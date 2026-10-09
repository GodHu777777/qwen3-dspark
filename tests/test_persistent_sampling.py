"""Actual tiny-Qwen session tests; probability law and q are never mocked."""
import copy
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dspark_qwen.cached_target import CachedTarget
from dspark_qwen.config import DraftConfig
from dspark_qwen.model import DSparkDraft
from dspark_qwen.packed_draft import PackedDraft, test_only_noncausal_varlen
from dspark_qwen.packed_sampling import PackedSpeculativeSession, RequestSpec
from dspark_qwen.persistent_qwen_target import PersistentQwenTarget, test_only_persistent_sdpa
from dspark_qwen.persistent_target_kv import Bucket
from dspark_qwen.target_strategy import AppendCropTargetStrategy, FiniteTargetBuckets, PersistentTargetStrategy
from dspark_qwen.tensor_sampling import TensorRandom, logits_to_probabilities, residual_distribution
from dspark_qwen.varlen_target import VarlenPackedTarget, test_only_dense_varlen


class RecordedRandom:
    def __init__(self, seed):
        self.base = TensorRandom(torch.Generator().manual_seed(seed))
        self.values, self.queued = [], []

    def uniform(self, reference):
        value = reference.new_tensor(self.queued.pop(0)) if self.queued else self.base.uniform(reference)
        self.values.append(float(value))
        return value


def bucket(q, c=40):
    return Bucket(tuple(q), (c,)*len(q), 'explicit tiny-Qwen test fixture')


class PersistentSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(887)
        cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256, attention_dropout=0.)
        cfg._attn_implementation = 'sdpa'
        self.model = Qwen3ForCausalLM(cfg).eval()
        self.draft = DSparkDraft(self.model, DraftConfig(layer_ids=(0,2), num_layers=2, block_size=3,
                                                       markov_rank=8, mask_token_id=63)).eval()
        self.prompts = {'A':torch.tensor([[1,2,3,4]]), 'B':torch.tensor([[7,8]]), 'C':torch.tensor([[11,12,13]])}
        self.buckets = FiniteTargetBuckets(
            prefill=(bucket((4,2,3),0), bucket((4,),0)),
            verification=tuple(bucket(q) for q in ((4,4,4),(4,4),(4,),(3,),(3,1),(2,1),(1,),(2,),(2,2),(1,1,1))))

    def session(self, persistent=True, explicit_default=False, buckets=None):
        packed = PackedDraft(copy.deepcopy(self.draft), test_kernel=test_only_noncausal_varlen)
        if persistent:
            target = PersistentQwenTarget(copy.deepcopy(self.model),(0,2),slots=3,context_capacity=48,
                max_query_tokens=12,test_kernel=test_only_persistent_sdpa)
            strategy = PersistentTargetStrategy(target, buckets or self.buckets)
        else:
            target = VarlenPackedTarget(copy.deepcopy(self.model),(0,2),test_kernel=test_only_dense_varlen)
            strategy = AppendCropTargetStrategy(target) if explicit_default else None
        return PackedSpeculativeSession(target,packed,target_strategy=strategy)

    def admit(self, session, budget=12, seed=100, names=None):
        names = tuple(self.prompts) if names is None else names
        return session.admit({r:RequestSpec(self.prompts[r],budget,RecordedRandom(seed+i)) for i,r in enumerate(names)})

    def assert_prefix(self, session):
        for r,state in session.requests.items():
            prefix = torch.cat((self.prompts[r],torch.tensor([state['output'][:-1]],dtype=torch.long)),dim=1)
            oracle = CachedTarget(copy.deepcopy(self.model),(0,2))
            features = oracle.prefill(prefix)
            for actual,expected in zip(session.target.request_kv(r),[(x.keys,x.values) for x in oracle.cache.layers]):
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
            projected = session.draft.draft.project_context_kv(features.context,0)
            for actual,expected in zip(session.draft.request_kv(r),projected):
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)

    def assert_closed(self, session):
        self.assertTrue(session.failed)
        self.assertEqual(session.draft.lengths,{})
        for action in (lambda:session.outputs(),lambda:session.step({'A':1}),lambda:session.remove('A'),
                       lambda:self.admit(session,names=('A',)),lambda:session.propose(('A',)),
                       lambda:session.verify_commit({}, {}, allocation_policy='external_nonanticipating')):
            with self.assertRaisesRegex(RuntimeError,'invalidated'):action()

    def test_seeded_actual_q_p_rng_and_outputs_match_original_and_explicit_default(self):
        sessions = [self.session(False),self.session(False,True),self.session()]
        for session in sessions:self.admit(session,budget=10)
        # Fixed zero allocation still verifies one anchor per request, allowing
        # synchronized final budgets after several stochastic speculative rounds.
        for allocations in ({'A':3,'B':3,'C':3},{'A':1,'B':0},{'A':0,'B':0,'C':0}):
            results = [s.step(allocations) for s in sessions]
            for s,result in zip(sessions,results):
                self.assertEqual(s.outputs(),sessions[0].outputs());self.assert_prefix(s)
                self.assertEqual(result['requests'],results[0]['requests'])
                for r,p in result['proposals'].items():
                    self.assertTrue(torch.equal(p.tokens,results[0]['proposals'][r].tokens))
                    torch.testing.assert_close(p.draft_probs,results[0]['proposals'][r].draft_probs,atol=1e-9,rtol=1e-6)
                    torch.testing.assert_close(result['target_probs'][r],results[0]['target_probs'][r],atol=1e-9,rtol=1e-6)
                    self.assertEqual(s.requests[r]['rng'].values,sessions[0].requests[r]['rng'].values)
            # Default strategy must be bit-exact against explicitly requested old path.
            for r in results[0]['proposals']:
                self.assertTrue(torch.equal(results[0]['target_probs'][r],results[1]['target_probs'][r]))

    def test_shadow_zero_inactive_resident_and_admission_head_rows(self):
        session=self.session();shapes=[]
        hook=session.target.model.get_output_embeddings().register_forward_pre_hook(lambda _m,a:shapes.append(tuple(a[0].shape)))
        self.admit(session);hook.remove()
        self.assertEqual(shapes,[(1,3,32)])
        old=[session.target.request_kv('C'),session.draft.request_kv('C')]
        batch=session.propose(('A','B'))
        self.assertEqual(batch.work['backbone']['query_rows'],6)
        actual_q=batch.proposals['A'].proposal.draft_probs[:1].clone()
        batch.proposals['A'].proposal.draft_probs.zero_()
        result=session.verify_commit(batch.proposals,{'A':1,'B':0},allocation_policy='external_nonanticipating')
        self.assertTrue(torch.equal(result['proposals']['A'].draft_probs,actual_q))
        self.assertEqual(result['work']['target']['physical_query_tokens'],3)
        self.assertEqual(result['work']['target']['session_target_commit_calls'],1)
        for before,after in zip(old,[session.target.request_kv('C'),session.draft.request_kv('C')]):
            for a,b in zip(before,after):
                for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        self.assert_prefix(session)

    def test_actual_law_partial_accept_reject_and_single_commit_after_all_decisions(self):
        # Real p/q come from complete model forwards. Chosen legal uniforms cover
        # every rejection index; no probability, proposal or decision is replaced.
        for reject in (0,1,2,None):
            for seed in range(100,120):
                session=self.session();self.admit(session,budget=20,seed=seed)
                batch=session.propose(('A','B'))
                chunks={r:torch.cat((torch.tensor([session.requests[r]['output'][-1]]),h.proposal.tokens))[None]
                        for r,h in batch.proposals.items()}
                preview=session.target_strategy.verify(chunks)
                p={r:logits_to_probabilities(v[0],1.) for r,v in session.target.predict(preview).items()}
                session.target.abort(preview)
                session.target_strategy.release_features(preview)
                alpha={r:p[r][torch.arange(3),h.proposal.tokens]/h.proposal.draft_probs[torch.arange(3),h.proposal.tokens]
                       for r,h in batch.proposals.items()}
                if reject is None or all(float(a[reject])<.999 for a in alpha.values()):break
            else:self.fail('Fixture did not find actual rejectable q samples')
            draws={r:len(session.requests[r]['rng'].values) for r in ('A','B')}
            for r in ('A','B'):
                session.requests[r]['rng'].queued=([0.]*3+[.4] if reject is None else
                    [0.]*reject+[(1.+float(alpha[r][reject]))/2.,.4])
            original=session.target.commit;calls=[]
            def commit(features,counts):
                self.assertEqual(session.target.lengths,{'A':4,'B':2,'C':3})
                for r in ('A','B'):
                    self.assertEqual(len(session.requests[r]['rng'].values)-draws[r],4 if reject is None else reject+2)
                calls.append(dict(counts));return original(features,counts)
            with patch.object(session.target,'commit',side_effect=commit):
                result=session.verify_commit(batch.proposals,{'A':3,'B':3},allocation_policy='external_nonanticipating')
            accepted=3 if reject is None else reject
            self.assertEqual(calls,[{'A':accepted+1,'B':accepted+1}])
            for record in result['requests'].values():
                self.assertEqual(record['accepted'],accepted);self.assertEqual(record['rejected_index'],reject)
            self.assert_prefix(session)

    def test_actual_eos_accept_residual_bonus_budget_and_remove_readmit(self):
        for kind in ('accepted','residual','bonus','budget'):
            session=self.session();self.admit(session,budget=3 if kind=='budget' else 20,names=('A',))
            batch=session.propose(('A',));proposal=batch.proposals['A'].proposal
            chunks={'A':torch.cat((torch.tensor([session.requests['A']['output'][-1]]),proposal.tokens))[None]}
            preview=session.target_strategy.verify(chunks)
            p=logits_to_probabilities(session.target.predict(preview)['A'][0],1.)
            session.target.abort(preview)
            session.target_strategy.release_features(preview)
            if kind in ('accepted','budget'):
                session.requests['A']['rng'].queued=[0.]*3
                if kind=='accepted':session.eos_ids=frozenset([int(proposal.tokens[1])])
            elif kind=='bonus':
                eos=next(i for i in range(64) if i not in proposal.tokens.tolist())
                session.eos_ids=frozenset([eos])
                u=float(p[-1,:eos].sum()+p[-1,eos]/2)
                session.requests['A']['rng'].queued=[0.]*3+[u]
            else:
                alphas=p[torch.arange(3),proposal.tokens]/proposal.draft_probs[torch.arange(3),proposal.tokens]
                j=next(i for i in range(3) if float(alphas[i])<1)
                residual=residual_distribution(p[j],proposal.draft_probs[j])
                eos=next(i for i in range(64) if residual[i]>0 and i not in proposal.tokens[:j].tolist())
                session.eos_ids=frozenset([eos])
                u=float(residual[:eos].sum()+residual[eos]/2)
                session.requests['A']['rng'].queued=[0.]*j+[(1.+float(alphas[j]))/2,u]
            # Shadow budget can be smaller than collected full proposal.
            allocation=2 if kind=='budget' else 3
            result=session.verify_commit(batch.proposals,{'A':allocation},allocation_policy='external_nonanticipating')
            record=result['requests']['A']
            self.assertEqual(record['stop_reason'],'budget' if kind=='budget' else 'eos')
            self.assertEqual(record['extra_token_kind'],None if kind in ('accepted','budget') else kind)
            self.assertTrue(session.requests['A']['finished']);self.assert_prefix(session)
            if kind=='budget':self.assertEqual(len(session.outputs()['A']),3)
            else:self.assertIn(session.outputs()['A'][-1],session.eos_ids)
            previous_incarnation=session.requests['A']['incarnation']
            session.remove('A');session.eos_ids=frozenset()
            self.admit(session,names=('A',))
            self.assertGreater(session.requests['A']['incarnation'],previous_incarnation)
            with self.assertRaisesRegex(ValueError,'Stale'):
                session.verify_commit(batch.proposals,{'A':1},allocation_policy='external_nonanticipating')
            self.assert_prefix(session)

    def test_finite_bucket_preflight_has_no_forward_draw_or_residual_admission(self):
        buckets=FiniteTargetBuckets((bucket((4,),0),),(bucket((1,),4),))
        session=self.session(buckets=buckets)
        with patch.object(session.target.model.model,'forward',side_effect=AssertionError('unexpected forward')):
            with self.assertRaisesRegex(ValueError,'No finite prefill'):self.admit(session)
        self.assertFalse(session.failed);self.assertEqual(session.target.lengths,{})
        self.assertEqual(session.draft.lengths,{})
        self.admit(session,names=('A',));rng=session.requests['A']['rng'];draws=list(rng.values)
        with patch.object(session.draft,'backbone',side_effect=AssertionError('unexpected backbone')):
            with self.assertRaisesRegex(ValueError,'No finite verification'):session.step({'A':3})
        self.assertEqual(rng.values,draws)
        batch=session.propose(('A',));draws=list(rng.values)
        with patch.object(session.target.model.model,'forward',side_effect=AssertionError('unexpected forward')):
            with self.assertRaisesRegex(ValueError,'No finite verification'):
                session.verify_commit(batch.proposals,{'A':1},allocation_policy='external_nonanticipating')
        self.assertEqual(rng.values,draws);self.assertFalse(session.failed)
        session.verify_commit(batch.proposals,{'A':0},allocation_policy='external_nonanticipating')
        draws=list(rng.values)
        with self.assertRaisesRegex(ValueError,'No finite verification'):session.step({'A':0})
        self.assertEqual(rng.values,draws);self.assert_prefix(session)

    def test_finite_provider_exact_order_min_capacity_and_no_shape_enumeration(self):
        small=bucket((2,1),10);large=bucket((2,1),20)
        provider=FiniteTargetBuckets((bucket((4,),0),),(large,small))
        session=self.session(buckets=provider)
        self.assertEqual(len(session.target.pool._workspaces),3)
        self.assertIs(provider.select('verification',(2,1),(5,9)),small)
        self.assertIs(provider.select('verification',(2,1),(11,9)),large)
        with self.assertRaisesRegex(ValueError,'No finite'):provider.select('verification',(1,2),(5,9))

    def test_feature_lease_release_after_head_and_draft_and_after_failure_abort(self):
        # CPU protocol test with real model features and a fake lease. This does
        # not execute a device graph or prove asynchronous event ownership.
        session=self.session();target=session.target;events=[];lease=[]
        verify,commit,abort,reset=target.verify,target.commit,target.abort,target.reset
        release=getattr(target,'release_features',None)
        append=session.draft.append_committed
        def verified(*args,**kwargs):
            self.assertFalse(lease)
            features=verify(*args,**kwargs);lease.append(features);events.append('verify');return features
        def committed(features,counts):
            self.assertIs(lease[0],features);result=commit(features,counts);events.append('commit');return result
        def projected(contexts):
            self.assertTrue(lease);result=append(contexts);events.append('draft');return result
        def released(features):
            self.assertIs(lease[0],features)
            if release is not None:release(features)
            events.append('release');lease.clear()
        def aborted(features):
            result=abort(features);events.append('abort');return result
        def cleared():
            self.assertFalse(lease);events.append('reset');return reset()
        from contextlib import ExitStack
        with ExitStack() as stack:
            for owner,name,func in ((target,'verify',verified),(target,'commit',committed),
                    (session.draft,'append_committed',projected),(target,'release_features',released),
                    (target,'abort',aborted),(target,'reset',cleared)):
                stack.enter_context(patch.object(owner,name,side_effect=func,create=True))
            head=target.model.get_output_embeddings().register_forward_hook(lambda *_:events.append('head'))
            stack.callback(head.remove)
            self.admit(session)
            self.assertEqual(events,['verify','commit','head','draft','release'])
            events.clear();session.step({'A':1,'B':1})
            self.assertEqual(events,['verify','head','commit','draft','release'])
            events.clear();batch=session.propose(('A','B'))
            with patch.object(target,'predict',side_effect=RuntimeError('head failure')):
                with self.assertRaisesRegex(RuntimeError,'head failure'):
                    session.verify_commit(batch.proposals,{'A':1,'B':1},allocation_policy='external_nonanticipating')
            self.assertEqual(events,['verify','abort','release','reset'])
        self.assert_closed(session)

    def test_admission_failure_aborts_prefill_or_clears_committed_state(self):
        for stage in ('commit','draft','slot'):
            session=self.session()
            from contextlib import ExitStack
            with ExitStack() as stack:
                if stage=='commit':
                    stack.enter_context(patch.object(session.target,'commit',side_effect=RuntimeError('prefill commit failure')))
                elif stage=='draft':
                    stack.enter_context(patch.object(session.draft,'append_committed',side_effect=RuntimeError('prefill draft failure')))
                else:
                    self.admit(session)
                with self.assertRaises((RuntimeError,ValueError)):
                    if stage=='slot':
                        session.admit({'D':RequestSpec(self.prompts['A'],10,RecordedRandom(100))})
                    else:self.admit(session)
            self.assert_closed(session)
            self.assertEqual(session.target.lengths,{})
            self.assertIsNone(session.target._pending);self.assertIsNone(session.target.pool._pending)

    def test_abort_before_commit_failure_after_commit_and_poisoned_cleanup(self):
        for stage in ('predict','decision','draft','commit_poison','abort'):
            session=self.session();self.admit(session);batch=session.propose(('A','B'))
            if stage=='predict':owner,name,error=session.target,'predict',RuntimeError('predict failure')
            elif stage=='decision':
                # Invalid RNG is rejected by the real law with scratch outstanding.
                session.requests['A']['rng'].queued=[2.]
                owner,name,error=session.target,'predict',None
            elif stage=='draft':owner,name,error=session.draft,'append_committed',RuntimeError('draft failure')
            elif stage=='abort':owner,name,error=session.target,'predict',RuntimeError('predict failure')
            else:
                owner,name=session.target,'commit'
                def error(*args):
                    session.target.pool.failed=True
                    raise RuntimeError('device commit failure')
            from contextlib import ExitStack
            with ExitStack() as stack:
                if error is not None:stack.enter_context(patch.object(owner,name,side_effect=error))
                if stage=='abort':stack.enter_context(patch.object(session.target,'abort',side_effect=RuntimeError('abort failure')))
                with self.assertRaises((RuntimeError,ValueError)):
                    session.verify_commit(batch.proposals,{'A':1,'B':1},allocation_policy='external_nonanticipating')
            self.assert_closed(session)
            if stage in ('commit_poison','abort'):self.assertTrue(session.cleanup_error)
            else:
                self.assertIsNone(session.target._pending);self.assertIsNone(session.target.pool._pending)
                self.assertEqual(session.target.lengths,{})


if __name__=='__main__':unittest.main()
