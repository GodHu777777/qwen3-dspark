"""Actual tiny-Qwen target-only control, independent sampling/cache oracles."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dspark_qwen.cached_sampling import cached_target_sample
from dspark_qwen.cached_target import CachedTarget
from dspark_qwen.packed_sampling import RequestSpec
from dspark_qwen.packed_target_sampling import PackedTargetOnlySession
from dspark_qwen.persistent_qwen_target import PersistentQwenTarget, test_only_persistent_sdpa
from dspark_qwen.persistent_target_kv import Bucket
from dspark_qwen.target_strategy import FiniteTargetBuckets, PersistentTargetStrategy
from dspark_qwen.tensor_sampling import TensorRandom


class RecordedRandom:
    def __init__(self, seed):
        self.base = TensorRandom(torch.Generator().manual_seed(seed))
        self.values = []; self.laws = []; self.queued = []

    def uniform(self, reference):
        value = reference.new_tensor(self.queued.pop(0)) if self.queued else self.base.uniform(reference)
        self.values.append(float(value)); self.laws.append(reference.detach().clone())
        return value


class PackedTargetOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(887)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256, attention_dropout=0.)
        config._attn_implementation = 'sdpa'
        self.model = Qwen3ForCausalLM(config).eval()
        self.prompts = {'A':torch.tensor([[1,2,3,4]]), 'B':torch.tensor([[7,8]]), 'C':torch.tensor([[11,12,13]])}
        self.buckets = FiniteTargetBuckets(
            (Bucket((4,2,3),(0,0,0),'CPU prompt fixture'), Bucket((4,),(0,),'CPU re-admission fixture')),
            tuple(Bucket((1,)*r,(20,)*r,'CPU anchor-only fixture') for r in (1,2,3)))

    def session(self, *, buckets=None, eos_ids=(), temperature=.8):
        target = PersistentQwenTarget(copy.deepcopy(self.model),(0,2),slots=3,context_capacity=24,
            max_query_tokens=9,test_kernel=test_only_persistent_sdpa)
        return PackedTargetOnlySession(target,target_strategy=PersistentTargetStrategy(target,buckets or self.buckets),
                                       eos_ids=eos_ids,temperature=temperature)

    def admit(self, session, budgets=None, names=None):
        names = tuple(self.prompts) if names is None else names
        budgets = budgets or {'A':3,'B':5,'C':7}
        return session.admit({r:RequestSpec(self.prompts[r],budgets[r],RecordedRandom(100+i))
                              for i,r in enumerate(names)})

    def assert_prefix(self, session):
        for r,state in session.requests.items():
            prefix = torch.cat((self.prompts[r],torch.tensor([state['output'][:-1]],dtype=torch.long)),dim=1)
            oracle = CachedTarget(copy.deepcopy(self.model),(0,2)); oracle.prefill(prefix)
            for actual,expected in zip(session.target.request_kv(r),[(x.keys,x.values) for x in oracle.cache.layers]):
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
            self.assertEqual(session.target.lengths[r],prefix.shape[1])

    def assert_failed(self, session):
        self.assertTrue(session.failed)
        for action in (session.outputs,lambda:session.step(('A',)),lambda:session.remove('A'),
                       lambda:self.admit(session,names=('A',))):
            with self.assertRaisesRegex(RuntimeError,'invalidated'):action()

    def test_multi_request_rollout_matches_independent_reference_law_rng_and_all_layer_kv(self):
        session=self.session();self.admit(session);self.assert_prefix(session)
        while any(not s['finished'] for s in session.requests.values()):
            active=[r for r,s in session.requests.items() if not s['finished']]
            result=session.step(active);self.assert_prefix(session)
            self.assertEqual(result['work']['target']['physical_query_tokens'],len(active))
            self.assertEqual(result['work']['target']['session_target_commit_calls'],1)
            self.assertEqual(result['work']['draft_forward_calls'],0)
            for r,p in result['target_probs'].items():
                self.assertEqual(p.dtype,torch.float64)
                self.assertTrue(torch.equal(p,session.requests[r]['rng'].laws[-1]))
        for i,(r,prompt) in enumerate(self.prompts.items()):
            random=RecordedRandom(100+i);state=session.requests[r]
            reference=cached_target_sample(CachedTarget(copy.deepcopy(self.model),(0,2)),prompt,state['budget'],
                                            temperature=.8,rng=random)
            self.assertEqual(session.outputs()[r],reference)
            self.assertEqual(state['rng'].values,random.values)
            self.assertEqual(len(random.values),state['budget'])
            for p,q in zip(state['rng'].laws,random.laws):
                torch.testing.assert_close(p,q,atol=1e-9,rtol=1e-6)

    def test_admission_r_final_head_rows_and_one_batched_step_commit_after_all_draws(self):
        session=self.session();shapes=[];forwards=[]
        head=session.target.model.get_output_embeddings().register_forward_pre_hook(lambda _,a:shapes.append(tuple(a[0].shape)))
        model=session.target.model.model.register_forward_hook(lambda *_:forwards.append(1))
        try:
            self.admit(session)
            self.assertEqual(shapes,[(1,3,32)]);self.assertEqual(len(forwards),1)
            self.assertEqual(session.last_prefill_work['prefill_lm_head_rows'],3)
            self.assertFalse(hasattr(session,'draft'))
            shapes.clear();forwards.clear();before=session.target.lengths
            commit=session.target.commit;calls=[]
            def checked(features,counts):
                self.assertEqual(session.target.lengths,before)
                self.assertEqual([len(session.requests[r]['rng'].values) for r in ('B','A')],[2,2])
                calls.append(dict(counts));return commit(features,counts)
            with patch.object(session.target,'commit',side_effect=checked):result=session.step(('B','A'))
            self.assertEqual(calls,[{'B':1,'A':1}]);self.assertEqual(shapes,[(1,2,32)])
            self.assertEqual(len(forwards),1);self.assertEqual(result['work']['shadow_proposal_positions'],0)
            self.assert_prefix(session)
        finally:head.remove();model.remove()

    def test_inactive_finished_remove_readmit_and_incarnation(self):
        session=self.session();self.admit(session,budgets={'A':2,'B':5,'C':1})
        c_before=session.target.request_kv('C');b_before=session.target.request_kv('B')
        session.step(('A',));self.assertTrue(session.requests['A']['finished'])
        self.assertTrue(session.requests['C']['finished'])
        for name,before in (('B',b_before),('C',c_before)):
            for a,b in zip(before,session.target.request_kv(name)):
                for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        with self.assertRaisesRegex(ValueError,'unfinished'):session.step(('A',))
        incarnation=session.requests['A']['incarnation'];session.remove('A')
        self.admit(session,names=('A',));self.assertGreater(session.requests['A']['incarnation'],incarnation)
        self.assert_prefix(session)

    def test_eos_on_admission_and_decode_obeys_reference_draw_count(self):
        for queued in ([0.],[.5,0.]):
            session=self.session(eos_ids=(0,));random=RecordedRandom(100);random.queued=list(queued)
            session.admit({'A':RequestSpec(self.prompts['A'],8,random)})
            while not session.requests['A']['finished']:session.step(('A',))
            other=RecordedRandom(100);other.queued=list(queued)
            expected=cached_target_sample(CachedTarget(copy.deepcopy(self.model),(0,2)),self.prompts['A'],8,
                                          temperature=.8,rng=other,eos_ids=(0,))
            self.assertEqual(session.outputs()['A'],expected)
            self.assertEqual(len(random.values),len(queued));self.assertEqual(random.values,other.values)
            self.assertEqual(expected[-1],0);self.assert_prefix(session)

    def test_missing_bucket_rejects_before_admission_forward_or_rng(self):
        buckets=FiniteTargetBuckets((Bucket((4,),(0,),'single admission'),),(Bucket((1,),(4,),'one step'),))
        session=self.session(buckets=buckets)
        with patch.object(session.target.model.model,'forward',side_effect=AssertionError('unexpected forward')):
            with self.assertRaisesRegex(ValueError,'No finite prefill'):self.admit(session)
        self.assertEqual(session.target.lengths,{});self.assertFalse(session.failed)
        self.admit(session,budgets={'A':8},names=('A',));session.step(('A',))
        before=list(session.requests['A']['rng'].values)
        with patch.object(session.target.model.model,'forward',side_effect=AssertionError('unexpected forward')):
            with self.assertRaisesRegex(ValueError,'No finite verification'):session.step(('A',))
        self.assertEqual(session.requests['A']['rng'].values,before);self.assertFalse(session.failed)
        self.assert_prefix(session)

    def test_invalid_inputs_and_foreign_strategy_have_no_state_change(self):
        session=self.session()
        for spec in (RequestSpec(self.prompts['A'],0,RecordedRandom(1)),
                     RequestSpec(torch.tensor([[64]]),3,RecordedRandom(1)),
                     RequestSpec(self.prompts['A'],3,object())):
            with self.assertRaises(ValueError):session.admit({'A':spec})
        self.assertEqual(session.target.lengths,{})
        with self.assertRaisesRegex(ValueError,'strategy'):
            PackedTargetOnlySession(session.target,target_strategy=SimpleNamespace(target=object()))
        self.admit(session)
        for active in ([],['A','A'],['unknown']):
            with self.assertRaises(ValueError):session.step(active)
        self.assertFalse(session.failed)

    def test_execution_failure_before_commit_after_commit_and_admission_fail_closed(self):
        for stage in ('predict','rng','commit','release','admission'):
            session=self.session()
            if stage=='admission':
                with patch.object(session.target.model.get_output_embeddings(),'forward',side_effect=RuntimeError('head failed')):
                    with self.assertRaisesRegex(RuntimeError,'head failed'):self.admit(session)
            else:
                self.admit(session)
                if stage=='predict':owner,name,side=session.target,'predict',RuntimeError('predict failed')
                elif stage=='commit':
                    owner,name=session.target,'commit'
                    def side(*args):
                        session.target.pool.failed=True
                        raise RuntimeError('device write failed')
                elif stage=='release':owner,name,side=session.target_strategy,'release_features',RuntimeError('release failed')
                else:
                    owner,name,side=session.requests['A']['rng'],'uniform',RuntimeError('draw failed')
                with patch.object(owner,name,side_effect=side):
                    with self.assertRaisesRegex(RuntimeError,'failed'):session.step(('A','B'))
            self.assert_failed(session)
            if stage in ('commit','release'):self.assertTrue(session.cleanup_error)
            else:
                self.assertEqual(session.target.lengths,{});self.assertIsNone(session.target._pending)

    def test_gpu_entry_guard_is_only_metadata_and_keeps_native_explicit(self):
        from dspark_qwen.rocm_varlen import BACKEND
        target=SimpleNamespace(lengths={},device=torch.device('cuda:0'),
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=64)),
            _varlen_kernel=SimpleNamespace(backend_name='public'))
        strategy=SimpleNamespace(target=target)
        with self.assertRaisesRegex(ValueError,'explicit pinned'):
            PackedTargetOnlySession(target,target_strategy=strategy)
        target._varlen_kernel.backend_name=BACKEND
        self.assertIs(PackedTargetOnlySession(target,target_strategy=strategy).target,target)


if __name__=='__main__':unittest.main()
