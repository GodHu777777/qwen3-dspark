import copy
from dataclasses import replace
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from dspark_qwen.cached_sampling import cached_speculative_sample
from dspark_qwen.cached_target import CachedTarget
from dspark_qwen.config import DraftConfig
from dspark_qwen.model import DSparkDraft
from dspark_qwen.packed_draft import PackedDraft,test_only_noncausal_varlen
from dspark_qwen.packed_sampling import PackedSpeculativeSession,RequestSpec,propose_packed
from dspark_qwen.tensor_sampling import TensorProposal,TensorRandom,TensorRound
from dspark_qwen.varlen_target import VarlenPackedTarget,test_only_dense_varlen


def rng(seed):return TensorRandom(torch.Generator().manual_seed(seed))


class PackedSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(887)
        cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,attention_dropout=0.)
        cfg._attn_implementation='sdpa';self.model=Qwen3ForCausalLM(cfg).eval()
        self.draft=DSparkDraft(self.model,DraftConfig(layer_ids=(0,2),num_layers=2,block_size=3,markov_rank=8,mask_token_id=63)).eval()
        target=VarlenPackedTarget(copy.deepcopy(self.model),(0,2),test_kernel=test_only_dense_varlen)
        self.packed=PackedDraft(self.draft,test_kernel=test_only_noncausal_varlen)
        self.session=PackedSpeculativeSession(target,self.packed)
        self.prompts={'A':torch.tensor([[1,2,3,4]]),'B':torch.tensor([[7,8]]),'C':torch.tensor([[11,12,13]])}

    def admit(self,budget=11):
        return self.session.admit({r:RequestSpec(t,budget,rng(100+i)) for i,(r,t) in enumerate(self.prompts.items())})

    def assert_prefix_content(self):
        for r,state in self.session.requests.items():
            prefix=torch.cat((self.prompts[r],torch.tensor([state['output'][:-1]],dtype=torch.long)),dim=1)
            target=CachedTarget(copy.deepcopy(self.model),(0,2));features=target.prefill(prefix)
            expected=[(x.keys,x.values) for x in target.cache.layers]
            for actual,reference in zip(self.session.target.request_kv(r),expected):
                for a,b in zip(actual,reference):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
            projected=self.draft.project_context_kv(features.context,0)
            for actual,reference in zip(self.packed.request_kv(r),projected):
                for a,b in zip(actual,reference):torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)

    def test_gpu_entry_refuses_public_backend_and_requires_both_explicit_names(self):
        # Metadata-only constructor guard: no GPU runtime/model is touched.
        from dspark_qwen.rocm_varlen import BACKEND
        from dspark_qwen.packed_draft import DRAFT_BACKEND
        device=torch.device('cuda:0')
        target=SimpleNamespace(lengths={},device=device,layer_ids=(0,2),
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=64)),
            _varlen_kernel=SimpleNamespace(backend_name='public'))
        draft=SimpleNamespace(lengths={},device=device,draft=SimpleNamespace(spec=SimpleNamespace(layer_ids=(0,2))),
            kernel=SimpleNamespace(backend_name=DRAFT_BACKEND))
        with self.assertRaisesRegex(ValueError,'explicit pinned'):PackedSpeculativeSession(target,draft)
        target._varlen_kernel.backend_name=BACKEND
        self.assertIs(PackedSpeculativeSession(target,draft).target,target)
        draft.kernel.backend_name='public_noncausal'
        with self.assertRaisesRegex(ValueError,'explicit pinned'):PackedSpeculativeSession(target,draft)

    def test_real_multi_request_rollout_matches_independent_reference_and_cache_content(self):
        self.admit()
        while any(not x['finished'] for x in self.session.requests.values()):
            active={r:3 for r,x in self.session.requests.items() if not x['finished']}
            result=self.session.step(active)
            self.assertEqual(result['work']['target']['model_forward_calls'],1)
            self.assert_prefix_content()
        for i,(r,prompt) in enumerate(self.prompts.items()):
            independent=CachedTarget(copy.deepcopy(self.model),(0,2))
            output,_=cached_speculative_sample(independent,self.draft,prompt,11,rng=rng(100+i))
            self.assertEqual(self.session.outputs()[r],output)

    def test_batched_markov_retains_actual_q_and_pre_token_confidence(self):
        self.admit()
        states={r:self.session.requests[r]['rng'].generator.get_state() for r in ['A','B']}
        batch=self.session.propose(['A','B'],{'A':3,'B':2},mode='fixed_budget')
        for r,n in [('A',3),('B',2)]:
            independent=rng(1);independent.generator.set_state(states[r])
            original=self.draft.propose_stochastic_cached(torch.tensor([[self.session.requests[r]['output'][-1]]]),
                self.packed.request_kv(r),self.packed.lengths[r],temperature=1.,rng=independent,max_draft_tokens=n)
            actual=batch.proposals[r].proposal
            self.assertTrue(torch.equal(actual.tokens,original.tokens))
            torch.testing.assert_close(actual.draft_probs,original.draft_probs,atol=1e-8,rtol=1e-6)
            torch.testing.assert_close(actual.confidence_logits,original.confidence_logits,atol=1e-6,rtol=1e-5)
            self.assertEqual(actual.draft_probs.dtype,torch.float64)

    def test_bf16_amp_real_multi_request_step_has_finite_laws_and_mixed_cache(self):
        self.model.to(torch.bfloat16);self.session.target.model.to(torch.bfloat16)
        self.session.amp=True
        self.admit(7)
        while any(not x['finished'] for x in self.session.requests.values()):
            result=self.session.step({r:3 for r,x in self.session.requests.items() if not x['finished']})
            self.assertTrue(all(bool(torch.isfinite(p.draft_probs).all()) for p in result['proposals'].values()))
            self.assertTrue(all(k.dtype==torch.float32 and v.dtype==torch.bfloat16 for k,v in self.packed.layers))
        self.assertTrue(all(len(x)==7 for x in self.session.outputs().values()))
        self.assertEqual(self.draft.fc.weight.dtype,torch.float32)
        self.assertEqual(self.draft.rotary.inv_freq.dtype,torch.float32)

    def test_single_batch_head_calls_and_prefill_only_final_rows(self):
        counts={};head_shapes=[];hooks=[]
        modules={'target':self.session.target.model.model,'lm_head':self.draft.lm_head,
            'markov':self.draft.markov_projection,'confidence':self.draft.confidence}
        for name,module in modules.items():hooks.append(module.register_forward_hook(lambda _m,_a,_o,n=name:counts.__setitem__(n,counts.get(n,0)+1)))
        h=self.session.target.model.get_output_embeddings().register_forward_pre_hook(lambda _m,args:head_shapes.append(tuple(args[0].shape)))
        try:
            self.admit();self.assertEqual(head_shapes,[(1,3,32)])
            self.assertEqual(self.session.last_prefill_work['prefill_lm_head_rows'],3)
            counts.clear();head_shapes.clear()
            result=self.session.step({'A':3,'B':1,'C':0})
            self.assertEqual(counts['target'],1);self.assertEqual(counts['lm_head'],1)
            self.assertEqual(counts['markov'],3);self.assertEqual(counts['confidence'],3)
            self.assertEqual(head_shapes,[(1,7,32)])
            self.assertEqual(result['work']['proposal']['backbone']['query_rows'],6)
            self.assertEqual(result['work']['proposal']['target_only_skipped_requests'],1)
        finally:
            h.remove()
            for hook in hooks:hook.remove()

    def test_shadow_zero_allocation_refreshes_full_draft_and_preserves_q_prefix(self):
        self.admit();batch=self.session.propose(['A','B'])
        self.assertEqual(batch.work['backbone']['query_rows'],6)
        expected=batch.proposals['A'].proposal.draft_probs[:1].clone()
        # External observation mutation must not change internally retained actual q.
        batch.proposals['A'].proposal.draft_probs.zero_()
        result=self.session.verify_commit(batch.proposals,{'A':1,'B':0},allocation_policy='external_nonanticipating')
        self.assertTrue(torch.equal(result['proposals']['A'].draft_probs,expected))
        self.assertEqual(result['proposals']['B'].tokens.numel(),0)
        self.assertEqual(result['work']['target']['logical_query_tokens'],3)
        self.assertFalse(result['planner_causality_proven']);self.assert_prefix_content()
        with self.assertRaisesRegex(ValueError,'silently skip'):self.session.propose(['A'],{'A':0})

    def test_foreign_stale_reused_readded_handles_and_illegal_prefix_rejected(self):
        self.admit();batch=self.session.propose(['A'])
        handle=batch.proposals['A']
        with self.assertRaisesRegex(ValueError,'foreign'):self.session.verify_commit({'A':replace(handle)}, {'A':1},allocation_policy='external_nonanticipating')
        with self.assertRaisesRegex(ValueError,'legal proposal prefix'):self.session.verify_commit(batch.proposals,{'A':4},allocation_policy='external_nonanticipating')
        with self.assertRaisesRegex(ValueError,'predeclared'):self.session.verify_commit(batch.proposals,{'A':1},allocation_policy='fixed_before_proposal')
        self.session.verify_commit(batch.proposals,{'A':1},allocation_policy='external_nonanticipating')
        with self.assertRaisesRegex(ValueError,'Stale'):self.session.verify_commit(batch.proposals,{'A':1},allocation_policy='external_nonanticipating')
        new=self.session.propose(['A']);inc=new.proposals['A'].incarnation
        self.session.remove('A');self.session.admit({'A':RequestSpec(self.prompts['A'],11,rng(100))})
        self.assertGreater(self.session.requests['A']['incarnation'],inc)
        with self.assertRaisesRegex(ValueError,'Stale'):self.session.verify_commit(new.proposals,{'A':1},allocation_policy='external_nonanticipating')
        self.assert_prefix_content()

    def test_active_subset_keeps_inactive_caches_exact(self):
        self.admit();target=self.session.target.request_kv('C');draft=self.packed.request_kv('C')
        self.session.step({'A':2,'B':0})
        for left,right in [(target,self.session.target.request_kv('C')),(draft,self.packed.request_kv('C'))]:
            for a,b in zip(left,right):
                for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        self.assert_prefix_content()

    def test_typed_policy_receives_identity_budget_and_can_refuse_before_forward(self):
        self.admit();batch=self.session.propose(['A']);before=self.session.target.lengths
        original_confidence=tuple(batch.proposals['A'].proposal.confidence_logits.tolist())
        batch.proposals['A'].proposal.confidence_logits.fill_(123)
        class Policy:
            def validate_allocation(inner,context,handles,allocations):
                self.assertEqual(context['epoch'],batch.proposals['A'].epoch)
                self.assertEqual(context['proposal_confidence_logits']['A'],original_confidence)
                self.assertEqual(context['proposal_identity']['A']['nonce'],batch.proposals['A'].nonce)
                self.assertEqual(set(context['proposal_identity']['A']),{'request','incarnation','epoch','cache_length','nonce','proposal_limit','mode'})
                self.assertEqual(context['roster'][0]['remaining_output_budget'],10)
                self.assertEqual(context['roster'][0]['incarnation'],batch.proposals['A'].incarnation)
                raise ValueError('ineligible causal proof')
        with self.assertRaisesRegex(ValueError,'causal proof'):
            self.session.verify_commit(batch.proposals,{'A':1},allocation_policy=Policy())
        self.assertEqual(before,self.session.target.lengths);self.assertFalse(self.session.failed)
        class Valid:
            def validate_allocation(inner,context,handles,allocations):return dict(epoch=context['epoch'],fixed_capacity=1,proof_scope='test fixture')
        result=self.session.verify_commit(batch.proposals,{'A':1},allocation_policy=Valid())
        self.assertEqual(result['allocation_policy'],'typed_validator');self.assertFalse(result['planner_causality_proven'])

    def test_execution_error_invalidates_session_without_replay(self):
        self.admit();batch=self.session.propose(['A','B'])
        with patch.object(self.packed,'append_committed',side_effect=RuntimeError('injected commit projection')):
            with self.assertRaisesRegex(RuntimeError,'commit projection'):
                self.session.verify_commit(batch.proposals,{'A':1,'B':1},allocation_policy='external_nonanticipating')
        self.assertTrue(self.session.failed);self.assertEqual(self.session.target.lengths,{})
        self.assertEqual(self.packed.lengths,{});self.assertEqual(self.packed.layers,[])
        with self.assertRaisesRegex(RuntimeError,'invalidated'):self.session.step({'A':1})
        with self.assertRaisesRegex(RuntimeError,'invalidated'):self.session.outputs()

    def test_controlled_commit_every_rejection_and_eos_path_keeps_real_caches(self):
        # Real packed forward/proposal/heads still execute. Only acceptance is controlled
        # to cover finite rollback branches independently of random tiny-model acceptance.
        for rejected in [0,1,2,None]:
            self.setUp();self.admit(20)
            def decision(proposal,p,rng,*,max_new_tokens,eos_ids):
                n=proposal.tokens.numel();accepted=n if rejected is None else rejected
                extra='bonus' if rejected is None else 'residual'
                tokens=torch.cat((proposal.tokens[:accepted],torch.tensor([5])))
                return TensorRound(tokens,accepted,rejected,extra,'round_complete')
            with patch('dspark_qwen.packed_sampling.verify_proposal',side_effect=decision):self.session.step({'A':3,'B':3})
            self.assert_prefix_content()
        for extra,accepted in [(None,2),('residual',1),('bonus',3)]:
            self.setUp();self.admit(20)
            def eos(proposal,p,rng,*,max_new_tokens,eos_ids):
                tokens=proposal.tokens[:accepted] if extra is None else torch.cat((proposal.tokens[:accepted],torch.tensor([5])))
                return TensorRound(tokens,accepted,1 if extra=='residual' else None,extra,'eos')
            with patch('dspark_qwen.packed_sampling.verify_proposal',side_effect=eos):result=self.session.step({'A':3,'B':3})
            self.assertTrue(self.session.requests['A']['finished']);self.assert_prefix_content()


if __name__=='__main__':unittest.main()
