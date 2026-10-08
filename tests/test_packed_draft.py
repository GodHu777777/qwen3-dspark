import copy
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dspark_qwen.config import DraftConfig
from dspark_qwen.model import DSparkDraft
from dspark_qwen.packed_draft import (DRAFT_BACKEND, PackedDraft,
    PinnedRocmDraftKernel, test_only_noncausal_varlen)
from dspark_qwen.varlen_target import PackedLayout


class PackedDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(773)
        cfg = Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,
            head_dim=8,max_position_embeddings=256,attention_dropout=0.)
        cfg._attn_implementation = 'sdpa'
        self.draft = DSparkDraft(Qwen3ForCausalLM(cfg).eval(),
            DraftConfig(layer_ids=(0,2),num_layers=2,block_size=3,markov_rank=8,mask_token_id=63)).eval()
        self.packed = PackedDraft(self.draft,test_kernel=test_only_noncausal_varlen)
        self.features = {r:torch.randn(1,n,64) for r,n in [('A',7),('B',6),('C',4)]}
        for r in self.features:self.packed.add_request(r)

    def prime(self):
        self.packed.append_committed({r:f[:,:n] for (r,f),n in zip(self.features.items(),[4,2,1])})

    def assert_pairs(self,left,right,exact=False):
        self.assertEqual(len(left),len(right))
        for a,b in zip(left,right):
            for x,y in zip(a,b):
                if exact:self.assertTrue(torch.equal(x,y))
                else:torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)

    def assert_reference(self,anchors,features):
        before=copy.deepcopy(self.packed.layers)
        actual=self.packed.backbone(anchors)
        for r,anchor in anchors.items():
            f=features[r];length=f.shape[1]
            kv=self.draft.project_context_kv(f,0)
            self.assert_pairs(self.packed.request_kv(r),kv)
            with torch.no_grad():expected=self.draft.backbone_cached(anchor,kv,length)
            torch.testing.assert_close(actual.for_request(r),expected,atol=2e-6,rtol=2e-5)
            ids=torch.cat((anchor.new_zeros((1,length)),anchor),1)
            with torch.no_grad():full=self.draft.backbone(ids,f,torch.tensor([length]))[0,0]
            torch.testing.assert_close(actual.for_request(r),full,atol=2e-6,rtol=2e-5)
        self.assert_pairs(before,self.packed.layers,exact=True)
        return actual

    def test_complete_backbone_matches_cached_and_full_independent_requests(self):
        self.prime()
        output=self.assert_reference({'B':torch.tensor([[9]]),'A':torch.tensor([[8]])},
            {'A':self.features['A'][:,:4],'B':self.features['B'][:,:2]})
        self.assertEqual(output.context_lengths,(2,4))
        self.assertEqual(output.hidden.shape,(6,32))
        self.assertEqual(output.work['query_rows'],6)
        self.assertEqual(output.work['gathered_key_rows'],12)
        self.assertEqual(output.work['attention_pair_domain'],36)
        self.assertEqual(output.work['gathered_kv_elements'],2*2*8*12*2)
        self.assertEqual(output.work['context_block_concatenation_kv_elements'],2*2*8*(7+6)*2)

    def test_module_calls_are_batched_not_per_request(self):
        calls={};hooks=[]
        modules={'fc':self.draft.fc,'context_norm':self.draft.context_norm,'embedding':self.draft.embedding}
        for i,layer in enumerate(self.draft.layers):
            for name in ['q_proj','k_proj','v_proj','o_proj']:modules[f'{i}.{name}']=getattr(layer.attention,name)
            modules[f'{i}.mlp']=layer.mlp
        for name,module in modules.items():
            hooks.append(module.register_forward_hook(lambda _m,_a,_o,n=name:calls.__setitem__(n,calls.get(n,0)+1)))
        kernel=Mock(side_effect=test_only_noncausal_varlen);self.packed.kernel=kernel
        try:
            self.prime();self.assertEqual(calls['fc'],1);self.assertEqual(calls['context_norm'],1)
            for i in range(2):self.assertEqual(calls[f'{i}.k_proj'],1);self.assertEqual(calls[f'{i}.v_proj'],1)
            calls.clear();self.packed.backbone({r:torch.tensor([[8]]) for r in ['C','A','B']})
            self.assertNotIn('fc',calls);self.assertEqual(calls['embedding'],1)
            for i in range(2):
                for name in ['q_proj','k_proj','v_proj','o_proj','mlp']:self.assertEqual(calls[f'{i}.{name}'],1)
            self.assertEqual(kernel.call_count,2)
        finally:
            for hook in hooks:hook.remove()

    def test_append_crop_zero_exit_readd_actual_kv_content_and_positions(self):
        self.prime();old={r:self.packed.request_kv(r) for r in self.features}
        self.packed.append_committed({'A':self.features['A'][:,4:7],'B':self.features['B'][:,2:5]})
        for r,n in [('A',4),('B',2)]:
            self.assert_pairs([(k[:,:,:n],v[:,:,:n]) for k,v in self.packed.request_kv(r)],old[r],exact=True)
        self.assert_pairs(self.packed.request_kv('C'),old['C'],exact=True)
        self.packed.crop('A',3);self.packed.crop('B',0)
        old_marker=self.packed._markers['C'];self.packed.remove_request('C');self.packed.add_request('C')
        self.assertNotEqual(old_marker,self.packed._markers['C']);self.assertFalse(bool((self.packed.key_requests==old_marker).any()))
        self.packed.append_committed({'B':self.features['B'][:,:1],'C':self.features['C'][:,:2]})
        self.assert_reference({r:torch.tensor([[7]]) for r in ['A','C','B']},
            dict(A=self.features['A'][:,:3],B=self.features['B'][:,:1],C=self.features['C'][:,:2]))
        for r,n in [('A',3),('B',1),('C',2)]:
            self.assertTrue(torch.equal(self.packed.key_positions[self.packed.key_requests==self.packed._markers[r]],torch.arange(n)))

    def test_zero_context_and_full_block_bidirectional_semantics(self):
        out=self.packed.backbone({'A':torch.tensor([[3]]),'B':torch.tensor([[4]])})
        for r,token in [('A',3),('B',4)]:
            with torch.no_grad():expected=self.draft.backbone(torch.tensor([[token]]),self.features[r][:,:0],torch.tensor([0]))[0,0]
            torch.testing.assert_close(out.for_request(r),expected,atol=2e-6,rtol=2e-5)
        q=torch.zeros(3,4,8);k=torch.zeros(5,2,8);v=torch.arange(5.).view(5,1,1).expand_as(k).contiguous()
        layout=PackedLayout.from_metadata(torch.zeros(3,dtype=torch.long),torch.arange(2,5),torch.zeros(5,dtype=torch.long),torch.arange(5))
        actual=test_only_noncausal_varlen(q,k,v,layout,scale=8**-.5)
        self.assertTrue(torch.equal(actual,torch.full_like(q,2.)))  # First query sees future block rows too.
        self.assertEqual(out.work['query_rows'],6)  # Admission is intentionally not a backbone width parameter.

    def test_same_shape_other_active_and_inactive_poison_exact_isolation(self):
        self.prime();saved=copy.deepcopy(self.packed.layers);captured=[]
        def observe(q,k,v,layout,*,scale):
            captured.append(tuple(x.clone() for x in (q[:3],k[:7],v[:7])))
            return test_only_noncausal_varlen(q,k,v,layout,scale=scale)
        self.packed.kernel=observe
        for victim,active in [('B',True),('C',False)]:
            self.packed.layers=copy.deepcopy(saved);captured.clear()
            anchors={'A':torch.tensor([[8]])}
            if active:anchors['B']=torch.tensor([[9]])
            before=self.packed.backbone(anchors).for_request('A').clone();inputs=copy.deepcopy(captured)
            selected=self.packed.key_requests==self.packed._markers[victim]
            for k,v in self.packed.layers:k[:,:,selected]=100;v[:,:,selected]=-100
            if active:anchors['B']=torch.tensor([[37]])
            captured.clear();after=self.packed.backbone(anchors).for_request('A')
            self.assertTrue(torch.equal(before,after))
            for x,y in zip(inputs,captured):
                for a,b in zip(x,y):self.assertTrue(torch.equal(a,b))
            self.assert_pairs(self.packed.request_kv('A'),[(k[:,:,:4],v[:,:,:4]) for k,v in saved],exact=True)

    def test_append_validation_and_projection_exception_are_atomic(self):
        self.prime();old=copy.deepcopy(self.packed.layers);requests=self.packed.key_requests.clone();positions=self.packed.key_positions.clone();lengths=self.packed.lengths
        with patch.object(self.draft.fc,'forward',side_effect=AssertionError('must validate first')):
            with self.assertRaises(ValueError):self.packed.append_committed({'A':self.features['A'][:,:1],'B':torch.zeros(1,2,12)})
        with patch.object(self.draft.layers[1].attention,'project_kv',side_effect=RuntimeError('injected projection')):
            with self.assertRaisesRegex(RuntimeError,'injected projection'):self.packed.append_committed({'A':self.features['A'][:,:1]})
        self.assert_pairs(old,self.packed.layers,exact=True);self.assertEqual(lengths,self.packed.lengths)
        self.assertTrue(torch.equal(requests,self.packed.key_requests));self.assertTrue(torch.equal(positions,self.packed.key_positions))

    def test_late_metadata_allocation_failure_retains_entire_old_state(self):
        self.prime()
        old_layers=self.packed.layers;old_lengths=self.packed._lengths
        old_requests=self.packed.key_requests;old_positions=self.packed.key_positions
        old_markers=dict(self.packed._markers);old_next=self.packed._next_marker
        old_work=self.packed.last_projection_work
        original=torch.cat;calls=[]
        def failing(tensors,*args,**kwargs):
            if tensors[0].ndim==1 and tensors[0].dtype==torch.long:
                calls.append(1)
                if len(calls)==4:raise RuntimeError('late metadata allocation')
            return original(tensors,*args,**kwargs)
        with patch('torch.cat',side_effect=failing):
            with self.assertRaisesRegex(RuntimeError,'late metadata allocation'):
                self.packed.append_committed({'A':self.features['A'][:,:1]})
        self.assertIs(old_layers,self.packed.layers);self.assertIs(old_lengths,self.packed._lengths)
        self.assertIs(old_requests,self.packed.key_requests);self.assertIs(old_positions,self.packed.key_positions)
        self.assertEqual(old_markers,self.packed._markers);self.assertEqual(old_next,self.packed._next_marker)
        self.assertIs(old_work,self.packed.last_projection_work)
        # Late candidate validation failure in crop must also preserve all refs.
        with patch.object(self.packed,'_validate_state',side_effect=RuntimeError('candidate validation')):
            with self.assertRaisesRegex(RuntimeError,'candidate validation'):self.packed.crop('A',1)
        self.assertIs(old_layers,self.packed.layers);self.assertIs(old_lengths,self.packed._lengths)
        self.assertIs(old_requests,self.packed.key_requests);self.assertIs(old_positions,self.packed.key_positions)

    def test_backbone_failure_does_not_append_uncommitted_block_kv(self):
        self.prime();old=copy.deepcopy(self.packed.layers);lengths=self.packed.lengths
        self.packed.kernel=Mock(side_effect=RuntimeError('operator failure'))
        with self.assertRaisesRegex(RuntimeError,'operator failure'):self.packed.backbone({'A':torch.tensor([[8]])})
        self.assert_pairs(old,self.packed.layers,exact=True);self.assertEqual(lengths,self.packed.lengths)

    def test_bf16_amp_keeps_trainable_parameters_and_rope_fp32(self):
        # Match real deployment: frozen target modules BF16; trainables stay FP32.
        self.draft.embedding.to(torch.bfloat16);self.draft.lm_head.to(torch.bfloat16)
        observed=[]
        def oracle(q,k,v,layout,*,scale):
            observed.append((q.dtype,k.dtype,v.dtype))
            return test_only_noncausal_varlen(q,k,v,layout,scale=scale)
        self.packed.kernel=oracle
        with torch.autocast('cpu',dtype=torch.bfloat16):
            self.prime()
            output=self.packed.backbone({'A':torch.tensor([[8]]),'B':torch.tensor([[9]])})
        self.assertEqual(output.hidden.dtype,torch.float32)
        self.assertTrue(bool(torch.isfinite(output.hidden).all()))
        self.assertEqual(observed,[(torch.bfloat16,)*3]*2)
        self.assertTrue(all(k.dtype==torch.float32 and v.dtype==torch.bfloat16 for k,v in self.packed.layers))
        self.assertGreater(output.work['attention_cast_elements'],0)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            for r,n,token in [('A',4,8),('B',2,9)]:
                reference=self.draft.project_context_kv(self.features[r][:,:n],0)
                self.assert_pairs(self.packed.request_kv(r),reference,exact=True)
                expected=self.draft.backbone_cached(torch.tensor([[token]]),reference,n)
                torch.testing.assert_close(output.for_request(r),expected,atol=.02,rtol=.02)
        self.assertEqual(self.draft.fc.weight.dtype,torch.float32)
        self.assertEqual(self.draft.layers[0].attention.q_proj.weight.dtype,torch.float32)
        self.assertEqual(self.draft.rotary.inv_freq.dtype,torch.float32)

    def test_native_adapter_is_explicit_noncausal_no_fallback(self):
        with self.assertRaisesRegex(ValueError,'explicit noncausal'):PackedDraft(self.draft)
        with self.assertRaisesRegex(ValueError,'CPU-only'):PackedDraft(self.draft,native_backend=DRAFT_BACKEND,test_kernel=test_only_noncausal_varlen)
        kernel=PinnedRocmDraftKernel.__new__(PinnedRocmDraftKernel);kernel._validate_inputs=Mock()
        q=torch.zeros(3,4,8);k=torch.zeros(3,2,8)
        layout=PackedLayout.from_metadata(torch.zeros(3,dtype=torch.long),torch.arange(3),torch.zeros(3,dtype=torch.long),torch.arange(3))
        kernel._operator=Mock(return_value=(q,));kernel(q,k,k,layout,scale=8**-.5)
        args=kernel._operator.call_args.args;kwargs=kernel._operator.call_args.kwargs
        self.assertIs(args[8],False);self.assertIsNone(kwargs['window_size_left']);self.assertIsNone(kwargs['window_size_right'])
        kernel._operator.side_effect=RuntimeError('native failure')
        with self.assertRaisesRegex(RuntimeError,'native failure'):kernel(q,k,k,layout,scale=8**-.5)
        self.assertEqual(kernel._operator.call_count,2)


if __name__=='__main__':unittest.main()
