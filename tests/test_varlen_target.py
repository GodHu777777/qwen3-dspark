import copy
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dspark_qwen.packed_target import PackedTarget
from dspark_qwen.varlen_target import PackedLayout, VarlenPackedTarget, native_varlen, test_only_dense_varlen


def ids(*tokens):
    return torch.tensor([tokens],dtype=torch.long)


class VarlenTargetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(302)
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,
            head_dim=8,max_position_embeddings=128,attention_dropout=0.0)
        config._attn_implementation='sdpa'
        model=Qwen3ForCausalLM(config).eval()
        self.dense=PackedTarget(model,(0,1))
        self.calls=[]
        def spy(q,k,v,layout,*,scale):
            self.calls.append((tuple(q.shape),tuple(k.shape),layout))
            return test_only_dense_varlen(q,k,v,layout,scale=scale)
        self.varlen=VarlenPackedTarget(copy.deepcopy(model),(0,1),test_kernel=spy)

    def add(self,name):
        self.dense.add_request(name);self.varlen.add_request(name)

    def compare(self,chunks):
        self.calls.clear()
        forwards=[]
        hook=self.varlen.model.model.register_forward_pre_hook(lambda _m,_a:forwards.append(1))
        try:
            actual=self.varlen.append(chunks)
        finally:
            hook.remove()
        expected=self.dense.append(chunks)
        self.assertEqual(forwards,[1])
        self.assertEqual(len(self.calls),3)
        self.assertEqual(actual.work['physical_query_tokens'],sum(x.shape[1] for x in chunks.values()))
        self.assertEqual(actual.work['mask_bytes'],0)
        self.assertIsNone(actual.work['mask_shape'])
        self.assertEqual(actual.work['native_varlen_calls_per_layer'],0)
        self.assertEqual(actual.work['attention_backend'],'test_only_cpu_dense_oracle')
        active_k=sum(self.dense.lengths[name] for name in chunks)
        for shape,kshape,layout in self.calls:
            self.assertEqual(shape[0],actual.work['physical_query_tokens'])
            self.assertEqual(kshape[0],active_k)
            self.assertEqual(kshape[1],2)  # native GQA layout retains KV heads
        torch.testing.assert_close(actual.last,expected.last,atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(actual.context,expected.context,atol=2e-6,rtol=1e-5)
        for name,logits in self.varlen.predict(actual).items():
            torch.testing.assert_close(logits,self.dense.predict(expected)[name],atol=2e-6,rtol=1e-5)
        self.assertEqual(self.varlen.lengths,self.dense.lengths)
        for name in self.dense.lengths:
            for (ka,va),(ke,ve) in zip(self.varlen.request_kv(name),self.dense.request_kv(name)):
                torch.testing.assert_close(ka,ke,atol=2e-6,rtol=1e-5)
                torch.testing.assert_close(va,ve,atol=2e-6,rtol=1e-5)
        return actual

    def test_layout_active_gather_order_and_bottom_right_causality(self):
        layout=PackedLayout.from_metadata(torch.tensor([2,2,1]),torch.tensor([2,3,2]),
            torch.tensor([1,1,2,2,3,3,1,2,2]),torch.tensor([0,1,0,1,0,1,2,2,3]))
        self.assertEqual(layout.query_lengths,(2,1))
        self.assertEqual(layout.key_lengths,(4,3))
        self.assertEqual(layout.cu_query.tolist(),[0,2,3])
        self.assertEqual(layout.cu_key.tolist(),[0,4,7])
        self.assertEqual(layout.gather_indices.tolist(),[2,3,7,8,0,1,6])
        self.assertEqual(layout.pair_domain,11)
        physical=torch.arange(9,dtype=torch.float32).view(1,1,9,1)
        k,v=layout.gather_kv(physical,physical)
        q=torch.zeros(3,1,1)
        # Zero Q gives uniform attention: B position 2 sees values [2,3,7],
        # B position 3 sees [2,3,7,8], A position 2 sees [0,1,6].
        output=test_only_dense_varlen(q,k,v,layout,scale=1)
        torch.testing.assert_close(output.flatten(),torch.tensor([4.,5.,7/3]),atol=1e-6,rtol=0)

    def test_real_qwen_mixed_queries_inactive_kv_and_lifecycle(self):
        for name in ['a','b','idle']:self.add(name)
        self.compare({'a':ids(1,2,3),'b':ids(4),'idle':ids(5,6)})
        batch=self.compare({'b':ids(7,8,9),'a':ids(10)})
        self.assertEqual(batch.work['inactive_kv_tokens_excluded'],2)
        self.assertEqual(batch.work['active_kv_gather_tokens'],8)
        self.assertEqual(batch.work['varlen_pair_domain_per_head_layer'],16)
        for target in [self.dense,self.varlen]:
            target.crop('a',3);target.crop('b',2);target.remove_request('idle')
        self.add('idle')
        self.compare({'idle':ids(20),'a':ids(21),'b':ids(22,23)})
        for target in [self.dense,self.varlen]:target.crop('a',0)
        self.compare({'a':ids(31,32)})

    def test_poison_other_request_does_not_affect_active_query(self):
        for name in ['a','b']:self.add(name)
        self.compare({'a':ids(1,2),'b':ids(3,4)})
        selected=self.varlen.key_requests==self.varlen._markers['b']
        for layer in self.varlen.cache.layers:
            layer.keys[:,:,selected]=1000;layer.values[:,:,selected]=-1000
        actual=self.varlen.append({'a':ids(5)})
        expected=self.dense.append({'a':ids(5)})
        torch.testing.assert_close(actual.last,expected.last,atol=2e-6,rtol=1e-5)
        self.assertEqual(actual.work['inactive_kv_tokens_excluded'],2)

    def test_no_native_cpu_fallback_and_failed_kernel_clears_state(self):
        native_model=copy.deepcopy(self.dense.model)
        target=VarlenPackedTarget(native_model)
        target.add_request('a')
        with self.assertRaisesRegex(RuntimeError,'Native varlen requires'):
            target.append({'a':ids(1)})
        self.assertEqual(target.lengths,{})
        self.add('a');self.compare({'a':ids(1,2)})
        def fail(*args,**kwargs):raise RuntimeError('unsupported kernel fixture')
        self.varlen._varlen_kernel=fail
        with self.assertRaisesRegex(RuntimeError,'unsupported kernel fixture'):
            self.varlen.append({'a':ids(3)})
        self.assertEqual(self.varlen.lengths,{})

    def test_invalid_nonterminal_or_discontiguous_layout(self):
        for qreq,qpos,kreq,kpos in [([1,2,1],[0,0,1],[1,1,2],[0,1,0]),
                ([1],[0],[1,1],[0,1]),([1],[1],[1,1],[0,2]),([],[],[],[])]:
            with self.subTest(qreq=qreq),self.assertRaises(ValueError):
                PackedLayout.from_metadata(*(torch.tensor(x,dtype=torch.long) for x in [qreq,qpos,kreq,kpos]))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
