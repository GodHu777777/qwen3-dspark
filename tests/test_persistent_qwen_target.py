import copy
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from dspark_qwen.persistent_target_kv import Bucket
from dspark_qwen.persistent_qwen_target import PersistentQwenTarget,test_only_persistent_sdpa


LAYERS=(0,2,3)


def model():
    torch.manual_seed(710);torch.set_num_threads(2)
    c=Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=48,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,attention_dropout=0.)
    c._attn_implementation='sdpa'
    return Qwen3ForCausalLM(c).eval().requires_grad_(False)


@torch.no_grad()
def reference(m,ids,cache=None):
    captured={};hooks=[]
    for i in LAYERS:
        def hook(_module,_inputs,output,index=i):captured[index]=output[0] if isinstance(output,tuple) else output
        hooks.append(m.model.layers[i].register_forward_hook(hook))
    try:
        out=m.model(input_ids=ids,past_key_values=copy.deepcopy(cache),use_cache=True,return_dict=True)
        return dict(last=out.last_hidden_state,context=torch.cat([captured[i] for i in LAYERS],dim=-1),
                    logits=m.lm_head(out.last_hidden_state),cache=out.past_key_values,raw_final_block=captured[3])
    finally:
        for h in hooks:h.remove()


class PersistentQwenTargetTests(unittest.TestCase):
    def fixture(self):
        original=model();ref=copy.deepcopy(original)
        target=PersistentQwenTarget(original,LAYERS,slots=3,context_capacity=24,max_query_tokens=12,
            test_kernel=test_only_persistent_sdpa)
        buckets=dict(prefill=Bucket((3,5,4),(0,0,0),'CPU full-model prompt fixture'),
                     verify=Bucket((2,3),(16,16),'CPU full-model growing-context fixture'),
                     single=Bucket((1,),(20,),'CPU active-subset fixture'),
                     readmit=Bucket((2,),(0,),'CPU reincarnation fixture'))
        for b in buckets.values():target.register_bucket(b)
        for r in ('A','B','C'):target.add_request(r)
        return target,ref,buckets

    def prompts(self,poison=False):
        return dict(A=torch.tensor([[1,2,3]]),B=torch.tensor([[4,5,6,7,8]]),C=torch.tensor([[80,81,82,83] if poison else [20,21,22,23]]))

    def close_features(self,target,features,expected):
        full=target.predict(features);last=target.predict(features,last_only=True)
        for r,want in expected.items():
            actual=features.for_request(r)
            torch.testing.assert_close(actual.last,want['last'],rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(actual.context,want['context'],rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(full[r],want['logits'],rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(last[r],want['logits'][:,-1],rtol=2e-5,atol=2e-6)

    def close_kv(self,target,request,cache):
        for (k,v),layer in zip(target.request_kv(request),cache.layers):
            torch.testing.assert_close(k,layer.keys,rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(v,layer.values,rtol=2e-5,atol=2e-6)

    def test_full_model_features_final_norm_logits_partial_commit_and_growth(self):
        target,ref,b=self.fixture();self.assertFalse(target.cache.is_compileable);prompts=self.prompts();expected={r:reference(ref,x) for r,x in prompts.items()}
        features=target.prefill(prompts,bucket=b['prefill']);self.close_features(target,features,expected)
        self.assertEqual(target.lengths,{'A':3,'B':5,'C':4})
        for r in prompts:self.close_kv(target,r,expected[r]['cache'])
        # Selected final decoder block is pre-final-norm; features.last is after
        # original model norm. Using output_hidden_states[-1] for both is wrong.
        raw=features.for_request('A').context[:,:,-32:]
        self.assertFalse(torch.allclose(raw,features.for_request('A').last))
        torch.testing.assert_close(raw,expected['A']['raw_final_block'],rtol=2e-5,atol=2e-6)
        cache={r:x['cache'] for r,x in expected.items()};pointers=target.pool.pointers(b['verify'])
        for chunks,counts in [(dict(A=torch.tensor([[9,10]]),B=torch.tensor([[11,12,13]])),{'A':1,'B':2}),
                              (dict(A=torch.tensor([[14,15]]),B=torch.tensor([[16,17,18]])),{'A':2,'B':3})]:
            before=target.lengths.copy();stored=(target.pool.keys.clone(),target.pool.values.clone())
            expected={r:reference(ref,x,cache[r]) for r,x in chunks.items()}
            features=target.verify(chunks,bucket=b['verify']);self.close_features(target,features,expected)
            self.assertEqual(target.lengths,before)
            self.assertTrue(torch.equal(stored[0],target.pool.keys));self.assertTrue(torch.equal(stored[1],target.pool.values))
            self.assertEqual(features.work['inactive_resident_key_tokens'],4)
            self.assertEqual(features.work['physical_query_tokens'],5)
            self.assertEqual(features.work['query_lengths'],[2,3])
            self.assertEqual(features.work['queried_context_lengths'],[before['A'],before['B']])
            with self.assertRaises(RuntimeError):target.remove_request('C')
            with self.assertRaises(ValueError):target.commit(copy.copy(features),counts)
            with self.assertRaises(ValueError):target.commit(features,{'A':99,'B':0})
            target.commit(features,counts)
            for r,n in counts.items():
                cache[r]=expected[r]['cache'];cache[r].crop(before[r]+n);self.close_kv(target,r,cache[r])
            self.close_kv(target,'C',cache['C'])
            self.assertEqual(pointers,target.pool.pointers(b['verify']))
            with self.assertRaises(ValueError):target.abort(features)
        self.assertEqual(target.lengths,{'A':6,'B':10,'C':4})
        target.close();self.assertEqual(target.model.config._attn_implementation,'sdpa')

    def test_abort_failed_forward_preserve_every_resident_and_allow_retry(self):
        target,ref,b=self.fixture();target.prefill(self.prompts(),bucket=b['prefill'])
        keys=target.pool.keys.clone();values=target.pool.values.clone();lengths=target.lengths.copy()
        features=target.verify({'A':torch.tensor([[30]])},bucket=b['single']);target.abort(features)
        self.assertTrue(torch.equal(keys,target.pool.keys));self.assertTrue(torch.equal(values,target.pool.values))
        with patch.object(target.model.model.layers[1],'forward',side_effect=RuntimeError('injected layer failure')):
            with self.assertRaisesRegex(RuntimeError,'injected layer'):
                target.verify({'A':torch.tensor([[31]])},bucket=b['single'])
        self.assertIsNone(target._pending);self.assertIsNone(target.pool._pending)
        self.assertEqual(target.lengths,lengths)
        self.assertTrue(torch.equal(keys,target.pool.keys));self.assertTrue(torch.equal(values,target.pool.values))
        features=target.verify({'A':torch.tensor([[31]])},bucket=b['single']);target.commit(features,{'A':1})
        self.assertEqual(target.lengths,{'A':4,'B':5,'C':4});target.close()

    def test_inactive_poison_and_slot_reincarnation_have_no_cross_request_effect(self):
        one,_,b1=self.fixture();two,_,b2=self.fixture()
        one.prefill(self.prompts(),bucket=b1['prefill']);two.prefill(self.prompts(poison=True),bucket=b2['prefill'])
        f1=one.verify({'A':torch.tensor([[41]])},bucket=b1['single']);f2=two.verify({'A':torch.tensor([[41]])},bucket=b2['single'])
        torch.testing.assert_close(f1.last,f2.last,rtol=0,atol=0)
        torch.testing.assert_close(f1.context,f2.context,rtol=0,atol=0)
        torch.testing.assert_close(one.predict(f1)['A'],two.predict(f2)['A'],rtol=0,atol=0)
        one.abort(f1);two.abort(f2)
        old=one._handles['C'];before=one.request_kv('A');one.remove_request('C');fresh=one.add_request('C')
        self.assertEqual(old.index,fresh.index);self.assertNotEqual(old.incarnation,fresh.incarnation)
        with self.assertRaises(ValueError):one.pool.length(old)
        one.prefill({'C':torch.tensor([[90,91]])},bucket=b1['readmit'])
        self.assertEqual(one.lengths['C'],2)
        for (k,v),(bk,bv) in zip(one.request_kv('A'),before):
            self.assertTrue(torch.equal(k,bk));self.assertTrue(torch.equal(v,bv))
        one.close();two.close()

    def test_explicit_kernel_exact_bucket_and_reset_contract(self):
        m=model()
        with self.assertRaisesRegex(ValueError,'explicit CPU'):PersistentQwenTarget(m,LAYERS,slots=1,context_capacity=8,max_query_tokens=3)
        target,_,b=self.fixture();target.prefill(self.prompts(),bucket=b['prefill']);pointers=target.pool.pointers(b['single'])
        with self.assertRaises(ValueError):target.verify({'A':torch.tensor([[1,2]])},bucket=b['single'])
        with self.assertRaises(ValueError):target.prefill({'A':torch.tensor([[1]])},bucket=b['single'])
        with self.assertRaises(ValueError):target.verify({'A':torch.tensor([[999]])},bucket=b['single'])
        target.reset();self.assertEqual(target.lengths,{})
        target.add_request('fresh');target.prefill({'fresh':torch.tensor([[2,3]])},bucket=b['readmit'])
        self.assertEqual(target.pool.pointers(b['single']),pointers)
        self.assertEqual(target.lengths,{'fresh':2});target.close()
        with self.assertRaises(RuntimeError):target.reset()

if __name__=='__main__':unittest.main()
