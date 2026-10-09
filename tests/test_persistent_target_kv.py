import copy
import unittest
from unittest.mock import patch

import torch

from dspark_qwen.persistent_target_kv import Bucket, PersistentTargetKV, Slot


class PersistentTargetKVTests(unittest.TestCase):
    def pool(self):
        return PersistentTargetKV(layers=2,slots=3,context_capacity=24,max_query_tokens=8,heads=2,dim=4)

    def prefix(self, pool, handle, n, offset):
        x = torch.arange(pool.layers*n*pool.heads*pool.dim,dtype=pool.dtype).reshape(pool.layers,n,pool.heads,pool.dim)+offset
        pool.load_prefix(handle,x,x+0.25)
        return x, x+0.25

    def stage(self,pool,tx):
        q = tx.bucket.query_tokens
        x = torch.arange(pool.layers*q*pool.heads*pool.dim,dtype=pool.dtype).reshape(pool.layers,q,pool.heads,pool.dim)+10000
        for layer in range(pool.layers):
            pool.stage_layer(tx,layer,x[layer],x[layer]+0.5)
        return x, x+0.5

    def test_indexed_device_alias_uses_actual_allocated_device(self):
        pool=PersistentTargetKV(layers=1,slots=1,context_capacity=4,max_query_tokens=1,heads=1,dim=2,device='cpu:0')
        self.assertEqual(pool.device,pool.keys.device)
        handle=pool.add_request('r0');x=torch.ones((1,1,1,2),device='cpu:0')
        pool.load_prefix(handle,x,x)
        bucket=Bucket((1,),(3,),'indexed CPU device regression');pool.register_bucket(bucket)
        tx=pool.begin([handle],bucket);pool.stage_layer(tx,0,x[0],x[0])
        k,_,_,ck,_=pool.prepare_attention(tx,0)
        self.assertEqual(k.device,pool.device);self.assertEqual(ck.tolist(),[0,2])
        pool.commit(tx,{handle:1});self.assertEqual(pool.length(handle),2)

    def test_growing_context_stable_buffers_exact_q_and_independent_scratch(self):
        pool=self.pool();a=pool.add_request('A');b=pool.add_request('B');idle=pool.add_request('idle')
        ak,av=self.prefix(pool,a,3,100);bk,bv=self.prefix(pool,b,5,500);ik,iv=self.prefix(pool,idle,9,900)
        bucket=Bucket((2,3),(12,12),'CPU fixture; no measured native applicability')
        pool.register_bucket(bucket);pointers=pool.pointers(bucket)
        before=pool.keys.clone();tx=pool.begin([a,b],bucket);sk,sv=self.stage(pool,tx)
        self.assertTrue(torch.equal(before,pool.keys))
        for layer in range(pool.layers):
            k,v,cq,ck,pos=pool.prepare_attention(tx,layer)
            expected_k=torch.cat((ak[layer],sk[layer,:2],bk[layer],sk[layer,2:]))
            expected_v=torch.cat((av[layer],sv[layer,:2],bv[layer],sv[layer,2:]))
            self.assertTrue(torch.equal(k[:13],expected_k));self.assertTrue(torch.equal(v[:13],expected_v))
            self.assertTrue(torch.equal(k[13:],torch.zeros_like(k[13:])))
            self.assertEqual(cq.tolist(),[0,2,5]);self.assertEqual(ck.tolist(),[0,5,13]);self.assertEqual(pos.tolist(),[3,4,5,6,7])
        work=pool.work(tx)
        self.assertEqual(work['logical_query_tokens'],work['physical_query_tokens'])
        self.assertEqual(work['inactive_resident_key_tokens'],9)
        self.assertEqual(work['physical_key_capacity'],29);self.assertEqual(work['padded_key_capacity'],16)
        pool.commit(tx,{a:1,b:2})
        self.assertTrue(torch.equal(pool.request_kv(a)[0],torch.cat((ak,sk[:,:1]),dim=1)))
        self.assertTrue(torch.equal(pool.request_kv(b)[0],torch.cat((bk,sk[:,2:4]),dim=1)))
        self.assertTrue(torch.equal(pool.request_kv(idle)[0],ik))
        self.assertEqual(pool.pointers(bucket),pointers)
        second=pool.begin([a,b],bucket);self.stage(pool,second)
        _,_,_,ck,pos=pool.prepare_attention(second,0)
        self.assertEqual(ck.tolist(),[0,6,16]);self.assertEqual(pos.tolist(),[4,5,7,8,9])
        self.assertEqual(pool.pointers(bucket),pointers)
        pool.abort(second)

    def test_abort_no_commit_and_slot_reuse_cannot_read_old_bytes(self):
        pool=self.pool();old=pool.add_request('same');self.prefix(pool,old,7,77)
        bucket=Bucket((1,),(10,),'CPU fixture');pool.register_bucket(bucket)
        before=pool.keys.clone();tx=pool.begin([old],bucket);self.stage(pool,tx);pool.abort(tx)
        self.assertTrue(torch.equal(before,pool.keys));self.assertEqual(pool.length(old),7)
        pool.remove_request(old);fresh=pool.add_request('same')
        self.assertEqual(old.index,fresh.index);self.assertNotEqual(old.incarnation,fresh.incarnation)
        with self.assertRaises(ValueError):pool.length(old)
        with self.assertRaises(ValueError):pool.length(copy.copy(fresh))
        with self.assertRaises(ValueError):pool.length(Slot(fresh.request,fresh.index,fresh.incarnation))
        other=self.pool();foreign=other.add_request('same')
        with self.assertRaises(ValueError):pool.length(foreign)
        tx=pool.begin([fresh],bucket);sk,_=self.stage(pool,tx);k,_,_,ck,pos=pool.prepare_attention(tx,0)
        self.assertEqual(ck.tolist(),[0,1]);self.assertEqual(pos.tolist(),[0])
        self.assertTrue(torch.equal(k[:1],sk[0]));self.assertEqual(int(torch.count_nonzero(k[1:])),0)
        pool.commit(tx,{fresh:1});self.assertEqual(pool.length(fresh),1)

    def test_invalid_transaction_and_commit_fail_before_resident_mutation(self):
        pool=self.pool();a=pool.add_request('A');self.prefix(pool,a,2,1)
        bucket=Bucket((2,),(4,),'CPU fixture');pool.register_bucket(bucket)
        before=pool.keys.clone();tx=pool.begin([a],bucket)
        with self.assertRaises(RuntimeError):pool.remove_request(a)
        with self.assertRaises(RuntimeError):pool.add_request('B')
        with self.assertRaises(RuntimeError):pool.begin([a],bucket)
        with self.assertRaises(ValueError):pool.prepare_attention(tx,0)
        with self.assertRaises(ValueError):pool.commit(tx,{a:1})
        self.stage(pool,tx)
        with self.assertRaises(ValueError):pool.commit(tx,{a:3})
        with self.assertRaises(ValueError):pool.commit(tx,{a:True})
        with self.assertRaises(ValueError):pool.commit(copy.copy(tx),{a:1})
        self.assertTrue(torch.equal(before,pool.keys))
        pool.commit(tx,{a:0});self.assertTrue(torch.equal(before,pool.keys))
        with self.assertRaises(ValueError):pool.abort(tx)

    def test_failed_metadata_preparation_never_publishes_or_revives_transaction(self):
        pool=self.pool();a=pool.add_request('A');self.prefix(pool,a,3,3)
        bucket=Bucket((2,),(8,),'CPU fixture');pool.register_bucket(bucket)
        old=pool.begin([a],bucket);pool.abort(old);before=pool.keys.clone()
        original=torch.tensor;calls=[]
        def fail(*args,**kwargs):
            calls.append(None)
            if len(calls)==3:raise RuntimeError('injected metadata allocation failure')
            return original(*args,**kwargs)
        with patch('torch.tensor',side_effect=fail),self.assertRaisesRegex(RuntimeError,'injected'):
            pool.begin([a],bucket)
        self.assertIsNone(pool._pending);self.assertTrue(torch.equal(pool.keys,before))
        with self.assertRaises(ValueError):pool.abort(old)
        tx=pool.begin([a],bucket);self.stage(pool,tx)
        _,_,cq,ck,pos=pool.prepare_attention(tx,0)
        self.assertEqual(cq.tolist(),[0,2]);self.assertEqual(ck.tolist(),[0,5]);self.assertEqual(pos.tolist(),[3,4])
        pool.abort(tx)

    def test_ordered_shape_bounds_and_active_subset(self):
        pool=self.pool();a=pool.add_request('A');b=pool.add_request('B');self.prefix(pool,a,8,0);self.prefix(pool,b,4,0)
        small=Bucket((1,),(7,),'CPU fixture');pool.register_bucket(small)
        with self.assertRaises(ValueError):pool.begin([a],small)
        pair=Bucket((2,1),(8,8),'CPU fixture');pool.register_bucket(pair)
        with self.assertRaises(ValueError):pool.begin([a,a],pair)
        with self.assertRaises(ValueError):pool.begin([a],pair)
        tx=pool.begin([b,a],pair);self.stage(pool,tx);_,_,cq,ck,pos=pool.prepare_attention(tx,1)
        self.assertEqual(cq.tolist(),[0,2,3]);self.assertEqual(ck.tolist(),[0,6,15]);self.assertEqual(pos.tolist(),[4,5,8])
        pool.abort(tx)
        tx=pool.begin([b],small);self.stage(pool,tx);self.assertEqual(pool.work(tx)['inactive_resident_key_tokens'],8)
        pool.abort(tx)
        with self.assertRaises(ValueError):pool.register_bucket(Bucket((8,8),(8,8),'fixture'))
        with self.assertRaises(ValueError):Bucket((1,),(8,),'')

    def test_attention_staging_matches_independent_variable_prefix_sdpa(self):
        pool=self.pool();a=pool.add_request('A');b=pool.add_request('B')
        ak,av=self.prefix(pool,a,3,0);bk,bv=self.prefix(pool,b,5,1)
        bucket=Bucket((2,1),(10,10),'CPU fixture');pool.register_bucket(bucket)
        tx=pool.begin([a,b],bucket);sk,sv=self.stage(pool,tx)
        torch.manual_seed(1);queries=torch.randn(3,4,4)
        k,v,cq,ck,_=pool.prepare_attention(tx,0)
        for i,(pk,pv,off,n) in enumerate(((ak[0],av[0],0,2),(bk[0],bv[0],2,1))):
            true_k=torch.cat((pk,sk[0,off:off+n]));true_v=torch.cat((pv,sv[0,off:off+n]))
            q=queries[off:off+n].transpose(0,1)[None]
            visible=torch.arange(len(true_k))[None,:] <= torch.arange(len(pk),len(pk)+n)[:,None]
            def attention(kk,vv):
                return torch.nn.functional.scaled_dot_product_attention(q,kk.transpose(0,1)[None],vv.transpose(0,1)[None],attn_mask=visible[None,None],enable_gqa=True)
            self.assertTrue(torch.equal(attention(k[ck[i]:ck[i+1]],v[ck[i]:ck[i+1]]),attention(true_k,true_v)))
        pool.abort(tx)

    def test_real_tiny_qwen_kv_commit_matches_independent_prefix_forward(self):
        # Existing local HF runtime suffices for storage semantics; production is
        # pinned separately to 5.17. This test does not validate native dispatch.
        from transformers import Qwen3Config,Qwen3ForCausalLM
        torch.manual_seed(340);torch.set_num_threads(2)
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,attention_dropout=0.)
        config._attn_implementation='sdpa';model=Qwen3ForCausalLM(config).eval()
        pool=PersistentTargetKV(layers=2,slots=2,context_capacity=24,max_query_tokens=5,heads=2,dim=8)
        a=pool.add_request('A');b=pool.add_request('B');handles=(a,b)
        prefixes=(torch.tensor([[1,2,3]]),torch.tensor([[4,5,6,7,8]]))
        chunks=(torch.tensor([[9,10]]),torch.tensor([[11,12,13]]));old=[];new=[]
        def extract(cache):
            return tuple(torch.stack([getattr(layer,n)[0].transpose(0,1) for layer in cache.layers]) for n in ('keys','values'))
        with torch.no_grad():
            for h,prompt,chunk in zip(handles,prefixes,chunks):
                out=model(prompt,use_cache=True);k,v=extract(out.past_key_values);old.append((k.clone(),v.clone()));pool.load_prefix(h,k,v)
                verified=model(chunk,past_key_values=out.past_key_values,use_cache=True)
                k,v=extract(verified.past_key_values);new.append((k[:,prompt.shape[1]:],v[:,prompt.shape[1]:]))
            bucket=Bucket((2,3),(12,12),'tiny actual Qwen CPU reference');pool.register_bucket(bucket)
            tx=pool.begin(handles,bucket);snapshot=pool.keys.clone()
            for layer in range(pool.layers):
                pool.stage_layer(tx,layer,torch.cat([x[0][layer] for x in new]),torch.cat([x[1][layer] for x in new]))
                k,v,_,ck,_=pool.prepare_attention(tx,layer)
                for i in range(2):
                    self.assertTrue(torch.equal(k[ck[i]:ck[i+1]],torch.cat((old[i][0][layer],new[i][0][layer]))))
            self.assertTrue(torch.equal(snapshot,pool.keys));pool.commit(tx,{a:1,b:2})
            for h,prompt,chunk,n in zip(handles,prefixes,chunks,(1,2)):
                expected=model(torch.cat((prompt,chunk[:,:n]),dim=1),use_cache=True)
                ek,ev=extract(expected.past_key_values);actual=pool.request_kv(h)
                torch.testing.assert_close(actual[0],ek,rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(actual[1],ev,rtol=1e-5,atol=1e-6)


if __name__=='__main__':unittest.main()
