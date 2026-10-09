"""Finite actual-B families: real CPU Qwen/session + explicit replay emulator.

Native argument spies prove host launch bounds only, never GPU capture behavior.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dspark_qwen.persistent_target_kv import Bucket, QueryFamily, PersistentTargetKV
from dspark_qwen.persistent_qwen_target import PersistentQwenTarget, test_only_persistent_sdpa
from dspark_qwen.persistent_qwen_graph import CPUReplayEmulator, FullTargetExecution
from dspark_qwen.target_strategy import FiniteTargetBuckets, PersistentTargetStrategy
from test_persistent_qwen_graph import model, ManualBackend
from test_persistent_sampling import RecordedRandom


def families():
    return (QueryFamily(2,3,2,24,16,'small actual B'),
            QueryFamily(2,6,6,32,16,'varying actual ordered Q'))


class QueryFamilyTests(unittest.TestCase):
    def test_bounds_actual_offsets_reorder_and_shared_arena(self):
        p=PersistentTargetKV(layers=1,slots=3,context_capacity=16,max_query_tokens=6,heads=1,dim=1)
        small,large=families();p.register_families((small,large))
        handles=[p.add_request(r) for r in ('a','b','idle')]
        for i,h in enumerate(handles):p.load_prefix(h,torch.full((1,i+1,1,1),float(i)),torch.full((1,i+1,1,1),float(i)))
        pointers={f:p.pointers(f) for f in (small,large)}
        self.assertEqual(pointers[small],pointers[large])
        before=p.keys.clone()
        for queries in ((0,6),(2,3),(1,6),(True,5),(1,1,4)):
            with self.assertRaises(ValueError):p.begin(handles[:2],large,query_lengths=queries)
            self.assertIsNone(p._pending)
        with self.assertRaises(ValueError):p.begin(handles[:2],large)
        with self.assertRaises(ValueError):p.register_families((small,))
        with self.assertRaises(ValueError):p.register_bucket(QueryFamily(1,1,1,4,4,'undeclared'))
        self.assertTrue(torch.equal(before,p.keys))
        for family,queries,active in ((small,(1,2),handles[:2]),(large,(5,1),handles[1::-1]),(small,(2,1),handles[:2])):
            contexts=tuple(p.length(h) for h in active)
            tx=p.begin(active,family,query_lengths=queries)
            self.assertEqual(tx.query_lengths,queries)
            values=torch.arange(family.query_tokens,dtype=torch.float32).reshape(-1,1,1)+20
            p.stage_layer(tx,0,values,values+1)
            _,_,cuq,cuk,pos=p.prepare_attention(tx,0)
            self.assertEqual(cuq.tolist(),[0,queries[0],sum(queries)])
            self.assertEqual(cuk.tolist(),[0,contexts[0]+queries[0],sum(contexts)+sum(queries)])
            self.assertEqual(pos.tolist(),[i for c,q in zip(contexts,queries) for i in range(c,c+q)])
            work=p.work(tx)
            self.assertEqual(work['physical_query_tokens'],sum(queries))
            self.assertEqual(work['logical_query_tokens'],sum(queries))
            self.assertEqual(work['shared_arena_allocated_bytes'],p.workspace_bytes(p._family_arena.bucket))
            old={h:p.request_kv(h)[0] for h in active};counts={h:min(q,2) for h,q in zip(active,queries)}
            p.commit(tx,counts);offset=0
            for h,q in zip(active,queries):
                expected=torch.cat((old[h],values[offset:offset+counts[h]][None]),dim=1)
                self.assertTrue(torch.equal(p.request_kv(h)[0],expected));offset+=q
            self.assertEqual(p.pointers(family),pointers[family])
        self.assertTrue(torch.equal(p.request_kv(handles[2])[0],before[:,2,:3]))
        old=handles[0];p.remove_request(old);fresh=p.add_request('a')
        with self.assertRaises(ValueError):p.begin((old,handles[1]),small,query_lengths=(1,2))
        tx=p.begin((fresh,handles[1]),small,query_lengths=(1,2));self.assertEqual(tx.context_lengths[0],0);p.abort(tx)
        tight=QueryFamily(2,6,6,7,6,'context bounds')
        self.assertFalse(tight.accepts((3,3),(2,0)))  # total K bound
        self.assertFalse(tight.accepts((5,1),(2,0)))  # per-entry maxK bound
        for args in ((2,1,1,8,8),(2,6,2,8,8),(2,6,6,5,8),(2,6,6,8,5),(True,6,6,8,8)):
            with self.assertRaises(ValueError):QueryFamily(*args,'invalid')

    def fixture(self,backend=None):
        original=model();small,large=families()
        eager=PersistentQwenTarget(copy.deepcopy(original),(0,2,3),slots=3,context_capacity=24,
            max_query_tokens=12,test_kernel=test_only_persistent_sdpa)
        graph=PersistentQwenTarget(original,(0,2,3),slots=3,context_capacity=24,max_query_tokens=12,
            test_kernel=test_only_persistent_sdpa,graph_backend=backend or CPUReplayEmulator(),
            max_graph_buckets=2,graph_byte_budget=1024**2)
        prefill=Bucket((3,4,2),(0,0,0),'family test admission')
        for target in (eager,graph):
            target.register_bucket(prefill)
            for r in ('a','b','idle'):target.add_request(r)
            f=target.prefill(dict(a=torch.tensor([[1,2,3]]),b=torch.tensor([[4,5,6,7]]),idle=torch.tensor([[30,31]])),bucket=prefill)
            target.release_features(f)
        graph.register_families((small,large))
        for family,qs in ((small,(1,2)),(large,(1,5))):
            graph.register_graph_bucket(family,reserve_bytes=4096)
            graph.capture_graph(self.chunks(qs),bucket=family)
        return eager,graph,small,large

    def chunks(self,qs,names=('a','b')):
        return {r:torch.arange(40+i*10,40+i*10+q)[None] for i,(r,q) in enumerate(zip(names,qs))}

    def test_full_model_changing_q_and_b_matches_exact_eager_all_kv(self):
        eager,graph,small,large=self.fixture()
        pointers={f:graph._graphs[f].pointers() for f in (small,large)}
        self.assertEqual(pointers[small]['keys'],pointers[large]['keys'])
        self.assertNotEqual(pointers[small]['last'],pointers[large]['last'])
        for family,qs,names in ((small,(1,2),('a','b')),(large,(1,5),('a','b')),
                                (large,(3,3),('b','a')),(large,(5,1),('a','b')),(small,(2,1),('a','b'))):
            chunks=self.chunks(qs,names);exact=Bucket(qs,(16-max(qs),)*2,'independent exact eager control')
            eager.register_bucket(exact)
            f=eager.verify(chunks,bucket=exact);g=graph.verify(chunks,bucket=family)
            self.assertEqual(f.spans,g.spans)
            torch.testing.assert_close(f.last,g.last,rtol=0,atol=0)
            torch.testing.assert_close(f.context,g.context,rtol=0,atol=0)
            for r in chunks:torch.testing.assert_close(eager.predict(f)[r],graph.predict(g)[r],rtol=0,atol=0)
            self.assertEqual(g.work['query_lengths'],list(qs));self.assertEqual(g.work['physical_query_tokens'],sum(qs))
            self.assertEqual(g.work['maximum_query_length'],family.maximum_query_length)
            counts={r:1 for r in chunks};eager.commit(f,counts);graph.commit(g,counts)
            with self.assertRaises(RuntimeError):graph.prepare_graph_verify(self.chunks((1,2)),bucket=small)
            with self.assertRaises(RuntimeError):graph.capture_graph(self.chunks((1,2)),bucket=small)
            for r in ('a','b','idle'):
                for a,b in zip(eager.request_kv(r),graph.request_kv(r)):
                    for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
            eager.release_features(f);graph.release_features(g)
            self.assertEqual(graph._graphs[family].pointers(),pointers[family])
        eager.close();graph.close()

    def test_native_arguments_use_fixed_family_bounds_not_current_maxima(self):
        p=PersistentTargetKV(layers=1,slots=2,context_capacity=16,max_query_tokens=6,heads=8,dim=128,dtype=torch.bfloat16)
        family=families()[1];p.register_families((family,));handles=[p.add_request(r) for r in ('a','b')]
        calls=[]
        def operator(*args,**kwargs):calls.append(args);return (torch.empty_like(args[0]),)
        owner=SimpleNamespace(pool=p,_varlen_kernel=SimpleNamespace(_operator=operator))
        for qs in ((1,5),(3,3),(5,1)):
            tx=p.begin(handles,family,query_lengths=qs);layout=PersistentQwenTarget._layout_for(owner,tx)
            self.assertEqual(layout.maximum_query_length,6);self.assertEqual(layout.maximum_key_length,16)
            self.assertLess(max(qs),layout.maximum_query_length)
            q=torch.empty((6,16,128),dtype=torch.bfloat16);k=torch.empty((32,8,128),dtype=torch.bfloat16)
            PersistentQwenTarget._native_attention(owner,q,k,k,layout,scale=128**-.5)
            self.assertEqual(calls[-1][5:7],(6,16))
            self.assertIs(calls[-1][3],p._workspaces[family].cu_query)
            p.abort(tx)

    def test_shared_arena_budget_counts_backing_once_and_each_graph_separately(self):
        target=PersistentQwenTarget(model(),(0,2),slots=2,context_capacity=24,max_query_tokens=6,
            test_kernel=test_only_persistent_sdpa,graph_backend=CPUReplayEmulator(),
            max_graph_buckets=2,graph_byte_budget=1024**2)
        small,large=families();shape=target.pool.family_arena_shape((small,large));size=target.pool.workspace_bytes(shape)
        target.register_families((small,large));self.assertEqual(target._workspace_bytes,size)
        self.assertLess(size,sum(target.pool.workspace_bytes(f) for f in (small,large)))
        for f in (small,large):target.register_graph_bucket(f,reserve_bytes=1234)
        self.assertEqual(target._graph_reserved_bytes,sum(1234+FullTargetExecution.output_buffer_bytes(target,f) for f in (small,large)))
        with self.assertRaises(ValueError):target.register_families((QueryFamily(1,1,1,8,8,'extra'),))
        target.close()
        too_small=PersistentQwenTarget(model(),(),slots=2,context_capacity=24,max_query_tokens=6,
            test_kernel=test_only_persistent_sdpa,workspace_byte_budget=size-1)
        with self.assertRaisesRegex(ValueError,'budget'):too_small.register_families((small,large))
        self.assertFalse(too_small.pool._workspaces);self.assertIsNone(too_small.pool._family_arena);too_small.close()

    def test_family_writer_lifecycle_cancel_and_view_drift(self):
        backend=ManualBackend();eager,graph,small,large=self.fixture(backend)
        t=graph.prepare_graph_verify(self.chunks((1,2)),bucket=small)
        with self.assertRaises(ValueError):graph._graphs[large].submit(t.receipt)
        done=graph.submit_graph(t)
        with self.assertRaises(RuntimeError):graph.prepare_graph_verify(self.chunks((3,3)),bucket=large)
        with self.assertRaisesRegex(RuntimeError,'unfinished'):graph.cancel_graph(t)
        self.assertFalse(graph.pool.failed);backend.ready=True;graph.cancel_graph(t)
        self.assertIsNone(graph._graph_completion)
        with self.assertRaises(ValueError):graph.finish_graph(done)
        f=graph.verify(self.chunks((3,3)),bucket=large);graph.abort(f);graph.release_features(f)
        # Identical base pointer is not an equivalent view layout.
        ws=graph.pool._workspaces[small];ws.keys=ws.keys.transpose(1,2)
        with self.assertRaisesRegex(RuntimeError,'addresses changed'):
            graph.prepare_graph_verify(self.chunks((1,2)),bucket=small)
        self.assertIsNone(graph.pool._pending);eager.close();graph.close()

    def test_real_session_q_p_rng_and_caches_match_exact_control(self):
        from dspark_qwen.config import DraftConfig
        from dspark_qwen.model import DSparkDraft
        from dspark_qwen.packed_draft import PackedDraft,test_only_noncausal_varlen
        from dspark_qwen.packed_sampling import PackedSpeculativeSession,RequestSpec
        prefill=Bucket((3,4,2),(0,0,0),'session admission')
        families_=(QueryFamily(2,2,1,80,40,'zero allocation'),QueryFamily(2,3,2,80,40,'partial'),
                   QueryFamily(2,6,4,80,40,'allocated varying Q'))
        exacts=tuple(Bucket(q,(36,36),'exact session control') for q in ((1,1),(1,2),(2,4),(3,3),(4,2)))
        def make(family):
            m=model();draft=DSparkDraft(m,DraftConfig(layer_ids=(0,2),num_layers=2,block_size=3,markov_rank=8,mask_token_id=127)).eval()
            opts=dict(graph_backend=CPUReplayEmulator(),max_graph_buckets=1,graph_byte_budget=1024**2) if family else {}
            target=PersistentQwenTarget(m,(0,2),slots=3,context_capacity=48,max_query_tokens=12,
                test_kernel=test_only_persistent_sdpa,**opts)
            provider=FiniteTargetBuckets((prefill,),families_ if family else exacts)
            strategy=PersistentTargetStrategy(target,provider)
            if family:target.register_graph_bucket(families_[2],reserve_bytes=4096)
            session=PackedSpeculativeSession(target,PackedDraft(draft,test_kernel=test_only_noncausal_varlen),target_strategy=strategy)
            session.admit({r:RequestSpec(ids,30,RecordedRandom(100+i)) for i,(r,ids) in enumerate(
                dict(a=torch.tensor([[1,2,3]]),b=torch.tensor([[4,5,6,7]]),idle=torch.tensor([[30,31]])).items())})
            if family:target.capture_graph(self.chunks((2,4)),bucket=families_[2])
            return session
        exact,family=make(False),make(True)
        for allocations in ({'a':1,'b':3},{'b':2,'a':2},{'a':3,'b':1},{'a':0,'b':1},{'a':0,'b':0}):
            results=[s.step(allocations) for s in (exact,family)]
            self.assertEqual(exact.outputs(),family.outputs());self.assertEqual(results[0]['requests'],results[1]['requests'])
            work=results[1]['work']['target'];actual_b=sum(1+n for n in allocations.values())
            self.assertEqual(work['physical_query_tokens'],actual_b);self.assertEqual(work['logical_query_tokens'],actual_b)
            self.assertEqual(work['query_lengths'],[n+1 for n in allocations.values()])
            self.assertEqual(work['execution_kind'],'cpu_replay_emulator_not_gpu_graph' if actual_b==6 else 'explicit_eager')
            for r in allocations:
                self.assertTrue(torch.equal(results[0]['proposals'][r].tokens,results[1]['proposals'][r].tokens))
                self.assertTrue(torch.equal(results[0]['proposals'][r].draft_probs,results[1]['proposals'][r].draft_probs))
                self.assertTrue(torch.equal(results[0]['target_probs'][r],results[1]['target_probs'][r]))
            for r in exact.requests:
                self.assertEqual(exact.requests[r]['rng'].values,family.requests[r]['rng'].values)
                for adapter in ('target','draft'):
                    for x,y in zip(getattr(exact,adapter).request_kv(r),getattr(family,adapter).request_kv(r)):
                        for a,b in zip(x,y):torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertIsNone(family.target._feature_lease)
        # No family for this B: preflight refuses before consuming proposal RNG.
        draws=list(family.requests['a']['rng'].values)
        with self.assertRaisesRegex(ValueError,'No finite'):family.step({'a':3})
        self.assertEqual(draws,family.requests['a']['rng'].values)
        exact.target.close();family.target.close()


if __name__=='__main__':unittest.main()
