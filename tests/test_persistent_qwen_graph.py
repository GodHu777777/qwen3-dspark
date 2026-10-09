"""Real tiny HF model plus explicit CPU replay-emulator ownership tests."""
import copy
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from dspark_qwen.persistent_qwen_target import PersistentQwenTarget,test_only_persistent_sdpa
from dspark_qwen.persistent_qwen_graph import CPUReplayEmulator,graph_pool_accounting
from dspark_qwen.persistent_target_kv import PersistentTargetKV,Bucket


def model():
    torch.manual_seed(421);torch.set_num_threads(2)
    cfg=Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=48,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,attention_dropout=0.)
    cfg._attn_implementation='sdpa'
    return Qwen3ForCausalLM(cfg).eval().requires_grad_(False)


class ManualBackend(CPUReplayEmulator):
    def __init__(self):super().__init__();self.ready=False;self.fail=False
    def complete(self,event,receipt):return self.ready and super().complete(event,receipt)
    def wait(self,event):
        if self.fail:raise RuntimeError('injected event failure')
        self.ready=True


class ExternalWriteTests(unittest.TestCase):
    def test_shared_backend_same_generation_cannot_publish_another_pool(self):
        backend=CPUReplayEmulator()
        def prepare():
            p=PersistentTargetKV(layers=1,slots=1,context_capacity=4,max_query_tokens=1,heads=1,dim=2)
            h=p.add_request('r');bucket=Bucket((1,),(2,),'shared backend counterexample')
            p.register_bucket(bucket);writer=p.register_external_writer(bucket,(0,),backend.complete)
            tx=p.begin([h],bucket);receipt=p.prepare_external_write(tx,writer)
            program,_=backend.capture(lambda:None,p.device,1,writer)
            return p,h,receipt,program
        left,lh,lr,lp=prepare();right,rh,rr,rp=prepare()
        self.assertEqual(lr.generation,rr.generation)
        with self.assertRaisesRegex(ValueError,'different captured writer'):
            backend.submit(lp,rr,right.device)
        event=backend.submit(lp,lr,left.device)
        right.submit_external_write(rr,event)
        with self.assertRaisesRegex(RuntimeError,'not been observed'):right.complete_external_write(rr)
        with self.assertRaises(ValueError):right.commit(rr.transaction,{rh:1})
        self.assertEqual(right.length(rh),0)
        self.assertFalse(right._staged);right.invalidate();left.abort(lr.transaction)

    def test_private_pool_budget_counts_inactive_blocks_and_rejects_bad_schema(self):
        segment=dict(address=4096,total_size=1024,allocated_size=0,active_size=0,
            device=0,segment_pool_id=(0,7),blocks=[dict(size=1024,state='inactive')])
        with self.assertRaisesRegex(RuntimeError,'measured=1024, reservation=512') as failure:
            graph_pool_accounting((0,7),[segment],0,10000,11024,512)
        self.assertEqual(failure.exception.memory_accounting['private_pool_reserved_bytes'],1024)
        self.assertEqual(failure.exception.pool_snapshot,[segment])
        result=graph_pool_accounting((0,7),[segment],0,10000,11024,1024)
        self.assertEqual(result['private_pool_reserved_bytes'],1024)
        self.assertTrue(result['includes_inactive_blocks'])
        for broken in ([segment,segment],[{k:v for k,v in segment.items() if k!='device'}],
                       [dict(segment,device=1)],[dict(segment,segment_pool_id=(0,8))],
                       [dict(segment,blocks=[dict(size=512)])],[]):
            with self.assertRaises(RuntimeError):graph_pool_accounting((0,7),broken,0,10000,12048,2048)

    def test_exact_program_transaction_generation_event_and_complete_layers(self):
        p=PersistentTargetKV(layers=2,slots=1,context_capacity=8,max_query_tokens=2,heads=1,dim=2)
        h=p.add_request('a');b=Bucket((2,),(4,),'receipt test');p.register_bucket(b)
        backend=ManualBackend();writer=p.register_external_writer(b,(0,1),backend.complete)
        with self.assertRaises(ValueError):p.register_external_writer(b,(0,),backend.complete)
        tx=p.begin([h],b)
        with self.assertRaises(ValueError):p.prepare_external_write(tx,copy.copy(writer))
        receipt=p.prepare_external_write(tx,writer)
        with self.assertRaises(ValueError):p.complete_external_write(receipt)
        with self.assertRaises(ValueError):p.stage_layer(tx,0,torch.ones(2,1,2),torch.ones(2,1,2))
        program,_=backend.capture(lambda:None,p.device,1,writer)
        event=backend.submit(program,receipt,p.device)
        p.submit_external_write(receipt,event)
        with self.assertRaises(ValueError):p.submit_external_write(receipt,event)
        with self.assertRaises(RuntimeError):p.complete_external_write(receipt)
        with self.assertRaises(RuntimeError):p.abort(tx)
        with self.assertRaises(ValueError):p.commit(tx,{h:1})
        self.assertFalse(backend.complete(copy.copy(event),receipt))
        backend.ready=True
        with self.assertRaises(ValueError):p.complete_external_write(copy.copy(receipt))
        p.scratch_keys.fill_(2);p.scratch_values.fill_(3)
        p.complete_external_write(receipt);p.commit(tx,{h:1})
        self.assertTrue(torch.equal(p.request_kv(h)[0],torch.full((2,1,1,2),2.)))
        tx2=p.begin([h],b);r2=p.prepare_external_write(tx2,writer)
        self.assertGreater(r2.generation,receipt.generation)
        with self.assertRaises(ValueError):p.complete_external_write(receipt)
        p.submit_external_write(r2,event)
        with self.assertRaises(RuntimeError):p.complete_external_write(r2)
        p.invalidate()
        with self.assertRaises(RuntimeError):p.commit(tx2,{h:0})

    def test_stale_scratch_address_and_foreign_pool_receipts_fail_closed(self):
        def make():
            p=PersistentTargetKV(layers=1,slots=1,context_capacity=4,max_query_tokens=1,heads=1,dim=2)
            h=p.add_request('a');b=Bucket((1,),(2,),'fixture');p.register_bucket(b)
            backend=CPUReplayEmulator();writer=p.register_external_writer(b,(0,),backend.complete)
            tx=p.begin([h],b);r=p.prepare_external_write(tx,writer)
            return p,backend,tx,r
        p,backend,tx,r=make();other,_,_,foreign=make()
        with self.assertRaises(ValueError):p.submit_external_write(foreign,None)
        p.scratch_keys=p.scratch_keys.clone()
        with self.assertRaisesRegex(RuntimeError,'allocation changed'):p.submit_external_write(r,None)
        self.assertTrue(p.failed);other.abort(foreign.transaction)


class FullTargetGraphTests(unittest.TestCase):
    def fixture(self,backend=None):
        original=model();eager=PersistentQwenTarget(copy.deepcopy(original),(0,2,3),slots=3,
            context_capacity=24,max_query_tokens=12,test_kernel=test_only_persistent_sdpa)
        graph=PersistentQwenTarget(original,(0,2,3),slots=3,context_capacity=24,max_query_tokens=12,
            test_kernel=test_only_persistent_sdpa,graph_backend=backend or CPUReplayEmulator(),
            max_graph_buckets=1,graph_byte_budget=1024**2,workspace_byte_budget=1024**2)
        prompt=Bucket((3,5,4),(0,0,0),'full-model prefill');verify=Bucket((2,3),(16,16),'full-model graph family')
        for target in (eager,graph):
            target.register_bucket(prompt)
            for name in ('a','b','idle'):target.add_request(name)
            f=target.prefill(dict(a=torch.tensor([[1,2,3]]),b=torch.tensor([[4,5,6,7,8]]),idle=torch.tensor([[30,31,32,33]])),bucket=prompt)
            target.release_features(f)
        eager.register_bucket(verify);graph.register_graph_bucket(verify,reserve_bytes=65536)
        return eager,graph,verify

    def chunks(self,offset=0):return dict(a=torch.tensor([[9+offset,10+offset]]),b=torch.tensor([[11+offset,12+offset,13+offset]]))

    def compare(self,eager,graph,a,b):
        torch.testing.assert_close(a.last,b.last,rtol=0,atol=0)
        torch.testing.assert_close(a.context,b.context,rtol=0,atol=0)
        for r in a.spans:torch.testing.assert_close(eager.predict(a)[r],graph.predict(b)[r],rtol=0,atol=0)
        self.assertFalse(torch.allclose(b.context[:,:,-32:],b.last))

    def test_real_all_layer_model_growing_replays_commit_and_feature_lease(self):
        eager,graph,bucket=self.fixture();chunks=self.chunks()
        before=(graph.pool.keys.clone(),graph.pool.values.clone())
        graph.capture_graph(chunks,bucket=bucket)
        for old,new in zip(before,(graph.pool.keys,graph.pool.values)):self.assertTrue(torch.equal(old,new))
        pointers=graph._graphs[bucket].pointers();positions=[];old_features=None
        for chunks,counts in [(chunks,{'a':1,'b':2}),(self.chunks(7),{'a':2,'b':3})]:
            f=eager.verify_eager(chunks,bucket=bucket)
            ticket=graph.prepare_graph_verify(chunks,bucket=bucket)
            with self.assertRaises(ValueError):graph.submit_graph(copy.copy(ticket))
            completion=graph.submit_graph(ticket)
            with self.assertRaises(ValueError):graph.finish_graph(copy.copy(completion))
            self.assertEqual(graph.pool._staged,set())
            g=graph.finish_graph(completion);self.compare(eager,graph,f,g)
            if old_features is not None:
                with self.assertRaises(ValueError):graph.predict(old_features)
            positions.append(graph.pool._workspaces[bucket].positions.tolist())
            self.assertEqual(graph._graphs[bucket].pointers(),pointers)
            with self.assertRaises(RuntimeError):graph.release_features(g)
            eager.commit(f,counts);graph.commit(g,counts)
            with self.assertRaises(RuntimeError):graph.verify(chunks,bucket=bucket)
            with self.assertRaises(RuntimeError):graph.reset()
            # A real draft consumer still needs raw selected features after commit.
            self.assertGreater(float(g.context.square().sum()),0)
            for r in ('a','b','idle'):
                for x,y in zip(eager.request_kv(r),graph.request_kv(r)):
                    for xx,yy in zip(x,y):torch.testing.assert_close(xx,yy,rtol=0,atol=0)
            eager.release_features(f);graph.release_features(g);old_features=g
            with self.assertRaises(ValueError):graph.release_features(g)
        self.assertEqual(positions,[[3,4,5,6,7],[4,5,7,8,9]])
        self.assertEqual(graph.lengths,{'a':6,'b':10,'idle':4})
        eager.close();graph.close()

    def test_unfinished_completion_cancel_and_failed_event_invalidate(self):
        backend=ManualBackend();eager,graph,bucket=self.fixture(backend)
        graph.capture_graph(self.chunks(),bucket=bucket)
        t=graph.prepare_graph_verify(self.chunks(),bucket=bucket);graph.cancel_graph(t)
        with self.assertRaises(ValueError):graph.submit_graph(t)
        t=graph.prepare_graph_verify(self.chunks(),bucket=bucket);done=graph.submit_graph(t)
        with self.assertRaisesRegex(RuntimeError,'not been observed'):graph.finish_graph(done,wait=False)
        with self.assertRaises(RuntimeError):graph.cancel_graph(t)
        self.assertEqual(graph.pool._staged,set());self.assertFalse(graph.pool.failed)
        backend.ready=True;g=graph.finish_graph(done,wait=False);graph.abort(g);graph.release_features(g)
        t=graph.prepare_graph_verify(self.chunks(),bucket=bucket);done=graph.submit_graph(t)
        backend.fail=True
        with self.assertRaisesRegex(RuntimeError,'event failure'):graph.finish_graph(done)
        self.assertTrue(graph.pool.failed)
        with self.assertRaises(RuntimeError):graph.reset()
        eager.close()

    def test_capture_failure_no_fallback_and_exact_registry_budgets(self):
        eager,graph,bucket=self.fixture()
        with self.assertRaisesRegex(ValueError,'captured'):graph.verify(self.chunks(),bucket=bucket)
        with self.assertRaises(ValueError):graph.register_graph_bucket(Bucket((1,),(16,),'extra'),reserve_bytes=1)
        self.assertEqual(len(graph._graphs),1)
        tiny=PersistentQwenTarget(model(),(0,),slots=1,context_capacity=8,max_query_tokens=2,
            test_kernel=test_only_persistent_sdpa,graph_backend=CPUReplayEmulator(),
            max_graph_buckets=1,graph_byte_budget=100,workspace_byte_budget=100)
        with self.assertRaisesRegex(ValueError,'budget'):tiny.register_graph_bucket(Bucket((2,),(4,),'over'),reserve_bytes=1)
        self.assertFalse(tiny.pool._workspaces);tiny.close()
        with patch.object(graph.model.model.layers[1],'forward',side_effect=RuntimeError('capture failed')):
            with self.assertRaisesRegex(RuntimeError,'capture failed'):graph.capture_graph(self.chunks(),bucket=bucket)
        self.assertTrue(graph.pool.failed)
        with self.assertRaises(RuntimeError):graph.verify_eager(self.chunks(),bucket=bucket)
        eager.close()

    def test_owner_stream_release_and_predict_boundary(self):
        eager,graph,bucket=self.fixture();graph.capture_graph(self.chunks(),bucket=bucket)
        g=graph.verify(self.chunks(),bucket=bucket);graph.commit(g,{'a':1,'b':1})
        with patch.object(graph,'_stream_identity',return_value='foreign-stream'):
            with self.assertRaisesRegex(RuntimeError,'owner stream'):graph.release_features(g)
            with self.assertRaises(RuntimeError):graph.predict(g)
            with self.assertRaises(RuntimeError):graph.prepare_graph_verify(self.chunks(),bucket=bucket)
        self.assertIs(graph._feature_lease,g)
        graph.release_features(g);graph.close();eager.close()

    def test_abort_replay_keeps_all_resident_kv_and_reincarnates_features(self):
        eager,graph,bucket=self.fixture();graph.capture_graph(self.chunks(),bucket=bucket)
        before=(graph.pool.keys.clone(),graph.pool.values.clone())
        g=graph.verify(self.chunks(),bucket=bucket);graph.abort(g)
        with self.assertRaises(RuntimeError):graph.verify(self.chunks(),bucket=bucket)
        graph.release_features(g)
        for old,new in zip(before,(graph.pool.keys,graph.pool.values)):self.assertTrue(torch.equal(old,new))
        new=graph.verify(self.chunks(9),bucket=bucket)
        self.assertIsNot(new,g)
        with self.assertRaises(ValueError):graph.commit(g,{'a':0,'b':0})
        graph.abort(new);graph.release_features(new);graph.close();eager.close()

    def test_changed_model_and_output_addresses_are_not_replayed(self):
        eager,graph,bucket=self.fixture();graph.capture_graph(self.chunks(),bucket=bucket)
        with torch.no_grad():next(graph.model.parameters()).add_(.01)
        with self.assertRaisesRegex(RuntimeError,'model signature'):
            graph.prepare_graph_verify(self.chunks(),bucket=bucket)
        self.assertIsNone(graph.pool._pending)
        eager.close();graph.close()

        eager,graph,bucket=self.fixture();graph.capture_graph(self.chunks(),bucket=bucket)
        ex=graph._graphs[bucket];ex.last=ex.last.clone()
        with self.assertRaisesRegex(RuntimeError,'addresses changed'):
            graph.prepare_graph_verify(self.chunks(),bucket=bucket)
        self.assertIsNone(graph.pool._pending)
        eager.close();graph.close()

    def test_execution_rejects_foreign_writer_before_backend_submission(self):
        backend=CPUReplayEmulator()
        eager,left,bucket=self.fixture(backend);other,right,_=self.fixture(backend)
        for target in (left,right):target.capture_graph(self.chunks(),bucket=bucket)
        lt=left.prepare_graph_verify(self.chunks(),bucket=bucket)
        rt=right.prepare_graph_verify(self.chunks(),bucket=bucket)
        self.assertEqual(lt.receipt.generation,rt.receipt.generation)
        with patch.object(backend,'submit',side_effect=AssertionError('must reject before replay')):
            with self.assertRaisesRegex(ValueError,'this captured writer'):
                left._graphs[bucket].submit(rt.receipt)
        left.cancel_graph(lt);right.cancel_graph(rt)
        for target in (eager,left,other,right):target.close()

    def test_cancel_checker_exception_poisons_but_false_remains_pending(self):
        class QueryFailure(ManualBackend):
            explode=False
            def complete(self,event,receipt):
                if self.explode:raise RuntimeError('injected cancel query failure')
                return super().complete(event,receipt)
        backend=QueryFailure();eager,graph,bucket=self.fixture(backend)
        graph.capture_graph(self.chunks(),bucket=bucket)
        ticket=graph.prepare_graph_verify(self.chunks(),bucket=bucket);graph.submit_graph(ticket)
        with self.assertRaisesRegex(RuntimeError,'unfinished'):graph.cancel_graph(ticket)
        self.assertFalse(graph.pool.failed);self.assertIs(graph._graph_ticket,ticket)
        backend.explode=True
        with self.assertRaisesRegex(RuntimeError,'cancel query failure'):graph.cancel_graph(ticket)
        self.assertTrue(graph.pool.failed);self.assertTrue(graph._graphs[bucket].invalid)
        backend.explode=False;backend.ready=True
        with self.assertRaises(RuntimeError):graph.verify_eager(self.chunks(),bucket=bucket)
        with self.assertRaises(RuntimeError):graph.reset()
        eager.close()

    def test_real_packed_session_draft_consumes_graph_lease_for_two_rounds(self):
        from dspark_qwen.config import DraftConfig
        from dspark_qwen.model import DSparkDraft
        from dspark_qwen.packed_draft import PackedDraft,test_only_noncausal_varlen
        from dspark_qwen.packed_sampling import PackedSpeculativeSession,RequestSpec
        from dspark_qwen.target_strategy import FiniteTargetBuckets,PersistentTargetStrategy
        from dspark_qwen.tensor_sampling import TensorRandom
        prefill=Bucket((3,5),(0,0),'real session prefill')
        verification=Bucket((1,1),(16,16),'real session graph family')
        provider=FiniteTargetBuckets((prefill,),(verification,))
        def make(graph):
            m=model();draft=DSparkDraft(m,DraftConfig(layer_ids=(0,2),num_layers=2,
                block_size=3,markov_rank=8,mask_token_id=127)).eval()
            opts=dict(graph_backend=CPUReplayEmulator(),max_graph_buckets=1,graph_byte_budget=1024**2) if graph else {}
            target=PersistentQwenTarget(m,(0,2),slots=2,context_capacity=24,max_query_tokens=8,
                test_kernel=test_only_persistent_sdpa,**opts)
            strategy=PersistentTargetStrategy(target,provider)
            if graph:target.register_graph_bucket(verification,reserve_bytes=65536)
            session=PackedSpeculativeSession(target,PackedDraft(draft,test_kernel=test_only_noncausal_varlen),target_strategy=strategy)
            session.admit({r:RequestSpec(ids,6,TensorRandom(torch.Generator().manual_seed(91+i)))
                for i,(r,ids) in enumerate(dict(a=torch.tensor([[1,2,3]]),b=torch.tensor([[4,5,6,7,8]])).items())})
            return session
        eager,graph=make(False),make(True);target=graph.target
        target.capture_graph({r:torch.tensor([[s['output'][-1]]]) for r,s in graph.requests.items()},bucket=verification)
        events=[];project=graph.draft.append_committed;release=target.release_features
        def project_with_lease(contexts):
            self.assertIsNotNone(target._feature_lease);self.assertIsNone(target._pending)
            result=project(contexts);events.append('actual_draft_projection');return result
        def release_after_projection(features):
            self.assertEqual(events[-1],'actual_draft_projection');events.append('release')
            return release(features)
        with patch.object(graph.draft,'append_committed',side_effect=project_with_lease),patch.object(target,'release_features',side_effect=release_after_projection):
            for _ in range(2):
                results=[]
                for s in (eager,graph):
                    issued=s.propose(('a','b'),mode='shadow')
                    results.append(s.verify_commit(issued.proposals,{'a':0,'b':0},allocation_policy='external_nonanticipating'))
                self.assertEqual(eager.outputs(),graph.outputs());self.assertIsNone(target._feature_lease)
                for r in ('a','b'):
                    self.assertTrue(torch.equal(results[0]['target_probs'][r],results[1]['target_probs'][r]))
                    for reference,candidate in zip(eager.draft.request_kv(r),graph.draft.request_kv(r)):
                        for a,b in zip(reference,candidate):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(events,['actual_draft_projection','release']*2)
        eager.target.close();graph.target.close()


if __name__=='__main__':unittest.main()
