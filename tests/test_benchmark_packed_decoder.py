import builtins
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
bench=load('native_end_to_end',ROOT/'scripts/benchmark_packed_decoder.py')
fixture=load('native_end_to_end_fixture',ROOT/'tests/test_probe_packed_decoder.py')
MANIFEST,_=bench.SHARED.load_workloads(ROOT/'configs/performance-workloads.example.json')

class NativeBenchmarkStdlibTests(unittest.TestCase):
    def test_binding_no_torch_and_complete_separate_54_sample_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,g,c,h=fixture.PackedGateTests().binding_fixture(Path(tmp));w=copy.deepcopy(MANIFEST)
            w['model']['config_sha256']=bench.GUARD.sha(m/'config.json');path=Path(tmp)/'workload.json';path.write_text(json.dumps(w))
            original=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name.split('.')[0] in ('torch','transformers','dspark_qwen'):raise AssertionError('heavyweight import')
                return original(name,*args,**kwargs)
            with patch.dict(bench.PROFILE.GATE.PROTOCOL,h),patch('builtins.__import__',side_effect=guarded),patch('sys.stdout',new_callable=io.StringIO) as stdout:
                self.assertEqual(bench.main(['--dry-run','--model',str(m),'--generation-manifest',str(g),'--checkpoint',str(c),'--workloads',str(path)]),0)
            binding=json.loads(stdout.getvalue())['binding']
            self.assertEqual(len(binding['expected_samples']),54);self.assertNotIn('local_requests',binding)
            self.assertEqual(binding['protocol']['timeout_seconds'],1800)
            self.assertFalse(binding['protocol']['capacity_scheduler_integrated'])
            self.assertIn('scripts/benchmark_packed_decoder.py',binding['source_sha256'])

    def fake_sample(self,*args,**kwargs):
        case=args[2];manifest=args[3]
        return dict(batch_wall_seconds=1.,output_tokens=case['request_count']*manifest['output_tokens'])

    def test_run_preserves_all54_identity_order_and_exact_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(bench,'batch',side_effect=self.fake_sample) as batch:
            out=Path(tmp)/'run';result=bench.run(None,None,MANIFEST,out,setup={'model_loads':1})
            self.assertEqual(batch.call_count,54);self.assertEqual(result['status'],'completed')
            samples=[json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
            self.assertEqual(result['aggregates'],bench.SHARED.verify_complete_samples(samples,MANIFEST))
            self.assertFalse(result['speculative_verification_sps']);self.assertTrue((out/'runtime.json').exists())
            self.assertEqual(sum(x['phase']=='primary' for x in samples),30)

    def test_missing_final_diagnostic_cannot_claim_complete_even_with_all_primary(self):
        calls=[]
        def failure(*args,**kwargs):
            calls.append(None)
            if len(calls)==54:raise RuntimeError('last diagnostic failed')
            return self.fake_sample(*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp,patch.object(bench,'batch',side_effect=failure):
            out=Path(tmp)/'run'
            with self.assertRaisesRegex(RuntimeError,'last diagnostic'):bench.run(None,None,MANIFEST,out)
            result=json.loads((out/'result.json').read_text());self.assertEqual(result['status'],'failed')
            self.assertEqual(result['sample_count'],53);self.assertTrue(all(x['complete'] for x in result['aggregates']))
            self.assertEqual(len(result['expected_samples']),54)

    def test_deadline_keeps_full_expected_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            with self.assertRaises(TimeoutError):bench.run(None,None,MANIFEST,out,deadline=0.)
            r=json.loads((out/'result.json').read_text());self.assertEqual(r['status'],'partial_deadline')
            self.assertEqual(len(r['expected_samples']),54);self.assertEqual(r['sample_count'],0)

class NativeBenchmarkTensorTests(unittest.TestCase):
    def factory(self):
        import torch
        from dspark_qwen.packed_draft import PackedDraft,test_only_noncausal_varlen
        from dspark_qwen.varlen_target import VarlenPackedTarget,test_only_dense_varlen
        session,model=fixture.PackedGateTests().fixture();draft=session.draft.draft;constructed=[]
        def target_builder(m,ids):
            value=VarlenPackedTarget(m,ids,test_kernel=test_only_dense_varlen);constructed.append(value);return value
        make=bench.make_session_factory(model,draft,target_builder=target_builder,
            draft_builder=lambda d:PackedDraft(d,test_kernel=test_only_noncausal_varlen))
        make.constructed=constructed
        return make,torch.device('cpu')

    def case(self):
        return dict(case_id='tiny',request_count=2,prompt_length=8,requests=[dict(request=r,seed=seed,prompt_token_ids=list(range(base,base+8))) for r,base,seed in [('A',100,3),('B',200,4)]])

    def test_real_tiny_admission_counts_once_fullshadow_and_fresh_repeats(self):
        make,device=self.factory();m=copy.deepcopy(MANIFEST);m['output_tokens']=4
        target_identity=id(make.target);model_identity=id(make.target.model)
        a=bench.batch(make,device,self.case(),m)
        self.assertEqual(make.target.lengths,{'A':11,'B':11})
        b=bench.batch(make,device,self.case(),m)
        self.assertEqual(a['output_tokens'],8);self.assertEqual(a['admission_output_tokens'],2)
        self.assertEqual(a['committed_round_output_tokens'],6);self.assertEqual(a['retained_resident_requests'],2)
        self.assertEqual([r['output_sha256'] for r in a['requests']],[r['output_sha256'] for r in b['requests']])
        self.assertEqual(len(make.constructed),1);self.assertEqual(id(make.target),target_identity);self.assertEqual(id(make.target.model),model_identity)
        self.assertTrue(all(r['ttft_seconds'] is None and r['completion_latency_seconds'] is None for r in a['requests']))
        for row in a['rounds']:
            self.assertEqual(row['actual_logical_b'],row['actual_physical_b'])
            self.assertEqual(row['work']['proposal_batches'][0]['backbone']['query_rows'],7*len(row['active_requests']))
            self.assertEqual(row['resident_requests'],2)
        self.assertTrue(all(n==11 for n in a['final_context_lengths'].values()))

    def test_real_tiny_diagnostic_observer_separate_and_zero_remaining_extra(self):
        make,device=self.factory();m=copy.deepcopy(MANIFEST);m['output_tokens']=2
        row=bench.batch(make,device,self.case(),m,diagnostic=True)
        self.assertEqual(row['output_tokens'],4);self.assertEqual(len(row['rounds']),1)
        self.assertEqual(row['rounds'][0]['allocation'],{'A':0,'B':0})
        self.assertEqual(row['rounds'][0]['work']['proposal_batches'][0]['backbone']['query_rows'],14)
        for r in row['requests']:
            self.assertGreater(r['ttft_seconds'],0)
            self.assertGreaterEqual(r['completion_latency_seconds'],r['ttft_seconds'])
        self.assertIsNone(row['pure_prefill_seconds'])

    def test_round_deadline_preserves_prior_completed_batch(self):
        make,device=self.factory();m=copy.deepcopy(MANIFEST);m['output_tokens']=4
        for c in m['cases']:
            for request in c['requests']:request['prompt_token_ids']=[x%1024 for x in request['prompt_token_ids']]
        original=bench.batch;calls=[]
        def expire_second(*args,**kwargs):
            calls.append(None)
            if len(calls)==2:kwargs['deadline']=0.
            return original(*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp,patch.object(bench,'batch',side_effect=expire_second):
            out=Path(tmp)/'run'
            with self.assertRaisesRegex(TimeoutError,'incomplete batch'):bench.run(make,device,m,out)
            r=json.loads((out/'result.json').read_text());self.assertEqual(r['status'],'partial_deadline')
            self.assertEqual(r['sample_count'],1);self.assertEqual(len(r['expected_samples']),54)
            self.assertEqual(len((out/'samples.jsonl').read_text().splitlines()),1)

    def test_real_unequal_completion_retains_finished_request_KV_cost(self):
        make,device=self.factory();m=copy.deepcopy(MANIFEST);m['output_tokens']=8
        row=bench.batch(make,device,self.case(),m)
        self.assertEqual(row['output_tokens'],16)
        self.assertEqual([x['active_requests'] for x in row['rounds']],[['A','B'],['A']])
        self.assertEqual(row['rounds'][0]['requests']['B']['stop_reason'],'budget')
        last=row['rounds'][-1];work=last['work']
        self.assertEqual(last['resident_requests'],2);self.assertEqual(work['target']['resident_requests'],2)
        self.assertEqual(work['target']['queried_requests'],1)
        self.assertEqual(work['target']['inactive_kv_tokens_excluded'],15)
        self.assertEqual(work['proposal_batches'][0]['backbone']['physical_context_rows'],25)
        self.assertEqual(work['proposal_batches'][0]['backbone']['query_rows'],7)
        self.assertEqual(row['final_context_lengths'],{'A':15,'B':15})

if __name__=='__main__':unittest.main()
