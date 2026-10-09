"""Independent CPU measurement-contract fixtures; no vLLM or GPU import."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from dspark_qwen.performance_workloads import load_workloads, schedule, summarize, verify_complete_samples

path = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('benchmark_vllm_offline',path/'scripts/benchmark_vllm_offline.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


class Params:
    def __init__(self, **kwargs): self.values = kwargs


def output(identity, prompt, n, finished):
    return NS(request_id=identity,prompt_token_ids=prompt,finished=finished,
        outputs=[NS(token_ids=[7]*n,finish_reason='length' if finished else None)])


class FakeLLM:
    def __init__(self):
        self.llm_engine=NS(has_unfinished_requests=lambda:False)
        self.calls=[]
    def generate(self,prompts,params,*,use_tqdm):
        self.calls.append((prompts,params,use_tqdm))
        return [output(str(i),p['prompt_token_ids'],128,True) for i,p in enumerate(prompts)]


class FakeDiagnosticEngine:
    def __init__(self):self.inputs={};self.pending=set();self.index=0
    def has_unfinished_requests(self):return bool(self.pending)
    def add_request(self,identity,prompt,parameter):
        self.inputs[identity]=(prompt['prompt_token_ids'],parameter);self.pending.add(identity)
    def step(self):
        a,b=list(self.inputs)
        self.index+=1
        if self.index==1:return [output(a,self.inputs[a][0],64,False),output(b,self.inputs[b][0],64,False)]
        done=a if self.index==2 else b
        self.pending.remove(done)
        return [output(done,self.inputs[done][0],64,True)]


class BenchmarkTests(unittest.TestCase):
    def setUp(self):self.manifest,self.digest=load_workloads(path/'configs/performance-workloads.example.json')

    def test_complete_shared_domain_and_one_engine_schedule(self):
        rows=list(schedule(self.manifest))
        self.assertEqual(len(rows),54)
        self.assertEqual(sum(p=='warmup' for p,_,_ in rows),12)
        self.assertEqual(sum(p=='primary' for p,_,_ in rows),30)
        self.assertEqual(sum(p=='diagnostic' for p,_,_ in rows),12)
        for case in self.manifest['cases']:
            self.assertEqual(sum(p=='primary' and c['case_id']==case['case_id'] for p,_,c in rows),5)
        self.assertEqual({2+sum(v) for v in self.manifest['local_native_profile']['prefix_lengths']},set(range(2,17)))
        self.assertEqual(len(set(map(tuple,self.manifest['local_native_profile']['prefix_lengths']))),64)
        # Supplied prompt lengths include every token: no template expansion.
        for case in self.manifest['cases']:
            self.assertTrue(all(len(r['prompt_token_ids'])==case['prompt_length'] for r in case['requests']))

    def test_primary_measures_batch_wall_and_does_not_invent_request_latency(self):
        case=self.manifest['cases'][2];llm=FakeLLM();clock=iter([10.,12.5])
        result=worker.primary_batch(llm,case,self.manifest,Params,clock=lambda:next(clock))
        self.assertEqual(result['batch_wall_seconds'],2.5)
        self.assertEqual(result['output_tokens'],256)
        self.assertTrue(all(r['completion_latency_seconds'] is None and r['ttft_seconds'] is None for r in result['requests']))
        for parameter in llm.calls[0][1]:
            self.assertTrue(parameter.values['ignore_eos'])
            self.assertEqual(parameter.values['max_tokens'],128)
            self.assertFalse(parameter.values['detokenize'])
        self.assertFalse(llm.calls[0][2])
        llm.generate=lambda *a,**k:[]
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            worker.primary_batch(llm,case,self.manifest,Params,clock=iter([0.,1.]).__next__)

    def test_diagnostic_times_each_request_from_own_submission_and_observed_events(self):
        case=self.manifest['cases'][2];engine=FakeDiagnosticEngine();llm=NS(llm_engine=engine)
        clock=iter([10.,10.1,10.2,11.,12.,13.,13.1])
        result=worker.diagnostic_batch(llm,case,self.manifest,Params,'delta',tag='test',clock=clock.__next__)
        a,b=result['requests']
        self.assertAlmostEqual(a['ttft_seconds'],.9)
        self.assertAlmostEqual(a['completion_latency_seconds'],1.9)
        self.assertAlmostEqual(b['ttft_seconds'],.8)
        self.assertAlmostEqual(b['completion_latency_seconds'],2.8)
        self.assertAlmostEqual(result['batch_wall_seconds'],3.1)
        self.assertEqual(result['output_tokens'],256)
        self.assertIsNone(result['pure_prefill_seconds'])
        self.assertTrue(all(parameter.values['output_kind']=='delta' for _,parameter in engine.inputs.values()))

    def test_pooled_rate_excludes_warmup_and_partial_cells_are_explicit(self):
        case=self.manifest['cases'][0]
        rows=[dict(phase='primary',repeat=i,case_id=case['case_id'],batch_wall_seconds=t,output_tokens=128) for i,t in enumerate((2.,4.))]
        rows.append(dict(phase='warmup',repeat=0,case_id=case['case_id'],batch_wall_seconds=100.,output_tokens=128))
        summary=summarize(rows,self.manifest)
        self.assertAlmostEqual(summary[0]['pooled_output_tokens_per_second'],256/6)
        self.assertEqual(summary[0]['batch_latency_median_seconds'],3.)
        self.assertFalse(summary[0]['complete'])
        self.assertIsNone(summary[1]['pooled_output_tokens_per_second'])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            summarize(rows+[rows[0]],self.manifest)
        with self.assertRaisesRegex(ValueError,'output count'):
            summarize([dict(rows[0],output_tokens=127)],self.manifest)

    def test_missing_diagnostic_or_reordered_sample_cannot_claim_complete(self):
        rows=[dict(phase=phase,repeat=repeat,case_id=case['case_id'],batch_wall_seconds=1.,
                   output_tokens=case['request_count']*128) for phase,repeat,case in schedule(self.manifest)]
        self.assertTrue(all(a['complete'] for a in verify_complete_samples(rows,self.manifest)))
        self.assertTrue(all(a['complete'] for a in summarize(rows[:-1],self.manifest)))
        with self.assertRaisesRegex(ValueError,'diagnostic sample domain'):
            verify_complete_samples(rows[:-1],self.manifest)
        with self.assertRaises(ValueError):
            verify_complete_samples(rows[1:]+rows[:1],self.manifest)

    def test_diagnostic_missing_prompt_echo_fails_closed_for_pinned_api(self):
        case=self.manifest['cases'][2];engine=FakeDiagnosticEngine();original=engine.step
        def missing_echo():
            result=original();result[0].prompt_token_ids=None;return result
        engine.step=missing_echo
        with self.assertRaisesRegex(ValueError,'identity'):
            worker.diagnostic_batch(NS(llm_engine=engine),case,self.manifest,Params,'delta',tag='test')

    def test_workload_mutation_rejects_unmeasured_or_incomplete_domains(self):
        for mutate in (lambda m:m['cases'].pop(),
                       lambda m:m['local_native_profile']['prefix_lengths'].pop(),
                       lambda m:m['cases'][0]['requests'][0]['prompt_token_ids'].pop(),
                       lambda m:m['sampling'].update(ignore_eos=False)):
            data=copy.deepcopy(self.manifest);mutate(data)
            with tempfile.TemporaryDirectory() as tmp:
                file=Path(tmp)/'workload.json';file.write_text(json.dumps(data))
                with self.assertRaises(ValueError):load_workloads(file)


if __name__=='__main__':unittest.main()
