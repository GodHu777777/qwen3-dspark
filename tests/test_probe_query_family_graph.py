"""Bounded CPU contracts for the additive native variable-Q capability runner."""
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
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
probe=load('variable_query_probe_tests',ROOT/'scripts/probe_query_family_graph.py')
old_fixture=load('variable_query_old_fixture',ROOT/'tests/test_probe_full_target_graph.py')


class BindingTests(unittest.TestCase):
    def test_additive_stdlib_binding_freezes_six_states_and_old_protocol_unchanged(self):
        old=copy.deepcopy(probe.BASE.PROTOCOL)
        with tempfile.TemporaryDirectory() as tmp:
            model,generation,_,_=old_fixture.fixture.PackedGateTests().binding_fixture(Path(tmp))
            workload=json.loads((ROOT/'configs/performance-workloads.example.json').read_text())
            workload['model']['config_sha256']=probe.GUARD.sha(model/'config.json')
            path=Path(tmp)/'workload.json';path.write_text(json.dumps(workload));original=builtins.__import__
            def guarded(name,*args,**kw):
                if name.split('.')[0] in ('torch','transformers','dspark_qwen'):raise AssertionError('Heavyweight binding import')
                return original(name,*args,**kw)
            with patch('builtins.__import__',side_effect=guarded),patch('sys.stdout',new_callable=io.StringIO) as output:
                self.assertEqual(probe.main(['--model',str(model),'--generation-manifest',str(generation),'--workloads',str(path)]),0)
            row=json.loads(output.getvalue());self.assertEqual(row['status'],'dry_run_no_torch_no_gpu')
            self.assertEqual(row['binding']['protocol'],probe.PROTOCOL);self.assertEqual(probe.BASE.PROTOCOL,old)
            self.assertEqual([r['family'] for r in probe.PROTOCOL['cases']],['S3','L5','E2','L5','L5','S3'])
            self.assertEqual(probe.PROTOCOL['cases'][4]['order'],['r1','r0'])
            self.assertEqual(probe.PROTOCOL['comparisons'],old['comparisons'])
            self.assertIn('scripts/probe_full_target_graph.py',row['binding']['source_sha256'])
            self.assertIn('scripts/probe_query_family_graph.py',row['binding']['source_sha256'])


class ContractTests(unittest.TestCase):
    def target(self,backend=None):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from dspark_qwen.persistent_qwen_target import PersistentQwenTarget,test_only_persistent_sdpa
        from dspark_qwen.persistent_qwen_graph import CPUReplayEmulator
        torch.manual_seed(993);torch.set_num_threads(2)
        cfg=Qwen3Config(vocab_size=1200,hidden_size=32,intermediate_size=48,num_hidden_layers=4,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,attention_dropout=0.)
        cfg._attn_implementation='sdpa'
        return PersistentQwenTarget(Qwen3ForCausalLM(cfg).eval().requires_grad_(False),(0,2,3),slots=3,
            context_capacity=148,max_query_tokens=273,test_kernel=test_only_persistent_sdpa,
            graph_backend=backend or CPUReplayEmulator(),max_graph_buckets=2,
            graph_byte_budget=probe.PROTOCOL['graph_byte_budget'],workspace_byte_budget=probe.PROTOCOL['workspace_byte_budget'])

    def run_fixture(self,target,path,*,require_gpu=False,**kw):
        import torch
        manifest=json.loads((ROOT/'configs/performance-workloads.example.json').read_text())
        requests=next(c['requests'] for c in manifest['cases'] if c['case_id']=='r2-c256')
        with torch.no_grad():return probe.exercise(target,requests,path,probe.new_report(),require_gpu=require_gpu,**kw)

    def test_six_states_two_graphs_shared_arena_reversal_and_explicit_eager(self):
        import torch
        target=self.target()
        with tempfile.TemporaryDirectory() as tmp:
            result=self.run_fixture(target,tmp);probe.verify_report(result,require_gpu=False)
            self.assertEqual(len(result['observations']),12);self.assertEqual(len(result['captures']),2)
            self.assertEqual(target.lengths,dict(r0=133,r1=137,idle=17));self.assertIsNone(target._feature_lease)
            rows=result['observations'][6:]
            self.assertEqual([r['work']['physical_query_tokens'] for r in rows],[3,5,2,5,5,3])
            self.assertEqual([r['execution'] for r in rows],['graph_replay','graph_replay','explicit_eager','graph_replay','graph_replay','graph_replay'])
            self.assertEqual(rows[3]['work']['query_lengths'],[2,3]);self.assertEqual(rows[4]['work']['query_lengths'],[4,1])
            for index,case in enumerate(probe.PROTOCOL['cases']):
                actual=torch.load(Path(tmp)/f'mixed-{index}-inputs.pt',weights_only=True)
                self.assertEqual(actual['positions'].tolist(),[p for c,q in zip(case['contexts'],case['query_lengths']) for p in range(c,c+q)])
                self.assertEqual(actual['cu_query'].tolist(),[0,case['query_lengths'][0],sum(case['query_lengths'])])
            self.assertEqual(len(list(Path(tmp).glob('mixed-*-committed.pt'))),6)
            self.assertEqual(len(list(Path(tmp).glob('eager-*-independent-layer-witnesses.pt'))),6)
            for name,qs in (('S3',[1,2]),('L5',[1,4])):
                capture=torch.load(Path(tmp)/f'capture-{name}-inputs.pt',weights_only=True)
                self.assertEqual(capture['cu_query'].tolist(),[0,qs[0],sum(qs)])
            with self.assertRaises(ValueError):probe.verify_report(result)
            for field in ('domain','captures','counter','physical'):
                broken=copy.deepcopy(result)
                if field=='domain':broken['observations'].pop()
                elif field=='captures':broken['captures'].pop()
                elif field=='counter':broken['observations'][6]['python_after']['model_forward']+=1
                else:broken['observations'][6]['work']['physical_query_tokens']+=1
                with self.assertRaises(ValueError):probe.verify_report(broken,require_gpu=False)
        target.close()

    def test_raw_layer_and_value_contamination_fail_before_capture_with_raw_saved(self):
        for failure in ('layer','values'):
            target=self.target();original_features=probe.feature_tensors;original_verify=target.verify_eager
            def wrong_features(*args):
                result=original_features(*args);result['context']=result['context'].roll(32,dims=-1);return result
            def wrong_values(*args,**kw):
                result=original_verify(*args,**kw)
                if tuple(args[0])==('r0','r1'):target.pool.values.add_(1.)
                return result
            with tempfile.TemporaryDirectory() as tmp:
                context=patch.object(probe,'feature_tensors',side_effect=wrong_features) if failure=='layer' else patch.object(target,'verify_eager',side_effect=wrong_values)
                with context,self.assertRaises(AssertionError):self.run_fixture(target,tmp)
                result=json.loads((Path(tmp)/'result.json').read_text());self.assertEqual(result['stages']['capture'],'pending')
                self.assertTrue((Path(tmp)/'eager-0-outputs.pt').exists());self.assertTrue((Path(tmp)/'failed-resident-and-scratch.pt').exists())

    def test_second_capture_budget_failure_preserves_first_capture_and_snapshot(self):
        from dspark_qwen.persistent_qwen_graph import CPUReplayEmulator,graph_pool_accounting
        segment=dict(address=4096,total_size=4096,allocated_size=0,active_size=0,device=0,
            segment_pool_id=(0,11),blocks=[dict(size=4096,state='inactive')])
        class FailSecond(CPUReplayEmulator):
            calls=0
            def capture(self,*args):
                self.calls+=1
                if self.calls==2:return graph_pool_accounting((0,11),[segment],0,8192,8192,2048)
                return super().capture(*args)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'measured=4096, reservation=2048'):self.run_fixture(self.target(FailSecond()),tmp)
            result=json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(result['stages'],dict(native_eager='passed',capture='failed',mixed_execution='pending'))
            self.assertEqual(len(result['observations']),6);self.assertEqual([r['family'] for r in result['captures']],['S3'])
            self.assertEqual(result['current_attempt']['family'],'L5')
            self.assertTrue((Path(tmp)/'failed-graph-pool-snapshot.json').exists());self.assertTrue((Path(tmp)/'failed-prepared-buffer-state.pt').exists())

    def test_changed_same_B_metadata_saved_before_replay_refusal(self):
        import torch
        target=self.target();original=target.prepare_graph_verify
        def stale(*args,**kw):
            ticket=original(*args,**kw)
            if tuple(t.shape[1] for t in args[0].values())==(2,3):target.pool._workspaces[kw['bucket']].cu_query[1]=1
            return ticket
        with tempfile.TemporaryDirectory() as tmp,patch.object(target,'prepare_graph_verify',side_effect=stale):
            with self.assertRaisesRegex(AssertionError,'actual_ordered_metadata'):self.run_fixture(target,tmp)
            result=json.loads((Path(tmp)/'result.json').read_text());self.assertEqual(result['current_attempt']['case'],3)
            self.assertEqual(len(result['observations']),9)
            actual=torch.load(Path(tmp)/'mixed-3-inputs.pt',weights_only=True);self.assertEqual(actual['cu_query'].tolist(),[0,1,5])
            self.assertFalse((Path(tmp)/'mixed-3-outputs.pt').exists())

    def test_bad_replay_keeps_output_and_python_emulator_cannot_claim_real_replay(self):
        from dspark_qwen.persistent_qwen_graph import CPUReplayEmulator
        class BadReplay(CPUReplayEmulator):
            def submit(self,graph,receipt,device):
                event=super().submit(graph,receipt,device);graph.__self__.last.add_(1.);return event
        class PretendGPU(CPUReplayEmulator):
            def capture(self,body,*args):body();body();return super().capture(body,*args)
        for backend,require_gpu,error in ((BadReplay(),False,'fixed_limits'),(PretendGPU(),True,'python_free_real_replay')):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(AssertionError,error):self.run_fixture(self.target(backend),tmp,require_gpu=require_gpu)
                result=json.loads((Path(tmp)/'result.json').read_text())
                self.assertEqual(result['stages']['capture'],'passed');self.assertEqual(result['stages']['mixed_execution'],'failed')
                self.assertEqual(len(result['observations']),7);self.assertTrue((Path(tmp)/'mixed-0-outputs.pt').exists())

    def test_deadline_keeps_exact_expected_domain_and_durable_partial_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TimeoutError):self.run_fixture(self.target(),tmp,deadline=0.)
            result=json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(result['status'],'partial_deadline');self.assertEqual(len(result['expected_observations']),12)
            self.assertEqual(result['stages']['capture'],'pending');self.assertTrue((Path(tmp)/'failed-resident-and-scratch.pt').exists())


if __name__=='__main__':unittest.main()
