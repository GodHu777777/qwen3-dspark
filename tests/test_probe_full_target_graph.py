"""CPU contract tests for the separate full-target GPU capability runner."""
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
probe=load('full_target_graph_probe_tests',ROOT/'scripts/probe_full_target_graph.py')
fixture=load('full_graph_binding_fixture',ROOT/'tests/test_probe_packed_decoder.py')

class BindingTests(unittest.TestCase):
    def test_binding_has_no_heavyweight_import_and_preserves_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,g,_,_=fixture.PackedGateTests().binding_fixture(Path(tmp))
            workload=json.loads((ROOT/'configs/performance-workloads.example.json').read_text())
            workload['model']['config_sha256']=probe.GUARD.sha(m/'config.json');w=Path(tmp)/'workload.json';w.write_text(json.dumps(workload))
            original=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name.split('.')[0] in ('torch','transformers','dspark_qwen'):raise AssertionError('Heavyweight binding import')
                return original(name,*args,**kwargs)
            with patch('builtins.__import__',side_effect=guarded),patch('sys.stdout',new_callable=io.StringIO) as output:
                self.assertEqual(probe.main(['--model',str(m),'--generation-manifest',str(g),'--workloads',str(w)]),0)
            row=json.loads(output.getvalue());self.assertEqual(row['status'],'dry_run_no_torch_no_gpu')
            p=row['binding']['protocol'];self.assertEqual(p['contexts'],[[128,128],[129,131],[130,135]])
            self.assertEqual(p['selected_layers'],[1,7,14,21,26]);self.assertEqual(p['comparisons']['max_rms'],.005)
            self.assertEqual(p['timeout_seconds'],300);self.assertFalse(p['performance_measurement'])

class ContractTests(unittest.TestCase):
    def target(self,backend=None):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from dspark_qwen.persistent_qwen_target import PersistentQwenTarget,test_only_persistent_sdpa
        from dspark_qwen.persistent_qwen_graph import CPUReplayEmulator
        torch.manual_seed(993);torch.set_num_threads(2)
        cfg=Qwen3Config(vocab_size=1200,hidden_size=32,intermediate_size=48,num_hidden_layers=4,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,attention_dropout=0.)
        cfg._attn_implementation='sdpa';model=Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
        return PersistentQwenTarget(model,(0,2,3),slots=3,context_capacity=148,max_query_tokens=273,
            test_kernel=test_only_persistent_sdpa,graph_backend=backend or CPUReplayEmulator(),max_graph_buckets=1,
            graph_byte_budget=probe.PROTOCOL['graph_byte_budget'],workspace_byte_budget=probe.PROTOCOL['workspace_byte_budget'])

    def run_fixture(self,target,path,*,require_gpu=False,**kwargs):
        import torch
        manifest=json.loads((ROOT/'configs/performance-workloads.example.json').read_text())
        requests=next(c['requests'] for c in manifest['cases'] if c['case_id']=='r2-c256')
        with torch.no_grad():return probe.exercise(target,requests,path,probe.new_report(),require_gpu=require_gpu,**kwargs)

    def test_complete_real_model_fixture_preserves_all_evidence_and_labels_emulator(self):
        import torch
        target=self.target()
        with tempfile.TemporaryDirectory() as tmp:
            result=self.run_fixture(target,tmp);probe.verify_report(result,require_gpu=False)
            self.assertEqual(result['runtime_scope'],'cpu_emulator_contract_only')
            self.assertEqual(len(result['observations']),6)
            replay=result['observations'][3:]
            self.assertEqual([r['positions'] for r in replay],[[128,128,129,130,131],[129,131,132,133,134],[130,135,136,137,138]])
            self.assertTrue(all(not r['python_unchanged'] for r in replay))
            self.assertEqual(target.lengths,dict(r0=130,r1=135,idle=17))
            self.assertIsNone(target._feature_lease)
            with self.assertRaises(ValueError):probe.verify_report(result)
            evidence=torch.load(Path(tmp)/'eager-0-independent-layer-witnesses.pt',weights_only=True)
            outputs=torch.load(Path(tmp)/'eager-0-outputs.pt',weights_only=True)
            for index,layer in enumerate(target.layer_ids):
                self.assertTrue(torch.equal(evidence[f'raw_layer_{layer}'],outputs['context'][:,:,index*32:(index+1)*32]))
            self.assertTrue(torch.equal(evidence['final_norm'],outputs['final_norm']))
            self.assertEqual(len(list(Path(tmp).glob('replay-*-committed.pt'))),3)
            capture=torch.load(Path(tmp)/'capture-prepared-inputs.pt',weights_only=True)
            self.assertEqual(capture['positions'].tolist(),[128,128,129,130,131])
            broken=copy.deepcopy(result);broken['observations'].pop()
            with self.assertRaises(ValueError):probe.verify_report(broken,require_gpu=False)
        target.close()

    def test_wrong_selected_raw_slice_fails_before_capture_with_saved_outputs(self):
        target=self.target();original=probe.feature_tensors
        def swapped(*args):
            values=original(*args);values['context']=values['context'].roll(32,dims=-1);return values
        with tempfile.TemporaryDirectory() as tmp,patch.object(probe,'feature_tensors',side_effect=swapped):
            with self.assertRaisesRegex(AssertionError,'each_selected_raw_layer_identity'):self.run_fixture(target,tmp)
            result=json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(result['stages']['capture'],'pending');self.assertTrue((Path(tmp)/'eager-0-outputs.pt').exists())

    def test_values_only_resident_contamination_is_detected(self):
        target=self.target();original=target.verify_eager
        def corrupt(chunks,**kwargs):
            result=original(chunks,**kwargs)
            if tuple(chunks)==('r0','r1'):target.pool.values.add_(1.)
            return result
        with tempfile.TemporaryDirectory() as tmp,patch.object(target,'verify_eager',side_effect=corrupt):
            with self.assertRaisesRegex(AssertionError,'eager_scratch_only'):self.run_fixture(target,tmp)
            result=json.loads((Path(tmp)/'result.json').read_text());self.assertEqual(result['stages']['capture'],'pending')

    def test_budget_failure_keeps_scoped_snapshot_and_complete_eager_phase(self):
        from dspark_qwen.persistent_qwen_graph import CPUReplayEmulator,graph_pool_accounting
        segment=dict(address=4096,total_size=4096,allocated_size=0,active_size=0,device=0,
            segment_pool_id=(0,11),blocks=[dict(size=4096,state='inactive')])
        class BudgetFailure(CPUReplayEmulator):
            def capture(self,*args):return graph_pool_accounting((0,11),[segment],0,8192,8192,2048)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'measured=4096, reservation=2048'):self.run_fixture(self.target(BudgetFailure()),tmp)
            result=json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(result['stages'],dict(native_eager='passed',capture='failed',graph_replay='pending'))
            self.assertEqual(len(result['observations']),3)
            self.assertEqual(result['failed_graph_memory_accounting']['global_reserved_delta'],0)
            self.assertEqual(json.loads((Path(tmp)/'failed-graph-pool-snapshot.json').read_text())[0]['total_size'],4096)
            self.assertTrue((Path(tmp)/'failed-resident-and-scratch.pt').exists())
            self.assertTrue((Path(tmp)/'failed-capture-buffer-state.pt').exists())

    def test_replay_numerical_failure_preserves_outputs_and_never_completes(self):
        from dspark_qwen.persistent_qwen_graph import CPUReplayEmulator
        class BadReplay(CPUReplayEmulator):
            def submit(self,graph,receipt,device):
                event=super().submit(graph,receipt,device);graph.__self__.last.add_(1.);return event
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(AssertionError,'fixed_limits'):self.run_fixture(self.target(BadReplay()),tmp)
            result=json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(result['stages']['native_eager'],'passed');self.assertEqual(result['stages']['graph_replay'],'failed')
            self.assertEqual(len(result['observations']),4);self.assertTrue((Path(tmp)/'replay-0-outputs.pt').exists())

    def test_python_execution_cannot_pass_real_replay_counter_requirement(self):
        from dspark_qwen.persistent_qwen_graph import CPUReplayEmulator
        class PretendCapture(CPUReplayEmulator):
            def capture(self,body,*args):
                body();body();return super().capture(body,*args)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(AssertionError,'python_free_real_replay'):
                self.run_fixture(self.target(PretendCapture()),tmp,require_gpu=True)
            result=json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(result['stages']['capture'],'passed')
            self.assertEqual(result['stages']['graph_replay'],'failed')
            self.assertFalse(result['observations'][-1]['python_unchanged'])

    def test_deadline_retains_full_expected_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TimeoutError):self.run_fixture(self.target(),tmp,deadline=0.)
            result=json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(result['status'],'partial_deadline');self.assertEqual(len(result['expected_observations']),6)
            self.assertEqual(result['stages']['capture'],'pending')

if __name__=='__main__':unittest.main()
