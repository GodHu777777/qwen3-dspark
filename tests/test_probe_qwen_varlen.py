"""CPU-only whole-Qwen gate control tests; tiny models never enter the formal CLI."""
import builtins
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('qwen_gate',ROOT/'scripts/probe_qwen_varlen.py')
gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)


def binding_fixture(root):
    model=root/'model';model.mkdir()
    config=dict(gate.PROTOCOL['model_architecture'],use_sliding_window=False,rope_scaling=None)
    (model/'config.json').write_text(json.dumps(config));(model/'model.safetensors').write_bytes(b'identity-only fixture')
    manifest=root/'generation.json'
    manifest.write_text(json.dumps(dict(model_files_sha256={p.name:gate.sha(p) for p in model.iterdir()})))
    return model,manifest


class WholeBindingTests(unittest.TestCase):
    def test_stdlib_only_actual_architecture_binding_and_tiny_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,manifest=binding_fixture(Path(tmp));original=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name.split('.')[0] in ('torch','transformers','dspark_qwen'):raise AssertionError('Runtime imported')
                return original(name,*args,**kwargs)
            with patch('builtins.__import__',side_effect=guarded),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(gate.main(['--model',str(model),'--generation-manifest',str(manifest)]),0)
            cfg=json.loads((model/'config.json').read_text());cfg['num_hidden_layers']=3
            (model/'config.json').write_text(json.dumps(cfg))
            manifest.write_text(json.dumps(dict(model_files_sha256={p.name:gate.sha(p) for p in model.iterdir()})))
            with self.assertRaisesRegex(ValueError,'not a tiny substitute'):gate.bind(model,manifest)

    def test_changed_weights_are_rejected_before_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,manifest=binding_fixture(Path(tmp));(model/'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):gate.bind(model,manifest)

    def test_timeout_reaps_only_owned_process_group_and_preserves_separate_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,manifest=binding_fixture(root)
            process=Mock(pid=432100);process.wait.side_effect=[subprocess.TimeoutExpired('mock',300),-9]
            with patch('subprocess.Popen',return_value=process),patch('os.killpg') as kill:
                result=gate.main(['--execute','--model',str(model),'--generation-manifest',str(manifest),'--output',str(root/'out')])
            self.assertEqual(result,1);kill.assert_called_once_with(432100,signal.SIGKILL)
            record=json.loads((root/'out/completion.json').read_text())
            self.assertTrue(record['timed_out']);self.assertFalse(record['execution_completed']);self.assertFalse(record['system_pass_claimed'])
            self.assertEqual(record['layer_attention_numerical_status'],'pending')


class WholeCpuFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.torch=torch

    def fixture(self,kernel=None):
        torch=self.torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from dspark_qwen.packed_target import PackedTarget
        from dspark_qwen.varlen_target import VarlenPackedTarget
        torch.manual_seed(111)
        cfg=Qwen3Config(vocab_size=1024,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,attention_dropout=0.)
        cfg._attn_implementation='sdpa'
        model=Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
        def exact(q,k,v,layout,*,scale):return gate.attention_oracle(q,k,v,layout,scale).to(q.dtype)
        native=VarlenPackedTarget(copy.deepcopy(model),(0,1),test_kernel=kernel or exact)
        dense=PackedTarget(copy.deepcopy(model),(0,1));independent={'_model':copy.deepcopy(model)}
        return native,dense,independent

    def execute(self,out,targets):
        report=gate.new_report()
        with self.torch.no_grad():gate.exercise(*targets,out,report)
        return report

    def test_real_tiny_forward_lifecycle_exact_poison_and_all_normal_layer_oracles(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);report=self.execute(out,self.fixture())
            self.assertEqual(report['status'],'completed')
            self.assertEqual(report['structural_status'],'passed')
            self.assertEqual(report['layer_attention_numerical_status'],'passed')
            self.assertEqual(report['numerical_comparison_status'],'completed')
            self.assertFalse(report['system_pass_claimed'])
            self.assertEqual(report['runtime_scope'],'cpu_test_fixture_only')
            required=[r for r in report['layer_attention'] if r['attention_gate_required']]
            self.assertEqual(len(required),12);self.assertEqual(len(report['layer_attention']),24)
            self.assertTrue(all(r['gather_content_exact'] and r['passes_fixed_limits'] for r in required))
            self.assertEqual(len(report['normal_steps']),4);self.assertEqual(len(report['poison_pairs']),2)
            self.assertTrue(all(r['A_QKV_exact'] and r['hidden_logits_and_all_A_KV_exact'] for r in report['poison_pairs']))
            self.assertTrue(all((out/r['private_comparison_tensors']['file']).exists() for r in report['normal_steps']))
            for row in report['normal_steps']:
                self.assertTrue(all('logits_vs_independent' in m for m in row['requests'].values()))

    def test_attention_mismatch_retains_failed_tensors_without_relaxing_gate(self):
        def wrong(q,k,v,layout,*,scale):return (gate.attention_oracle(q,k,v,layout,scale)+.1).to(q.dtype)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);report=self.execute(out,self.fixture(wrong))
            self.assertEqual(report['status'],'completed_with_failed_layer_attention_gate')
            self.assertEqual(report['structural_status'],'passed')
            self.assertEqual(report['layer_attention_numerical_status'],'failed')
            self.assertEqual(report['numerical_comparison_status'],'completed')
            failures=[r for r in report['layer_attention'] if r['attention_gate_required'] and not r['passes_fixed_limits']]
            self.assertTrue(failures)
            for r in failures:
                tensors=self.torch.load(out/r['failed_tensors']['file'],weights_only=True)
                self.assertEqual(set(tensors),{'q','k','v','cu_query','cu_key','output','fp32_oracle'})

    def test_independent_gather_mapping_detects_corruption_and_keeps_partial_evidence(self):
        targets=self.fixture();native=targets[0];original=native._attention_payload
        def bad(qr,qp,kr,kp):
            result=original(qr,qp,kr,kp);indices=result[1]['packed_layout'].gather_indices
            indices[[0,1]]=indices[[1,0]]
            return result
        native._attention_payload=bad
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with self.assertRaisesRegex(AssertionError,'independent request mapping'):self.execute(out,targets)
            report=json.loads((out/'worker-result.json').read_text())
            self.assertEqual(report['status'],'execution_failed');self.assertEqual(len(report['layer_attention']),1)
            self.assertTrue((out/report['layer_attention'][0]['failed_tensors']['file']).exists())

    def test_actual_model_position_ids_are_checked_before_attention(self):
        targets=self.fixture()
        def corrupt(_module,args,kwargs):
            kwargs['position_ids']=self.torch.zeros_like(kwargs['position_ids'])
            return args,kwargs
        hook=targets[0].model.model.register_forward_pre_hook(corrupt,with_kwargs=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                with self.assertRaisesRegex(AssertionError,'actual_model_input_tokens_and_RoPE_positions'):
                    self.execute(out,targets)
                report=json.loads((out/'worker-result.json').read_text())
                self.assertEqual(report['status'],'execution_failed')
                self.assertEqual(report['layer_attention'],[])
                self.assertTrue((out/report['failure_state_evidence']['file']).exists())
        finally:hook.remove()

    def test_operator_failure_preserves_earlier_layer_and_failing_inputs(self):
        calls=[]
        def fail(q,k,v,layout,*,scale):
            calls.append(1)
            if len(calls)==2:raise RuntimeError('injected layer failure')
            return gate.attention_oracle(q,k,v,layout,scale).to(q.dtype)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'injected layer failure'):self.execute(out,self.fixture(fail))
            report=json.loads((out/'worker-result.json').read_text())
            self.assertEqual([r['status'] for r in report['layer_attention']],['observed','failed'])
            self.assertTrue((out/report['layer_attention'][1]['failed_tensors']['file']).exists())


if __name__=='__main__':unittest.main()
