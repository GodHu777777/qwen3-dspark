import builtins
import copy
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
SPEC=importlib.util.spec_from_file_location('packed_gate',ROOT/'scripts/probe_packed_decoder.py')
gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)


class PackedGateTests(unittest.TestCase):
    def binding_fixture(self,root):
        model=root/'model';model.mkdir();cfg=dict(gate.BASE.PROTOCOL['model_architecture'],rope_scaling=None)
        (model/'config.json').write_text(json.dumps(cfg));(model/'model.safetensors').write_bytes(b'fixture')
        fingerprints={p.name:gate.sha(p) for p in model.iterdir()}
        generation=root/'generation.json';generation.write_text(json.dumps(dict(model_files_sha256=fingerprints)))
        checkpoint=root/'checkpoint';checkpoint.mkdir();(checkpoint/'draft.safetensors').write_bytes(b'fixture-draft')
        metadata=dict(step=1280,draft_weights_sha256=gate.sha(checkpoint/'draft.safetensors'),
            draft_config=dict(num_layers=5,block_size=7,layer_ids=[1,7,14,21,26]),identity=dict(target_fingerprint=fingerprints))
        (checkpoint/'metadata.json').write_text(json.dumps(metadata))
        hashes=dict(checkpoint_weights_sha256=gate.sha(checkpoint/'draft.safetensors'),checkpoint_metadata_sha256=gate.sha(checkpoint/'metadata.json'))
        return model,generation,checkpoint,hashes

    def test_stdlib_binding_refuses_other_checkpoint_and_tiny_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,g,c,h=self.binding_fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,'exact selected'):gate.bind(m,g,c)
            original=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name.split('.')[0] in ('torch','transformers','dspark_qwen'):raise AssertionError('heavyweight import')
                return original(name,*args,**kwargs)
            with patch.dict(gate.PROTOCOL,h),patch('builtins.__import__',side_effect=guarded),patch('sys.stdout',new_callable=io.StringIO):
                self.assertEqual(gate.main(['--model',str(m),'--generation-manifest',str(g),'--checkpoint',str(c)]),0)
            cfg=json.loads((m/'config.json').read_text());cfg['hidden_size']=32;(m/'config.json').write_text(json.dumps(cfg))
            g.write_text(json.dumps(dict(model_files_sha256={p.name:gate.sha(p) for p in m.iterdir()})))
            with self.assertRaisesRegex(ValueError,'not a tiny'):gate.bind(m,g,c)

    def test_protocol_threshold_drift_is_rejected(self):
        with patch.dict(gate.PROTOCOL,attention_max_rms=.006):
            with self.assertRaisesRegex(ValueError,'thresholds differ'):gate.validate_limits()

    def test_fixed_timeout_owned_group_and_no_system_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m,g,c,h=self.binding_fixture(root);process=Mock(pid=765432)
            process.wait.side_effect=[subprocess.TimeoutExpired('mock',300),-9]
            with patch.dict(gate.PROTOCOL,h),patch('subprocess.Popen',return_value=process),patch('os.killpg') as kill:
                self.assertEqual(gate.main(['--execute','--model',str(m),'--generation-manifest',str(g),'--checkpoint',str(c),'--output',str(root/'out')]),1)
            kill.assert_called_once_with(765432,signal.SIGKILL)
            r=json.loads((root/'out/completion.json').read_text());self.assertTrue(r['timed_out']);self.assertFalse(r['execution_completed']);self.assertFalse(r['system_pass_claimed'])

    def fixture(self,bias=0.):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from dspark_qwen.config import DraftConfig
        from dspark_qwen.model import DSparkDraft
        from dspark_qwen.packed_draft import PackedDraft,test_only_noncausal_varlen
        from dspark_qwen.packed_sampling import PackedSpeculativeSession
        from dspark_qwen.varlen_target import VarlenPackedTarget,test_only_dense_varlen
        torch.set_num_threads(2);torch.manual_seed(334)
        config=Qwen3Config(vocab_size=1024,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,attention_dropout=0.)
        config._attn_implementation='sdpa';model=Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
        reference=copy.deepcopy(model);draft=DSparkDraft(model,DraftConfig(layer_ids=(0,2),num_layers=2,block_size=7,markov_rank=8,mask_token_id=1023)).eval()
        target=VarlenPackedTarget(model,(0,2),test_kernel=test_only_dense_varlen)
        def kernel(q,k,v,layout,*,scale):return (test_only_noncausal_varlen(q,k,v,layout,scale=scale)+bias).to(q.dtype)
        packed=PackedDraft(draft,test_kernel=kernel)
        return PackedSpeculativeSession(target,packed,amp=True),reference

    def test_full_tiny_trained_shape_lifecycle_and_poison(self):
        session,reference=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);r=gate.exercise(session,reference,out,gate.new_report())
            self.assertEqual(r['status'],'completed');self.assertEqual(r['structural_status'],'passed')
            self.assertEqual(len(r['normal_layers']),6);self.assertEqual(len(r['target_comparisons']),5)
            self.assertEqual(len(r['rounds']),3);self.assertEqual(len(r['poison_pairs']),2)
            self.assertTrue(all(x['A_inputs_outputs_logits_cache_exact'] for x in r['poison_pairs']))
            self.assertFalse(r['system_pass_claimed']);self.assertEqual(r['prior_target_whole_qwen_gate'],'failed_unchanged')
            self.assertTrue(all((out/x['tensors']['file']).exists() for x in r['normal_layers']))
            import torch
            for layer in r['normal_layers']:
                self.assertEqual(layer['oracle_dtype'],'torch.float32')
                # Regression: explicit .float() inputs are insufficient while AMP remains active.
                saved=torch.load(out/layer['tensors']['file'],weights_only=True)
                self.assertEqual(saved['fp32_oracle'].dtype,torch.float32)

    def test_finite_numerical_failure_keeps_complete_matrix_and_original_limits(self):
        session,reference=self.fixture(.1)
        with tempfile.TemporaryDirectory() as tmp:
            r=gate.exercise(session,reference,Path(tmp),gate.new_report())
            self.assertEqual(r['status'],'completed_with_failed_draft_layer_gate')
            self.assertEqual(r['structural_status'],'passed');self.assertEqual(r['draft_layer_numerical_status'],'failed')
            self.assertEqual(r['endpoint_comparison_status'],'completed');self.assertEqual(len(r['normal_layers']),6)


if __name__=='__main__':unittest.main()
