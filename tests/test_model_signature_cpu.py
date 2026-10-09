import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('signature_cpu_benchmark',ROOT/'scripts/benchmark_model_signature_cpu.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


class SignatureMeasurementTests(unittest.TestCase):
    def test_exact_output_with_aliases_mixed_types_and_mutations(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from dspark_qwen.persistent_qwen_graph import FullTargetExecution
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,tie_word_embeddings=True)
        config._attn_implementation='sdpa'
        with torch.device('meta'):model=Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
        model.register_buffer('mixed_dtype',torch.ones(3,dtype=torch.float32,device='meta'))
        model.register_buffer('mixed_device',torch.ones(3,dtype=torch.int64,device='cpu'))
        model.register_buffer('alias',model.mixed_dtype)
        owner=SimpleNamespace(model=model,layer_ids=[0,1]);execution=SimpleNamespace(target=owner)
        def snapshot():
            original=FullTargetExecution._model_signature(execution)
            self.assertEqual(original,bench.signature_recomposed(owner))
            self.assertEqual(original,bench.signature_recomposed(owner,local_strings=True))
            self.assertTrue(all(type(row[4]) is str and type(row[5]) is str for row in original[-1]))
            return original
        before=snapshot()
        model.mixed_device.add_(1);self.assertNotEqual(before,snapshot());before=snapshot()
        model.config.rms_norm_eps*=2;self.assertNotEqual(before,snapshot());before=snapshot()
        model.train();self.assertNotEqual(before,snapshot())
        result=bench.measure(owner,iterations=2,repeats=3,warmup=0,deadline_seconds=10)
        self.assertTrue(result['output_equivalent_before_after'])
        self.assertEqual(len(result['components']),12)
        self.assertTrue(all(len(x['samples_ns_per_call'])==3 for x in result['components'].values()))


if __name__=='__main__':unittest.main()
