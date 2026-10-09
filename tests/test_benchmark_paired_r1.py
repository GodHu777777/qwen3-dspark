import builtins
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load('test_paired_r1_runner', ROOT/'scripts/benchmark_paired_r1.py')
fixture = load('test_paired_r1_binding_fixture', ROOT/'tests/test_probe_packed_decoder.py')
MANIFEST, MANIFEST_SHA = bench.SHARED.load_workloads(ROOT/'configs/performance-workloads.example.json')


class PairedR1ProtocolTests(unittest.TestCase):
    def test_binding_stays_stdlib_pins_trained_input_and_two_case_36_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, generation, checkpoint, hashes = fixture.PackedGateTests().binding_fixture(Path(tmp))
            manifest = copy.deepcopy(MANIFEST)
            manifest['model']['config_sha256'] = bench.GUARD.sha(model/'config.json')
            path = Path(tmp)/'workloads.json'; path.write_text(json.dumps(manifest))
            original = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in ('torch', 'transformers', 'dspark_qwen'):
                    raise AssertionError('heavyweight import in dry-run')
                return original(name, *args, **kwargs)
            with patch.dict(bench.PROFILE.GATE.PROTOCOL, hashes), \
                    patch.object(bench, 'baseline_reference', return_value={'fixture': True}), \
                    patch('builtins.__import__', side_effect=guarded), \
                    patch('sys.stdout', new_callable=io.StringIO) as output:
                self.assertEqual(bench.main(['--dry-run', '--model', str(model), '--generation-manifest', str(generation),
                    '--checkpoint', str(checkpoint), '--workloads', str(path)]), 0)
            binding = json.loads(output.getvalue())['binding']
            self.assertEqual(binding['selected_case_ids'], ['r1-c64', 'r1-c256'])
            self.assertEqual(len(binding['expected_samples']), 36)
            self.assertEqual(binding['checkpoint_weights_sha256'], hashes['checkpoint_weights_sha256'])
            self.assertEqual(binding['protocol']['graph_reservation_bytes'], 512*1024**2)
            self.assertFalse(binding['protocol']['all_graph'])
            self.assertFalse(binding['protocol']['full_six_case_panel'])
            self.assertEqual(binding['protocol']['timeout_seconds'], 1800)
            self.assertIn('dspark_qwen/persistent_qwen_graph.py', binding['source_sha256'])
            self.assertIn('scripts/benchmark_paired_r1.py', binding['source_sha256'])

    def test_schedule_keeps_original_manifest_and_alternates_each_complete_pair(self):
        old = copy.deepcopy(MANIFEST)
        schedule = list(bench.schedule(MANIFEST))
        self.assertEqual(len(schedule), 36)
        self.assertEqual(MANIFEST, old)
        self.assertEqual(len(MANIFEST['cases']), 6)
        self.assertEqual({a: sum(x[3] == a for x in schedule) for a in bench.ARMS}, dict.fromkeys(bench.ARMS, 18))
        self.assertEqual(sum(x[0] == 'primary' for x in schedule), 20)
        for i in range(0, 36, 2):
            first, second = schedule[i:i+2]
            self.assertEqual(first[:3], second[:3])
            self.assertEqual((first[3], second[3]), bench.ARMS if first[1] % 2 == 0 else bench.ARMS[::-1])

    def test_frozen_vllm_match_is_recomputed_and_mismatch_rejected(self):
        source = json.loads((bench.BASELINE/'source-identity.json').read_text())
        ref = bench.baseline_reference(MANIFEST_SHA, source['model_tokenizer_sha256'])
        self.assertEqual([x['case_id'] for x in ref['cells']], list(bench.CASE_IDS))
        self.assertAlmostEqual(ref['cells'][0]['pooled_output_tokens_per_second'], 131.26405631263103)
        self.assertAlmostEqual(ref['cells'][1]['pooled_output_tokens_per_second'], 127.33989818625602)
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            bench.baseline_reference('changed', source['model_tokenizer_sha256'])

    @staticmethod
    def fake_batch(factory, device, case, manifest, arm, **kwargs):
        return dict(arm=arm, batch_wall_seconds=1. if arm == 'target_only' else 2., output_tokens=128,
                    execution_coverage=dict(actual_gpu_graph_rounds=16, eager_tail_rounds=1))

    def test_complete_run_recomputes_both_arms_and_36_identities(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bench, 'batch', side_effect=self.fake_batch):
            out = Path(tmp)/'run'; result = bench.run(None, None, MANIFEST, out)
            samples = [json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
            self.assertEqual(result['sample_count'], 36)
            self.assertEqual(result['aggregates'], bench.verify_complete(samples, MANIFEST))
            for cells in result['aggregates']['primary_execution'].values():
                for cell in cells:
                    self.assertEqual(cell['coverage_samples'], 5)
                    self.assertEqual(cell['totals'], dict(actual_gpu_graph_rounds=80, eager_tail_rounds=5))
                    self.assertEqual(len(cell['primary_batch_wall_seconds']), 5)
            self.assertTrue(all(x['speculative_over_target_only_rate_ratio'] == .5
                                for x in result['aggregates']['paired_comparisons']))
            with self.assertRaisesRegex(ValueError, '36-batch'):
                bench.verify_complete(samples[:-1], MANIFEST)
            self.assertFalse(result['all_graph'])

    def test_last_diagnostic_failure_and_worker_deadline_preserve_partial(self):
        calls = []
        def failure(*args, **kwargs):
            calls.append(None)
            if len(calls) == 36: raise TimeoutError('final diagnostic deadline')
            return self.fake_batch(*args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(bench, 'batch', side_effect=failure):
            out = Path(tmp)/'run'
            with self.assertRaises(TimeoutError): bench.run(None, None, MANIFEST, out)
            partial = json.loads((out/'result.json').read_text())
            self.assertEqual(partial['status'], 'partial_deadline')
            self.assertEqual(partial['sample_count'], 35)
            self.assertEqual(len(partial['expected_samples']), 36)
            self.assertTrue(all(x['complete_primary_pair'] for x in partial['aggregates']['paired_comparisons']))
            with patch.object(bench, 'worker', side_effect=TimeoutError('outer deadline')):
                self.assertEqual(bench.main(['--worker-binding', str(Path(tmp)/'ignored'), '--output', str(out)]), 1)
            self.assertEqual(json.loads((out/'result.json').read_text()), partial)

    def test_python_replay_and_exact_physical_q_are_enforced(self):
        work = dict(execution_kind='rocm_full_target_graph', physical_query_tokens=8)
        self.assertTrue(bench.execution_record(work, 8, (3, 3, 3), (3, 3, 3), 'cuda')['actual_gpu_graph'])
        with self.assertRaisesRegex(ValueError, 'Python counters'):
            bench.execution_record(work, 8, (3, 3, 3), (4, 4, 4), 'cuda')
        with self.assertRaisesRegex(ValueError, 'padding'):
            bench.execution_record(dict(work, physical_query_tokens=9), 8, (3,), (3,), 'cuda')
        with self.assertRaisesRegex(ValueError, 'explicit backend'):
            bench.execution_record(dict(physical_query_tokens=1), 1, (3,), (4,), 'cuda')


class PairedR1RealFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def factory(self, *, capture=True):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from dspark_qwen.config import DraftConfig
        from dspark_qwen.model import DSparkDraft
        from dspark_qwen.packed_draft import PackedDraft, test_only_noncausal_varlen
        from dspark_qwen.persistent_qwen_graph import CPUReplayEmulator
        from dspark_qwen.persistent_qwen_target import PersistentQwenTarget, test_only_persistent_sdpa
        torch.manual_seed(334)
        config = Qwen3Config(vocab_size=2048, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512, attention_dropout=0.)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
        draft = DSparkDraft(model, DraftConfig(layer_ids=(0, 2), num_layers=2, block_size=7,
            markov_rank=8, mask_token_id=2047)).eval()
        constructed = []
        def builder(m, ids, **kw):
            kw['graph_backend'] = CPUReplayEmulator()
            target = PersistentQwenTarget(m, ids, test_kernel=test_only_persistent_sdpa, **kw)
            constructed.append(target)
            return target
        factory = bench.PairedFactory(model, draft, target_builder=builder,
            draft_builder=lambda d: PackedDraft(d, test_kernel=test_only_noncausal_varlen))
        factory.constructed = constructed
        self.addCleanup(factory.close)
        if capture:
            result = factory.capture(MANIFEST)
            self.assertEqual(result['buckets'], 10)
            self.assertEqual(result['graph_count'], 2)
            self.assertEqual([x['query_tokens'] for x in result['captures']], [1, 8])
            self.assertEqual(factory.target.lengths, {})
            self.assertGreater(result['setup_validation_seconds'], 0.)
            for row in result['captures']:
                self.assertEqual(row['status'], 'validated')
                self.assertEqual(row['validation']['status'], 'passed')
                self.assertEqual(len(row['validation']['comparisons']), 12)
                self.assertTrue(row['validation']['resident_unchanged_after_replay_abort'])
        return factory, torch.device('cpu')

    def test_setup_new_shapes_fixed_limits_failure_keeps_original_outputs(self):
        import torch
        make, _ = self.factory(capture=False)
        original = make.target.verify
        progress = []
        def corrupted(*args, **kwargs):
            features = original(*args, **kwargs)
            if features.work.get('execution_kind') == 'cpu_replay_emulator_not_gpu_graph':
                features.last.add_(1.)
            return features
        with tempfile.TemporaryDirectory() as tmp, patch.object(make.target, 'verify', side_effect=corrupted):
            with self.assertRaisesRegex(AssertionError, 'fixed numerical limits'):
                make.capture(MANIFEST, evidence_dir=Path(tmp), progress=lambda rows: progress.append(copy.deepcopy(rows)))
            row = progress[-1][0]
            self.assertEqual(row['status'], 'validation_failed')
            self.assertEqual(row['validation']['limits'], dict(atol=.02, rtol=.02, max_rms=.005))
            self.assertFalse(row['validation']['comparisons']['final_norm']['passed'])
            path = Path(tmp)/row['validation']['tensors']['file']
            self.assertEqual(bench.GUARD.sha(path), row['validation']['tensors']['sha256'])
            raw = torch.load(path, weights_only=True)
            self.assertIn('eager_scratch_keys', raw)
            self.assertIn('replay_scratch_values', raw)
            self.assertTrue(torch.equal(raw['resident_keys_before'], raw['resident_keys_after']))
            self.assertGreater(float((raw['replay_final_norm']-raw['eager_final_norm']).abs().max()), .5)
            self.assertFalse(make.ready)

    def test_real_complete_target_only128_same_factory_and_repeat_hash(self):
        make, device = self.factory()
        case = bench.selected_manifest(MANIFEST)['cases'][0]
        target_id = id(make.target); model_id = id(make.target.model); draft_id = id(make.draft)
        first = bench.batch(make, device, case, MANIFEST, 'target_only')
        second = bench.batch(make, device, case, MANIFEST, 'target_only', diagnostic=True)
        self.assertEqual(first['output_tokens'], 128)
        self.assertEqual(first['committed_round_output_tokens'], 127)
        self.assertEqual(len(first['rounds']), 127)
        self.assertEqual(first['final_context_lengths'], {'r0': 191})
        self.assertEqual(first['requests'][0]['output_sha256'], second['requests'][0]['output_sha256'])
        self.assertIsNone(first['requests'][0]['ttft_seconds'])
        self.assertGreater(second['requests'][0]['ttft_seconds'], 0.)
        self.assertEqual(len(make.constructed), 1)
        self.assertEqual((id(make.target), id(make.target.model), id(make.draft)), (target_id, model_id, draft_id))
        self.assertEqual(first['selected_layer_ids'], [0, 2])
        self.assertEqual(first['execution_coverage']['actual_gpu_graph_rounds'], 0)
        self.assertEqual(first['execution_coverage']['graph_plan_rounds'], 127)
        for row in first['rounds']:
            self.assertEqual(row['execution_kind'], 'cpu_replay_emulator_not_gpu_graph')
            self.assertEqual(row['work']['draft_forward_calls'], 0)

    def test_real_all_six_eager_tail_shapes_are_timed_and_full_shadow_even_q1(self):
        make, device = self.factory()
        case = bench.selected_manifest(MANIFEST)['cases'][0]
        real_eager = make.target.verify_eager
        for q in range(1, 9):
            manifest = dict(MANIFEST, output_tokens=q+1)
            events = []
            def observed(*args, **kwargs):
                events.append(bench.time.perf_counter())
                return real_eager(*args, **kwargs)
            with patch.object(make.target, 'verify_eager', side_effect=observed):
                row = bench.batch(make, device, case, manifest, 'fixed_gamma7_full_shadow')
            first = row['rounds'][0]
            self.assertEqual(first['query_tokens'], q)
            self.assertEqual(first['allocation'], {'r0': q-1})
            self.assertEqual(row['admission_output_tokens'], 1)
            self.assertEqual(row['committed_round_output_tokens'], q)
            self.assertEqual(row['final_context_lengths'], {'r0': 64+q})
            self.assertEqual(first['graph_plan_hit'], q in (1, 8))
            self.assertEqual(first['eager_tail'], q not in (1, 8))
            self.assertGreater(row['batch_wall_seconds'], 0.)
            if q not in (1, 8):
                self.assertGreaterEqual(len(events), 2)  # Admission plus real eager verification.
                self.assertGreaterEqual(row['execution_coverage']['eager_tail_query_rows'], q)
            for round_row in row['rounds']:
                self.assertEqual(round_row['work']['proposal_batches'][0]['backbone']['query_rows'], 7)
                self.assertEqual(round_row['actual_physical_b'], round_row['actual_logical_b'])
            self.assertIsNone(make.target._pending)
            self.assertIsNone(make.target._feature_lease)

    def test_real_speculative128_then_other_arm_uses_fresh_state_and_loaded_models(self):
        make, device = self.factory()
        case = bench.selected_manifest(MANIFEST)['cases'][1]
        spec = bench.batch(make, device, case, MANIFEST, 'fixed_gamma7_full_shadow')
        self.assertEqual(spec['output_tokens'], 128)
        self.assertEqual(spec['final_context_lengths'], {'r0': 383})
        self.assertGreater(spec['execution_coverage']['graph_plan_query_rows'], 0)
        target = bench.batch(make, device, case, MANIFEST, 'target_only')
        self.assertEqual(target['output_tokens'], 128)
        self.assertEqual(target['final_context_lengths'], {'r0': 383})
        self.assertEqual(target['selected_layer_ids'], spec['selected_layer_ids'])
        self.assertEqual(target['prefill_work']['prefill_lm_head_rows'], 1)
        self.assertEqual(spec['prefill_work']['prefill_lm_head_rows'], 1)
        self.assertEqual(len(make.constructed), 1)

    def test_real_next_batch_deadline_retains_only_completed_sample(self):
        make, device = self.factory()
        manifest = dict(MANIFEST, output_tokens=3)
        original = bench.batch; calls = []
        def expire(*args, **kwargs):
            calls.append(None)
            if len(calls) == 2: kwargs['deadline'] = 0.
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(bench, 'batch', side_effect=expire):
            out = Path(tmp)/'run'
            with self.assertRaisesRegex(TimeoutError, 'incomplete batch'):
                bench.run(make, device, manifest, out)
            result = json.loads((out/'result.json').read_text())
            self.assertEqual(result['status'], 'partial_deadline')
            self.assertEqual(result['sample_count'], 1)
            self.assertEqual(len(result['expected_samples']), 36)
            self.assertEqual(len((out/'samples.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
