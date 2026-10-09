import builtins
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_benchmark_paired_r1 import load, fixture

ROOT = Path(__file__).resolve().parents[1]
bench = load('test_r2_runner', ROOT/'scripts/benchmark_paired_r2.py')
old = load('test_r2_r1_fixture', ROOT/'tests/test_benchmark_paired_r1.py')
MANIFEST, SHA = bench.SHARED.load_workloads(ROOT/'configs/performance-workloads.example.json')


class R2ProtocolTests(unittest.TestCase):
    def test_stdlib_binding_and_original_r1_protocol(self):
        frozen = copy.deepcopy(bench.BASE.PROTOCOL)
        with tempfile.TemporaryDirectory() as tmp:
            model, generation, checkpoint, hashes = fixture.PackedGateTests().binding_fixture(Path(tmp))
            manifest = copy.deepcopy(MANIFEST); manifest['model']['config_sha256'] = bench.GUARD.sha(model/'config.json')
            path = Path(tmp)/'workloads.json'; path.write_text(json.dumps(manifest))
            original = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in ('torch', 'transformers', 'dspark_qwen'): raise AssertionError('heavy import')
                return original(name, *args, **kwargs)
            with patch.dict(bench.BASE.PROFILE.GATE.PROTOCOL, hashes), patch.object(bench, 'baseline_reference', return_value={}), \
                    patch('builtins.__import__', side_effect=guarded), patch('sys.stdout', new_callable=io.StringIO) as output:
                self.assertEqual(bench.main(['--dry-run', '--model', str(model), '--generation-manifest', str(generation),
                    '--checkpoint', str(checkpoint), '--workloads', str(path)]), 0)
            value = json.loads(output.getvalue())['binding']
            self.assertEqual(len(value['expected_samples']), 27)
            self.assertEqual(value['selected_case_ids'], ['r2-c256'])
            self.assertEqual(bench.BASE.PROTOCOL, frozen)
            self.assertEqual(bench.PROTOCOL['graph_byte_budget'], 1280*1024**2)
            self.assertEqual(bench.PROTOCOL['timeout_seconds'], 1800)
            self.assertFalse(bench.PROTOCOL['measured_sps'])

    def test_schedule_exact_domain_rotation_and_vllm256_denominator(self):
        rows = list(bench.schedule(MANIFEST)); self.assertEqual(len(rows), 27)
        positions = {a: [0, 0, 0] for a in bench.ARMS}
        for start in range(0, 27, 3):
            group = rows[start:start+3]; offset = (start//3)%3
            self.assertEqual(tuple(x[3] for x in group), bench.ARMS[offset:]+bench.ARMS[:offset])
            if group[0][0] == 'primary':
                for i, x in enumerate(group): positions[x[3]][i] += 1
        self.assertEqual(positions, bench.PROTOCOL['primary_arm_position_counts'])
        source = json.loads((bench.BASE.BASELINE/'source-identity.json').read_text())
        reference = bench.baseline_reference(SHA, source['model_tokenizer_sha256'])
        self.assertAlmostEqual(reference['cells'][0]['pooled_output_tokens_per_second'], 246.56672736039107)
        with self.assertRaises(ValueError): bench.baseline_reference('wrong', source['model_tokenizer_sha256'])
        buckets = bench.finite_buckets(); self.assertEqual(len(buckets.verification), 23)
        for r in (1, 2):
            import itertools
            for q in itertools.product(range(1, 9), repeat=r):
                self.assertEqual(buckets.select('verification', q, (375,)*r).query_tokens, sum(q))

    @staticmethod
    def fake_batch(factory, device, case, manifest, arm, **kwargs):
        return dict(arm=arm, batch_wall_seconds=float(bench.ARMS.index(arm)+1), output_tokens=256,
            execution_coverage={'eager_tail_rounds': 2}, full_shadow_positions=14,
            selected_proposal_tokens=0, accepted_draft_tokens=0, r1_tail_rounds=1)

    def test_complete_and_failed_last_diagnostic_preserve_exact_samples(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bench, 'batch', side_effect=self.fake_batch):
            out = Path(tmp)/'run'; result = bench.run(None, None, MANIFEST, out)
            self.assertEqual(result['sample_count'], 27)
            self.assertEqual(result['aggregates']['native_rates']['target_only'], 256.)
            rows = [json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
            with self.assertRaisesRegex(ValueError, '27-batch'): bench.verify_complete(rows[:-1], MANIFEST)
            def failed(*args, **kwargs):
                if failed.count == 26: raise TimeoutError('last diagnostic')
                failed.count += 1; return self.fake_batch(*args, **kwargs)
            failed.count = 0
            with patch.object(bench, 'batch', side_effect=failed):
                with self.assertRaises(TimeoutError): bench.run(None, None, MANIFEST, Path(tmp)/'failed')
            partial = json.loads((Path(tmp)/'failed/result.json').read_text())
            self.assertEqual(partial['sample_count'], 26); self.assertEqual(partial['status'], 'partial_deadline')
            self.assertTrue(partial['aggregates']['complete_primary_triple'])

    def test_execution_rejects_pseudo_replay_padding_and_r1_graph(self):
        row = dict(execution_kind='rocm_full_target_graph', physical_query_tokens=16)
        self.assertTrue(bench.execution_record(row, 16, (0, 0), (0, 0), 'cuda', active_count=2)['actual_gpu_graph'])
        for bad, after, r in [(row, (1, 1), 2), (dict(row, physical_query_tokens=17), (0, 0), 2),
                              (dict(row, physical_query_tokens=8), (0, 0), 1)]:
            with self.assertRaises(ValueError): bench.execution_record(bad, bad['physical_query_tokens'], (0, 0), after, 'cuda', active_count=r)

    def test_worker_failure_preserves_pool_snapshot_and_no_retry(self):
        error = RuntimeError('second capture pool reservation exceeded')
        error.memory_accounting = {'reserved_bytes': 512*1024**2, 'retained_bytes': 513*1024**2}
        error.pool_snapshot = {'segments': [{'total_size': 513*1024**2}]}
        with tempfile.TemporaryDirectory() as tmp, patch.object(bench, 'worker', side_effect=error) as worker:
            out = Path(tmp)
            (out/'capture-progress.json').write_text(json.dumps({'captures': [{'status': 'validated'}, {'status': 'capturing'}]}))
            self.assertEqual(bench.main(['--worker-binding', str(out/'binding.json'), '--output', str(out)]), 1)
            self.assertEqual(worker.call_count, 1)
            result = json.loads((out/'result.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['memory_accounting'], error.memory_accounting)
            self.assertEqual(json.loads((out/'failed-graph-pool-snapshot.json').read_text()), error.pool_snapshot)
            self.assertEqual(json.loads((out/'capture-progress.json').read_text())['captures'][0]['status'], 'validated')


class R2RealFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def factory(self, capture=True):
        with patch.object(old.bench, 'PairedFactory', bench.PairedFactory):
            make, device = old.PairedR1RealFactoryTests.factory(self, capture=False)
        if capture:
            setup = make.capture(MANIFEST)
            self.assertEqual((setup['buckets'], setup['graph_count']), (25, 2))
            for row in setup['captures']:
                self.assertEqual(row['status'], 'validated')
                self.assertEqual([x['context_lengths'] for x in row['validations']], [[256, 256], [368, 375]])
                self.assertTrue(all(v['status'] == 'passed' for v in row['validations']))
        return make, device

    def test_real_three_complete128_arms_shadow_denominators_resets_and_tails(self):
        make, device = self.factory(); case = bench.selected_manifest(MANIFEST)['cases'][0]
        results = {arm: bench.batch(make, device, case, MANIFEST, arm) for arm in bench.ARMS}
        self.assertEqual(len(make.constructed), 1)
        for arm, row in results.items():
            self.assertEqual(row['output_tokens'], 256); self.assertEqual(row['admission_output_tokens'], 2)
            self.assertEqual(row['committed_round_output_tokens'], 254)
            self.assertEqual(row['final_context_lengths'], {'r0': 383, 'r1': 383})
            self.assertEqual(row['full_shadow_positions'], 0 if arm == 'target_only' else 7*row['request_rounds'])
            self.assertEqual(row['execution_coverage']['actual_gpu_graph_rounds'], 0)
            for r in row['rounds']:
                self.assertEqual(r['actual_physical_b'], r['actual_logical_b'])
                if len(r['active_requests']) == 1:
                    self.assertTrue(r['eager_tail']); self.assertEqual(r['execution_kind'], 'explicit_eager')
        zero = results['full_shadow_zero_admission']
        self.assertEqual(zero['selected_proposal_tokens'], 0); self.assertEqual(zero['accepted_draft_tokens'], 0)
        self.assertEqual(zero['full_shadow_positions'], 127*14); self.assertIsNone(zero['accepted_over_selected'])
        self.assertEqual(zero['rounds'][-1]['work']['proposal_batches'][0]['proposal_positions'], 14)
        self.assertGreater(results['fixed_gamma7_full_shadow']['r1_tail_rounds'], 0)
        repeat = bench.batch(make, device, case, MANIFEST, 'full_shadow_zero_admission', diagnostic=True)
        self.assertEqual([x['output_sha256'] for x in zero['requests']], [x['output_sha256'] for x in repeat['requests']])
        self.assertIsNone(make.target._pending); self.assertIsNone(make.target._feature_lease)

    def test_grown_setup_failure_preserves_raw_tensors_and_completed_capture(self):
        import torch
        make, _ = self.factory(capture=False); original = make.target.verify; progress = []
        def corrupt(*args, **kwargs):
            f = original(*args, **kwargs)
            if kwargs['bucket'].query_tokens == 16 and max(make.target.lengths.values()) > 300: f.last.add_(1.)
            return f
        with tempfile.TemporaryDirectory() as tmp, patch.object(make.target, 'verify', side_effect=corrupt):
            with self.assertRaisesRegex(AssertionError, 'fixed numerical'):
                make.capture(MANIFEST, evidence_dir=Path(tmp), progress=lambda x: progress.append(copy.deepcopy(x)))
            self.assertEqual(progress[-1][0]['status'], 'validated')
            failed = progress[-1][1]; self.assertEqual(failed['status'], 'validation_failed')
            validation = failed['validations'][-1]
            raw = torch.load(Path(tmp)/'grown'/validation['tensors']['file'], weights_only=True)
            self.assertIn('eager_scratch_keys', raw); self.assertIn('replay_scratch_values', raw)
            self.assertFalse(validation['comparisons']['r1/final_norm']['passed'])
            self.assertFalse(make.ready)

    def test_deadline_and_explicit_r1_tail_real_composition(self):
        import torch
        from dspark_qwen.packed_sampling import RequestSpec
        from dspark_qwen.tensor_sampling import TensorRandom
        make, device = self.factory(); session = make('fixed_gamma7_full_shadow')
        case = bench.selected_manifest(MANIFEST)['cases'][0]
        session.admit({r['request']: RequestSpec(torch.tensor([r['prompt_token_ids']]), 1 if i == 0 else 3,
            TensorRandom(torch.Generator().manual_seed(r['seed']))) for i, r in enumerate(case['requests'])})
        issued = session.propose(['r1'], mode='shadow'); allocation = {'r1': 2}
        before = make.witness.snapshot()
        result = session.verify_commit(issued.proposals, allocation, allocation_policy=bench.BASE.NATIVE.FixedPrefix(session, allocation))
        record = bench.execution_record(result['work']['target'], 3, before, make.witness.snapshot(), device.type, active_count=1)
        self.assertEqual(record['execution_kind'], 'explicit_eager'); self.assertTrue(record['eager_tail'])
        with self.assertRaisesRegex(TimeoutError, 'incomplete batch'):
            bench.batch(make, device, case, MANIFEST, 'target_only', deadline=0.)


if __name__ == '__main__': unittest.main()
