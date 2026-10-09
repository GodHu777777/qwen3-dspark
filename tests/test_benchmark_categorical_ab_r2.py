import builtins
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_benchmark_paired_r1 import load, fixture as binding_fixture

ROOT = Path(__file__).resolve().parents[1]
ab = load('test_categorical_ab_runner', ROOT/'scripts/benchmark_categorical_ab_r2.py')
fixture = load('categorical_ab_factory_fixture', ROOT/'tests/test_benchmark_paired_r2.py')
MANIFEST = fixture.MANIFEST


class CategoricalRunnerProtocolTests(unittest.TestCase):
    def test_stdlib_binding_original_domain_exact_schedule_and_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, generation, checkpoint, hashes = binding_fixture.PackedGateTests().binding_fixture(Path(tmp))
            manifest = copy.deepcopy(MANIFEST); manifest['model']['config_sha256'] = ab.GUARD.sha(model/'config.json')
            path = Path(tmp)/'workloads.json'; path.write_text(json.dumps(manifest)); original = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in ('torch', 'transformers', 'dspark_qwen'): raise AssertionError('Heavy dry-run import')
                return original(name, *args, **kwargs)
            with patch.dict(ab.ENGINE.PROFILE.GATE.PROTOCOL, hashes), patch.object(ab.BASE, 'baseline_reference', return_value={}), \
                    patch('builtins.__import__', side_effect=guarded), patch('sys.stdout', new_callable=io.StringIO) as output:
                self.assertEqual(ab.main(['--dry-run', '--model', str(model), '--generation-manifest', str(generation),
                    '--checkpoint', str(checkpoint), '--workloads', str(path)]), 0)
            value = json.loads(output.getvalue())['binding']
            self.assertEqual(value['worker_entry'], 'scripts/benchmark_categorical_ab_r2.py')
            self.assertEqual(len(value['expected_samples']), 20)
            for name in ab.SOURCES: self.assertEqual(value['source_sha256'][name], ab.GUARD.sha(ROOT/name))
            self.assertEqual(value['protocol']['timeout_seconds'], 300)
            self.assertEqual(value['protocol']['cooperative_reserve_seconds'], 10)
            self.assertEqual(value['protocol']['graph_byte_budget'], 1280*1024**2)
            self.assertFalse(value['protocol']['observer']); self.assertFalse(value['protocol']['signature_wrappers'])
        rows = ab.identities(MANIFEST)
        self.assertEqual([(r['arm'], r['variant']) for r in rows[:4]], [(a, v) for a in ab.ARMS for v in ab.VARIANTS])
        for arm in ab.ARMS:
            for repeat, order in enumerate(('AB', 'BA', 'BA', 'AB')):
                self.assertEqual(''.join(r['variant'] for r in rows if r['phase'] == 'primary' and r['arm'] == arm and r['repeat'] == repeat), order)
        altered = copy.deepcopy(MANIFEST); altered['output_tokens'] = 127
        with self.assertRaises(ValueError): ab.identities(altered)
        with patch.object(ab.ENGINE, 'worker', return_value=0) as worker:
            self.assertEqual(ab.worker('binding', 'out'), 0)
            self.assertEqual(worker.call_args.kwargs['experiment'].PROTOCOL, ab.PROTOCOL)

    @staticmethod
    def samples():
        return [dict(row, output_tokens=256, batch_wall_seconds=1. if row['variant'] == 'A' else .98,
            sampler_gate_passed=True, sampler_gate_sha256='cpu-gate', sampler_gate_scope='cpu_emulator_not_native',
            variant_aliases_restored=True, semantic=dict(arm=row['arm'])) for row in ab.identities(MANIFEST)]

    def test_screening_exact_threshold_pooled_denominator_both_arms_and_three_pairs(self):
        with patch.object(ab.COARSE, 'invariant', side_effect=lambda x: x['semantic']):
            rows = self.samples(); result = ab.verify_complete(rows, MANIFEST)
            self.assertTrue(result['screening_passed'])
            for arm in result['arms'].values():
                self.assertEqual(arm['pooled_output_tokens_per_variant'], 1024)
                self.assertEqual(arm['pooled_output_tokens_per_second']['A'], 256.)
                self.assertEqual(arm['b_faster_pairs'], 4)
            # Strong pooled improvement cannot replace the required3/4 wins.
            for row in rows:
                if row['phase'] == 'primary' and row['variant'] == 'B' and row['arm'] == ab.ARMS[1]:
                    row['batch_wall_seconds'] = .5 if row['repeat'] < 2 else 1.01
            self.assertFalse(ab.verify_complete(rows, MANIFEST)['screening_passed'])
            rows = self.samples()
            for row in rows:
                if row['phase'] == 'primary' and row['variant'] == 'B': row['batch_wall_seconds'] = 1./1.02
            self.assertTrue(ab.verify_complete(rows, MANIFEST)['screening_passed'])
            for row in rows:
                if row['phase'] == 'primary' and row['variant'] == 'B': row['batch_wall_seconds'] = 1./1.019999
            self.assertFalse(ab.verify_complete(rows, MANIFEST)['screening_passed'])
            for invalid in (float('nan'), float('inf'), 0., -1.):
                rows = self.samples(); rows[-1]['batch_wall_seconds'] = invalid
                with self.assertRaises(ValueError): ab.verify_complete(rows, MANIFEST)
            rows = self.samples(); rows[-1]['semantic']['rng'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'raw tokens/RNG/work'): ab.verify_complete(rows, MANIFEST)
            with self.assertRaisesRegex(ValueError, 'Exactly20'): ab.verify_complete(self.samples()[:-1], MANIFEST)

    def test_final_semantic_failure_retains_all_completed_rows_and_failed_invariance(self):
        import torch
        sequence = self.samples(); sequence[-1]['semantic']['rng'] = 'drift'
        def measured(*args, **kwargs): return copy.deepcopy(sequence.pop(0))
        with tempfile.TemporaryDirectory() as tmp, patch.object(ab, 'measure', side_effect=measured), \
                patch.object(ab, 'categorical_gate', return_value=dict(passed=True, scope='cpu_emulator_not_native')), \
                patch.object(ab.COARSE, 'invariant', side_effect=lambda x: x['semantic']):
            out = Path(tmp)/'run'
            with self.assertRaisesRegex(ValueError, 'raw tokens/RNG/work'): ab.run(None, torch.device('cpu'), MANIFEST, out)
            self.assertEqual(len((out/'samples.jsonl').read_text().splitlines()), 20)
            result = json.loads((out/'result.json').read_text()); self.assertEqual(result['sample_count'], 20)
            self.assertEqual(result['status'], 'failed'); self.assertNotIn('aggregates', result)
            self.assertFalse(json.loads((out/'invariance.json').read_text())['passed'])


class CategoricalRunnerCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def factory(self): return fixture.R2RealFactoryTests.factory(self)

    def test_real_twenty_batches_direct_aliases_no_observer_exact_semantics_and_gate(self):
        from dspark_qwen import tensor_sampling, packed_sampling, packed_target_sampling
        from dspark_qwen.experimental_categorical import ORIGINAL_SAMPLE_CATEGORICAL, sample_categorical_fixed_shape
        make, device = self.factory(); batch = ab.BASE.batch; selected = []
        def inspected(*args, **kwargs):
            aliases = [m.sample_categorical for m in (tensor_sampling, packed_sampling, packed_target_sampling)]
            self.assertTrue(all(f is aliases[0] for f in aliases)); self.assertIn(aliases[0], (ORIGINAL_SAMPLE_CATEGORICAL, sample_categorical_fixed_shape))
            selected.append('A' if aliases[0] is ORIGINAL_SAMPLE_CATEGORICAL else 'B'); return batch(*args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(ab.BASE, 'batch', side_effect=inspected), \
                patch.object(ab.COARSE.CoarseObserver, '__enter__', side_effect=AssertionError('No observer')):
            out = Path(tmp)/'run'; result = ab.run(make, device, MANIFEST, out)
            rows = [json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
            self.assertEqual(result['status'], 'completed'); self.assertEqual(selected, [x[3] for x in ab.PLAN])
            self.assertEqual(result['aggregates'], ab.verify_complete(rows, MANIFEST))
            self.assertEqual(result['aggregates']['device_evidence_scope'], 'cpu_emulator_not_native')
            gate = json.loads((out/'categorical-gate/result.json').read_text())
            self.assertEqual(gate['completed_case_count'], 15); self.assertTrue(gate['passed'])
            self.assertTrue(all(x['exact_equal'] and x['expected_contract'] for x in gate['cases']))
            self.assertTrue(all(x['exact_probabilities_equal'] for x in gate['cases']))
            self.assertEqual(gate['cases'][-1]['original']['probability_stride'], [2])
            self.assertEqual(gate['cases'][-1]['original']['token'], 151934)
            import torch
            for gate_case in gate['cases'][-2:]:
                artifact = gate_case['raw_probabilities']; raw_path = out/'categorical-gate'/artifact['file']
                self.assertEqual(ab.GUARD.sha(raw_path), artifact['sha256'])
                raw = torch.load(raw_path, weights_only=True); self.assertTrue(torch.equal(raw['original'], raw['candidate']))
                self.assertEqual(raw['original'].numel(), 151936)
            self.assertEqual(result['categorical_gate_sha256'], ab.ENGINE.PROFILE.digest(gate))
            self.assertEqual(set(gate['sources']), {'dspark_qwen/tensor_sampling.py', 'dspark_qwen/experimental_categorical.py'})
            self.assertTrue(all(x['variant_aliases_restored'] for x in rows))
            self.assertTrue(all('coarse_observation' not in x and 'inner_observation' not in x for x in rows))
            self.assertGreater(rows[2]['r1_tail_rounds'], 0)
            self.assertTrue(all(not list(d.glob('*observation*')) for d in (out/'batches').iterdir()))
        for module in (tensor_sampling, packed_sampling, packed_target_sampling): self.assertIs(module.sample_categorical, ORIGINAL_SAMPLE_CATEGORICAL)

    def test_gate_failure_and_deadline_abort_before_any_batch(self):
        import torch
        from dspark_qwen import experimental_categorical
        with tempfile.TemporaryDirectory() as tmp, patch.object(ab, 'measure') as measured, \
                patch.object(experimental_categorical, 'sample_categorical_fixed_shape', return_value=torch.tensor(3)):
            out = Path(tmp)/'failure'
            with self.assertRaisesRegex(ValueError, 'oracle gate failed'): ab.run(None, torch.device('cpu'), MANIFEST, out)
            measured.assert_not_called(); self.assertFalse((out/'samples.jsonl').exists())
            report = json.loads((out/'categorical-gate/result.json').read_text())
            self.assertEqual(report['status'], 'failed'); self.assertEqual(len(report['cases']), 1)
            self.assertFalse(report['cases'][0]['exact_equal'])
        with tempfile.TemporaryDirectory() as tmp, patch.object(ab, 'measure') as measured:
            out = Path(tmp)/'timeout'
            with self.assertRaises(TimeoutError): ab.run(None, torch.device('cpu'), MANIFEST, out, deadline=0.)
            measured.assert_not_called(); self.assertEqual(json.loads((out/'result.json').read_text())['status'], 'partial_deadline')

    def test_variant_restores_all_aliases_when_batch_raises(self):
        from dspark_qwen import tensor_sampling, packed_sampling, packed_target_sampling
        from dspark_qwen.experimental_categorical import ORIGINAL_SAMPLE_CATEGORICAL, sample_categorical_fixed_shape
        make, device = self.factory()
        def fail(*args, **kwargs):
            for module in (tensor_sampling, packed_sampling, packed_target_sampling): self.assertIs(module.sample_categorical, sample_categorical_fixed_shape)
            raise RuntimeError('injected batch failure')
        with tempfile.TemporaryDirectory() as tmp, patch.object(ab.BASE, 'batch', side_effect=fail):
            out = Path(tmp)/'batch'
            with self.assertRaisesRegex(RuntimeError, 'injected batch failure'):
                ab.measure(make, device, ab.validate_domain(MANIFEST), MANIFEST, ab.ARMS[0], 'B', out)
            status = json.loads((out/'status.json').read_text()); self.assertEqual(status['status'], 'failed')
            self.assertTrue(status['variant_aliases_restored']); self.assertFalse(status['complete_batch'])
        for module in (tensor_sampling, packed_sampling, packed_target_sampling): self.assertIs(module.sample_categorical, ORIGINAL_SAMPLE_CATEGORICAL)


if __name__ == '__main__': unittest.main()
