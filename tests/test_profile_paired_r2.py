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
profile = load('test_coarse_r2', ROOT/'scripts/profile_paired_r2.py')
fixture = load('test_coarse_r2_factory', ROOT/'tests/test_benchmark_paired_r2.py')
MANIFEST = fixture.MANIFEST


class CoarseProtocolTests(unittest.TestCase):
    def test_stdlib_binding_and_real_worker_dispatch_use300_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, generation, checkpoint, hashes = binding_fixture.PackedGateTests().binding_fixture(Path(tmp))
            manifest = copy.deepcopy(MANIFEST); manifest['model']['config_sha256'] = profile.GUARD.sha(model/'config.json')
            path = Path(tmp)/'workloads.json'; path.write_text(json.dumps(manifest))
            original = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in ('torch', 'transformers', 'dspark_qwen'): raise AssertionError('Heavy dry binding')
                return original(name, *args, **kwargs)
            with patch.dict(profile.BASE.BASE.PROFILE.GATE.PROTOCOL, hashes), \
                    patch.object(profile.BASE, 'baseline_reference', return_value={}), \
                    patch('builtins.__import__', side_effect=guarded), patch('sys.stdout', new_callable=io.StringIO) as output:
                self.assertEqual(profile.main(['--dry-run', '--model', str(model), '--generation-manifest', str(generation),
                    '--checkpoint', str(checkpoint), '--workloads', str(path)]), 0)
            binding = json.loads(output.getvalue())['binding']
            self.assertEqual(binding['protocol']['timeout_seconds'], 300)
            self.assertEqual(binding['protocol']['cooperative_reserve_seconds'], 10)
            self.assertEqual(len(binding['expected_samples']), 6)
            self.assertEqual(binding['worker_entry'], 'scripts/profile_paired_r2.py')
            self.assertFalse(binding['protocol']['kineto'])
            self.assertEqual(profile.BASE.PROTOCOL['timeout_seconds'], 1800)
            self.assertEqual(profile.BASE.BASE.PROTOCOL['timeout_seconds'], 1800)
        with patch.object(profile.BASE.BASE, 'worker', return_value=0) as worker:
            self.assertEqual(profile.worker('binding', 'out'), 0)
            self.assertEqual(worker.call_args.kwargs['experiment'].PROTOCOL['timeout_seconds'], 300)
            self.assertIs(worker.call_args.kwargs['experiment'].PairedFactory, profile.BASE.PairedFactory)

    def test_nested_outer_calls_fence_exactly_twice_and_keep_prior_drain_separate(self):
        calls = []; observer = profile.CoarseObserver(None, synchronize=lambda: calls.append('sync'))
        observer.origin = profile.time.perf_counter()
        def inside():
            calls.append('outer_body')
            return observer.call('nested', 'nested_call', lambda: calls.append('nested_body'))
        observer.call('full_shadow_propose', 'propose', inside)
        self.assertEqual(calls, ['sync', 'outer_body', 'nested_body', 'sync'])
        self.assertEqual(len(observer.stages), 1)
        observer.closed = True
        summary = observer.summary(1.)
        self.assertGreaterEqual(summary['prior_unassigned_boundary_drain_seconds'], 0.)
        self.assertEqual(summary['complete_batch_wall_seconds'], summary['prior_unassigned_boundary_drain_seconds']+
            summary['synchronized_service_seconds']+summary['unassigned_batch_remainder_seconds'])


class CoarseCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def factory(self): return fixture.R2RealFactoryTests.factory(self)

    @staticmethod
    def kv(session):
        result = {}
        for owner in ('target', 'draft'):
            if not hasattr(session, owner): continue
            result[owner] = {r: [(k.clone(), v.clone()) for k, v in getattr(session, owner).request_kv(r)] for r in session.requests}
        return result

    def test_real_six_complete_batches_semantics_rng_kv_work_and_exclusive_coverage(self):
        import torch
        make, device = self.factory(); snapshots = []; sessions = []
        real_factory = profile.SessionFactory; real_state = profile.semantic_state
        class CapturedFactory(real_factory):
            def __call__(self, arm):
                session = super().__call__(arm); sessions.append(session); return session
        def recorded_state(session):
            snapshots.append(self.kv(session)); return real_state(session)
        with tempfile.TemporaryDirectory() as tmp, patch.object(profile, 'SessionFactory', CapturedFactory), \
                patch.object(profile, 'semantic_state', side_effect=recorded_state):
            result = profile.run(make, device, MANIFEST, Path(tmp)/'run')
            self.assertEqual(result['status'], 'completed'); self.assertEqual(result['sample_count'], 6)
            rows = [json.loads(x) for x in (Path(tmp)/'run/samples.jsonl').read_text().splitlines()]
            self.assertEqual(result['aggregates'], profile.verify_complete(rows, MANIFEST))
            for left, right in ((2, 3), (5, 4)):
                self.assertEqual(profile.invariant(rows[left]), profile.invariant(rows[right]))
                for owner in snapshots[left]:
                    for request in snapshots[left][owner]:
                        for a, b in zip(snapshots[left][owner][request], snapshots[right][owner][request]):
                            for x, y in zip(a, b): self.assertTrue(torch.equal(x, y))
            for index in (3, 4):
                row = rows[index]; obs = row['coarse_observation']
                self.assertEqual(len(obs['rounds']), len(row['rounds']))
                self.assertEqual(obs['signature_calls'], 3*row['execution_coverage']['graph_plan_rounds'])
                self.assertTrue(all(x['stage'] is not None for x in obs['signatures']))
                self.assertTrue(all(obs['stages'][x['stage']]['category'] == 'target_service' for x in obs['signatures']))
                self.assertGreater(obs['unassigned_batch_remainder_seconds'], 0.)
                self.assertGreater(obs['unassigned_session_remainder_seconds'], 0.)
                self.assertTrue(all(x['unassigned_round_remainder_seconds'] >= 0 for x in obs['rounds']))
                self.assertEqual(sum(x['calls'] for x in obs['categories'].values()), len(obs['stages']))
                if row['arm'] == 'fixed_gamma7_full_shadow':
                    self.assertEqual(obs['categories']['full_shadow_propose']['calls'], len(row['rounds']))
                    self.assertGreater(row['r1_tail_rounds'], 0)
                else: self.assertNotIn('full_shadow_propose', obs['categories'])
                for stage in obs['stages']:
                    self.assertIn(stage['callsite'], ('admission', 'round'))
                    self.assertGreaterEqual(stage['synchronized_service_seconds'], stage['body_host_seconds'])
                    if stage['callable'] == 'admission_head': self.assertEqual(stage['callsite'], 'admission')
            self.assertTrue(all(len(s.requests['r0']['output']) == 128 for s in sessions))
            self.assertEqual(len(make.constructed), 1)

    def test_rng_only_corruption_is_rejected_after_both_completed_rows_are_saved(self):
        for corrupted_count, arm, plain_index, coarse_index in ((4, 'target_only', 2, 3),
                (6, 'fixed_gamma7_full_shadow', 5, 4)):
            with self.subTest(arm=arm):
                make, device = self.factory(); real_state = profile.semantic_state; count = 0
                def corrupt(session):
                    nonlocal count
                    value = real_state(session); count += 1
                    if count == corrupted_count:
                        import hashlib
                        row = value['r0']; row['rng_state_bytes'][0] ^= 1
                        row['rng_state_sha256'] = hashlib.sha256(bytes(row['rng_state_bytes'])).hexdigest()
                    return value
                with tempfile.TemporaryDirectory() as tmp, patch.object(profile, 'semantic_state', side_effect=corrupt):
                    out = Path(tmp)/'run'
                    with self.assertRaisesRegex(ValueError, 'raw output/RNG/work'): profile.run(make, device, MANIFEST, out)
                    rows = [json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
                    self.assertEqual(len(rows), corrupted_count)
                    self.assertEqual(rows[plain_index]['requests'][0]['output_sha256'], rows[coarse_index]['requests'][0]['output_sha256'])
                    self.assertEqual(json.loads((out/'result.json').read_text())['sample_count'], corrupted_count)
                    failure = next(x for x in json.loads((out/'invariance.json').read_text())['comparisons'] if x['arm'] == arm)
                    self.assertEqual(failure['differing_fields'], ['semantic_state'])
                    self.assertTrue((out/f'batches/{coarse_index}-coarse_sync-{arm}/completed-batch.json').exists())
                    self.assertTrue((out/f'batches/{plain_index}-plain-{arm}/completed-batch.json').exists())

    def test_body_failure_restores_wrappers_and_retains_partial_stages(self):
        from dspark_qwen.packed_target_sampling import PackedTargetOnlySession
        from dspark_qwen.persistent_qwen_target import PersistentQwenTarget
        make, device = self.factory(); originals = (PackedTargetOnlySession.step, PersistentQwenTarget.predict)
        def fail(*args, **kwargs): raise RuntimeError('injected target head failure')
        case = profile.BASE.selected_manifest(MANIFEST)['cases'][0]
        with tempfile.TemporaryDirectory() as tmp, patch.object(PersistentQwenTarget, 'predict', side_effect=fail) as replaced:
            with self.assertRaisesRegex(RuntimeError, 'target head failure'):
                profile.measure(make, device, case, MANIFEST, 'target_only', 'coarse_sync', Path(tmp)/'batch')
            self.assertIs(PackedTargetOnlySession.step, originals[0]); self.assertIs(PersistentQwenTarget.predict, replaced)
            partial = json.loads((Path(tmp)/'batch/partial-observation.json').read_text())
            self.assertTrue(partial['observer_restored'])
            self.assertTrue(any(x['status'] == 'failed' for x in partial['stages']))
            self.assertEqual(partial['rounds'][0]['status'], 'failed')
        self.assertIs(PersistentQwenTarget.predict, originals[1])

    def test_cooperative_deadline_and_stage_limit_keep_failures_without_retry(self):
        make, device = self.factory(); case = profile.BASE.selected_manifest(MANIFEST)['cases'][0]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TimeoutError):
                profile.measure(make, device, case, MANIFEST, 'target_only', 'coarse_sync', Path(tmp)/'deadline', deadline=0.)
            self.assertTrue(json.loads((Path(tmp)/'deadline/partial-observation.json').read_text())['observer_restored'])
            with patch.dict(profile.PROTOCOL, max_stage_records=1):
                with self.assertRaisesRegex(RuntimeError, 'record limit'):
                    profile.measure(make, device, case, MANIFEST, 'target_only', 'coarse_sync', Path(tmp)/'limit')
            partial = json.loads((Path(tmp)/'limit/partial-observation.json').read_text())
            self.assertTrue(partial['observer_restored']); self.assertEqual(len(partial['stages']), 1)


if __name__ == '__main__': unittest.main()
