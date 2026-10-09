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
inner = load('test_shadow_inner', ROOT/'scripts/profile_shadow_inner_r2.py')
fixture = load('inner_r2_factory_fixture', ROOT/'tests/test_benchmark_paired_r2.py')
MANIFEST = fixture.MANIFEST


class InnerProtocolTests(unittest.TestCase):
    def test_stdlib_binding_preserves_original_domain_and_binds_new_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, generation, checkpoint, hashes = binding_fixture.PackedGateTests().binding_fixture(Path(tmp))
            manifest = copy.deepcopy(MANIFEST); manifest['model']['config_sha256'] = inner.GUARD.sha(model/'config.json')
            path = Path(tmp)/'workloads.json'; path.write_text(json.dumps(manifest))
            original = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in ('torch', 'transformers', 'dspark_qwen'): raise AssertionError('Heavy binding import')
                return original(name, *args, **kwargs)
            with patch.dict(inner.ENGINE.PROFILE.GATE.PROTOCOL, hashes), patch.object(inner.PAIRED, 'baseline_reference', return_value={}), \
                    patch('builtins.__import__', side_effect=guarded), patch('sys.stdout', new_callable=io.StringIO) as output:
                self.assertEqual(inner.main(['--dry-run', '--model', str(model), '--generation-manifest', str(generation),
                    '--checkpoint', str(checkpoint), '--workloads', str(path)]), 0)
            value = json.loads(output.getvalue())['binding']
            self.assertEqual(value['worker_entry'], 'scripts/profile_shadow_inner_r2.py')
            self.assertEqual(len(value['expected_samples']), 6)
            self.assertEqual(value['protocol']['timeout_seconds'], 300)
            self.assertEqual(value['protocol']['cooperative_reserve_seconds'], 10)
            self.assertEqual(value['protocol']['max_rounds_per_batch'], 127)
            self.assertEqual(inner.COARSE.PROTOCOL['max_rounds_per_batch'], 254)
            self.assertEqual(inner.PAIRED.PROTOCOL['timeout_seconds'], 1800)
        for field, replacement in [('output_tokens', 129), ('sampling', dict(MANIFEST['sampling'], temperature=.9))]:
            altered = copy.deepcopy(MANIFEST); altered[field] = replacement
            with self.assertRaises(ValueError): inner.validate_domain(altered)
        with patch.object(inner.ENGINE, 'worker', return_value=0) as worker:
            inner.worker('binding', 'out')
            self.assertEqual(worker.call_args.kwargs['experiment'].PROTOCOL['max_inner_records'], 8192)

    def test_predeclared_classification_rejects_nan_and_distinguishes_order_disagreement(self):
        pairs = [dict(probability_fraction=.5, observed_over_plain=.95), dict(probability_fraction=.6, observed_over_plain=1.05)]
        self.assertEqual(inner.classify(pairs)['classification'], 'supported')
        low = [dict(x, probability_fraction=.49) for x in pairs]
        self.assertEqual(inner.classify(low)['classification'], 'not_supported')
        mixed = [low[0], pairs[1]]
        self.assertEqual(inner.classify(mixed)['reason'], 'threshold_disagreement_across_reversed_orders')
        disturbed = [dict(low[0], observed_over_plain=1.050001), low[1]]
        self.assertEqual(inner.classify(disturbed)['reason'], 'observed_plain_disturbance_outside_predeclared_5pct')
        for invalid in (float('nan'), float('inf'), -1., 1.1):
            with self.assertRaises(ValueError): inner.classify([dict(pairs[0], probability_fraction=invalid), pairs[1]])


class InnerCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def factory(self): return fixture.R2RealFactoryTests.factory(self)

    @staticmethod
    def kv(session):
        return {owner: {r: [(k.clone(), v.clone()) for k, v in getattr(session, owner).request_kv(r)]
            for r in session.requests} for owner in ('target', 'draft')}

    def test_six_real_batches_full_proposal_q_confidence_and_each_committed_kv_match(self):
        import torch
        from dspark_qwen.packed_sampling import PackedSpeculativeSession
        make, device = self.factory(); streams = []; sessions = {}
        original_factory = inner.COARSE.SessionFactory; propose = PackedSpeculativeSession.propose; verify = PackedSpeculativeSession.verify_commit
        class CapturingFactory(original_factory):
            def __call__(self, arm):
                session = super().__call__(arm); index = len(streams)
                streams.append(dict(proposals=[], kv=[])); sessions[id(session)] = index
                return session
        def proposal(session, *args, **kwargs):
            value = propose(session, *args, **kwargs)
            streams[sessions[id(session)]]['proposals'].append({r: tuple(t.clone() for t in
                (p.proposal.tokens, p.proposal.draft_probs, p.proposal.confidence_logits)) for r, p in value.proposals.items()})
            return value
        def committed(session, *args, **kwargs):
            value = verify(session, *args, **kwargs)
            streams[sessions[id(session)]]['kv'].append(self.kv(session))
            return value
        with tempfile.TemporaryDirectory() as tmp, patch.object(inner.COARSE, 'SessionFactory', CapturingFactory), \
                patch.object(PackedSpeculativeSession, 'propose', proposal), patch.object(PackedSpeculativeSession, 'verify_commit', committed):
            out = Path(tmp)/'run'; result = inner.run(make, device, MANIFEST, out)
            self.assertEqual(result['status'], 'completed')
            rows = [json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
            self.assertEqual(result['aggregates'], inner.verify_complete(rows, MANIFEST))
            for index in (3, 4, 5):
                self.assertEqual(inner.COARSE.invariant(rows[2]), inner.COARSE.invariant(rows[index]))
                self.assertEqual(len(streams[2]['proposals']), len(streams[index]['proposals']))
                for a, b in zip(streams[2]['proposals'], streams[index]['proposals']):
                    self.assertEqual(list(a), list(b))
                    for r in a:
                        for x, y in zip(a[r], b[r]): self.assertTrue(torch.equal(x, y))
                self.assertEqual(len(streams[2]['kv']), len(streams[index]['kv']))
                for a, b in zip(streams[2]['kv'], streams[index]['kv']):
                    self.assertEqual(a.keys(), b.keys())
                    for owner in a:
                        self.assertEqual(a[owner].keys(), b[owner].keys())
                        for r in a[owner]:
                            self.assertEqual(len(a[owner][r]), len(b[owner][r]))
                            for x, y in zip(a[owner][r], b[owner][r]):
                                for left, right in zip(x, y): self.assertTrue(torch.equal(left, right))
            self.assertTrue(streams[2]['kv'][0]['target']['r0'][0][0].numel() > 0)
            self.assertGreater(rows[2]['r1_tail_rounds'], 0)
            for index in (3, 4):
                row = rows[index]; obs = row['inner_observation']; inner.validate_inner_sample(row)
                self.assertEqual(obs['signature_calls'], 3*row['execution_coverage']['graph_plan_rounds'])
                self.assertEqual(len(obs['inner_stages']), sum(30+8*len(r['active_requests']) for r in row['rounds']))
                self.assertTrue(all(x['callsite'] == 'round' for x in obs['inner_stages']))
                self.assertTrue(all(x['category'] != 'full_shadow_propose' for x in obs['inner_stages']))
                self.assertTrue(all(x['parent_residual_seconds'] >= 0 for x in obs['inner_parent_partitions']))
                self.assertEqual(obs['inner_categories']['base_head']['calls'], len(row['rounds']))
                self.assertEqual(obs['inner_categories']['fp64_law']['calls'], 7*len(row['rounds']))
                self.assertEqual(obs['inner_categories']['proposal_copy']['calls'], row['request_rounds'])
                for stage in obs['signatures']: self.assertEqual(obs['stages'][stage['stage']]['category'], 'target_service')
            broken = copy.deepcopy(rows[3]); broken['inner_observation']['inner_stages'].pop()
            with self.assertRaisesRegex(ValueError, 'coverage'): inner.validate_inner_sample(broken)

    def test_inner_failure_restores_aliases_and_leaves_clean_original_cleanup(self):
        from dspark_qwen import packed_sampling
        make, device = self.factory(); original = packed_sampling._copy_proposal
        def fail(*args, **kwargs): raise RuntimeError('injected proposal copy failure')
        case = inner.validate_domain(MANIFEST)
        with tempfile.TemporaryDirectory() as tmp, patch.object(packed_sampling, '_copy_proposal', side_effect=fail) as replaced:
            with self.assertRaisesRegex(RuntimeError, 'proposal copy failure'):
                inner.measure(make, device, case, MANIFEST, inner.ARM, 'inner_observed', Path(tmp)/'batch')
            self.assertIs(packed_sampling._copy_proposal, replaced)
            partial = json.loads((Path(tmp)/'batch/partial-inner-observation.json').read_text())
            self.assertTrue(partial['observer_restored'])
            self.assertTrue(any(x['status'] == 'failed' and x['category'] == 'proposal_copy' for x in partial['inner_stages']))
            self.assertEqual(make.target.lengths, {}); self.assertIsNone(make.target._pending); self.assertIsNone(make.target._feature_lease)
        self.assertIs(packed_sampling._copy_proposal, original)

    def test_post_batch_coverage_failure_preserves_completion_and_marks_failed(self):
        make, device = self.factory(); case = inner.validate_domain(MANIFEST)
        with tempfile.TemporaryDirectory() as tmp, patch.object(inner, 'validate_inner_sample', side_effect=ValueError('injected coverage failure')):
            directory = Path(tmp)/'batch'
            with self.assertRaisesRegex(ValueError, 'injected coverage failure'):
                inner.measure(make, device, case, MANIFEST, inner.ARM, 'inner_observed', directory)
            status = json.loads((directory/'status.json').read_text())
            self.assertEqual(status['status'], 'failed'); self.assertTrue(status['complete_batch'])
            self.assertEqual(status['phase'], 'inner_observed')
            self.assertEqual(status['error_type'], 'ValueError')
            for filename in ('completed-batch.json', 'observation.json', 'partial-observation.json', 'partial-inner-observation.json'):
                self.assertTrue((directory/filename).exists())
            self.assertTrue(json.loads((directory/'partial-inner-observation.json').read_text())['observer_restored'])
            completed = json.loads((directory/'completed-batch.json').read_text())
            self.assertEqual(make.target.lengths, completed['final_context_lengths'])
            self.assertIsNone(make.target._pending); self.assertIsNone(make.target._feature_lease)

    def test_actual_round_guard_and_inner_limit_fail_before_expansion(self):
        from dspark_qwen.packed_sampling import RequestSpec
        from dspark_qwen.tensor_sampling import TensorRandom
        import torch
        make, device = self.factory(); session = make(inner.ARM); case = inner.validate_domain(MANIFEST)
        session.admit({r['request']: RequestSpec(torch.tensor([r['prompt_token_ids']]),128,
            TensorRandom(torch.Generator().manual_seed(r['seed']))) for r in case['requests']})
        observer = inner.InnerObserver(make); observer.rounds = [{}]*127
        before = {r: s['rng'].generator.get_state().clone() for r, s in session.requests.items()}
        with self.assertRaisesRegex(ValueError, '<=127'): observer.guard_propose(session, ['r0', 'r1'])
        for r, state in session.requests.items(): self.assertTrue(torch.equal(before[r], state['rng'].generator.get_state()))
        observer.rounds = []
        with self.assertRaises(ValueError): observer.guard_propose(session, ['r1', 'r0'])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(inner.PROTOCOL, max_inner_records=1):
            with self.assertRaisesRegex(RuntimeError, 'Inner child record limit'):
                inner.measure(make, device, case, MANIFEST, inner.ARM, 'inner_observed', Path(tmp)/'limit')
            partial = json.loads((Path(tmp)/'limit/partial-inner-observation.json').read_text())
            self.assertEqual(len(partial['inner_stages']), 1); self.assertTrue(partial['observer_restored'])
            self.assertIsNone(make.target._feature_lease)

    def test_reverse_pair_rng_drift_retains_all_six_completed_rows_before_reject(self):
        make, device = self.factory(); original = inner.COARSE.semantic_state; calls = 0
        def corrupted(session):
            nonlocal calls
            value = original(session); calls += 1
            if calls == 6:
                import hashlib
                value['r0']['rng_state_bytes'][0] ^= 1
                value['r0']['rng_state_sha256'] = hashlib.sha256(bytes(value['r0']['rng_state_bytes'])).hexdigest()
            return value
        with tempfile.TemporaryDirectory() as tmp, patch.object(inner.COARSE, 'semantic_state', side_effect=corrupted):
            out = Path(tmp)/'run'
            with self.assertRaisesRegex(ValueError, 'raw tokens/RNG/round work'): inner.run(make, device, MANIFEST, out)
            rows = [json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual(json.loads((out/'result.json').read_text())['sample_count'], 6)
            self.assertEqual(rows[4]['requests'][0]['output_sha256'], rows[5]['requests'][0]['output_sha256'])
            self.assertTrue((out/f'batches/4-inner_observed-1-{inner.ARM}/completed-batch.json').exists())
            self.assertTrue((out/f'batches/5-plain-1-{inner.ARM}/completed-batch.json').exists())


if __name__ == '__main__': unittest.main()
