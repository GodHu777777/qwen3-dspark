"""Exact original-oracle gate for the opt-in fixed-shape fallback experiment."""
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
import inspect
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from dspark_qwen import tensor_sampling as original
from dspark_qwen.experimental_categorical import ORIGINAL_SAMPLE_CATEGORICAL, sample_categorical_fixed_shape
from test_benchmark_paired_r1 import load

ROOT = Path(__file__).resolve().parents[1]
fixture = load('categorical_r2_fixture', ROOT/'tests/test_benchmark_paired_r2.py')
coarse = load('categorical_semantics', ROOT/'scripts/profile_paired_r2.py')


@contextmanager
def sampler_variant(sampler):
    from dspark_qwen import packed_sampling, packed_target_sampling
    with ExitStack() as stack:
        for module in (original, packed_sampling, packed_target_sampling):
            stack.enter_context(patch.object(module, 'sample_categorical', sampler))
        yield


def clone_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().clone()
    if is_dataclass(value): return {f.name: clone_tree(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict): return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clone_tree(v) for v in value]
    return value


class CallbackRandom:
    def __init__(self, value=None, mutate=None):
        self.value, self.mutate = value, mutate
        self.generator = torch.Generator().manual_seed(125); self.calls = 0
    def uniform(self, reference):
        self.calls += 1
        drawn = torch.rand((), dtype=torch.float64, generator=self.generator)
        if self.mutate: self.mutate(reference)
        return drawn if self.value is None else self.value()


class CategoricalOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def equal_tree(self, a, b):
        self.assertIs(type(a), type(b))
        if isinstance(a, torch.Tensor):
            self.assertEqual((a.dtype, a.device, a.shape, a.stride()), (b.dtype, b.device, b.shape, b.stride()))
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a: self.equal_tree(a[k], b[k])
        elif isinstance(a, list):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b): self.equal_tree(x, y)
        else: self.assertEqual(a, b)

    def compare(self, make_probs, *, value=None, mutate=None, expected_calls=1, expected_error=None):
        results = []
        for sampler in (ORIGINAL_SAMPLE_CATEGORICAL, sample_categorical_fixed_shape):
            probs = make_probs(); rng = CallbackRandom(value, mutate)
            try: result = sampler(probs, rng); error = None
            except Exception as exc: result = None; error = (type(exc), str(exc))
            results.append(dict(result=result, error=error, probs=probs, calls=rng.calls, state=rng.generator.get_state()))
        self.equal_tree(results[0], results[1])
        self.assertEqual(results[0]['calls'], expected_calls)
        if expected_error is not None: self.assertIs(results[0]['error'][0], expected_error)
        return results[0]

    def test_frozen_original_remains_independent_and_production_default(self):
        self.assertIs(original.sample_categorical, ORIGINAL_SAMPLE_CATEGORICAL)
        source = inspect.getsource(ORIGINAL_SAMPLE_CATEGORICAL)
        self.assertIn('torch.nonzero(probs > 0, as_tuple=False)[-1, 0]', source)
        self.assertNotIn('fixed_shape', source)
        self.assertNotEqual(source, inspect.getsource(sample_categorical_fixed_shape))

    def test_dense_sparse_exhaustive_strict_cdf_boundaries_and_large_vocab(self):
        laws = [[1.], [1., 0.], [0., 1.], [1e-320, .5, .5, 0.]]
        laws += [[x/4 for x in row] for n in (2, 3, 4) for row in itertools.product(range(5), repeat=n) if sum(row) == 4]
        comparisons = 0
        for law in laws:
            probs = torch.tensor(law, dtype=torch.float64)
            values = {0., torch.nextafter(torch.tensor(1., dtype=torch.float64), torch.tensor(0., dtype=torch.float64)).item()}
            for value in probs.cumsum(0).tolist():
                for toward in (-float('inf'), value, float('inf')):
                    u = torch.nextafter(torch.tensor(value, dtype=torch.float64), torch.tensor(toward, dtype=torch.float64)).item()
                    if 0 <= u < 1: values.add(u)
            for u in values:
                result = self.compare(lambda: probs.clone(), value=lambda u=u: torch.tensor(u, dtype=torch.float64))
                self.assertIsNone(result['error']); comparisons += 1
        for size in (1, 2, 9, 151936):
            for position in set((0, size//2, size-1)):
                p = torch.zeros(size, dtype=torch.float64); p[position] = 1.
                for u in (0., .5, 1.-2**-53):
                    self.assertEqual(self.compare(lambda: p.clone(), value=lambda u=u: torch.tensor(u, dtype=torch.float64))['result'].item(), position)
            p = torch.arange(1, size+1, dtype=torch.float64); p /= p.sum()
            self.assertIsNone(self.compare(lambda: p.clone())['error'])
            storage = torch.empty(2*size, dtype=torch.float64); storage[::2] = p
            self.assertIsNone(self.compare(lambda: storage.clone()[::2])['error'])
        self.assertGreater(comparisons, 200)

    def test_rounding_fallback_and_post_callback_support_or_metadata(self):
        deficit = [0., .5, .5-5e-13, 0.]
        make = lambda: torch.tensor(deficit, dtype=torch.float64)
        high = lambda: torch.tensor(1.-2**-53, dtype=torch.float64)
        self.assertEqual(self.compare(make, value=high)['result'].item(), 2)
        self.assertEqual(self.compare(make, value=high, mutate=lambda p: p.copy_(torch.tensor([1., 0., 0., 0.], dtype=torch.float64)))['result'].item(), 0)
        # A changed 2-D shape uses original FIRST coordinate, not the flat index.
        result = self.compare(make, value=high, mutate=lambda p: p.resize_(2, 2))
        self.assertEqual(result['result'].item(), 1)
        for mutate in (lambda p: p.zero_(), lambda p: p.resize_(0), lambda p: p.resize_(()).fill_(1),
                       lambda p: p.resize_(2, 2).zero_()):
            self.compare(make, value=lambda: torch.tensor(.25, dtype=torch.float64), mutate=mutate, expected_error=IndexError)
        for mutate in (lambda p: p.resize_(2).copy_(torch.tensor([0., 1.], dtype=torch.float64)),
                       lambda p: setattr(p, 'data', p.float()),
                       lambda p: p.copy_(torch.tensor([float('nan'), 0., 1., 0.], dtype=torch.float64))):
            self.compare(make, value=high, mutate=mutate)
        self.compare(make, value=lambda: torch.tensor(1., dtype=torch.float64), mutate=lambda p: p.zero_(), expected_error=ValueError)

    def test_invalid_inputs_and_uniform_rejections_keep_exact_order(self):
        invalid = [None, [1.], torch.tensor(1., dtype=torch.float64), torch.empty(0, dtype=torch.float64),
            torch.tensor([1.], dtype=torch.float32), torch.tensor([1], dtype=torch.long),
            torch.tensor([[1.]], dtype=torch.float64)]
        invalid += [torch.tensor(x, dtype=torch.float64) for x in ([0., 0.], [.5, .6], [-.1, 1.1], [float('nan'), 1.], [float('inf'), 0.], [1.-2e-12, 0.])]
        for p in invalid:
            self.compare(lambda p=p: p.clone() if isinstance(p, torch.Tensor) else p, expected_calls=0, expected_error=ValueError)
        bad_uniform = [None, 0., torch.tensor(.5), torch.tensor([.5], dtype=torch.float64),
            torch.tensor(.5, dtype=torch.float64, device='meta')]
        bad_uniform += [torch.tensor(x, dtype=torch.float64) for x in (-.1, 1., float('nan'), float('inf'))]
        for u in bad_uniform:
            self.compare(lambda: torch.tensor([.5, .5], dtype=torch.float64), value=lambda u=u: u, expected_error=ValueError)

    def test_seeded_rng_exact_state_and_input_retention(self):
        generators = [torch.Generator().manual_seed(73), torch.Generator().manual_seed(73)]
        law = torch.tensor([0., .2, 0., .3, .5, 0.], dtype=torch.float64); before = law.clone()
        outputs = [[sampler(law, original.TensorRandom(g)) for _ in range(64)] for sampler, g in
                   zip((ORIGINAL_SAMPLE_CATEGORICAL, sample_categorical_fixed_shape), generators)]
        self.equal_tree(outputs[0], outputs[1]); self.assertTrue(torch.equal(law, before))
        self.assertTrue(torch.equal(generators[0].get_state(), generators[1].get_state()))

    def test_real_target_and_gamma_complete_sessions_exact_rounds_laws_confidence_rng_and_populated_kv(self):
        from dspark_qwen.packed_sampling import PackedSpeculativeSession
        from dspark_qwen.packed_target_sampling import PackedTargetOnlySession
        make, device = fixture.R2RealFactoryTests.factory(self)
        case = fixture.bench.selected_manifest(fixture.MANIFEST)['cases'][0]
        for arm in ('target_only', 'fixed_gamma7_full_shadow'):
            runs = []
            for sampler in (ORIGINAL_SAMPLE_CATEGORICAL, sample_categorical_fixed_shape):
                streams = dict(proposals=[], committed=[])
                propose, verify, step = PackedSpeculativeSession.propose, PackedSpeculativeSession.verify_commit, PackedTargetOnlySession.step
                def proposed(session, *args, **kwargs):
                    result = propose(session, *args, **kwargs)
                    streams['proposals'].append(clone_tree({r: p.proposal for r, p in result.proposals.items()})); return result
                def capture(session, result):
                    kv = {owner: {r: clone_tree(getattr(session, owner).request_kv(r)) for r in session.requests}
                          for owner in ('target', 'draft') if hasattr(session, owner)}
                    streams['committed'].append(dict(result=clone_tree(result), kv=kv))
                    return result
                def verified(session, *args, **kwargs): return capture(session, verify(session, *args, **kwargs))
                def stepped(session, *args, **kwargs): return capture(session, step(session, *args, **kwargs))
                proxy = coarse.SessionFactory(make)
                with sampler_variant(sampler), patch.object(PackedSpeculativeSession, 'propose', proposed), \
                        patch.object(PackedSpeculativeSession, 'verify_commit', verified), patch.object(PackedTargetOnlySession, 'step', stepped):
                    row = fixture.bench.batch(proxy, device, case, fixture.MANIFEST, arm)
                    row['semantic_state'] = coarse.semantic_state(proxy.session)
                runs.append((coarse.invariant(row), streams, row))
            self.equal_tree(runs[0][0], runs[1][0]); self.equal_tree(runs[0][1], runs[1][1])
            self.assertEqual(len(runs[0][1]['committed']), len(runs[0][2]['rounds']))
            self.assertTrue(runs[0][1]['committed'][0]['kv']['target']['r0'][0][0].numel() > 0)
            if arm != 'target_only':
                self.assertGreater(runs[0][2]['r1_tail_rounds'], 0)
                self.assertGreater(len({r['query_tokens'] for r in runs[0][2]['rounds']}), 1)
                self.assertTrue(any(x['rejected_index'] is not None and x['extra_token_kind'] == 'residual'
                    for r in runs[0][2]['rounds'] for x in r['requests'].values()))
        self.assertIs(original.sample_categorical, ORIGINAL_SAMPLE_CATEGORICAL)


    def test_remaining_one_output_real_shadow_session_and_exception_alias_restoration(self):
        from dspark_qwen import packed_sampling, packed_target_sampling
        from dspark_qwen.packed_sampling import RequestSpec
        make, device = fixture.R2RealFactoryTests.factory(self)
        case = fixture.bench.selected_manifest(fixture.MANIFEST)['cases'][0]; values = []
        for sampler in (ORIGINAL_SAMPLE_CATEGORICAL, sample_categorical_fixed_shape):
            session = make('fixed_gamma7_full_shadow')
            with sampler_variant(sampler):
                session.admit({r['request']: RequestSpec(torch.tensor([r['prompt_token_ids']]), 2,
                    original.TensorRandom(torch.Generator().manual_seed(r['seed']))) for r in case['requests']})
                issued = session.propose(['r0', 'r1'], mode='shadow')
                result = session.verify_commit(issued.proposals, {'r0': 0, 'r1': 0}, allocation_policy='external_nonanticipating')
                self.assertTrue(all(x['committed'] == 1 and x['stop_reason'] == 'budget' for x in result['requests'].values()))
                values.append(clone_tree(dict(proposed={r: h.proposal for r, h in issued.proposals.items()}, result=result,
                    semantics=coarse.semantic_state(session), kv={owner: {r: getattr(session, owner).request_kv(r)
                    for r in session.requests} for owner in ('target', 'draft')})))
        self.equal_tree(values[0], values[1])
        with self.assertRaisesRegex(RuntimeError, 'binding failure'):
            with sampler_variant(sample_categorical_fixed_shape): raise RuntimeError('binding failure')
        for module in (original, packed_sampling, packed_target_sampling):
            self.assertIs(module.sample_categorical, ORIGINAL_SAMPLE_CATEGORICAL)


if __name__ == '__main__': unittest.main()
