#!/usr/bin/env python3
"""Observer-free, opt-in categorical A/B on the unchanged original R2 workload."""
from contextlib import contextmanager
import copy
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('categorical_ab_coarse_helpers', ROOT/'scripts/profile_paired_r2.py')
COARSE = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(COARSE)
BASE = COARSE.BASE
ENGINE = BASE.BASE
GUARD = BASE.GUARD
ARMS = ('target_only', 'fixed_gamma7_full_shadow')
VARIANTS = ('A', 'B')
PLAN = [('warmup', 0, arm, variant) for arm in ARMS for variant in VARIANTS]
for repeat, variants in enumerate((('A', 'B'), ('B', 'A'), ('B', 'A'), ('A', 'B'))):
    for arm in ARMS if repeat % 2 == 0 else ARMS[::-1]:
        PLAN.extend(('primary', repeat, arm, variant) for variant in variants)
PLAN = tuple(PLAN)
SOURCES = ('dspark_qwen/experimental_categorical.py', 'scripts/benchmark_categorical_ab_r2.py',
    'tests/test_experimental_categorical.py', 'tests/test_benchmark_categorical_ab_r2.py', 'docs/categorical-fixed-shape.md')
PROTOCOL = copy.deepcopy(BASE.PROTOCOL)
PROTOCOL.update(method='observer_free_original_r2_categorical_ab', arms=list(ARMS), variants=list(VARIANTS),
    timeout_seconds=300, cooperative_reserve_seconds=10,
    resident_cost='Same target, trained draft and both R2 graph pools remain resident across all20 batches and both arms/variants',
    measurement=dict(warmup_per_arm_variant=1, primary_per_arm_variant=4),
    schedule=[dict(phase=p, repeat=r, arm=a, variant=v) for p, r, a, v in PLAN],
    pair_order='AB/BA/BA/AB per arm; arm order target,gamma / gamma,target / target,gamma / gamma,target',
    variant_binding='Direct function aliases in tensor_sampling, packed_sampling, packed_target_sampling bound before batch and restored after; no wrapper in timed sampler path',
    categorical_gate='Bounded exact original/candidate cases on actual execution device before all warmups, independent generators, source identity and raw reconstructible results saved',
    screening_rule='Each arm pooled B/A throughput >=1.02 AND at least3/4 paired complete batches faster for B; both arms must pass',
    screening_not_statistical_confidence=True, invariant='All same-arm raw tokens/final RNG bytes/round decisions/work/execution equal across warmup and primary A/B rows',
    observer=False, stage_fences=False, signature_wrappers=False, trace_export=False, kineto=False, retry=False,
    timing_scope='Original complete-session timer including unchanged validation and any candidate scalar synchronization; no stage subtraction',
    production_default_changed=False, distribution_equivalence_claimed=False)
for key in ('primary_arm_position_counts', 'zero_arm_scope'): PROTOCOL.pop(key, None)


def validate_domain(manifest):
    canonical, _ = BASE.SHARED.load_workloads(ROOT/'configs/performance-workloads.example.json')
    actual = BASE.selected_manifest(manifest)['cases'][0]
    if actual != BASE.selected_manifest(canonical)['cases'][0] or manifest['output_tokens'] != 128 or manifest['sampling'] != canonical['sampling']:
        raise ValueError('Categorical A/B requires original R2 prompts/seeds,128-output budget and sampling')
    return actual


def identities(manifest):
    case = validate_domain(manifest)
    return [dict(phase=p, repeat=r, case_id=case['case_id'], arm=a, variant=v) for p, r, a, v in PLAN]


def bind(model, generation, checkpoint, workloads=None):
    value = COARSE.bind(model, generation, checkpoint, workloads)
    manifest, _ = BASE.SHARED.load_workloads(value['workloads'])
    value.update(protocol=PROTOCOL, protocol_sha256=ENGINE.PROFILE.digest(PROTOCOL),
        expected_samples=identities(manifest), worker_entry='scripts/benchmark_categorical_ab_r2.py')
    for name in SOURCES: value['source_sha256'][name] = GUARD.sha(ROOT/name)
    return value


@contextmanager
def sampler_variant(variant):
    from dspark_qwen import tensor_sampling, packed_sampling, packed_target_sampling
    from dspark_qwen.experimental_categorical import ORIGINAL_SAMPLE_CATEGORICAL, sample_categorical_fixed_shape
    if variant not in VARIANTS: raise ValueError('Unknown categorical variant')
    modules = (tensor_sampling, packed_sampling, packed_target_sampling)
    originals = [module.sample_categorical for module in modules]
    if any(value is not ORIGINAL_SAMPLE_CATEGORICAL for value in originals):
        raise ValueError('Categorical aliases must begin as the unwrapped original oracle')
    selected = ORIGINAL_SAMPLE_CATEGORICAL if variant == 'A' else sample_categorical_fixed_shape
    try:
        for module in modules: module.sample_categorical = selected
        yield
    finally:
        for module, value in zip(modules, originals): module.sample_categorical = value


def categorical_gate(device, directory, *, deadline=None):
    """Independent bounded RNG/callback gate; all serialization is outside samples."""
    import torch
    from dspark_qwen.experimental_categorical import ORIGINAL_SAMPLE_CATEGORICAL, sample_categorical_fixed_shape
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    cases = [dict(name='first_zero_uniform', probs=[1., 0., 0., 0.], uniform=0., token=0),
        dict(name='strict_cdf_tie', probs=[.5, 0., .5, 0.], uniform=.5, token=2),
        dict(name='adjacent_below_tie', probs=[.5, 0., .5, 0.], uniform=.5-2**-54, token=0),
        dict(name='adjacent_above_tie', probs=[.5, 0., .5, 0.], uniform=.5+2**-53, token=2),
        dict(name='last_hot', probs=[0., 0., 0., 1.], uniform=1.-2**-53, token=3),
        dict(name='tiny_support', probs=[1e-300, .5, .5, 0.], uniform=0., token=0),
        dict(name='rounding_zero_tail', probs=[0., .5, .5-5e-13, 0.], uniform=1.-2**-53, token=2),
        dict(name='changed_support', probs=[0., .5, .5-5e-13, 0.], uniform=1.-2**-53, mutation='support', token=0),
        dict(name='changed_shape', probs=[0., .5, .5-5e-13, 0.], uniform=1.-2**-53, mutation='reshape', token=1),
        dict(name='no_support', probs=[.5, .5, 0., 0.], uniform=.25, mutation='zero', error='IndexError'),
        dict(name='empty_shape', probs=[.5, .5, 0., 0.], uniform=.25, mutation='empty', error='IndexError'),
        dict(name='invalid_uniform_precedes_fallback', probs=[.5, .5, 0., 0.], uniform=1., mutation='zero', error='ValueError'),
        dict(name='invalid_law_before_rng', probs=[0., 0., 0., 0.], uniform=.25, error='ValueError', calls=0),
        dict(name='native_vocab_dense', construction='uniform_dense', size=151936, uniform=1.-2**-53, token=151935),
        dict(name='native_vocab_strided_zero_tail', construction='strided_sparse_deficit', size=151936, uniform=1.-2**-53, token=151934)]
    report = dict(status='running', device=str(device), scope='native_device' if device.type == 'cuda' else 'cpu_emulator_not_native',
        sources={n: GUARD.sha(ROOT/n) for n in ('dspark_qwen/tensor_sampling.py', 'dspark_qwen/experimental_categorical.py')},
        expected_case_count=len(cases), cases=[])
    path = directory/'result.json'; GUARD.write(path, report)
    try:
        for specification in cases:
            if deadline is not None and time.monotonic() >= deadline: raise TimeoutError('Categorical gate deadline')
            outcomes = []; probability_snapshots = []
            for sampler in (ORIGINAL_SAMPLE_CATEGORICAL, sample_categorical_fixed_shape):
                if specification.get('construction') == 'uniform_dense':
                    probs = torch.full((specification['size'],), 1./specification['size'], dtype=torch.float64, device=device)
                elif specification.get('construction') == 'strided_sparse_deficit':
                    storage = torch.zeros(2*specification['size'], dtype=torch.float64, device=device)
                    probs = storage[::2]; probs[1] = .5; probs[-2] = .5-5e-13
                else:
                    probs = torch.tensor(specification['probs'], dtype=torch.float64, device=device)
                generator = torch.Generator(device=device).manual_seed(723)
                before = generator.get_state().cpu().tolist()
                class Random:
                    calls = 0
                    def uniform(self, reference):
                        self.calls += 1
                        torch.rand((), dtype=torch.float64, device=device, generator=generator)
                        mutation = specification.get('mutation')
                        if mutation == 'zero': reference.zero_()
                        elif mutation == 'empty': reference.resize_(0)
                        elif mutation == 'reshape': reference.resize_(2, 2)
                        elif mutation == 'support': reference.copy_(reference.new_tensor([1., 0., 0., 0.]))
                        return torch.tensor(specification['uniform'], dtype=torch.float64, device=device)
                rng = Random(); outcome = dict(token=None, error=None)
                try:
                    token = sampler(probs, rng)
                    outcome.update(token=token.item(), token_dtype=str(token.dtype), token_device=str(token.device), token_shape=list(token.shape))
                except Exception as exc: outcome['error'] = dict(type=type(exc).__name__, message=str(exc))
                snapshot = probs.detach().cpu().contiguous(); probability_snapshots.append(snapshot)
                outcome.update(probabilities=snapshot.tolist() if probs.numel() <= 16 else None,
                    probabilities_sha256=hashlib.sha256(snapshot.numpy().tobytes()).hexdigest(),
                    probability_shape=list(probs.shape), probability_stride=list(probs.stride()), probability_dtype=str(probs.dtype),
                    calls=rng.calls, generator_before=before, generator_after=generator.get_state().cpu().tolist())
                outcomes.append(outcome)
            probabilities_equal = torch.equal(probability_snapshots[0], probability_snapshots[1])
            equal = outcomes[0] == outcomes[1] and probabilities_equal
            expected = (outcomes[0]['calls'] == specification.get('calls', 1) and
                ((outcomes[0]['error'] is not None and outcomes[0]['error']['type'] == specification['error']) if 'error' in specification else
                 (outcomes[0]['error'] is None and outcomes[0]['token'] == specification['token'])))
            raw_probabilities = None
            if specification.get('size', 0) > 16:
                raw_path = directory/(specification['name']+'.pt')
                torch.save(dict(original=probability_snapshots[0], candidate=probability_snapshots[1]), raw_path)
                raw_probabilities = dict(file=raw_path.name, sha256=GUARD.sha(raw_path))
            report['cases'].append(dict(specification=specification, original=outcomes[0], candidate=outcomes[1],
                exact_equal=equal, exact_probabilities_equal=probabilities_equal, expected_contract=expected, raw_probabilities=raw_probabilities))
            GUARD.write(path, report)
            if not equal or not expected: raise ValueError('Categorical native/CPU oracle gate failed: '+specification['name'])
        report.update(status='completed', passed=True, completed_case_count=len(report['cases']))
        GUARD.write(path, report); return report
    except BaseException as exc:
        report.update(status='failed', passed=False, error_type=type(exc).__name__, error=str(exc))
        GUARD.write(path, report); raise


def measure(factory, device, case, manifest, arm, variant, directory, *, deadline=None):
    if case != validate_domain(manifest) or arm not in ARMS or variant not in VARIANTS:
        raise ValueError('Original categorical A/B domain and declared arm/variant required')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', arm=arm, variant=variant, complete_batch=False)
    GUARD.write(directory/'status.json', status); proxy = COARSE.SessionFactory(factory); sample = None; bound = False
    try:
        with sampler_variant(variant):
            bound = True
            sample = BASE.batch(proxy, device, case, manifest, arm, diagnostic=False, deadline=deadline)
        sample['semantic_state'] = COARSE.semantic_state(proxy.session)
        sample['semantic_sha256'] = BASE.SHARED.sha_json(COARSE.invariant(sample))
        sample['variant_aliases_restored'] = True
        GUARD.write(directory/'completed-batch.json', sample)
        status.update(status='completed', complete_batch=True, variant_aliases_restored=True)
        GUARD.write(directory/'status.json', status); return sample
    except BaseException as exc:
        if sample is not None: GUARD.write(directory/'completed-batch.json', sample)
        status.update(status='failed', complete_batch=sample is not None, error_type=type(exc).__name__, error=str(exc), variant_aliases_restored=bound)
        GUARD.write(directory/'status.json', status); raise


def compare_semantics(samples):
    rows = []
    for arm in ARMS:
        values = [x for x in samples if x['arm'] == arm]
        if not values: continue
        expected = COARSE.invariant(values[0])
        differences = []
        for value in values:
            actual = COARSE.invariant(value)
            fields = [k for k in sorted(expected.keys() | actual.keys()) if
                k not in expected or k not in actual or expected[k] != actual[k]]
            differences.append(dict(phase=value['phase'], repeat=value['repeat'], variant=value['variant'], fields=fields))
        rows.append(dict(arm=arm, passed=all(not x['fields'] for x in differences), comparisons=differences))
    return rows


def verify_complete(samples, manifest, baseline=None):
    if [{k: row[k] for k in ('phase', 'repeat', 'case_id', 'arm', 'variant')} for row in samples] != identities(manifest):
        raise ValueError('Exactly20 ordered complete categorical A/B batches required')
    if any(row['output_tokens'] != 256 or not math.isfinite(row['batch_wall_seconds']) or row['batch_wall_seconds'] <= 0 for row in samples):
        raise ValueError('Every categorical batch requires256 outputs and finite positive wall')
    if any(not row['sampler_gate_passed'] or not row['variant_aliases_restored'] for row in samples):
        raise ValueError('Categorical gate and alias restoration required')
    if len({(row['sampler_gate_sha256'], row['sampler_gate_scope']) for row in samples}) != 1:
        raise ValueError('One common device categorical gate required')
    comparisons = compare_semantics(samples)
    if any(not row['passed'] for row in comparisons): raise ValueError('Categorical variant changed raw tokens/RNG/work')
    arms = {}
    for arm in ARMS:
        pairs = []
        for repeat in range(4):
            rows = {x['variant']: x for x in samples if x['phase'] == 'primary' and x['arm'] == arm and x['repeat'] == repeat}
            a, b = rows['A']['batch_wall_seconds'], rows['B']['batch_wall_seconds']
            pairs.append(dict(repeat=repeat, order='AB' if repeat in (0, 3) else 'BA', a_seconds=a, b_seconds=b,
                b_over_a_wall_ratio=b/a, b_over_a_throughput_ratio=a/b, b_faster=b<a))
        sums = {v: sum(x['batch_wall_seconds'] for x in samples if x['phase'] == 'primary' and x['arm'] == arm and x['variant'] == v) for v in VARIANTS}
        rates = {v: 1024/sums[v] for v in VARIANTS}; ratio = rates['B']/rates['A']; faster = sum(x['b_faster'] for x in pairs)
        strong = (baseline or {}).get('cells', [{}])[0].get('pooled_output_tokens_per_second')
        arms[arm] = dict(pairs=pairs, pooled_output_tokens_per_variant=1024, summed_batch_seconds=sums,
            pooled_output_tokens_per_second=rates, pooled_b_over_a_throughput_ratio=ratio, b_faster_pairs=faster,
            screening_passed=ratio >= 1.02 and faster >= 3,
            frozen_vllm_output_tokens_per_second=strong,
            over_frozen_vllm_rate_ratios={v: rates[v]/strong for v in VARIANTS} if strong else None)
    passed = all(x['screening_passed'] for x in arms.values())
    return dict(arms=arms, semantics=comparisons, screening_passed=passed,
        decision='candidate_passes_practical_screen' if passed else 'retain_original_no_demonstrated_useful_e2e_gain',
        device_evidence_scope=samples[0]['sampler_gate_scope'], screening_not_statistical_confidence=True,
        production_default_changed=False, distribution_equivalence_claimed=False)


def run(factory, device, manifest, out, *, deadline=None, setup=None, baseline=None):
    case = validate_domain(manifest); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'samples.jsonl').exists(): raise ValueError('Fresh categorical A/B output required')
    GUARD.write(out/'specification.json', dict(protocol=PROTOCOL, expected_samples=identities(manifest), frozen_vllm_reference=baseline))
    if setup is not None: GUARD.write(out/'runtime.json', setup)
    samples = []; GUARD.write(out/'result.json', dict(status='running', sample_count=0))
    try:
        gate = categorical_gate(device, out/'categorical-gate', deadline=deadline)
        gate_sha = ENGINE.PROFILE.digest(gate)
        for index, (phase, repeat, arm, variant) in enumerate(PLAN):
            if deadline is not None and time.monotonic() >= deadline: raise TimeoutError('Categorical A/B cooperative deadline')
            row = measure(factory, device, case, manifest, arm, variant, out/'batches'/f'{index}-{phase}-{repeat}-{arm}-{variant}', deadline=deadline)
            row.update(phase=phase, repeat=repeat, case_id=case['case_id'], request_count=2, prompt_length=256, variant=variant,
                sampler_gate_passed=gate['passed'], sampler_gate_sha256=gate_sha, sampler_gate_scope=gate['scope'])
            samples.append(row)
            with (out/'samples.jsonl').open('a') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
            comparisons = compare_semantics(samples)
            GUARD.write(out/'invariance.json', dict(comparisons=comparisons, passed=all(x['passed'] for x in comparisons)))
            if any(not x['passed'] for x in comparisons): raise ValueError('Categorical variant changed raw tokens/RNG/work')
            GUARD.write(out/'result.json', dict(status='running', sample_count=len(samples)))
        result = dict(status='completed', sample_count=20, aggregates=verify_complete(samples, manifest, baseline),
            categorical_gate_sha256=gate_sha, production_default_changed=False)
        GUARD.write(out/'result.json', result); return result
    except BaseException as exc:
        GUARD.write(out/'result.json', dict(status='partial_deadline' if isinstance(exc, TimeoutError) else 'failed', sample_count=len(samples),
            error_type=type(exc).__name__, error=str(exc), expected_samples=identities(manifest), production_default_changed=False))
        raise


def api():
    return SimpleNamespace(PROTOCOL=PROTOCOL, bind=bind, identities=identities, run=run, worker=worker,
        verify_complete=verify_complete, PairedFactory=BASE.PairedFactory)


def worker(*args, **kwargs): return ENGINE.worker(*args, experiment=api(), **kwargs)
def main(argv=None): return ENGINE.main(argv, experiment=api(), script=__file__)
if __name__ == '__main__': raise SystemExit(main())
