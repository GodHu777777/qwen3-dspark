#!/usr/bin/env python3
"""Bounded full-shadow child service diagnostic with reversed plain/observed pairs."""
import copy
import functools
import importlib.util
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('shadow_inner_coarse', ROOT/'scripts/profile_paired_r2.py')
COARSE = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(COARSE)
PAIRED = COARSE.BASE
ENGINE = PAIRED.BASE
GUARD = COARSE.GUARD
ARM = 'fixed_gamma7_full_shadow'
PLAN = (('warmup', 0, 'target_only'), ('warmup', 0, ARM), ('plain', 0, ARM),
        ('inner_observed', 0, ARM), ('inner_observed', 1, ARM), ('plain', 1, ARM))
PROTOCOL = copy.deepcopy(COARSE.PROTOCOL)
PROTOCOL.update(method='full_shadow_inner_service_reversed_pair_diagnostic',
    schedule=[dict(phase=p, repeat=r, arm=a) for p, r, a in PLAN],
    measurement=dict(target_warmup=1, shadow_warmup=1, shadow_plain=2, shadow_inner_observed=2),
    pair_order='plain0 observed0; observed1 plain1 after target/shadow warmups',
    max_inner_records=8192, max_rounds_per_batch=127,
    record_bound=dict(r2_children_per_round=46, r1_children_per_round=38, maximum_rounds=127,
        maximum_children=5842, maximum_outer_stages=1278, maximum_signatures=381,
        maximum_stage_signature_records=7501, maximum_observer_rows_with_summaries=7755),
    inner_partition='Within full-shadow propose only: packed backbone; shared base head; Markov embedding/projection/confidence modules; FP64 law; categorical draw; proposal copy; explicit residual',
    parent_denominator='Sum of owning propose synchronized-service envelopes (body through post-drain, including bookkeeping, excluding outer pre-drain)',
    decision_rule='Both fractions >=0.50 with both observed/plain ratios in [0.95,1.05]: supported; both <0.50 with valid ratios: not_supported; mixed thresholds or any ratio outside: inconclusive',
    full_shadow_phase='Inclusive owning propose envelope with inner fences; child ledger is a decomposition, never added to parent totals',
    observer_context='Observer enter/restore and common token/RNG collection excluded; wrappers and outer/inner fences included in observed complete-session wall',
    timing_scope='Perturbed serialized full-shadow child service; no original critical-path/kernel-active/speedup claim')


def validate_domain(manifest):
    canonical, _ = PAIRED.SHARED.load_workloads(ROOT/'configs/performance-workloads.example.json')
    actual = PAIRED.selected_manifest(manifest)['cases'][0]
    expected = PAIRED.selected_manifest(canonical)['cases'][0]
    if (actual != expected or manifest['output_tokens'] != 128 or manifest['sampling'] != canonical['sampling']):
        raise ValueError('Inner diagnostic requires exact original R2 inputs/seeds,128 output budget and sampling')
    return actual


def identities(manifest):
    case = validate_domain(manifest)
    return [dict(phase=p, repeat=r, case_id=case['case_id'], arm=a) for p, r, a in PLAN]


def bind(model, generation, checkpoint, workloads=None):
    value = COARSE.bind(model, generation, checkpoint, workloads)
    manifest, _ = PAIRED.SHARED.load_workloads(value['workloads'])
    value.update(protocol=PROTOCOL, protocol_sha256=ENGINE.PROFILE.digest(PROTOCOL),
        expected_samples=identities(manifest), worker_entry='scripts/profile_shadow_inner_r2.py')
    for name in ('scripts/profile_shadow_inner_r2.py', 'tests/test_profile_shadow_inner_r2.py', 'docs/shadow-inner-r2-profile.md'):
        value['source_sha256'][name] = GUARD.sha(ROOT/name)
    return value


class InnerObserver(COARSE.CoarseObserver):
    def __init__(self, factory, **kwargs):
        super().__init__(factory, **kwargs)
        self.inner_stages = []; self.inner_active = False

    def call(self, category, label, function):
        if not category.startswith('inner:'): return super().call(category, label, function)
        if (self.failed or self.inner_active or self.active is None or
                self.stages[self.active]['category'] != 'full_shadow_propose'):
            return function()
        if len(self.inner_stages) >= PROTOCOL['max_inner_records']:
            self.failed = True; raise RuntimeError('Inner child record limit')
        row = dict(index=len(self.inner_stages), parent_stage=self.active, round=self.round,
            category=category.removeprefix('inner:'), callable=label, callsite=self.callsite,
            status='running', start_seconds=time.perf_counter()-self.origin)
        self.inner_stages.append(row); self.inner_active = True
        begin = time.perf_counter()
        try:
            self.synchronize(); body_start = time.perf_counter()
            row['pre_boundary_drain_seconds'] = body_start-begin
            try: result = function()
            finally:
                body_end = time.perf_counter(); row['body_host_seconds'] = body_end-body_start
            post_start = time.perf_counter(); self.synchronize(); ended = time.perf_counter()
            row.update(post_boundary_drain_seconds=ended-post_start, boundary_bookkeeping_seconds=post_start-body_end,
                synchronized_service_seconds=ended-body_start, status='completed')
            return result
        except BaseException as exc:
            self.failed = True; row.update(status='failed', error_type=type(exc).__name__, error=str(exc)); raise
        finally:
            row['end_seconds'] = time.perf_counter()-self.origin; self.inner_active = False

    def guard_propose(self, session, active, proposal_limits=None, *, mode='shadow'):
        ordered = tuple(r for r, s in session.requests.items() if not s['finished'])
        if (len(self.rounds) >= 127 or tuple(active) != ordered or not ordered or
                list(session.requests) != ['r0', 'r1'] or mode != 'shadow' or proposal_limits is not None or
                session.temperature != 1. or session.eos_ids or session.draft.draft.spec.block_size != 7 or
                any(s['budget'] != 128 or s['prompt_length'] != 256 for s in session.requests.values())):
            self.failed = True; raise ValueError('Inner bound requires <=127 original full-shadow all-active rounds')

    def __enter__(self):
        from dspark_qwen import packed_sampling as spec, tensor_sampling as probability
        from dspark_qwen.packed_draft import PackedDraft
        super().__enter__()
        try:
            # Guard before original propose/outer observer, preserving its exact invocation.
            owner = spec.PackedSpeculativeSession; original = owner.propose
            @functools.wraps(original)
            def guarded(session, active, proposal_limits=None, *, mode='shadow'):
                self.guard_propose(session, active, proposal_limits, mode=mode)
                return original(session, active, proposal_limits, mode=mode)
            existed = 'propose' in vars(owner); raw = vars(owner).get('propose')
            owner.propose = guarded; self.patches.append((owner, 'propose', existed, raw, guarded))
            self.wrap(PackedDraft, 'backbone', category='inner:backbone', label='PackedDraft.backbone')
            self.wrap(self.factory.target.model.get_output_embeddings(), 'forward', category='inner:base_head', label='shared_base_head')
            for name in ('markov_embedding', 'markov_projection', 'confidence'):
                self.wrap(getattr(self.factory.draft, name), 'forward', category='inner:markov_confidence', label=name)
            for module in (spec, probability):
                for name, category in (('logits_to_probabilities', 'fp64_law'), ('sample_categorical', 'categorical_draw')):
                    self.wrap(module, name, category='inner:'+category, label=name)
            self.wrap(spec, '_copy_proposal', category='inner:proposal_copy', label='_copy_proposal')
            return self
        except BaseException:
            self.restore(); raise

    def summary(self, batch_wall_seconds):
        outer = super().summary(batch_wall_seconds)
        if len(self.rounds) > 127: raise ValueError('Actual original-workload round bound exceeded')
        if any(x['status'] != 'completed' for x in self.inner_stages): raise ValueError('Incomplete inner observation')
        for left, right in zip(self.inner_stages, self.inner_stages[1:]):
            if left['end_seconds'] > right['start_seconds']: raise ValueError('Overlapping inner children')
        parents = [x for x in self.stages if x['category'] == 'full_shadow_propose']
        if any(x['parent_stage'] not in {p['index'] for p in parents} for x in self.inner_stages):
            raise ValueError('Orphan inner child')
        partitions = []; categories = {}
        for parent in parents:
            children = [x for x in self.inner_stages if x['parent_stage'] == parent['index']]
            if not children: raise ValueError('Missing shadow inner child coverage')
            for child in children:
                if (child['round'] != parent['round'] or child['callsite'] != 'round' or
                        child['start_seconds'] < parent['start_seconds'] or child['end_seconds'] > parent['end_seconds']):
                    raise ValueError('Inner child outside owning propose')
                group = categories.setdefault(child['category'], dict(calls=0, synchronized_service_seconds=0.,
                    pre_boundary_drain_seconds=0., body_host_seconds=0., post_boundary_drain_seconds=0., boundary_bookkeeping_seconds=0.))
                group['calls'] += 1
                for key in tuple(group)[1:]: group[key] += child[key]
            pre = sum(x['pre_boundary_drain_seconds'] for x in children)
            service = sum(x['synchronized_service_seconds'] for x in children)
            total = parent['synchronized_service_seconds']; residual = total-pre-service
            if residual < 0: raise ValueError('Inner children exceed parent envelope')
            partitions.append(dict(parent_stage=parent['index'], round=parent['round'], child_count=len(children),
                parent_synchronized_service_seconds=total, child_pre_boundary_drain_seconds=pre,
                child_synchronized_service_seconds=service, parent_residual_seconds=residual))
        if len(partitions) != len(self.rounds): raise ValueError('Exactly one owning propose per round required')
        parent_total = sum(x['parent_synchronized_service_seconds'] for x in partitions)
        probability_total = sum(x['synchronized_service_seconds'] for x in self.inner_stages if x['category'] in ('fp64_law', 'categorical_draw'))
        if parent_total <= 0: raise ValueError('Positive observed propose envelope required')
        outer.update(inner_stages=self.inner_stages, inner_categories=categories, inner_parent_partitions=partitions,
            parent_propose_seconds=parent_total, inner_probability_seconds=probability_total,
            inner_probability_fraction=probability_total/parent_total,
            inner_accounting='Parent service = child pre-drains + disjoint child services + residual; parent/child ledgers never added',
            parent_denominator=PROTOCOL['parent_denominator'])
        return outer


def measure(factory, device, case, manifest, arm, phase, directory, *, deadline=None):
    if case != validate_domain(manifest): raise ValueError('Case differs from bound original workload')
    if ((phase == 'warmup' and arm not in COARSE.ARMS) or
            (phase in ('plain', 'inner_observed') and arm != ARM) or phase not in ('warmup', 'plain', 'inner_observed')):
        raise ValueError('Unknown inner diagnostic phase/arm')
    directory = Path(directory); observers = []; failure = None
    def observer_factory(make):
        value = InnerObserver(make); observers.append(value); return value
    try:
        sample = COARSE.measure(factory, device, case, manifest, arm,
            'coarse_sync' if phase == 'inner_observed' else phase, directory,
            deadline=deadline, observer_class=observer_factory)
        if 'coarse_observation' in sample:
            sample['inner_observation'] = sample.pop('coarse_observation')
            validate_inner_sample(sample)
        return sample
    except BaseException as exc:
        failure = dict(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status_path = directory/'status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text()); status.update(phase=phase, observer_kind='full_shadow_inner' if observers else 'none')
            if failure is not None: status.update(failure)
            GUARD.write(status_path, status)
        for observation in observers:
            if failure is not None or observation.failed or (directory/'partial-observation.json').exists():
                GUARD.write(directory/'partial-observation.json', dict(stages=observation.stages, signatures=observation.signatures,
                    rounds=observation.rounds, observer_restored=observation.closed and not observation.patches))
                GUARD.write(directory/'partial-inner-observation.json', dict(inner_stages=observation.inner_stages,
                    outer_stages=observation.stages, rounds=observation.rounds,
                    observer_restored=observation.closed and not observation.patches))



def validate_inner_sample(sample):
    observation = sample['inner_observation']; rounds = sample['rounds']
    if len(rounds) > 127 or len(observation['inner_parent_partitions']) != len(rounds):
        raise ValueError('Original full-shadow round bound/parent coverage differs')
    for index, row in enumerate(rounds):
        r = len(row['active_requests'])
        if r not in (1, 2) or any(x['committed'] < 1 for x in row['requests'].values()):
            raise ValueError('Each active request must commit at least one output')
        if sum(x['proposal_positions'] for x in row['work']['proposal_batches']) != 7*r:
            raise ValueError('Complete seven-position proposals required')
        children = [x for x in observation['inner_stages'] if x['round'] == index]
        counts = {name: sum(x['category'] == name for x in children) for name in
            ('backbone', 'base_head', 'markov_confidence', 'fp64_law', 'categorical_draw', 'proposal_copy')}
        if counts != dict(backbone=1, base_head=1, markov_confidence=21, fp64_law=7,
                          categorical_draw=7*r, proposal_copy=r) or len(children) != 30+8*r:
            raise ValueError('Missing, duplicate or unexpected inner child call coverage')
        for module in ('markov_embedding', 'markov_projection', 'confidence'):
            if sum(x['callable'] == module for x in children) != 7:
                raise ValueError('Seven calls per Markov/confidence module required')
    if (len(observation['inner_stages']) > 5842 or len(observation['stages']) > 1278 or
            len(observation['signatures']) > 381):
        raise ValueError('Proven original-workload record bounds exceeded')
    parent_total = sum(x['synchronized_service_seconds'] for x in observation['stages'] if x['category'] == 'full_shadow_propose')
    probability_total = sum(x['synchronized_service_seconds'] for x in observation['inner_stages'] if x['category'] in ('fp64_law', 'categorical_draw'))
    if (parent_total <= 0 or observation['parent_propose_seconds'] != parent_total or
            observation['inner_probability_seconds'] != probability_total or
            observation['inner_probability_fraction'] != probability_total/parent_total):
        raise ValueError('Inner hypothesis fraction does not reconstruct from raw parent/child records')


def compare_pairs(samples, *, complete=False):
    nonwarm = [x for x in samples if x['phase'] != 'warmup']
    reference = COARSE.invariant(nonwarm[0]) if nonwarm else None
    # Same seeds require cross-pair equality too; no semantic drift can be classified as timing noise.
    for row in nonwarm:
        if COARSE.invariant(row) != reference: raise ValueError('Inner observation changed raw tokens/RNG/round work')
    pairs = []
    for repeat in (0, 1):
        plain = [x for x in samples if (x['phase'], x['repeat']) == ('plain', repeat)]
        observed = [x for x in samples if (x['phase'], x['repeat']) == ('inner_observed', repeat)]
        if len(plain) > 1 or len(observed) > 1: raise ValueError('Duplicate inner pair row')
        if not plain or not observed:
            if complete: raise ValueError('Two complete reversed pairs required')
            continue
        a, b = plain[0], observed[0]
        pairs.append(dict(repeat=repeat, order='plain_then_observed' if repeat == 0 else 'observed_then_plain',
            raw_semantics_equal=True, plain_seconds=a['batch_wall_seconds'], observed_seconds=b['batch_wall_seconds'],
            observed_over_plain=b['batch_wall_seconds']/a['batch_wall_seconds'],
            probability_fraction=b['inner_observation']['inner_probability_fraction']))
    return pairs


def classify(pairs):
    if len(pairs) != 2: return dict(classification='incomplete', reason='Both reversed pairs required')
    if any(not math.isfinite(x['observed_over_plain']) or x['observed_over_plain'] <= 0 or
           not math.isfinite(x['probability_fraction']) or not 0 <= x['probability_fraction'] <= 1 for x in pairs):
        raise ValueError('Finite positive wall ratios and bounded probability fractions required')
    if any(not .95 <= x['observed_over_plain'] <= 1.05 for x in pairs):
        return dict(classification='inconclusive', reason='observed_plain_disturbance_outside_predeclared_5pct')
    flags = [x['probability_fraction'] >= .5 for x in pairs]
    if all(flags): return dict(classification='supported', reason='FP64_law_plus_draw_at_least_half_in_both_pairs')
    if not any(flags): return dict(classification='not_supported', reason='FP64_law_plus_draw_below_half_in_both_pairs')
    return dict(classification='inconclusive', reason='threshold_disagreement_across_reversed_orders')


def verify_complete(samples, manifest, baseline=None):
    del baseline
    if [{k: x[k] for k in ('phase', 'repeat', 'case_id', 'arm')} for x in samples] != identities(manifest):
        raise ValueError('Exactly six ordered inner diagnostic batches required')
    if any(x['output_tokens'] != 256 for x in samples): raise ValueError('Every inner batch requires256 outputs')
    for sample in samples:
        if sample['phase'] == 'inner_observed': validate_inner_sample(sample)
    pairs = compare_pairs(samples, complete=True)
    return dict(pairs=pairs, hypothesis=classify(pairs), diagnostic_only=True, primary_throughput_claimed=False,
        threshold=.50, allowed_wall_ratio=[.95, 1.05], timing_scope=PROTOCOL['timing_scope'])


def run(factory, device, manifest, out, *, deadline=None, setup=None, baseline=None):
    case = validate_domain(manifest); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'samples.jsonl').exists(): raise ValueError('Fresh inner diagnostic output required')
    GUARD.write(out/'specification.json', dict(protocol=PROTOCOL, expected_samples=identities(manifest), frozen_vllm_reference=baseline))
    if setup is not None: GUARD.write(out/'runtime.json', setup)
    samples = []; GUARD.write(out/'result.json', dict(status='running', sample_count=0))
    try:
        for index, (phase, repeat, arm) in enumerate(PLAN):
            if deadline is not None and time.monotonic() >= deadline: raise TimeoutError('Inner cooperative deadline')
            row = measure(factory, device, case, manifest, arm, phase, out/'batches'/f'{index}-{phase}-{repeat}-{arm}', deadline=deadline)
            row.update(phase=phase, repeat=repeat, case_id=case['case_id'], request_count=2, prompt_length=256)
            samples.append(row)
            with (out/'samples.jsonl').open('a') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
            pairs = compare_pairs(samples)
            GUARD.write(out/'invariance.json', dict(pairs=pairs, all_nonwarm_semantics_equal=True))
            GUARD.write(out/'result.json', dict(status='running', sample_count=len(samples)))
        result = dict(status='completed', sample_count=6, aggregates=verify_complete(samples, manifest), diagnostic_only=True,
            primary_throughput_claimed=False, distribution_equivalence_claimed=False)
        GUARD.write(out/'result.json', result); return result
    except BaseException as exc:
        GUARD.write(out/'result.json', dict(status='partial_deadline' if isinstance(exc, TimeoutError) else 'failed', sample_count=len(samples),
            error_type=type(exc).__name__, error=str(exc), diagnostic_only=True, expected_samples=identities(manifest)))
        raise


def api():
    return SimpleNamespace(PROTOCOL=PROTOCOL, bind=bind, identities=identities, run=run, worker=worker,
        verify_complete=verify_complete, PairedFactory=PAIRED.PairedFactory)


def worker(*args, **kwargs): return ENGINE.worker(*args, experiment=api(), **kwargs)
def main(argv=None): return ENGINE.main(argv, experiment=api(), script=__file__)
if __name__ == '__main__': raise SystemExit(main())
