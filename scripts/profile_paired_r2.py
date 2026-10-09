#!/usr/bin/env python3
"""Six complete R2 batches with bounded coarse synchronized service observation."""
import contextlib
import copy
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('coarse_r2_base', ROOT/'scripts/benchmark_paired_r2.py')
BASE = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(BASE)
GUARD = BASE.GUARD
ARMS = ('target_only', 'fixed_gamma7_full_shadow')
PLAN = (('warmup', ARMS[0]), ('warmup', ARMS[1]), ('plain', ARMS[0]),
        ('coarse_sync', ARMS[0]), ('coarse_sync', ARMS[1]), ('plain', ARMS[1]))
PROTOCOL = copy.deepcopy(BASE.PROTOCOL)
PROTOCOL.update(method='six_batch_r2_coarse_synchronized_service_diagnostic', arms=list(ARMS),
    timeout_seconds=300, cooperative_reserve_seconds=10,
    resident_cost='Same target, trained draft and both R2 graph pools remain resident across all six batches and both arms',
    measurement=dict(warmup_per_arm=1, plain_per_arm=1, coarse_sync_per_arm=1),
    schedule=[dict(phase=p, arm=a) for p, a in PLAN], diagnostic_only=True,
    pair_order='warmup target/spec; plain target; coarse target/spec; plain spec',
    max_stage_records=8192, max_signature_records=8192, max_rounds_per_batch=254,
    full_shadow_phase='Entire propose including its nested head, Markov, FP64 laws and draws; nested coarse stages suppressed',
    target_phase='Strategy prefill/verify, target.predict, and admission-only head; prefill owns its nested commit',
    probability_phase='Outside full_shadow: logits_to_probabilities, sample_categorical, verify_proposal; nested probability calls suppressed',
    commit_phase='Outside prefill: strategy commit, draft append_committed, feature release',
    remainder='Unwrapped reset/session and loop work, private confidence D2H/capability validation, token materialization, record construction, final synchronization and instrumentation gaps',
    fences='Only outer observed callable boundaries. Pre-drain is prior unassigned queued work; body plus post-drain is current synchronized service. Fence wall includes waiting and API overhead.',
    nested_signatures='Host spans/counts only, attributed to current outer stage; no nested fence and never added to outer totals',
    invariant='Private actual token arrays, final per-request RNG state bytes and digest, all round decisions/Q/work/execution, final lengths and prefill work equal within each arm',
    observer_context='Enter/restore and common post-timer semantic collection excluded; wrappers and fences included in coarse complete-batch wall',
    timing_scope='Perturbed serialized service diagnostic, not original critical path, kernel-active time or primary speedup',
    trace_export=False, kineto=False, retry=False)
PROTOCOL.pop('primary_arm_position_counts', None)
PROTOCOL.pop('zero_arm_scope', None)


def identities(manifest):
    case = BASE.selected_manifest(manifest)['cases'][0]
    return [dict(phase=p, repeat=0, case_id=case['case_id'], arm=a) for p, a in PLAN]


def bind(model, generation, checkpoint, workloads=None):
    value = BASE.bind(model, generation, checkpoint, workloads)
    manifest, _ = BASE.SHARED.load_workloads(value['workloads'])
    value.update(protocol=PROTOCOL, protocol_sha256=BASE.BASE.PROFILE.digest(PROTOCOL),
        expected_samples=identities(manifest), worker_entry='scripts/profile_paired_r2.py')
    for name in ('scripts/profile_paired_r2.py', 'tests/test_profile_paired_r2.py', 'docs/paired-r2-profile.md'):
        value['source_sha256'][name] = GUARD.sha(ROOT/name)
    return value


class SessionFactory:
    """Same factory call, with a retained reference for common post-timer evidence."""
    def __init__(self, factory): self.original = factory; self.session = None
    def __getattr__(self, name): return getattr(self.original, name)
    def __call__(self, arm):
        self.session = self.original(arm)
        return self.session


def semantic_state(session):
    """Read existing output/RNG state after the batch timer; never draw or forward."""
    outputs = session.outputs()
    result = {}
    for request, state in session.requests.items():
        raw = state['rng'].generator.get_state().detach().cpu().contiguous()
        if str(raw.dtype) != 'torch.uint8' or raw.ndim != 1:
            raise ValueError('Expected explicit one-dimensional RNG state bytes')
        values = raw.tolist()
        result[request] = dict(output_token_ids=outputs[request], rng_state_bytes=values,
            rng_state_sha256=hashlib.sha256(bytes(values)).hexdigest(),
            finished=state['finished'], budget=state['budget'])
    return result


def invariant(sample):
    keys = ('arm', 'output_tokens', 'rounds', 'execution_coverage', 'admission_output_tokens',
        'committed_round_output_tokens', 'prefill_work', 'final_context_lengths', 'accepted_draft_tokens',
        'selected_proposal_tokens', 'selected_layer_ids', 'probability_policy', 'full_shadow_positions',
        'request_rounds', 'r1_tail_rounds', 'semantic_state')
    value = {key: sample[key] for key in keys}
    value['requests'] = [{k: row[k] for k in ('request', 'output_tokens', 'output_sha256')} for row in sample['requests']]
    for row in value['requests']:
        state = value['semantic_state'][row['request']]
        if (len(state['output_token_ids']) != row['output_tokens'] or
                BASE.SHARED.sha_json(state['output_token_ids']) != row['output_sha256'] or
                hashlib.sha256(bytes(state['rng_state_bytes'])).hexdigest() != state['rng_state_sha256']):
            raise ValueError('Raw semantic token/RNG bytes and hashes disagree')
    return value


class CoarseObserver:
    """Temporary single-owner wrappers, exclusive outer intervals, no event trace."""
    def __init__(self, factory, *, synchronize=None):
        self.factory = factory
        self.synchronize = synchronize or (lambda: BASE.BASE.NATIVE.sync(factory.target.device))
        self.patches = []; self.stages = []; self.signatures = []; self.rounds = []
        self.active = None; self.round = None; self.callsite = 'outside_session'; self.closed = False; self.failed = False
        self.origin = None

    def call(self, category, label, function):
        if self.active is not None or self.failed: return function()  # Owning outer operation keeps all nested work.
        if len(self.stages) >= PROTOCOL['max_stage_records']:
            self.failed = True
            raise RuntimeError('Coarse stage record limit')
        index = len(self.stages)
        row = dict(index=index, category=category, callable=label, callsite=self.callsite,
                   round=self.round, status='running', start_seconds=time.perf_counter()-self.origin)
        self.stages.append(row); self.active = index
        start = time.perf_counter()
        try:
            self.synchronize(); body_start = time.perf_counter()
            row['pre_boundary_drain_seconds'] = body_start-start
            try:
                result = function()
            finally:
                body_end = time.perf_counter(); row['body_host_seconds'] = body_end-body_start
            before_post = time.perf_counter(); self.synchronize(); ended = time.perf_counter()
            row.update(post_boundary_drain_seconds=ended-before_post,
                synchronized_service_seconds=ended-body_start,
                boundary_bookkeeping_seconds=before_post-body_end, status='completed')
            return result
        except BaseException as exc:
            self.failed = True
            row.update(status='failed', error_type=type(exc).__name__, error=str(exc)); raise
        finally:
            row['end_seconds'] = time.perf_counter()-self.origin
            self.active = None

    def wrap(self, owner, name, *, category=None, label=None, start=False, end=False, admission=False, signature=False):
        original = getattr(owner, name); existed = name in vars(owner); raw = vars(owner).get(name)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            previous = self.callsite
            if admission: self.callsite = 'admission'
            if start:
                if self.round is not None: raise RuntimeError('Unfinished observed round')
                if len(self.rounds) >= PROTOCOL['max_rounds_per_batch']: raise RuntimeError('Coarse round record limit')
                self.round = len(self.rounds); self.callsite = 'round'
                self.rounds.append(dict(index=self.round, context_lengths=dict(args[0].target.lengths), status='running',
                    start_seconds=time.perf_counter()-self.origin))
            try:
                if signature:
                    if self.failed: return original(*args, **kwargs)
                    if len(self.signatures) >= PROTOCOL['max_signature_records']:
                        self.failed = True
                        raise RuntimeError('Signature record limit')
                    before = time.perf_counter()
                    try: return original(*args, **kwargs)
                    finally:
                        self.signatures.append(dict(stage=self.active, round=self.round, callsite=self.callsite,
                            host_seconds=time.perf_counter()-before))
                result = self.call(category, label or name, lambda: original(*args, **kwargs)) if category else original(*args, **kwargs)
                if end:
                    if self.round is None: raise RuntimeError('Round end without observed start')
                    target = result['work']['target']
                    self.rounds[self.round].update(status='completed', end_seconds=time.perf_counter()-self.origin,
                        query_tokens=target['physical_query_tokens'],
                        execution_kind=target.get('execution_kind'))
                return result
            except BaseException as exc:
                self.failed = True
                if self.round is not None: self.rounds[self.round].update(status='failed', error=repr(exc))
                raise
            finally:
                if end: self.round = None; self.callsite = 'outside_session'
                elif admission: self.callsite = previous
        setattr(owner, name, observed); self.patches.append((owner, name, existed, raw, observed))

    def __enter__(self):
        from dspark_qwen import packed_sampling as spec, packed_target_sampling as target, tensor_sampling as probability
        from dspark_qwen.target_strategy import PersistentTargetStrategy
        from dspark_qwen.persistent_qwen_target import PersistentQwenTarget
        from dspark_qwen.persistent_qwen_graph import FullTargetExecution
        from dspark_qwen.packed_draft import PackedDraft
        self.origin = time.perf_counter()
        try:
            for cls in (spec.PackedSpeculativeSession, target.PackedTargetOnlySession): self.wrap(cls, 'admit', admission=True)
            self.wrap(spec.PackedSpeculativeSession, 'propose', category='full_shadow_propose', start=True)
            self.wrap(spec.PackedSpeculativeSession, 'verify_commit', end=True)
            self.wrap(target.PackedTargetOnlySession, 'step', start=True, end=True)
            for name in ('prefill', 'verify'):
                self.wrap(PersistentTargetStrategy, name, category='target_service', label='strategy.'+name)
            self.wrap(PersistentQwenTarget, 'predict', category='target_service', label='target.predict')
            self.wrap(self.factory.target.model.get_output_embeddings(), 'forward', category='target_service', label='admission_head')
            for module in (spec, target, probability):
                for name in ('logits_to_probabilities', 'sample_categorical', 'verify_proposal'):
                    if hasattr(module, name): self.wrap(module, name, category='fp64_probability_service', label=name)
            for cls, name in ((PersistentTargetStrategy, 'commit'), (PackedDraft, 'append_committed'),
                              (PersistentTargetStrategy, 'release_features')):
                self.wrap(cls, name, category='commit_projection_release', label=cls.__name__+'.'+name)
            self.wrap(FullTargetExecution, '_model_signature', signature=True)
            return self
        except BaseException:
            self.restore(); raise

    def restore(self):
        errors = []
        for owner, name, existed, raw, wrapper in reversed(self.patches):
            try:
                if getattr(owner, name) is not wrapper: errors.append(name+' was replaced during observation')
                if existed: setattr(owner, name, raw)
                else: delattr(owner, name)
            except BaseException as exc: errors.append(name+': '+repr(exc))
        self.patches.clear(); self.closed = True
        if errors: raise RuntimeError('Observer restoration failed: '+repr(errors))

    def __exit__(self, *args): self.restore()

    def summary(self, batch_wall_seconds):
        if not self.closed or self.patches: raise RuntimeError('Restore observer before summary')
        if self.failed: raise ValueError('Observer failed during batch')
        if any(x['status'] != 'completed' for x in self.stages+self.rounds): raise ValueError('Incomplete coarse observation')
        for left, right in zip(self.stages, self.stages[1:]):
            if left['end_seconds'] > right['start_seconds']: raise ValueError('Overlapping coarse stages')
        by_category = {}; drain = 0.; service = 0.
        for row in self.stages:
            group = by_category.setdefault(row['category'], dict(calls=0, synchronized_service_seconds=0.,
                body_host_seconds=0., post_boundary_drain_seconds=0., pre_boundary_drain_seconds=0., boundary_bookkeeping_seconds=0.))
            group['calls'] += 1
            for key in ('synchronized_service_seconds', 'body_host_seconds', 'post_boundary_drain_seconds', 'pre_boundary_drain_seconds', 'boundary_bookkeeping_seconds'):
                group[key] += row[key]
            drain += row['pre_boundary_drain_seconds']; service += row['synchronized_service_seconds']
        remainder = batch_wall_seconds-drain-service
        if remainder < 0: raise ValueError('Coarse accounting exceeds complete batch timer')
        round_remainder = 0.
        for row in self.rounds:
            parts = [x for x in self.stages if x['round'] == row['index']]
            wall = row['end_seconds']-row['start_seconds']
            pre = sum(x['pre_boundary_drain_seconds'] for x in parts)
            current = sum(x['synchronized_service_seconds'] for x in parts)
            rest = wall-pre-current
            if rest < 0: raise ValueError('Coarse stages exceed owning round')
            row.update(wall_seconds=wall, pre_boundary_drain_seconds=pre, synchronized_service_seconds=current,
                unassigned_round_remainder_seconds=rest)
            round_remainder += rest
        session_remainder = remainder-round_remainder
        if session_remainder < 0: raise ValueError('Round remainder exceeds batch remainder')
        return dict(stages=self.stages, categories=by_category, rounds=self.rounds,
            unassigned_round_remainder_seconds=round_remainder, unassigned_session_remainder_seconds=session_remainder,
            signatures=self.signatures, signature_calls=len(self.signatures),
            nested_signature_host_seconds=sum(x['host_seconds'] for x in self.signatures),
            prior_unassigned_boundary_drain_seconds=drain, synchronized_service_seconds=service,
            unassigned_batch_remainder_seconds=remainder, complete_batch_wall_seconds=batch_wall_seconds,
            observer_restored=True, timing_scope=PROTOCOL['timing_scope'],
            accounting='batch = pre-boundary drains + disjoint body/post-drain services + remainder; nested signature spans are already contained and never added')


def measure(factory, device, case, manifest, arm, phase, directory, *, deadline=None, observer_class=CoarseObserver):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    proxy = SessionFactory(factory); observer = observer_class(factory) if phase == 'coarse_sync' else None
    status = dict(status='running', phase=phase, arm=arm, complete_batch=False)
    GUARD.write(directory/'status.json', status)
    sample = None
    try:
        with observer if observer is not None else contextlib.nullcontext():
            sample = BASE.batch(proxy, device, case, manifest, arm, diagnostic=False, deadline=deadline)
        # Identical collector for warmup/plain/coarse. These reads and JSON writes follow the timer.
        sample['semantic_state'] = semantic_state(proxy.session)
        sample['semantic_sha256'] = BASE.SHARED.sha_json(invariant(sample))
        GUARD.write(directory/'completed-batch.json', sample)
        if observer is not None:
            observation = observer.summary(sample['batch_wall_seconds'])
            if len(observation['rounds']) != len(sample['rounds']): raise ValueError('Observed round coverage differs')
            for a, b in zip(observation['rounds'], sample['rounds']):
                if a['query_tokens'] != b['query_tokens'] or a['execution_kind'] != b['execution_kind']:
                    raise ValueError('Observed round shape/execution differs')
            sample['coarse_observation'] = observation
            GUARD.write(directory/'observation.json', observation)
        status.update(status='completed', complete_batch=True, observer_restored=observer is None or observer.closed)
        GUARD.write(directory/'status.json', status)
        return sample
    except BaseException as exc:
        if sample is not None: GUARD.write(directory/'completed-batch.json', sample)
        if observer is not None:
            GUARD.write(directory/'partial-observation.json', dict(stages=observer.stages, signatures=observer.signatures,
                rounds=observer.rounds, observer_restored=observer.closed and not observer.patches))
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc), complete_batch=sample is not None)
        GUARD.write(directory/'status.json', status); raise


def compare_pairs(samples, *, require_complete):
    comparisons = []
    for arm in ARMS:
        plain = [x for x in samples if x['phase'] == 'plain' and x['arm'] == arm]
        coarse = [x for x in samples if x['phase'] == 'coarse_sync' and x['arm'] == arm]
        if len(plain) > 1 or len(coarse) > 1: raise ValueError('Duplicate control/observation')
        if not plain or not coarse:
            if require_complete: raise ValueError('Both complete controls required for each arm')
            continue
        a, b = plain[0], coarse[0]
        left, right = invariant(a), invariant(b)
        differences = [key for key in left if left[key] != right[key]]
        comparisons.append(dict(arm=arm, passed=not differences, differing_fields=differences,
            plain_semantic_sha256=BASE.SHARED.sha_json(left), coarse_semantic_sha256=BASE.SHARED.sha_json(right),
            plain_batch_wall_seconds=a['batch_wall_seconds'], coarse_batch_wall_seconds=b['batch_wall_seconds'],
            coarse_over_plain_wall_ratio=b['batch_wall_seconds']/a['batch_wall_seconds'],
            wall_delta_seconds=b['batch_wall_seconds']-a['batch_wall_seconds'],
            interpretation='Whole observed batch disturbance, not a primary speed ratio or subtraction-based speedup'))
    return comparisons


def verify_complete(samples, manifest, baseline=None):
    del baseline
    if [{k: x[k] for k in ('phase', 'repeat', 'case_id', 'arm')} for x in samples] != identities(manifest):
        raise ValueError('Exactly six ordered complete diagnostic batches required')
    if any(x['output_tokens'] != 256 for x in samples): raise ValueError('Every complete R2 batch requires256 outputs')
    comparisons = compare_pairs(samples, require_complete=True)
    if any(not x['passed'] for x in comparisons): raise ValueError('Observation changed raw output/RNG/work semantics')
    return dict(comparisons=comparisons, diagnostic_only=True, primary_throughput_claimed=False,
        timing_scope=PROTOCOL['timing_scope'])


def run(factory, device, manifest, out, *, deadline=None, setup=None, baseline=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'samples.jsonl').exists(): raise ValueError('Fresh coarse diagnostic directory required')
    GUARD.write(out/'specification.json', dict(protocol=PROTOCOL, expected_samples=identities(manifest), frozen_vllm_reference=baseline))
    if setup is not None: GUARD.write(out/'runtime.json', setup)
    samples = []; case = BASE.selected_manifest(manifest)['cases'][0]
    GUARD.write(out/'result.json', dict(status='running', sample_count=0))
    try:
        for index, (phase, arm) in enumerate(PLAN):
            if deadline is not None and time.monotonic() >= deadline: raise TimeoutError('Coarse diagnostic deadline')
            row = measure(factory, device, case, manifest, arm, phase, out/'batches'/f'{index}-{phase}-{arm}', deadline=deadline)
            row.update(phase=phase, repeat=0, case_id=case['case_id'], request_count=2, prompt_length=256)
            samples.append(row)
            with (out/'samples.jsonl').open('a') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
            comparisons = compare_pairs(samples, require_complete=False)
            GUARD.write(out/'invariance.json', dict(comparisons=comparisons))
            if any(not x['passed'] for x in comparisons): raise ValueError('Observation changed raw output/RNG/work semantics')
            GUARD.write(out/'result.json', dict(status='running', sample_count=len(samples)))
        result = dict(status='completed', sample_count=6, aggregates=verify_complete(samples, manifest),
            diagnostic_only=True, primary_throughput_claimed=False, distribution_equivalence_claimed=False)
        GUARD.write(out/'result.json', result); return result
    except BaseException as exc:
        GUARD.write(out/'result.json', dict(status='partial_deadline' if isinstance(exc, TimeoutError) else 'failed',
            sample_count=len(samples), error_type=type(exc).__name__, error=str(exc), diagnostic_only=True,
            expected_samples=identities(manifest)))
        raise


def api():
    return SimpleNamespace(PROTOCOL=PROTOCOL, bind=bind, identities=identities, run=run, worker=worker,
        verify_complete=verify_complete, PairedFactory=BASE.PairedFactory)


def worker(*args, **kwargs): return BASE.BASE.worker(*args, experiment=api(), **kwargs)
def main(argv=None): return BASE.BASE.main(argv, experiment=api(), script=__file__)
if __name__ == '__main__': raise SystemExit(main())
