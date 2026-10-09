#!/usr/bin/env python3
"""Complete R1 paired sessions with two explicit target graphs and eager tails.

Default: stdlib fingerprint binding only. This is a two-case attribution
experiment, not the six-case panel, all-graph execution or a new sampling law.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('paired_r1_native_helpers', ROOT/'scripts/benchmark_packed_decoder.py')
NATIVE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NATIVE)
SHARED, GUARD, PROFILE = NATIVE.SHARED, NATIVE.GUARD, NATIVE.PROFILE
ARMS = ('target_only', 'fixed_gamma7_full_shadow')
CASE_IDS = ('r1-c64', 'r1-c256')
BASELINE = ROOT/'reports/vllm-offline-benchmark-20261009-700bfa6'
PROTOCOL = dict(version=1, method='paired_r1_two_graphs_and_explicit_eager_tails',
    case_ids=list(CASE_IDS), arms=list(ARMS), output_tokens=128, gamma=7,
    allocation='max(0,min(7,remaining_output_budget-1)) chosen before full shadow',
    proposal_mode='shadow; all seven positions even when allocation is zero',
    graph_query_lengths=[1, 8], eager_query_lengths=[2, 3, 4, 5, 6, 7],
    prefill_query_lengths=[64, 256], verification_key_capacity=383,
    slots=1, context_capacity=384, max_query_tokens=256, max_buckets=10,
    max_graph_buckets=2, graph_reservation_bytes=512*1024**2,
    graph_byte_budget=1280*1024**2, workspace_byte_budget=64*1024**2,
    temperature=1., eos_ids=[], probability_policy='float64_softmax_normalize_cdf_v1',
    timeout_seconds=1800, cooperative_reserve_seconds=10,
    minimum_free_bytes=8*1024**3, process_allocation_cap_bytes=6*1024**3,
    model_loads=1, draft_loads=1, selected_layers='same trained draft layer IDs in both arms',
    setup_validation_limits=dict(atol=.02, rtol=.02, max_rms=.005),
    setup_validation='once per Q1/Q8: native eager versus first replay, raw selected/final norm/logits/all-layer scratch KV; resident unchanged',
    measurement=dict(warmup_batches_per_case=2, measured_batches_per_case=5,
                     diagnostic_batches_per_case=2),
    pair_order='original two-case schedule; within each pair canonical arms for even repeat, reversed for odd repeat',
    setup_scope='load/register/prime/two captures outside batch timings; reported separately',
    resident_cost='same target, both graph pools and trained draft weights remain resident in both arms',
    capacity_scheduler_integrated=False, overlap=False, all_graph=False,
    full_six_case_panel=False, distribution_equivalence_claimed=False)


def selected_manifest(manifest):
    """Derive a view only after the original six-case manifest was validated."""
    cases = {c['case_id']: c for c in manifest['cases']}
    if not all(c in cases and cases[c]['request_count'] == 1 for c in CASE_IDS):
        raise ValueError('Both declared R1 cases required')
    return dict(manifest, cases=[cases[c] for c in CASE_IDS])


def schedule(manifest):
    for phase, repeat, case in SHARED.schedule(selected_manifest(manifest)):
        for arm in (ARMS if repeat % 2 == 0 else ARMS[::-1]):
            yield phase, repeat, case, arm


def identities(manifest):
    return [dict(phase=p, repeat=r, case_id=c['case_id'], arm=a)
            for p, r, c, a in schedule(manifest)]


def baseline_reference(workload_sha, target_fingerprint, *, case_ids=CASE_IDS, request_count=1):
    """Bind frozen strong baseline and recompute its matched primary rates."""
    source = json.loads((BASELINE/'source-identity.json').read_text())
    if (source['workload_sha256'] != workload_sha or
            any(source['model_tokenizer_sha256'].get(k) != v for k, v in target_fingerprint.items())):
        raise ValueError('Frozen vLLM reference workload/model identity differs')
    raw = [json.loads(line) for line in (BASELINE/'scalar-samples.jsonl').read_text().splitlines()]
    published = {c['case_id']: c for c in json.loads((BASELINE/'primary-metrics.json').read_text())['cells']}
    cells = []
    for case_id in case_ids:
        rows = [x for x in raw if x['phase'] == 'primary' and x['case_id'] == case_id]
        if [x['repeat'] for x in rows] != list(range(5)) or any(x['output_tokens'] != 128*request_count for x in rows):
            raise ValueError('Frozen vLLM five-repeat R1 panel incomplete')
        seconds = [x['batch_wall_seconds'] for x in rows]
        if any(not math.isfinite(x) or x <= 0 for x in seconds):
            raise ValueError('Invalid frozen baseline duration')
        rate = sum(x['output_tokens'] for x in rows)/sum(seconds)
        if rate != published[case_id]['pooled_output_tokens_per_second']:
            raise ValueError('Frozen vLLM raw/public aggregate mismatch')
        cells.append(dict(case_id=case_id, batch_wall_seconds=seconds,
                          pooled_output_tokens_per_second=rate, completed_samples=5))
    return dict(source_commit=source['source_commit'], workload_sha256=workload_sha,
        files_sha256={str((BASELINE/n).relative_to(ROOT)): GUARD.sha(BASELINE/n)
            for n in ('source-identity.json', 'scalar-samples.jsonl', 'primary-metrics.json')},
        cells=cells, comparison_scope='Frozen strong execution-stack baseline; different FP64/native implementation, not target-law equivalence')


def bind(model, generation, checkpoint, workloads=None):
    value = PROFILE.bind(model, generation, checkpoint, workloads)
    value.pop('local_requests')
    manifest, workload_sha = SHARED.load_workloads(value['workloads'])
    if (manifest['output_tokens'] != PROTOCOL['output_tokens'] or
            any(manifest['measurement'][k] != v for k, v in PROTOCOL['measurement'].items())):
        raise ValueError('Paired protocol requires original 128 outputs and 2/5/2 repeats')
    value.update(protocol=PROTOCOL, protocol_sha256=PROFILE.digest(PROTOCOL),
        selected_case_ids=list(CASE_IDS), expected_samples=identities(manifest),
        frozen_vllm_reference=baseline_reference(workload_sha, value['target_fingerprint']))
    for name in ('scripts/benchmark_paired_r1.py', 'scripts/benchmark_packed_decoder.py',
                 'tests/test_benchmark_paired_r1.py', 'docs/paired-r1-benchmark.md'):
        value['source_sha256'][name] = GUARD.sha(ROOT/name)
    return value


def finite_buckets():
    from dspark_qwen.persistent_target_kv import Bucket
    from dspark_qwen.target_strategy import FiniteTargetBuckets
    return FiniteTargetBuckets(
        tuple(Bucket((q,), (0,), 'paired R1 exact shared prompt') for q in PROTOCOL['prefill_query_lengths']),
        tuple(Bucket((q,), (383-q,), 'paired R1 full output128; explicit graph Q1/Q8 else eager') for q in range(1, 9)))


class ForwardWitness:
    """Small Python counters distinguish actual replay from emulator/eager calls."""
    def __init__(self, model):
        self.counts = [0]*(1+len(model.model.layers))
        def hook(index):
            def count(*args): self.counts[index] += 1
            return count
        self.hooks = [model.model.register_forward_pre_hook(hook(0))]
        self.hooks += [layer.register_forward_hook(hook(i+1)) for i, layer in enumerate(model.model.layers)]

    def snapshot(self): return tuple(self.counts)

    def close(self):
        for hook in self.hooks: hook.remove()


def setup_tensor_comparison(actual, expected):
    import torch
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    row = dict(finite=finite, passed=False)
    if finite:
        difference = (actual.double()-expected.double()).abs()
        limits = PROTOCOL['setup_validation_limits']
        rms = float(difference.square().mean().sqrt())
        row.update(max_abs=float(difference.max()), rms=rms,
            bit_equal=torch.equal(actual, expected),
            passed=bool((difference <= limits['atol']+limits['rtol']*expected.double().abs()).all())
                and rms <= limits['max_rms'])
    return row


def save_setup_tensors(directory, q, tensors):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'q{q}-eager-first-replay.pt'
    torch.save({k: v.detach().cpu().clone() for k, v in tensors.items()}, path)
    return dict(file=str(path.name), sha256=GUARD.sha(path))


class PairedFactory:
    """One real persistent target/loaded draft; production and CPU-test factory."""
    def __init__(self, model, draft, *, target_builder=None, draft_builder=None,
                 protocol=None, buckets=None, arms=None, graph_shapes=None):
        from dspark_qwen.persistent_qwen_target import PersistentQwenTarget
        from dspark_qwen.persistent_qwen_graph import TorchGraphBackend
        from dspark_qwen.target_strategy import PersistentTargetStrategy
        from dspark_qwen.rocm_varlen import BACKEND
        from dspark_qwen.packed_draft import PackedDraft, DRAFT_BACKEND
        if draft.spec.block_size != 7:
            raise ValueError('Complete seven-position trained shadow required')
        if target_builder is None:
            target_builder = lambda m, ids, **kw: PersistentQwenTarget(m, ids, native_backend=BACKEND, **kw)
        self.draft_builder = draft_builder or (lambda d: PackedDraft(d, native_backend=DRAFT_BACKEND))
        self.draft = draft
        self.protocol = PROTOCOL if protocol is None else protocol
        self.arms = ARMS if arms is None else arms
        self.buckets = finite_buckets() if buckets is None else buckets
        self.target = target_builder(model, draft.spec.layer_ids, graph_backend=TorchGraphBackend(),
            **{k: self.protocol[k] for k in ('slots', 'context_capacity', 'max_query_tokens', 'max_buckets',
                'max_graph_buckets', 'graph_byte_budget', 'workspace_byte_budget')})
        PersistentTargetStrategy(self.target, self.buckets)  # Register exactly ten finite workspaces.
        for bucket in self.buckets.verification:
            selected = (bucket.query_tokens in PROTOCOL['graph_query_lengths'] if graph_shapes is None
                        else (bucket.request_count, bucket.query_tokens) in graph_shapes)
            if selected:
                self.target.register_graph_bucket(bucket, reserve_bytes=self.protocol['graph_reservation_bytes'])
        self.witness = ForwardWitness(model)
        self.ready = False

    def validate_first_replay(self, bucket, chunks, resident_before, report, evidence_dir=None):
        """Setup-only new-shape check in this E2E process, never a timed sample."""
        import torch
        target = self.target; q = bucket.query_tokens
        tensors = dict(input_ids=next(iter(chunks.values())).detach().clone(),
            positions=target.pool._workspaces[bucket].positions.clone(),
            cu_query=target.pool._workspaces[bucket].cu_query.clone(),
            cu_key=target.pool._workspaces[bucket].cu_key.clone(),
            resident_keys_before=resident_before[0], resident_values_before=resident_before[1])
        def snapshot(features, prefix):
            values = dict(context=features.context, final_norm=features.last,
                logits=target.model.get_output_embeddings()(features.last),
                scratch_keys=target.pool.scratch_keys[:, :q], scratch_values=target.pool.scratch_values[:, :q])
            tensors.update({prefix+'_'+k: v.detach().clone() for k, v in values.items()})
        def unchanged():
            return all(torch.equal(a, b) for a, b in zip(resident_before, (target.pool.keys, target.pool.values)))
        started = time.perf_counter()
        report.update(status='validating', limits=PROTOCOL['setup_validation_limits'],
                      resident_unchanged_after_capture=unchanged())
        try:
            eager = target.verify_eager(chunks, bucket=bucket)
            snapshot(eager, 'eager')
            target.abort(eager); target.release_features(eager)
            del eager
            report['resident_unchanged_after_eager_abort'] = unchanged()
            before = self.witness.snapshot()
            actual = target.verify(chunks, bucket=bucket)
            after = self.witness.snapshot()
            snapshot(actual, 'replay')
            work = dict(actual.work)
            target.abort(actual); target.release_features(actual)
            del actual
            report['resident_unchanged_after_replay_abort'] = unchanged()
            tensors.update(resident_keys_after=target.pool.keys.detach().clone(),
                           resident_values_after=target.pool.values.detach().clone())
            # Save original outputs before numerical or replay-counter assertions.
            if evidence_dir is not None:
                report['tensors'] = save_setup_tensors(evidence_dir, q, tensors)
            report['execution'] = execution_record(work, q, before, after, target.device.type)
            width = target.model.config.hidden_size
            comparisons = {}
            for index, layer in enumerate(target.layer_ids):
                comparisons[f'selected_raw_layer_{layer}'] = setup_tensor_comparison(
                    tensors['replay_context'][:, :, index*width:(index+1)*width],
                    tensors['eager_context'][:, :, index*width:(index+1)*width])
            for key in ('final_norm', 'logits'):
                comparisons[key] = setup_tensor_comparison(tensors['replay_'+key], tensors['eager_'+key])
            for layer in range(target.pool.layers):
                for key in ('scratch_keys', 'scratch_values'):
                    comparisons[f'{key}_layer_{layer}'] = setup_tensor_comparison(
                        tensors['replay_'+key][layer], tensors['eager_'+key][layer])
            report['comparisons'] = comparisons
            if not all(report[k] for k in ('resident_unchanged_after_capture',
                    'resident_unchanged_after_eager_abort', 'resident_unchanged_after_replay_abort')):
                raise AssertionError('Setup capture/eager/replay changed resident KV')
            if not all(x['passed'] for x in comparisons.values()):
                raise AssertionError('Q1/Q8 setup eager/first-replay fixed numerical limits failed')
            report['status'] = 'passed'
        except BaseException as exc:
            report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            if evidence_dir is not None and 'tensors' not in report:
                try: report['tensors'] = save_setup_tensors(evidence_dir, q, tensors)
                except BaseException as write_error: report['artifact_write_error'] = repr(write_error)
            raise
        finally:
            report['setup_validation_seconds'] = time.perf_counter()-started

    def capture(self, manifest, *, deadline=None, progress=None, evidence_dir=None):
        import torch
        target = self.target
        source = next(c for c in manifest['cases'] if c['case_id'] == 'r1-c256')['requests'][0]
        captures = []
        for bucket in self.buckets.verification:
            if bucket not in target._graphs: continue
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Cooperative deadline during paired graph setup')
            row = dict(query_tokens=bucket.query_tokens, status='preparing')
            captures.append(row)
            if progress: progress(captures)
            target.reset()
            target.add_request('capture')
            prompt = torch.tensor([source['prompt_token_ids']], device=target.device, dtype=torch.long)
            features = target.prefill({'capture': prompt}, bucket=self.buckets.prefill[1])
            target.release_features(features)
            del features
            row['status'] = 'capturing'
            if progress: progress(captures)
            resident_before = (target.pool.keys.clone(), target.pool.values.clone())
            started = time.perf_counter()
            target.capture_graph({'capture': prompt[:, :bucket.query_tokens]}, bucket=bucket)
            NATIVE.sync(target.device)
            ex = target._graphs[bucket]
            row.update(status='captured', capture_seconds=time.perf_counter()-started,
                graph_memory=ex.graph_memory, pointers=ex.pointers())
            if progress: progress(captures)
            row['validation'] = {}
            try:
                self.validate_first_replay(bucket, {'capture': prompt[:, :bucket.query_tokens]},
                    resident_before, row['validation'], evidence_dir=evidence_dir)
                row['status'] = 'validated'
            except BaseException:
                row['status'] = 'validation_failed'
                raise
            finally:
                del resident_before
                if progress: progress(captures)
        target.reset()
        self.ready = True
        return dict(captures=captures,
            setup_validation_seconds=sum(x['validation']['setup_validation_seconds'] for x in captures),
            workspace_bytes=target._workspace_bytes,
            graph_reserved_bytes=target._graph_reserved_bytes,
            resident_and_scratch_bytes=sum(t.numel()*t.element_size() for t in
                (target.pool.keys, target.pool.values, target.pool.scratch_keys, target.pool.scratch_values)),
            buckets=len(target.pool._workspaces), graph_count=len(target._graphs))

    def __call__(self, arm):
        from dspark_qwen.packed_sampling import PackedSpeculativeSession
        from dspark_qwen.packed_target_sampling import PackedTargetOnlySession
        from dspark_qwen.target_strategy import PersistentTargetStrategy
        if not self.ready or arm not in self.arms:
            raise ValueError('Both explicit captures and a declared arm required')
        self.target.reset()  # Charged inside every batch, including after the other arm.
        strategy = PersistentTargetStrategy(self.target, self.buckets)
        options = dict(target_strategy=strategy, temperature=1., eos_ids=())
        if arm == 'target_only':
            return PackedTargetOnlySession(self.target, **options)
        return PackedSpeculativeSession(self.target, self.draft_builder(self.draft), amp=True, **options)

    def close(self):
        self.witness.close()
        self.target.close()


def execution_record(target_work, q, before, after, device_type):
    deltas = [a-b for a, b in zip(after, before)]
    registered = q in PROTOCOL['graph_query_lengths']
    kind = target_work.get('execution_kind')
    if registered:
        expected = 'rocm_full_target_graph' if device_type == 'cuda' else 'cpu_replay_emulator_not_gpu_graph'
        if kind != expected:
            raise ValueError('Declared captured Q did not execute its explicit backend')
        calls = 0 if device_type == 'cuda' else 1
    else:
        if q not in PROTOCOL['eager_query_lengths'] or kind is not None:
            raise ValueError('Unexpected eager tail or undeclared graph')
        kind = 'native_eager_tail' if device_type == 'cuda' else 'cpu_eager_tail'
        calls = 1
    if not deltas or any(n != calls for n in deltas):
        raise ValueError('Actual replay/eager model and decoder Python counters differ')
    if target_work['physical_query_tokens'] != q:
        raise ValueError('Physical query padding or dropped queries detected')
    return dict(execution_kind=kind, query_tokens=q, graph_plan_hit=registered,
        actual_gpu_graph=kind == 'rocm_full_target_graph', python_forward_deltas=deltas,
        eager_tail=not registered)


def batch(factory, device, case, manifest, arm, *, diagnostic=False, deadline=None, experiment=None):
    import torch
    from dspark_qwen.packed_sampling import RequestSpec
    from dspark_qwen.tensor_sampling import TensorRandom
    cases, arms = (CASE_IDS, ARMS) if experiment is None else (experiment.CASE_IDS, experiment.ARMS)
    count = 1 if experiment is None else experiment.PROTOCOL['request_count']
    if case['case_id'] not in cases or case['request_count'] != count or arm not in arms:
        raise ValueError('Declared R1 pair required')
    prepared = time.perf_counter()
    specs = {r['request']: RequestSpec(torch.tensor([r['prompt_token_ids']], dtype=torch.long, device=device),
        manifest['output_tokens'], TensorRandom(torch.Generator(device=device).manual_seed(r['seed']))) for r in case['requests']}
    NATIVE.sync(device)
    preparation_seconds = time.perf_counter()-prepared
    memory = {}
    if device.type == 'cuda':
        memory = dict(pre_reset_allocated_bytes=torch.cuda.memory_allocated(device),
                      pre_reset_reserved_bytes=torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
    rounds, first, finished = [], {}, {}
    NATIVE.sync(device)
    started = time.perf_counter()
    session = factory(arm)
    admitted = session.admit(specs)
    if diagnostic:
        observed = time.perf_counter()-started
        first = {r: observed for r in admitted}
        finished = {r: observed for r, s in session.requests.items() if s['finished']}
    while any(not s['finished'] for s in session.requests.values()):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Cooperative paired deadline; incomplete batch is not a sample')
        active = [r for r, s in session.requests.items() if not s['finished']]
        before = factory.witness.snapshot()
        if arm != 'target_only':
            allocations = {r: (0 if arm == 'full_shadow_zero_admission' else
                max(0, min(7, session.requests[r]['budget']-len(session.requests[r]['output'])-1))) for r in active}
            policy = NATIVE.FixedPrefix(session, allocations)
            issued = session.propose(active, mode='shadow')
            result = session.verify_commit(issued.proposals, allocations, allocation_policy=policy)
            del issued
        else:
            allocations = {r: 0 for r in active}
            result = session.step(active)
        q = len(active)+sum(allocations.values())
        if experiment is None:
            execution = execution_record(result['work']['target'], q, before, factory.witness.snapshot(), device.type)
        else:
            execution = experiment.execution_record(result['work']['target'], q, before,
                factory.witness.snapshot(), device.type, active_count=len(active))
        rounds.append(dict(active_requests=active, allocation=allocations, requests=result['requests'],
            work=result['work'], actual_logical_b=q, actual_physical_b=result['work']['target']['physical_query_tokens'],
            resident_requests=len(session.requests), **execution))
        if diagnostic:
            observed = time.perf_counter()-started
            for r in active:
                if session.requests[r]['finished']: finished[r] = observed
        del result  # Do not retain full-vocabulary q/p across rounds.
    NATIVE.sync(device)
    elapsed = time.perf_counter()-started
    outputs = session.outputs()
    records = []
    for request in case['requests']:
        r = request['request']; tokens = outputs[r]; state = session.requests[r]
        if len(tokens) != manifest['output_tokens'] or not state['finished'] or state['budget'] != len(tokens):
            raise ValueError('Incomplete paired output budget')
        records.append(dict(request=r, output_tokens=len(tokens), output_sha256=SHARED.sha_json(tokens),
            ttft_seconds=first[r] if diagnostic else None,
            completion_latency_seconds=finished[r] if diagnostic else None,
            latency_scope='host batched admission/round return' if diagnostic else 'unsupported_in_primary'))
    admission = sum(len(x) for x in admitted.values())
    committed = sum(x['committed'] for row in rounds for x in row['requests'].values())
    output_count = sum(x['output_tokens'] for x in records)
    if admission+committed != output_count:
        raise ValueError('Admission and complete-round output accounting differs')
    coverage = dict(verification_rounds=len(rounds), verification_query_rows=sum(x['query_tokens'] for x in rounds),
        graph_plan_rounds=sum(x['graph_plan_hit'] for x in rounds),
        graph_plan_query_rows=sum(x['query_tokens'] for x in rounds if x['graph_plan_hit']),
        actual_gpu_graph_rounds=sum(x['actual_gpu_graph'] for x in rounds),
        actual_gpu_graph_query_rows=sum(x['query_tokens'] for x in rounds if x['actual_gpu_graph']),
        eager_tail_rounds=sum(x['eager_tail'] for x in rounds),
        eager_tail_query_rows=sum(x['query_tokens'] for x in rounds if x['eager_tail']),
        prefill_eager_calls=1, prefill_eager_rows=case['prompt_length']*count)
    row = dict(arm=arm, batch_wall_seconds=elapsed, output_tokens=output_count, requests=records,
        preparation_seconds=preparation_seconds, rounds=rounds, execution_coverage=coverage,
        admission_output_tokens=admission, committed_round_output_tokens=committed,
        prefill_work=session.last_prefill_work, final_context_lengths=dict(session.target.lengths),
        accepted_draft_tokens=sum(x.get('accepted', 0) for row in rounds for x in row['requests'].values()),
        selected_proposal_tokens=sum(x.get('proposed', 0) for row in rounds for x in row['requests'].values()),
        selected_layer_ids=list(session.target.layer_ids), probability_policy=PROTOCOL['probability_policy'],
        time_scope='fresh session/reset, admission/first draw, complete native graph/eager rounds, full shadow for speculative arm, original FP64 validation/sampling, commit/projection/release and final synchronization',
        memory_scope='same loaded target/draft and both graph pools; pre-reset through completion including prepared inputs and inherited resident KV',
        all_graph=False, full_six_case_panel=False, capacity_scheduler_integrated=False)
    if experiment is not None:
        experiment.extend_sample(row)
    if device.type == 'cuda':
        row.update(**memory, whole_operation_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                   whole_operation_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return row


def summarize(samples, manifest, baseline=None):
    """Each arm has an independent complete two-case panel; no missing-arm pooling."""
    domain = selected_manifest(manifest)
    aggregates = {arm: SHARED.summarize([x for x in samples if x['arm'] == arm], domain) for arm in ARMS}
    execution = {}
    for arm in ARMS:
        execution[arm] = []
        for case_id in CASE_IDS:
            rows = [x for x in samples if x['arm'] == arm and x['case_id'] == case_id and x['phase'] == 'primary']
            coverage = [x['execution_coverage'] for x in rows if 'execution_coverage' in x]
            keys = sorted({k for x in coverage for k in x})
            execution[arm].append(dict(case_id=case_id, primary_samples=len(rows),
                primary_batch_wall_seconds=[x['batch_wall_seconds'] for x in rows],
                coverage_samples=len(coverage), totals={k: sum(x[k] for x in coverage) for k in keys}))
    reference = {x['case_id']: x for x in (baseline or {}).get('cells', [])}
    comparisons = []
    for case_id in CASE_IDS:
        by_arm = {arm: next(x for x in aggregates[arm] if x['case_id'] == case_id) for arm in ARMS}
        rates = {a: x['pooled_output_tokens_per_second'] for a, x in by_arm.items()}
        complete = all(x['complete'] for x in by_arm.values())
        strong = reference.get(case_id, {}).get('pooled_output_tokens_per_second')
        comparisons.append(dict(case_id=case_id, complete_primary_pair=complete, native_rates=rates,
            speculative_over_target_only_rate_ratio=rates[ARMS[1]]/rates[ARMS[0]] if complete else None,
            frozen_vllm_output_tokens_per_second=strong,
            native_over_frozen_vllm_rate_ratios={a: rates[a]/strong if complete and strong else None for a in ARMS}))
    return dict(arms=aggregates, primary_execution=execution, paired_comparisons=comparisons)


def verify_complete(samples, manifest, baseline=None):
    expected = identities(manifest)
    actual = [{k: x[k] for k in ('phase', 'repeat', 'case_id', 'arm')} for x in samples]
    if actual != expected:
        raise ValueError('Complete ordered 36-batch paired panel required')
    return summarize(samples, manifest, baseline)


def run(factory, device, manifest, out, *, deadline=None, setup=None, baseline=None, experiment=None):
    api = runner_api() if experiment is None else experiment
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'samples.jsonl').exists(): raise ValueError('Fresh paired sample output required')
    samples = []
    specification = dict(protocol=api.PROTOCOL, expected_samples=api.identities(manifest),
        frozen_vllm_reference=baseline, full_six_case_panel=False)
    GUARD.write(out/'specification.json', specification)
    if setup is not None: GUARD.write(out/'runtime.json', setup)
    GUARD.write(out/'result.json', dict(status='running', sample_count=0))
    try:
        for phase, repeat, case, arm in api.schedule(manifest):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Cooperative deadline before next paired batch')
            sample = api.batch(factory, device, case, manifest, arm, diagnostic=phase == 'diagnostic', deadline=deadline)
            sample.update(phase=phase, repeat=repeat, case_id=case['case_id'], request_count=case['request_count'], prompt_length=case['prompt_length'])
            samples.append(sample)
            with (out/'samples.jsonl').open('a') as stream:
                stream.write(json.dumps(sample, allow_nan=False)+'\n')
        result = dict(status='completed', sample_count=len(samples), aggregates=api.verify_complete(samples, manifest, baseline),
            all_graph=False, full_six_case_panel=False, capacity_scheduler_integrated=False,
            prior_target_numerical_gate='failed_unchanged', distribution_equivalence_claimed=False)
        GUARD.write(out/'result.json', result)
        return result
    except BaseException as exc:
        GUARD.write(out/'result.json', dict(status='partial_deadline' if isinstance(exc, TimeoutError) else 'failed',
            error_type=type(exc).__name__, error=str(exc), sample_count=len(samples),
            aggregates=api.summarize(samples, manifest, baseline), expected_samples=specification['expected_samples'],
            all_graph=False, full_six_case_panel=False, distribution_equivalence_claimed=False))
        raise


def worker(binding_path, out, *, experiment=None):
    api = runner_api() if experiment is None else experiment
    started = time.monotonic()
    deadline = started+api.PROTOCOL['timeout_seconds']-api.PROTOCOL['cooperative_reserve_seconds']
    binding = json.loads(Path(binding_path).read_text())
    if api.bind(binding['model'], binding['generation_manifest'], binding['checkpoint'], binding['workloads']) != binding:
        raise ValueError('Frozen paired source/input changed before loading')
    manifest, workload_sha = SHARED.load_workloads(binding['workloads'])
    import torch
    import transformers
    sys.path.insert(0, str(ROOT))
    from transformers import AutoModelForCausalLM
    from dspark_qwen.checkpoint import load_checkpoint
    from dspark_qwen.config import DraftConfig
    from dspark_qwen.model import DSparkDraft
    from dspark_qwen.rocm_varlen import PinnedRocmVarlenKernel
    if transformers.__version__ != '5.17.0' or not torch.cuda.is_available():
        raise RuntimeError('Pinned GPU runtime required')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    runtime = PinnedRocmVarlenKernel(device).runtime
    free, total = torch.cuda.mem_get_info()
    if free < api.PROTOCOL['minimum_free_bytes']: raise RuntimeError('8 GiB free guard failed')
    torch.cuda.set_per_process_memory_fraction(api.PROTOCOL['process_allocation_cap_bytes']/total)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    GUARD.write(out/'result.json', dict(status='running', stage='model_load', sample_count=0))
    model = AutoModelForCausalLM.from_pretrained(binding['model'], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    draft = DSparkDraft(model, DraftConfig.from_dict(binding['draft_config'])).to(device).eval()
    load_checkpoint(binding['checkpoint'], draft)
    factory = api.PairedFactory(model, draft)
    def progress(captures):
        GUARD.write(out/'capture-progress.json', dict(status='preparing', captures=captures))
    capture = factory.capture(manifest, deadline=deadline, progress=progress, evidence_dir=out/'setup-validation')
    GUARD.write(out/'capture-progress.json', dict(status='completed', **capture))
    setup = dict(runtime=runtime, setup_seconds=time.monotonic()-started, capture=capture,
        model_loads=1, draft_loads=1, workload_sha256=workload_sha,
        torch=torch.__version__, transformers=transformers.__version__,
        shared_resident_cost=api.PROTOCOL['resident_cost'], runtime_scope='actual_pretrained_trained_GPU')
    result = api.run(factory, device, manifest, out, deadline=deadline, setup=setup,
                 baseline=binding['frozen_vllm_reference'])
    if api.bind(binding['model'], binding['generation_manifest'], binding['checkpoint'], binding['workloads']) != binding:
        raise ValueError('Frozen paired source/input changed during benchmark')
    GUARD.write(out/'post-input-integrity.json', dict(passed=True))
    factory.close()
    return 0 if result['status'] == 'completed' else 1


def runner_api():
    """Explicit runner seams; defaults resolve live globals for existing observers."""
    from types import SimpleNamespace
    names = ('PROTOCOL', 'bind', 'identities', 'schedule', 'batch', 'summarize',
             'verify_complete', 'run', 'worker', 'PairedFactory')
    return SimpleNamespace(**{name: globals()[name] for name in names})


def main(argv=None, *, experiment=None, script=None):
    api = runner_api() if experiment is None else experiment
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'generation-manifest', 'checkpoint', 'workloads', 'asr-url'):
        p.add_argument('--'+name)
    p.add_argument('--output', type=Path)
    p.add_argument('--asr-pid', type=int)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--worker-binding', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker_binding:
        try:
            return api.worker(args.worker_binding, args.output)
        except BaseException as exc:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output/'error.txt').write_text(traceback.format_exc())
            path = args.output/'result.json'
            result = json.loads(path.read_text()) if path.exists() else dict(sample_count=0)
            if result.get('status') not in ('failed', 'partial_deadline'):
                result.update(status='partial_deadline' if isinstance(exc, TimeoutError) else 'failed',
                              error_type=type(exc).__name__, error=str(exc))
            if hasattr(exc, 'memory_accounting'):
                result['memory_accounting'] = exc.memory_accounting
                GUARD.write(args.output/'failed-graph-pool-snapshot.json', exc.pool_snapshot)
            GUARD.write(path, result)
            return 1
    if not all((args.model, args.generation_manifest, args.checkpoint)):
        p.error('Actual pinned model, generation manifest and trained step1280 checkpoint required')
    binding = api.bind(args.model, args.generation_manifest, args.checkpoint, args.workloads)
    if not args.execute:
        print(json.dumps(dict(status='dry_run_no_backend_import_no_gpu', binding=binding), indent=2))
        return 0
    if args.dry_run or args.output is None or args.asr_pid is None or not args.asr_url:
        p.error('Execute requires fresh output, ASR identity/health and no dry-run')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    GUARD.write(out/'binding.json', binding)
    guard = GUARD.ASRGuard(args.asr_pid, args.asr_url, pre_free_bytes=api.PROTOCOL['minimum_free_bytes'])
    command = [sys.executable, '-u', str(Path(script or __file__).resolve()), '--worker-binding', str(out/'binding.json'),
               '--output', str(out/'worker')]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    env.pop('PYTHONPATH', None)
    status = GUARD.supervise(command, out/'supervision', guard, timeout=api.PROTOCOL['timeout_seconds'], env=env)
    intact = api.bind(args.model, args.generation_manifest, args.checkpoint, args.workloads) == binding
    GUARD.write(out/'post-input-integrity.json', dict(passed=intact))
    if not intact or status['status'] != 'completed' or not status['released']: return 1
    manifest, _ = SHARED.load_workloads(binding['workloads'])
    samples = [json.loads(line) for line in (out/'worker/samples.jsonl').read_text().splitlines()]
    recomputed = api.verify_complete(samples, manifest, binding['frozen_vllm_reference'])
    result = json.loads((out/'worker/result.json').read_text())
    if recomputed != result['aggregates'] or result['sample_count'] != len(binding['expected_samples']):
        raise ValueError('Controller paired raw sample/aggregate mismatch')
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
