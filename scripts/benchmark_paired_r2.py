#!/usr/bin/env python3
"""Bounded original R2/C256 complete-session three-arm diagnostic; no SPS table."""
import copy
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('paired_r2_shared_runner', ROOT/'scripts/benchmark_paired_r1.py')
BASE = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(BASE)
SHARED, GUARD = BASE.SHARED, BASE.GUARD
CASE_IDS = ('r2-c256',)
ARMS = ('target_only', 'fixed_gamma7_full_shadow', 'full_shadow_zero_admission')
PROTOCOL = copy.deepcopy(BASE.PROTOCOL)
PROTOCOL.update(method='paired_r2_three_arm_finite_family_diagnostic', case_ids=list(CASE_IDS),
    arms=list(ARMS), request_count=2, graph_shapes=[[2, 2], [2, 16]],
    verification_shapes=[[r, b] for r in (2, 1) for b in range(r, 8*r+1)],
    graph_query_lengths=[2, 16], eager_query_lengths=None,
    prefill_query_lengths=[256, 256], slots=2, max_query_tokens=512, max_buckets=25,
    verification_key_capacity=766, maximum_key_length=383,
    pair_order='global replicate ordinal 0..8: rotate canonical three arms left by ordinal modulo3; warmup2,primary5,diagnostic2',
    primary_arm_position_counts={'target_only': [2, 2, 1], 'fixed_gamma7_full_shadow': [1, 2, 2],
                                 'full_shadow_zero_admission': [2, 1, 2]},
    allocation='fixed gamma7 clipped by remaining budget, or zero; chosen before all-seven-position shadow draws',
    setup_validation='each B2/B16 at C256/256 and after committed eager growth to C368/375; pooled and per-request fixed limits, raw outputs before assertions',
    setup_growth_query_lengths=[112, 119], setup_growth_contexts=[368, 375],
    resident_cost='same target, both R2 graph pools and trained draft weights resident for all three arms',
    zero_arm_scope='aggregate full-shadow-path overhead diagnostic; proposal RNG changes trajectories; not isolated drafter latency',
    measured_sps=False, current_confidence_admission=False,
    setup_scope='load/register/prime/two captures, native eager/graph validation and actual context growth outside samples; separately reported')


def selected_manifest(manifest):
    rows = [c for c in manifest['cases'] if c['case_id'] in CASE_IDS]
    if len(rows) != 1 or rows[0]['request_count'] != 2 or rows[0]['prompt_length'] != 256:
        raise ValueError('Original R2/C256 required')
    return dict(manifest, cases=rows)


def schedule(manifest):
    for ordinal, (phase, repeat, case) in enumerate(SHARED.schedule(selected_manifest(manifest))):
        offset = ordinal % len(ARMS)
        for arm in ARMS[offset:]+ARMS[:offset]:
            yield phase, repeat, case, arm


def identities(manifest):
    return [dict(phase=p, repeat=r, case_id=c['case_id'], arm=a) for p, r, c, a in schedule(manifest)]


def baseline_reference(workload_sha, target_fingerprint):
    return BASE.baseline_reference(workload_sha, target_fingerprint, case_ids=CASE_IDS, request_count=2)


def bind(model, generation, checkpoint, workloads=None):
    value = BASE.PROFILE.bind(model, generation, checkpoint, workloads); value.pop('local_requests')
    manifest, sha = SHARED.load_workloads(value['workloads'])
    if manifest['output_tokens'] != 128 or any(manifest['measurement'][k] != v for k, v in PROTOCOL['measurement'].items()):
        raise ValueError('Original 128-output and 2/5/2 repeats required')
    value.update(protocol=PROTOCOL, protocol_sha256=BASE.PROFILE.digest(PROTOCOL),
        selected_case_ids=list(CASE_IDS), expected_samples=identities(manifest),
        frozen_vllm_reference=baseline_reference(sha, value['target_fingerprint']))
    for name in ('scripts/benchmark_paired_r1.py', 'scripts/benchmark_paired_r2.py',
                 'scripts/benchmark_packed_decoder.py', 'tests/test_benchmark_paired_r2.py',
                 'docs/paired-r2-benchmark.md'):
        value['source_sha256'][name] = GUARD.sha(ROOT/name)
    return value


def finite_buckets():
    from dspark_qwen.persistent_target_kv import Bucket, QueryFamily
    from dspark_qwen.target_strategy import FiniteTargetBuckets
    return FiniteTargetBuckets((Bucket((256, 256), (0, 0), 'original R2 admission'),
        Bucket((112, 119), (256, 256), 'setup-only committed context growth')),
        tuple(QueryFamily(r, b, min(8, b-r+1), 383*r, 383,
            'R2 diagnostic declared actual B; only R2B2/R2B16 graphs') for r, b in PROTOCOL['verification_shapes']))


def execution_record(work, q, before, after, device_type, *, active_count):
    shape = [active_count, q]
    if shape not in PROTOCOL['verification_shapes']:
        raise ValueError('Undeclared actual R/B')
    graph = shape in PROTOCOL['graph_shapes']
    expected = ('rocm_full_target_graph' if device_type == 'cuda' else 'cpu_replay_emulator_not_gpu_graph') if graph else 'explicit_eager'
    if work.get('execution_kind') != expected:
        raise ValueError('Declared family execution backend differs')
    deltas = [a-b for a, b in zip(after, before)]
    if not deltas or len(before) != len(after) or any(x != (0 if graph and device_type == 'cuda' else 1) for x in deltas):
        raise ValueError('Actual replay/eager Python counters differ')
    if work['physical_query_tokens'] != q:
        raise ValueError('Physical padding or dropped query rows')
    return dict(execution_kind=expected, query_tokens=q, graph_plan_hit=graph,
        actual_gpu_graph=graph and device_type == 'cuda', python_forward_deltas=deltas, eager_tail=not graph)


class PairedFactory(BASE.PairedFactory):
    def __init__(self, model, draft, **kwargs):
        super().__init__(model, draft, protocol=PROTOCOL, buckets=finite_buckets(), arms=ARMS,
            graph_shapes=tuple(map(tuple, PROTOCOL['graph_shapes'])), **kwargs)

    def validate(self, family, chunks, out, row, *, resident_before=None):
        import torch
        target = self.target; q = family.query_tokens
        resident = resident_before if resident_before is not None else (target.pool.keys.clone(), target.pool.values.clone())
        ws = target.pool._workspaces[family]
        tensors = dict(input_ids=torch.cat(tuple(chunks.values()), dim=1).clone(),
            resident_keys_before=resident[0], resident_values_before=resident[1])
        row.update(status='validating', ordered_requests=list(chunks), query_lengths=[t.shape[1] for t in chunks.values()],
            context_lengths=[target.lengths[r] for r in chunks], limits=PROTOCOL['setup_validation_limits'])
        started = time.perf_counter()
        def save():
            if out is not None: row['tensors'] = BASE.save_setup_tensors(out, q, tensors)
        def snapshot(f, prefix):
            data = dict(context=f.context, final_norm=f.last, logits=target.model.get_output_embeddings()(f.last),
                scratch_keys=target.pool.scratch_keys[:, :q], scratch_values=target.pool.scratch_values[:, :q])
            tensors.update({prefix+'_'+k: v.detach().clone() for k, v in data.items()})
        try:
            for phase in ('eager', 'replay'):
                before = self.witness.snapshot()
                f = (target.verify_eager if phase == 'eager' else target.verify)(chunks, bucket=family)
                after = self.witness.snapshot(); snapshot(f, phase)
                tensors.update({phase+'_'+key: getattr(ws, key).clone() for key in ('positions', 'cu_query', 'cu_key')})
                if phase == 'replay': row['execution'] = execution_record(f.work, q, before, after, target.device.type, active_count=2)
                target.abort(f); target.release_features(f); del f
                row[phase+'_resident_unchanged'] = all(torch.equal(a, b) for a, b in zip(resident, (target.pool.keys, target.pool.values)))
            tensors.update(resident_keys_after=target.pool.keys.clone(), resident_values_after=target.pool.values.clone())
            save()  # Original raw outputs survive failed numerical assertions.
            checks = {}; offset = 0; width = target.model.config.hidden_size
            spans = [('pooled', 0, q)]
            for request, chunk in chunks.items():
                spans.append((request, offset, offset+chunk.shape[1])); offset += chunk.shape[1]
            for label, lo, hi in spans:
                pairs = {f'selected_{layer}': (tensors['replay_context'][:, lo:hi, i*width:(i+1)*width],
                    tensors['eager_context'][:, lo:hi, i*width:(i+1)*width]) for i, layer in enumerate(target.layer_ids)}
                pairs.update({key: (tensors['replay_'+key][:, lo:hi], tensors['eager_'+key][:, lo:hi]) for key in ('final_norm', 'logits')})
                pairs.update({f'{key}_{layer}': (tensors['replay_'+key][layer, lo:hi], tensors['eager_'+key][layer, lo:hi])
                    for key in ('scratch_keys', 'scratch_values') for layer in range(target.pool.layers)})
                checks.update({label+'/'+name: BASE.setup_tensor_comparison(a, b) for name, (a, b) in pairs.items()})
            row['comparisons'] = checks
            if not all(row[phase+'_resident_unchanged'] for phase in ('eager', 'replay')):
                raise AssertionError('Setup abort changed resident KV')
            if not all(x['passed'] for x in checks.values()): raise AssertionError('Setup fixed numerical limits failed')
            row['status'] = 'passed'
        except BaseException as exc:
            row.update(status='failed', error_type=type(exc).__name__, error=str(exc)); save(); raise
        finally: row['setup_validation_seconds'] = time.perf_counter()-started

    def capture(self, manifest, *, deadline=None, progress=None, evidence_dir=None):
        import torch
        target = self.target; rows = []; started = time.perf_counter()
        source = selected_manifest(manifest)['cases'][0]['requests']
        prompts = {x['request']: torch.tensor([x['prompt_token_ids']], dtype=torch.long, device=target.device) for x in source}
        def check():
            if deadline is not None and time.monotonic() >= deadline: raise TimeoutError('Deadline during R2 setup')
        for family in self.buckets.verification:
            if family not in target._graphs: continue
            check(); row = dict(query_tokens=family.query_tokens, status='preparing', validations=[]); rows.append(row)
            if progress: progress(rows)
            target.reset()
            for request in prompts: target.add_request(request)
            f = target.prefill(prompts, bucket=self.buckets.prefill[0]); target.release_features(f); del f
            q = family.query_tokens//2; chunks = {r: ids[:, :q] for r, ids in prompts.items()}
            resident = (target.pool.keys.clone(), target.pool.values.clone())
            row['status'] = 'capturing'
            if progress: progress(rows)
            check(); before = time.perf_counter(); target.capture_graph(chunks, bucket=family); BASE.NATIVE.sync(target.device)
            ex = target._graphs[family]
            row.update(status='captured', capture_seconds=time.perf_counter()-before,
                graph_memory=ex.graph_memory, pointers=ex.pointers(),
                capture_resident_unchanged=all(torch.equal(a, b) for a, b in zip(resident, (target.pool.keys, target.pool.values))))
            if progress: progress(rows)
            try:
                for stage in ('initial', 'grown'):
                    check()
                    if stage == 'grown':
                        growth = {r: ids[:, :n] for (r, ids), n in zip(prompts.items(), PROTOCOL['setup_growth_query_lengths'])}
                        f = target.verify_eager(growth, bucket=self.buckets.prefill[1])
                        target.commit(f, dict(zip(prompts, PROTOCOL['setup_growth_query_lengths'])))
                        target.release_features(f); del f
                    validation = dict(stage=stage); row['validations'].append(validation)
                    directory = Path(evidence_dir)/stage if evidence_dir is not None else None
                    self.validate(family, chunks, directory, validation, resident_before=resident if stage == 'initial' else None)
                    if stage == 'initial': resident = None
                if not row['capture_resident_unchanged']: raise AssertionError('Capture changed resident KV')
                row['pointers_stable'] = ex.pointers() == row['pointers']
                if not row['pointers_stable']: raise AssertionError('Graph I/O pointers moved during setup')
                row['status'] = 'validated'
            except BaseException:
                row['status'] = 'validation_failed'; raise
            finally:
                if progress: progress(rows)
        target.reset(); self.ready = True
        return dict(captures=rows, setup_validation_seconds=time.perf_counter()-started,
            workspace_bytes=target._workspace_bytes, graph_reserved_bytes=target._graph_reserved_bytes,
            resident_and_scratch_bytes=sum(t.numel()*t.element_size() for t in
                (target.pool.keys, target.pool.values, target.pool.scratch_keys, target.pool.scratch_values)),
            buckets=len(target.pool._workspaces), graph_count=len(target._graphs))


def extend_sample(row):
    positions = sum(batch['proposal_positions'] for round_row in row['rounds'] for batch in round_row['work'].get('proposal_batches', []))
    expected = 0 if row['arm'] == 'target_only' else sum(7*len(x['active_requests']) for x in row['rounds'])
    if positions != expected: raise ValueError('Incomplete full seven-position shadow work')
    if row['arm'] == 'full_shadow_zero_admission' and (row['selected_proposal_tokens'] or row['accepted_draft_tokens']):
        raise ValueError('Zero-admission arm admitted a proposal')
    row.update(full_shadow_positions=positions, request_rounds=sum(len(x['active_requests']) for x in row['rounds']),
        selected_proposal_denominator=row['selected_proposal_tokens'],
        full_shadow_denominator=positions, accepted_over_selected=(row['accepted_draft_tokens']/row['selected_proposal_tokens'] if row['selected_proposal_tokens'] else None),
        accepted_over_full_shadow=row['accepted_draft_tokens']/positions if positions else None,
        r1_tail_rounds=sum(len(x['active_requests']) == 1 for x in row['rounds']), measured_sps=False,
        current_confidence_admission=False, zero_arm_scope=PROTOCOL['zero_arm_scope'])


def batch(*args, **kwargs): return BASE.batch(*args, experiment=api(), **kwargs)


def summarize(samples, manifest, baseline=None):
    domain = selected_manifest(manifest)
    aggregates = {a: SHARED.summarize([x for x in samples if x['arm'] == a], domain) for a in ARMS}
    execution = {}
    for arm in ARMS:
        rows = [x for x in samples if x['arm'] == arm and x['phase'] == 'primary']
        cover = [x['execution_coverage'] for x in rows]
        execution[arm] = dict(primary_samples=len(rows), primary_batch_wall_seconds=[x['batch_wall_seconds'] for x in rows],
            totals={k: sum(x[k] for x in cover) for k in sorted({k for x in cover for k in x})},
            full_shadow_positions=sum(x['full_shadow_positions'] for x in rows),
            selected_proposal_tokens=sum(x['selected_proposal_tokens'] for x in rows),
            accepted_draft_tokens=sum(x['accepted_draft_tokens'] for x in rows),
            r1_tail_rounds=sum(x['r1_tail_rounds'] for x in rows))
    rates = {a: aggregates[a][0]['pooled_output_tokens_per_second'] for a in ARMS}
    complete = all(aggregates[a][0]['complete'] for a in ARMS)
    strong = (baseline or {}).get('cells', [{}])[0].get('pooled_output_tokens_per_second')
    return dict(arms=aggregates, primary_execution=execution, complete_primary_triple=complete,
        native_rates=rates, native_over_target_only_rate_ratios={a: rates[a]/rates[ARMS[0]] if complete else None for a in ARMS},
        frozen_vllm_output_tokens_per_second=strong,
        native_over_frozen_vllm_rate_ratios={a: rates[a]/strong if complete and strong else None for a in ARMS},
        primary_arm_position_counts=PROTOCOL['primary_arm_position_counts'],
        zero_arm_scope=PROTOCOL['zero_arm_scope'], measured_sps=False, current_confidence_admission=False)


def verify_complete(samples, manifest, baseline=None):
    if [{k: x[k] for k in ('phase', 'repeat', 'case_id', 'arm')} for x in samples] != identities(manifest):
        raise ValueError('Complete ordered 27-batch R2 triple required')
    return summarize(samples, manifest, baseline)


def api():
    names = ('PROTOCOL', 'CASE_IDS', 'ARMS', 'bind', 'identities', 'schedule', 'batch', 'summarize',
        'verify_complete', 'run', 'worker', 'PairedFactory', 'execution_record', 'extend_sample')
    return SimpleNamespace(**{name: globals()[name] for name in names})


def run(*args, **kwargs): return BASE.run(*args, experiment=api(), **kwargs)
def worker(*args, **kwargs): return BASE.worker(*args, experiment=api(), **kwargs)
def main(argv=None): return BASE.main(argv, experiment=api(), script=__file__)
if __name__ == '__main__': raise SystemExit(main())
