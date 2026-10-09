"""Shared exact synthetic inputs and measurement-order helpers; stdlib only."""
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_workloads(path):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    if manifest['schema'] != 'dspark-shared-performance-workloads-v1':
        raise ValueError('Unsupported workload schema')
    if manifest['chat_template_applied'] or manifest['prefix_caching'] or manifest['output_tokens'] != 128:
        raise ValueError('Exact untemplated uncached 128-output-token workload required')
    expected = dict(temperature=1., top_p=1., top_k=0, min_p=0., ignore_eos=True,
        repetition_penalty=1., presence_penalty=0., frequency_penalty=0.)
    if manifest['sampling'] != expected:
        raise ValueError('Sampling protocol changed')
    cases = manifest['cases']
    if {(c['request_count'], c['prompt_length']) for c in cases} != {(r,c) for r in (1,2,4) for c in (64,256)} or len(cases) != 6:
        raise ValueError('Complete fixed R/context domain required')
    if len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Unique case IDs required')
    for case in cases:
        if len(case['requests']) != case['request_count']:
            raise ValueError('Request count mismatch')
        names = set()
        for request in case['requests']:
            ids = request['prompt_token_ids']
            if request['request'] in names or not request['request']:
                raise ValueError('Unique nonempty request IDs required')
            names.add(request['request'])
            if len(ids) != case['prompt_length'] or any(type(t) is not int or not 0 <= t < manifest['model']['vocab_size'] for t in ids):
                raise ValueError('Exact prompt length and vocabulary required')
            if type(request['seed']) is not int or not 0 <= request['seed'] < 2**32:
                raise ValueError('Explicit uint32 per-request seed required')
    measurement = manifest['measurement']
    for key in ('warmup_batches_per_case','measured_batches_per_case','diagnostic_batches_per_case'):
        if type(measurement[key]) is not int or measurement[key] < 1:
            raise ValueError('Positive fixed warmup/repeat counts required')
    if measurement['engine_loads_per_run'] != 1:
        raise ValueError('One engine load per run required')
    if measurement['primary_order'] != 'replicate-major; rotate canonical case order left by replicate index' or measurement['diagnostic_order'] != 'replicate-major canonical order after all primary samples':
        raise ValueError('Measurement ordering changed')
    local = manifest['local_native_profile']
    if (local['request_count'],local['resident_request_count'],local['committed_context_vector'],local['gamma']) != (2,2,[128,128],7):
        raise ValueError('Local native profile domain changed')
    if sorted(map(tuple,local['prefix_lengths'])) != [(a,b) for a in range(8) for b in range(8)]:
        raise ValueError('Full64 allocation domain required')
    if (local['snapshot_source_case_id'],local['snapshot_source_request_order'],local['snapshot_prompt_prefix_tokens']) != ('r2-c256',['r0','r1'],128):
        raise ValueError('Local snapshot source changed')
    return manifest, hashlib.sha256(raw).hexdigest()


def schedule(manifest):
    cases = manifest['cases']; m = manifest['measurement']
    for case in cases:
        for repeat in range(m['warmup_batches_per_case']):
            yield 'warmup', repeat, case
    for repeat in range(m['measured_batches_per_case']):
        order = cases[repeat % len(cases):] + cases[:repeat % len(cases)]
        for case in order:
            yield 'primary', repeat, case
    for repeat in range(m['diagnostic_batches_per_case']):
        for case in cases:
            yield 'diagnostic', repeat, case


def summarize(samples, manifest):
    """Retain all samples; pooled completed tokens / pooled batch wall time."""
    expected_cases = {case['case_id']: case for case in manifest['cases']}
    phase_limits = dict(warmup=manifest['measurement']['warmup_batches_per_case'],
        primary=manifest['measurement']['measured_batches_per_case'],
        diagnostic=manifest['measurement']['diagnostic_batches_per_case'])
    seen = set()
    for sample in samples:
        key = (sample['phase'], sample['repeat'], sample['case_id'])
        if key in seen or sample['phase'] not in phase_limits or sample['case_id'] not in expected_cases:
            raise ValueError('Duplicate or unknown sample identity')
        seen.add(key)
        if type(sample['repeat']) is not int or not 0 <= sample['repeat'] < phase_limits[sample['phase']]:
            raise ValueError('Repeat outside frozen domain')
        if sample['output_tokens'] != expected_cases[sample['case_id']]['request_count'] * manifest['output_tokens']:
            raise ValueError('Incomplete fixed output count')
        if not math.isfinite(sample['batch_wall_seconds']) or sample['batch_wall_seconds'] <= 0:
            raise ValueError('Finite positive batch timings required')
    result = []
    for case in manifest['cases']:
        rows = [s for s in samples if s['phase']=='primary' and s['case_id']==case['case_id']]
        times = [s['batch_wall_seconds'] for s in rows]
        if any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError('Finite positive batch timings required')
        result.append(dict(case_id=case['case_id'], completed_samples=len(rows),
            expected_samples=manifest['measurement']['measured_batches_per_case'],
            complete=len(rows)==manifest['measurement']['measured_batches_per_case'],
            pooled_output_tokens_per_second=sum(s['output_tokens'] for s in rows)/sum(times) if times else None,
            batch_latency_mean_seconds=statistics.mean(times) if times else None,
            batch_latency_median_seconds=statistics.median(times) if times else None,
            batch_latency_min_seconds=min(times) if times else None,
            batch_latency_max_seconds=max(times) if times else None,
            note='Offline fixed-batch wall time; not per-request latency, spec SPS, or an arrival-load frontier'))
    return result


def expected_sample_identities(manifest):
    return [dict(phase=phase, repeat=repeat, case_id=case['case_id'])
        for phase, repeat, case in schedule(manifest)]


def verify_complete_samples(samples, manifest):
    actual = [{key: sample[key] for key in ('phase', 'repeat', 'case_id')} for sample in samples]
    if actual != expected_sample_identities(manifest):
        raise ValueError('Full ordered warmup/primary/diagnostic sample domain incomplete')
    return summarize(samples, manifest)
