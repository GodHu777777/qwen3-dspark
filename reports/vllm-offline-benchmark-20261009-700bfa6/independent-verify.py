"""Recompute saved scalar evidence without importing benchmark implementation.
Usage: python3 independent-verify.py /path/to/private/run-evidence
Requires the original source.tar, binding, lifecycle and worker evidence.
"""
import collections
import hashlib
import json
import math
import pathlib
import statistics
import sys
import tarfile

p = pathlib.Path(sys.argv[1])
def read(name):
    return json.loads((p / name).read_text())
def lines(name):
    return [json.loads(s) for s in (p / name).read_text().splitlines()]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values), min=min(values), max=max(values))
prep = read('preparation-summary.json')
binding = read('binding.json')
assert sha(p / 'binding.json') == prep['binding_sha256']
assert sha(p / 'source.tar') == prep['archive_sha256']
with tarfile.open(p / 'source.tar') as archive:
    members = [m for m in archive if m.isfile()]
    assert len(members) == prep['source_files'] == 288
    for member in members:
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == prep['source_sha256'][member.name]
    manifest_bytes = archive.extractfile('configs/performance-workloads.example.json').read()
manifest = json.loads(manifest_bytes)
workload_hash = hashlib.sha256(manifest_bytes).hexdigest()
assert workload_hash == prep['request_sha256']
assert len(binding['files']) == prep['binding_files'] == 3456
cases = manifest['cases']
assert [(c['request_count'], c['prompt_length']) for c in cases] == [(r,c) for r in (1,2,4) for c in (64,256)]
for c in cases:
    for i, req in enumerate(c['requests']):
        assert req['request'] == f'r{i}' and req['seed'] == 20261009+i
        assert req['prompt_token_ids'] == [100+((20261009+53*i+17*j)%1000) for j in range(c['prompt_length'])]
expected = [('warmup', j, c['case_id']) for c in cases for j in range(2)]
for j in range(5):
    expected += [('primary', j, c['case_id']) for c in cases[j:]+cases[:j]]
expected += [('diagnostic', j, c['case_id']) for j in range(2) for c in cases]
samples = lines('execution/worker/samples.jsonl')
assert [(s['phase'],s['repeat'],s['case_id']) for s in samples] == expected
spec = read('execution/worker/specification.json')
assert [(s['phase'],s['repeat'],s['case_id']) for s in spec['expected_samples']] == expected
assert spec['workload_sha256'] == workload_hash
by_case = {c['case_id']:c for c in cases}
for s in samples:
    c = by_case[s['case_id']]
    assert math.isfinite(s['batch_wall_seconds']) and s['batch_wall_seconds'] > 0
    assert s['request_count'] == c['request_count'] == len(s['requests'])
    assert s['prompt_length'] == c['prompt_length']
    assert s['output_tokens'] == 128*c['request_count']
    assert [r['request'] for r in s['requests']] == [r['request'] for r in c['requests']]
    for r in s['requests']:
        assert r['output_tokens'] == 128
        if s['phase'] != 'diagnostic':
            assert r['ttft_seconds'] is None and r['completion_latency_seconds'] is None
        else:
            assert all(math.isfinite(r[k]) for k in ('ttft_seconds','completion_latency_seconds','submitted_offset_seconds','post_first_to_finish_seconds'))
            assert 0 <= r['submitted_offset_seconds']
            assert 0 < r['ttft_seconds'] <= r['completion_latency_seconds']
            assert r['submitted_offset_seconds'] + r['completion_latency_seconds'] <= s['batch_wall_seconds'] + 1e-9
            assert math.isclose(r['completion_latency_seconds'] - r['ttft_seconds'], r['post_first_to_finish_seconds'], abs_tol=1e-9)
            assert s['pure_prefill_seconds'] is None
result = read('execution/worker/result.json')
assert result['status'] == 'completed' and result['sample_count'] == 54
primary, diagnostic = [], []
for c, recorded in zip(cases,result['aggregates']):
    cell = [s for s in samples if s['phase']=='primary' and s['case_id']==c['case_id']]
    times = [s['batch_wall_seconds'] for s in cell]
    tokens = sum(s['output_tokens'] for s in cell)
    rate = tokens/sum(times)
    assert recorded['case_id'] == c['case_id'] and recorded['complete'] and len(cell)==5
    assert math.isclose(rate, recorded['pooled_output_tokens_per_second'], rel_tol=1e-14)
    st = stats(times)
    for k,v in st.items():
        assert math.isclose(v,recorded[f'batch_latency_{k}_seconds'],rel_tol=1e-14)
    primary.append(dict(case_id=c['case_id'], request_count=c['request_count'], prompt_length=c['prompt_length'], repeats=5, batch_wall_seconds=times, total_output_tokens=tokens, total_batch_wall_seconds=sum(times), pooled_output_tokens_per_second=rate, batch_wall_summary_seconds=st, per_request_completion_latency_seconds=None, ttft_seconds=None))
    reqs = [r for s in samples if s['phase']=='diagnostic' and s['case_id']==c['case_id'] for r in s['requests']]
    diagnostic.append(dict(case_id=c['case_id'], diagnostic_batches=2, request_observations=len(reqs), ttft_seconds=stats([r['ttft_seconds'] for r in reqs]), completion_latency_seconds=stats([r['completion_latency_seconds'] for r in reqs]), pure_prefill_seconds=None))
controller = read('execution/controller-result.json')
worker = read('execution/worker-exit.json')
release = read('independent-release.json')
post = read('execution/post-input-integrity.json')
assert controller['status']=='completed' and controller['worker_os_exit']==0 and controller['released']
assert not controller['timed_out'] and not controller['cleanup_actions'] and not controller['remaining_owned']
assert worker == {'returncode':0,'timed_out':False}
assert int((p/'controller-os-exit').read_text()) == release['controller_os_exit'] == release['worker_os_exit'] == 0
assert post['passed'] and post['binding_sha256'] == prep['binding_sha256'] and post['workload_sha256'] == workload_hash
assert len(release['owned_pids'])==7 and not any(release['owned_pid_exists'].values())
pre = read('execution/prestate.json')
end = read('execution/poststate.json')
health = lines('execution/health.jsonl')
assert pre['asr_identity']['pid'] == end['asr_identity']['pid'] == binding['asr_pid']
assert pre['asr_identity']['start_ticks'] == end['asr_identity']['start_ticks']
assert release['kfd_owners'] == pre['kfd_owners'] == [binding['asr_pid']]
assert release['vram'] == pre['vram']
assert all(h['asr']['ready'] and not h['asr']['busy'] and h['asr_identity']['start_ticks']==pre['asr_identity']['start_ticks'] for h in health)
assert release['asr']['ready'] and not release['asr']['busy']
root = read('root-scalar-verification.json')
assert root['source_archive_git_exact_match'] and root['sample_sha256'] == sha(p/'execution/worker/samples.jsonl')
assert all(math.isclose(root['rates'][v['case_id']],v['pooled_output_tokens_per_second'],rel_tol=1e-14) for v in primary)
print(json.dumps(dict(passed=True, source_commit=prep['source_commit'], workload_sha256=workload_hash, sample_count=len(samples), phase_counts=dict(collections.Counter(s['phase'] for s in samples)), total_output_tokens=sum(s['output_tokens'] for s in samples), primary_output_tokens=sum(s['output_tokens'] for s in samples if s['phase']=='primary'), primary=primary, diagnostic=diagnostic, health_samples=len(health), maximum_sampled_global_vram_used_bytes=max(h['vram'][0]['used_bytes'] for h in health), checks=['Archive members match preparation hashes; root separately matched git archive bytes','54 exact ordered phase/repeat/case identities match independent enumeration and specification','Materialized tokens and seeds match fixed recipe','Every request has 128 output tokens and expected identity; all batch times positive finite','Primary request latency and TTFT remain null','Diagnostic request-specific clocks obey first-output/completion/batch bounds','All six pooled rates and mean/median/min/max independently match worker; rates also match root','Worker/controller actual OS exits zero; timeout false and supervisor cleanup empty','Seven observed owned PIDs absent in saved independent release','Preserved ASR identity/health and sampled global VRAM restored','Saved post-run all-bound-file integrity passed']), indent=2))
