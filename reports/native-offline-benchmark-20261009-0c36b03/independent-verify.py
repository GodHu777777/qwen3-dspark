"""Independent stdlib recomputation of original native fixed-output evidence.
Usage: python3 independent_audit.py PRIVATE_RUN_DIR VLLM_PUBLIC_REPORT_DIR
Does not import the benchmark or implementation under test.
"""
import collections,hashlib,json,math,pathlib,statistics,sys,tarfile
p=pathlib.Path(sys.argv[1]); v=pathlib.Path(sys.argv[2])
read=lambda name:json.loads((p/name).read_text())
lines=lambda path:[json.loads(x) for x in path.read_text().splitlines()]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
stats=lambda x:dict(mean=statistics.mean(x),median=statistics.median(x),min=min(x),max=max(x))
close=lambda a,b:math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-12)
expected=read('expected-source.json'); binding=read('run/binding.json')
assert sha(p/'source.tar')==expected['archive_sha256']
with tarfile.open(p/'source.tar') as archive:
 actual={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive if m.isfile()}
 assert actual==expected['files_sha256'] and len(actual)==307
 manifest=json.loads(archive.extractfile('configs/performance-workloads.example.json').read())
 assert hashlib.sha256(archive.extractfile('configs/performance-workloads.example.json').read()).hexdigest()==expected['workload_sha256']
assert binding==read('prepared-binding.json')
assert binding['protocol_sha256']==expected['protocol_sha256']
assert binding['workloads_sha256']==expected['workload_sha256']
assert binding['protocol']['timeout_seconds']==1800
cases=manifest['cases']; by_case={c['case_id']:c for c in cases}
assert [(c['request_count'],c['prompt_length']) for c in cases]==[(r,c) for r in (1,2,4) for c in (64,256)]
for c in cases:
 for i,r in enumerate(c['requests']):
  assert r['request']==f'r{i}' and r['seed']==20261009+i
  assert r['prompt_token_ids']==[100+((20261009+53*i+17*j)%1000) for j in range(c['prompt_length'])]
schedule=[('warmup',j,c['case_id']) for c in cases for j in range(2)]
for j in range(5):schedule += [('primary',j,c['case_id']) for c in cases[j:]+cases[:j]]
schedule += [('diagnostic',j,c['case_id']) for j in range(2) for c in cases]
samples=lines(p/'run/worker/samples.jsonl')
assert [(s['phase'],s['repeat'],s['case_id']) for s in samples]==schedule
for declared in [binding['expected_samples'],read('run/worker/specification.json')['expected_samples']]:
 assert [(s['phase'],s['repeat'],s['case_id']) for s in declared]==schedule
round_count=0;request_round_count=0;zero_prefix_rounds=0;inactive_resident_rounds=0
for s in samples:
 c=by_case[s['case_id']]; names=[r['request'] for r in c['requests']]; R=c['request_count']; C=c['prompt_length']
 assert s['request_count']==R and s['prompt_length']==C and s['output_tokens']==128*R
 assert s['admission_output_tokens']==R and s['committed_round_output_tokens']==127*R
 assert [r['request'] for r in s['requests']]==names
 assert math.isfinite(s['batch_wall_seconds']) and s['batch_wall_seconds']>0
 assert not s['capacity_scheduler_integrated'] and not s['speculative_verification_sps']
 assert s['pure_prefill_seconds'] is None
 assert s['retained_resident_requests']==R
 assert s['final_context_lengths']=={r:C+127 for r in names}
 assert s['whole_operation_peak_allocated_bytes']<=6*1024**3
 for r in s['requests']:
  assert r['output_tokens']==128 and len(r['output_sha256'])==64
  if s['phase']!='diagnostic':assert r['ttft_seconds'] is None and r['completion_latency_seconds'] is None
  else:assert 0<r['ttft_seconds']<=r['completion_latency_seconds']<=s['batch_wall_seconds']
 emitted={r:1 for r in names}; accepted=proposed=queries=0
 for row in s['rounds']:
  active=[r for r in names if emitted[r]<128]
  assert row['active_requests']==active and set(row['requests'])==set(active)
  assert row['allocation']=={r:max(0,min(7,128-emitted[r]-1)) for r in active}
  assert row['resident_requests']==R
  assert row['actual_logical_b']==len(active)+sum(row['allocation'].values())
  assert row['actual_physical_b']==row['actual_logical_b']
  assert row['work']['target']['physical_query_tokens']==row['actual_physical_b']
  assert len(row['work']['proposal_batches'])==1
  shadow=row['work']['proposal_batches'][0]
  assert shadow['proposal_positions']==7*len(active) and shadow['backbone_calls']==1
  assert shadow['markov_projection_calls']==7 and shadow['confidence_head_calls']==7
  assert shadow['probability_policy']=='float64_softmax_normalize_cdf_v1'
  for r,d in row['requests'].items():
   n=row['allocation'][r]
   assert d['proposed']==n and 0<=d['accepted']<=n
   assert d['committed']==d['accepted']+1
   assert d['cache_before']==C+emitted[r]-1 and d['cache_after']==d['cache_before']+d['committed']
   emitted[r]+=d['committed']; assert emitted[r]<=128
   assert (d['stop_reason']=='budget')==(emitted[r]==128)
   accepted+=d['accepted'];proposed+=d['proposed'];request_round_count+=1
  queries+=row['actual_logical_b'];round_count+=1
  zero_prefix_rounds+=any(n==0 for n in row['allocation'].values())
  inactive_resident_rounds+=len(active)<R
 assert emitted=={r:128 for r in names}
 assert accepted==s['accepted_draft_tokens'] and proposed==s['selected_proposal_tokens']
 assert queries==s['target_verification_query_rows']
result=read('run/worker/result.json'); assert result['status']=='completed' and result['sample_count']==54
assert result['prior_target_numerical_gate']=='failed_unchanged' and not result['whole_system_pass_claimed']
vsource=json.loads((v/'source-identity.json').read_text())
assert vsource['workload_sha256']==binding['workloads_sha256']
assert vsource['model_tokenizer_sha256']==binding['target_fingerprint']
assert sha(v/'scalar-samples.jsonl')==vsource['evidence_sha256']['execution/worker/samples.jsonl']
vrows=lines(v/'scalar-samples.jsonl'); vmetrics=json.loads((v/'primary-metrics.json').read_text())
assert [(s['phase'],s['repeat'],s['case_id']) for s in vrows]==schedule
primary=[]; diagnostic=[];comparison=[]
for c,agg,vm in zip(cases,result['aggregates'],vmetrics['cells']):
 cell=[s for s in samples if s['phase']=='primary' and s['case_id']==c['case_id']]
 times=[s['batch_wall_seconds'] for s in cell];tokens=sum(s['output_tokens'] for s in cell);rate=tokens/sum(times)
 assert agg['case_id']==vm['case_id']==c['case_id'] and agg['complete'] and agg['completed_samples']==5
 assert close(rate,agg['pooled_output_tokens_per_second'])
 for k,value in stats(times).items():assert close(value,agg[f'batch_latency_{k}_seconds'])
 vc=[s for s in vrows if s['phase']=='primary' and s['case_id']==c['case_id']]
 vt=sum(s['output_tokens'] for s in vc);vs=sum(s['batch_wall_seconds'] for s in vc);vrate=vt/vs
 assert vt==tokens and close(vrate,vm['pooled_output_tokens_per_second'])
 primary.append(dict(case_id=c['case_id'],request_count=c['request_count'],prompt_length=c['prompt_length'],repeats=5,batch_wall_seconds=times,total_output_tokens=tokens,total_batch_wall_seconds=sum(times),pooled_output_tokens_per_second=rate,batch_wall_summary_seconds=stats(times),round_counts=[len(s['rounds']) for s in cell],accepted_draft_tokens=[s['accepted_draft_tokens'] for s in cell],selected_proposal_tokens=[s['selected_proposal_tokens'] for s in cell],ttft_seconds=None,per_request_completion_latency_seconds=None))
 reqs=[r for s in samples if s['phase']=='diagnostic' and s['case_id']==c['case_id'] for r in s['requests']]
 diagnostic.append(dict(case_id=c['case_id'],batches=2,request_observations=len(reqs),ttft_seconds=stats([r['ttft_seconds'] for r in reqs]),completion_latency_seconds=stats([r['completion_latency_seconds'] for r in reqs]),pure_prefill_seconds=None))
 hashes=[tuple(r['output_sha256'] for r in s['requests']) for s in samples if s['case_id']==c['case_id']]
 comparison.append(dict(case_id=c['case_id'],native_output_tokens_per_second=rate,vllm_output_tokens_per_second=vrate,native_over_vllm_throughput=rate/vrate,native_over_vllm_batch_time=sum(times)/vs,native_unique_output_hash_vectors=len(set(hashes))))
controller=read('run/supervision/controller-result.json'); release=read('independent-release.json'); launch=read('run/supervision/launch.json')
assert controller['status']=='completed' and controller['released'] and not controller['timed_out']
assert not controller['remaining_owned'] and not controller['cleanup_actions']
assert controller['worker_os_exit']==release['worker_os_exit']==release['controller_os_exit']==int((p/'controller.os-exit-code').read_text())==0
assert read('run/supervision/worker-exit.json')==dict(returncode=0,timed_out=False)
assert read('run/post-input-integrity.json')['passed'] and read('run/worker/post-input-integrity.json')['passed']
assert all(r['current'] is None for r in release['process_checks']) and release['shell_current'] is None
assert {r['expected']['pid'] for r in release['process_checks']}=={r['pid'] for r in controller['owned_processes']}|{launch['controller']['pid']}
pre=read('preparation.json'); post=read('post-inputs-and-release.json')
for key in ('source_commit','archive_sha256','archive_files_verified','source_sha256','protocol_sha256','workload_sha256','checkpoint_weights_sha256','target_fingerprint'):assert pre[key]==post[key]
health=lines(p/'run/supervision/health.jsonl')
for state in [pre['state'],post['state'],release]+health:
 assert state['asr']['ready'] and not state['asr']['busy']
 assert state['asr_identity']['pid']==expected['preserved_asr']['pid'] and state['asr_identity']['start_ticks']==expected['preserved_asr']['start_ticks']
assert pre['state']['vram']==post['state']['vram']==release['vram']
assert release['kfd_owners']==post['state']['kfd_owners']==[expected['preserved_asr']['pid']]
print(json.dumps(dict(passed=True,source_commit=expected['source_commit'],archive_sha256=sha(p/'source.tar'),sample_sha256=sha(p/'run/worker/samples.jsonl'),sample_count=54,phase_counts=dict(collections.Counter(s['phase'] for s in samples)),round_count=round_count,request_round_count=request_round_count,zero_prefix_rounds=zero_prefix_rounds,inactive_resident_rounds=inactive_resident_rounds,primary=primary,diagnostic=diagnostic,comparison=comparison,health_samples=len(health),supervisor_elapsed_seconds=controller['elapsed_seconds'],maximum_sampled_global_vram_used_bytes=max(h['vram'][0]['used_bytes'] for h in health),maximum_whole_operation_peak_allocated_bytes=max(s['whole_operation_peak_allocated_bytes'] for s in samples),maximum_whole_operation_peak_reserved_bytes=max(s['whole_operation_peak_reserved_bytes'] for s in samples),checks=['307 archive file hashes match frozen expected identities','54 exact schedule identities, synthetic input recipe, request order and 128 output budget','Every round active/retained roster, pre-draw max prefix, full-seven shadow, actual B, commit/cache accounting reconstructed','All six native and vLLM primary pooled throughputs independently recomputed from original samples','Diagnostic common-start host clock bounds, primary null latency fields','Worker/controller OS0, no timeout/cleanup and independently absent owned/controller/shell processes','Preserved ASR identity/health and VRAM returned to baseline; post-run source/input verification intact'],limitations=['Native whole-Qwen RMS numerical gate remains failed; no distribution equivalence','Native eager fixed maximum prefix versus vLLM graph-enabled target only; different execution and sampling paths','Fixed synthetic batch throughput, not serving arrival frontier or speculative verification SPS','Native diagnostic common batch start differs from vLLM individual add_request starts']),indent=2))
