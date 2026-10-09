"""Independent CPU-only tensor audit; never imports the probe or backend.
Usage: python independent_tensor_audit.py PRIVATE_EVIDENCE_DIR
"""
import collections,hashlib,json,math,pathlib,sys,tarfile
import torch
p=pathlib.Path(sys.argv[1]); w=p/'run/worker'
read=lambda path:json.loads((p/path).read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def tensor(name):return torch.load(w/name,map_location='cpu',weights_only=True)
report=read('run/worker/result.json');binding=read('run/binding.json');expected=read('expected-source.json')
assert report['status']=='completed' and report['stages']==dict(real_qkv='passed',eager_tail='passed',capture_replay='passed')
assert report['runtime_scope']=='real_pretrained_gpu'
assert not report['whole_model_graph'] and not report['overlap'] and not report['t_minus_two_capacity_graph']
variants=['exact_actual_max','exact_fixed_max','capacity_zero','capacity_finite','capacity_nan']
order=[dict(stage='eager_tail',case=i,variant=v) for i in range(3) for v in variants]+[dict(stage='capture_replay',case=i,variant='replay') for i in range(3)]
assert report['expected_observations']==order
assert [dict(stage=r['stage'],case=r['case'],variant=r.get('variant','replay')) for r in report['observations']]==order
protocol=read('run/worker/protocol.json');assert protocol==binding['protocol']
assert protocol['comparisons']['atol']==.02 and protocol['comparisons']['rtol']==.02 and protocol['comparisons']['max_rms']==.005
assert sha(p/'source.tar')==expected['archive_sha256']
with tarfile.open(p/'source.tar') as t:
 hashes={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t if m.isfile()}
 assert hashes==expected['files_sha256'] and len(hashes)==324
 manifest=json.loads(t.extractfile('configs/performance-workloads.example.json').read())
assert expected['workload_sha256']==binding['workload_sha256']==hashes['configs/performance-workloads.example.json']
assert expected['protocol_sha256']==binding['protocol_sha256']==hashlib.sha256(json.dumps(protocol,sort_keys=True,allow_nan=False).encode()).hexdigest()
assert read('prepared-binding.json')==binding
case=next(c for c in manifest['cases'] if c['case_id']=='r2-c256');assert binding['requests']==case['requests']
for i,r in enumerate(case['requests']):assert r['prompt_token_ids']==[100+((20261009+53*i+17*j)%1000) for j in range(256)] and r['seed']==20261009+i
contexts=[(128,128),(129,131),(130,135)];commits=[(1,3),(1,4),(0,0)];ql=[1,4]
checks=0; comparisons=[]
def same(a,b):
 global checks
 assert a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
 assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8));checks+=1

def comparison(actual,reference,bitwise):
 assert actual.shape==reference.shape==(5,16,128) and actual.dtype==reference.dtype==torch.bfloat16
 assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
 rows=[]
 for name,lo,hi in [('pooled',0,5),('0',0,1),('1',1,5)]:
  a=actual[lo:hi];r=reference[lo:hi];delta=(a.double()-r.double()).abs()
  mx=delta.max().item();rms=delta.square().mean().sqrt().item();equal=torch.equal(a,r)
  ok=equal if bitwise else bool((delta<=.02+.02*r.double().abs()).all()) and rms<=.005
  rows.append(dict(request=name,finite=True,max_abs=mx,rms=rms,bit_equal=equal,passed=ok))
 return dict(passed=all(x['passed'] for x in rows),bitwise_required=bitwise,rows=rows)

def compare_record(actual,ref,record,bitwise,label):
 computed=comparison(actual,ref,bitwise)
 assert computed==record and computed['passed']
 assert torch.equal(actual.contiguous().view(torch.uint8),ref.contiguous().view(torch.uint8))
 comparisons.append(dict(observation=label,**computed))

prefixes=None; exact_inputs=[];outputs=[];capacity_outputs=[]
for i,cs in enumerate(contexts):
 captured=[tensor(f'real-qkv-{i}-request-{j}.pt') for j in range(2)]
 for j,(r,c,n) in enumerate(zip(captured,cs,ql)):
  shapes=dict(query=(n,16,128),prefix_k=(c,8,128),prefix_v=(c,8,128),new_k=(n,8,128),new_v=(n,8,128))
  assert set(r)==set(shapes)
  for name,shape in shapes.items():
   assert tuple(r[name].shape)==shape and r[name].dtype==torch.bfloat16 and r[name].device.type=='cpu' and torch.isfinite(r[name]).all()
 if prefixes is None:prefixes=[(r['prefix_k'],r['prefix_v']) for r in captured]
 assert [len(k) for k,v in prefixes]==list(cs)
 q=torch.cat([r['query'] for r in captured]);k=torch.cat([torch.cat((old[0],r['new_k'])) for old,r in zip(prefixes,captured)]);v=torch.cat([torch.cat((old[1],r['new_v'])) for old,r in zip(prefixes,captured)])
 N=sum(cs)+5;cq=torch.tensor([0,1,5],dtype=torch.int32);ck=torch.tensor([0,cs[0]+1,N],dtype=torch.int32);pos=torch.tensor([cs[0],*range(cs[1],cs[1]+4)],dtype=torch.long)
 exact_inputs.append(dict(q=q,k=k,v=v,cu_query=cq,cu_key=ck,positions=pos))
 values={}
 for j,variant in enumerate(variants):
  inp=tensor(f'eager-{i}-{variant}-inputs.pt');out=tensor(f'eager-{i}-{variant}-output.pt')['output'];row=report['observations'][5*i+j]
  assert set(inp)==set(exact_inputs[-1]); same(inp['q'],q);same(inp['cu_query'],cq);same(inp['cu_key'],ck);same(inp['positions'],pos)
  same(inp['k'][:N],k);same(inp['v'][:N],v)
  capacity=variant.startswith('capacity');assert len(inp['k'])==len(inp['v'])==(293 if capacity else N)
  if variant=='capacity_zero':assert torch.count_nonzero(inp['k'][N:])==torch.count_nonzero(inp['v'][N:])==0
  if variant=='capacity_finite':assert (inp['k'][N:]==100).all() and (inp['v'][N:]==-100).all()
  if variant=='capacity_nan':assert torch.isnan(inp['k'][N:]).all() and torch.isnan(inp['v'][N:]).all()
  ow=row['operator_work'];assert ow==dict(physical_q_rows=5,physical_k_rows=len(inp['k']),logical_k_rows=N,padding_k_rows=len(inp['k'])-N,max_q=4,max_k=max(c+n for c,n in zip(cs,ql)) if variant=='exact_actual_max' else 148)
  work=row['work'];assert work['logical_active_key_tokens']==N and work['physical_key_capacity']==293 and work['padded_key_capacity']==293-N
  assert work['gather_rows_per_kv_per_layer']==586 and work['staging_output_rows_per_layer']==293
  assert work['allowed_causal_pairs']==sum(n*c+n*(n+1)//2 for c,n in zip(cs,ql))
  assert row['committed_unchanged_before_fixture_commit']
  values[variant]=out;ref=values['capacity_zero'] if variant in ('capacity_finite','capacity_nan') else values['exact_actual_max']
  compare_record(out,ref,row['comparison'],variant in ('capacity_finite','capacity_nan'),f'eager-{i}-{variant}')
 outputs.append(values['exact_actual_max']);capacity_outputs.append(values['capacity_zero'])
 prefixes=[(torch.cat((old[0],r['new_k'][:n])),torch.cat((old[1],r['new_v'][:n]))) for old,r,n in zip(prefixes,captured,commits[i])]
first_addresses=None;first_output=None
for i in range(3):
 inp=tensor(f'replay-{i}-inputs.pt');out=tensor(f'replay-{i}-output.pt')['output'];row=report['observations'][15+i]
 for name,x in exact_inputs[i].items():same(inp[{'k':'exact_k','v':'exact_v'}.get(name,name)],x)
 compare_record(out,outputs[i],row['comparison'],False,f'replay-{i}-exact')
 compare_record(out,capacity_outputs[i],row['eager_capacity_comparison'],True,f'replay-{i}-capacity')
 assert row['positions']==exact_inputs[i]['positions'].tolist() and row['cu_key']==exact_inputs[i]['cu_key'].tolist()
 assert row['stable_input_pointers'] and row['stable_output_pointer'] and row['committed_unchanged_before_fixture_commit']
 assert len(row['input_pointers'])==18 and len(set(row['input_pointers'].values()))==18
 assert all(type(x) is int and x>0 for x in row['input_pointers'].values())
 if first_addresses is None:first_addresses=row['input_pointers'];first_output=row['output_pointer']
 assert row['input_pointers']==first_addresses and row['output_pointer']==first_output
assert len(list(w.glob('*.pt')))==42
ctrl=read('run/supervision/controller-result.json');release=read('independent-release.json')
assert ctrl['status']=='completed' and ctrl['released'] and not ctrl['timed_out'] and not ctrl['cleanup_actions'] and not ctrl['remaining_owned']
assert ctrl['worker_os_exit']==release['worker_os_exit']==release['controller_os_exit']==int((p/'controller.os-exit-code').read_text())==0
assert read('run/supervision/worker-exit.json')==dict(returncode=0,timed_out=False)
assert read('run/post-input-integrity.json')['passed'] and read('run/worker/post-input-integrity.json')['passed']
assert all(x['current'] is None for x in release['process_checks']) and release['shell_current'] is None
pre=read('preparation.json');post=read('post-inputs-and-release.json')
for name in ('source_commit','archive_sha256','archive_files_verified','source_sha256','protocol_sha256','workload_sha256','target_fingerprint'):assert pre[name]==post[name]
health=[json.loads(x) for x in (p/'run/supervision/health.jsonl').read_text().splitlines()]
for h in health+[pre['state'],post['state'],release]:
 assert h['asr']['ready'] and not h['asr']['busy']
 assert h['asr_identity']['pid']==expected['preserved_asr']['pid'] and h['asr_identity']['start_ticks']==expected['preserved_asr']['start_ticks']
assert pre['state']['vram']==post['state']['vram']==release['vram']
assert pre['state']['kfd_owners']==post['state']['kfd_owners']==release['kfd_owners']==[expected['preserved_asr']['pid']]
print(json.dumps(dict(passed=True,audit_runtime_torch=torch.__version__,audit_device='cpu',source_commit=expected['source_commit'],result_sha256=sha(w/'result.json'),observations=18,tensor_artifacts=42,exact_input_tensor_comparisons=checks,comparison_groups=len(comparisons),pooled_and_per_request_comparisons=sum(len(x['rows']) for x in comparisons),max_abs=max(r['max_abs'] for c in comparisons for r in c['rows']),max_rms=max(r['rms'] for c in comparisons for r in c['rows']),all_compared_output_storage_bytes_equal=True,all_compared_outputs_bit_equal=all(r['bit_equal'] for c in comparisons for r in c['rows']),recorded_input_addresses=18,stable_recorded_output_address=True,logical_active_k=[261,265,270],capacity_k=293,tail_rows=[32,28,23],health_samples=len(health),supervisor_elapsed_seconds=ctrl['elapsed_seconds'],maximum_sampled_global_vram_used_bytes=max(h['vram'][0]['used_bytes'] for h in health),comparisons=comparisons,artifact_sha256={f.name:sha(f) for f in sorted(w.glob('*.pt'))},checks=['324 source archive files and exact shared token recipe verified','42 original CPU-loaded tensor artifacts checked for real QKV shape/dtype/finite data','Cross-context committed-prefix K and V reconstructed from initial bytes and artificial selected new rows; all saved eager and replay active QKV match exactly','Actual cumulative lengths, positions, physical/logical K and all zero/finite/NaN tails inspected','All 21 comparison groups and 63 pooled/request records independently recomputed from raw output bytes','18 ordered observations complete and runtime K/V isolation/pointer flags true; actual saved output pointer constant','OS exits0, no timeout/cleanup, independent process release, preserved ASR and source/input/VRAM identity'],limits=['Pre-commit full resident K/V snapshots were not saved: their whole-pool isolation is checked by executed dual-K/V assertions, not independently recomputed offline','Saved input pointer dictionary is the initial snapshot; current input stability is an executed runtime comparison flag, while every output pointer is directly recorded','Graph covers gather+native attention only; QKV projections/RoPE/full model/commit/sampling/scheduler overlap excluded','No throughput or speedup measurement; earlier whole-Qwen numerical failure remains unchanged']),indent=2))
