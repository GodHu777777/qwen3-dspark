#!/usr/bin/env python3
"""Bounded real-Qwen first-layer native capacity-tail/capture capability probe.

Default: stdlib binding only. GPU execution needs a separate coordinated window.
Captures gather + attention only; pretrained QKV/RoPE and commits remain outside.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
BASE=module('capacity_graph_binding',ROOT/'scripts/probe_qwen_varlen.py')
GUARD=module('capacity_graph_guard',ROOT/'scripts/guard_vllm_smoke.py')
SHARED=module('capacity_graph_workloads',ROOT/'dspark_qwen/performance_workloads.py')
PROTOCOL=dict(version=1,scope='Real pretrained Qwen first-layer QKV; gather+native attention graph only',
    query_lengths=[1,4],context_ceilings=[144,144],contexts=[[128,128],[129,131],[130,135]],
    committed_query_rows=[[1,3],[1,4],[0,0]],commit_semantics='Artificial KV transaction fixture; not sampled acceptance events',
    source_case='r2-c256',source_layer=0,physical_query_tokens=5,physical_key_capacity=293,
    maximum_query_length=4,maximum_key_length=148,
    comparisons=dict(atol=.02,rtol=.02,max_rms=.005,scope='Pooled and each request; native exact-length reference'),
    replay_reference='Native exact-length numerical limits plus bit-identical eager zero-tail capacity reference',
    tail_poison=dict(finite_keys=100.,finite_values=-100.,nonfinite='NaN keys and values; required unread-tail isolation gate',
        equality='zero/finite/NaN capacity outputs must be bit-identical and finite'),
    warmup_calls=2,replays=3,timeout_seconds=300,cooperative_reserve_seconds=10,
    minimum_free_bytes=8*1024**3,process_allocation_cap_bytes=6*1024**3,
    capture_boundary='Persistent gathers + native attention; excludes QKV projections/RoPE, host metadata, commit, sampling and full model',
    whole_model_graph=False,overlap=False,t_minus_two_capacity_graph=False)

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()

def bind(model,generation,workloads=None):
    b=BASE.bind(model,generation)
    path=Path(workloads or ROOT/'configs/performance-workloads.example.json').resolve();m,h=SHARED.load_workloads(path)
    if m['model']['config_sha256']!=b['target_fingerprint']['config.json']:raise ValueError('Shared model mismatch')
    case=next(c for c in m['cases'] if c['case_id']==PROTOCOL['source_case'])
    b.update(protocol=PROTOCOL,protocol_sha256=digest(PROTOCOL),workloads=str(path),workload_sha256=h,
        requests=case['requests'])
    for name in ('scripts/probe_native_capacity_graph.py','scripts/guard_vllm_smoke.py','dspark_qwen/performance_workloads.py'):
        b['source_sha256'][name]=GUARD.sha(ROOT/name)
    return b

def expected_observations():
    variants=('exact_actual_max','exact_fixed_max','capacity_zero','capacity_finite','capacity_nan')
    return [dict(stage='eager_tail',case=i,variant=v) for i in range(3) for v in variants]+[dict(stage='capture_replay',case=i,variant='replay') for i in range(3)]

def verify_report(report):
    if report['status']!='completed' or any(v!='passed' for v in report['stages'].values()):raise ValueError('Incomplete probe stages')
    actual=[dict(stage=r['stage'],case=r['case'],variant=r.get('variant','replay')) for r in report['observations']]
    if actual!=expected_observations() or report['expected_observations']!=expected_observations():raise ValueError('Incomplete or reordered operator/replay domain')
    for row in report['observations']:
        if not row['comparison']['passed'] or not row['committed_unchanged_before_fixture_commit']:raise ValueError('Failed numerical/isolation evidence in completed report')
        if row['stage']=='capture_replay' and not (row['eager_capacity_comparison']['passed'] and row['stable_input_pointers'] and row['stable_output_pointer'] and row['committed_unchanged_before_fixture_commit']):raise ValueError('Failed replay identity/isolation evidence')

def new_report():
    return dict(status='running',stages={n:'pending' for n in ('real_qkv','eager_tail','capture_replay')},
        observations=[],expected_observations=expected_observations(),whole_model_graph=False,overlap=False,t_minus_two_capacity_graph=False)

def deadline_check(deadline):
    if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Cooperative probe deadline')

def extract_cases(model,requests,*,out=None,deadline=None):
    """Run stock SDPA target forwards; capture actual first-layer RoPE-applied QKV.

    The temporary callback delegates unchanged to installed SDPA. It restores this
    model's prior backend in finally; no existing target adapter is modified.
    """
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original=model.config._attn_implementation;sdpa=ALL_ATTENTION_FUNCTIONS['sdpa'];captured={}
    name='dspark_capacity_probe_capture_qkv'
    def callback(attention,q,k,v,mask,**kwargs):
        if attention.layer_idx==PROTOCOL['source_layer']:
            captured['qkv']=tuple(x.detach().clone() for x in (q,k,v))
        return sdpa(attention,q,k,v,mask,**kwargs)
    ALL_ATTENTION_FUNCTIONS.register(name,callback)
    model.config._attn_implementation=name;device=model.get_input_embeddings().weight.device
    cases=[]
    try:
        with torch.no_grad():
            for case_index,contexts in enumerate(PROTOCOL['contexts']):
                rows=[]
                for request_index,(request,c,n) in enumerate(zip(requests,contexts,PROTOCOL['query_lengths'])):
                    deadline_check(deadline)
                    ids=torch.tensor([request['prompt_token_ids'][:c+n]],dtype=torch.long,device=device)
                    captured.clear();model.model(input_ids=ids,use_cache=False,return_dict=True)
                    if 'qkv' not in captured:raise RuntimeError('Actual first-layer QKV callback not reached')
                    q,k,v=(x[0].transpose(0,1).contiguous() for x in captured['qkv'])
                    rows.append(dict(query=q[c:c+n].contiguous(),prefix_k=k[:c].contiguous(),prefix_v=v[:c].contiguous(),
                                     new_k=k[c:c+n].contiguous(),new_v=v[c:c+n].contiguous()))
                    if out is not None:save_tensors(Path(out)/f'real-qkv-{case_index}-request-{request_index}.pt',rows[-1])
                cases.append(dict(contexts=tuple(contexts),rows=rows))
    finally:model.config._attn_implementation=original
    return cases

def save_tensors(path,values):
    import torch
    torch.save({k:v.detach().cpu().clone() for k,v in values.items()},path)

def comparison(actual,expected,qlengths,*,bitwise=False):
    import torch
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or actual.device!=expected.device:raise ValueError('Operator output shape/dtype/device differs')
    rows=[];start=0
    for name,x,y in [('pooled',actual,expected)]+[(str(i),actual[sum(qlengths[:i]):sum(qlengths[:i+1])],expected[sum(qlengths[:i]):sum(qlengths[:i+1])]) for i in range(len(qlengths))]:
        finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        if finite:
            d=(x.double()-y.double()).abs();rms=float(d.square().mean().sqrt());limit=PROTOCOL['comparisons']
            equal=torch.equal(x,y);ok=equal if bitwise else bool((d<=limit['atol']+limit['rtol']*y.double().abs()).all()) and rms<=limit['max_rms']
            rows.append(dict(request=name,finite=True,max_abs=float(d.max()),rms=rms,bit_equal=equal,passed=ok))
        else:rows.append(dict(request=name,finite=False,passed=False))
    return dict(passed=all(x['passed'] for x in rows),bitwise_required=bitwise,rows=rows)

def make_pool(cases):
    import torch
    from dspark_qwen.persistent_target_kv import PersistentTargetKV,Bucket
    first=cases[0]['rows'][0]['new_k'];heads,dim=first.shape[1:]
    pool=PersistentTargetKV(layers=1,slots=2,context_capacity=148,max_query_tokens=5,heads=heads,dim=dim,dtype=first.dtype,device=first.device)
    handles=[pool.add_request(f'r{i}') for i in range(2)]
    for h,row in zip(handles,cases[0]['rows']):pool.load_prefix(h,row['prefix_k'][None],row['prefix_v'][None])
    bucket=Bucket(tuple(PROTOCOL['query_lengths']),tuple(PROTOCOL['context_ceilings']),'Preregistered native capacity-tail capability experiment; no performance applicability')
    pool.register_bucket(bucket)
    return pool,handles,bucket

def prepare(pool,handles,bucket,case):
    import torch
    if tuple(pool.length(h) for h in handles)!=case['contexts']:raise ValueError('Growing context transaction differs from protocol')
    tx=pool.begin(handles,bucket)
    pool.stage_layer(tx,0,torch.cat([r['new_k'] for r in case['rows']]),torch.cat([r['new_v'] for r in case['rows']]))
    # Independent exact assembly from committed copies, not the candidate gathers.
    pairs=[pool.request_kv(h) for h in handles]
    exact_k=torch.cat([torch.cat((kv[0][0],row['new_k'])) for kv,row in zip(pairs,case['rows'])])
    exact_v=torch.cat([torch.cat((kv[1][0],row['new_v'])) for kv,row in zip(pairs,case['rows'])])
    cap_k,cap_v,cq,ck,pos=pool.prepare_attention(tx,0)
    if not torch.equal(cap_k[:len(exact_k)],exact_k) or not torch.equal(cap_v[:len(exact_v)],exact_v):raise AssertionError('Candidate gather differs from independent exact assembly')
    return tx,torch.cat([r['query'] for r in case['rows']]),exact_k,exact_v,cap_k,cap_v,cq,ck,pos

def commit_fixture(pool,tx,handles,counts):
    """Check every committed byte against old prefix + selected actual scratch."""
    import torch
    expected=[];offset=0
    for h,n,q in zip(handles,counts,PROTOCOL['query_lengths']):
        old=pool.request_kv(h)
        expected.append(tuple(torch.cat((x,poolbuf[:,offset:offset+n]),dim=1) for x,poolbuf in zip(old,(pool.scratch_keys,pool.scratch_values))))
        offset+=q
    pool.commit(tx,dict(zip(handles,counts)))
    if any(not torch.equal(a,e) for h,pair in zip(handles,expected) for a,e in zip(pool.request_kv(h),pair)):
        raise AssertionError('Committed fixture KV bytes differ')

def exercise(cases,native_call,capture_factory,out,report,*,deadline=None,synchronize=lambda:None):
    """CPU-contract-testable sequencing. Formal CLI supplies only pinned GPU calls."""
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write=lambda:GUARD.write(out/'result.json',report)
    references=[];capacity_references=[];active='eager_tail';report['stages'][active]='running';write()
    try:
        if len(cases)!=3 or [list(c['contexts']) for c in cases]!=PROTOCOL['contexts']:raise ValueError('Exact three-state growing context domain required')
        pool,handles,bucket=make_pool(cases)
        for index,(case,counts) in enumerate(zip(cases,PROTOCOL['committed_query_rows'])):
            deadline_check(deadline)
            tx,q,k,v,ckey,cval,cq,ck,pos=prepare(pool,handles,bucket,case)
            before=(pool.keys.clone(),pool.values.clone());actual_max=max(c+n for c,n in zip(case['contexts'],PROTOCOL['query_lengths']))
            candidates={}
            for variant in ('exact_actual_max','exact_fixed_max','capacity_zero','capacity_finite','capacity_nan'):
                deadline_check(deadline)
                kk,vv=(k,v) if variant.startswith('exact') else (ckey,cval)
                if variant=='capacity_finite':kk[len(k):].fill_(100.);vv[len(v):].fill_(-100.)
                if variant=='capacity_nan':kk[len(k):].fill_(float('nan'));vv[len(v):].fill_(float('nan'))
                stem=f'eager-{index}-{variant}'
                report['current_attempt']=dict(stage=active,case=index,variant=variant);write()
                save_tensors(out/(stem+'-inputs.pt'),dict(q=q,k=kk,v=vv,cu_query=cq,cu_key=ck,positions=pos))
                output=native_call(q,kk,vv,cq,ck,4,actual_max if variant=='exact_actual_max' else 148)
                synchronize();output=output.detach().clone();save_tensors(out/(stem+'-output.pt'),{'output':output})
                candidates[variant]=output
                pristine=all(torch.equal(old,current) for old,current in zip(before,(pool.keys,pool.values)))
                compare=comparison(output,candidates['capacity_zero'] if variant in ('capacity_finite','capacity_nan') else candidates['exact_actual_max'],PROTOCOL['query_lengths'],bitwise=variant in ('capacity_finite','capacity_nan'))
                report['observations'].append(dict(stage=active,case=index,variant=variant,comparison=compare,committed_unchanged_before_fixture_commit=pristine,work=pool.work(tx),operator_work=dict(physical_q_rows=len(q),physical_k_rows=len(kk),logical_k_rows=len(k),padding_k_rows=len(kk)-len(k),max_q=4,max_k=actual_max if variant=='exact_actual_max' else 148),inputs=stem+'-inputs.pt',output=stem+'-output.pt'))
                write()
                if not compare['passed']:raise ValueError(f'Eager gate failed: {stem}')
                if not pristine:raise AssertionError('Verification contaminated committed KV')
            if not all(torch.equal(old,current) for old,current in zip(before,(pool.keys,pool.values))):raise AssertionError('Verification contaminated committed KV')
            references.append(candidates['exact_actual_max']);capacity_references.append(candidates['capacity_zero'])
            commit_fixture(pool,tx,handles,counts)
        report['stages'][active]='passed';active='capture_replay';report['stages'][active]='running';write()
        pool,handles,bucket=make_pool(cases);ws=pool._workspaces[bucket]
        query=torch.empty_like(torch.cat([r['query'] for r in cases[0]['rows']]))
        pointers=dict(query=query.data_ptr(),**pool.pointers(bucket));graph=None;output_pointer=None
        # Only fixed-size deterministic tensor work is captured. No transaction
        # inspection, metadata construction, host values or commits in this body.
        def body():
            for resident,scratch,old,new,result in ((pool.keys,pool.scratch_keys,ws.old_k,ws.new_k,ws.keys),(pool.values,pool.scratch_values,ws.old_v,ws.new_v,ws.values)):
                torch.index_select(resident[0].reshape(-1,pool.heads,pool.dim),0,ws.committed_indices,out=old)
                torch.index_select(scratch[0],0,ws.scratch_indices,out=new)
                torch.where(ws.from_scratch,new,old,out=result);result.masked_fill_(~ws.valid,0)
            return native_call(query,ws.keys,ws.values,ws.cu_query,ws.cu_key,4,148)
        for index,(case,counts) in enumerate(zip(cases,PROTOCOL['committed_query_rows'])):
            deadline_check(deadline)
            report['current_attempt']=dict(stage=active,case=index,variant='replay');write()
            tx,q,k,v,_,_,cq,ck,pos=prepare(pool,handles,bucket,case);query.copy_(q);before=(pool.keys.clone(),pool.values.clone())
            save_tensors(out/f'replay-{index}-inputs.pt',dict(q=query,exact_k=k,exact_v=v,cu_query=cq,cu_key=ck,positions=pos))
            if graph is None:
                graph=capture_factory(body)
            deadline_check(deadline);output=graph.replay();synchronize();save_tensors(out/f'replay-{index}-output.pt',{'output':output})
            compared=comparison(output,references[index],PROTOCOL['query_lengths'])
            capacity_compared=comparison(output,capacity_references[index],PROTOCOL['query_lengths'],bitwise=True)
            if output_pointer is None:output_pointer=output.data_ptr()
            stable_output=output.data_ptr()==output_pointer
            stable=dict(query=query.data_ptr(),**pool.pointers(bucket))==pointers
            pristine=all(torch.equal(old,current) for old,current in zip(before,(pool.keys,pool.values)))
            report['observations'].append(dict(stage=active,case=index,comparison=compared,eager_capacity_comparison=capacity_compared,stable_input_pointers=stable,stable_output_pointer=stable_output,input_pointers=pointers,output_pointer=output.data_ptr(),
                committed_unchanged_before_fixture_commit=pristine,positions=pos.cpu().tolist(),cu_key=ck.cpu().tolist(),work=pool.work(tx)))
            write()
            if not compared['passed'] or not capacity_compared['passed'] or not stable or not stable_output or not pristine:raise ValueError('Replay output/address/isolation gate failed')
            commit_fixture(pool,tx,handles,counts)
        report['stages'][active]='passed';report['status']='completed';write();return report
    except BaseException as exc:
        report['stages'][active]='failed';report['status']='partial_deadline' if isinstance(exc,TimeoutError) else 'failed'
        report['error_type']=type(exc).__name__;report['error']=str(exc);write();raise

def validate_real_cases(cases,device):
    import torch
    if len(cases)!=3:raise ValueError('Three declared real-QKV states required')
    for case,contexts in zip(cases,PROTOCOL['contexts']):
        if case['contexts']!=tuple(contexts) or len(case['rows'])!=2:raise ValueError('Actual target context domain mismatch')
        for row,c,n in zip(case['rows'],contexts,PROTOCOL['query_lengths']):
            for name,shape in (('query',(n,16,128)),('prefix_k',(c,8,128)),('prefix_v',(c,8,128)),('new_k',(n,8,128)),('new_v',(n,8,128))):
                x=row[name]
                if tuple(x.shape)!=shape or x.dtype!=torch.bfloat16 or x.device!=device or x.requires_grad or not x.is_contiguous() or not bool(torch.isfinite(x).all()):raise ValueError('Real target QKV shape/device/dtype/finite contract differs')

def gpu_capture(body,device):
    import torch
    stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(PROTOCOL['warmup_calls']):body()
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):output=body()
    torch.cuda.current_stream(device).wait_stream(stream)
    class Replay:
        def replay(self):graph.replay();return output
    return Replay()

def worker(binding_path,out):
    start=time.monotonic();b=json.loads(Path(binding_path).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report=new_report();GUARD.write(out/'result.json',report);GUARD.write(out/'protocol.json',PROTOCOL)
    try:
        if bind(b['model'],b['generation_manifest'],b['workloads'])!=b:raise ValueError('Bound source/input changed')
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        sys.path.insert(0,str(ROOT))
        from dspark_qwen.rocm_varlen import PinnedRocmVarlenKernel
        if transformers.__version__!='5.17.0':raise RuntimeError('Pinned Transformers required')
        device=torch.device('cuda:0');torch.cuda.set_device(device);kernel=PinnedRocmVarlenKernel(device)
        free,total=torch.cuda.mem_get_info(device)
        if free<PROTOCOL['minimum_free_bytes']:raise RuntimeError('Insufficient free memory')
        torch.cuda.set_per_process_memory_fraction(PROTOCOL['process_allocation_cap_bytes']/total,device)
        GUARD.write(out/'runtime.json',dict(runtime=kernel.runtime,transformers=transformers.__version__,scope=PROTOCOL['scope']))
        report['runtime_scope']='real_pretrained_gpu';report['stages']['real_qkv']='running';GUARD.write(out/'result.json',report)
        model=AutoModelForCausalLM.from_pretrained(b['model'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        cases=extract_cases(model,b['requests'],out=out,deadline=start+PROTOCOL['timeout_seconds']-PROTOCOL['cooperative_reserve_seconds'])
        validate_real_cases(cases,device);report['stages']['real_qkv']='passed';GUARD.write(out/'result.json',report)
        def native(q,k,v,cq,ck,maxq,maxk):
            return kernel._operator(q,k,v,cq,ck,maxq,maxk,0.,True,False,scale=128**-.5,
                window_size_left=None,window_size_right=None,seqused_k=None,alibi_slopes=None,block_table=None,num_splits=None)[0]
        with torch.no_grad():
            exercise(cases,native,lambda fn:gpu_capture(fn,device),out,report,
                deadline=start+PROTOCOL['timeout_seconds']-PROTOCOL['cooperative_reserve_seconds'],synchronize=lambda:torch.cuda.synchronize(device))
        intact=bind(b['model'],b['generation_manifest'],b['workloads'])==b
        GUARD.write(out/'post-input-integrity.json',{'passed':intact})
        if not intact:raise ValueError('Inputs changed during probe')
        return 0
    except BaseException as exc:
        if report['status'] in ('running','completed'):
            report['status']='failed';report['error_type']=type(exc).__name__;report['error']=str(exc)
            for key in report['stages']:
                if report['stages'][key]=='running':report['stages'][key]='failed'
            GUARD.write(out/'result.json',report)
        (out/'error.txt').write_text(traceback.format_exc());return 1

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('model','generation-manifest','workloads','asr-url'):p.add_argument('--'+n)
    p.add_argument('--output',type=Path);p.add_argument('--asr-pid',type=int);p.add_argument('--execute',action='store_true');p.add_argument('--worker-binding',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker_binding:return worker(a.worker_binding,a.output)
    if not a.model or not a.generation_manifest:p.error('Actual model and generation fingerprint manifest required')
    b=bind(a.model,a.generation_manifest,a.workloads)
    if not a.execute:print(json.dumps(dict(status='dry_run_no_torch_no_gpu',binding=b),indent=2));return 0
    if a.output is None or a.asr_pid is None or not a.asr_url:p.error('Fresh output and preserved ASR identity required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False,mode=0o700);GUARD.write(out/'binding.json',b)
    guard=GUARD.ASRGuard(a.asr_pid,a.asr_url,pre_free_bytes=PROTOCOL['minimum_free_bytes'])
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-binding',str(out/'binding.json'),'--output',str(out/'worker')]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2');env.pop('PYTHONPATH',None)
    status=GUARD.supervise(command,out/'supervision',guard,timeout=PROTOCOL['timeout_seconds'],env=env)
    intact=bind(a.model,a.generation_manifest,a.workloads)==b;GUARD.write(out/'post-input-integrity.json',{'passed':intact})
    if not intact or status['status']!='completed' or not status['released']:return 1
    r=json.loads((out/'worker/result.json').read_text())
    if r['status']!='completed' or any(v!='passed' for v in r['stages'].values()):return 1
    verify_report(r)
    return 0

if __name__=='__main__':raise SystemExit(main())
