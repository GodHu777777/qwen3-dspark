#!/usr/bin/env python3
"""Bounded full-shadow packed cost profile; default stdlib binding, no GPU.

All 64 R2/gamma7 allocations are declared. Missing cells never support lookup.
Snapshot restoration is outside timed rounds; all existing round checks remain.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value
GATE = module('packed_gate_binding', ROOT/'scripts/probe_packed_decoder.py')
GUARD = module('packed_profile_guard', ROOT/'scripts/guard_vllm_smoke.py')
PROTOCOL = dict(version=1, requests=['r0','r1'], context_lengths=[128,128], gamma=7,
    cells=[dict(ell=[a,b], logical_b=2+a+b) for total in range(15)
           for a in range(8) for b in range(8) if a+b == total],
    warmups_per_cell=2, primary_samples_per_cell=5, diagnostic_samples_per_cell=1,
    sample_order='replicate-major full-domain sweeps; rotate by17*repeat, reverse odd repeats; phases warmup,primary,diagnostic',
    input_policy='First128 exact tokens and original seeds from shared r2-c256 case', max_new_tokens=128,
    temperature=1., eos_ids=[], timeout_seconds=300, cooperative_reserve_seconds=10,
    minimum_free_bytes=8*1024**3, process_allocation_cap_bytes=6*1024**3,
    mode='eager_full_shadow_predeclared_allocation', probability_policy='float64_softmax_normalize_cdf_v1',
    timer='synchronize_full_round_wall_seconds', sps_unit='completed_rounds_per_second',
    scope='Exact R2 C128 frozen-cache cost, not planner cost or end-to-end throughput')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def bind(model, generation, checkpoint, workloads=None):
    value = GATE.bind(model,generation,checkpoint)
    helper=module('shared_profile_inputs',ROOT/'dspark_qwen/performance_workloads.py')
    path=Path(workloads or ROOT/'configs/performance-workloads.example.json').resolve()
    manifest,workload_sha=helper.load_workloads(path)
    local=manifest['local_native_profile']
    if local['committed_context_vector']!=PROTOCOL['context_lengths'] or set(map(tuple,local['prefix_lengths']))!={tuple(x['ell']) for x in PROTOCOL['cells']}:
        raise ValueError('Shared local profile domain changed')
    if manifest['model']['config_sha256']!=value['target_fingerprint']['config.json']:
        raise ValueError('Shared workload actual model fingerprint mismatch')
    case=next(x for x in manifest['cases'] if x['case_id']=='r2-c256')
    value.update(workloads=str(path),workloads_sha256=workload_sha,
        local_requests=[dict(request=x['request'],seed=x['seed'],prompt_token_ids=x['prompt_token_ids'][:128]) for x in case['requests']])
    if [x['request'] for x in value['local_requests']]!=PROTOCOL['requests']:raise ValueError('Shared canonical request order changed')
    value['protocol'] = PROTOCOL; value['protocol_sha256'] = digest(PROTOCOL)
    for name in ['scripts/profile_packed_decoder.py','scripts/guard_vllm_smoke.py']:
        value['source_sha256'][name] = GUARD.sha(ROOT/name)
    return value

def domain(binding):
    return dict(model_fingerprint=binding['target_fingerprint'],
        checkpoint_weights_sha256=binding['checkpoint_weights_sha256'],
        source_sha256=binding['source_sha256'],protocol_sha256=binding['protocol_sha256'],
        workloads_sha256=binding['workloads_sha256'],local_input_sha256=digest(binding['local_requests']),
        active_requests=2,resident_requests=2,context_lengths=[128,128],total_context=256,
        gamma=7,mode=PROTOCOL['mode'],probability_policy=PROTOCOL['probability_policy'],
        target_backend=GATE.PROTOCOL['target_backend'],draft_backend=GATE.PROTOCOL['draft_backend'])

def lookup(report, expected_domain, ell, *, purpose='local_allocation_diagnostic'):
    """Only complete exact-domain cells yield a measured rate; never interpolate."""
    if purpose!='local_allocation_diagnostic':
        raise ValueError('Local rate excludes planner/history; CapacityRoundDriver profile use is forbidden')
    if report['domain'] != expected_domain:
        raise ValueError('Measured domain differs; no R/context/layout transfer')
    rows = [x for x in report['cells'] if x['ell'] == list(ell)]
    if len(rows) != 1 or rows[0]['status'] != 'completed':
        raise ValueError('Unmeasured or incomplete allocation cell')
    return rows[0]['primary_statistics']

def schedule(spec):
    """Fixed repeated sweeps, not B-sorted consecutive timing blocks."""
    count=len(spec['cells'])
    for phase,repeats in [('warmups',spec['warmups_per_cell']),('primary',spec['primary_samples_per_cell']),('diagnostic',spec['diagnostic_samples_per_cell'])]:
        for repeat in range(repeats):
            offset=(17*repeat)%count;order=list(range(count));order=order[offset:]+order[:offset]
            if repeat%2:order.reverse()
            for index in order:yield phase,repeat,index

def synchronize(session):
    if session.target.device.type == 'cuda':
        import torch
        torch.cuda.synchronize(session.target.device)

TARGET_FIELDS = ('cache','_lengths','_markers','_next_marker','key_requests','key_positions')
DRAFT_FIELDS = ('layers','_lengths','_markers','_next_marker','key_requests','key_positions','last_projection_work')
SESSION_FIELDS = ('requests','epoch','_incarnation','_nonce','_pending','failed','last_prefill_work')

def snapshot(session):
    if session._pending or session.failed:
        raise ValueError('Snapshot requires healthy session without pending proposals')
    return copy.deepcopy(dict(target={k:getattr(session.target,k) for k in TARGET_FIELDS},
        draft={k:getattr(session.draft,k) for k in DRAFT_FIELDS},
        session={k:getattr(session,k) for k in SESSION_FIELDS}))

def tensor_digest(t):
    import torch
    t=t.detach().cpu().contiguous()
    return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())

def snapshot_identity(state):
    values={}
    for owner in ('target','draft'):
        part=state[owner]
        values[owner]={k:part[k] for k in ('_lengths','_markers','_next_marker')}
        values[owner].update({k:tensor_digest(part[k]) for k in ('key_requests','key_positions')})
        layers=[(x.keys,x.values) for x in part['cache'].layers] if owner=='target' else part['layers']
        values[owner]['kv']=[[tensor_digest(k),tensor_digest(v)] for k,v in layers]
    values['session']={k:state['session'][k] for k in ('epoch','_incarnation','_nonce','failed')}
    values['requests']={r:dict(**{k:v for k,v in x.items() if k!='rng'},rng=tensor_digest(x['rng'].generator.get_state()))
                        for r,x in state['session']['requests'].items()}
    return digest(values)

def restore(session, frozen):
    """Single-owner profiling only; never restore externally issued capabilities."""
    import torch
    if session._pending or session.failed:
        raise ValueError('Cannot restore with outstanding capabilities or failed session')
    synchronize(session);start=time.perf_counter();candidate=copy.deepcopy(frozen)
    for owner,fields in [('target',TARGET_FIELDS),('draft',DRAFT_FIELDS),('session',SESSION_FIELDS)]:
        obj=session if owner=='session' else getattr(session,owner)
        for k in fields:setattr(obj,k,candidate[owner][k])
    session._check_lengths();session.draft.validate_cache()
    # Exact tensor comparisons outside the timed round; do not hash/copy KV to host.
    for owner in ('target','draft'):
        obj=getattr(session,owner);saved=frozen[owner]
        assert obj._lengths==saved['_lengths'] and obj._markers==saved['_markers']
        for key in ('key_requests','key_positions'):assert torch.equal(getattr(obj,key),saved[key])
        actual=[(x.keys,x.values) for x in obj.cache.layers] if owner=='target' else obj.layers
        expected=[(x.keys,x.values) for x in saved['cache'].layers] if owner=='target' else saved['layers']
        assert all(torch.equal(a,b) for x,y in zip(actual,expected) for a,b in zip(x,y))
    for request,x in session.requests.items():
        assert torch.equal(x['rng'].generator.get_state(),frozen['session']['requests'][request]['rng'].generator.get_state())
    synchronize(session)
    return time.perf_counter()-start

class FrozenAllocation:
    """Charge real private confidence copy/capability validation, not a planner."""
    def __init__(self,session,allocations):
        self.allocations=dict(allocations);self.epoch=session.epoch
        self.roster={r:(x['incarnation'],session.target.lengths[r]) for r,x in session.requests.items()}
    def validate_allocation(self,context,handles,allocations):
        if context['epoch']!=self.epoch or allocations!=self.allocations or set(handles)!=set(self.roster):
            raise ValueError('Frozen allocation or roster changed')
        for r,h in handles.items():
            if (h.incarnation,h.cache_length)!=self.roster[r] or h.epoch!=self.epoch:
                raise ValueError('Stale profile capability')
            if len(context['proposal_confidence_logits'][r])!=h.proposal_limit:
                raise ValueError('Incomplete full shadow confidence')
        return dict(kind='predeclared_profile_allocation',planner_cost_measured=False,
                    confidence_host_values=sum(map(len,context['proposal_confidence_logits'].values())))

class Diagnostics:
    """Separate event/host-span pass; nested intervals must not be summed."""
    def __init__(self,session):
        self.session=session;self.events=[];self.originals=[]
    def __enter__(self):
        for obj,name in [(self.session.draft,'backbone'),(self.session.target,'append'),
                         (self.session.target,'predict'),(self.session.target,'crop'),
                         (self.session.draft,'append_committed')]:
            original=getattr(obj,name);self.originals.append((obj,name,original))
            def wrapped(*args,_fn=original,_name=name,_owner=type(obj).__name__,**kwargs):
                return self.call(_owner+'.'+_name,lambda:_fn(*args,**kwargs))
            setattr(obj,name,wrapped)
        return self
    def call(self,name,fn):
        import torch
        gpu=self.session.target.device.type=='cuda'
        first=last=None
        if gpu:first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);first.record()
        start=time.perf_counter()
        try:return fn()
        finally:
            elapsed=time.perf_counter()-start
            if gpu:last.record()
            self.events.append((name,elapsed,first,last))
    def __exit__(self,*args):
        for obj,name,original in self.originals:setattr(obj,name,original)
    def result(self):
        return [dict(region=n,host_span_seconds=s,gpu_stream_interval_ms=a.elapsed_time(b) if a else None)
                for n,s,a,b in self.events]

def round_once(session,ell,diagnostic=False):
    import contextlib
    # All allocation construction, private-source validation and existing q/p
    # finite/normalization checks remain inside the primary timed boundary.
    synchronize(session);start=time.perf_counter()
    allocations=dict(zip(session.requests,ell))
    policy=FrozenAllocation(session,allocations)
    capture=Diagnostics(session) if diagnostic else None
    with capture if capture else contextlib.nullcontext():
        call=capture.call if capture else lambda _name,fn:fn()
        batch=call('full_shadow_propose',lambda:session.propose(allocations,mode='shadow'))
        result=call('private_source_verify_commit',lambda:session.verify_commit(batch.proposals,allocations,allocation_policy=policy))
    synchronize(session);seconds=time.perf_counter()-start
    row=dict(wall_seconds=seconds,requests=result['requests'],work=result['work'],
        logical_b=2+sum(ell),physical_b=result['work']['target']['physical_query_tokens'],
        accepted=sum(x['accepted'] for x in result['requests'].values()),
        committed=sum(x['committed'] for x in result['requests'].values()),
        timed_scope='Existing full-shadow proposal + private confidence copy/capability validation + target verification + q/p checks + commit/crop',
        planner_cost_measured=False)
    if capture:
        row['diagnostic_regions']=capture.result()
        top=sum(x['host_span_seconds'] for x in row['diagnostic_regions'] if x['region'] in ('full_shadow_propose','private_source_verify_commit'))
        row['diagnostic_other_round_wall_seconds']=seconds-top
        row['diagnostic_note']='Nested GPU stream/host spans, not disjoint kernel times; remainder includes policy creation, instrumentation and final synchronization'
    return row

def statistics_for(rows):
    samples=[x['wall_seconds'] for x in rows];median=statistics.median(samples)
    return dict(samples=len(samples),median_seconds=median,min_seconds=min(samples),max_seconds=max(samples),
        mean_seconds=statistics.mean(samples),rounds_per_second=1/median,
        capacity_round_driver_profile_eligible=False,excluded_planner_history=True,
        units='rounds/s = 1/median complete round seconds; not B/time or output tokens/s')

def exercise(session,out,bound_domain,*,spec=None,deadline=None):
    import torch
    spec=PROTOCOL if spec is None else spec;out=Path(out);out.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',domain=bound_domain,protocol_sha256=digest(spec),
        timing_scope='frozen-cache eager full-shadow allocation profile; scheduler/history maintenance excluded and not claimed',
        cells=[dict(**cell,status='unmeasured',warmups=[],primary=[],diagnostic=[]) for cell in spec['cells']],
        excluded_costs=['model load','snapshot preparation/restoration','disk evidence','real planner/history management'],
        capacity_round_driver_profile_eligible=False,
        whole_system_pass_claimed=False)
    save=lambda:GUARD.write(out/'result.json',r)
    save();synchronize(session);start=time.perf_counter();frozen=snapshot(session);synchronize(session)
    r['snapshot_seconds']=time.perf_counter()-start;r['snapshot_sha256']=snapshot_identity(frozen);save()
    try:
        # Every repeat sweeps the full domain; diagnostics are a separate final pass.
        for sequence_index,(phase,repeat,index) in enumerate(schedule(spec)):
            cell=r['cells'][index]
            if deadline is not None and time.monotonic()>=deadline:
                r['status']='partial_deadline';save();return r
            restore_seconds=restore(session,frozen)
            if session.target.device.type=='cuda':torch.cuda.reset_peak_memory_stats(session.target.device)
            row=round_once(session,cell['ell'],phase=='diagnostic');row['restore_seconds']=restore_seconds
            row.update(repeat_index=repeat,sequence_index=sequence_index)
            if session.target.device.type=='cuda':
                row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(session.target.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(session.target.device))
            cell[phase].append(row);cell['status']='partial'
            if len(cell['primary'])==spec['primary_samples_per_cell'] and len(cell['diagnostic'])==spec['diagnostic_samples_per_cell']:
                cell['primary_statistics']=statistics_for(cell['primary']);cell['diagnostic_statistics']=statistics_for(cell['diagnostic']);cell['status']='completed'
            save()
        r['status']='completed';save();return r
    except BaseException as exc:
        r.update(status='failed',error_type=type(exc).__name__,error=str(exc));save();raise


def worker(binding_path,out):
    started=time.monotonic();binding=json.loads(Path(binding_path).read_text());out=Path(out)
    if bind(binding['model'],binding['generation_manifest'],binding['checkpoint'],binding['workloads'])!=binding:
        raise ValueError('Bound input/source/protocol changed')
    import torch
    import transformers
    sys.path.insert(0,str(ROOT))
    from transformers import AutoModelForCausalLM
    from dspark_qwen.checkpoint import load_checkpoint
    from dspark_qwen.config import DraftConfig
    from dspark_qwen.model import DSparkDraft
    from dspark_qwen.packed_draft import PackedDraft,DRAFT_BACKEND
    from dspark_qwen.packed_sampling import PackedSpeculativeSession,RequestSpec
    from dspark_qwen.varlen_target import VarlenPackedTarget
    from dspark_qwen.rocm_varlen import BACKEND,PinnedRocmVarlenKernel
    from dspark_qwen.tensor_sampling import TensorRandom
    if transformers.__version__!='5.17.0' or not torch.cuda.is_available():raise RuntimeError('Require pinned actual GPU runtime')
    device=torch.device('cuda:0');torch.cuda.set_device(device);pin=PinnedRocmVarlenKernel(device).runtime
    free,total=torch.cuda.mem_get_info()
    if free<PROTOCOL['minimum_free_bytes']:raise RuntimeError('8 GiB free guard failed')
    torch.cuda.set_per_process_memory_fraction(PROTOCOL['process_allocation_cap_bytes']/total)
    model=AutoModelForCausalLM.from_pretrained(binding['model'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    draft=DSparkDraft(model,DraftConfig.from_dict(binding['draft_config'])).to(device).eval();load_checkpoint(binding['checkpoint'],draft)
    session=PackedSpeculativeSession(VarlenPackedTarget(model,draft.spec.layer_ids,native_backend=BACKEND),PackedDraft(draft,native_backend=DRAFT_BACKEND),amp=True)
    session.admit({x['request']:RequestSpec(torch.tensor([x['prompt_token_ids']],device=device,dtype=torch.long),PROTOCOL['max_new_tokens'],TensorRandom(torch.Generator(device=device).manual_seed(x['seed'])))
        for x in binding['local_requests']})
    if list(session.target.lengths.values())!=PROTOCOL['context_lengths']:raise RuntimeError('Actual committed context differs from frozen domain')
    measured_domain=domain(binding);measured_domain['runtime']=pin
    r=exercise(session,out,measured_domain,deadline=started+PROTOCOL['timeout_seconds']-PROTOCOL['cooperative_reserve_seconds'])
    if bind(binding['model'],binding['generation_manifest'],binding['checkpoint'],binding['workloads'])!=binding:raise ValueError('Source/input identity changed during profile')
    GUARD.write(out/'post-input-integrity.json',dict(passed=True))
    return 0 if r['status']=='completed' else 2


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model');p.add_argument('--generation-manifest');p.add_argument('--checkpoint')
    p.add_argument('--workloads',type=Path);p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--asr-pid',type=int);p.add_argument('--asr-url');p.add_argument('--worker-binding',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.worker_binding:
        try:return worker(args.worker_binding,args.output)
        except BaseException:
            args.output.mkdir(parents=True,exist_ok=True);(args.output/'error.txt').write_text(traceback.format_exc());return 1
    if not all((args.model,args.generation_manifest,args.checkpoint)):p.error('Actual model/generation manifest/selected checkpoint required')
    binding=bind(args.model,args.generation_manifest,args.checkpoint,args.workloads)
    if not args.execute:
        print(json.dumps(dict(status='dry_run_no_backend_import_no_gpu',binding=binding,domain=domain(binding)),indent=2));return 0
    if args.dry_run:p.error('Choose dry-run or execute')
    if args.output is None or args.asr_pid is None or not args.asr_url:p.error('Fresh output and ASR identity/health endpoint required')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False,mode=0o700);GUARD.write(out/'binding.json',binding)
    guard=GUARD.ASRGuard(args.asr_pid,args.asr_url,pre_free_bytes=PROTOCOL['minimum_free_bytes'])
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-binding',str(out/'binding.json'),'--output',str(out/'worker')]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2');env.pop('PYTHONPATH',None)
    result=GUARD.supervise(command,out/'supervision',guard,timeout=PROTOCOL['timeout_seconds'],env=env)
    intact=bind(args.model,args.generation_manifest,args.checkpoint,args.workloads)==binding
    GUARD.write(out/'post-input-integrity.json',dict(passed=intact))
    return 0 if intact and result['status']=='completed' and result['released'] else 1

if __name__=='__main__':raise SystemExit(main())
