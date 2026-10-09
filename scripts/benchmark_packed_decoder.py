#!/usr/bin/env python3
"""Shared-workload native eager full-shadow fixed-prefix end-to-end baseline.

Stdlib binding is the default. Uses original FP64 sampling/validation unchanged;
no capacity scheduler, graph replay, overlap or target-law-equivalence claim.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
import importlib.util
spec=importlib.util.spec_from_file_location('native_local_profile',ROOT/'scripts/profile_packed_decoder.py')
PROFILE=importlib.util.module_from_spec(spec);spec.loader.exec_module(PROFILE)
SHARED=PROFILE.module('native_shared_workloads',ROOT/'dspark_qwen/performance_workloads.py')
GUARD=PROFILE.GUARD
PROTOCOL=dict(version=1,method='eager_full_shadow_fixed_max_prefix',gamma=7,
    allocation='max(0,min(7,remaining_output_budget-1)) chosen before proposal',
    first_admission_output_counts_toward_budget=True,finished_requests='retain_resident_KV_until_batch_end',
    timeout_seconds=1800,cooperative_reserve_seconds=10,minimum_free_bytes=8*1024**3,
    process_allocation_cap_bytes=6*1024**3,capacity_scheduler_integrated=False,
    probability_policy='float64_softmax_normalize_cdf_v1',temperature=1.,eos_ids=[],
    scope='Fixed-output end-to-end synthetic baseline; not local verification SPS')

def bind(model,generation,checkpoint,workloads=None):
    value=PROFILE.bind(model,generation,checkpoint,workloads)
    value.pop('local_requests')
    value['protocol']=PROTOCOL;value['protocol_sha256']=PROFILE.digest(PROTOCOL)
    value['source_sha256']['scripts/benchmark_packed_decoder.py']=GUARD.sha(Path(__file__).resolve())
    manifest,_=SHARED.load_workloads(value['workloads'])
    value['expected_samples']=SHARED.expected_sample_identities(manifest)
    return value

class FixedPrefix(PROFILE.FrozenAllocation):
    """Declared before this round's draw; finished resident requests are inactive."""
    def __init__(self,session,allocations):
        super().__init__(session,allocations)
        self.roster={r:self.roster[r] for r in allocations}

def make_session_factory(model,draft,*,target_builder=None,draft_builder=None):
    """One registered adapter for all batches, with a charged per-batch reset.

    CPU tests inject only backend constructors and execute this same factory.
    The target keeps its final KV until the following reset; memory evidence must
    include and label that inherited pre-reset allocation.
    """
    from dspark_qwen.packed_sampling import PackedSpeculativeSession
    if target_builder is None:
        from dspark_qwen.varlen_target import VarlenPackedTarget
        from dspark_qwen.rocm_varlen import BACKEND
        target_builder=lambda m,ids:VarlenPackedTarget(m,ids,native_backend=BACKEND)
    if draft_builder is None:
        from dspark_qwen.packed_draft import PackedDraft,DRAFT_BACKEND
        draft_builder=lambda d:PackedDraft(d,native_backend=DRAFT_BACKEND)
    target=target_builder(model,draft.spec.layer_ids)
    def factory():
        target.reset()
        return PackedSpeculativeSession(target,draft_builder(draft),amp=True)
    factory.target=target
    return factory

def sync(device):
    if device.type=='cuda':
        import torch
        torch.cuda.synchronize(device)

def batch(factory,device,case,manifest,*,diagnostic=False,deadline=None):
    import torch
    from dspark_qwen.packed_sampling import RequestSpec
    from dspark_qwen.tensor_sampling import TensorRandom
    # Match baseline's untimed token-list/SamplingParams preparation. Native
    # tensor/RNG preparation is likewise explicit and excluded, never model work.
    prepared=time.perf_counter()
    specs={r['request']:RequestSpec(torch.tensor([r['prompt_token_ids']],dtype=torch.long,device=device),
        manifest['output_tokens'],TensorRandom(torch.Generator(device=device).manual_seed(r['seed']))) for r in case['requests']}
    sync(device);preparation_seconds=time.perf_counter()-prepared
    memory={}
    if device.type=='cuda':
        memory=dict(pre_reset_allocated_bytes=torch.cuda.memory_allocated(device),
            pre_reset_reserved_bytes=torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
    rounds=[];first={};finished={}
    sync(device);started=time.perf_counter()
    session=factory()  # Fresh cache/request-state construction is charged.
    admitted=session.admit(specs)
    if diagnostic:
        observed=time.perf_counter()
        first={r:observed-started for r in admitted}
        finished={r:observed-started for r,x in session.requests.items() if x['finished']}
    while any(not x['finished'] for x in session.requests.values()):
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError('Cooperative end-to-end deadline; incomplete batch is not a sample')
        active=[r for r,x in session.requests.items() if not x['finished']]
        allocations={r:max(0,min(PROTOCOL['gamma'],session.requests[r]['budget']-len(session.requests[r]['output'])-1)) for r in active}
        policy=FixedPrefix(session,allocations)
        issued=session.propose(active,mode='shadow')
        result=session.verify_commit(issued.proposals,allocations,allocation_policy=policy)
        rounds.append(dict(active_requests=active,allocation=allocations,requests=result['requests'],work=result['work'],
            actual_logical_b=len(active)+sum(allocations.values()),
            actual_physical_b=result['work']['target']['physical_query_tokens'],
            resident_requests=len(session.requests)))
        if diagnostic:
            observed=time.perf_counter()
            for r in active:
                if session.requests[r]['finished']:finished[r]=observed-started
        # Work/decision records above contain scalars only. Do not let benchmark
        # observers keep full-vocabulary q/p alive through the next model round.
        del issued,result
    sync(device);elapsed=time.perf_counter()-started
    # Output validation and hashes follow the measured synchronous completion.
    outputs=session.outputs();records=[]
    for request in case['requests']:
        r=request['request'];tokens=outputs[r];state=session.requests[r]
        if len(tokens)!=manifest['output_tokens'] or not state['finished'] or state['budget']!=len(tokens):
            raise ValueError('Native fixed output budget incomplete or exceeded')
        records.append(dict(request=r,output_tokens=len(tokens),output_sha256=SHARED.sha_json(tokens),
            ttft_seconds=first[r] if diagnostic else None,
            completion_latency_seconds=finished[r] if diagnostic else None,
            latency_status='host_observed_batched_admission_and_round_completion' if diagnostic else 'unsupported_in_primary_without_observer',
            latency_scope='shared batch submission boundary to host observation; not pure prefill or individual asynchronous submission' if diagnostic else None))
    target_queries=sum(x['actual_logical_b'] for x in rounds)
    output_count=sum(x['output_tokens'] for x in records)
    admission_tokens=sum(len(x) for x in admitted.values())
    committed=sum(x['committed'] for row in rounds for x in row['requests'].values())
    if admission_tokens+committed!=output_count:
        raise ValueError('Admission-first-output plus committed-round output accounting differs')
    record=dict(batch_wall_seconds=elapsed,output_tokens=output_count,requests=records,
        time_scope='fresh request/cache creation, batched admission/first sample, full-shadow rounds, private-source validation, q/p checks and completion synchronization',
        preparation_seconds=preparation_seconds,rounds=rounds,admission_output_tokens=admission_tokens,
        committed_round_output_tokens=committed,prefill_work=session.last_prefill_work,
        target_verification_query_rows=target_queries,
        accepted_draft_tokens=sum(x['accepted'] for row in rounds for x in row['requests'].values()),
        selected_proposal_tokens=sum(x['proposed'] for row in rounds for x in row['requests'].values()),
        retained_resident_requests=len(session.requests),final_context_lengths=dict(session.target._lengths),
        capacity_scheduler_integrated=False,speculative_verification_sps=False,
        first_output_note='Admission emits the first budget token; diagnostic TTFT includes complete batched admission/context projection, not isolated prefill',
        pure_prefill_seconds=None,pure_prefill_status='not_isolated_in_this_pass')
    if device.type=='cuda':record.update(**memory,
        whole_operation_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        whole_operation_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        memory_scope='Before cache reset through completion; includes inherited previous target KV and prepared inputs, not an independent per-cell peak')
    # Proposal/verification observations are released every round. The factory
    # retains the target's final KV until next batch reset; account for it above.
    return record

def run(factory,device,manifest,out,*,deadline=None,setup=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);samples=[]
    specification=dict(protocol=PROTOCOL,expected_samples=SHARED.expected_sample_identities(manifest),
        model_loads_per_run=1,capacity_scheduler_integrated=False,arrival_load_frontier=False)
    GUARD.write(out/'specification.json',specification)
    if setup is not None:GUARD.write(out/'runtime.json',setup)
    try:
        for phase,repeat,case in SHARED.schedule(manifest):
            if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Cooperative deadline before next batch')
            sample=batch(factory,device,case,manifest,diagnostic=phase=='diagnostic',deadline=deadline)
            sample.update(phase=phase,repeat=repeat,case_id=case['case_id'],request_count=case['request_count'],prompt_length=case['prompt_length'])
            samples.append(sample)
            with (out/'samples.jsonl').open('a') as stream:stream.write(json.dumps(sample,allow_nan=False)+'\n')
        aggregates=SHARED.verify_complete_samples(samples,manifest)
        result=dict(status='completed',sample_count=len(samples),aggregates=aggregates,
            capacity_scheduler_integrated=False,speculative_verification_sps=False,arrival_load_frontier=False,
            prior_target_numerical_gate='failed_unchanged',whole_system_pass_claimed=False)
        GUARD.write(out/'result.json',result);return result
    except BaseException as exc:
        GUARD.write(out/'result.json',dict(status='partial_deadline' if isinstance(exc,TimeoutError) else 'failed',
            error_type=type(exc).__name__,error=str(exc),sample_count=len(samples),
            aggregates=SHARED.summarize(samples,manifest),expected_samples=specification['expected_samples'],
            capacity_scheduler_integrated=False,speculative_verification_sps=False,whole_system_pass_claimed=False))
        raise

def worker(binding_path,out):
    started=time.monotonic();binding=json.loads(Path(binding_path).read_text())
    if bind(binding['model'],binding['generation_manifest'],binding['checkpoint'],binding['workloads'])!=binding:
        raise ValueError('Frozen source/input changed before loading')
    manifest,workload_sha=SHARED.load_workloads(binding['workloads'])
    import torch
    import transformers
    sys.path.insert(0,str(ROOT))
    from transformers import AutoModelForCausalLM
    from dspark_qwen.checkpoint import load_checkpoint
    from dspark_qwen.config import DraftConfig
    from dspark_qwen.model import DSparkDraft
    from dspark_qwen.rocm_varlen import PinnedRocmVarlenKernel
    if transformers.__version__!='5.17.0' or not torch.cuda.is_available():raise RuntimeError('Require pinned GPU runtime')
    device=torch.device('cuda:0');torch.cuda.set_device(device);runtime=PinnedRocmVarlenKernel(device).runtime
    free,total=torch.cuda.mem_get_info()
    if free<PROTOCOL['minimum_free_bytes']:raise RuntimeError('8 GiB free guard failed')
    torch.cuda.set_per_process_memory_fraction(PROTOCOL['process_allocation_cap_bytes']/total)
    model=AutoModelForCausalLM.from_pretrained(binding['model'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    draft=DSparkDraft(model,DraftConfig.from_dict(binding['draft_config'])).to(device).eval();load_checkpoint(binding['checkpoint'],draft)
    factory=make_session_factory(model,draft)
    setup_seconds=time.monotonic()-started
    result=run(factory,device,manifest,out,deadline=started+PROTOCOL['timeout_seconds']-PROTOCOL['cooperative_reserve_seconds'],
        setup=dict(runtime=runtime,setup_seconds=setup_seconds,model_loads=1,workload_sha256=workload_sha,
            torch=torch.__version__,transformers=transformers.__version__))
    if bind(binding['model'],binding['generation_manifest'],binding['checkpoint'],binding['workloads'])!=binding:
        raise ValueError('Frozen source/input changed during benchmark')
    GUARD.write(Path(out)/'post-input-integrity.json',dict(passed=True));return 0 if result['status']=='completed' else 1

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','generation-manifest','checkpoint','workloads','asr-url']:p.add_argument('--'+name)
    p.add_argument('--output',type=Path);p.add_argument('--asr-pid',type=int)
    p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--worker-binding',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.worker_binding:
        try:return worker(args.worker_binding,args.output)
        except BaseException:
            args.output.mkdir(parents=True,exist_ok=True);(args.output/'error.txt').write_text(traceback.format_exc());return 1
    if not all((args.model,args.generation_manifest,args.checkpoint)):p.error('Actual model, manifest, selected checkpoint required')
    binding=bind(args.model,args.generation_manifest,args.checkpoint,args.workloads)
    if not args.execute:print(json.dumps(dict(status='dry_run_no_backend_import_no_gpu',binding=binding),indent=2));return 0
    if args.dry_run or args.output is None or args.asr_pid is None or not args.asr_url:p.error('Execute requires fresh output, ASR identity/health and no dry-run')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False,mode=0o700);GUARD.write(out/'binding.json',binding)
    guard=GUARD.ASRGuard(args.asr_pid,args.asr_url,pre_free_bytes=PROTOCOL['minimum_free_bytes'])
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-binding',str(out/'binding.json'),'--output',str(out/'worker')]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2');env.pop('PYTHONPATH',None)
    status=GUARD.supervise(command,out/'supervision',guard,timeout=PROTOCOL['timeout_seconds'],env=env)
    intact=bind(args.model,args.generation_manifest,args.checkpoint,args.workloads)==binding
    GUARD.write(out/'post-input-integrity.json',dict(passed=intact))
    if not intact or status['status']!='completed' or not status['released']:return 1
    manifest,_=SHARED.load_workloads(binding['workloads'])
    samples=[json.loads(line) for line in (out/'worker/samples.jsonl').read_text().splitlines()]
    recomputed=SHARED.verify_complete_samples(samples,manifest);result=json.loads((out/'worker/result.json').read_text())
    if recomputed!=result['aggregates'] or result['sample_count']!=len(binding['expected_samples']):raise ValueError('Controller raw sample/aggregate mismatch')
    return 0 if result['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
