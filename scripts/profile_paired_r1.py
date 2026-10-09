#!/usr/bin/env python3
"""Bounded diagnostic observation of the unchanged paired R1 runner."""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('profile_base_paired_r1', ROOT/'scripts/benchmark_paired_r1.py')
BASE = importlib.util.module_from_spec(spec); spec.loader.exec_module(BASE)
ORIGINAL_BIND = BASE.bind
GUARD = BASE.GUARD
PROFILE_PROTOCOL = dict(version=1, method='diagnostic_only_complete_session_observer',
    baseline_execution_commit='192a3578eb71527e02fded28a3beaf487d53859a',
    panel='unchanged 36 full128 batches; warmup2 primary5 diagnostic2 for two cases and two arms',
    diagnostic0='nested host spans and real GPU stream event intervals',
    diagnostic1='complete CPU+ROCm GPU trace with verifiable CPU/runtime/GPU correlation',
    primary_observer=False, max_traces=4, trace_bytes=256*1024**2, total_trace_bytes=1024**3,
    rss_bytes=8*1024**3, monitor_seconds=.25, max_spans=100000,
    rss_abort='independent stdlib process samples bound worker RSS; PID/startticks plus pidfd; write failure then SIGKILL only bound worker; monitor86; parent subreaper reaps',
    profiler_options=dict(record_shapes=False,profile_memory=False,with_stack=False),
    trace_failure='preserve partial artifacts and completed samples; no CPU-only fallback or retry')


def bind(model, generation, checkpoint, workloads=None):
    binding = ORIGINAL_BIND(model, generation, checkpoint, workloads)
    binding.update(profile_protocol=PROFILE_PROTOCOL, profile_protocol_sha256=BASE.PROFILE.digest(PROFILE_PROTOCOL),
                   worker_entry='scripts/profile_paired_r1.py')
    for name in ['scripts/profile_paired_r1.py','dspark_qwen/paired_observer.py',
                 'dspark_qwen/profile_resource_monitor.py','tests/test_profile_paired_r1.py','docs/paired-r1-profile.md']:
        binding['source_sha256'][name] = GUARD.sha(ROOT/name)
    return binding


def invariant(sample):
    fields = ['arm','output_tokens','rounds','execution_coverage','admission_output_tokens',
              'committed_round_output_tokens','prefill_work','final_context_lengths',
              'accepted_draft_tokens','selected_proposal_tokens','selected_layer_ids','probability_policy']
    value = {key:sample[key] for key in fields}
    value['requests'] = [{k:r[k] for k in ('request','output_tokens','output_sha256')} for r in sample['requests']]
    return BASE.SHARED.sha_json(value)


def diagnostic(factory, device, case, manifest, arm, repeat, directory, *, deadline, result_path,
               observer_class=None, monitor_class=None, profiler_factory=None):
    import torch
    from dspark_qwen.paired_observer import Observer, ResourceMonitor, analyze_trace, write
    observer_class = observer_class or Observer; monitor_class = monitor_class or ResourceMonitor
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    status = dict(status='running',case_id=case['case_id'],arm=arm,phase='diagnostic',repeat=repeat,
                  stage='observer_start',complete_session=False)
    write(directory/'status.json', status)
    observation = None; sample = None
    try:
        with monitor_class(directory.parent, result_path, deadline=deadline,state_directory=directory) as monitor:
            if repeat == 1:
                if device.type != 'cuda' or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
                    raise RuntimeError('CPU+ROCm GPU profiler capability unavailable')
                make = profiler_factory or torch.profiler.profile
                profiler = make(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,profile_memory=False,with_stack=False)
            else: profiler = contextlib.nullcontext()
            status['stage'] = 'complete_session'; write(directory/'status.json', status)
            try:
                with profiler:
                    with observer_class(factory, BASE.NATIVE, events=repeat==0, timeline=repeat==1) as observation:
                        with torch.profiler.record_function('paired::complete_session') if repeat==1 else contextlib.nullcontext():
                            sample = BASE.batch(factory, device, case, manifest, arm, diagnostic=True,deadline=deadline)
            except BaseException:
                # If Kineto can export after a partial session, retain it while
                # the same RSS/file/deadline monitor is still active.
                if repeat==1:
                    try:
                        profiler.export_chrome_trace(str(directory/'trace.json.partial'))
                        monitor.check()
                    except BaseException:
                        (directory/'partial-trace-export-error.txt').write_text(traceback.format_exc())
                raise
            # Wrapper/collector are gone before event resolution/export. The
            # resource monitor remains active through these potentially long calls.
            if not observation.closed or observation.patches: raise RuntimeError('Observer failed to restore')
            write(directory/'completed-batch.json', sample)
            status.update(stage='event_resolution' if repeat==0 else 'trace_export',complete_session=True)
            write(directory/'status.json', status)
            summary = observation.summarize()
            if len(summary['rounds']) != len(sample['rounds']): raise ValueError('Diagnostic round coverage mismatch')
            for a,b in zip(summary['rounds'],sample['rounds']):
                if a['query_tokens'] != b['query_tokens']: raise ValueError('Diagnostic query mismatch')
                a.update(execution_kind=b['execution_kind'],eager_tail=b['eager_tail'])
            write(directory/'spans.json', summary)
            if repeat == 1:
                partial = directory/'trace.json.partial'
                profiler.export_chrome_trace(str(partial))
                monitor.check()  # Includes per-file and total size before parsing.
                status['stage'] = 'trace_validation'; write(directory/'status.json',status)
                trace = analyze_trace(partial, summary)
                write(directory/'gpu-trace-summary.json',trace)
                partial.rename(directory/'trace.json')
            monitor.check()
            status.update(status='completed',stage='complete',observer_restored=True,
                          max_rss_bytes=monitor.max_rss,trace=repeat==1)
            write(directory/'status.json',status)
        return sample
    except BaseException as exc:
        if sample is not None and not (directory/'completed-batch.json').exists(): write(directory/'completed-batch.json',sample)
        if observation is not None:
            # Scalars already observed remain useful even if event resolution,
            # profiler stop/export or the batch itself fails. No fake GPU values.
            write(directory/'partial-observation.json',dict(spans=observation.spans,rounds=observation.rounds,
                observer_restored=observation.closed and not observation.patches))
        status.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        write(directory/'status.json',status); (directory/'error.txt').write_text(traceback.format_exc())
        raise


def validate_diagnostics(out):
    root=Path(out)/'diagnostics'
    expected={f'{case}-{arm}-1/trace.json' for case in BASE.CASE_IDS for arm in BASE.ARMS}
    files={str(p.relative_to(root)):p for p in root.glob('*/trace.json')}
    if set(files)!=expected or list(root.glob('*/trace.json.partial')):
        raise ValueError('Exactly four complete final GPU traces required; partials cannot count')
    sizes={name:p.stat().st_size for name,p in files.items()}
    if any(n>PROFILE_PROTOCOL['trace_bytes'] for n in sizes.values()) or sum(sizes.values())>PROFILE_PROTOCOL['total_trace_bytes']:
        raise ValueError('Final exported trace bounds exceeded')
    for case in BASE.CASE_IDS:
        for arm in BASE.ARMS:
            for repeat in (0,1):
                status=json.loads((root/f'{case}-{arm}-{repeat}'/'status.json').read_text())
                if status.get('status')!='completed' or not status.get('observer_restored') or not status.get('complete_session'):
                    raise ValueError('Every diagnostic must complete and restore')
    return sizes


def run(factory, device, manifest, out, *, deadline=None, setup=None, baseline=None):
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'samples.jsonl').exists(): raise ValueError('Fresh profiling output required')
    GUARD.write(out/'specification.json',dict(protocol=BASE.PROTOCOL,profile_protocol=PROFILE_PROTOCOL,
        expected_samples=BASE.identities(manifest),frozen_vllm_reference=baseline))
    if setup is not None: GUARD.write(out/'runtime.json',setup)
    samples=[]; references={}
    GUARD.write(out/'result.json',dict(status='running',sample_count=0))
    try:
        for phase,repeat,case,arm in BASE.schedule(manifest):
            if deadline is not None and time.monotonic() >= deadline: raise TimeoutError('Profile cooperative deadline')
            if phase == 'diagnostic':
                sample=diagnostic(factory,device,case,manifest,arm,repeat,
                    out/'diagnostics'/f'{case["case_id"]}-{arm}-{repeat}',deadline=deadline,result_path=out/'result.json')
            else:
                # No observer context, wrapper, collector, thread or profiler on
                # the primary/warmup call. Invoke the original function directly.
                sample=BASE.batch(factory,device,case,manifest,arm,diagnostic=False,deadline=deadline)
            signature=invariant(sample); key=(case['case_id'],arm)
            if phase=='primary':
                if key in references and references[key]!=signature: raise ValueError('Primary same-path work/output disagreement')
                references[key]=signature
            if phase=='diagnostic' and references.get(key)!=signature:
                GUARD.write(out/'diagnostics'/f'{case["case_id"]}-{arm}-{repeat}'/'invariant-failure.json',
                    dict(expected=references.get(key),actual=signature))
                raise ValueError('Observation changed same-path output/work/decisions')
            sample.update(phase=phase,repeat=repeat,case_id=case['case_id'],request_count=1,prompt_length=case['prompt_length'])
            samples.append(sample)
            with (out/'samples.jsonl').open('a') as stream: stream.write(json.dumps(sample,allow_nan=False)+'\n')
            GUARD.write(out/'result.json',dict(status='running',sample_count=len(samples)))
        trace_sizes=validate_diagnostics(out)
        result=dict(status='completed',sample_count=len(samples),aggregates=BASE.verify_complete(samples,manifest,baseline),
                    diagnostic_observer_only=True,trace_count=len(trace_sizes),trace_sizes=trace_sizes,all_graph=False,distribution_equivalence_claimed=False,
                    prior_target_numerical_gate='failed_unchanged')
        GUARD.write(out/'result.json',result);return result
    except BaseException as exc:
        GUARD.write(out/'result.json',dict(status='partial_deadline' if isinstance(exc,TimeoutError) else 'failed',
            error_type=type(exc).__name__,error=str(exc),sample_count=len(samples),
            aggregates=BASE.summarize(samples,manifest,baseline),profiling_complete=False))
        raise


def worker(binding_path,out):
    # Only this isolated imported runner module's entry functions change. Its
    # original factory/batch/algorithm implementations remain untouched.
    old_bind,old_run=BASE.bind,BASE.run
    BASE.bind,BASE.run=bind,run
    try:return BASE.worker(binding_path,out)
    finally:BASE.bind,BASE.run=old_bind,old_run


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','generation-manifest','checkpoint','workloads','asr-url'):p.add_argument('--'+name)
    p.add_argument('--output',type=Path);p.add_argument('--asr-pid',type=int)
    p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--worker-binding',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.worker_binding:
        try:return worker(args.worker_binding,args.output)
        except BaseException as exc:
            args.output.mkdir(parents=True,exist_ok=True);(args.output/'error.txt').write_text(traceback.format_exc())
            path=args.output/'result.json';value=json.loads(path.read_text()) if path.exists() else dict(sample_count=0)
            if value.get('status') not in ('failed','partial_deadline'):
                value.update(status='partial_deadline' if isinstance(exc,TimeoutError) else 'failed',error_type=type(exc).__name__,error=str(exc))
            if hasattr(exc,'memory_accounting'):
                value['memory_accounting']=exc.memory_accounting
                GUARD.write(args.output/'failed-graph-pool-snapshot.json',exc.pool_snapshot)
            GUARD.write(path,value);return 1
    if not all((args.model,args.generation_manifest,args.checkpoint)):p.error('Pinned model/data/step1280 required')
    binding=bind(args.model,args.generation_manifest,args.checkpoint,args.workloads)
    if not args.execute:print(json.dumps(dict(status='dry_run_no_backend_import_no_gpu',binding=binding),indent=2));return 0
    if args.dry_run or args.output is None or args.asr_pid is None or not args.asr_url:p.error('Fresh output and ASR identity/health required')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False,mode=0o700);GUARD.write(out/'binding.json',binding)
    guard=GUARD.ASRGuard(args.asr_pid,args.asr_url,pre_free_bytes=BASE.PROTOCOL['minimum_free_bytes'])
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-binding',str(out/'binding.json'),'--output',str(out/'worker')]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2');env.pop('PYTHONPATH',None)
    status=GUARD.supervise(command,out/'supervision',guard,timeout=BASE.PROTOCOL['timeout_seconds'],env=env)
    intact=bind(args.model,args.generation_manifest,args.checkpoint,args.workloads)==binding
    GUARD.write(out/'post-input-integrity.json',dict(passed=intact))
    if not intact or status['status']!='completed' or not status['released']:return 1
    manifest,_=BASE.SHARED.load_workloads(binding['workloads'])
    samples=[json.loads(line) for line in (out/'worker/samples.jsonl').read_text().splitlines()]
    result=json.loads((out/'worker/result.json').read_text())
    if result['aggregates']!=BASE.verify_complete(samples,manifest,binding['frozen_vllm_reference']):raise ValueError('Profile aggregate mismatch')
    return 0 if result['status']=='completed' and result['sample_count']==36 else 1


if __name__=='__main__':raise SystemExit(main())
