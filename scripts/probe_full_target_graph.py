#!/usr/bin/env python3
"""Bounded full pretrained native-eager/full-HF-graph capability probe.

Default is stdlib binding only. Device execution requires a coordinated window.
No sampling, throughput or cross-backend numerical-equivalence claim.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
CAP=module('full_graph_capacity_binding',ROOT/'scripts/probe_native_capacity_graph.py')
GUARD=CAP.GUARD
PROTOCOL=dict(version=1,scope='Full original pretrained HF Qwen embedding/all decoder blocks/final norm graph; head outside',
    backend=CAP.BASE.BACKEND,dtype='bfloat16',transformers='5.17.0',selected_layers=[1,7,14,21,26],
    query_lengths=[1,4],context_ceilings=[144,144],contexts=[[128,128],[129,131],[130,135]],
    committed_query_rows=[[1,3],[1,4],[0,0]],commit_semantics='Artificial transaction fixture, not sampled acceptance',
    source_case='r2-c256',inactive_request=dict(name='idle',source_request=1,token_slice=[200,217]),
    comparisons=dict(atol=.02,rtol=.02,max_rms=.005,reductions='Pooled and each request, every selected raw layer, final norm, logits and each layer K/V'),
    graph_reservation_bytes=512*1024**2,graph_byte_budget=768*1024**2,workspace_byte_budget=256*1024**2,
    timeout_seconds=300,cooperative_reserve_seconds=10,minimum_free_bytes=8*1024**3,
    process_allocation_cap_bytes=6*1024**3,replays=3,
    graph_boundary='Embedding/QKV/QK norm/RoPE/attention/output projections/residuals/MLP/final norm; selected raw copies and scratch/gathers',
    excluded='Prefill, LM head, host metadata, commits, sampling, scheduler',
    python_replay_requirement='Original model forward and every decoder hook counters unchanged during all three GPU replays',
    full_pretrained_graph=True,overlap=False,performance_measurement=False)

def bind(model,generation,workloads=None):
    result=CAP.bind(model,generation,workloads)
    result.update(protocol=PROTOCOL,protocol_sha256=CAP.digest(PROTOCOL))
    result['source_sha256']['scripts/probe_full_target_graph.py']=GUARD.sha(Path(__file__).resolve())
    return result

def new_report():
    return dict(status='running',stages={name:'pending' for name in ('native_eager','capture','graph_replay')},
        observations=[],expected_observations=[dict(stage=stage,case=i) for stage in ('native_eager','graph_replay') for i in range(3)],
        structural_checks=[],performance_measurement=False,distribution_equivalence_claimed=False)

def compare(actual,expected,lengths,*,exact=False):
    """Inputs are flattened request rows. No reduction can hide a failed request."""
    import torch
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise ValueError('Comparison shape/dtype mismatch')
    expected=expected.to(actual.device);rows=[]
    for name,start,end in [('pooled',0,len(actual))]+[(str(i),sum(lengths[:i]),sum(lengths[:i+1])) for i in range(len(lengths))]:
        a,b=actual[start:end],expected[start:end]
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        row=dict(request=name,finite=finite,passed=False)
        if finite:
            d=(a.double()-b.double()).abs();rms=float(d.square().mean().sqrt());limit=PROTOCOL['comparisons']
            row.update(max_abs=float(d.max()),rms=rms,bit_equal=torch.equal(a,b))
            row['passed']=row['bit_equal'] if exact else bool((d<=limit['atol']+limit['rtol']*b.double().abs()).all()) and rms<=limit['max_rms']
        rows.append(row)
    return dict(passed=all(row['passed'] for row in rows),exact_required=exact,rows=rows)

def compare_outputs(actual,expected,target,*,exact=False):
    lengths=PROTOCOL['query_lengths'];width=target.model.config.hidden_size;rows={}
    for index,layer in enumerate(target.layer_ids):
        key=f'selected_raw_layer_{layer}'
        rows[key]=compare(actual['context'][0,:,index*width:(index+1)*width],expected['context'][0,:,index*width:(index+1)*width],lengths,exact=exact)
    for key in ('final_norm','logits'):rows[key]=compare(actual[key][0],expected[key][0],lengths,exact=exact)
    for layer in range(target.pool.layers):
        for key in ('scratch_keys','scratch_values'):
            rows[f'{key}_layer_{layer}']=compare(actual[key][layer],expected[key][layer],lengths,exact=exact)
    return dict(passed=all(row['passed'] for row in rows.values()),tensors=rows)

class ForwardWitness:
    """Hooks count Python execution, and independently identify eager raw layers."""
    def __init__(self,target):
        self.counts=dict(model_forward=0,decoder_hooks=[0]*target.pool.layers);self.raw={};self.norm=None;self.record=False
        def model_hook(*args):self.counts['model_forward']+=1
        def layer_hook(index):
            def hook(module,args,output):
                self.counts['decoder_hooks'][index]+=1
                if self.record and index in target.layer_ids:self.raw[index]=(output[0] if isinstance(output,tuple) else output).detach().clone()
            return hook
        def norm_hook(module,args,output):
            if self.record:self.norm=output.detach().clone()
        self.hooks=[target.model.model.register_forward_pre_hook(model_hook),target.model.model.norm.register_forward_hook(norm_hook)]
        self.hooks.extend(layer.register_forward_hook(layer_hook(i)) for i,layer in enumerate(target.model.model.layers))
    def snapshot(self):return copy.deepcopy(self.counts)
    def close(self):
        for hook in self.hooks:hook.remove()

def save_tensors(out,name,tensors):
    import torch
    path=Path(out)/name;torch.save({key:t.detach().cpu().clone() for key,t in tensors.items()},path)
    return dict(file=name,sha256=GUARD.sha(path))

def feature_tensors(target,features):
    import torch
    logits=target.predict(features)
    return dict(context=features.context.detach().clone(),final_norm=features.last.detach().clone(),
        logits=torch.cat(list(logits.values()),dim=1).detach().clone(),
        scratch_keys=target.pool.scratch_keys[:,:sum(PROTOCOL['query_lengths'])].clone(),
        scratch_values=target.pool.scratch_values[:,:sum(PROTOCOL['query_lengths'])].clone())

def resident(target):return dict(keys=target.pool.keys.clone(),values=target.pool.values.clone())

def exercise(target,requests,out,report,*,require_gpu=True,deadline=None,synchronize=lambda:None):
    """Formal target is pinned BF16 GPU; CPU tests explicitly use the emulator."""
    import torch
    from dspark_qwen.persistent_target_kv import Bucket
    out=Path(out);out.mkdir(parents=True,exist_ok=True);witness=ForwardWitness(target)
    save=lambda:GUARD.write(out/'result.json',report)
    def check(ok,kind,**details):
        report['structural_checks'].append(dict(kind=kind,passed=bool(ok),**details));save()
        if not ok:raise AssertionError('Structural invariant failed: '+kind)
    def unchanged(before):return torch.equal(before['keys'],target.pool.keys) and torch.equal(before['values'],target.pool.values)
    def ids(values):return torch.tensor([values],dtype=torch.long,device=target.device)
    prompts={f'r{i}':ids(r['prompt_token_ids'][:128]) for i,r in enumerate(requests)}
    prompts['idle']=ids(requests[1]['prompt_token_ids'][200:217])
    prefill=Bucket((128,128,17),(0,0,0),'Full-target capability eager prefill')
    bucket=Bucket((1,4),(144,144),'Full-target capability fixed graph family')
    target.register_bucket(prefill);target.register_graph_bucket(bucket,reserve_bytes=PROTOCOL['graph_reservation_bytes'])
    ex=target._graphs[bucket];baseline_pointers=ex.pointers();references=[];committed=[];active='native_eager'
    def prime():
        for name in prompts:target.add_request(name)
        f=target.prefill(prompts,bucket=prefill);target.release_features(f)
    def chunks(index):
        return {f'r{i}':ids(r['prompt_token_ids'][c:c+q]) for i,(r,c,q) in enumerate(zip(requests,PROTOCOL['contexts'][index],PROTOCOL['query_lengths']))}
    def expected_positions(index):
        return torch.cat([torch.arange(c,c+q,device=target.device) for c,q in zip(PROTOCOL['contexts'][index],PROTOCOL['query_lengths'])])
    def finish(f,index,before,own_logits):
        counts=PROTOCOL['committed_query_rows'][index];offset=0;expected={key:value.clone() for key,value in before.items()}
        for name,n,q in zip(('r0','r1'),counts,PROTOCOL['query_lengths']):
            handle=target._handles[name];c=target.lengths[name]
            for key,scratch in (('keys',target.pool.scratch_keys),('values',target.pool.scratch_values)):
                expected[key][:,handle.index,c:c+n].copy_(scratch[:,offset:offset+n])
            offset+=q
        if any(counts):target.commit(f,dict(zip(('r0','r1'),counts)))
        else:target.abort(f)
        check(unchanged(expected),'exact_artificial_commit_or_rollback',stage=active,case=index)
        if active=='graph_replay':
            check(target._feature_lease is f and target._pending is None,'lease_survives_commit_or_abort',case=index)
            refused=False
            try:target.prepare_graph_verify(chunks(index),bucket=bucket)
            except RuntimeError:refused=True
            check(refused,'reuse_blocked_while_features_leased',case=index)
            # Actual LM-head consumer after commit must read the still-owned output.
            check(torch.equal(torch.cat(list(target.predict(f).values()),dim=1),own_logits),
                'post_commit_head_reads_owned_output',case=index)
        target.release_features(f)
        return expected
    try:
        report['runtime_scope']='real_pretrained_gpu' if require_gpu else 'cpu_emulator_contract_only';report['stages'][active]='running';save()
        prime();initial=resident(target);save_tensors(out,'initial-resident.pt',initial)
        for index in range(3):
            CAP.deadline_check(deadline);report['current_attempt']=dict(stage=active,case=index);save()
            check([target.lengths[r] for r in ('r0','r1')]==PROTOCOL['contexts'][index],'declared_contexts',stage=active,case=index)
            before=resident(target);witness.record=True;witness.raw={};witness.norm=None
            save_tensors(out,f'eager-{index}-request-tokens.pt',chunks(index))
            f=target.verify_eager(chunks(index),bucket=bucket);tensors=feature_tensors(target,f);witness.record=False
            evidence=save_tensors(out,f'eager-{index}-outputs.pt',tensors)
            ws=target.pool._workspaces[bucket]
            check(torch.equal(ws.positions,expected_positions(index)),'actual_request_positions',stage=active,case=index)
            save_tensors(out,f'eager-{index}-prepared-metadata.pt',dict(positions=ws.positions,cu_query=ws.cu_query,cu_key=ws.cu_key))
            width=target.model.config.hidden_size
            check(set(witness.raw)==set(target.layer_ids) and all(torch.equal(tensors['context'][:,:,i*width:(i+1)*width],witness.raw[layer]) for i,layer in enumerate(target.layer_ids)),
                'each_selected_raw_layer_identity',case=index)
            check(torch.equal(tensors['final_norm'],witness.norm),'separate_original_final_norm',case=index)
            save_tensors(out,f'eager-{index}-independent-layer-witnesses.pt',dict(**{f'raw_layer_{k}':v for k,v in witness.raw.items()},final_norm=witness.norm))
            check(unchanged(before),'eager_scratch_only',case=index)
            target.abort(f);target.release_features(f);check(unchanged(before),'eager_abort_restores_committed_state',case=index)
            slot=target._handles['idle'].index
            target.pool.keys[:,slot].fill_(100.);target.pool.values[:,slot].fill_(-100.)
            poisoned=resident(target);f=target.verify_eager(chunks(index),bucket=bucket)
            isolated=feature_tensors(target,f);isolation_evidence=save_tensors(out,f'eager-{index}-inactive-poison-outputs.pt',isolated)
            comparison=compare_outputs(isolated,tensors,target,exact=True)
            report['observations'].append(dict(stage=active,case=index,comparison=comparison,outputs=evidence,inactive_poison_outputs=isolation_evidence,
                work=f.work,python_counts=witness.snapshot()));save()
            check(comparison['passed'] and unchanged(poisoned),'inactive_KV_isolation_and_resident_integrity',case=index)
            finish(f,index,poisoned,isolated['logits'])
            target.pool.keys[:,slot].copy_(before['keys'][:,slot]);target.pool.values[:,slot].copy_(before['values'][:,slot])
            after=resident(target);save_tensors(out,f'eager-{index}-committed.pt',after)
            references.append({key:t.cpu().clone() for key,t in tensors.items()});committed.append({key:t.cpu().clone() for key,t in after.items()})
        report['stages'][active]='passed';active='capture';report['stages'][active]='running';save()
        CAP.deadline_check(deadline);target.reset()
        # Fixture reconstruction, outside graph: reset drops ownership but retains old bytes.
        target.pool.keys.zero_();target.pool.values.zero_();prime()
        check(unchanged(initial),'eager_initial_state_reconstructed')
        before=resident(target);counts_before=witness.snapshot()
        save_tensors(out,'capture-request-tokens.pt',chunks(0))
        target.capture_graph(chunks(0),bucket=bucket);synchronize();counts_after=witness.snapshot()
        ws=target.pool._workspaces[bucket]
        save_tensors(out,'capture-prepared-inputs.pt',dict(input_ids=ex.input_ids,positions=ws.positions,cu_query=ws.cu_query,cu_key=ws.cu_key))
        report['capture']=dict(python_before=counts_before,python_after=counts_after,graph_memory=ex.graph_memory,pointers=ex.pointers())
        expected_calls=3 if require_gpu else 1
        check(counts_after['model_forward']-counts_before['model_forward']==expected_calls and
            all(a-b==expected_calls for a,b in zip(counts_after['decoder_hooks'],counts_before['decoder_hooks'])),
            'capture_setup_runs_original_model_and_every_decoder')
        check(unchanged(before) and ex.pointers()==baseline_pointers,'capture_preserves_residents_and_addresses')
        report['stages'][active]='passed';active='graph_replay';report['stages'][active]='running';save()
        for index in range(3):
            CAP.deadline_check(deadline);report['current_attempt']=dict(stage=active,case=index);save()
            check([target.lengths[r] for r in ('r0','r1')]==PROTOCOL['contexts'][index],'declared_contexts',stage=active,case=index)
            before=resident(target);ticket=target.prepare_graph_verify(chunks(index),bucket=bucket)
            ws=target.pool._workspaces[bucket]
            check(torch.equal(ws.positions,expected_positions(index)) and torch.equal(ex.input_ids,torch.cat(list(chunks(index).values()),dim=1)),
                'actual_request_positions_and_token_ids',stage=active,case=index)
            inputs=save_tensors(out,f'replay-{index}-inputs.pt',dict(input_ids=ex.input_ids,positions=ws.positions,cu_query=ws.cu_query,cu_key=ws.cu_key))
            python_before=witness.snapshot();f=target.finish_graph(target.submit_graph(ticket));synchronize();python_after=witness.snapshot()
            tensors=feature_tensors(target,f);outputs=save_tensors(out,f'replay-{index}-outputs.pt',tensors)
            compared=compare_outputs(tensors,references[index],target)
            stable=ex.pointers()==baseline_pointers;pristine=unchanged(before)
            report['observations'].append(dict(stage=active,case=index,comparison=compared,inputs=inputs,outputs=outputs,
                python_before=python_before,python_after=python_after,python_unchanged=python_before==python_after,
                stable_pointers=stable,resident_unchanged_before_commit=pristine,positions=ws.positions.cpu().tolist(),work=f.work));save()
            check(compared['passed'],'full_target_outputs_and_all_layer_scratch_within_fixed_limits',case=index)
            check(stable and pristine,'replay_preserves_residents_and_fixed_addresses',case=index)
            check(python_before==python_after if require_gpu else python_before!=python_after,
                'python_free_real_replay' if require_gpu else 'emulator_explicitly_reruns_python',case=index)
            finish(f,index,before,tensors['logits'])
            after=resident(target);save_tensors(out,f'replay-{index}-committed.pt',after)
            # KV comparison is per layer and per active request; idle bytes exact.
            kv_comparisons=[]
            for name in ('r0','r1'):
                handle=target._handles[name];length=target.lengths[name]
                for layer in range(target.pool.layers):
                    for key in ('keys','values'):
                        result=compare(after[key][layer,handle.index,:length],committed[index][key][layer,handle.index,:length],(length,))
                        kv_comparisons.append(dict(request=name,layer=layer,kind_kv=key,comparison=result))
            check(all(r['comparison']['passed'] for r in kv_comparisons),'committed_KV_fixed_limits',case=index,comparisons=kv_comparisons)
            slot=target._handles['idle'].index
            check(all(torch.equal(after[key][:,slot],initial[key][:,slot]) for key in after),'inactive_resident_exact',case=index)
        report['stages'][active]='passed';report['status']='completed';save();return report
    except BaseException as exc:
        report['stages'][active]='failed';report['status']='partial_deadline' if isinstance(exc,TimeoutError) else 'failed'
        report.update(error_type=type(exc).__name__,error=str(exc))
        if hasattr(exc,'memory_accounting'):report['failed_graph_memory_accounting']=exc.memory_accounting
        if hasattr(exc,'pool_snapshot'):GUARD.write(out/'failed-graph-pool-snapshot.json',exc.pool_snapshot)
        save()
        try:
            save_tensors(out,'failed-resident-and-scratch.pt',dict(**resident(target),scratch_keys=target.pool.scratch_keys,scratch_values=target.pool.scratch_values))
            if active=='capture':
                ws=target.pool._workspaces[bucket]
                save_tensors(out,'failed-capture-buffer-state.pt',dict(input_ids=ex.input_ids,positions=ws.positions,cu_query=ws.cu_query,cu_key=ws.cu_key))
        except BaseException as evidence_error:
            report['failure_tensor_write_error']=str(evidence_error);save()
        raise
    finally:witness.close()

def verify_report(report,*,require_gpu=True):
    expected=[dict(stage=stage,case=i) for stage in ('native_eager','graph_replay') for i in range(3)]
    if report['status']!='completed' or any(v!='passed' for v in report['stages'].values()):raise ValueError('Incomplete full-target stages')
    if report['expected_observations']!=expected or [{k:r[k] for k in ('stage','case')} for r in report['observations']]!=expected:raise ValueError('Incomplete observation domain')
    if not report['structural_checks'] or any(not r['passed'] for r in report['structural_checks']):raise ValueError('Failed structural checks')
    for row in report['observations']:
        if not row['comparison']['passed']:raise ValueError('Failed numerical comparison')
        if row['stage']=='graph_replay' and (not row['stable_pointers'] or not row['resident_unchanged_before_commit'] or (require_gpu and not row['python_unchanged'])):raise ValueError('Invalid graph replay evidence')
    if require_gpu and (report['runtime_scope']!='real_pretrained_gpu' or report['capture']['graph_memory']['kind']!='graph_private_pool_reserved_segments'):raise ValueError('Real GPU/private-pool evidence required')

def worker(binding_path,out):
    start=time.monotonic();b=json.loads(Path(binding_path).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report=new_report();GUARD.write(out/'result.json',report);GUARD.write(out/'protocol.json',PROTOCOL)
    try:
        if bind(b['model'],b['generation_manifest'],b['workloads'])!=b:raise ValueError('Bound source/input changed')
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        sys.path.insert(0,str(ROOT))
        from dspark_qwen.persistent_qwen_target import PersistentQwenTarget
        from dspark_qwen.persistent_qwen_graph import TorchGraphBackend
        from dspark_qwen.rocm_varlen import PinnedRocmVarlenKernel
        if transformers.__version__!=PROTOCOL['transformers']:raise RuntimeError('Pinned Transformers required')
        device=torch.device('cuda:0');torch.cuda.set_device(device);kernel=PinnedRocmVarlenKernel(device)
        free,total=torch.cuda.mem_get_info(device)
        if free<PROTOCOL['minimum_free_bytes']:raise RuntimeError('Insufficient free memory')
        torch.cuda.set_per_process_memory_fraction(PROTOCOL['process_allocation_cap_bytes']/total,device)
        GUARD.write(out/'runtime.json',dict(runtime=kernel.runtime,transformers=transformers.__version__,scope=PROTOCOL['scope']))
        model=AutoModelForCausalLM.from_pretrained(b['model'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        target=PersistentQwenTarget(model,PROTOCOL['selected_layers'],slots=3,context_capacity=148,max_query_tokens=273,
            native_backend=PROTOCOL['backend'],graph_backend=TorchGraphBackend(),max_graph_buckets=1,
            graph_byte_budget=PROTOCOL['graph_byte_budget'],workspace_byte_budget=PROTOCOL['workspace_byte_budget'])
        with torch.no_grad():exercise(target,b['requests'],out,report,deadline=start+PROTOCOL['timeout_seconds']-PROTOCOL['cooperative_reserve_seconds'],synchronize=lambda:torch.cuda.synchronize(device))
        verify_report(report);target.close()
        intact=bind(b['model'],b['generation_manifest'],b['workloads'])==b;GUARD.write(out/'post-input-integrity.json',{'passed':intact})
        if not intact:raise ValueError('Inputs changed during probe')
        return 0
    except BaseException as exc:
        if report['status'] in ('running','completed'):report['status']='failed'
        report.update(error_type=type(exc).__name__,error=str(exc));GUARD.write(out/'result.json',report)
        (out/'error.txt').write_text(traceback.format_exc());return 1

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','generation-manifest','workloads','asr-url'):p.add_argument('--'+name)
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
    verify_report(json.loads((out/'worker/result.json').read_text()));return 0

if __name__=='__main__':raise SystemExit(main())
