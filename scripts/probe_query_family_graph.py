#!/usr/bin/env python3
"""Bounded variable-Q full-target graph capability probe; default is stdlib binding.

Additive protocol: the old fixed-Q probe and its historical limits are unchanged.
No sampling, speed, stock-HF equivalence, capacity scheduling or overlap claim.
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
BASE=module('query_family_full_target_helpers',ROOT/'scripts/probe_full_target_graph.py')
CAP=BASE.CAP
GUARD=BASE.GUARD
PROTOCOL=dict(version=1,scope='Variable ordered Q in finite physical-B full original pretrained HF target graphs',
    backend=BASE.PROTOCOL['backend'],dtype='bfloat16',transformers='5.17.0',selected_layers=[1,7,14,21,26],
    source_case='r2-c256',initial_contexts=[128,128],
    inactive_request=dict(name='idle',source_request=1,token_slice=[200,217]),
    families=[dict(name='S3',request_count=2,query_tokens=3,maximum_query_length=3,key_capacity=291,maximum_key_length=148,graph=True),
              dict(name='L5',request_count=2,query_tokens=5,maximum_query_length=5,key_capacity=293,maximum_key_length=148,graph=True),
              dict(name='E2',request_count=2,query_tokens=2,maximum_query_length=2,key_capacity=290,maximum_key_length=148,graph=False)],
    cases=[dict(family='S3',order=['r0','r1'],query_lengths=[1,2],contexts=[128,128],commit=[1,1]),
           dict(family='L5',order=['r0','r1'],query_lengths=[1,4],contexts=[129,129],commit=[1,3]),
           dict(family='E2',order=['r0','r1'],query_lengths=[1,1],contexts=[130,132],commit=[1,1]),
           dict(family='L5',order=['r0','r1'],query_lengths=[2,3],contexts=[131,133],commit=[1,2]),
           dict(family='L5',order=['r1','r0'],query_lengths=[4,1],contexts=[135,132],commit=[2,1]),
           dict(family='S3',order=['r0','r1'],query_lengths=[2,1],contexts=[133,137],commit=[0,0])],
    capture_examples=[dict(family='S3',order=['r0','r1'],query_lengths=[1,2],contexts=[128,128]),
                      dict(family='L5',order=['r0','r1'],query_lengths=[1,4],contexts=[128,128])],
    comparisons=copy.deepcopy(BASE.PROTOCOL['comparisons']),
    commit_semantics='Artificial transaction fixture, not sampled acceptance',
    graph_reservation_bytes=512*1024**2,graph_byte_budget=1280*1024**2,workspace_byte_budget=64*1024**2,
    timeout_seconds=300,cooperative_reserve_seconds=10,minimum_free_bytes=8*1024**3,
    process_allocation_cap_bytes=6*1024**3,graph_programs=2,replays=5,explicit_eager_cases=1,
    graph_boundary=BASE.PROTOCOL['graph_boundary'],excluded=BASE.PROTOCOL['excluded'],
    full_pretrained_graph=True,overlap=False,performance_measurement=False)


def bind(model,generation,workloads=None):
    result=BASE.bind(model,generation,workloads)
    result.update(protocol=PROTOCOL,protocol_sha256=CAP.digest(PROTOCOL))
    result['source_sha256']['scripts/probe_query_family_graph.py']=GUARD.sha(Path(__file__).resolve())
    return result


def expected_observations():
    return [dict(stage=stage,case=i,family=case['family']) for stage in ('native_eager','mixed_execution') for i,case in enumerate(PROTOCOL['cases'])]


def new_report():
    return dict(status='running',stages={name:'pending' for name in ('native_eager','capture','mixed_execution')},
        observations=[],captures=[],expected_observations=expected_observations(),structural_checks=[],
        performance_measurement=False,distribution_equivalence_claimed=False)


def feature_tensors(target,features):
    import torch
    q=features.last.shape[1]
    return dict(context=features.context.detach().clone(),final_norm=features.last.detach().clone(),
        logits=torch.cat(list(target.predict(features).values()),dim=1).detach().clone(),
        scratch_keys=target.pool.scratch_keys[:,:q].clone(),scratch_values=target.pool.scratch_values[:,:q].clone())


def compare_outputs(actual,expected,target,lengths,*,exact=False):
    width=target.model.config.hidden_size;rows={}
    for index,layer in enumerate(target.layer_ids):
        rows[f'selected_raw_layer_{layer}']=BASE.compare(actual['context'][0,:,index*width:(index+1)*width],
            expected['context'][0,:,index*width:(index+1)*width],lengths,exact=exact)
    for key in ('final_norm','logits'):rows[key]=BASE.compare(actual[key][0],expected[key][0],lengths,exact=exact)
    for layer in range(target.pool.layers):
        for key in ('scratch_keys','scratch_values'):
            rows[f'{key}_layer_{layer}']=BASE.compare(actual[key][layer],expected[key][layer],lengths,exact=exact)
    return dict(passed=all(r['passed'] for r in rows.values()),tensors=rows)


def metadata(target,family):
    ws=target.pool._workspaces[family]
    return {name:getattr(ws,name) for name in ('positions','cu_query','cu_key','committed_indices','scratch_indices','from_scratch','valid')}


def exercise(target,requests,out,report,*,require_gpu=True,deadline=None,synchronize=lambda:None):
    import torch
    from dspark_qwen.persistent_target_kv import Bucket,QueryFamily
    out=Path(out);out.mkdir(parents=True,exist_ok=True);witness=BASE.ForwardWitness(target)
    save=lambda:GUARD.write(out/'result.json',report)
    def check(ok,kind,**details):
        report['structural_checks'].append(dict(kind=kind,passed=bool(ok),**details));save()
        if not ok:raise AssertionError('Structural invariant failed: '+kind)
    def unchanged(before):return all(torch.equal(before[k],getattr(target.pool,k)) for k in ('keys','values'))
    def ids(values):return torch.tensor([values],dtype=torch.long,device=target.device)
    source={f'r{i}':r['prompt_token_ids'] for i,r in enumerate(requests)}
    prompts={name:ids(tokens[:128]) for name,tokens in source.items()};prompts['idle']=ids(source['r1'][200:217])
    def chunks(case):return {name:ids(source[name][c:c+q]) for name,c,q in zip(case['order'],case['contexts'],case['query_lengths'])}
    families={};references=[];committed=[];active='native_eager';current_family=None
    def prime():
        for name in prompts:target.add_request(name)
        f=target.prefill(prompts,bucket=prefill);target.release_features(f)
    def prepared(case,family,index):
        ws=target.pool._workspaces[family];old=[];new=[];choose=[];positions=[];cq=[0];ck=[0]
        for name,c,q in zip(case['order'],case['contexts'],case['query_lengths']):
            slot=target._handles[name].index;offset=cq[-1]
            old.extend([slot*target.pool.context_capacity+j for j in range(c)]+[0]*q)
            new.extend([0]*c+list(range(offset,offset+q)));choose.extend([False]*c+[True]*q)
            positions.extend(range(c,c+q));cq.append(offset+q);ck.append(ck[-1]+c+q)
        tail=family.key_capacity-ck[-1];old.extend([0]*tail);new.extend([0]*tail);choose.extend([False]*tail)
        expected=dict(positions=positions,cu_query=cq,cu_key=ck,committed_indices=old,scratch_indices=new,
            from_scratch=choose,valid=[True]*ck[-1]+[False]*tail)
        check(all(torch.equal(getattr(ws,k),torch.tensor(v,device=target.device,dtype=getattr(ws,k).dtype).reshape(getattr(ws,k).shape)) for k,v in expected.items()),
            'actual_ordered_metadata_and_gathers',stage=active,case=index)
        check(sum(case['query_lengths'])==family.query_tokens and max(case['query_lengths'])<family.maximum_query_length
            and max(c+q for c,q in zip(case['contexts'],case['query_lengths']))<family.maximum_key_length,
            'actual_B_and_strict_fixed_launch_bounds',stage=active,case=index)
    def finish(features,case,before,own_logits,index,graph):
        expected={key:value.clone() for key,value in before.items()};offset=0
        for name,c,q,n in zip(case['order'],case['contexts'],case['query_lengths'],case['commit']):
            slot=target._handles[name].index
            for key,scratch in (('keys',target.pool.scratch_keys),('values',target.pool.scratch_values)):
                expected[key][:,slot,c:c+n].copy_(scratch[:,offset:offset+n])
            offset+=q
        if any(case['commit']):target.commit(features,dict(zip(case['order'],case['commit'])))
        else:target.abort(features)
        check(unchanged(expected),'exact_own_commit_or_abort',stage=active,case=index)
        if graph:
            check(target._feature_lease is features and target._pending is None,'lease_survives_commit_or_abort',case=index)
            refused=False
            other=dict(order=['r0','r1'],query_lengths=[1,1],contexts=[target.lengths[r] for r in ('r0','r1')])
            try:target.verify_eager(chunks(other),bucket=families['E2'])
            except RuntimeError:refused=True
            check(refused,'shared_arena_switch_blocked_while_leased',case=index)
            check(torch.equal(torch.cat(list(target.predict(features).values()),dim=1),own_logits),'post_commit_head_reads_owned_output',case=index)
        target.release_features(features)
    try:
        report['runtime_scope']='real_pretrained_gpu' if require_gpu else 'cpu_emulator_contract_only'
        report['stages'][active]='running';save()
        prefill=Bucket((128,128,17),(0,0,0),'Variable-Q capability eager prefill');target.register_bucket(prefill)
        for row in PROTOCOL['families']:
            families[row['name']]=QueryFamily(**{k:v for k,v in row.items() if k not in ('name','graph')},source='Frozen variable-Q capability '+row['name'])
        target.register_families(tuple(families.values()))
        for row in PROTOCOL['families']:
            if row['graph']:target.register_graph_bucket(families[row['name']],reserve_bytes=PROTOCOL['graph_reservation_bytes'])
        baseline={name:target.pool.pointers(f) for name,f in families.items()}
        graph_pointers={name:target._graphs[f].pointers() for name,f in families.items() if f in target._graphs}
        check(all(baseline[n]==baseline['S3'] for n in families),'families_share_one_stable_arena')
        report['memory_declarations']=dict(workspace_bytes=target._workspace_bytes,graph_reserved_bytes=target._graph_reserved_bytes,
            shared_arena_bytes=target.pool.workspace_bytes(target.pool._family_arena.bucket),graph_programs=len(target._graphs))
        prime();initial=BASE.resident(target);BASE.save_tensors(out,'initial-resident.pt',initial)
        for index,case in enumerate(PROTOCOL['cases']):
            CAP.deadline_check(deadline);current_family=families[case['family']];report['current_attempt']=dict(stage=active,case=index,family=case['family']);save()
            check([target.lengths[r] for r in case['order']]==case['contexts'],'declared_contexts',stage=active,case=index)
            before=BASE.resident(target);witness.record=True;witness.raw={};witness.norm=None
            BASE.save_tensors(out,f'eager-{index}-request-tokens.pt',chunks(case))
            f=target.verify_eager(chunks(case),bucket=current_family);values=feature_tensors(target,f);witness.record=False
            evidence=BASE.save_tensors(out,f'eager-{index}-outputs.pt',values)
            BASE.save_tensors(out,f'eager-{index}-metadata.pt',metadata(target,current_family));prepared(case,current_family,index)
            BASE.save_tensors(out,f'eager-{index}-independent-layer-witnesses.pt',dict(**{f'raw_layer_{k}':v for k,v in witness.raw.items()},final_norm=witness.norm))
            width=target.model.config.hidden_size
            check(set(witness.raw)==set(target.layer_ids) and all(torch.equal(values['context'][:,:,i*width:(i+1)*width],witness.raw[layer]) for i,layer in enumerate(target.layer_ids)),
                'each_selected_raw_layer_identity',case=index)
            check(torch.equal(values['final_norm'],witness.norm),'separate_original_final_norm',case=index)
            check(unchanged(before),'eager_scratch_only',case=index)
            target.abort(f);target.release_features(f);check(unchanged(before),'eager_abort_resident_integrity',case=index)
            slot=target._handles['idle'].index;target.pool.keys[:,slot].fill_(100.);target.pool.values[:,slot].fill_(-100.)
            poisoned=BASE.resident(target);f=target.verify_eager(chunks(case),bucket=current_family);isolated=feature_tensors(target,f)
            isolation=BASE.save_tensors(out,f'eager-{index}-inactive-poison-outputs.pt',isolated)
            compared=compare_outputs(isolated,values,target,case['query_lengths'],exact=True)
            report['observations'].append(dict(stage=active,case=index,family=case['family'],comparison=compared,outputs=evidence,
                inactive_poison_outputs=isolation,work=f.work));save()
            check(compared['passed'] and unchanged(poisoned),'inactive_isolation_and_resident_integrity',case=index)
            finish(f,case,poisoned,isolated['logits'],index,False)
            target.pool.keys[:,slot].copy_(before['keys'][:,slot]);target.pool.values[:,slot].copy_(before['values'][:,slot])
            after=BASE.resident(target);BASE.save_tensors(out,f'eager-{index}-committed.pt',after)
            references.append({k:v.cpu().clone() for k,v in values.items()});committed.append({k:v.cpu().clone() for k,v in after.items()})
        report['stages'][active]='passed';active='capture';report['stages'][active]='running';save()
        target.reset();target.pool.keys.zero_();target.pool.values.zero_();prime();check(unchanged(initial),'initial_state_reconstructed')
        for example in PROTOCOL['capture_examples']:
            CAP.deadline_check(deadline);name=example['family'];current_family=families[name];ex=target._graphs[current_family]
            report['current_attempt']=dict(stage=active,family=name);save();before=BASE.resident(target);counts_before=witness.snapshot()
            BASE.save_tensors(out,f'capture-{name}-request-tokens.pt',chunks(example))
            target.capture_graph(chunks(example),bucket=current_family);synchronize();counts_after=witness.snapshot()
            BASE.save_tensors(out,f'capture-{name}-inputs.pt',dict(input_ids=ex.input_ids,**metadata(target,current_family)))
            prepared(example,current_family,name)
            report['captures'].append(dict(family=name,python_before=counts_before,python_after=counts_after,graph_memory=ex.graph_memory,pointers=ex.pointers()))
            expected=3 if require_gpu else 1
            check(counts_after['model_forward']-counts_before['model_forward']==expected and all(a-b==expected for a,b in zip(counts_after['decoder_hooks'],counts_before['decoder_hooks'])),
                'capture_runs_original_model_every_layer',family=name)
            check(unchanged(before) and ex.pointers()==graph_pointers[name],'capture_preserves_residents_and_addresses',family=name)
        report['stages'][active]='passed';active='mixed_execution';report['stages'][active]='running';save()
        for index,case in enumerate(PROTOCOL['cases']):
            CAP.deadline_check(deadline);name=case['family'];current_family=families[name];is_graph=current_family in target._graphs
            report['current_attempt']=dict(stage=active,case=index,family=name);save()
            check([target.lengths[r] for r in case['order']]==case['contexts'],'declared_contexts',stage=active,case=index)
            before=BASE.resident(target);python_before=witness.snapshot()
            BASE.save_tensors(out,f'mixed-{index}-request-tokens.pt',chunks(case))
            if is_graph:
                ticket=target.prepare_graph_verify(chunks(case),bucket=current_family)
                ex=target._graphs[current_family]
                inputs=BASE.save_tensors(out,f'mixed-{index}-inputs.pt',dict(input_ids=ex.input_ids,**metadata(target,current_family)))
                prepared(case,current_family,index)
                check(torch.equal(ex.input_ids,torch.cat(list(chunks(case).values()),dim=1)),'actual_graph_token_ids',case=index)
                f=target.finish_graph(target.submit_graph(ticket))
            else:
                f=target.verify(chunks(case),bucket=current_family)
                inputs=BASE.save_tensors(out,f'mixed-{index}-inputs.pt',metadata(target,current_family));prepared(case,current_family,index)
            synchronize();python_after=witness.snapshot();values=feature_tensors(target,f)
            outputs=BASE.save_tensors(out,f'mixed-{index}-outputs.pt',values)
            compared=compare_outputs(values,references[index],target,case['query_lengths'])
            stable=all(target.pool.pointers(fam)==baseline[n] for n,fam in families.items())
            stable=stable and all(target._graphs[families[n]].pointers()==p for n,p in graph_pointers.items())
            pristine=unchanged(before)
            expected_calls=0 if is_graph and require_gpu else 1
            counts_ok=python_after['model_forward']-python_before['model_forward']==expected_calls and all(a-b==expected_calls for a,b in zip(python_after['decoder_hooks'],python_before['decoder_hooks']))
            report['observations'].append(dict(stage=active,case=index,family=name,comparison=compared,inputs=inputs,outputs=outputs,
                execution='graph_replay' if is_graph else 'explicit_eager',python_before=python_before,python_after=python_after,
                python_calls_match=counts_ok,stable_pointers=stable,resident_unchanged_before_commit=pristine,work=f.work));save()
            check(compared['passed'],'outputs_and_all_layer_scratch_fixed_limits',case=index)
            check(stable and pristine,'stable_addresses_and_scratch_only',case=index)
            check(counts_ok,'python_free_real_replay' if is_graph and require_gpu else 'explicit_python_execution_count',case=index)
            check(f.work['physical_query_tokens']==sum(case['query_lengths']) and f.work['maximum_query_length']==current_family.maximum_query_length
                  and f.work['maximum_key_length']==current_family.maximum_key_length,'physical_B_and_fixed_launch_work',case=index)
            finish(f,case,before,values['logits'],index,is_graph)
            after=BASE.resident(target);BASE.save_tensors(out,f'mixed-{index}-committed.pt',after);kv=[]
            for name in ('r0','r1'):
                slot=target._handles[name].index;length=target.lengths[name]
                for layer in range(target.pool.layers):
                    for key in ('keys','values'):
                        kv.append(dict(request=name,layer=layer,kind_kv=key,comparison=BASE.compare(after[key][layer,slot,:length],committed[index][key][layer,slot,:length],(length,))))
            check(all(row['comparison']['passed'] for row in kv),'committed_KV_fixed_limits',case=index,comparisons=kv)
            slot=target._handles['idle'].index;check(all(torch.equal(after[key][:,slot],initial[key][:,slot]) for key in after),'inactive_resident_exact',case=index)
        report['stages'][active]='passed';report['status']='completed';save();return report
    except BaseException as exc:
        report['stages'][active]='failed';report['status']='partial_deadline' if isinstance(exc,TimeoutError) else 'failed'
        report.update(error_type=type(exc).__name__,error=str(exc))
        if hasattr(exc,'memory_accounting'):report['failed_graph_memory_accounting']=exc.memory_accounting
        if hasattr(exc,'pool_snapshot'):GUARD.write(out/'failed-graph-pool-snapshot.json',exc.pool_snapshot)
        save()
        try:
            BASE.save_tensors(out,'failed-resident-and-scratch.pt',dict(**BASE.resident(target),scratch_keys=target.pool.scratch_keys,scratch_values=target.pool.scratch_values))
            if current_family is not None:
                values=metadata(target,current_family)
                if current_family in target._graphs:values=dict(input_ids=target._graphs[current_family].input_ids,**values)
                BASE.save_tensors(out,'failed-prepared-buffer-state.pt',values)
        except BaseException as evidence_error:report['failure_tensor_write_error']=str(evidence_error);save()
        raise
    finally:witness.close()


def verify_report(report,*,require_gpu=True):
    if report['status']!='completed' or any(v!='passed' for v in report['stages'].values()):raise ValueError('Incomplete variable-Q stages')
    expected=expected_observations()
    if report['expected_observations']!=expected or [{k:r[k] for k in ('stage','case','family')} for r in report['observations']]!=expected:raise ValueError('Incomplete variable-Q domain')
    if [r['family'] for r in report['captures']]!=['S3','L5']:raise ValueError('Both graph programs required')
    if not report['structural_checks'] or any(not r['passed'] for r in report['structural_checks']):raise ValueError('Failed structural checks')
    def calls(row,count):
        a,b=row['python_after'],row['python_before']
        return (a['model_forward']-b['model_forward']==count and len(a['decoder_hooks'])==len(b['decoder_hooks'])>0
                and all(x-y==count for x,y in zip(a['decoder_hooks'],b['decoder_hooks'])))
    if any(not calls(row,3 if require_gpu else 1) for row in report['captures']):raise ValueError('Capture Python counters differ')
    for row in report['observations']:
        if not row['comparison']['passed']:raise ValueError('Failed numerical comparison')
        if row['stage']=='mixed_execution':
            wanted='explicit_eager' if row['family']=='E2' else 'graph_replay'
            if row['execution']!=wanted or not all(row[k] for k in ('python_calls_match','stable_pointers','resident_unchanged_before_commit')):raise ValueError('Invalid execution identity')
            if not calls(row,0 if require_gpu and wanted=='graph_replay' else 1):raise ValueError('Replay/eager Python counters differ')
            family=next(f for f in PROTOCOL['families'] if f['name']==row['family']);work=row['work']
            if (work['query_lengths']!=PROTOCOL['cases'][row['case']]['query_lengths']
                    or work['physical_query_tokens']!=family['query_tokens']
                    or work['maximum_query_length']!=family['maximum_query_length']
                    or work['maximum_key_length']!=family['maximum_key_length']):raise ValueError('Actual family work differs')
    if require_gpu:
        if report['runtime_scope']!='real_pretrained_gpu' or any(r['graph_memory']['kind']!='graph_private_pool_reserved_segments' for r in report['captures']):raise ValueError('Real GPU independent private-pool evidence required')
        pools=[tuple(r['graph_memory']['pool_id']) for r in report['captures']]
        if len(set(pools))!=2:raise ValueError('Independent graph pools required')

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
            native_backend=PROTOCOL['backend'],graph_backend=TorchGraphBackend(),max_graph_buckets=2,
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
