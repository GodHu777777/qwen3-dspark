#!/usr/bin/env python3
"""Bounded trained-draft noncausal and integrated packed-decoder gate.

Default stdlib-only binding. Formal CLI never accepts a tiny model/checkpoint.
Normal layer numerics, structural content and endpoint comparison are separate.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
_SPEC=importlib.util.spec_from_file_location('whole_gate_helpers',ROOT/'scripts/probe_qwen_varlen.py')
BASE=importlib.util.module_from_spec(_SPEC);_SPEC.loader.exec_module(BASE)
sha,digest,save=BASE.sha,BASE.digest,BASE.save
PROTOCOL=dict(version=1,checkpoint_step=1280,
    checkpoint_weights_sha256='d6f21ab3af5187ed181efdbde54118a9ada21b458d9957515a17d6b3693d0de4',
    checkpoint_metadata_sha256='8bb0c5113d02decac5b8a64c585284dd1664f1a5f88f13014713ec1facc7ecc5',
    target_backend='rocm_aten_no_window_pinned_v1',draft_backend='rocm_aten_no_window_noncausal_pinned_v1',
    target_dtype='bfloat16',draft_parameters='float32',draft_autocast='bfloat16',transformers='5.17.0',
    probability_policy='float64_softmax_normalize_cdf_v1',temperature=1.,eos_ids=[],max_new_tokens=64,
    seeds=dict(A=202610091,B=202610092,C=202610093),readded_C_seed=202610094,
    prompts=dict(A=list(range(100,116)),B=list(range(200,221)),C=list(range(300,313))),readded_C_prompt=[313],
    rounds=[dict(name='shadow_one',mode='shadow',allocations=dict(A=1,B=7,C=0)),
            dict(name='shadow_readd',mode='shadow',allocations=dict(A=0,B=1,C=3)),
            dict(name='fixed_zero',mode='fixed_budget',allocations=dict(A=0,B=2))],
    poison_pairs=[dict(name='inactive_C',active=['A'],victim='C'),dict(name='active_B',active=['A','B'],victim='B')],
    poison_key=100.,poison_value=-100.,attention_atol=.02,attention_rtol=.02,attention_max_rms=.005,
    timeout_seconds=300,minimum_free_bytes=8*1024**3,process_allocation_cap_bytes=6*1024**3,
    target_forwards=5,draft_forwards=7,normal_draft_forwards=3,
    scope='Synthetic fixed pre-draw allocations. No quality/final-test reads, asynchronous overlap or speed claim',
    numerical_policy='Every normal draft layer pooled and per request against actual same-QKV FP32 noncausal MATH; finite failures continue',
    endpoint_policy='Same actual native input chunks and committed lengths replayed on independent SDPA; differences quantified only',
    prior_target_gate='Whole-Qwen target RMS failure retained; not reclassified by this gate')



def validate_limits():
    keys=('attention_atol','attention_rtol','attention_max_rms')
    if any(PROTOCOL[k]!=BASE.PROTOCOL[k] for k in keys):
        raise ValueError('Bound draft protocol and fixed numerical helper thresholds differ')


def source_identity():
    return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/probe_qwen_varlen.py',*sorted((ROOT/'dspark_qwen').glob('*.py'))]}


def bind(model,generation_manifest,checkpoint):
    validate_limits()
    target=BASE.bind(model,generation_manifest);checkpoint=Path(checkpoint).resolve()
    metadata=json.loads((checkpoint/'metadata.json').read_text())
    weights=sha(checkpoint/'draft.safetensors');meta=sha(checkpoint/'metadata.json')
    if (metadata['step']!=1280 or weights!=PROTOCOL['checkpoint_weights_sha256'] or meta!=PROTOCOL['checkpoint_metadata_sha256'] or
            metadata['draft_weights_sha256']!=weights or metadata['identity']['target_fingerprint']!=target['target_fingerprint']):
        raise ValueError('Require the exact selected trained step1280 checkpoint and matching actual target')
    spec=metadata['draft_config']
    if spec['num_layers']!=5 or spec['block_size']!=7 or spec['layer_ids']!=[1,7,14,21,26]:
        raise ValueError('Unexpected selected draft architecture')
    return dict(model=target['model'],generation_manifest=target['generation_manifest'],generation_manifest_sha256=target['generation_manifest_sha256'],
        target_fingerprint=target['target_fingerprint'],checkpoint=str(checkpoint),checkpoint_weights_sha256=weights,
        checkpoint_metadata_sha256=meta,draft_config=spec,protocol=PROTOCOL,protocol_sha256=digest(PROTOCOL),source_sha256=source_identity())


def new_report():
    return dict(status='running',structural_status='pending',draft_layer_numerical_status='pending',endpoint_comparison_status='pending',
        normal_layers=[],structural_checks=[],target_comparisons=[],rounds=[],poison_pairs=[],system_pass_claimed=False)


def exercise(session,reference_model,out,report,*,require_native_events=False):
    import contextlib
    import torch
    from torch.nn.attention import SDPBackend,sdpa_kernel
    from dspark_qwen.cached_target import CachedTarget
    from dspark_qwen.packed_sampling import RequestSpec
    from dspark_qwen.tensor_sampling import TensorRandom
    validate_limits()
    out=Path(out);target=session.target;packed=session.draft;draft=packed.draft;device=target.device
    report['runtime_scope']='actual_pretrained_trained_gpu' if require_native_events else 'tiny_CPU_fixture_only'
    report['prior_target_whole_qwen_gate']='failed_unchanged'
    state=dict(stage='prime',phase='normal',draft_calls=0,target_calls=0,draft_layer=0,attention_inputs={})
    refs={};original_target_append=target.append;original_predict=target.predict
    original_backbone=packed.backbone;original_kernel=packed.kernel;original_context=packed.append_committed
    def persist():save(out/'worker-result.json',report)
    def cpu(t):return t.detach().cpu().clone()
    def evidence(name,tensors):
        path=out/name;torch.save({k:cpu(v) for k,v in tensors.items()},path);return dict(file=path.name,sha256=sha(path))
    def structural(ok,kind,**values):
        report['structural_checks'].append(dict(passed=bool(ok),kind=kind,stage=state['stage'],**values));persist()
        if not ok:raise AssertionError(kind)
    def profiled(call,expected,kind):
        ctx=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if require_native_events else contextlib.nullcontext()
        with ctx as events:
            result=call()
            if require_native_events:torch.cuda.synchronize(device)
        if require_native_events:
            count=sum(e.count for e in events.key_averages() if e.key=='aten::_flash_attention_forward')
            structural(count==expected,kind,native_flash_calls=count,expected=expected)
        return result
    def append(chunks):
        state['target_calls']+=1
        state['native_chunks']={r:t.clone() for r,t in chunks.items()}
        state['native_features']=profiled(lambda:original_target_append(chunks),target.model.config.num_hidden_layers,'one_packed_target_forward')
        return state['native_features']
    def predict(features,last_only=False):
        result=original_predict(features,last_only=last_only);state['native_logits']=result;return result
    def context(chunks):
        before={r:packed.request_kv(r) for r in packed.lengths};lengths=packed.lengths
        for r,value in chunks.items():
            expected=state['native_features'].for_request(r).context[:,:value.shape[1]]
            structural(torch.equal(value,expected),'only_actual_committed_target_features_projected',request=r,tokens=value.shape[1])
        result=original_context(chunks)
        for r,old in before.items():BASE.assert_kv_equal(packed.request_kv(r),old,prefix=lengths[r] if r in chunks else None)
        structural(True,'draft_append_old_prefix_and_inactive_KV_exact')
        return result
    def backbone(anchors):
        state['draft_calls']+=1;state['draft_layer']=0;state['draft_requests']=list(anchors);state['attention_inputs']={}
        result=profiled(lambda:original_backbone(anchors),len(draft.layers),'one_flat_draft_backbone')
        structural(state['draft_layer']==len(draft.layers),'one_noncausal_callback_per_draft_layer',callbacks=state['draft_layer'])
        return result
    def kernel(q,k,v,layout,*,scale):
        index=state['draft_layer'];state['draft_layer']+=1
        row=dict(stage=state['stage'],layer=index,query_lengths=list(layout.query_lengths),key_lengths=list(layout.key_lengths))
        tensors=dict(q=q,k=k,v=v,cu_query=layout.cu_query,cu_key=layout.cu_key)
        try:
            output=original_kernel(q,k,v,layout,scale=scale);tensors['output']=output
            if not all(bool(torch.isfinite(t).all()) for t in (q,k,v,output)):raise RuntimeError('Nonfinite draft attention tensors')
            if state['phase']=='poison':
                nq,nk=layout.query_lengths[0],layout.key_lengths[0]
                state['attention_inputs'].update({f'layer{index}_{name}':cpu(t) for name,t in [('q',q[:nq]),('k',k[:nk]),('v',v[:nk]),('out',output[:nq])]})
                return output
            expected=[];qo=ko=0
            with torch.autocast(device.type,enabled=False), sdpa_kernel(SDPBackend.MATH):
                for nq,nk in zip(layout.query_lengths,layout.key_lengths):
                    expected.append(torch.nn.functional.scaled_dot_product_attention(q[qo:qo+nq].transpose(0,1)[None].float(),
                        k[ko:ko+nk].transpose(0,1)[None].float(),v[ko:ko+nk].transpose(0,1)[None].float(),
                        dropout_p=0.,is_causal=False,scale=scale,enable_gqa=True).squeeze(0).transpose(0,1));qo+=nq;ko+=nk
            oracle=torch.cat(expected)
            if oracle.dtype!=torch.float32:raise AssertionError('Independent MATH oracle must execute in FP32 outside autocast')
            tensors['fp32_oracle']=oracle;row['oracle_dtype']=str(oracle.dtype)
            row['pooled']=BASE.difference(output,oracle,attention=True);row['requests']=[];offset=0
            for request,n in zip(state['draft_requests'],layout.query_lengths):
                a=output[offset:offset+n];b=oracle[offset:offset+n];rounded=b.to(torch.bfloat16).float()
                row['requests'].append(dict(request=request,errors=BASE.difference(a,b,attention=True),
                    nearest_BF16_rounding=BASE.difference(rounded,b),native_vs_nearest_BF16=BASE.difference(a,rounded)))
                offset+=n
            row['passes_fixed_limits']=row['pooled']['passes_fixed_limits'] and all(x['errors']['passes_fixed_limits'] for x in row['requests'])
            row['tensors']=evidence(f"private-draft-{state['stage']}-layer{index}.pt",tensors)
            report['normal_layers'].append(row);persist();return output
        except Exception as exc:
            row.update(error_type=type(exc).__name__,error=str(exc),tensors=evidence(f"private-draft-{state['stage']}-layer{index}-failed.pt",tensors))
            report['normal_layers'].append(row);persist();raise
    target.append=append;target.predict=predict;packed.backbone=backbone;packed.kernel=kernel;packed.append_committed=context
    def tensor(ids):return torch.tensor([ids],device=device,dtype=torch.long)
    def random(seed):return TensorRandom(torch.Generator(device=device).manual_seed(seed))
    def replay(chunks,*,keeps=None):
        rows={};raw={}
        for r,ids in chunks.items():
            if r not in refs:refs[r]=dict(target=CachedTarget(reference_model,target.layer_ids),draft_kv=[])
            ref=refs[r];old=ref['target'].length
            feat=ref['target'].prefill(ids) if not ref['target']._initialized else ref['target'].append(ids)
            native=state['native_features'].for_request(r)
            raw.update({f'{r}_native_hidden':native.last,f'{r}_reference_hidden':feat.last,f'{r}_native_features':native.context,f'{r}_reference_features':feat.context})
            rows[r]=dict(hidden=BASE.difference(native.last,feat.last),features=BASE.difference(native.context,feat.context))
            if keeps is not None:
                logits=ref['target'].predict(feat);native_logits=state['native_logits'][r]
                raw.update({f'{r}_native_logits':native_logits,f'{r}_reference_logits':logits})
                rows[r]['logits']=BASE.logits_metrics(native_logits,logits)
            keep=ref['target'].length if keeps is None else keeps[r]
            committed=keep-old
            with session._autocast():projected=draft.project_context_kv(feat.context[:,:committed],old)
            if ref['draft_kv']:projected=[(torch.cat((a,k),2),torch.cat((b,v),2)) for (a,b),(k,v) in zip(ref['draft_kv'],projected)]
            ref['draft_kv']=projected;ref['target'].crop(keep)
        kv_rows={}
        for r,ref in refs.items():
            kv_rows[r]={}
            for domain,actual,expected in [('target',target.request_kv(r),[(x.keys,x.values) for x in ref['target'].cache.layers]),
                    ('draft',packed.request_kv(r),ref['draft_kv'])]:
                structural(len(actual)==len(expected),'actual_resident_KV_layer_count',request=r,domain=domain)
                kv_rows[r][domain]=[]
                for i,((ak,av),(bk,bv)) in enumerate(zip(actual,expected)):
                    for label,t in [('native_key',ak),('native_value',av),('reference_key',bk),('reference_value',bv)]:raw[f'{r}_{domain}_layer{i}_{label}']=t
                    kv_rows[r][domain].append(dict(layer=i,key=BASE.difference(ak,bk),value=BASE.difference(av,bv)))
        proof=evidence('private-endpoints-'+state['stage']+'.pt',raw)
        report['target_comparisons'].append(dict(stage=state['stage'],requests=rows,committed_KV=kv_rows,tensors=proof));persist()
    try:
        with torch.no_grad():
            prompts={r:tensor(ids) for r,ids in PROTOCOL['prompts'].items()}
            session.admit({r:RequestSpec(ids,64,random(PROTOCOL['seeds'][r])) for r,ids in prompts.items()});replay(prompts)
            for index,round_spec in enumerate(PROTOCOL['rounds']):
                if index==1:
                    old_inc=session.requests['C']['incarnation'];old_marker=packed._markers['C'];old_target_marker=target._markers['C']
                    session.remove('C');del refs['C']
                    structural(not bool((packed.key_requests==old_marker).any()) and not bool((target.key_requests==old_target_marker).any()),'removed_C_marker_absent')
                    state['stage']='readd_C';ids=tensor(PROTOCOL['readded_C_prompt'])
                    session.admit({'C':RequestSpec(ids,64,random(PROTOCOL['readded_C_seed']))});replay({'C':ids})
                    structural(session.requests['C']['incarnation']>old_inc and packed._markers['C']>old_marker and target._markers['C']>old_target_marker,'C_readded_fresh_incarnation_and_markers')
                state['stage']=round_spec['name'];state['phase']='normal'
                before=target.lengths;old_target={r:target.request_kv(r) for r in before};old_draft={r:packed.request_kv(r) for r in before}
                allocations=round_spec['allocations']
                shadow={}
                if round_spec['mode']=='shadow':
                    batch=session.propose(allocations,mode='shadow')
                    shadow={r:h.proposal for r,h in batch.proposals.items()}
                    result=session.verify_commit(batch.proposals,allocations,allocation_policy='external_nonanticipating')
                else:result=session.step(allocations)
                stochastic={}
                for r,p in result['proposals'].items():
                    stochastic.update({f'{r}_proposal_tokens':p.tokens,f'{r}_actual_q':p.draft_probs,f'{r}_raw_confidence':p.confidence_logits,
                        f'{r}_target_p':result['target_probs'][r],f'{r}_committed_tokens':result['decisions'][r].tokens,
                        f'{r}_verified_input':state['native_chunks'][r]})
                for r,p in shadow.items():
                    stochastic.update({f'{r}_shadow_tokens':p.tokens,f'{r}_shadow_q':p.draft_probs,f'{r}_shadow_confidence':p.confidence_logits})
                stochastic_proof=evidence('private-stochastic-'+state['stage']+'.pt',stochastic)
                for r in before:
                    BASE.assert_kv_equal(target.request_kv(r),old_target[r],prefix=before[r] if r in allocations else None)
                    BASE.assert_kv_equal(packed.request_kv(r),old_draft[r],prefix=before[r] if r in allocations else None)
                structural(True,'committed_old_prefix_and_inactive_target_draft_KV_exact')
                replay(state['native_chunks'],keeps={r:x['cache_after'] for r,x in result['requests'].items()})
                report['rounds'].append(dict(name=round_spec['name'],requests=result['requests'],work=result['work'],allocation_policy=result['allocation_policy'],stochastic_tensors=stochastic_proof))
                persist()
            saved=copy.deepcopy(packed.layers)
            for pair in PROTOCOL['poison_pairs']:
                collected=[]
                for poisoned in [False,True]:
                    packed.layers=copy.deepcopy(saved);state['phase']='poison';state['stage']=pair['name']+('_poison' if poisoned else '_control')
                    anchors={r:tensor([session.requests[r]['output'][-1]]) for r in pair['active']}
                    if poisoned:
                        selected=packed.key_requests==packed._markers[pair['victim']]
                        for k,v in packed.layers:k[:,:,selected]=100.;v[:,:,selected]=-100.
                        if 'B' in anchors:anchors['B']=tensor([507])
                    with session._autocast():hidden=packed.backbone(anchors).for_request('A');logits=draft.lm_head(hidden)
                    values=dict(state['attention_inputs'],hidden=cpu(hidden),logits=cpu(logits))
                    for i,(k,v) in enumerate(packed.request_kv('A')):values[f'cache{i}_k']=cpu(k);values[f'cache{i}_v']=cpu(v)
                    collected.append(values)
                exact=set(collected[0])==set(collected[1]) and all(torch.equal(t,collected[1][k]) for k,t in collected[0].items())
                tensors={f'{label}_{k}':t for label,entry in zip(['control','poison'],collected) for k,t in entry.items()}
                report['poison_pairs'].append(dict(name=pair['name'],A_inputs_outputs_logits_cache_exact=exact,tensors=evidence('private-poison-'+pair['name']+'.pt',tensors)))
                structural(exact,'paired_draft_isolation_exact',pair=pair['name'])
            packed.layers=saved
            structural(state['target_calls']==5 and state['draft_calls']==7,'all_fixed_forward_counts',target_forwards=state['target_calls'],draft_forwards=state['draft_calls'])
            structural(len(report['normal_layers'])==3*len(draft.layers),'all_normal_draft_layers_measured')
            report.update(structural_status='passed',draft_layer_numerical_status='passed' if all(x['passes_fixed_limits'] for x in report['normal_layers']) else 'failed',endpoint_comparison_status='completed')
            report['status']='completed' if report['draft_layer_numerical_status']=='passed' else 'completed_with_failed_draft_layer_gate';persist()
    except Exception as exc:
        failed={}
        for r,ids in state.get('native_chunks',{}).items():failed[f'{r}_last_target_input']=ids
        if 'native_features' in state:
            failed['last_native_hidden']=state['native_features'].last
            failed['last_native_features']=state['native_features'].context
        for i,(k,v) in enumerate(packed.layers):failed[f'draft{i}_key']=k;failed[f'draft{i}_value']=v
        for i,layer in enumerate(target.cache.layers):
            if layer.is_initialized:failed[f'target{i}_key']=layer.keys;failed[f'target{i}_value']=layer.values
        report['failure_evidence']=evidence('private-failure-state.pt',failed)
        report.update(status='execution_failed',error_type=type(exc).__name__,error=str(exc))
        for key in ['structural_status','draft_layer_numerical_status','endpoint_comparison_status']:
            if report[key]=='pending':report[key]='failed_or_incomplete'
        persist();raise
    finally:
        target.append=original_target_append;target.predict=original_predict;packed.backbone=original_backbone;packed.kernel=original_kernel;packed.append_committed=original_context
    return report


def worker(out):
    report=new_report();save(out/'worker-result.json',report)
    try:
        binding=json.loads((out/'binding.json').read_text())
        if bind(binding['model'],binding['generation_manifest'],binding['checkpoint'])!=binding:raise ValueError('Bound inputs/source/protocol changed')
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        sys.path.insert(0,str(ROOT))
        from dspark_qwen.checkpoint import load_checkpoint
        from dspark_qwen.config import DraftConfig
        from dspark_qwen.model import DSparkDraft
        from dspark_qwen.packed_draft import PackedDraft,DRAFT_BACKEND
        from dspark_qwen.packed_sampling import PackedSpeculativeSession
        from dspark_qwen.varlen_target import VarlenPackedTarget
        from dspark_qwen.rocm_varlen import BACKEND,PinnedRocmVarlenKernel
        if transformers.__version__!='5.17.0' or not torch.cuda.is_available():raise RuntimeError('Require pinned Transformers and separately authorized ROCm GPU')
        device=torch.device('cuda:0');torch.cuda.set_device(device);runtime=PinnedRocmVarlenKernel(device).runtime
        free,total=torch.cuda.mem_get_info()
        if free<PROTOCOL['minimum_free_bytes']:raise RuntimeError('8 GiB free-memory guard failed')
        torch.cuda.set_per_process_memory_fraction(PROTOCOL['process_allocation_cap_bytes']/total);torch.cuda.reset_peak_memory_stats()
        def load():return AutoModelForCausalLM.from_pretrained(binding['model'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
        model=load();draft=DSparkDraft(model,DraftConfig.from_dict(binding['draft_config'])).to(device).eval()
        load_checkpoint(binding['checkpoint'],draft)
        target=VarlenPackedTarget(model,draft.spec.layer_ids,native_backend=BACKEND)
        packed=PackedDraft(draft,native_backend=DRAFT_BACKEND)
        session=PackedSpeculativeSession(target,packed,temperature=1.,eos_ids=(),amp=True)
        reference=load()
        save(out/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,hip=torch.version.hip,runtime_pin=runtime,
            free_bytes_before_load=free,process_allocation_cap_bytes=PROTOCOL['process_allocation_cap_bytes'],
            draft_fc_dtype=str(draft.fc.weight.dtype),draft_rope_dtype=str(draft.rotary.inv_freq.dtype)))
        exercise(session,reference,out,report,require_native_events=True)
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        if source_identity()!=binding['source_sha256']:raise RuntimeError('Source changed during execution')
        return 0 if report['status']=='completed' else 1
    except Exception as exc:
        report.update(status='execution_failed',error_type=type(exc).__name__,error=str(exc));(out/'error.txt').write_text(traceback.format_exc());return 1
    finally:save(out/'worker-result.json',report)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true');mode.add_argument('--dry-run',action='store_true')
    parser.add_argument('--model');parser.add_argument('--generation-manifest');parser.add_argument('--checkpoint');parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS);args=parser.parse_args(argv)
    if args.worker:return worker(args.worker.resolve())
    if not all((args.model,args.generation_manifest,args.checkpoint)):parser.error('Real --model, --generation-manifest and selected --checkpoint required')
    binding=bind(args.model,args.generation_manifest,args.checkpoint)
    if not args.execute:print(json.dumps(dict(status='dry_run_no_backend_import_no_gpu',**binding),indent=2));return 0
    if args.output is None:parser.error('--execute requires fresh --output')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False,mode=0o700);save(out/'binding.json',binding)
    command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(out)];timed_out=False;start=time.monotonic()
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        child=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2'))
        save(out/'started.json',dict(pid=child.pid,command=command))
        try:code=child.wait(timeout=PROTOCOL['timeout_seconds'])
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            code=child.wait()
        except BaseException:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait();raise
    path=out/'worker-result.json';report=json.loads(path.read_text()) if path.exists() else new_report()
    finished=not timed_out and source_identity()==binding['source_sha256'] and report['status'] in ('completed','completed_with_failed_draft_layer_gate')
    save(out/'completion.json',dict(exit_code=code,timed_out=timed_out,execution_completed=finished,
        **{k:report[k] for k in ['status','structural_status','draft_layer_numerical_status','endpoint_comparison_status','system_pass_claimed']},
        wall_seconds=time.monotonic()-start,timing_scope='Bounded correctness instrumentation, not a benchmark'))
    return 0 if code==0 and finished and report['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
