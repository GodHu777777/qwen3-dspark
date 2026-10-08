#!/usr/bin/env python3
"""Whole pretrained Qwen/KV reference gate. Default is stdlib-only dry-run.

CPU fixtures call exercise() directly; the CLI always binds the real Qwen3-0.6B
architecture and generation target fingerprints. GPU execution is separately
coordinated. Structural, same-QKV attention and end-to-end comparisons are distinct.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BACKEND = 'rocm_aten_no_window_pinned_v1'
PROTOCOL = dict(version=1, backend=BACKEND, dtype='bfloat16', transformers='5.17.0', temperature=1.0,
    probability_policy='float64_softmax_normalize_cdf_v1',
    attention_atol=0.02, attention_rtol=0.02, attention_max_rms=0.005,
    attention_gate_reduction='pooled and each request independently',
    timeout_seconds=300, selected_layers=[1,7,14,21,26],
    model_architecture=dict(model_type='qwen3',hidden_size=1024,intermediate_size=3072,
        num_hidden_layers=28,num_attention_heads=16,num_key_value_heads=8,head_dim=128,vocab_size=151936),
    normal_schedule=[dict(name='prime',chunks=dict(A=list(range(100,116)),B=list(range(200,221)),C=list(range(300,313)))),
        dict(name='cached_tail',chunks=dict(A=[116],B=list(range(221,229)))),
        dict(name='crop_exit_readd',crop=dict(A=8,B=3),remove_readd='C',chunks=dict(A=[117,118,119],B=[229,230],C=[313])),
        dict(name='crop_zero',crop=dict(A=0),chunks=dict(A=[120,121]))],
    poison_pairs=[dict(name='inactive_C',chunks=dict(A=[116]),poison='C'),
        dict(name='other_active_B',chunks=dict(A=[116],B=list(range(221,229))),poison='B',alternate_tokens=list(range(500,508)))],
    poison_key=100.,poison_value=-100.,
    structural_policy='Exact positions/gather/crop/prefix preservation and paired request-isolation content checks',
    layer_attention_policy='Every layer of four normal forwards: actual BF16 QKV against independent FP32 MATH bottom-right, fixed original limits',
    poison_attention_policy='No same-QKV numerical gate on artificial poison amplitudes; structural exact isolation is separate',
    end_to_end_policy='BF16 dense/per-request KV, hidden, selected features, logits and float64 TV are quantified only; no numerical pass or losslessness claim')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def source_identity():
    paths=[Path(__file__).resolve(),*sorted((ROOT/'dspark_qwen').glob('*.py'))]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def bind(model_path,generation_manifest):
    model=Path(model_path).resolve();manifest=Path(generation_manifest).resolve()
    fingerprint=json.loads(manifest.read_text())['model_files_sha256']
    if 'config.json' not in fingerprint or not any(n.endswith('.safetensors') for n in fingerprint):
        raise ValueError('Require actual generation model/config/weight fingerprints')
    for name,expected in fingerprint.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(model/name)!=expected:
            raise ValueError('Target/tokenizer identity mismatch')
    cfg=json.loads((model/'config.json').read_text())
    if any(cfg.get(k)!=v for k,v in PROTOCOL['model_architecture'].items()) or cfg.get('use_sliding_window',False) or cfg.get('rope_scaling') is not None:
        raise ValueError('Formal gate requires the pinned real dense Qwen3-0.6B architecture, not a tiny substitute')
    return dict(model=str(model),generation_manifest=str(manifest),generation_manifest_sha256=sha(manifest),
        target_fingerprint=fingerprint,protocol=PROTOCOL,protocol_sha256=digest(PROTOCOL),source_sha256=source_identity())


def new_report():
    return dict(status='running',structural_status='pending',layer_attention_numerical_status='pending',
        numerical_comparison_status='pending',layer_attention=[],normal_steps=[],poison_pairs=[],
        structural_checks=[],system_pass_claimed=False)


def difference(actual,expected,*,attention=False):
    import torch
    if actual.shape!=expected.shape or not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise RuntimeError('Incompatible or nonfinite comparison tensors')
    delta=(actual.double()-expected.double()).abs()
    if delta.numel()==0:return dict(elements=0,max_abs=None,rms=None,equal_elements=0)
    row=dict(elements=delta.numel(),max_abs=float(delta.max()),rms=float(delta.square().mean().sqrt()),
             equal_elements=int((delta==0).sum()))
    if attention:
        allowed=PROTOCOL['attention_atol']+PROTOCOL['attention_rtol']*expected.double().abs()
        row['mismatched_elements']=int((delta>allowed).sum())
        row['passes_fixed_limits']=row['mismatched_elements']==0 and row['rms']<=PROTOCOL['attention_max_rms']
    return row


def attention_oracle(q,k,v,layout,scale):
    import torch
    from torch.nn.attention import SDPBackend,sdpa_kernel
    rows=[];qo=ko=0
    with sdpa_kernel(SDPBackend.MATH):
        for nq,nk in zip(layout.query_lengths,layout.key_lengths):
            mask=torch.arange(nk,device=q.device)[None]<=torch.arange(nk-nq,nk,device=q.device)[:,None]
            row=torch.nn.functional.scaled_dot_product_attention(q[qo:qo+nq].transpose(0,1)[None].float(),
                k[ko:ko+nk].transpose(0,1)[None].float(),v[ko:ko+nk].transpose(0,1)[None].float(),
                attn_mask=mask[None,None],dropout_p=0.,is_causal=False,scale=scale,enable_gqa=q.shape[1]!=k.shape[1])
            rows.append(row.squeeze(0).transpose(0,1));qo+=nq;ko+=nk
    return torch.cat(rows)


def snapshot(target):
    import copy
    return dict(cache=copy.deepcopy(target.cache),lengths=dict(target._lengths),markers=dict(target._markers),
        next_marker=target._next_marker,key_requests=target.key_requests.clone(),key_positions=target.key_positions.clone())


def restore(target,state):
    import copy
    target.cache=copy.deepcopy(state['cache']);target._lengths=dict(state['lengths']);target._markers=dict(state['markers'])
    target._next_marker=state['next_marker'];target.key_requests=state['key_requests'].clone();target.key_positions=state['key_positions'].clone()
    target.validate_cache()


def request_kv(target):
    return {name:tuple((k.clone(),v.clone()) for k,v in target.request_kv(name)) for name in target.lengths}


def assert_kv_equal(actual,expected,*,prefix=None):
    import torch
    if prefix==0 and not expected:
        return  # Initial cache layers are uninitialized; there is no retained prefix.
    if len(actual)!=len(expected):raise AssertionError('KV layer count changed')
    for (ak,av),(ek,ev) in zip(actual,expected):
        if prefix is not None:ak,av,ek,ev=ak[:,:,:prefix],av[:,:,:prefix],ek[:,:,:prefix],ev[:,:,:prefix]
        if not torch.equal(ak,ek) or not torch.equal(av,ev):raise AssertionError('KV content changed outside intended suffix')


def logits_metrics(actual,expected):
    from dspark_qwen.tensor_sampling import logits_to_probabilities
    row=difference(actual,expected);a=actual.reshape(-1,actual.shape[-1]);b=expected.reshape_as(a)
    row['argmax_changed_rows']=int((a.argmax(-1)!=b.argmax(-1)).sum())
    row['probability_rows']=[]
    for index,(x,y) in enumerate(zip(a,b)):
        p=logits_to_probabilities(x,1.);q=logits_to_probabilities(y,1.)
        row['probability_rows'].append(dict(row=index,total_variation=float((p-q).abs().sum()/2),max_probability_difference=float((p-q).abs().max())))
    return row


def exercise(native,dense,independent,out,report,*,require_native_events=False):
    """CPU-testable control flow. Formal worker supplies real BF16 GPU models."""
    import copy
    import contextlib
    import torch
    from dspark_qwen.cached_target import CachedTarget
    out=Path(out);model=native.model;device=native.device;layers=model.config.num_hidden_layers
    report['runtime_scope']='real_pretrained_gpu' if require_native_events else 'cpu_test_fixture_only'
    original_kernel=native._varlen_kernel;original_payload=native._attention_payload
    current={};expected_lengths={};expected_markers={};next_marker=0
    def persist():save(out/'worker-result.json',report)
    def cpu(t):return t.detach().cpu().clone()
    def evidence(name,tensors):
        path=out/name;torch.save({k:cpu(v) for k,v in tensors.items()},path)
        return dict(file=path.name,sha256=sha(path))
    def structural(ok,kind,**details):
        report['structural_checks'].append(dict(kind=kind,passed=bool(ok),**details));persist()
        if not ok:raise AssertionError('Structural invariant failed: '+kind)
    def payload(qr,qp,kr,kp):
        structural(torch.equal(qr,current['qr']) and torch.equal(qp,current['qp']) and
                   torch.equal(kr,current['kr']) and torch.equal(kp,current['kp']),
                   'actual_model_request_positions',stage=current['name'])
        result=original_payload(qr,qp,kr,kp)
        current['layout']=result[1]['packed_layout']
        return result
    def observed_kernel(q,k,v,layout,*,scale):
        index=current['layer'];current['layer']+=1
        row=dict(stage=current['name'],phase=current['phase'],layer=index,
            attention_gate_required=current['phase']=='normal',q_shape=list(q.shape),kv_shape=list(k.shape),
            query_lengths=list(layout.query_lengths),key_lengths=list(layout.key_lengths))
        tensors=dict(q=q,k=k,v=v,cu_query=layout.cu_query,cu_key=layout.cu_key)
        try:
            if index>=layers:raise AssertionError('More attention calls than model layers')
            physical=native.cache.layers[index]
            indices=torch.cat([torch.nonzero(current['kr']==marker,as_tuple=False).flatten() for marker in current['ordered_markers']])
            if not torch.equal(layout.gather_indices,indices) or tuple(layout.query_lengths)!=current['query_lengths'] or tuple(layout.key_lengths)!=current['key_lengths']:
                raise AssertionError('Packed gather indices/lengths differ from independent request mapping')
            expected_k=physical.keys.index_select(2,indices).squeeze(0).transpose(0,1)
            expected_v=physical.values.index_select(2,indices).squeeze(0).transpose(0,1)
            if not torch.equal(k,expected_k) or not torch.equal(v,expected_v):
                raise AssertionError('Actual gathered KV content differs from physical request gather')
            row['gather_content_exact']=True
            if current['phase']=='poison':
                nq,nk=layout.query_lengths[0],layout.key_lengths[0]
                selected={'q':cpu(q[:nq]),'k':cpu(k[:nk]),'v':cpu(v[:nk])}
                row['A_QKV_sha256']={key:hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for key,t in selected.items()}
                current['isolation_inputs'].update({f'layer{index}_{key}':t for key,t in selected.items()})
            if not all(bool(torch.isfinite(t).all()) for t in (q,k,v)):
                raise AssertionError('Nonfinite actual QKV')
            actual=original_kernel(q,k,v,layout,scale=scale);tensors['output']=actual
            if actual.shape!=q.shape or actual.dtype!=q.dtype or actual.device!=q.device or not bool(torch.isfinite(actual).all()):
                raise AssertionError('Invalid attention output shape/dtype/device/finite')
            row['finite_shape_dtype_device_checked']=True
            if row['attention_gate_required']:
                expected=attention_oracle(q,k,v,layout,scale);tensors['fp32_oracle']=expected
                row['errors']=difference(actual,expected,attention=True)
                offset=0;row['per_request_errors']=[]
                for nq in layout.query_lengths:
                    row['per_request_errors'].append(difference(actual[offset:offset+nq],expected[offset:offset+nq],attention=True));offset+=nq
                row['passes_fixed_limits']=row['errors']['passes_fixed_limits'] and all(x['passes_fixed_limits'] for x in row['per_request_errors'])
                if not row['passes_fixed_limits']:
                    row['failed_tensors']=evidence(f"private-attention-{current['name']}-layer{index}.pt",tensors)
            row['status']='observed';report['layer_attention'].append(row);persist()
            return actual
        except Exception as exc:
            row.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                failed_tensors=evidence(f"private-attention-{current['name']}-layer{index}-failure.pt",tensors))
            report['layer_attention'].append(row);persist();raise
    native._attention_payload=payload;native._varlen_kernel=observed_kernel
    def add(name):
        nonlocal next_marker
        native.add_request(name);dense.add_request(name)
        independent[name]=CachedTarget(independent['_model'],native.layer_ids)
        expected_lengths[name]=0;expected_markers[name]=next_marker;next_marker+=1
    def native_append(chunks,name,phase,lengths,markers):
        current.clear();current.update(name=name,phase=phase,layer=0,isolation_inputs={},ordered_markers=[markers[r] for r in chunks],
            query_lengths=tuple(len(ids) for ids in chunks.values()),key_lengths=tuple(lengths[r]+len(ids) for r,ids in chunks.items()))
        before=request_kv(native)
        structural(native.lengths==lengths and native._markers==markers,'request_state_before_forward',stage=name)
        current['qr']=torch.cat([torch.full((len(tokens),),markers[r],dtype=torch.long,device=device) for r,tokens in chunks.items()])
        current['qp']=torch.cat([torch.arange(lengths[r],lengths[r]+len(tokens),device=device) for r,tokens in chunks.items()])
        current['kr']=torch.cat((native.key_requests,current['qr']));current['kp']=torch.cat((native.key_positions,current['qp']))
        forwards=[]
        expected_tokens=torch.tensor([sum((list(tokens) for tokens in chunks.values()),[])],dtype=torch.long,device=device)
        def inspect_forward(_module,_args,kwargs):
            forwards.append(1)
            structural(torch.equal(kwargs['position_ids'],current['qp'][None]) and torch.equal(kwargs['input_ids'],expected_tokens),
                       'actual_model_input_tokens_and_RoPE_positions',stage=name)
        hook=model.model.register_forward_pre_hook(inspect_forward,with_kwargs=True)
        profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if require_native_events else contextlib.nullcontext()
        try:
            with profile as events:
                features=native.append({r:torch.tensor([tokens],dtype=torch.long,device=device) for r,tokens in chunks.items()})
                if require_native_events:torch.cuda.synchronize(device)
        finally:hook.remove()
        structural(forwards==[1] and current['layer']==layers,'one_forward_one_attention_per_layer',stage=name,model_forwards=len(forwards),attention_calls=current['layer'])
        if require_native_events:
            counts={event.key:event.count for event in events.key_averages()}
            structural(counts.get('aten::_flash_attention_forward',0)==layers,'native_flash_operator_count',stage=name,operator_counts=counts)
        after=request_kv(native)
        for r,old in before.items():
            if r in chunks:assert_kv_equal(after[r],old,prefix=lengths[r])
            else:assert_kv_equal(after[r],old)
        structural(True,'append_preserves_actual_old_KV_content',stage=name)
        expected={r:n+len(chunks.get(r,[])) for r,n in lengths.items()}
        structural(native.lengths==expected,'request_lengths_after_forward',stage=name)
        logits=native.predict(features)
        return features,logits
    def mutate(step):
        for r,n in step.get('crop',{}).items():
            before=request_kv(native);old_requests=native.key_requests.clone();old_positions=native.key_positions.clone()
            keep=(old_requests!=expected_markers[r])|(old_positions<n)
            native.crop(r,n);dense.crop(r,n);independent[r].crop(n);expected_lengths[r]=n
            for name,pairs in before.items():
                if name==r:
                    expected=tuple((k[:,:,:n],v[:,:,:n]) for k,v in pairs);assert_kv_equal(native.request_kv(name),expected)
                else:assert_kv_equal(native.request_kv(name),pairs)
            structural(torch.equal(native.key_requests,old_requests[keep]) and torch.equal(native.key_positions,old_positions[keep]),'crop_exact_KV_and_position_subset',request=r,length=n)
        if 'remove_readd' in step:
            r=step['remove_readd'];before=request_kv(native);old_marker=expected_markers[r]
            native.remove_request(r);dense.remove_request(r);del independent[r];del expected_lengths[r];del expected_markers[r]
            for name,pairs in before.items():
                if name!=r:assert_kv_equal(native.request_kv(name),pairs)
            structural(not bool((native.key_requests==old_marker).any()),'removed_request_marker_absent',request=r)
            add(r)
            structural(native._markers[r]!=old_marker and native.lengths[r]==0,'readded_request_fresh_marker_and_position_zero',request=r)
    try:
        for name in ['A','B','C']:add(name)
        prime_state=None
        for step in PROTOCOL['normal_schedule']:
            mutate(step);chunks=step['chunks'];name=step['name']
            features,logits=native_append(chunks,name,'normal',dict(expected_lengths),dict(expected_markers))
            packed=dense.append({r:torch.tensor([ids],dtype=torch.long,device=device) for r,ids in chunks.items()});packed_logits=dense.predict(packed)
            individual={};individual_logits={}
            for r,ids in chunks.items():
                target=independent[r];tensor=torch.tensor([ids],dtype=torch.long,device=device)
                individual[r]=target.prefill(tensor) if not target._initialized else target.append(tensor)
                individual_logits[r]=target.predict(individual[r])
                expected_lengths[r]+=len(ids)
            structural(dense.lengths==expected_lengths and all(independent[r].length==n for r,n in expected_lengths.items()),'reference_request_lengths_match',stage=name)
            raw={};row=dict(stage=name,requests={},kv_comparisons={})
            for r in chunks:
                for label,feat,output in [('native',features.for_request(r),logits[r]),('dense',packed.for_request(r),packed_logits[r]),('independent',individual[r],individual_logits[r])]:
                    raw[f'{r}_{label}_hidden']=feat.last;raw[f'{r}_{label}_features']=feat.context;raw[f'{r}_{label}_logits']=output
            for r in expected_lengths:
                for label,pairs in [('native',native.request_kv(r)),('dense',dense.request_kv(r)),('independent',[(layer.keys,layer.values) for layer in independent[r].cache.layers])]:
                    for i,(key,value) in enumerate(pairs):
                        raw[f'{r}_layer{i}_{label}_key']=key;raw[f'{r}_layer{i}_{label}_value']=value
            row['private_comparison_tensors']=evidence(f'private-comparison-{name}.pt',raw)
            report['normal_steps'].append(row);persist()
            for r in chunks:
                a=features.for_request(r);b=packed.for_request(r);c=individual[r]
                row['requests'][r]=dict(hidden_vs_dense=difference(a.last,b.last),hidden_vs_independent=difference(a.last,c.last),
                    features_vs_dense=difference(a.context,b.context),features_vs_independent=difference(a.context,c.context),
                    logits_vs_dense=logits_metrics(logits[r],packed_logits[r]),logits_vs_independent=logits_metrics(logits[r],individual_logits[r]))
                for label,hidden,context,output in [('native',a.last,a.context,logits[r]),('dense',b.last,b.context,packed_logits[r]),('independent',c.last,c.context,individual_logits[r])]:
                    raw[f'{r}_{label}_hidden']=hidden;raw[f'{r}_{label}_features']=context;raw[f'{r}_{label}_logits']=output
            for r in expected_lengths:
                actual=native.request_kv(r);reference=dense.request_kv(r)
                separate=[(layer.keys,layer.values) for layer in independent[r].cache.layers]
                row['kv_comparisons'][r]=[]
                for i,((ak,av),(bk,bv),(ck,cv)) in enumerate(zip(actual,reference,separate)):
                    row['kv_comparisons'][r].append(dict(layer=i,key_vs_dense=difference(ak,bk),value_vs_dense=difference(av,bv),
                        key_vs_independent=difference(ak,ck),value_vs_independent=difference(av,cv)))
                    for label,key,value in [('native',ak,av),('dense',bk,bv),('independent',ck,cv)]:
                        raw[f'{r}_layer{i}_{label}_key']=key;raw[f'{r}_layer{i}_{label}_value']=value
            persist()
            if name=='prime':prime_state=snapshot(native);prime_lengths=dict(expected_lengths);prime_markers=dict(expected_markers)
        for pair in PROTOCOL['poison_pairs']:
            collected=[]
            for poisoned in [False,True]:
                restore(native,prime_state);chunks=copy.deepcopy(pair['chunks']);name=pair['name']+('_poison' if poisoned else '_control')
                if poisoned:
                    selected=native.key_requests==prime_markers[pair['poison']]
                    for layer in native.cache.layers:
                        layer.keys[:,:,selected]=PROTOCOL['poison_key'];layer.values[:,:,selected]=PROTOCOL['poison_value']
                    if 'alternate_tokens' in pair:chunks['B']=pair['alternate_tokens']
                features,logits=native_append(chunks,name,'poison',prime_lengths,prime_markers)
                collected.append((cpu(features.for_request('A').last),cpu(logits['A']),tuple((cpu(k),cpu(v)) for k,v in native.request_kv('A')),dict(current['isolation_inputs'])))
            left,right=collected
            tensor_map={}
            for label,item in [('control',left),('poison',right)]:
                tensor_map[label+'_hidden']=item[0];tensor_map[label+'_logits']=item[1]
                for i,(k,v) in enumerate(item[2]):tensor_map[f'{label}_layer{i}_key']=k;tensor_map[f'{label}_layer{i}_value']=v
                tensor_map.update({label+'_attention_'+key:t for key,t in item[3].items()})
            proof=evidence('private-isolation-'+pair['name']+'.pt',tensor_map)
            inputs_exact=set(left[3])==set(right[3]) and all(torch.equal(t,right[3][key]) for key,t in left[3].items())
            exact=inputs_exact and torch.equal(left[0],right[0]) and torch.equal(left[1],right[1])
            try:assert_kv_equal(left[2],right[2])
            except AssertionError:exact=False
            report['poison_pairs'].append(dict(name=pair['name'],A_QKV_exact=inputs_exact,hidden_logits_and_all_A_KV_exact=exact,private_tensors=proof));persist()
            structural(exact,'paired_request_isolation_content',pair=pair['name'])
        normal=[r for r in report['layer_attention'] if r['attention_gate_required']]
        structural(len(normal)==4*layers,'all_normal_layers_compared',count=len(normal))
        report['structural_status']='passed'
        report['layer_attention_numerical_status']='passed' if all(r['passes_fixed_limits'] for r in normal) else 'failed'
        report['numerical_comparison_status']='completed'
        report['status']='completed' if report['layer_attention_numerical_status']=='passed' else 'completed_with_failed_layer_attention_gate'
        persist()
    except Exception as exc:
        failed_state=dict(key_requests=native.key_requests,key_positions=native.key_positions)
        for i,layer in enumerate(native.cache.layers):
            if layer.is_initialized:
                failed_state[f'layer{i}_key']=layer.keys;failed_state[f'layer{i}_value']=layer.values
        report['failure_state_evidence']=evidence('private-structural-failure-state.pt',failed_state)
        report.update(status='execution_failed',error_type=type(exc).__name__,error=str(exc))
        if report['structural_status']=='pending':report['structural_status']='failed_or_incomplete'
        if report['layer_attention_numerical_status']=='pending':report['layer_attention_numerical_status']='failed_or_incomplete'
        if report['numerical_comparison_status']=='pending':report['numerical_comparison_status']='failed_or_incomplete'
        persist();raise
    finally:
        native._attention_payload=original_payload;native._varlen_kernel=original_kernel
    return report


def worker(out):
    report=new_report();save(out/'worker-result.json',report)
    try:
        binding=json.loads((out/'binding.json').read_text())
        if bind(binding['model'],binding['generation_manifest'])!=binding:
            raise ValueError('Bound target/source/protocol changed before loading')
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        sys.path.insert(0,str(ROOT))
        from dspark_qwen.packed_target import PackedTarget
        from dspark_qwen.varlen_target import VarlenPackedTarget
        from dspark_qwen.rocm_varlen import PinnedRocmVarlenKernel
        if not torch.cuda.is_available():raise RuntimeError('Formal gate requires a separately authorized GPU window')
        device=torch.device('cuda:0');torch.cuda.set_device(device)
        runtime_kernel=PinnedRocmVarlenKernel(device)
        free,total=torch.cuda.mem_get_info()
        if free<8*1024**3:raise RuntimeError('Require 8 GiB free before loading three actual target copies')
        torch.cuda.set_per_process_memory_fraction(6*1024**3/total)
        torch.cuda.reset_peak_memory_stats(device)
        if transformers.__version__!=PROTOCOL['transformers']:raise RuntimeError('Whole-Qwen gate pins Transformers 5.17.0 cache/callback API')
        save(out/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,hip=torch.version.hip,
            runtime_pin=runtime_kernel.runtime,free_bytes_before_load=free,process_allocation_cap_bytes=6*1024**3))
        def load():
            model=AutoModelForCausalLM.from_pretrained(binding['model'],local_files_only=True,dtype=torch.bfloat16,
                attn_implementation='sdpa').to(device).eval().requires_grad_(False)
            if any(getattr(model.config,k,None)!=v for k,v in PROTOCOL['model_architecture'].items()):
                raise ValueError('Loaded model differs from bound real Qwen3-0.6B architecture')
            return model
        native=VarlenPackedTarget(load(),PROTOCOL['selected_layers'],native_backend=BACKEND)
        dense=PackedTarget(load(),PROTOCOL['selected_layers']);independent={'_model':load()}
        if getattr(native._varlen_kernel,'backend_name',None)!=BACKEND:raise RuntimeError('Wrong native backend')
        with torch.no_grad():exercise(native,dense,independent,out,report,require_native_events=True)
        if source_identity()!=binding['source_sha256']:raise RuntimeError('Source changed during gate')
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        # Completion states are separate; no blanket system/bitwise/lossless verdict.
        return 0 if report['status']=='completed' else 1
    except Exception as exc:
        report.update(status='execution_failed',error_type=type(exc).__name__,error=str(exc))
        (out/'error.txt').write_text(traceback.format_exc());return 1
    finally:save(out/'worker-result.json',report)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--execute',action='store_true');modes.add_argument('--dry-run',action='store_true')
    parser.add_argument('--model');parser.add_argument('--generation-manifest');parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:return worker(args.worker.resolve())
    if not args.model or not args.generation_manifest:parser.error('--model and --generation-manifest are required; tiny fixtures are not a CLI mode')
    binding=bind(args.model,args.generation_manifest)
    if not args.execute:
        print(json.dumps(dict(status='dry_run_no_backend_import_no_gpu',**binding),indent=2));return 0
    if args.output is None:parser.error('--execute requires a fresh --output')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False,mode=0o700);save(out/'binding.json',binding)
    command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(out)]
    start=time.monotonic();timed_out=False
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        process=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2'))
        save(out/'started.json',dict(pid=process.pid,command=command))
        try:code=process.wait(timeout=PROTOCOL['timeout_seconds'])
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            code=process.wait()
        except BaseException:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait();raise
    path=out/'worker-result.json';report=json.loads(path.read_text()) if path.exists() else new_report()
    valid=code==0 and not timed_out and source_identity()==binding['source_sha256'] and report['status']=='completed'
    finished=not timed_out and source_identity()==binding['source_sha256'] and report['status'] in ('completed','completed_with_failed_layer_attention_gate')
    save(out/'completion.json',dict(exit_code=code,timed_out=timed_out,execution_completed=finished,worker_status=report['status'],
        structural_status=report['structural_status'],layer_attention_numerical_status=report['layer_attention_numerical_status'],
        numerical_comparison_status=report['numerical_comparison_status'],system_pass_claimed=False,
        wall_seconds=time.monotonic()-start,timing_scope='Oracle/instrumentation/persistence included; not a benchmark'))
    return 0 if valid else 1


if __name__=='__main__':raise SystemExit(main())
