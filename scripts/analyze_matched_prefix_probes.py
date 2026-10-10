#!/usr/bin/env python3
"""Bounded CPU-only analysis of six existing first-position full-law probes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STEPS = (128, 512, 1280)
ORDINALS = (0, 1)
VOCAB = 151936
TOL = 1e-12
CAPS = dict(file_bytes=64*1024**2, total_input_bytes=384*1024**2,
            rss_bytes=2*1024**3, output_bytes=256*1024**2, timeout_seconds=300, cooperative_seconds=290)
QUALITY_FILES = ('run.json','result.json','aggregate.json','runtime.json','worker-exit.json',
                 'private-rounds.jsonl','private-blocks.jsonl','private-outputs.jsonl')
SOURCES = ('scripts/analyze_matched_prefix_probes.py','tests/test_analyze_matched_prefix_probes.py','docs/matched-prefix-diagnostic.md')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''): result.update(chunk)
    return result.hexdigest()


def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def loads(value):
    def pairs(items):
        result={}
        for k,v in items: require(k not in result,'Duplicate JSON key');result[k]=v
        return result
    def bad(value): raise ValueError('Nonfinite JSON constant')
    return json.loads(value,object_pairs_hook=pairs,parse_constant=bad)


def read(path): return loads(Path(path).read_text())
def lines(path): return [loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)


def bind(probe_root, quality_root, historical_source):
    """No torch import, no follow of paths to models/generation data."""
    probe_root,quality_root,source=map(lambda p:Path(p).resolve(),(probe_root,quality_root,historical_source))
    sources={p.name:sha(p) for p in source.glob('*.py')};require(sources,'Historical source required')
    files={};probes=[];states={};common=None
    for step in STEPS:
        directory=quality_root/str(step)/'collection'
        for name in QUALITY_FILES: files[str(directory/name)]=sha(directory/name)
        binding,result,aggregate=(read(directory/name) for name in ('run.json','result.json','aggregate.json'))
        require(binding['binding_sha256']==digest({k:v for k,v in binding.items() if k!='binding_sha256'}),'Run binding mismatch')
        require(Path(binding['checkpoint']).name==f'step-{step:06d}','Checkpoint step mismatch')
        manifest=binding['manifest'];protocol=manifest['protocol']
        require(manifest['manifest_sha256']==digest({k:v for k,v in manifest.items() if k!='manifest_sha256'}),'Panel digest mismatch')
        panel=directory.parent/'panel.private.json';files[str(panel)]=sha(panel)
        require(files[str(panel)]==binding['manifest_file_sha256'] and read(panel)==manifest,'Panel bytes mismatch')
        require(binding['source_sha256']==sources,'Historical source mismatch')
        require(binding.get('collection_group','quality')=='quality' and len(binding['cases'])==32,'Original quality32 required')
        require(protocol['max_new_tokens']==128 and protocol['block_size']==7 and protocol['temperature']==1 and
                protocol['probability_policy']=='float64_softmax_normalize_cdf_v1','Original probability protocol required')
        require(all(x['status']=='completed' and x['execution_checks_passed'] is True and x['binding_sha256']==binding['binding_sha256'] for x in (result,aggregate)),'Incomplete/unbound collection')
        require(read(directory/'worker-exit.json')=={'returncode':0},'Successful original worker required')
        runtime=read(directory/'runtime.json');identity=dict(cases=binding['cases'],manifest=manifest,source=sources,
            runtime={k:runtime[k] for k in ('torch','transformers','hip','device')},draft_config=binding['draft_config'])
        if common is not None: require(identity==common,'Cross-checkpoint input/source/runtime identity mismatch')
        common=identity
        outputs=lines(directory/'private-outputs.jsonl');rounds=lines(directory/'private-rounds.jsonl');blocks=lines(directory/'private-blocks.jsonl')
        require(len(result['runs'])==len(outputs)==aggregate['completed_prompts']==32,'Incomplete quality32 outputs')
        require(len(rounds)==len(blocks)==result['blocks']==aggregate['blocks'],'Round/block coverage mismatch')
        for ordinal in ORDINALS:
            case=binding['cases'][ordinal];run=result['runs'][ordinal];output=outputs[ordinal]
            require(case['index']==run['case']==ordinal and case['split']=='validation' and case['seed']==run['seed'] and output['id']==case['id'],'Case/seed/output mismatch')
            require({k:case[k] for k in manifest['groups']['quality'][ordinal]}==manifest['groups']['quality'][ordinal] and digest(case['prompt_token_ids'])==case['prompt_sha256'],'Case/prompt panel mismatch')
            require(run['execution_invariants_passed'] is True and len(output['tokens'])==run['output_tokens'] and run['rounds']>0,'Invalid original run')
            require(dict(id=case['id'],round=0) in manifest['tv_probes'] and dict(round=0,status='collected') in run['preselected_probes'],'Original probe not selected/collected')
            selected=[r for r in rounds if r['case']==ordinal and r['round']==0];require(len(selected)==1,'Exactly one raw initial round required');row=selected[0]
            selected=[b for b in blocks if b['block_id']==f"{case['id']}:0"];require(len(selected)==1,'Exactly one original block required');block=selected[0]
            require(row['cache_before']==len(case['prompt_token_ids']) and len(row['proposal_tokens'])==row['verified_proposal_length']==block['proposal_length']==block['verified_length']==7,'Initial state/proposal length mismatch')
            require(block['prompt_id']==case['id'] and block['split']=='validation' and block['sampling_mode']=='stochastic' and block['collection_policy']=='full_proposal','Block policy mismatch')
            require(row['finite_and_shape_checked'] is True and row['q_dtype']=='torch.float64' and row['confidence_logits']==block['confidence_logits'] and row['accepted']==block['accepted_prefix_length'],'Original audit/acceptance mismatch')
            require(row['prefix_labels'][0]==int(row['accepted']>=1) and row['cache_after']==row['cache_before']+len(row['committed_tokens']) and output['tokens'][1:1+len(row['committed_tokens'])]==row['committed_tokens'],'Initial commit/output mismatch')
            require(len(row['selected_q'])==len(row['selected_p'])==7 and all(type(x) in (int,float) and math.isfinite(x) for values in (row['selected_q'],row['selected_p']) for x in values) and row['selected_q'][0]>0 and row['selected_p'][0]>=0,'Invalid selected probabilities')
            prefix=case['prompt_token_ids']+output['tokens'][:1]
            if ordinal in states: require(prefix==states[ordinal],'Initial semantic prefix differs')
            states[ordinal]=prefix
            metrics=[m for m in result['numerical_probes'] if m['case']==ordinal and m['round']==0]
            require(len(metrics)==8 and [m['position'] for m in metrics]==list(range(8)),'Incomplete original numerical controls')
            require(all(m['semantic_prefix_length']==len(prefix)+m['position'] for m in metrics),'Probe semantic length mismatch')
            path=probe_root/str(step)/f'private-probe-case{ordinal}-round0.pt';size=path.stat().st_size
            require(0<size<=CAPS['file_bytes'],'Probe file size cap')
            files[str(path)]=sha(path)
            probes.append(dict(step=step,ordinal=ordinal,path=str(path),size_bytes=size,sha256=files[str(path)],
                initial_output=output['tokens'][:1],proposal_tokens=row['proposal_tokens'],selected_q0=row['selected_q'][0],selected_p0=row['selected_p'][0],
                historical_alpha=min(1.,row['selected_p'][0]/row['selected_q'][0]),first_prefix_label=row['prefix_labels'][0],numerical_control=metrics[0]))
    require(sum(x['size_bytes'] for x in probes)<=CAPS['total_input_bytes'],'Total probe byte cap')
    for name,value in sources.items():files[str(source/name)]=value
    return dict(version=1,kind='six_saved_matched_initial_probe_rows',caps=CAPS,tolerance=TOL,vocab=VOCAB,probes=probes,
        input_sha256=files,source_sha256={name:sha(ROOT/name) for name in SOURCES},reference='step128 sequential row0 for each ordinal',
        raw_hash_limitation='Current recovered-file identity, not historical precommitted probe hashes')


def byte_equal(a,b):
    import torch
    return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def law(tensor,shape):
    import torch
    require(isinstance(tensor,torch.Tensor) and tensor.device.type=='cpu' and tensor.dtype==torch.float64 and tuple(tensor.shape)==tuple(shape),'Probability dtype/device/shape mismatch')
    require(bool(torch.isfinite(tensor).all()) and bool((tensor>=0).all()) and bool((tensor<=1).all()),'Invalid probability entries')
    masses=tensor.sum(-1);require(bool(((masses-1).abs()<=TOL).all()),'Probability mass exceeds tolerance')
    return masses


def overlap(p,q):
    import torch
    mass_p,mass_q=float(p.sum()),float(q.sum());l1=float((p-q).abs().sum());value=float(torch.minimum(p,q).sum())
    algebra=(mass_p+mass_q-l1)/2;mass_residual=(mass_p+mass_q)/2-1
    require(abs(value-algebra)<=TOL and abs(value-(1-l1/2)-mass_residual)<=TOL,'Overlap mass identity failed')
    return dict(overlap=value,one_minus_overlap_normalized_interpretation=1-value,l1=l1,total_variation=l1/2,
        p_mass=mass_p,q_mass=mass_q,mass_residual=mass_residual,algebra_error=value-algebra)


def tv(p,q):return float((p-q).abs().sum())/2


def validate_payload(payload,probe,*,vocab=VOCAB):
    import torch
    require(set(payload)=={'actual_q','block_probs','block_logits','proposal_tokens','sequential_probs','sequential_logits','semantic_committed_output_prefix'},'Unexpected/incomplete probe schema')
    q,p=payload['actual_q'],payload['block_probs'];law(q,(7,vocab));law(p,(8,vocab))
    require(isinstance(payload['sequential_probs'],list) and len(payload['sequential_probs'])==8 and isinstance(payload['sequential_logits'],list) and len(payload['sequential_logits'])==8,'Incomplete sequential payload')
    for row in payload['sequential_probs']:law(row,(vocab,))
    logits=payload['block_logits'];require(isinstance(logits,torch.Tensor) and logits.device.type=='cpu' and logits.dtype==torch.bfloat16 and tuple(logits.shape)==(8,vocab) and bool(torch.isfinite(logits).all()),'Block logits mismatch')
    for row in payload['sequential_logits']:
        require(isinstance(row,torch.Tensor) and row.device.type=='cpu' and row.dtype==torch.bfloat16 and tuple(row.shape)==(vocab,) and bool(torch.isfinite(row).all()),'Sequential logits mismatch')
    tokens=payload['proposal_tokens'];require(isinstance(tokens,torch.Tensor) and tokens.device.type=='cpu' and tokens.dtype==torch.long and tuple(tokens.shape)==(7,) and tokens.tolist()==probe['proposal_tokens'],'Proposal token vector mismatch')
    require(bool(((tokens>=0)&(tokens<vocab)).all()) and payload['semantic_committed_output_prefix']==probe['initial_output'],'Probe semantic prefix mismatch')
    token=probe['proposal_tokens'][0]
    require(float(q[0,token])==probe['selected_q0'] and float(p[0,token])==probe['selected_p0'],'Exact selected p/q gather mismatch')
    ps=payload['sequential_probs'][0];difference=(logits[0].double()-payload['sequential_logits'][0].double()).abs()
    current=dict(total_variation=tv(p[0],ps),max_probability_difference=float((p[0]-ps).abs().max()),
        max_absolute_logit_difference=float(difference.max()),mean_absolute_logit_difference=float(difference.mean()),
        argmax_equal=int(p[0].argmax())==int(ps.argmax()))
    expected=probe['numerical_control']
    for key,value in current.items():
        if key in ('total_variation','mean_absolute_logit_difference'):
            require(abs(value-expected[key])<=TOL,'Historical reduction reconciliation failed: '+key)
        else:require(value==expected[key],'Historical exact numerical control differs: '+key)
    return dict(q=q[0].clone(),block=p[0].clone(),seq=ps.clone()),dict(step=probe['step'],ordinal=probe['ordinal'],passed=True,
        historical_control=current,selected_q0=probe['selected_q0'],selected_p0=probe['selected_p0'],historical_alpha=probe['historical_alpha'],
        first_prefix_label=probe['first_prefix_label'],probability_rows_validated=23)


def compare_state(ordinal,rows):
    pref=rows[128]['seq'];checkpoints=[]
    for step in STEPS:
        r=rows[step];values={name:overlap(r[name],r['q']) for name in ('block','seq')};values['ref']=overlap(pref,r['q'])
        sensitivity={}
        for name in ('seq','block'):
            distance=tv(r[name],pref);mass_correction=abs(float(r[name].sum())-float(pref.sum()))/2
            bound=distance+mass_correction+TOL;observed=abs(values[name]['overlap']-values['ref']['overlap'])
            require(observed<=bound,'Teacher-law sensitivity bound failed')
            sensitivity[name]=dict(tv_to_reference=distance,mass_defect_correction=mass_correction,bound=bound,absolute_overlap_shift=observed)
        checkpoints.append(dict(step=step,overlaps=values,teacher_sensitivity=sensitivity,
            tv_block_vs_sequential=tv(r['block'],r['seq']),sequential_reference_byte_equal=byte_equal(r['seq'],pref)))
    changes=[]
    for left,right in zip(STEPS,STEPS[1:]):
        a,b=next(x for x in checkpoints if x['step']==left),next(x for x in checkpoints if x['step']==right)
        reference_delta=b['overlaps']['ref']['overlap']-a['overlaps']['ref']['overlap'];sensitivity=[]
        for teacher_step in STEPS:
            p=rows[teacher_step]['seq'];delta=overlap(p,rows[right]['q'])['overlap']-overlap(p,rows[left]['q'])['overlap']
            sensitivity.append(dict(teacher_step=teacher_step,delta=delta,sign=(delta>0)-(delta<0)))
        pair_controls={}
        for name in ('seq','block'):
            delta=b['overlaps'][name]['overlap']-a['overlaps'][name]['overlap'];bound=a['teacher_sensitivity'][name]['bound']+b['teacher_sensitivity'][name]['bound']
            require(abs(delta-reference_delta)<=bound,'Paired teacher sensitivity bound failed')
            pair_controls[name]=dict(own_pair_delta=delta,common_reference_delta=reference_delta,absolute_delta_shift=abs(delta-reference_delta),bound=bound)
        signs={x['sign'] for x in sensitivity}
        changes.append(dict(from_step=left,to_step=right,common_reference_delta=reference_delta,tv_between_draft_laws=tv(rows[left]['q'],rows[right]['q']),
            saved_sequential_teacher_sensitivity=sensitivity,sensitivity_status='same_sign' if len(signs)==1 else 'numerically_sensitive',pair_controls=pair_controls))
    return dict(ordinal=ordinal,checkpoints=checkpoints,changes=changes,all_sequential_laws_byte_equal=all(x['sequential_reference_byte_equal'] for x in checkpoints))


def peak_rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def checkpoint(deadline,out):
    if time.monotonic()>=deadline:raise TimeoutError('Cooperative CPU analysis deadline')
    require(peak_rss()<=CAPS['rss_bytes'],'CPU RSS cap exceeded')
    require(sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())<=CAPS['output_bytes'],'Additional output byte cap exceeded')


def analyze(binding,out,*,deadline=None):
    import torch
    torch.set_num_threads(2);out=Path(out);deadline=time.monotonic()+CAPS['cooperative_seconds'] if deadline is None else deadline
    status=dict(status='running',validated_probes=0,rows=[]);write(out/'private-validation.json',status)
    rows={ordinal:{} for ordinal in ORDINALS}
    try:
        for probe in binding['probes']:
            checkpoint(deadline,out);path=Path(probe['path']);require(sha(path)==probe['sha256'],'Probe changed since binding')
            with zipfile.ZipFile(path) as archive:
                require(sum(x.file_size for x in archive.infolist())<=CAPS['file_bytes'],'Expanded payload byte cap exceeded')
            payload=torch.load(path,map_location='cpu',weights_only=True)
            retained,validation=validate_payload(payload,probe);del payload
            rows[probe['ordinal']][probe['step']]=retained;status['rows'].append(validation);status['validated_probes']+=1
            write(out/'private-validation.json',status);checkpoint(deadline,out)
        states=[compare_state(ordinal,rows[ordinal]) for ordinal in ORDINALS]
        require(all(sha(path)==value for path,value in binding['input_sha256'].items()),'Original inputs changed during analysis')
        require(all(sha(ROOT/name)==value for name,value in binding['source_sha256'].items()),'Analysis source changed')
        result=dict(status='completed',state_count=2,probe_count=6,reference=binding['reference'],states=states,
            bounds=CAPS,tolerance=TOL,peak_rss_bytes=peak_rss(),binding_sha256=digest(binding),input_hashes_unchanged=True,
            limits=['Two existing matched initial states only; no quality32 or later-prefix generalization.',
                'Actual retained q only; no dense-draft/training-objective/exposure mechanism attribution.',
                'No new forward, training, generation, policy search, speed inference or independent-round CI.',binding['raw_hash_limitation']])
        status.update(status='completed',passed=True);write(out/'private-validation.json',status);write(out/'summary.json',result);checkpoint(deadline,out);return result
    except BaseException as exc:
        status.update(status='partial_deadline' if isinstance(exc,TimeoutError) else 'evidence_gap',passed=False,error_type=type(exc).__name__,error=str(exc))
        write(out/'private-validation.json',status)
        write(out/'summary.json',dict(status=status['status'],validated_probes=status['validated_probes'],error_type=status['error_type'],error=status['error'],performance_or_quality_verdict=False))
        raise


def supervise(binding,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False,mode=0o700);write(out/'binding.private.json',binding)
    env=dict(os.environ,HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    started=time.monotonic();reason=None;max_rss=0
    with (out/'worker.log').open('w') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(out)],stdout=log,stderr=subprocess.STDOUT,env=env)
        while child.poll() is None:
            try:
                raw=subprocess.check_output(['ps','-o','rss=','-p',str(child.pid)],text=True).strip();rss=int(raw or 0)*1024;max_rss=max(max_rss,rss)
            except (subprocess.CalledProcessError,ValueError):rss=0
            if time.monotonic()-started>=CAPS['timeout_seconds']:reason='timeout'
            if rss>CAPS['rss_bytes']:reason='rss_limit'
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>CAPS['output_bytes']:reason='output_limit'
            if reason:child.kill();break
            time.sleep(.1)
        code=child.wait()
    if reason:write(out/'summary.json',dict(status='partial_'+reason,performance_or_quality_verdict=False))
    write(out/'supervision.json',dict(worker_os_exit=code,status='completed' if code==0 and reason is None else 'failed',limit_reason=reason,
        elapsed_seconds=time.monotonic()-started,maximum_observed_rss_bytes=max_rss,child_reaped=True))
    return 0 if code==0 and reason is None else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('probe-root','quality-root','historical-source','output'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        binding=read(args.worker/'binding.private.json');analyze(binding,args.worker);return 0
    if not all((args.probe_root,args.quality_root,args.historical_source)):parser.error('Three explicit existing input roots required')
    binding=bind(args.probe_root,args.quality_root,args.historical_source)
    if not args.execute:
        print(json.dumps(dict(status='dry_run_no_tensor_load',binding_sha256=digest(binding),probe_count=6,total_probe_bytes=sum(p['size_bytes'] for p in binding['probes']),source_sha256=binding['source_sha256'],caps=CAPS),indent=2));return 0
    if args.output is None:parser.error('Fresh output required')
    return supervise(binding,args.output)


if __name__=='__main__':raise SystemExit(main())
