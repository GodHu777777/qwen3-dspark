#!/usr/bin/env python3
"""Matched initial32 replay. Default is stdlib-only preflight; --evidence-only skips remote dependencies.

No sampling, decoder loop, optimizer or training-data reads. --execute is a separate,
explicit GPU operation; this implementation's CPU validation does not authorize it.
Native readiness is unverified: the worker refuses execution if pinned Torch does
not expose an exact pci_bus_id matching the unique physical AMD sysfs device.
"""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import analyze_matched_prefix_probes as old

STEPS = (512, 1280)
PLAN = [(s, i) for s, indices in ((512, range(2)), (1280, range(2)),
        (512, range(2,32)), (1280, range(2,32))) for i in indices]
CAPS = dict(wall_seconds=300, cooperative_seconds=290, rss_bytes=8*1024**3,
    output_bytes=1024**3, allocator_bytes=6*1024**3, free_before_load_bytes=8*1024**3,
    device_used_delta_bytes=8*1024**3)
CHECKPOINTS = {
    512: ('667d2dd6e8ad7d11e2115e936af1b6cf7e44357d68e3412f5c9ef433f26690ae',
          '30d8813eaa5a4db7a50ddbf164f17b8c045e79ae38434e378aa8e703bc699ddb'),
    1280: ('d6f21ab3af5187ed181efdbde54118a9ada21b458d9957515a17d6b3693d0de4',
           '8bb0c5113d02decac5b8a64c585284dd1664f1a5f88f13014713ec1facc7ecc5')}
RUNTIME = dict(torch='2.12.0+rocm7.2', transformers='5.17.0', hip='7.2.53211', device='AMD Radeon Graphics')
SOURCES = ('scripts/diagnose_matched_initial32.py', 'tests/test_diagnose_matched_initial32.py',
           'scripts/analyze_matched_prefix_probes.py', 'docs/matched-initial32-diagnostic.md')
require, sha, read, write, digest = old.require, old.sha, old.read, old.write, old.digest


class ReplayMismatch(ValueError): pass


def source_hashes():
    return {str(ROOT / name): sha(ROOT / name) for name in SOURCES} | {
        str(p): sha(p) for p in sorted((ROOT/'dspark_qwen').glob('*.py'))}


def bind_evidence(probe_root, quality_root, historical_source):
    previous = old.bind(probe_root, quality_root, historical_source)
    for name in ('model.py','cached_target.py','cached_decode.py','tensor_sampling.py'):
        require(sha(ROOT/'dspark_qwen'/name)==sha(Path(historical_source)/name), 'Historical core source differs')
    states=[]; runs=[]; common=None
    for step in STEPS:
        directory=Path(quality_root)/str(step)/'collection'; run=read(directory/'run.json');runs.append(run)
        require((run['checkpoint_sha256'],run['checkpoint_metadata_sha256'])==CHECKPOINTS[step], 'Wrong historical checkpoint identity')
        require(run['manifest_file_sha256']=='b5119c7f83c1f6d0ea08b58eb795bad2d1475fb6e4b71ece46ed9ae1fe00ee1e', 'Wrong original panel')
        outputs=old.lines(directory/'private-outputs.jsonl'); rounds=old.lines(directory/'private-rounds.jsonl')
        result=read(directory/'result.json'); blocks=old.lines(directory/'private-blocks.jsonl')
        initial=[x for x in rounds if x['round']==0]
        require(len(initial)==32 and sorted(x['case'] for x in initial)==list(range(32)), 'Missing/duplicate initial ordinal')
        identity=[]
        for ordinal,case in enumerate(run['cases']):
            output=outputs[ordinal]; recorded=result['runs'][ordinal]
            require(case['index']==recorded['case']==ordinal and case['split']=='validation' and
                    output['id']==case['id'] and case['seed']==recorded['seed'], 'Case/order/seed identity differs')
            require({k:case[k] for k in run['manifest']['groups']['quality'][ordinal]}==run['manifest']['groups']['quality'][ordinal] and
                    digest(case['prompt_token_ids'])==case['prompt_sha256'], 'Case differs from frozen panel')
            require(recorded['execution_invariants_passed'] is True and len(output['tokens'])==recorded['output_tokens'] and len(output['tokens'])>1, 'Incomplete original output')
            row=next(x for x in initial if x['case']==ordinal)
            matched=[x for x in blocks if x['block_id']==f"{case['id']}:0"]
            require(len(matched)==1, 'Missing/duplicate raw block');block=matched[0]
            require(len(row['proposal_tokens'])==row['verified_proposal_length']==block['proposal_length']==block['verified_length']==7 and
                    row['cache_before']==len(case['prompt_token_ids']) and row['finite_and_shape_checked'] is True and row['q_dtype']=='torch.float64', 'Original initial shape/audit differs')
            require(block['prompt_id']==case['id'] and block['split']=='validation' and block['sampling_mode']=='stochastic' and block['collection_policy']=='full_proposal' and
                    row['confidence_logits']==block['confidence_logits'] and row['accepted']==block['accepted_prefix_length'], 'Original block identity differs')
            require(row['cache_after']==row['cache_before']+len(row['committed_tokens']) and output['tokens'][1:1+len(row['committed_tokens'])]==row['committed_tokens'], 'Original committed prefix differs')
            tokens=case['prompt_token_ids']+output['tokens'][:1]+row['proposal_tokens']
            require(all(type(t) is int and 0<=t<old.VOCAB for t in tokens) and 32<=len(case['prompt_token_ids'])<=326, 'Token/length bounds differ')
            for name in ('selected_q','selected_p','confidence_logits'):
                require(len(row[name])==7 and all(type(v) in (int,float) and math.isfinite(v) for v in row[name]), 'Invalid saved scalars')
            require(all(0<=v<=1 for name in ('selected_q','selected_p') for v in row[name]) and row['selected_q'][0]>0, 'Illegal original selected probabilities')
            anchor=next((p for p in previous['probes'] if p['step']==step and p['ordinal']==ordinal),None)
            states.append(dict(step=step,ordinal=ordinal,prompt=case['prompt_token_ids'],initial_token=output['tokens'][0],
                proposal_tokens=row['proposal_tokens'],selected_q0=row['selected_q'][0],selected_p0=row['selected_p'][0],confidence0=row['confidence_logits'][0],anchor=anchor))
            identity.append((case, output['tokens'][0]))
        require(len({digest(x[0]['prompt_token_ids']) for x in identity})==32, 'Duplicate original prompt')
        if common is not None: require(identity==common, 'Initial state differs across checkpoints')
        common=identity
    return dict(version=1,states=states,plan=PLAN,caps=CAPS,reference='new step512 sequential row0 per ordinal',
        checkpoints={str(s):dict(path=r['checkpoint'],weights_sha256=r['checkpoint_sha256'],metadata_sha256=r['checkpoint_metadata_sha256']) for s,r in zip(STEPS,runs)},
        model=runs[0]['model'],draft_config=runs[0]['draft_config'],target_fingerprint=runs[0]['manifest']['identity']['target_fingerprint'],
        development_records_sha256=runs[0]['manifest']['identity']['development_records_sha256'],
        input_sha256=previous['input_sha256'],source_sha256=source_hashes(),dependencies_verified=False)


def verify_dependencies(binding):
    """Hash only explicit checkpoint/target files; never dereference config data paths."""
    files=dict(binding['input_sha256'])
    for step,checkpoint in binding['checkpoints'].items():
        path=Path(checkpoint['path']); metadata=path/'metadata.json';weights=path/'draft.safetensors'
        require(sha(metadata)==checkpoint['metadata_sha256'] and sha(weights)==checkpoint['weights_sha256'], 'Checkpoint bytes changed')
        m=read(metadata)
        require(m['step']==int(step) and m['draft_weights_sha256']==checkpoint['weights_sha256'] and m['draft_config']==binding['draft_config'], 'Checkpoint step/config differs')
        require(m['identity']['target_fingerprint']==binding['target_fingerprint'] and m['identity']['records_sha256']==binding['development_records_sha256'] and
                m['identity']['config']['model']==binding['model'], 'Checkpoint target/data identity differs')
        files[str(metadata)]=sha(metadata);files[str(weights)]=sha(weights)
    fingerprint=binding['target_fingerprint']
    require('config.json' in fingerprint and 'tokenizer.json' in fingerprint and any(n.endswith('.safetensors') for n in fingerprint), 'Incomplete target/tokenizer fingerprint')
    for name,expected in fingerprint.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe target fingerprint path')
        path=Path(binding['model'])/name;require(sha(path)==expected, 'Target/tokenizer bytes changed');files[str(path)]=expected
    cfg=read(Path(binding['model'])/'config.json')
    require(cfg['model_type']=='qwen3' and cfg['vocab_size']==old.VOCAB and not cfg.get('use_sliding_window',False), 'Wrong target architecture')
    binding['input_sha256']=files;binding['dependencies_verified']=True
    return binding


def verify_hashes(binding):
    require(all(sha(path)==expected for path,expected in binding['input_sha256'].items()), 'Bound input bytes changed')
    require(binding['source_sha256']==source_hashes(), 'Bound source bytes changed')


def law_record(tensor, shape):
    try: old.law(tensor,shape);return dict(passed=True,shape=list(tensor.shape),maximum_mass_error=float((tensor.sum(-1)-1).abs().max()))
    except ValueError as exc:return dict(passed=False,error=str(exc))


def capture_state(model,draft,state,*,amp=True,target_class=None,cache_class=None,adapter=None,observe=None):
    """The original draw-free prefix of q sampling, plus two historical target shapes."""
    import torch
    if target_class is None:
        from dspark_qwen.cached_target import CachedTarget
        target_class=CachedTarget
    if cache_class is None:
        from dspark_qwen.cached_decode import DraftContextCache
        cache_class=DraftContextCache
    if adapter is None:
        from dspark_qwen.tensor_sampling import logits_to_probabilities
        adapter=logits_to_probabilities
    device=next(model.parameters()).device;vocab=model.config.vocab_size
    ids=torch.tensor([state['prompt']],dtype=torch.long,device=device);anchor=ids.new_tensor([[state['initial_token']]])
    autocast=lambda:torch.autocast(device.type,dtype=torch.bfloat16) if amp else nullcontext()
    cpu=lambda x:x.detach().cpu().clone()
    target=cache=sequential=None
    observe=(lambda event:None) if observe is None else observe
    try:
        with torch.no_grad():
            model.eval();draft.eval();target=target_class(model,draft.spec.layer_ids)
            observe('target_prefill');features=target.prefill(ids);cache=cache_class(draft)
            with autocast():
                cache.append(features.context)
                require(target.length==cache.length==ids.shape[1], 'Initial cache length differs')
                prefill_length=target.length
                observe('draft_block');hidden=draft.backbone_cached(anchor,cache.layers,cache.length)
                require(hidden.ndim==2 and hidden.shape[0]==7, 'Draft backbone shape differs')
                base=draft.lm_head(hidden);require(tuple(base.shape)==(7,vocab), 'Draft head shape differs')
                emb=draft.markov_embedding(anchor[0])
                confidence=draft.confidence(torch.cat((hidden[0:1],emb),-1)).flatten().float()
                require(confidence.numel()==1 and bool(torch.isfinite(confidence).all()), 'Nonfinite pre-token confidence')
                logits=base[0:1]+draft.markov_projection(emb)
                q=adapter(logits,1.)[0]
            verify_ids=torch.cat((anchor,ids.new_tensor([state['proposal_tokens']])),dim=1)
            observe('target_block_append');features=target.append(verify_ids);block_logits=target.predict(features)[0]
            block=adapter(block_logits,1.)
            require(target.length==ids.shape[1]+8 and tuple(block_logits.shape)==(8,vocab), 'Block target shape/cache differs')
            raw=dict(q=cpu(q),block=cpu(block[0]),q_logits=cpu(logits[0]),block_logits=cpu(block_logits[0]),confidence0=float(confidence.item()),
                shapes=dict(prompt=list(ids.shape),backbone=list(hidden.shape),base=list(base.shape),q_adapter=list(logits.shape),block_adapter=list(block_logits.shape)))
            raw['block_laws']=law_record(cpu(block),(8,vocab))
            raw['cache_lengths']=dict(prefill=prefill_length,draft_context=cache.length,block_target=target.length)
            target.reset();cache.reset();target=cache=None
            del features,hidden,base,emb,logits,block,block_logits,q
            sequential=target_class(model,());observe('target_prefill');features=sequential.prefill(ids)
            require(sequential.length==ids.shape[1], 'Fresh sequential prefill length differs')
            raw['cache_lengths']['sequential_prefill']=sequential.length
            observe('target_anchor_append');features=sequential.append(anchor);seq_logits=sequential.predict(features,last_only=True)[0]
            seq=adapter(seq_logits,1.)
            require(sequential.length==ids.shape[1]+1, 'Sequential target cache differs')
            raw.update(seq=cpu(seq),seq_logits=cpu(seq_logits));raw['shapes']['seq_adapter']=list(seq_logits.shape)
            raw['cache_lengths']['sequential_target']=sequential.length
            return raw
    finally:
        for cache_object in (target,cache,sequential):
            if cache_object is not None:cache_object.reset()


def validate_new(raw,*,vocab=old.VOCAB):
    import torch
    for name in ('q','block','seq'):old.law(raw[name],(vocab,))
    require(raw['block_laws']['passed'], 'Invalid transient block law')
    for name in ('q_logits','block_logits','seq_logits'):
        tensor=raw[name]
        require(tensor.device.type=='cpu' and tensor.dtype==torch.bfloat16 and tuple(tensor.shape)==(vocab,) and bool(torch.isfinite(tensor).all()), 'Invalid retained logit dtype/shape/entries')
    require(math.isfinite(raw['confidence0']), 'Invalid raw confidence')
    shapes=raw['shapes'];n=shapes['prompt'][1]
    require(raw['cache_lengths']==dict(prefill=n,draft_context=n,block_target=n+8,sequential_prefill=n,sequential_target=n+1), 'Recorded cache lengths differ')
    require(shapes['q_adapter']==[1,vocab] and shapes['block_adapter']==[8,vocab] and shapes['seq_adapter']==[vocab] and shapes['base']==[7,vocab] and shapes['backbone'][0]==7, 'Recorded numerical shape differs')


def replay_checks(raw,state,*,anchor_payload=None,vocab=old.VOCAB):
    """Return all drift measurements; caller persists them before enforcing the gate."""
    token=state['proposal_tokens'][0]
    checks=dict(selected_q_equal=float(raw['q'][token])==state['selected_q0'],selected_p_equal=float(raw['block'][token])==state['selected_p0'],confidence_equal=raw['confidence0']==state['confidence0'])
    record=dict(checks=checks,selected_q0=float(raw['q'][token]),selected_p0=float(raw['block'][token]),confidence0=raw['confidence0'])
    if state['anchor'] is not None:
        old_rows,_=old.validate_payload(anchor_payload,state['anchor'],vocab=vocab)
        record['full_law_differences']={}
        for name in ('q','block','seq'):
            checks[name+'_byte_equal']=old.byte_equal(raw[name],old_rows[name])
            record['full_law_differences'][name]=dict(tv=old.tv(raw[name],old_rows[name]),max_absolute_difference=float((raw[name]-old_rows[name]).abs().max()))
        for name,key in (('block_logits','block_logits'),('seq_logits','sequential_logits')):
            checks[name+'_equal']=old.byte_equal(raw[name],anchor_payload[key][0])
        delta=(raw['block_logits'].double()-raw['seq_logits'].double()).abs()
        metrics=dict(total_variation=old.tv(raw['block'],raw['seq']),max_probability_difference=float((raw['block']-raw['seq']).abs().max()),
            max_absolute_logit_difference=float(delta.max()),mean_absolute_logit_difference=float(delta.mean()),argmax_equal=int(raw['block'].argmax())==int(raw['seq'].argmax()))
        for key,value in metrics.items():
            expected=state['anchor']['numerical_control'][key]
            checks[key+'_reconciled']=abs(value-expected)<=old.TOL if key in ('total_variation','mean_absolute_logit_difference') else value==expected
        record['historical_metrics']=metrics
    record['passed']=all(checks.values());return record


def compare_pair(ordinal,rows):
    pref=rows[512]['seq'];checkpoints=[]
    for step in STEPS:
        r=rows[step];overlaps={name:old.overlap(r[name],r['q']) for name in ('block','seq')};overlaps['ref']=old.overlap(pref,r['q']);controls={}
        for name in ('block','seq'):
            distance=old.tv(r[name],pref);correction=abs(float(r[name].sum())-float(pref.sum()))/2;bound=distance+correction+old.TOL
            shift=abs(overlaps[name]['overlap']-overlaps['ref']['overlap']);require(shift<=bound,'Teacher sensitivity bound failed')
            controls[name]=dict(tv=distance,mass_correction=correction,bound=bound,shift=shift)
        checkpoints.append(dict(step=step,overlaps=overlaps,teacher_controls=controls,tv_block_seq=old.tv(r['block'],r['seq']),seq_reference_byte_equal=old.byte_equal(r['seq'],pref)))
    left,right=checkpoints;delta=right['overlaps']['ref']['overlap']-left['overlaps']['ref']['overlap'];paired={}
    for name in ('block','seq'):
        own=right['overlaps'][name]['overlap']-left['overlaps'][name]['overlap'];bound=sum(c['teacher_controls'][name]['bound'] for c in checkpoints)
        require(abs(own-delta)<=bound,'Paired teacher sensitivity bound failed');paired[name]=dict(own_delta=own,reference_delta=delta,shift=abs(own-delta),bound=bound)
    teachers=[]
    for step in STEPS:
        p=rows[step]['seq'];d=old.overlap(p,rows[1280]['q'])['overlap']-old.overlap(p,rows[512]['q'])['overlap'];teachers.append(dict(teacher_step=step,delta=d,sign=(d>0)-(d<0)))
    return dict(ordinal=ordinal,checkpoints=checkpoints,delta=delta,sign=(delta>0)-(delta<0),tv_q=old.tv(rows[512]['q'],rows[1280]['q']),
        paired_controls=paired,teachers=teachers,numerical_sensitivity='same_sign' if len({t['sign'] for t in teachers})==1 else 'numerically_sensitive')


def aggregate(states):
    require(len(states)==32 and [s['ordinal'] for s in states]==list(range(32)), 'Complete ordered32 coverage required')
    def describe(values):return dict(equal_state_mean=math.fsum(values)/32,median=statistics.median(values),minimum=min(values),maximum=max(values))
    return dict(count=32,reference_overlap={str(step):describe([s['checkpoints'][index]['overlaps']['ref']['overlap'] for s in states]) for index,step in enumerate(STEPS)},
        delta=describe([s['delta'] for s in states]),sign_counts={name:sum(s['sign']==sign for s in states) for name,sign in (('positive',1),('zero',0),('negative',-1))})


def check_budget(out,deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Cooperative deadline')
    require(old.peak_rss()<=CAPS['rss_bytes'], 'Host RSS cap')
    require(sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())<=CAPS['output_bytes'], 'Output byte cap')


def panel(binding,out,load_step,capture,*,vocab=old.VOCAB,deadline=None):
    import torch
    out=Path(out);deadline=time.monotonic()+CAPS['cooperative_seconds'] if deadline is None else deadline
    status=dict(status='running',completed=[],raw_files={},attempted_operations={key:0 for key in ('checkpoint_load','target_prefill','draft_block','target_block_append','target_anchor_append')});write(out/'progress.private.json',status)
    def observe(event):
        status['attempted_operations'][event]+=1;write(out/'progress.private.json',status)
    states={(s['step'],s['ordinal']):s for s in binding['states']};previous=None
    try:
        require(len(states)==len(binding['states'])==64 and set(states)==set(PLAN) and [tuple(x) for x in binding['plan']]==PLAN, 'Plan/coverage differs')
        require(all((s['anchor'] is not None)==(s['ordinal']<2) for s in states.values()), 'Historical anchor coverage differs')
        for index,(step,ordinal) in enumerate(PLAN):
            check_budget(out,deadline)
            if index==4:require(status['completed']==[list(x) for x in PLAN[:4]], 'Four historical anchors must pass first')
            if step!=previous:observe('checkpoint_load');load_step(step);previous=step
            state=states[step,ordinal];raw=capture(state,observe)
            path=out/f'private-step{step}-ordinal{ordinal}.pt';torch.save(raw,path)
            status['raw_files'][path.name]=sha(path);write(out/'progress.private.json',status)
            validate_new(raw,vocab=vocab)
            anchor=None
            if state['anchor'] is not None:
                bound=state['anchor'];require(sha(bound['path'])==bound['sha256'], 'Historical anchor bytes changed')
                anchor=torch.load(bound['path'],map_location='cpu',weights_only=True)
            checks=replay_checks(raw,state,anchor_payload=anchor,vocab=vocab)
            check_path=out/f'private-check-step{step}-ordinal{ordinal}.json';write(check_path,checks)
            status['raw_files'][check_path.name]=sha(check_path);write(out/'progress.private.json',status)
            del anchor,raw
            if not checks['passed']:raise ReplayMismatch('Historical replay numerical mismatch')
            status['completed'].append([step,ordinal]);write(out/'progress.private.json',status);check_budget(out,deadline)
        require(status['attempted_operations']==dict(checkpoint_load=4,target_prefill=128,draft_block=64,target_block_append=64,target_anchor_append=64), 'Actual operation count differs')
        public=[]
        for ordinal in range(32):
            rows={step:torch.load(out/f'private-step{step}-ordinal{ordinal}.pt',map_location='cpu',weights_only=True) for step in STEPS}
            public.append(compare_pair(ordinal,rows));del rows;check_budget(out,deadline)
        verify_hashes(binding);check_budget(out,deadline)
        result=dict(status='completed',operation_counts=status['attempted_operations'],states=public,aggregate=aggregate(public),binding_sha256=digest(binding),input_hashes_unchanged=True,
            limits=['Original32 development initial states only; no later-prefix or population inference.', 'No samples, training, loss attribution, checkpoint selection or speed claim.'])
        write(out/'summary.json',result);status['status']='completed';write(out/'progress.private.json',status);return result
    except BaseException as exc:
        kind='historical_numerical_mismatch' if isinstance(exc,ReplayMismatch) else 'partial_deadline' if isinstance(exc,TimeoutError) else 'evidence_gap'
        status.update(status=kind,error_type=type(exc).__name__);write(out/'progress.private.json',status)
        write(out/'summary.json',dict(status=kind,completed_count=len(status['completed']),aggregate_available=False,quality_verdict=False,error_type=type(exc).__name__))
        raise


def telemetry(root=Path('/sys/class/drm')):
    candidates={}
    for card in sorted(Path(root).glob('card[0-9]*')):
        if not card.name[4:].isdigit():continue
        device=(card/'device').resolve()
        if not (device/'mem_info_vram_total').exists():continue
        total=int((device/'mem_info_vram_total').read_text())
        if total<=0:continue
        require((device/'vendor').read_text().strip()=='0x1002' and (device/'driver').resolve().name=='amdgpu', 'Unknown VRAM device')
        used=int((device/'mem_info_vram_used').read_text());require(0<=used<=total, 'Invalid device telemetry')
        driver=(device/'driver').resolve()
        candidates[str(device)]=dict(total=total,used=used,physical_device=str(device),drm_node=card.name,
            driver_srcversion=(driver/'module'/'srcversion').read_text().strip(),kernel_release=os.uname().release)
    require(len(candidates)==1, 'Unique physical AMD VRAM device required')
    return next(iter(candidates.values()))


def device_gate(properties, baseline, current):
    require(all(current[k]==baseline[k] for k in ('physical_device','total','driver_srcversion','kernel_release')), 'Physical device/driver baseline changed')
    bus=getattr(properties,'pci_bus_id',None)
    require(isinstance(bus,str) and bus.lower()==Path(baseline['physical_device']).name.lower(), 'Exact logical/physical PCI mapping unavailable')
    require(properties.total_memory==baseline['total'], 'Physical/logical GPU memory differs')
    return dict(pci_bus_id=bus,physical_device=baseline['physical_device'],driver_srcversion=baseline['driver_srcversion'],kernel_release=baseline['kernel_release'],logical_device=0)


def worker(out):
    binding=read(out/'binding.private.json');require(binding['dependencies_verified'], 'Dependencies unverified');verify_hashes(binding)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from dspark_qwen.model import DSparkDraft
    from dspark_qwen.config import DraftConfig
    from dspark_qwen.checkpoint import load_checkpoint
    require(torch.cuda.is_available() and torch.cuda.device_count()==1, 'One visible GPU required')
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,hip=torch.version.hip,device=torch.cuda.get_device_name(0))
    write(out/'runtime.private.json',runtime);require(runtime==RUNTIME, 'Loaded historical runtime mismatch')
    baseline=read(out/'telemetry-baseline.json');current=telemetry()
    properties=torch.cuda.get_device_properties(0)
    runtime.update(device_gate(properties,baseline,current))
    write(out/'runtime.private.json',runtime)
    free,total=torch.cuda.mem_get_info();require(free>=CAPS['free_before_load_bytes'], 'Insufficient GPU free memory')
    torch.cuda.set_per_process_memory_fraction(CAPS['allocator_bytes']/total)
    torch.manual_seed(0)
    model=AutoModelForCausalLM.from_pretrained(binding['model'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    require(model.config._attn_implementation=='sdpa', 'Wrong loaded attention backend')
    draft=DSparkDraft(model,DraftConfig.from_dict(binding['draft_config'])).to('cuda').eval()
    def load(step):
        checkpoint=binding['checkpoints'][str(step)];path=Path(checkpoint['path'])
        require(sha(path/'metadata.json')==checkpoint['metadata_sha256'] and sha(path/'draft.safetensors')==checkpoint['weights_sha256'], 'Checkpoint changed before load')
        load_checkpoint(path,draft) # weights only: optimizer/generator omitted
        require(all(p.dtype==torch.float32 for p in draft.parameters() if p.requires_grad), 'Draft trainable dtype differs')
    try:panel(binding,out,load,lambda state,observe:capture_state(model,draft,state,observe=observe),deadline=binding['deadline_monotonic'])
    finally:write(out/'allocation.private.json',dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def supervise(binding,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    started=time.monotonic();binding=dict(binding,deadline_monotonic=started+CAPS['cooperative_seconds']);write(out/'binding.private.json',binding)
    reason=None;child=None;code=None;maximum_rss=0
    try:
        baseline=telemetry();require(baseline['total']-baseline['used']>=CAPS['free_before_load_bytes'],'Insufficient telemetry free memory')
        write(out/'telemetry-baseline.json',baseline)
        with (out/'worker.log').open('w') as log, (out/'telemetry.jsonl').open('w') as monitor:
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',HIP_VISIBLE_DEVICES='0',ROCR_VISIBLE_DEVICES='0',CUDA_VISIBLE_DEVICES='0')
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(out)],stdout=log,stderr=subprocess.STDOUT,env=env)
            while child.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>=CAPS['wall_seconds']:reason='wall_limit'
                try:rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(child.pid)],text=True,timeout=2).strip() or 0)*1024
                except subprocess.CalledProcessError:
                    if child.poll() is not None:break
                    raise
                maximum_rss=max(maximum_rss,rss)
                if rss>CAPS['rss_bytes']:reason='rss_limit'
                if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>CAPS['output_bytes']:reason='output_limit'
                current=telemetry();monitor.write(json.dumps(dict(elapsed_seconds=elapsed,rss_bytes=rss,**current))+'\n');monitor.flush()
                if current['total']!=baseline['total'] or current['physical_device']!=baseline['physical_device']:raise ValueError('Device telemetry identity changed')
                if current['used']-baseline['used']>CAPS['device_used_delta_bytes']:reason='device_pressure_limit'
                if reason:break
                time.sleep(.2)
            if reason and child.poll() is None:child.kill()
            code=child.wait()
        if reason is None:
            final=telemetry();write(out/'telemetry-final.json',final)
            require(all(final[k]==baseline[k] for k in ('physical_device','total','driver_srcversion','kernel_release')), 'Final telemetry identity changed')
            if final['used']-baseline['used']>CAPS['device_used_delta_bytes']:reason='device_pressure_limit'
            if time.monotonic()-started>=CAPS['wall_seconds']:reason='wall_limit'
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>CAPS['output_bytes']:reason='output_limit'
        if reason or code!=0:
            summary=out/'summary.json'
            if summary.exists() and read(summary).get('status')=='completed':summary.replace(out/'unconfirmed-summary.private.json')
            if reason or not summary.exists():write(summary,dict(status='partial_'+reason if reason else 'evidence_gap',aggregate_available=False,quality_verdict=False))
    except BaseException as exc:
        reason='supervision_'+type(exc).__name__
        if child is not None:
            if child.poll() is None:child.kill()
            code=child.wait()
        summary=out/'summary.json'
        if summary.exists() and read(summary).get('status')=='completed':summary.replace(out/'unconfirmed-summary.private.json')
        write(summary,dict(status='evidence_gap',aggregate_available=False,quality_verdict=False,error_type=type(exc).__name__))
    finally:
        write(out/'supervision.json',dict(worker_os_exit=code,worker_started=child is not None,child_reaped=child is not None and code is not None,
            limit_reason=reason,elapsed_seconds=time.monotonic()-started,maximum_observed_rss_bytes=maximum_rss,
            rss_observation='Live child sampled; peak between polls is bounded additionally by cooperative worker checks.'))
        write(out/'artifact-sha256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifact-sha256.json'})
    return 0 if code==0 and reason is None else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('probe-root','quality-root','historical-source','output'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--evidence-only',action='store_true',help='Bind recovered local evidence; dependencies remain unverified')
    parser.add_argument('--execute',action='store_true',help='Explicit separate coordinated GPU run, after review')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:worker(args.worker);return 0
    if not all((args.probe_root,args.quality_root,args.historical_source)):parser.error('Three explicit evidence roots required')
    if args.execute and (args.evidence_only or args.output is None):parser.error('Execution requires full dependency verification and fresh output')
    try:
        binding=bind_evidence(args.probe_root,args.quality_root,args.historical_source)
        if not args.evidence_only:verify_dependencies(binding)
    except Exception as exc:
        if args.output is not None and not args.output.exists():
            args.output.mkdir(parents=True,exist_ok=False,mode=0o700)
            write(args.output/'summary.json',dict(status='preflight_evidence_gap',error_type=type(exc).__name__,aggregate_available=False,quality_verdict=False))
        raise
    if args.execute:return supervise(binding,args.output)
    print(json.dumps(dict(status='dry_run_no_tensor_load',dependencies_verified=binding['dependencies_verified'],state_count=32,checkpoint_state_count=64,
        plan=PLAN,caps=CAPS,binding_sha256=digest(binding),source_sha256=binding['source_sha256']),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
