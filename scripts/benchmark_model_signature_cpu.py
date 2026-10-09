#!/usr/bin/env python3
"""Bounded CPU/meta measurement of the existing full-target model signature.

No weight load, forward pass, graph execution, persistent cache, or production
signature change. Isolated component timings are not an additive GPU profile.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
ARCHITECTURE=dict(vocab_size=151936,hidden_size=1024,intermediate_size=3072,
    num_hidden_layers=28,num_attention_heads=16,num_key_value_heads=8,head_dim=128)


def named_tensors(model):
    return tuple(model.named_parameters())+tuple(model.named_buffers())


def tensor_signature(items):
    return tuple((name,t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device),t._version)
                 for name,t in items)


def tensor_signature_local_strings(items):
    """Within-call formatting reuse only. Every tensor field is still read now."""
    formats={}
    def text(value):
        found=formats.get(value)
        if found is None:found=formats[value]=str(value)
        return found
    return tuple((name,t.data_ptr(),tuple(t.shape),tuple(t.stride()),text(t.dtype),text(t.device),t._version)
                 for name,t in items)


def signature_recomposed(owner,*,local_strings=False):
    tensors=(tensor_signature_local_strings if local_strings else tensor_signature)(named_tensors(owner.model))
    return (id(owner.model),owner.model.training,tuple(owner.layer_ids),
        repr(owner.model.config.to_dict()),owner.model.config._attn_implementation,
        tuple(id(layer) for layer in owner.model.model.layers),tensors)


def measure(owner,*,iterations=50,repeats=7,warmup=10,deadline_seconds=60):
    from dspark_qwen.persistent_qwen_graph import FullTargetExecution
    execution=SimpleNamespace(target=owner)
    original=lambda:FullTargetExecution._model_signature(execution)
    items=named_tensors(owner.model);config_dict=owner.model.config.to_dict()
    functions={
        'config_to_dict':owner.model.config.to_dict,
        'config_repr_prebuilt':lambda:repr(config_dict),
        'config_to_dict_repr':lambda:repr(owner.model.config.to_dict()),
        'named_parameters_tuple':lambda:tuple(owner.model.named_parameters()),
        'named_buffers_tuple':lambda:tuple(owner.model.named_buffers()),
        'named_tensor_enumeration':lambda:named_tensors(owner.model),
        'metadata_and_version_prescanned':lambda:tensor_signature(items),
        'metadata_without_version_prescanned':lambda:tuple((n,t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device)) for n,t in items),
        'version_only_prescanned':lambda:tuple(t._version for _,t in items),
        'full_original':original,
        'full_recomposed':lambda:signature_recomposed(owner),
        'full_local_string_candidate':lambda:signature_recomposed(owner,local_strings=True)}
    expected=original()
    if expected!=signature_recomposed(owner) or expected!=signature_recomposed(owner,local_strings=True):
        raise AssertionError('Candidate/decomposition signature output differs')
    started=time.monotonic();deadline=started+deadline_seconds
    for function in functions.values():
        for _ in range(warmup):function()
    rows={name:[] for name in functions};orders=[];rng=random.Random(1729)
    for repeat in range(repeats):
        order=list(functions);rng.shuffle(order);orders.append(order)
        for name in order:
            if time.monotonic()>=deadline:raise TimeoutError('Bounded signature CPU measurement deadline')
            function=functions[name];start=time.perf_counter_ns()
            for _ in range(iterations):function()
            rows[name].append((time.perf_counter_ns()-start)/iterations)
    if original()!=expected or signature_recomposed(owner,local_strings=True)!=expected:
        raise AssertionError('Measured object changed or candidate differs after timing')
    summary={name:dict(samples_ns_per_call=values,median_ns_per_call=statistics.median(values),
        min_ns_per_call=min(values),max_ns_per_call=max(values)) for name,values in rows.items()}
    return dict(status='completed_cpu_meta_measurement',iterations=iterations,repeats=repeats,warmup=warmup,
        elapsed_seconds=time.monotonic()-started,orders=orders,components=summary,
        local_string_candidate_over_original_median=summary['full_local_string_candidate']['median_ns_per_call']/summary['full_original']['median_ns_per_call'],
        output_equivalent_before_after=True,production_signature_changed=False,
        scope='Isolated CPU/meta object timings; component microbenchmarks have separate call/allocation overhead and are not an additive stage partition or a GPU speedup measurement.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=50)
    p.add_argument('--repeats',type=int,default=7);p.add_argument('--warmup',type=int,default=10)
    args=p.parse_args(argv)
    if not (1<=args.iterations<=200 and 3<=args.repeats<=11 and 0<=args.warmup<=50):p.error('Bounded iterations1..200 repeats3..11 warmup0..50 required')
    if args.output.exists():p.error('Fresh evidence output required')
    import torch
    import transformers
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise RuntimeError('No initialized GPU context allowed')
    if args.config:
        raw=args.config.read_bytes();config_values=json.loads(raw)
        if any(config_values.get(k)!=v for k,v in ARCHITECTURE.items()):raise ValueError('Pinned Qwen3-0.6B architecture required')
        config=Qwen3Config.from_dict(config_values);config_source=dict(kind='provided_config_json',sha256=hashlib.sha256(raw).hexdigest())
    else:
        config=Qwen3Config(**ARCHITECTURE,tie_word_embeddings=True)
        config_source=dict(kind='architecture_matched_synthetic_config',note='Serialization contents are not claimed identical to a pretrained runtime config.')
    config._attn_implementation='sdpa'
    with torch.device('meta'):model=Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    owner=SimpleNamespace(model=model,layer_ids=[1,7,14,21,26])
    if any(t.device.type!='meta' for _,t in named_tensors(model)):raise AssertionError('Meta tensors only')
    result=measure(owner,iterations=args.iterations,repeats=args.repeats,warmup=args.warmup)
    result.update(architecture=ARCHITECTURE,config_source=config_source,
        config_serialized_repr_bytes=len(repr(config.to_dict()).encode()),
        parameter_tensor_count=len(tuple(model.named_parameters())),buffer_tensor_count=len(tuple(model.named_buffers())),
        unique_parameter_elements=sum(t.numel() for t in model.parameters()),
        runtime=dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,transformers=transformers.__version__,threads=torch.get_num_threads(),device='meta'),
        gpu_initialized=torch.cuda.is_initialized(),source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('scripts/benchmark_model_signature_cpu.py','dspark_qwen/persistent_qwen_graph.py')})
    if result['gpu_initialized']:raise AssertionError('Unexpected GPU initialization')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','elapsed_seconds','parameter_tensor_count','buffer_tensor_count','unique_parameter_elements','local_string_candidate_over_original_median','gpu_initialized')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
