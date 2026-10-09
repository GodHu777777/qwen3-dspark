"""Diagnostic-only observers. No model/probability implementation lives here."""
import contextlib
import functools
import json
import os
from pathlib import Path
import time

from .profile_resource_monitor import LIMITS, ResourceMonitor, write


def union_us(intervals):
    total = 0.; end = None
    for a, b in sorted(intervals):
        if b < a: raise ValueError('Negative interval')
        if end is None or a >= end: total += b-a
        elif b > end: total += b-end
        end = b if end is None else max(end, b)
    return total



class Observer:
    """Temporarily wrap original callables once; keep only scalar timing records."""
    def __init__(self, factory, native, *, events=False, timeline=False, event_backend=None,
                 record_factory=None, max_spans=LIMITS['max_spans']):
        import torch
        self.factory, self.native = factory, native
        self.events, self.timeline = events, timeline
        self.event_backend = event_backend or torch.cuda
        self.record_factory = record_factory or torch.profiler.record_function
        self.max_spans = max_spans
        self.spans = []; self.stack = []; self.patches = []; self.event_pairs = {}
        self.rounds = []; self.round_span = None; self.session = None
        self.started = None; self.origin = None; self.closed = False; self.ranges = {}

    def begin(self, name):
        if len(self.spans) >= self.max_spans: raise RuntimeError('Diagnostic span bound exceeded')
        i = len(self.spans)
        row = dict(id=i, parent=self.stack[-1] if self.stack else None, name=name,
                   round=len(self.rounds)-1 if self.round_span is not None else None,
                   start_us=(time.perf_counter()-self.started)*1e6)
        self.spans.append(row); self.stack.append(i)
        if self.timeline:
            context = self.record_factory('paired::'+name+'#'+str(i)); context.__enter__(); self.ranges[i] = context
        if self.events:
            start = self.event_backend.Event(enable_timing=True); stop = self.event_backend.Event(enable_timing=True)
            start.record(); self.event_pairs[i] = (start, stop)
        return i

    def end(self, i):
        if not self.stack or self.stack[-1] != i: raise RuntimeError('Diagnostic span nesting changed')
        if self.events and i in self.event_pairs: self.event_pairs[i][1].record()
        if i in self.ranges: self.ranges.pop(i).__exit__(None,None,None)
        self.spans[i]['end_us'] = (time.perf_counter()-self.started)*1e6
        self.stack.pop()

    @contextlib.contextmanager
    def span(self, name):
        i = self.begin(name)
        try: yield
        finally: self.end(i)

    def wrap(self, owner, attribute, label, *, round_start=False, round_end=False, session=False):
        original = getattr(owner, attribute)
        existed = attribute in vars(owner)
        raw = vars(owner).get(attribute)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            if session: self.session = args[0]
            if round_start:
                if self.round_span is not None: raise RuntimeError('Unclosed diagnostic round')
                self.rounds.append(dict(index=len(self.rounds), context_lengths=dict(args[0].target.lengths)))
                self.round_span = self.begin('complete_round')
                self.spans[self.round_span]['round'] = len(self.rounds)-1
            try:
                with self.span(label() if callable(label) else label): result = original(*args, **kwargs)
                if round_end:
                    work = result['work']['target']
                    self.rounds[-1].update(query_tokens=work['physical_query_tokens'],
                        execution_kind=work.get('execution_kind', 'native_eager_tail'), span_id=self.round_span)
                return result
            finally:
                if round_end and self.round_span is not None:
                    self.end(self.round_span); self.round_span = None
        setattr(owner, attribute, observed)
        self.patches.append((owner, attribute, existed, raw, observed))

    def __enter__(self):
        import torch
        from . import packed_sampling as spec, packed_target_sampling as target, tensor_sampling as sampling
        from .persistent_qwen_target import PersistentQwenTarget
        from .persistent_qwen_graph import FullTargetExecution, TorchGraphBackend, CPUReplayEmulator
        from .persistent_target_kv import PersistentTargetKV
        from .target_strategy import FiniteTargetBuckets
        from .packed_draft import PackedDraft
        self.started = time.perf_counter()
        if self.events:
            self.origin = self.event_backend.Event(enable_timing=True); self.origin.record()
        try:
            for cls in (spec.PackedSpeculativeSession, target.PackedTargetOnlySession):
                self.wrap(cls, 'admit', 'admission', session=True)
            self.wrap(target.PackedTargetOnlySession, 'step', 'target_round', round_start=True, round_end=True, session=True)
            self.wrap(spec.PackedSpeculativeSession, 'propose', 'draft_propose', round_start=True, session=True)
            self.wrap(spec.PackedSpeculativeSession, 'verify_commit', 'verify_accept_commit', round_end=True, session=True)
            for cls, mapping in [
                (PersistentQwenTarget, {'reset':'reset','_validate_chunks':'validate_chunks', '_layout_for':'metadata_layout',
                  'prepare_graph_verify':'prepare_graph', 'submit_graph':'target_submit','finish_graph':'target_finish',
                  'verify_eager':'target_eager', 'predict':'target_predict','commit':'target_commit','release_features':'feature_release'}),
                (PersistentTargetKV, {'begin':'metadata_begin','commit':'kv_commit','prepare_attention':'gather_attention'}),
                (FullTargetExecution, {'validate_pointers':'validate_pointers','_model_signature':'model_signature','prepare':'graph_prepare'}),
                (FiniteTargetBuckets, {'select':'bucket_select'}),
                (PackedDraft, {'backbone':'draft_backbone','append_committed':'draft_context_projection','validate_cache':'draft_source_validation'}),
                (self.native.FixedPrefix, {'validate_allocation':'allocation_source_validation'}),
                (sampling.TensorRandom, {'uniform':'rng_draw'})]:
                for attr, label in mapping.items(): self.wrap(cls, attr, label)
            for cls in (TorchGraphBackend, CPUReplayEmulator):
                self.wrap(cls, 'submit', 'backend_graph_submit'); self.wrap(cls, 'wait', 'graph_completion_wait')
            for module in (sampling, spec, target):
                for attr, label in [('logits_to_probabilities','fp64_logits_to_law'),('check_probabilities','probability_checks'),
                                    ('sample_categorical','categorical'),('verify_proposal','accept_reject'),('residual_distribution','residual')]:
                    if hasattr(module, attr): self.wrap(module, attr, label)
            # The frozen target/draft share the same head: label its call site by
            # enclosing admission/target_predict/draft_propose hierarchy.
            self.wrap(self.factory.target.model.get_output_embeddings(), 'forward', 'lm_head')
            for attr in ('markov_embedding','markov_projection','confidence'):
                self.wrap(getattr(self.factory.draft, attr), 'forward', 'draft_'+attr)
            self.wrap(torch.Tensor, 'cpu', lambda: 'private_confidence_d2h' if any(self.spans[i]['name']=='verify_accept_commit' for i in self.stack) else 'tensor_cpu_transfer')
            self.wrap(torch.Tensor, 'tolist', 'tensor_host_materialize')
            return self
        except BaseException:
            self.restore(); raise

    def restore(self):
        errors = []
        for owner, attr, existed, raw, wrapper in reversed(self.patches):
            try:
                if getattr(owner, attr) is not wrapper: errors.append(attr+' changed while observed')
                if existed: setattr(owner, attr, raw)
                else: delattr(owner, attr)
            except BaseException as exc: errors.append(attr+': '+repr(exc))
        self.patches.clear(); self.closed = True
        if errors: raise RuntimeError('Observer restoration failed: '+repr(errors))

    def __exit__(self, typ, value, tb):
        try:
            while self.stack: self.end(self.stack[-1])
            self.round_span = None
        finally: self.restore()

    def summarize(self):
        if not self.closed: raise RuntimeError('Restore wrappers before resolving events')
        if self.events:
            self.event_backend.synchronize()
            for i, (start, stop) in self.event_pairs.items():
                self.spans[i].update(gpu_stream_start_us=self.origin.elapsed_time(start)*1000,
                                    gpu_stream_end_us=self.origin.elapsed_time(stop)*1000)
            self.event_pairs.clear(); self.origin = None
        grouped = {}; by_parent = {}
        for row in self.spans: by_parent.setdefault(row['parent'],[]).append(row)
        for row in self.spans:
            children = by_parent.get(row['id'],[])
            row['inclusive_host_us'] = row['end_us']-row['start_us']
            row['exclusive_host_us'] = row['inclusive_host_us']-union_us([(x['start_us'],x['end_us']) for x in children])
            group = grouped.setdefault(row['name'], dict(calls=0,inclusive_host_us=0.,exclusive_host_us=0.))
            group['calls'] += 1
            for key in ('inclusive_host_us','exclusive_host_us'): group[key] += row[key]
        if self.events:
            for name, group in grouped.items():
                group['gpu_stream_interval_union_us'] = union_us([(x['gpu_stream_start_us'],x['gpu_stream_end_us']) for x in self.spans if x['name']==name])
        return dict(spans=self.spans, stages=grouped, rounds=self.rounds,
                    timing_semantics='Host inclusive/exclusive tree; GPU events are stream intervals including dispatch gaps, not kernel-active time. Nested stage totals and CPU waits/GPU durations must not be added.')


def analyze_trace(path, observation=None):
    """Associate GPU -> runtime -> CPU ID, then CPU containment -> observed stage.

    GPU timestamp containment only validates session scope, never attribution.
    Ambiguous correlation IDs or missing CPU parents remain explicitly unknown.
    """
    trace = json.loads(Path(path).read_text())
    complete = [x for x in trace.get('traceEvents',[]) if x.get('ph')=='X'
                and isinstance(x.get('ts'),(int,float)) and isinstance(x.get('dur'),(int,float))]
    def category(x): return x.get('cat','').lower()
    def field(x, name):
        value=next((v for k,v in x.get('args',{}).items()
                    if k.lower().replace(' ','').replace('_','')==name),None)
        return value if type(value) is int and value>0 else None
    def end(x): return x['ts']+x['dur']
    def thread_id(x):
        values=(x.get('pid'),x.get('tid'))
        return values if all(type(v) is int and v>0 for v in values) else None
    gpu = [x for x in complete if category(x) in ('kernel','gpu_memcpy','gpu_memset')]
    kernels = [x for x in gpu if category(x)=='kernel']
    runtime = [x for x in complete if category(x) in ('cuda_runtime','hip_runtime','cuda_driver','hip_driver')]
    cpu = [x for x in complete if category(x) in ('cpu_op','user_annotation')]
    sessions = [x for x in complete if x.get('name')=='paired::complete_session']
    if len(sessions)!=1: raise RuntimeError('Exactly one complete-session trace range required')
    session=sessions[0]
    if any(x['ts']<session['ts'] or end(x)>end(session) for x in gpu):
        raise RuntimeError('GPU activity is not bounded by this complete synchronized session')
    annotations=[x for x in cpu if x.get('name','').startswith('paired::')]
    round_ranges=sorted([x for x in annotations if x['name'].startswith('paired::complete_round#')],key=lambda x:x['ts'])
    observed_rounds={r['span_id']:r for r in (observation or {}).get('rounds',[])}
    if observation is not None:
        ids=[int(x['name'].rsplit('#',1)[1]) for x in round_ranges]
        if ids!=list(observed_rounds): raise RuntimeError('Complete round annotations do not match observed round identities')
    round_ids={id(x):i for i,x in enumerate(round_ranges)}
    # Sweep each CPU thread's nested named ranges once; no all-spans rescans.
    stacks={}; cpu_ids={}
    for x in sorted(cpu,key=lambda x:(x['ts'],-x['dur'],0 if x.get('name','').startswith('paired::') else 1)):
        if not (session['ts']<=x['ts'] and end(x)<=end(session)):continue
        thread=thread_id(x)
        if thread is None:continue
        stack=stacks.setdefault(thread,[])
        while stack and (end(stack[-1])<end(x) or end(stack[-1])<=x['ts']):stack.pop()
        named=x.get('name','').startswith('paired::')
        if named:stack.append(x)
        stage=stack[-1] if stack else None
        round_range=next((a for a in reversed(stack) if id(a) in round_ids),None)
        item=dict(event=x,stage=stage['name'] if stage else None,
                  round=round_ids[id(round_range)] if round_range is not None else None)
        external=field(x,'externalid')
        if external is not None:cpu_ids.setdefault(external,[]).append(item)
    all_runtime={}
    for x in runtime:
        correlation=field(x,'correlation')
        if correlation is not None:all_runtime.setdefault(correlation,[]).append(x)
    correlations={}
    for correlation,entries in all_runtime.items():
        # Count every runtime before filtering CPU chains. A second unmatched
        # runtime with the same ID makes the entire association ambiguous.
        if len(entries)!=1:continue
        x=entries[0];external=field(x,'externalid');thread=thread_id(x)
        if thread is None or external is None:continue
        candidates=[c for c in cpu_ids.get(external,[]) if thread_id(c['event'])==thread
            and c['event']['ts']<=x['ts'] and end(x)<=end(c['event'])]
        if len(candidates)==1:
            correlations[correlation]=[dict(runtime=x,cpu=candidates[0],external=external)]
    activities=[]; grouped={}; by_round={i:[] for i in range(len(round_ranges))}; by_stage={}
    for x in gpu:
        correlation=field(x,'correlation'); candidates=correlations.get(correlation,[])
        association=candidates[0] if len(candidates)==1 else None
        row=dict(name=x.get('name','unknown'),category=category(x),start_us=x['ts'],duration_us=x['dur'],
                 correlation=correlation,association='cpu_runtime_correlation' if association else 'unknown',
                 round=association['cpu']['round'] if association else None,
                 stage=association['cpu']['stage'] if association else None,
                 cpu_external_id=association['external'] if association else None,
                 cpu_operator=association['cpu']['event'].get('name') if association else None,
                 runtime_name=association['runtime'].get('name') if association else None)
        activities.append(row)
        if row['round'] is not None:by_round[row['round']].append(row)
        if row['stage'] is not None:by_stage.setdefault(row['stage'],[]).append(row)
        grouped.setdefault(row['name'],[]).append(row)
    correlated=[x for x in activities if x['association']!='unknown']
    if not kernels or not runtime or not correlated:
        raise RuntimeError('CPU+ROCm GPU timeline/correlation unavailable; CPU-only or uncorrelated trace is not sufficient')
    def metrics(rows):
        return dict(activities=len(rows),active_union_us=union_us([(r['start_us'],r['start_us']+r['duration_us']) for r in rows]))
    rounds=[]
    for i,range_ in enumerate(round_ranges):
        span_id=int(range_['name'].rsplit('#',1)[1]); original=observed_rounds.get(span_id,{})
        rounds.append(dict(original,index=i,span_id=span_id,correlated_gpu=metrics(by_round[i])))
    return dict(gpu_activities=len(gpu),kernel_activities=len(kernels),runtime_activities=len(runtime),
        correlated_gpu_activities=len(correlated),unknown_gpu_associations=len(gpu)-len(correlated),
        gpu_active_union_us=metrics(activities)['active_union_us'],
        kernel_active_union_us=metrics([r for r in activities if r['category']=='kernel'])['active_union_us'],
        kernels_by_actual_name={k:metrics(v) for k,v in grouped.items()},
        stages={k:metrics(v) for k,v in by_stage.items()},rounds=rounds,activities=activities,
        inference='GPU duration from actual GPU events; round/stage assignment requires GPU/runtime/CPU ID chain and CPU range ancestry. Unknown associations remain unassigned. Never infer attribution from GPU/host time overlap; zero correlated activity is not proof of zero GPU work.')
