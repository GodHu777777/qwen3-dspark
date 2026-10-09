"""Fixed-buffer full-HF target graph execution and explicit CPU lifecycle emulator.

No sampling, commit, capability validation or metadata construction in the graph.
The CPU emulator is only a host/ownership test implementation, never GPU evidence.
"""
from dataclasses import dataclass
import weakref
import torch
from transformers import Cache


@dataclass(frozen=True)
class GraphTicket:
    transaction: object
    receipt: object
    execution: object
    spans: dict


@dataclass(frozen=True)
class GraphCompletion:
    ticket: GraphTicket
    event: object


@dataclass(frozen=True)
class _BackendEvent:
    receipt: object
    payload: object


class _EventOwner:
    def __init__(self):
        self._events=weakref.WeakValueDictionary()
        self._programs={}

    def _event(self,receipt,payload):
        event=_BackendEvent(receipt,payload);self._events[id(event)]=event
        return event

    def _owns(self,event,receipt):
        return (isinstance(event,_BackendEvent) and self._events.get(id(event)) is event
                and event.receipt is receipt)

    def _bind_program(self,graph,writer):
        key=id(graph)
        self._programs[key]=(weakref.ref(graph,lambda _:self._programs.pop(key,None)),writer)

    def _check_program(self,graph,receipt):
        program=self._programs.get(id(graph))
        if program is None or program[0]() is not graph or program[1] is not receipt.writer:
            raise ValueError('Submission receipt belongs to a different captured writer/program')


def graph_pool_accounting(pool_id,segments,device_index,global_before,global_after,reserve_bytes):
    """Count scoped allocator segments, including inactive retained graph blocks.

    Snapshot acquisition is explicitly scoped to graph.pool(); unsupported or
    inconsistent schema fails closed. No allocated-by-live-tensors fallback.
    """
    if (not isinstance(pool_id,tuple) or len(pool_id)!=2 or pool_id==(0,0)
            or any(type(n) is not int or n<0 for n in pool_id)
            or not isinstance(segments,list) or not segments):
        raise RuntimeError('Independent graph private pool snapshot required')
    addresses=set();reserved=0;records=[]
    for segment in segments:
        if not isinstance(segment,dict):raise RuntimeError('Unsupported graph pool segment schema')
        for name in ('address','total_size','allocated_size','active_size','device'):
            if type(segment.get(name)) is not int:raise RuntimeError('Incomplete graph pool segment schema')
        address,size=segment['address'],segment['total_size']
        if (address<=0 or size<=0 or address in addresses or segment['device']!=device_index
                or not 0<=segment['allocated_size']<=segment['active_size']<=size):
            raise RuntimeError('Duplicate, foreign-device or inconsistent graph pool segment')
        if 'segment_pool_id' in segment and tuple(segment['segment_pool_id'])!=pool_id:
            raise RuntimeError('Snapshot contains a different private pool')
        blocks=segment.get('blocks')
        if (not isinstance(blocks,list) or not blocks
                or any(not isinstance(b,dict) or type(b.get('size')) is not int or b['size']<=0 for b in blocks)
                or sum(b['size'] for b in blocks)!=size):
            raise RuntimeError('Incomplete graph retained block accounting')
        addresses.add(address);reserved+=size
        records.append({k:segment[k] for k in ('address','total_size','allocated_size','active_size','device')})
    if (any(type(v) is not int or v<0 for v in (global_before,global_after,reserve_bytes))
            or reserved>global_after):
        raise RuntimeError('Graph private pool exceeds inconsistent global reserved accounting')
    result=dict(kind='graph_private_pool_reserved_segments',pool_id=list(pool_id),
        private_pool_reserved_bytes=reserved,segments=records,global_reserved_before=global_before,
        global_reserved_after=global_after,global_reserved_delta=global_after-global_before,
        reservation_bytes=reserve_bytes,includes_inactive_blocks=True,transient_peak_measured=False)
    if reserved>reserve_bytes:
        error=RuntimeError(f'Graph private-pool reserved bytes exceed registered reservation: measured={reserved}, reservation={reserve_bytes}')
        error.memory_accounting=result;error.pool_snapshot=segments
        raise error
    return result


class TorchGraphBackend(_EventOwner):
    """One caller stream, event-backed completion, no hidden eager fallback."""
    kind = 'rocm_full_target_graph'

    def __init__(self):
        super().__init__();self._pool_ids=set()

    def capture(self, body, device, reserve_bytes, writer):
        if device.type != 'cuda':
            raise ValueError('Actual CUDA/ROCm device required for graph capture')
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        before = torch.cuda.memory_reserved(device)
        with torch.cuda.stream(stream):
            body(); body()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            body()
        torch.cuda.current_stream(device).wait_stream(stream)
        stream.synchronize()
        pool_id=graph.pool()
        if pool_id in self._pool_ids:raise RuntimeError('Graph pool sharing is not registered or budgeted')
        segments=torch.cuda.memory.memory_snapshot(mempool_id=pool_id,include_traces=False)
        memory=graph_pool_accounting(pool_id,segments,device.index,
            before,torch.cuda.memory_reserved(device),reserve_bytes)
        self._pool_ids.add(pool_id);self._bind_program(graph,writer)
        return graph,memory

    def submit(self, graph, receipt, device):
        self._check_program(graph,receipt)
        graph.replay()
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        return self._event(receipt,event)

    def wait(self, event):
        if not self._owns(event,event.receipt):raise ValueError('Foreign completion event')
        event.payload.synchronize()

    def complete(self, event, receipt):
        return self._owns(event,receipt) and event.payload.query()


class CPUReplayEmulator(_EventOwner):
    """Explicit CPU-only emulator: executes Python on each replay, not a graph."""
    kind = 'cpu_replay_emulator_not_gpu_graph'

    def capture(self, body, device, reserve_bytes, writer):
        if device.type != 'cpu':raise ValueError('CPU emulator must never receive GPU tensors')
        body()
        self._bind_program(body,writer)
        return body,dict(kind='cpu_emulator_no_gpu_accounting',private_pool_reserved_bytes=0)

    def submit(self, graph, receipt, device):
        self._check_program(graph,receipt)
        graph()
        return self._event(receipt,True)

    def wait(self, event):
        pass

    def complete(self, event, receipt):
        return self._owns(event,receipt) and event.payload is True


class _CapturedCache(Cache):
    """Capture-time HF interface emitting only deterministic scratch/gather work.

    These Python calls do not recur in real replay. Complete layer coverage is
    checked at capture; the trusted program receipt covers later submissions.
    """
    def __init__(self, execution):
        super().__init__(layers=[])
        self.execution = execution

    @property
    def is_compileable(self):return False

    def get_seq_length(self, layer_idx=0):
        raise RuntimeError('Captured target requires explicit persistent positions')

    def get_mask_sizes(self, *args, **kwargs):
        raise RuntimeError('Captured target requires explicit attention mapping')

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        ex = self.execution;pool=ex.target.pool;ws=ex.workspace;q=ex.bucket.query_tokens
        if layer_idx in ex.seen_layers:raise RuntimeError('Duplicate captured layer update')
        ex.seen_layers.add(layer_idx)
        pool.scratch_keys[layer_idx,:q].copy_(key_states[0].transpose(0,1))
        pool.scratch_values[layer_idx,:q].copy_(value_states[0].transpose(0,1))
        for resident,scratch,old,new,result in (
                (pool.keys,pool.scratch_keys,ws.old_k,ws.new_k,ws.keys),
                (pool.values,pool.scratch_values,ws.old_v,ws.new_v,ws.values)):
            torch.index_select(resident[layer_idx].reshape(-1,pool.heads,pool.dim),0,ws.committed_indices,out=old)
            torch.index_select(scratch[layer_idx],0,ws.scratch_indices,out=new)
            torch.where(ws.from_scratch,new,old,out=result)
            result.masked_fill_(~ws.valid,0)
        return ws.keys.transpose(0,1)[None],ws.values.transpose(0,1)[None]


class FullTargetExecution:
    """A bounded bucket's fixed input/output buffers plus one captured program."""
    def __init__(self, target, bucket, backend, reserve_bytes):
        self.target,self.bucket,self.backend,self.reserve_bytes=target,bucket,backend,reserve_bytes
        self.workspace=target.pool._workspaces[bucket]
        q=bucket.query_tokens;hidden=target.model.config.hidden_size
        self.input_ids=torch.empty((1,q),dtype=torch.long,device=target.device)
        self.last=torch.empty((1,q,hidden),dtype=target.pool.dtype,device=target.device)
        self.context=(torch.empty((1,q,hidden*len(target.layer_ids)),dtype=target.pool.dtype,device=target.device)
            if target.layer_ids else None)
        self.cache=_CapturedCache(self)
        self.layout=None;self.graph=None;self.retained_graph_bytes=0;self.graph_memory=None
        self.seen_layers=set();self.seen_features=set();self.invalid=False
        self.writer=target.pool.register_external_writer(bucket,range(target.pool.layers),backend.complete)
        self.addresses=self.pointers()
        self.model_signature=self._model_signature()

    def _model_signature(self):
        owner=self.target
        tensors=tuple((name,t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device),t._version)
            for name,t in tuple(owner.model.named_parameters())+tuple(owner.model.named_buffers()))
        return (id(owner.model),owner.model.training,tuple(owner.layer_ids),
            repr(owner.model.config.to_dict()),owner.model.config._attn_implementation,
            tuple(id(layer) for layer in owner.model.model.layers),tensors)

    @staticmethod
    def output_buffer_bytes(target,bucket):
        return 8*bucket.query_tokens + bucket.query_tokens*target.model.config.hidden_size*target.pool.keys.element_size()*(1+len(target.layer_ids))

    def pointers(self):
        result=dict(input_ids=self.input_ids.data_ptr(),last=self.last.data_ptr(),**self.target.pool.pointers(self.bucket))
        if self.context is not None:result['context']=self.context.data_ptr()
        return result

    def validate_pointers(self):
        if self.invalid or self.pointers()!=self.addresses or self._model_signature()!=self.model_signature:
            raise RuntimeError('Graph executor, model signature or captured buffer addresses changed')

    def prepare(self, chunks, layout):
        self.validate_pointers();self.layout=layout;offset=0
        for ids in chunks.values():
            n=ids.shape[1];self.input_ids[:,offset:offset+n].copy_(ids);offset+=n

    def capture_feature(self,index,hidden):
        if index in self.seen_features:raise RuntimeError('Duplicate selected-layer output')
        self.seen_features.add(index)
        width=self.target.model.config.hidden_size
        offset=self.target.layer_ids.index(index)*width
        self.context[:,:,offset:offset+width].copy_(hidden)

    def body(self):
        owner=self.target
        if owner._graph_execution is not None:raise RuntimeError('Nested target graph body')
        self.seen_layers=set();self.seen_features=set();owner._graph_execution=self
        try:
            output=owner.model.model(input_ids=self.input_ids,position_ids=self.workspace.positions[None],
                attention_mask={'full_attention':None},past_key_values=self.cache,use_cache=True,
                return_dict=True,persistent_target=owner)
            self.last.copy_(output.last_hidden_state)
            if self.seen_layers!=set(range(owner.pool.layers)) or self.seen_features!=set(owner.layer_ids):
                raise RuntimeError('Captured full-layer/selected-feature signature incomplete')
        finally:owner._graph_execution=None

    def capture(self):
        self.validate_pointers()
        if self.graph is not None:raise ValueError('Bucket already captured')
        self.graph,self.graph_memory=self.backend.capture(self.body,self.target.device,self.reserve_bytes,self.writer)
        self.retained_graph_bytes=self.graph_memory['private_pool_reserved_bytes']
        self.validate_pointers()

    def submit(self, receipt):
        self.validate_pointers()
        if self.graph is None:raise RuntimeError('Explicit bucket capture required before submission')
        if receipt.writer is not self.writer:
            raise ValueError('Submission receipt does not belong to this captured writer')
        return self.backend.submit(self.graph,receipt,self.target.device)
