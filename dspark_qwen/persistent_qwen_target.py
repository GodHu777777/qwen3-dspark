"""Opt-in full HF Qwen3 target over committed KV + independent verify scratch.

Original decoder layers/weights/RoPE/MLP/norm are reused. CPU oracle, explicitly
pinned native eager and bounded full-model graph execution are separate choices.
The graph implementation is CPU-prepared, not yet a measured native graph result.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import weakref

import torch
from transformers import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .packed_target import PackedFeatures
from .persistent_target_kv import PersistentTargetKV, Bucket


@dataclass(frozen=True)
class PersistentAttentionLayout:
    query_lengths: tuple
    key_lengths: tuple
    cu_query: torch.Tensor
    cu_key: torch.Tensor
    physical_key_capacity: int
    maximum_key_length: int = 0


def test_only_persistent_sdpa(query, key, value, layout, *, scale):
    """Independent CPU-only request-local causal attention, excluding capacity tail."""
    if query.device.type != 'cpu':
        raise ValueError('CPU oracle is not a native attention fallback')
    rows=[]; qo=ko=0
    for q,k in zip(layout.query_lengths,layout.key_lengths):
        visible=torch.arange(k)[None,:] <= torch.arange(k-q,k)[:,None]
        out=torch.nn.functional.scaled_dot_product_attention(
            query[qo:qo+q].transpose(0,1)[None],key[ko:ko+k].transpose(0,1)[None],
            value[ko:ko+k].transpose(0,1)[None],attn_mask=visible[None,None],
            dropout_p=0.,is_causal=False,scale=scale,enable_gqa=True)
        rows.append(out[0].transpose(0,1));qo+=q;ko+=k
    return torch.cat(rows)


def persistent_attention_forward(module,query,key,value,attention_mask,
                                 scaling=None,dropout=0.,persistent_target=None,**kwargs):
    owner=persistent_target
    if not isinstance(owner,PersistentQwenTarget) or (owner._active_tx is None and owner._graph_execution is None):
        raise ValueError('Owned persistent verification transaction required')
    if (attention_mask is not None or dropout or
            owner.model.model.layers[module.layer_idx].self_attn is not module):
        raise ValueError('Unexpected mask/dropout/foreign model attention')
    q=query[0].transpose(0,1).contiguous()
    k=key[0].transpose(0,1)
    v=value[0].transpose(0,1)
    layout=owner._graph_execution.layout if owner._graph_execution is not None else owner._layout
    out=owner.kernel(q,k,v,layout,scale=scaling)
    if not isinstance(out,torch.Tensor) or out.shape!=q.shape or out.dtype!=q.dtype or out.device!=q.device:
        raise ValueError('Persistent attention returned incompatible output')
    return out[None],None


class _VerificationCache(Cache):
    """HF update interface; no DynamicCache concatenation or committed mutation."""
    def __init__(self,owner):
        super().__init__(layers=[])
        self.owner=owner

    @property
    def is_compileable(self):
        return False  # Empty Cache.layers must not imply a compiled/captured cache.

    def update(self,key_states,value_states,layer_idx,*args,**kwargs):
        owner=self.owner;tx=owner._active_tx
        if tx is None:raise RuntimeError('Cache update outside owned transaction')
        q=tx.bucket.query_tokens
        expected=(1,owner.model.config.num_key_value_heads,q,owner.model.config.head_dim)
        if tuple(key_states.shape)!=expected or value_states.shape!=key_states.shape:
            raise ValueError('HF layer KV differs from exact packed query shape')
        owner.pool.stage_layer(tx,layer_idx,key_states[0].transpose(0,1).contiguous(),
                               value_states[0].transpose(0,1).contiguous())
        keys,values,*_=owner.pool.prepare_attention(tx,layer_idx)
        return keys.transpose(0,1)[None],values.transpose(0,1)[None]

    def get_seq_length(self,layer_idx=0):
        # Positions are explicitly supplied per request. This is only the packed
        # committed population, never a scalar position for each request.
        return sum(self.owner.lengths.values())

    def get_mask_sizes(self,*args,**kwargs):
        raise RuntimeError('Persistent packed target requires explicit request-local attention metadata')

    def get_max_cache_shape(self,layer_idx=0):
        return self.owner.pool.slot_capacity*self.owner.pool.context_capacity


BACKEND_NAME='dspark_persistent_qwen_cpu_candidate'


class PersistentQwenTarget:
    """Full-model transaction adapter with explicit eager/graph execution choices.

    prefill commits all prompt queries. verify never changes committed state;
    commit copies only caller-selected verified input prefixes (including anchor).
    Session integration requires its explicit persistent target strategy.
    Test-kernel injection is CPU-only; native and graph choices never fall back.
    """
    def __init__(self,model,layer_ids=(),*,slots,context_capacity,max_query_tokens,test_kernel=None,
                 native_backend=None,graph_backend=None,max_buckets=32,
                 workspace_byte_budget=256*1024**2,max_graph_buckets=0,graph_byte_budget=0):
        config=model.config
        if (config.model_type!='qwen3' or getattr(config,'use_sliding_window',False)
                or any(kind!='full_attention' for kind in config.layer_types)
                or config._attn_implementation!='sdpa'
                or config.rope_parameters.get('rope_type')!='default'):
            raise ValueError('Owned dense Qwen3 with original SDPA/default RoPE required')
        self.layer_ids=tuple(layer_ids)
        if (tuple(sorted(set(self.layer_ids)))!=self.layer_ids
                or any(type(i) is not int or not 0<=i<config.num_hidden_layers for i in self.layer_ids)):
            raise ValueError('Unique increasing valid decoder layer IDs required')
        device=model.get_input_embeddings().weight.device
        if native_backend is None:
            if device.type!='cpu' or test_kernel is None:
                raise ValueError('Require explicit CPU test attention or pinned native backend')
            self.kernel=test_kernel;self.attention_backend='explicit_cpu_test_kernel'
        else:
            from .rocm_varlen import BACKEND,PinnedRocmVarlenKernel
            import transformers
            if (native_backend!=BACKEND or test_kernel is not None or device.type!='cuda'
                    or transformers.__version__!='5.17.0'
                    or model.get_input_embeddings().weight.dtype!=torch.bfloat16
                    or (config.num_attention_heads,config.num_key_value_heads,config.head_dim)!=(16,8,128)):
                raise ValueError('Explicit pinned BF16 native Qwen architecture/runtime required')
            self._varlen_kernel=PinnedRocmVarlenKernel(device)
            self.kernel=self._native_attention;self.attention_backend=BACKEND
        if any(type(x) is not int or x<1 for x in (max_buckets,workspace_byte_budget)):
            raise ValueError('Positive finite bucket/workspace budget required')
        if graph_backend is not None:
            from .persistent_qwen_graph import CPUReplayEmulator,TorchGraphBackend
            if not ((device.type=='cpu' and isinstance(graph_backend,CPUReplayEmulator)) or
                    (device.type=='cuda' and native_backend is not None and type(graph_backend) is TorchGraphBackend)):
                raise ValueError('Explicit device-compatible graph backend required')
            if any(type(x) is not int or x<1 for x in (max_graph_buckets,graph_byte_budget)):
                raise ValueError('Positive finite graph count and byte budget required')
        self.model=model.eval().requires_grad_(False)
        self.pool=PersistentTargetKV(layers=config.num_hidden_layers,slots=slots,context_capacity=context_capacity,
            max_query_tokens=max_query_tokens,heads=config.num_key_value_heads,dim=config.head_dim,
            dtype=model.get_input_embeddings().weight.dtype,device=device)
        self._handles={};self._pending=None;self._active_tx=None;self._layout=None
        self._captured={};self._closed=False;self._old_backend=config._attn_implementation
        self._graph_execution=None;self._graphs={};self._graph_backend=graph_backend
        self._graph_ticket=None;self._graph_completion=None;self._feature_lease=None
        self._features=weakref.WeakValueDictionary()
        self._max_buckets=max_buckets;self._workspace_budget=workspace_byte_budget
        self._max_graph_buckets=max_graph_buckets;self._graph_budget=graph_byte_budget
        self._graph_reserved_bytes=0;self._workspace_bytes=0
        self._owner_stream=self._stream_identity()
        self.cache=_VerificationCache(self)
        self._hooks=[model.model.layers[i].register_forward_hook(self._hook(i)) for i in self.layer_ids]
        ALL_ATTENTION_FUNCTIONS.register(BACKEND_NAME,persistent_attention_forward)
        config._attn_implementation=BACKEND_NAME

    @property
    def device(self):return self.pool.device

    @property
    def lengths(self):return {r:self.pool.length(h) for r,h in self._handles.items()}

    def _stream_identity(self):
        return torch.cuda.current_stream(self.device).cuda_stream if self.device.type=='cuda' else None

    def _check_stream(self):
        if self._stream_identity()!=self._owner_stream:
            raise RuntimeError('Target, head, draft feature consumers and release must use the owner stream')

    def _idle(self):
        self._check_stream()
        if self._closed:raise RuntimeError('Target adapter closed')
        if self._pending is not None or self._active_tx is not None:raise RuntimeError('Finish existing verification transaction first')
        if self._graph_ticket is not None or self._feature_lease is not None:
            raise RuntimeError('Finish graph submission and release feature lease first')
        self.pool._idle()

    def _hook(self,index):
        def capture(_module,_inputs,output):
            if self._graph_execution is not None:
                self._graph_execution.capture_feature(index,output[0] if isinstance(output,tuple) else output)
                return
            if self._active_tx is None:raise RuntimeError('Foreign model forward while persistent adapter owns model')
            self._captured[index]=output[0] if isinstance(output,tuple) else output
        return capture

    def register_bucket(self,bucket):
        self._idle()
        if not isinstance(bucket,Bucket):raise ValueError('Explicit Bucket required')
        if bucket not in self.pool._workspaces:
            size=self.pool.workspace_bytes(bucket)
            if len(self.pool._workspaces)>=self._max_buckets or self._workspace_bytes+size>self._workspace_budget:
                raise ValueError('Finite bucket/workspace budget exceeded')
            if self._graph_backend is not None and self._workspace_bytes+size+self._graph_reserved_bytes>self._graph_budget:
                raise ValueError('Combined graph/workspace byte budget exceeded')
            self.pool.register_bucket(bucket);self._workspace_bytes+=size

    def _native_attention(self,q,k,v,layout,*,scale):
        # The ordinary pinned wrapper keeps its exact-K contract. Only this
        # explicitly selected capacity path calls the underlying pinned schema.
        if (q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype
                or tuple(q.shape[1:])!=(16,128) or tuple(k.shape[1:])!=(8,128)
                or k.shape!=v.shape or len(k)!=layout.physical_key_capacity
                or len(q)!=sum(layout.query_lengths) or scale!=128**-.5):
            raise ValueError('Pinned native capacity attention shape/dtype/scale differs')
        return self._varlen_kernel._operator(q,k,v,layout.cu_query,layout.cu_key,
            max(layout.query_lengths),layout.maximum_key_length,0.,True,False,
            scale=scale,window_size_left=None,window_size_right=None,seqused_k=None,
            alibi_slopes=None,block_table=None,num_splits=None)[0]

    def _layout_for(self,tx):
        ws=self.pool._workspaces[tx.bucket]
        return PersistentAttentionLayout(tx.bucket.query_lengths,
            tuple(c+q for c,q in zip(tx.context_lengths,tx.bucket.query_lengths)),
            ws.cu_query,ws.cu_key,tx.bucket.key_capacity,
            max(c+q for c,q in zip(tx.bucket.context_ceilings,tx.bucket.query_lengths)))

    def register_graph_bucket(self,bucket,*,reserve_bytes):
        """Explicit registration only; reservation covers retained graph allocations."""
        self._idle()
        from .persistent_qwen_graph import FullTargetExecution
        if self._graph_backend is None or type(reserve_bytes) is not int or reserve_bytes<1:
            raise ValueError('Graph backend and positive allocation reservation required')
        if bucket in self._graphs or len(self._graphs)>=self._max_graph_buckets:
            raise ValueError('Duplicate graph bucket or finite graph count exceeded')
        workspace=0 if bucket in self.pool._workspaces else self.pool.workspace_bytes(bucket)
        extra=FullTargetExecution.output_buffer_bytes(self,bucket)+reserve_bytes
        if self._workspace_bytes+workspace+self._graph_reserved_bytes+extra>self._graph_budget:
            raise ValueError('Combined graph/workspace byte budget exceeded')
        self.register_bucket(bucket)
        self._graphs[bucket]=FullTargetExecution(self,bucket,self._graph_backend,reserve_bytes)
        self._graph_reserved_bytes+=extra

    @torch.no_grad()
    def capture_graph(self,chunks,*,bucket):
        """Setup-only full-model capture; no request commit and no lazy fallback."""
        self._idle();self._validate_chunks(chunks,bucket)
        if bucket not in self._graphs:raise ValueError('Explicit registered graph bucket required')
        tx=self.pool.begin(tuple(self._handles[r] for r in chunks),bucket)
        try:
            ex=self._graphs[bucket];ex.prepare(chunks,self._layout_for(tx));ex.capture()
            self.pool.abort(tx)
        except BaseException:
            self.invalidate();raise

    @torch.no_grad()
    def prepare_graph_verify(self,chunks,*,bucket):
        self._idle();self._validate_chunks(chunks,bucket)
        from .persistent_qwen_graph import GraphTicket
        ex=self._graphs.get(bucket)
        if ex is None or ex.graph is None:raise ValueError('Explicitly captured bucket required; no capture-on-miss')
        tx=self.pool.begin(tuple(self._handles[r] for r in chunks),bucket)
        try:
            ex.prepare(chunks,self._layout_for(tx))
            receipt=self.pool.prepare_external_write(tx,ex.writer)
            spans={};offset=0
            for r,c,q in zip(chunks,tx.context_lengths,bucket.query_lengths):
                spans[r]=(offset,q,c);offset+=q
            ticket=GraphTicket(tx,receipt,ex,spans);self._graph_ticket=ticket
            return ticket
        except BaseException:
            self.pool.abort(tx);raise

    @torch.no_grad()
    def submit_graph(self,ticket):
        self._check_stream()
        from .persistent_qwen_graph import GraphCompletion
        if ticket is not self._graph_ticket or self._graph_completion is not None:
            raise ValueError('Exact unsubmitted graph ticket required')
        try:
            event=ticket.execution.submit(ticket.receipt)
            self.pool.submit_external_write(ticket.receipt,event)
            result=GraphCompletion(ticket,event);self._graph_completion=result
            return result
        except BaseException:
            self.invalidate();raise

    @torch.no_grad()
    def finish_graph(self,completion,*,wait=True):
        self._check_stream()
        if completion is not self._graph_completion or completion is None:
            raise ValueError('Exact submitted graph completion required')
        ticket=completion.ticket;ex=ticket.execution
        try:
            if wait:ex.backend.wait(completion.event)
            ready=ex.backend.complete(completion.event,ticket.receipt)
        except BaseException:
            self.invalidate();raise
        if not ready:raise RuntimeError('Graph completion has not been observed')
        try:
            self.pool.complete_external_write(ticket.receipt);ex.validate_pointers()
            work=self.pool.work(ticket.transaction)
            work.update(model_forward_calls=1,layers=self.pool.layers,
                selected_layer_ids=list(self.layer_ids),attention_backend=self.attention_backend,
                execution_kind=ex.backend.kind,committed_write_policy='scratch_only_until_explicit_commit',
                graph_scope='original HF embedding/all decoder layers/QKV/RoPE/MLP/final norm; head and sampling outside',
                graph_retained_bytes=ex.retained_graph_bytes,
                graph_memory_accounting=ex.graph_memory,
                graph_reserved_bytes=self._graph_reserved_bytes,workspace_bytes=self._workspace_bytes)
            features=PackedFeatures(ex.last,ex.context,ticket.spans,work)
            self._pending=(features,ticket.transaction);self._feature_lease=features
            self._features[id(features)]=features
            self._graph_ticket=None;self._graph_completion=None
            return features
        except BaseException:
            self.invalidate();raise

    def cancel_graph(self,ticket):
        self._check_stream()
        if ticket is not self._graph_ticket:raise ValueError('Exact graph ticket required')
        try:self.pool.abort(ticket.transaction)  # Refuses unfinished external writes.
        except BaseException:
            if self.pool.failed:self.invalidate()
            raise
        self._graph_ticket=None;self._graph_completion=None

    def invalidate(self):
        """Fail closed after uncertain async/device failure; no implicit recovery."""
        self.pool.invalidate()
        for ex in self._graphs.values():ex.invalid=True
        self._pending=None;self._graph_ticket=None;self._graph_completion=None
        self._active_tx=None;self._graph_execution=None

    def release_features(self,features):
        self._check_stream()
        if self._features.get(id(features)) is not features:
            raise ValueError('Foreign, copied or already released features')
        if self._pending is not None and self._pending[0] is features:
            raise RuntimeError('Commit or abort before releasing feature lease')
        if self._feature_lease is features:self._feature_lease=None
        del self._features[id(features)]

    def add_request(self,request):
        self._idle();handle=self.pool.add_request(request);self._handles[request]=handle;return handle

    def remove_request(self,request):
        self._idle();self.pool.remove_request(self._handles[request]);del self._handles[request]

    def reset(self):
        """Release request incarnations while preserving allocated bucket addresses."""
        self._idle()
        for request in tuple(self._handles):self.remove_request(request)
        self._captured.clear()

    def close(self):
        self._idle()
        for hook in self._hooks:hook.remove()
        self._hooks=[];self.model.config._attn_implementation=self._old_backend;self._closed=True

    def _validate_chunks(self,chunks,bucket):
        if not isinstance(chunks,Mapping) or not chunks or not isinstance(bucket,Bucket):
            raise ValueError('Nonempty ordered token mapping and explicit bucket required')
        if tuple(x.shape[1] if isinstance(x,torch.Tensor) and x.ndim==2 else -1 for x in chunks.values())!=bucket.query_lengths:
            raise ValueError('Ordered query lengths must exactly match the chosen bucket')
        for request,ids in chunks.items():
            if request not in self._handles:raise ValueError('Request not resident')
            if ids.shape[0]!=1 or ids.dtype!=torch.long or ids.device!=self.device:
                raise ValueError('Each token chunk must be [1,Q] long on target device')
            if bool(((ids<0)|(ids>=self.model.config.vocab_size)).any()):raise ValueError('Token outside vocabulary')

    @torch.no_grad()
    def verify(self,chunks,*,bucket):
        if bucket in self._graphs:
            return self.finish_graph(self.submit_graph(self.prepare_graph_verify(chunks,bucket=bucket)))
        return self.verify_eager(chunks,bucket=bucket)

    @torch.no_grad()
    def verify_eager(self,chunks,*,bucket):
        """Explicit same-backend eager control, including on registered graph buckets."""
        self._idle();self._validate_chunks(chunks,bucket)
        handles=tuple(self._handles[r] for r in chunks)
        tx=self.pool.begin(handles,bucket);ws=self.pool._workspaces[bucket]
        self._active_tx=tx
        self._layout=self._layout_for(tx)
        self._captured={};spans={};offset=0
        for request,c,q in zip(chunks,tx.context_lengths,bucket.query_lengths):
            spans[request]=(offset,q,c);offset+=q
        try:
            output=self.model.model(input_ids=torch.cat(tuple(chunks.values()),dim=1),
                position_ids=ws.positions[None],attention_mask={'full_attention':None},
                past_key_values=self.cache,use_cache=True,return_dict=True,persistent_target=self)
            if len(self.pool._staged)!=self.pool.layers or set(self._captured)!=set(self.layer_ids):
                raise RuntimeError('Full layer cache update or selected feature capture missing')
            context=torch.cat([self._captured[i] for i in self.layer_ids],dim=-1) if self.layer_ids else None
            work=self.pool.work(tx)
            work.update(model_forward_calls=1,layers=self.pool.layers,attention_heads=self.model.config.num_attention_heads,
                queried_context_lengths=list(tx.context_lengths),query_lengths=list(bucket.query_lengths),
                selected_layer_ids=list(self.layer_ids),attention_backend=self.attention_backend,
                committed_write_policy='scratch_only_until_explicit_commit',full_model_native_or_graph_verified=False)
            features=PackedFeatures(output.last_hidden_state.detach(),None if context is None else context.detach(),spans,work)
            self._pending=(features,tx)
            self._features[id(features)]=features
            return features
        except BaseException:
            # The model wrote scratch only. Aborting keeps prior committed KV,
            # including inactive residents. A poisoned store still fails closed.
            if self.pool._pending is tx and not self.pool.failed:self.pool.abort(tx)
            raise
        finally:
            self._active_tx=None;self._layout=None;self._captured={}

    @torch.no_grad()
    def commit(self,features,committed_query_lengths):
        self._check_stream()
        if self._pending is None or self._pending[0] is not features:
            raise ValueError('Only the outstanding verification features can commit')
        _,tx=self._pending
        if not isinstance(committed_query_lengths,Mapping) or set(committed_query_lengths)!=set(features.spans):
            raise ValueError('One committed query prefix per verified request required')
        counts={self._handles[r]:n for r,n in committed_query_lengths.items()}
        result=self.pool.commit(tx,counts);self._pending=None;return result

    def abort(self,features):
        self._check_stream()
        if self._pending is None or self._pending[0] is not features:
            raise ValueError('Only outstanding verification features can abort')
        self.pool.abort(self._pending[1]);self._pending=None

    @torch.no_grad()
    def prefill(self,chunks,*,bucket):
        self._idle()
        if any(self.lengths.get(r)!=0 for r in chunks):raise ValueError('Prefill is only for newly admitted empty requests')
        features=(self.verify_eager(chunks,bucket=bucket) if bucket in self._graphs
            else self.verify(chunks,bucket=bucket))
        self.commit(features,{r:ids.shape[1] for r,ids in chunks.items()})
        return features

    @torch.no_grad()
    def predict(self,features,last_only=False):
        """Original LM head over actual Q rows after original final model norm."""
        self._check_stream()
        if self._features.get(id(features)) is not features:
            raise ValueError('Foreign or released features')
        logits=self.model.get_output_embeddings()(features.last)
        return {r:(logits[:,offset+count-1] if last_only else logits[:,offset:offset+count])
                for r,(offset,count,_) in features.spans.items()}

    def request_kv(self,request):
        """Same per-layer [1,Hkv,C,D] auditing format as the existing target."""
        keys,values=self.pool.request_kv(self._handles[request])
        return tuple((k.transpose(0,1)[None],v.transpose(0,1)[None]) for k,v in zip(keys,values))
