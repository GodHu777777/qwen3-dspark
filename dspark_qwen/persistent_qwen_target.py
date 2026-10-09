"""Opt-in full HF Qwen3 target over committed KV + independent verify scratch.

CPU integration candidate only. Original decoder layers/weights/RoPE/MLP/norm
are reused; no sampling law or existing target adapter is changed. Full-model
native execution and graph capture are not enabled by the first-layer probe.
"""
from dataclasses import dataclass
from collections.abc import Mapping

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
    if not isinstance(owner,PersistentQwenTarget) or owner._active_tx is None:
        raise ValueError('Owned persistent verification transaction required')
    if (attention_mask is not None or dropout or
            owner.model.model.layers[module.layer_idx].self_attn is not module):
        raise ValueError('Unexpected mask/dropout/foreign model attention')
    q=query[0].transpose(0,1).contiguous()
    k=key[0].transpose(0,1)
    v=value[0].transpose(0,1)
    out=owner.kernel(q,k,v,owner._layout,scale=scaling)
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
    """Full-model eager transaction adapter, not yet a PackedSpeculativeSession drop-in.

    prefill commits all prompt queries. verify never changes committed state;
    commit copies only caller-selected verified input prefixes (including anchor).
    Existing append/crop sampling integration must be changed explicitly later.
    Kernel injection is CPU-only; no automatic dense/native fallback exists.
    """
    def __init__(self,model,layer_ids=(),*,slots,context_capacity,max_query_tokens,test_kernel=None):
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
        if device.type!='cpu' or test_kernel is None:
            raise ValueError('This full-target candidate requires explicit CPU test attention; native full-model path is not enabled')
        self.model=model.eval().requires_grad_(False);self.kernel=test_kernel
        self.pool=PersistentTargetKV(layers=config.num_hidden_layers,slots=slots,context_capacity=context_capacity,
            max_query_tokens=max_query_tokens,heads=config.num_key_value_heads,dim=config.head_dim,
            dtype=model.get_input_embeddings().weight.dtype,device=device)
        self._handles={};self._pending=None;self._active_tx=None;self._layout=None
        self._captured={};self._closed=False;self._old_backend=config._attn_implementation
        self.cache=_VerificationCache(self)
        self._hooks=[model.model.layers[i].register_forward_hook(self._hook(i)) for i in self.layer_ids]
        ALL_ATTENTION_FUNCTIONS.register(BACKEND_NAME,persistent_attention_forward)
        config._attn_implementation=BACKEND_NAME

    @property
    def device(self):return self.pool.device

    @property
    def lengths(self):return {r:self.pool.length(h) for r,h in self._handles.items()}

    def _idle(self):
        if self._closed:raise RuntimeError('Target adapter closed')
        if self._pending is not None or self._active_tx is not None:raise RuntimeError('Finish existing verification transaction first')
        self.pool._idle()

    def _hook(self,index):
        def capture(_module,_inputs,output):
            if self._active_tx is None:raise RuntimeError('Foreign model forward while persistent adapter owns model')
            self._captured[index]=output[0] if isinstance(output,tuple) else output
        return capture

    def register_bucket(self,bucket):
        self._idle();self.pool.register_bucket(bucket)

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
        self._idle();self._validate_chunks(chunks,bucket)
        handles=tuple(self._handles[r] for r in chunks)
        tx=self.pool.begin(handles,bucket);ws=self.pool._workspaces[bucket]
        self._active_tx=tx
        self._layout=PersistentAttentionLayout(bucket.query_lengths,
            tuple(c+q for c,q in zip(tx.context_lengths,bucket.query_lengths)),
            ws.cu_query,ws.cu_key,bucket.key_capacity)
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
                selected_layer_ids=list(self.layer_ids),attention_backend='explicit_cpu_test_kernel',
                committed_write_policy='scratch_only_until_explicit_commit',full_model_native_or_graph_verified=False)
            features=PackedFeatures(output.last_hidden_state.detach(),None if context is None else context.detach(),spans,work)
            self._pending=(features,tx)
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
        if self._pending is None or self._pending[0] is not features:
            raise ValueError('Only the outstanding verification features can commit')
        _,tx=self._pending
        if not isinstance(committed_query_lengths,Mapping) or set(committed_query_lengths)!=set(features.spans):
            raise ValueError('One committed query prefix per verified request required')
        counts={self._handles[r]:n for r,n in committed_query_lengths.items()}
        result=self.pool.commit(tx,counts);self._pending=None;return result

    def abort(self,features):
        if self._pending is None or self._pending[0] is not features:
            raise ValueError('Only outstanding verification features can abort')
        self.pool.abort(self._pending[1]);self._pending=None

    @torch.no_grad()
    def prefill(self,chunks,*,bucket):
        self._idle()
        if any(self.lengths.get(r)!=0 for r in chunks):raise ValueError('Prefill is only for newly admitted empty requests')
        features=self.verify(chunks,bucket=bucket)
        self.commit(features,{r:ids.shape[1] for r,ids in chunks.items()})
        return features

    @torch.no_grad()
    def predict(self,features,last_only=False):
        """Original LM head over actual Q rows after original final model norm."""
        logits=self.model.get_output_embeddings()(features.last)
        return {r:(logits[:,offset+count-1] if last_only else logits[:,offset:offset+count])
                for r,(offset,count,_) in features.spans.items()}

    def request_kv(self,request):
        """Same per-layer [1,Hkv,C,D] auditing format as the existing target."""
        keys,values=self.pool.request_kv(self._handles[request])
        return tuple((k.transpose(0,1)[None],v.transpose(0,1)[None]) for k,v in zip(keys,values))
