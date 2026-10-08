"""Packed layout + one native varlen attention call per Qwen layer.

Native kernel import is delayed until execution (requires a sufficiently new
Torch ROCm/CUDA build). CPU tests must explicitly inject the test-only oracle;
there is no automatic dense fallback. No native kernel success is implied by
the CPU tests. The target owns its model's attention config; do not share that
model with another target or concurrent forward.
"""
from dataclasses import dataclass

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .packed_target import PackedTarget


@dataclass(frozen=True)
class PackedLayout:
    request_markers: tuple
    query_lengths: tuple
    key_lengths: tuple
    cu_query: torch.Tensor
    cu_key: torch.Tensor
    gather_indices: torch.Tensor
    physical_key_tokens: int

    @classmethod
    def from_metadata(cls, query_requests, query_positions, key_requests, key_positions):
        arrays = (query_requests, query_positions, key_requests, key_positions)
        if any(not isinstance(x, torch.Tensor) or x.ndim != 1 or x.dtype != torch.long for x in arrays):
            raise ValueError('Expected flat long request/position metadata')
        if any(x.device != query_requests.device for x in arrays):
            raise ValueError('All layout metadata must share a device')
        if query_requests.numel() == 0 or query_requests.shape != query_positions.shape or key_requests.shape != key_positions.shape:
            raise ValueError('Nonempty queries and matching marker/position lengths required')
        markers, counts = torch.unique_consecutive(query_requests, return_counts=True)
        if torch.unique(markers).numel() != markers.numel():
            raise ValueError('Each request query must form one contiguous chunk')
        gathers, q_lengths, k_lengths = [], counts.tolist(), []
        offset = 0
        for marker, count in zip(markers, q_lengths):
            indices = torch.nonzero(key_requests == marker, as_tuple=False).flatten()
            length = indices.numel()
            if length < count or not torch.equal(key_positions[indices], torch.arange(length, device=indices.device)):
                raise ValueError('Active keys must be a complete request-local prefix')
            expected_query_positions = torch.arange(length-count, length, device=indices.device)
            if not torch.equal(query_positions[offset:offset+count], expected_query_positions):
                raise ValueError('Queries must be the terminal suffix of their own key prefix')
            gathers.append(indices)
            k_lengths.append(length)
            offset += count
        def cumulative(lengths):
            ends = [0]
            for length in lengths:
                ends.append(ends[-1]+length)
            if ends[-1] >= 2**31:
                raise ValueError('Varlen cumulative lengths exceed int32')
            return torch.tensor(ends, device=query_requests.device, dtype=torch.int32)
        return cls(tuple(markers.tolist()), tuple(q_lengths), tuple(k_lengths),
            cumulative(q_lengths), cumulative(k_lengths), torch.cat(gathers), key_requests.numel())

    @property
    def query_tokens(self):
        return sum(self.query_lengths)

    @property
    def gathered_key_tokens(self):
        return sum(self.key_lengths)

    @property
    def pair_domain(self):
        return sum(q*k for q,k in zip(self.query_lengths, self.key_lengths))

    def gather_kv(self, key, value):
        if (key.ndim != 4 or key.shape[0] != 1 or value.shape != key.shape or
                key.shape[2] != self.physical_key_tokens or key.device != self.gather_indices.device or
                value.device != key.device or value.dtype != key.dtype):
            raise ValueError('Expected matching [1,Hkv,physical_K,D] cache tensors')
        # Each layer gathers all active requests together. Inactive KV remains in
        # persistent state but is absent from this transient attention input.
        return tuple(x.index_select(2, self.gather_indices).squeeze(0).transpose(0,1).contiguous()
                     for x in (key,value))


def native_varlen(query, key, value, layout, *, scale):
    if not query.is_cuda:
        raise RuntimeError('Native varlen requires ROCm/CUDA tensors; CPU oracle is test-only and never an automatic fallback')
    try:
        from torch.nn.attention.varlen import varlen_attn
    except ImportError as exc:
        raise RuntimeError('This Torch build lacks torch.nn.attention.varlen.varlen_attn; use a supported build without changing the frozen training environment') from exc
    # No catch-and-fallback: unsupported backend/dtype/GQA/shape must fail.
    return varlen_attn(query,key,value,layout.cu_query,layout.cu_key,
        max(layout.query_lengths),max(layout.key_lengths),
        scale=scale,window_size=(-1,0),enable_gqa=True)


def test_only_dense_varlen(query, key, value, layout, *, scale):
    """CPU test oracle only. Explicit per-request SDPA, not a native varlen kernel."""
    if query.device.type != 'cpu':
        raise ValueError('Test-only dense oracle cannot run on GPU')
    result, q_offset, k_offset = [], 0, 0
    for q_len,k_len in zip(layout.query_lengths,layout.key_lengths):
        q=query[q_offset:q_offset+q_len].transpose(0,1)[None]
        k=key[k_offset:k_offset+k_len].transpose(0,1)[None]
        v=value[k_offset:k_offset+k_len].transpose(0,1)[None]
        visible=(torch.arange(k_len)[None,:] <= torch.arange(k_len-q_len,k_len)[:,None])[None,None]
        out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=visible,
            dropout_p=0.0,is_causal=False,scale=scale,enable_gqa=True)
        result.append(out.squeeze(0).transpose(0,1))
        q_offset+=q_len;k_offset+=k_len
    return torch.cat(result,dim=0)


def varlen_attention_forward(module, query, key, value, attention_mask,
                             scaling=None, dropout=0.0, packed_layout=None,
                             packed_varlen_kernel=None, **kwargs):
    if attention_mask is not None or dropout:
        raise ValueError('Varlen callback requires metadata-only causal attention without dropout')
    if not isinstance(packed_layout,PackedLayout) or packed_varlen_kernel is None:
        raise ValueError('Missing explicit packed layout/kernel')
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[2] != packed_layout.query_tokens:
        raise ValueError('Expected [1,Hq,Q,D] query with layout Q')
    q=query.squeeze(0).transpose(0,1).contiguous()
    k,v=packed_layout.gather_kv(key,value)
    out=packed_varlen_kernel(q,k,v,packed_layout,scale=scaling)
    if not isinstance(out,torch.Tensor) or out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise RuntimeError('Varlen kernel returned an incompatible output')
    return out[None],None


BACKEND_NAME='dspark_native_varlen'


class VarlenPackedTarget(PackedTarget):
    def __init__(self,model,layer_ids=(),*,test_kernel=None):
        # Parent checks the original supported Qwen/default-RoPE/SDPA model.
        super().__init__(model,layer_ids)
        if test_kernel is not None and self.device.type != 'cpu':
            raise ValueError('Explicit test kernel injection is CPU-only')
        self._varlen_kernel=native_varlen if test_kernel is None else test_kernel
        self._backend_label='native_varlen_unverified' if test_kernel is None else 'test_only_cpu_dense_oracle'
        ALL_ATTENTION_FUNCTIONS.register(BACKEND_NAME,varlen_attention_forward)
        self.model.config._attn_implementation=BACKEND_NAME

    def _attention_payload(self,query_requests,query_positions,key_requests,key_positions):
        layout=PackedLayout.from_metadata(query_requests,query_positions,key_requests,key_positions)
        config=self.model.config
        per_token_kv_bytes=2*config.num_key_value_heads*config.head_dim*self.model.get_input_embeddings().weight.element_size()
        return {'full_attention':None},dict(packed_layout=layout,packed_varlen_kernel=self._varlen_kernel),dict(
            attention_backend=self._backend_label,mask_shape=None,mask_bytes=0,
            active_kv_gather_tokens=layout.gathered_key_tokens,
            inactive_kv_tokens_excluded=layout.physical_key_tokens-layout.gathered_key_tokens,
            kv_gather_payload_bytes_per_layer=layout.gathered_key_tokens*per_token_kv_bytes,
            varlen_pair_domain_per_head_layer=layout.pair_domain,
            varlen_cross_request_pair_domain=0,
            attention_adapter_calls_per_layer=1,
            native_varlen_calls_per_layer=int(self._varlen_kernel is native_varlen),
            scope='Metadata-only varlen layout; dense Q*K counts are counterfactual oracle domains. CPU injected kernel is test-only; native execution is not implied.')
