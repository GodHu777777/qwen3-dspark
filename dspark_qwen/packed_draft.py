"""Packed multi-request draft backbone; inference-only, no per-request model loop.

Reuses immutable DSparkDraft modules and a separate projected-context KV pool.
Every request sees its complete own draft block plus its committed context.
Native execution is explicitly pinned NONCAUSAL varlen, with no dense fallback.
CPU tests must inject the independent noncausal oracle explicitly.
"""
from dataclasses import dataclass

import torch

from .model import rotate
from .rocm_varlen import PinnedRocmVarlenKernel
from .varlen_target import PackedLayout

DRAFT_BACKEND = 'rocm_aten_no_window_noncausal_pinned_v1'


class PinnedRocmDraftKernel(PinnedRocmVarlenKernel):
    """Separate unverified noncausal backend; target causal gate does not cover it."""
    backend_name = DRAFT_BACKEND

    def __call__(self, query, key, value, layout, *, scale):
        self._validate_inputs(query, key, value, layout, scale)
        # Same pinned schema/input contract, deliberately different causal flag.
        output = self._operator(query, key, value, layout.cu_query, layout.cu_key,
            max(layout.query_lengths), max(layout.key_lengths), 0.0, False, False,
            scale=scale, window_size_left=None, window_size_right=None,
            seqused_k=None, alibi_slopes=None, block_table=None, num_splits=None)[0]
        if output.shape != query.shape or output.dtype != query.dtype or output.device != query.device:
            raise RuntimeError('Private noncausal draft kernel returned incompatible output')
        return output


def test_only_noncausal_varlen(query, key, value, layout, *, scale):
    """Independent CPU-only reference, all own context/block keys visible."""
    if query.device.type != 'cpu':
        raise ValueError('Dense draft oracle is CPU-test-only')
    rows = []; qo = ko = 0
    for nq, nk in zip(layout.query_lengths, layout.key_lengths):
        output = torch.nn.functional.scaled_dot_product_attention(
            query[qo:qo+nq].transpose(0, 1)[None],
            key[ko:ko+nk].transpose(0, 1)[None],
            value[ko:ko+nk].transpose(0, 1)[None],
            attn_mask=None, dropout_p=0., is_causal=False, scale=scale, enable_gqa=True)
        rows.append(output.squeeze(0).transpose(0, 1)); qo += nq; ko += nk
    return torch.cat(rows)


@dataclass(frozen=True)
class PackedDraftFeatures:
    hidden: torch.Tensor  # [all request block rows, hidden]
    requests: tuple
    block_size: int
    context_lengths: tuple
    work: dict

    def for_request(self, request):
        index = self.requests.index(request)
        return self.hidden[index*self.block_size:(index+1)*self.block_size]


class PackedDraft:
    """Projected context only: newest emitted anchor remains outside this cache.

    append_committed accepts already-selected committed target features. Rejected
    verification tails must never be passed here. Backbone reads but never extends
    persistent context. The wrapper owns mutable cache state, not model weights.
    """
    def __init__(self, draft, *, native_backend=None, test_kernel=None):
        if draft.training:
            raise ValueError('Packed draft is inference-only; call eval first')
        self.draft = draft
        self.device = draft.embedding.weight.device
        if test_kernel is not None:
            if native_backend is not None or self.device.type != 'cpu':
                raise ValueError('CPU-only test injection cannot select a native backend')
            self.kernel = test_kernel
        else:
            if native_backend != DRAFT_BACKEND:
                raise ValueError('Select explicit noncausal draft backend; no default/fallback')
            self.kernel = PinnedRocmDraftKernel(self.device)
        self.layers = []
        self._lengths = {}; self._markers = {}; self._next_marker = 0
        self.key_requests = torch.empty(0, device=self.device, dtype=torch.long)
        self.key_positions = torch.empty(0, device=self.device, dtype=torch.long)
        self.last_projection_work = None

    @property
    def lengths(self):
        return dict(self._lengths)

    def add_request(self, request):
        if not isinstance(request, str) or not request or request in self._lengths:
            raise ValueError('Require a fresh nonempty request name')
        self._lengths[request] = 0; self._markers[request] = self._next_marker
        self._next_marker += 1

    def _request(self, request):
        if request not in self._lengths:
            raise KeyError(request)

    def validate_cache(self):
        self._validate_state(self.layers, self._lengths, self.key_requests, self.key_positions)

    def _validate_state(self, layers, lengths, key_requests, key_positions):
        if key_requests.shape != key_positions.shape or key_requests.numel() != sum(lengths.values()):
            raise RuntimeError('Draft cache metadata length mismatch')
        for request, length in lengths.items():
            positions = key_positions[key_requests == self._markers[request]]
            if not torch.equal(positions, torch.arange(length, device=self.device)):
                raise RuntimeError('Draft cache must be the complete request-local committed prefix')
        if not layers:
            if key_requests.numel():
                raise RuntimeError('Nonempty draft metadata without projected KV')
            return
        if len(layers) != len(self.draft.layers):
            raise RuntimeError('Draft cache layer count mismatch')
        for (k, v), layer in zip(layers, self.draft.layers):
            expected = (1, layer.attention.kv_heads, key_requests.numel(), layer.attention.dim)
            if k.shape != expected or v.shape != expected or not k.is_floating_point() or not v.is_floating_point() or k.device != self.device or v.device != self.device:
                raise RuntimeError('Draft cache KV shape/device/dtype mismatch')

    def request_kv(self, request):
        self._request(request)
        indices = torch.nonzero(self.key_requests == self._markers[request], as_tuple=False).flatten()
        return tuple((k.index_select(2, indices), v.index_select(2, indices)) for k, v in self.layers)

    @torch.no_grad()
    def append_committed(self, chunks):
        if not chunks:
            raise ValueError('Require at least one committed feature chunk')
        self.validate_cache()
        features = []; positions = []; markers = []; lengths = dict(self._lengths)
        for request, value in chunks.items():
            self._request(request)
            if (not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1 or
                    value.shape[1] < 1 or value.shape[2] != self.draft.fc.in_features or
                    value.device != self.device or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
                raise ValueError('Expected finite nonempty [1, tokens, target_features] on draft device')
            n = value.shape[1]; start = lengths[request]
            features.append(value.detach())
            positions.append(torch.arange(start, start+n, device=self.device))
            markers.append(torch.full((n,), self._markers[request], device=self.device, dtype=torch.long))
            lengths[request] += n
        # Exactly one context projection and norm across all committed chunks.
        context = self.draft.context_norm(self.draft.fc(torch.cat(features, dim=1)))
        pos = torch.cat(positions)
        cos, sin = self.draft.rotary(context, pos[None])
        projected = [layer.attention.project_kv(context, cos, sin) for layer in self.draft.layers]
        if any(not bool(torch.isfinite(t).all()) for kv in projected for t in kv):
            raise RuntimeError('Nonfinite projected draft context')
        had_cache = bool(self.layers)
        if self.layers:
            if any(any(a.dtype != b.dtype for a,b in zip(old,new)) for old,new in zip(self.layers,projected)):
                raise ValueError('Draft cache precision cannot change across appends')
            projected = [(torch.cat((ok, k), dim=2), torch.cat((ov, v), dim=2))
                         for (ok, ov), (k, v) in zip(self.layers, projected)]
        # Build and validate every candidate tensor/metadata value before commit.
        new_positions = torch.cat((self.key_positions, pos))
        new_requests = torch.cat((self.key_requests, torch.cat(markers)))
        work = dict(requests=len(chunks),new_context_tokens=pos.numel(),
            context_projection_calls=1,kv_projection_calls=len(self.draft.layers),
            cache_concatenation_kv_elements=sum(k.numel()+v.numel() for k,v in projected) if had_cache else 0,
            cache_concatenation_kv_bytes=sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in projected) if had_cache else 0)
        self._validate_state(projected, lengths, new_requests, new_positions)
        self.layers, self._lengths, self.key_requests, self.key_positions, self.last_projection_work = (
            projected, lengths, new_requests, new_positions, work)

    def crop(self, request, length):
        self._request(request)
        if type(length) is not int or not 0 <= length <= self._lengths[request]:
            raise ValueError('Invalid committed draft crop length')
        keep = (self.key_requests != self._markers[request]) | (self.key_positions < length)
        layers = [(k[:, :, keep], v[:, :, keep]) for k, v in self.layers]
        requests = self.key_requests[keep]; positions = self.key_positions[keep]
        lengths = dict(self._lengths); lengths[request] = length
        self._validate_state(layers, lengths, requests, positions)
        self.layers, self._lengths, self.key_requests, self.key_positions = layers, lengths, requests, positions

    def remove_request(self, request):
        self.crop(request, 0)
        del self._lengths[request]; del self._markers[request]
        self.validate_cache()

    @torch.no_grad()
    def backbone(self, anchors):
        if not anchors:
            raise ValueError('Require active anchor requests')
        if self.draft.training:
            raise ValueError('Packed draft is inference-only')
        self.validate_cache()
        size = self.draft.spec.block_size; requests = tuple(anchors)
        tokens = []; positions = []; markers = []
        for request, anchor in anchors.items():
            self._request(request)
            if (not isinstance(anchor, torch.Tensor) or anchor.shape != (1, 1) or anchor.dtype != torch.long or
                    anchor.device != self.device or not bool(((anchor >= 0) & (anchor < self.draft.embedding.num_embeddings)).all())):
                raise ValueError('Require one valid [1,1] long anchor per active request')
            block = anchor.new_full((1, size), self.draft.spec.mask_token_id); block[:, :1] = anchor
            tokens.append(block); start = self._lengths[request]
            positions.append(torch.arange(start, start+size, device=self.device))
            markers.append(torch.full((size,), self._markers[request], device=self.device, dtype=torch.long))
        query_positions = torch.cat(positions); query_requests = torch.cat(markers)
        layout = PackedLayout.from_metadata(query_requests, query_positions,
            torch.cat((self.key_requests, query_requests)), torch.cat((self.key_positions, query_positions)))
        z = self.draft.embedding(torch.cat(tokens, dim=1))
        cos, sin = self.draft.rotary(z, query_positions[None])
        concat_elements = concat_bytes = gather_elements = gather_bytes = cast_elements = cast_bytes = 0
        for index, layer in enumerate(self.draft.layers):
            attention = layer.attention; normalized = layer.input_norm(z)
            q = attention.q_norm(attention.q_proj(normalized).view(1, -1, attention.heads, attention.dim)).transpose(1, 2)
            q = rotate(q, cos, sin).squeeze(0).transpose(0, 1).contiguous()
            kb, vb = attention.project_kv(normalized, cos, sin)
            if self.layers:
                kc, vc = self.layers[index]
                if kb.dtype != kc.dtype or vb.dtype != vc.dtype:
                    raise ValueError('Backbone/cache precision mismatch')
                kb, vb = torch.cat((kc, kb), dim=2), torch.cat((vc, vb), dim=2)
                concat_elements += kb.numel()+vb.numel()
                concat_bytes += kb.numel()*kb.element_size()+vb.numel()*vb.element_size()
            # Existing FP32 RMSNorm can leave K FP32 while V is BF16 under AMP.
            # Preserve that persistent representation and reproduce SDPA's cast
            # at the attention boundary, which a private ATen call cannot assume.
            k, v = (t.index_select(2, layout.gather_indices).squeeze(0).transpose(0,1).contiguous()
                    for t in (kb, vb))
            gather_elements += k.numel()+v.numel()
            gather_bytes += k.numel()*k.element_size()+v.numel()*v.element_size()
            if torch.is_autocast_enabled(self.device.type):
                dtype = torch.get_autocast_dtype(self.device.type)
                for value in (q, k, v):
                    if value.dtype != dtype:
                        cast_elements += value.numel()
                        cast_bytes += value.numel()*torch.empty((),dtype=dtype).element_size()
                q, k, v = (value.to(dtype) for value in (q, k, v))
            output = self.kernel(q, k, v, layout, scale=attention.dim**-.5)
            if (not isinstance(output, torch.Tensor) or output.shape != q.shape or
                    output.device != q.device or output.dtype != q.dtype or not bool(torch.isfinite(output).all())):
                raise RuntimeError('Invalid packed draft attention output')
            z = z + attention.o_proj(output.reshape(1, layout.query_tokens, -1))
            z = z + layer.mlp(layer.post_norm(z))
        hidden = self.draft.norm(z).squeeze(0)
        if not bool(torch.isfinite(hidden).all()):
            raise RuntimeError('Nonfinite packed draft hidden')
        return PackedDraftFeatures(hidden, requests, size, tuple(self._lengths[r] for r in requests),
            dict(requests=len(requests),query_rows=layout.query_tokens,
                gathered_key_rows=layout.gathered_key_tokens,physical_context_rows=self.key_requests.numel(),
                attention_pair_domain=layout.pair_domain,attention_calls=len(self.draft.layers),
                context_block_concatenation_kv_elements=concat_elements,context_block_concatenation_kv_bytes=concat_bytes,
                gathered_kv_elements=gather_elements,gathered_kv_bytes=gather_bytes,
                attention_cast_elements=cast_elements,attention_cast_output_bytes=cast_bytes,
                mlp_calls=len(self.draft.layers),scope='Work counts only; no speed claim'))
