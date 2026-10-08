"""One-forward dense Qwen3 reference for variable query lengths across requests.

There is no query padding: physical Q is sum(request query lengths). Attention
still uses a dense Q x K mask over all resident keys, including masked cross-
request pairs. This is not a sparse/efficient varlen engine or a scheduler.
Pinned to the Transformers 5.17 DynamicLayer/dict-mask API.
"""
from dataclasses import dataclass
from collections.abc import Mapping

import torch
from transformers import DynamicCache

from .cached_target import CachedFeatures


@dataclass
class PackedFeatures:
    last: torch.Tensor
    context: torch.Tensor | None
    # request -> (physical query offset, query length, request-local start)
    spans: dict
    work: dict

    def for_request(self, request_id):
        offset, count, start = self.spans[request_id]
        return CachedFeatures(start, self.last[:, offset:offset+count],
            None if self.context is None else self.context[:, offset:offset+count])


class PackedTarget:
    def __init__(self, model, layer_ids=()):
        config = model.config
        if (config.model_type != 'qwen3' or getattr(config, 'use_sliding_window', False)
                or any(kind != 'full_attention' for kind in config.layer_types)):
            raise ValueError('Only dense full-attention Qwen3 is supported')
        if config._attn_implementation != 'sdpa':
            raise ValueError('This reference requires the explicit-mask SDPA backend')
        if config.rope_parameters.get('rope_type') != 'default':
            raise ValueError('Only default RoPE: dynamic scaling can couple request lengths')
        self.layer_ids = tuple(layer_ids)
        if (tuple(sorted(set(self.layer_ids))) != self.layer_ids or
                any(type(i) is not int or not 0 <= i < config.num_hidden_layers for i in self.layer_ids)):
            raise ValueError('layer_ids must be unique increasing decoder block indices')
        self.model = model.eval().requires_grad_(False)
        self.reset()

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    def reset(self):
        self.cache = DynamicCache(config=self.model.config)
        self._lengths, self._markers = {}, {}
        self._next_marker = 0
        self.key_requests = torch.empty(0, dtype=torch.long, device=self.device)
        self.key_positions = torch.empty_like(self.key_requests)

    def add_request(self, request_id):
        if not isinstance(request_id, str) or not request_id or request_id in self._lengths:
            raise ValueError('Request ID must be a new nonempty string')
        self._lengths[request_id] = 0
        self._markers[request_id] = self._next_marker
        self._next_marker += 1

    @property
    def lengths(self):
        self.validate_cache()
        return dict(self._lengths)

    def _known(self, request_id):
        if request_id not in self._lengths:
            raise ValueError('Unknown request ID')

    def validate_cache(self):
        total = sum(self._lengths.values())
        if len(self.cache.layers) != self.model.config.num_hidden_layers:
            raise RuntimeError('Unexpected number of packed cache layers')
        if self.key_requests.numel() != total or self.key_positions.numel() != total:
            raise RuntimeError('Cache marker population differs from request lengths')
        for request, length in self._lengths.items():
            positions = self.key_positions[self.key_requests == self._markers[request]]
            if not torch.equal(positions, torch.arange(length, device=self.device)):
                raise RuntimeError('Request positions are not exactly a cached prefix')
        for layer in self.cache.layers:
            if layer.get_seq_length() != total:
                raise RuntimeError('Physical KV length differs from markers')
            if total and (layer.keys.shape[0] != 1 or layer.values.shape != layer.keys.shape):
                raise RuntimeError('Unexpected packed KV shape')

    def _gather(self, keep):
        """Gather all layers before assigning any, preserving interleaved order."""
        indices = torch.nonzero(keep, as_tuple=False).flatten()
        layers = [(layer, layer.keys.index_select(-2, indices), layer.values.index_select(-2, indices))
                  for layer in self.cache.layers if layer.is_initialized]
        requests = self.key_requests.index_select(0, indices)
        positions = self.key_positions.index_select(0, indices)
        for layer, keys, values in layers:
            layer.keys, layer.values = keys, values
        self.key_requests, self.key_positions = requests, positions

    def crop(self, request_id, length):
        """Keep one request's absolute local prefix; all other KV stays intact."""
        self._known(request_id)
        self.validate_cache()
        if type(length) is not int or not 0 <= length <= self._lengths[request_id]:
            raise ValueError('Crop length must be an integer within the cached request')
        if length == self._lengths[request_id]:
            return
        keep = (self.key_requests != self._markers[request_id]) | (self.key_positions < length)
        self._gather(keep)
        self._lengths[request_id] = length
        self.validate_cache()

    def remove_request(self, request_id):
        """Release an exited/EOS request. A future reused ID starts at position 0."""
        self.crop(request_id, 0)
        del self._lengths[request_id], self._markers[request_id]

    def request_kv(self, request_id):
        """Detached copies in local position order, for content auditing."""
        self._known(request_id)
        self.validate_cache()
        indices = torch.nonzero(self.key_requests == self._markers[request_id], as_tuple=False).flatten()
        return tuple((layer.keys.index_select(-2, indices).detach(),
                      layer.values.index_select(-2, indices).detach())
                     for layer in self.cache.layers if layer.is_initialized)

    def _validate_chunks(self, chunks):
        if not isinstance(chunks, Mapping) or not chunks:
            raise ValueError('Need a nonempty request-to-token mapping')
        for request, ids in chunks.items():
            self._known(request)
            if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1
                    or ids.shape[1] < 1 or ids.dtype != torch.long or ids.device != self.device):
                raise ValueError('Each chunk needs nonempty [1,q] long IDs on the model device')
            if bool(((ids < 0) | (ids >= self.model.config.vocab_size)).any()):
                raise ValueError('Token outside vocabulary')

    @torch.no_grad()
    def append(self, chunks):
        """Run one real model forward for any active subset, including new requests.

        Caller owns speculative acceptance, EOS decisions and draft caches. After
        a forward failure this object resets every request, because HF may have
        extended only some cache layers. Invalid inputs fail before mutation.
        """
        self._validate_chunks(chunks)
        self.validate_cache()
        spans, markers, positions = {}, [], []
        offset = 0
        causal_pairs = within_request_pairs = 0
        for request, ids in chunks.items():
            count, start = ids.shape[1], self._lengths[request]
            spans[request] = (offset, count, start)
            markers.append(torch.full((count,), self._markers[request], device=self.device, dtype=torch.long))
            positions.append(torch.arange(start, start+count, device=self.device))
            offset += count
            causal_pairs += count*start + count*(count+1)//2
            within_request_pairs += count*(start+count)
        query_requests, query_positions = torch.cat(markers), torch.cat(positions)
        key_requests = torch.cat((self.key_requests, query_requests))
        key_positions = torch.cat((self.key_positions, query_positions))
        mask = ((query_requests[:, None] == key_requests[None, :]) &
                (query_positions[:, None] >= key_positions[None, :]))[None, None]
        ids = torch.cat(tuple(chunks.values()), dim=1)
        q, k = ids.shape[1], key_requests.numel()
        work = dict(queried_requests=len(chunks), resident_requests=len(self._lengths),
            logical_query_tokens=q, physical_query_tokens=q, physical_kv_tokens=k,
            physical_kv_tokens_before=k-q,
            query_lengths=[count for _, count, _ in spans.values()],
            queried_context_lengths=[start for _, _, start in spans.values()],
            resident_context_lengths_before=list(self._lengths.values()),
            dense_attention_pairs_per_head_layer=q*k, allowed_causal_pairs=causal_pairs,
            masked_cross_request_pairs=q*k-within_request_pairs,
            masked_future_pairs=within_request_pairs-causal_pairs,
            mask_shape=list(mask.shape), mask_bytes=mask.numel()*mask.element_size(),
            attention_heads=self.model.config.num_attention_heads,
            layers=self.model.config.num_hidden_layers,
            dense_attention_pairs_all_heads_layers=q*k*self.model.config.num_attention_heads*self.model.config.num_hidden_layers,
            model_forward_calls=1,
            scope='Dense score-domain extents, not measured kernel operations: Q has no padding; no sparse skipping is claimed')
        captured, hooks = {}, []
        def hook_for(index):
            def hook(_module, _inputs, output):
                captured[index] = output[0] if isinstance(output, tuple) else output
            return hook
        try:
            for index in self.layer_ids:
                hooks.append(self.model.model.layers[index].register_forward_hook(hook_for(index)))
            output = self.model.model(input_ids=ids, position_ids=query_positions[None],
                attention_mask={'full_attention': mask}, past_key_values=self.cache,
                use_cache=True, return_dict=True)
            self.cache = output.past_key_values
            self.key_requests, self.key_positions = key_requests, key_positions
            for request, (_, count, start) in spans.items():
                self._lengths[request] = start+count
            self.validate_cache()
            context = torch.cat([captured[i] for i in self.layer_ids], dim=-1) if self.layer_ids else None
            return PackedFeatures(output.last_hidden_state.detach(),
                None if context is None else context.detach(), spans, work)
        except Exception:
            self.reset()
            raise
        finally:
            for hook in hooks:
                hook.remove()

    @torch.no_grad()
    def predict(self, features, last_only=False):
        """One projection on physical Q rows, then split by request query spans."""
        logits = self.model.get_output_embeddings()(features.last)
        result = {}
        for request, (offset, count, _) in features.spans.items():
            result[request] = (logits[:, offset+count-1] if last_only
                               else logits[:, offset:offset+count])
        return result
