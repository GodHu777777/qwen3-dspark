"""Optional experimental fixed-query target; not the default serving backend.

This is a numerical control, not a replacement for DSpark's variable global
verification budget or a promise of stock-BF16 token identity. Query padding and
static key capacity have real costs and may obscure hardware throughput cliffs.

Both canonical sequential and speculative paths share full-prompt prefill and
its full-row LM-head projection. Later appends always compute query_width rows.
predict() projects all padded rows before selecting valid output positions.
Only real input rows enter the logical cache/returned draft context.
"""
from dataclasses import dataclass

import torch
from transformers import StaticCache
from transformers.cache_utils import StaticLayer

from .cached_target import CachedFeatures, CachedTarget


@dataclass
class CanonicalFeatures(CachedFeatures):
    padded_last: torch.Tensor
    is_prefill: bool


class CanonicalTarget(CachedTarget):
    def __init__(self, model, layer_ids=(), *, capacity, query_width=8, dummy_token_id=0):
        if type(capacity) is not int or type(query_width) is not int or not 1 <= query_width <= capacity:
            raise ValueError("Require integer capacity >= query_width >= 1")
        if type(dummy_token_id) is not int or not 0 <= dummy_token_id < model.config.vocab_size:
            raise ValueError("Dummy token must be in the target vocabulary")
        self.capacity = capacity
        self.query_width = query_width
        self.dummy_token_id = dummy_token_id
        super().__init__(model, layer_ids)

    def reset(self):
        self.cache = StaticCache(config=self.model.config, max_cache_len=self.capacity)
        if any(type(layer) is not StaticLayer for layer in self.cache.layers):
            raise ValueError("CanonicalTarget supports dense StaticLayer only")
        self._length = 0
        self._initialized = False

    @property
    def length(self):
        # Avoid reading one GPU scalar per layer in a hot path. The adapter owns
        # all writes; validate_cache() checks backing counters during tests.
        return self._length

    def validate_cache(self):
        for layer in self.cache.layers:
            if int(layer.get_seq_length()) != self._length:
                raise RuntimeError("Static cache counters do not match logical prefix")
            if layer.is_initialized and layer.keys.shape[2] != self.capacity:
                raise RuntimeError("Static cache backing capacity changed")

    def _retain(self, length):
        # HF 5.17 StaticLayer has no crop. This pinned adapter preserves storage
        # addresses and excludes stale suffix contents through the explicit mask.
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(length)
        self._length = length

    def crop(self, length):
        if not self._initialized:
            raise ValueError("Call prefill before crop")
        if type(length) is not int or not 0 <= length <= self.length:
            raise ValueError("crop length must be an absolute retained prefix length")
        if length != self.length:
            self._retain(length)

    @torch.no_grad()
    def prefill(self, input_ids):
        self._validate_ids(input_ids)
        if input_ids.shape[1] > self.capacity:
            raise ValueError("Prompt exceeds static cache capacity")
        self.reset()
        self._initialized = True
        return self._forward(input_ids, input_ids.shape[1], is_prefill=True)

    @torch.no_grad()
    def append(self, input_ids):
        self._validate_ids(input_ids)
        if not self._initialized:
            raise ValueError("Call prefill before append")
        valid = input_ids.shape[1]
        if valid > self.query_width:
            raise ValueError("Input chunk exceeds the fixed query width")
        if self.length + self.query_width > self.capacity:
            raise ValueError("Static capacity lacks full padded-query headroom")
        padded = input_ids.new_full((1, self.query_width), self.dummy_token_id)
        padded[:, :valid] = input_ids
        return self._forward(padded, valid, is_prefill=False)

    @torch.no_grad()
    def _forward(self, physical_ids, valid, *, is_prefill):
        start = self.length
        positions = torch.arange(start, start + physical_ids.shape[1], device=physical_ids.device)[None]
        keys = torch.arange(self.capacity, device=physical_ids.device)
        visible = ((keys[None, :] <= positions[0, :, None]) & (keys[None, :] < start + valid))
        mask = torch.zeros((1, 1, physical_ids.shape[1], self.capacity),
                           device=physical_ids.device, dtype=self.model.dtype)
        mask.masked_fill_(~visible[None, None], float("-inf"))
        captured, hooks = {}, []
        def hook_for(index):
            def hook(_module, _inputs, output):
                captured[index] = output[0] if isinstance(output, tuple) else output
            return hook
        try:
            for index in self.layer_ids:
                hooks.append(self.model.model.layers[index].register_forward_hook(hook_for(index)))
            result = self.model.model(input_ids=physical_ids, position_ids=positions,
                attention_mask={"full_attention": mask}, past_key_values=self.cache,
                use_cache=True, return_dict=True)
            self.cache = result.past_key_values
            # Only physically padded query rows are discarded here. Real
            # speculative suffixes remain until the caller commits/crops them.
            if valid < physical_ids.shape[1]:
                self._retain(start + valid)
            else:
                self._length = start + valid
            context = torch.cat([captured[i][:, :valid] for i in self.layer_ids], -1) if self.layer_ids else None
            padded_hidden = result.last_hidden_state.detach()
            return CanonicalFeatures(start, padded_hidden[:, :valid],
                context.detach() if context is not None else None, padded_hidden, is_prefill)
        except Exception:
            self.reset()
            raise
        finally:
            for hook in hooks:
                hook.remove()

    def logits(self, hidden):
        raise ValueError("CanonicalTarget requires predict(features) to preserve LM-head shape")

    @torch.no_grad()
    def predict(self, features, last_only=False):
        if not isinstance(features, CanonicalFeatures):
            raise ValueError("Expected CanonicalFeatures with full padded hidden states")
        if not features.is_prefill and features.padded_last.shape[1] != self.query_width:
            raise ValueError("Canonical query width changed")
        logits = self.model.get_output_embeddings()(features.padded_last)
        valid_logits = logits[:, :features.last.shape[1]]
        return valid_logits[:, -1] if last_only else valid_logits
