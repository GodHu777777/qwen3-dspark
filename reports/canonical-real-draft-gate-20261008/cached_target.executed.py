"""Single-sequence dense Qwen3 target KV cache, independent of draft policy.

The cache contains exactly the processed tokens. Returned hidden/context tensors
cover only the new chunk; logit at chunk position i predicts its following token.
crop() discards an uncommitted suffix. The caller owns any corresponding draft
feature cache and must crop that separately. No padding or sliding-window cache.
"""
from dataclasses import dataclass

import torch
from transformers import DynamicCache


@dataclass
class CachedFeatures:
    start: int
    last: torch.Tensor
    context: torch.Tensor | None

    @property
    def end(self):
        return self.start + self.last.shape[1]


class CachedTarget:
    def __init__(self, model, layer_ids=()):
        c = model.config
        if c.model_type != "qwen3" or getattr(c, "use_sliding_window", False):
            raise ValueError("Only dense Qwen3 without sliding-window attention is supported")
        self.layer_ids = tuple(layer_ids)
        if (tuple(sorted(set(self.layer_ids))) != self.layer_ids or
                any(i < 0 or i >= c.num_hidden_layers for i in self.layer_ids)):
            raise ValueError("layer_ids must be unique, increasing decoder block indices")
        self.model = model.eval().requires_grad_(False)
        self.reset()

    def reset(self):
        self.cache = DynamicCache(config=self.model.config)
        self._length = 0
        self._initialized = False

    @property
    def length(self):
        actual = self.cache.get_seq_length()
        if actual != self._length:
            raise RuntimeError(f"Cache length changed externally: {actual} != {self._length}")
        return self._length

    def _validate_ids(self, input_ids):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("Expected one nonempty, unpadded token sequence")
        if input_ids.dtype != torch.long:
            raise ValueError("Token IDs must have torch.long dtype")

    @torch.no_grad()
    def prefill(self, input_ids):
        self._validate_ids(input_ids)
        self.reset()
        self._initialized = True
        return self.append(input_ids)

    @torch.no_grad()
    def append(self, input_ids):
        """Process an anchor, a verification block, or the next decode token."""
        self._validate_ids(input_ids)
        if not self._initialized:
            raise ValueError("Call prefill before append")
        start = self.length
        end = start + input_ids.shape[1]
        positions = torch.arange(start, end, device=input_ids.device)[None]
        captured, hooks = {}, []
        def hook_for(index):
            def hook(_module, _inputs, output):
                captured[index] = output[0] if isinstance(output, tuple) else output
            return hook
        try:
            for index in self.layer_ids:
                hooks.append(self.model.model.layers[index].register_forward_hook(hook_for(index)))
            output = self.model.model(input_ids=input_ids, position_ids=positions,
                attention_mask=torch.ones((1, end), device=input_ids.device, dtype=torch.long),
                past_key_values=self.cache, use_cache=True, return_dict=True)
            self.cache = output.past_key_values
            self._length = end
            if self.length != end:
                raise RuntimeError("Unexpected target cache length")
            context = torch.cat([captured[i] for i in self.layer_ids], -1) if self.layer_ids else None
            return CachedFeatures(start, output.last_hidden_state.detach(),
                                  context.detach() if context is not None else None)
        except Exception:
            # A failed forward can leave only some layers extended. Never reuse it.
            self.reset()
            raise
        finally:
            for hook in hooks:
                hook.remove()

    def crop(self, length):
        """Keep an absolute prefix length (including zero), never extend a cache."""
        if not self._initialized:
            raise ValueError("Call prefill before crop")
        if isinstance(length, bool) or not isinstance(length, int) or not 0 <= length <= self.length:
            raise ValueError("crop length must be an integer in [0, cached length]")
        # Negative crop means remove that many tokens in both older HF caches
        # and Transformers 5.17+. In 5.17, crop(0) is a no-op, not an empty cache.
        if length != self.length:
            self.cache.crop(length - self.length)
        self._length = length
        if self.length != length:
            raise RuntimeError("Cache crop did not preserve the requested prefix")

    @torch.no_grad()
    def logits(self, hidden):
        return self.model.get_output_embeddings()(hidden)

    @torch.no_grad()
    def predict(self, features, last_only=False):
        """Ordinary dynamic-cache policy: project only requested hidden rows."""
        hidden = features.last[:, -1] if last_only else features.last
        return self.logits(hidden)

    @torch.no_grad()
    def greedy(self, input_ids, max_new_tokens, eos_ids=()):
        """Return exact greedy output; the final emitted token remains unprocessed."""
        self._validate_ids(input_ids)
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        self.reset()
        if max_new_tokens == 0:
            return []
        eos_ids = set(eos_ids)
        features = self.prefill(input_ids)
        tokens = []
        for index in range(max_new_tokens):
            token = self.predict(features, last_only=True).argmax(-1)
            value = int(token.item())
            tokens.append(value)
            if value in eos_ids or index + 1 == max_new_tokens:
                break
            features = self.append(token[:, None])
        return tokens
