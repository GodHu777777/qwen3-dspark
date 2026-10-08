"""Frozen target: selected block outputs feed draft; final norm feeds supervision."""
from dataclasses import dataclass
import torch


@dataclass
class TargetFeatures:
    context: torch.Tensor  # [1, S, number_of_layers * hidden]
    last: torch.Tensor  # [1, S, hidden], after target final norm


class FrozenTarget:
    def __init__(self, model, spec):
        spec.validate(model.config)
        self.model = model.eval().requires_grad_(False)
        self.spec = spec

    @torch.no_grad()
    def capture(self, input_ids):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Reference supports one unpadded sequence per micro-step")
        captured, hooks = {}, []
        def hook_for(index):
            def hook(_module, _inputs, output):
                captured[index] = output[0] if isinstance(output, tuple) else output
            return hook
        try:
            for i in self.spec.layer_ids:
                hooks.append(self.model.model.layers[i].register_forward_hook(hook_for(i)))
            # Run base transformer, avoiding an unnecessary [S, vocab] logits allocation.
            result = self.model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        finally:
            for hook in hooks:
                hook.remove()
        return TargetFeatures(torch.cat([captured[i] for i in self.spec.layer_ids], -1).detach(),
                              result.last_hidden_state.detach())

    @torch.no_grad()
    def logits(self, last_hidden):
        return self.model.get_output_embeddings()(last_hidden)
