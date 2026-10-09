"""Opt-in categorical fallback experiment; production sampling remains unchanged.

The original function remains the independent oracle. Only a nonempty 1-D law
AFTER the RNG callback uses the fixed-shape index reduction. Unusual callback
metadata changes and missing positive support retain the original expression,
including its errors. No support mask, uniform, CDF or validation is cached.
"""
import torch

from .tensor_sampling import check_probabilities, _uniform
from .tensor_sampling import sample_categorical as ORIGINAL_SAMPLE_CATEGORICAL


def sample_categorical_fixed_shape(probs, rng):
    """Same law/callback order as the original; fixed-shape normal-path fallback."""
    check_probabilities(probs)
    if probs.ndim != 1:
        raise ValueError("Categorical sampling expects one row")
    index = torch.searchsorted(probs.cumsum(-1), _uniform(rng, probs), right=True)
    if probs.ndim == 1 and probs.numel():
        positions = torch.arange(probs.numel(), dtype=torch.long, device=probs.device)
        last_positive = torch.where(probs > 0, positions, -1).amax()
        if bool(last_positive < 0):
            last_positive = torch.nonzero(probs > 0, as_tuple=False)[-1, 0]
    else:
        last_positive = torch.nonzero(probs > 0, as_tuple=False)[-1, 0]
    return torch.where(index < probs.numel(), index, last_positive)
