"""CPU Sequential Temperature Scaling for held-out rollout prefix events.

This module fits no model and selects no checkpoint/scheduler. See
docs/confidence-calibration.md for source definitions and local conventions.
"""
from dataclasses import dataclass
import math


DEFAULT_TEMPERATURE_GRID = tuple(2.0 ** (i / 10) for i in range(-30, 31))


@dataclass(frozen=True)
class RolloutBlock:
    block_id: str
    prompt_id: str
    split: str
    confidence_logits: tuple[float, ...]
    proposal_length: int
    accepted_prefix_length: int
    verified_length: int
    # Zero-based position of an accepted EOS; EOS itself remains in the sample.
    accepted_eos_position: int | None = None
    sampling_mode: str = "stochastic"
    collection_policy: str = "full_proposal"

    @property
    def effective_length(self):
        return (self.accepted_eos_position + 1 if self.accepted_eos_position is not None
                else self.verified_length)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _positive(value):
    return _number(value) and value > 0


def _sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def calibrated_probabilities(logits, temperatures):
    """Scale conditional confidence logits, then return conditional and prefix p."""
    if len(logits) > len(temperatures):
        raise ValueError("Missing per-position temperatures")
    if not all(_number(x) for x in logits) or not all(_positive(t) for t in temperatures):
        raise ValueError("Need finite logits and positive finite temperatures")
    conditional, prefix, product = [], [], 1.0
    for logit, temperature in zip(logits, temperatures):
        probability = _sigmoid(logit / temperature)
        conditional.append(probability)
        product *= probability
        prefix.append(product)
    return conditional, prefix


def reliability_metrics(probabilities, labels, *, num_bins=20):
    """Equal-width bin ECE/Brier, with DeepSpec's endpoint clamp and bin rule."""
    if type(num_bins) is not int or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")
    if len(probabilities) != len(labels):
        raise ValueError("Prediction/label lengths differ")
    if any(not _number(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid probability")
    if any(type(y) not in (int, bool) or y not in (0, 1) for y in labels):
        raise ValueError("Labels must be binary realized prefix events")
    n = len(labels)
    if not n:
        return dict(count=0, ece=None, brier=None, pred_mean=None, target_mean=None, bins=[])
    bins = [[] for _ in range(num_bins)]
    clamped = [min(1 - 1e-8, max(1e-8, p)) for p in probabilities]
    for p, y in zip(clamped, labels):
        bins[min(num_bins - 1, int(p * num_bins))].append((p, int(y)))
    entries = []
    for index, samples in enumerate(bins):
        if samples:
            entries.append(dict(bin=index, count=len(samples),
                pred_mean=math.fsum(p for p, _ in samples)/len(samples),
                target_mean=math.fsum(y for _, y in samples)/len(samples)))
    return dict(count=n,
        ece=math.fsum(e['count'] * abs(e['pred_mean']-e['target_mean']) for e in entries)/n,
        brier=math.fsum((p-y)**2 for p, y in zip(clamped, labels))/n,
        pred_mean=math.fsum(clamped)/n, target_mean=math.fsum(labels)/n, bins=entries)


def _validate(blocks, block_size, sampling_mode):
    if type(block_size) is not int or block_size < 1:
        raise ValueError("Positive block_size required")
    if sampling_mode not in ('stochastic', 'greedy'):
        raise ValueError("Unknown acceptance regime")
    seen = set()
    for block in blocks:
        if block.split != 'validation':
            raise ValueError("Calibration only accepts validation; train/final test are forbidden")
        if not block.block_id or not block.prompt_id or block.block_id in seen:
            raise ValueError("Missing identity or duplicate rollout block")
        seen.add(block.block_id)
        if block.sampling_mode != sampling_mode or block.collection_policy != 'full_proposal':
            raise ValueError("Mixed acceptance regime or confidence-selected/censored collection")
        if not 0 <= len(block.confidence_logits) <= block_size or not all(_number(x) for x in block.confidence_logits):
            raise ValueError("Invalid confidence logits")
        if type(block.proposal_length) is not int or not 0 <= block.proposal_length <= len(block.confidence_logits):
            raise ValueError("Invalid proposal length")
        if type(block.verified_length) is not int or not 0 <= block.verified_length <= block.proposal_length:
            raise ValueError("Invalid verified length")
        if type(block.accepted_prefix_length) is not int or not 0 <= block.accepted_prefix_length <= block.verified_length:
            raise ValueError("Invalid accepted prefix length")
        eos = block.accepted_eos_position
        if eos is not None and (type(eos) is not int or not 0 <= eos < block.accepted_prefix_length):
            raise ValueError("EOS is not inside the accepted prefix")
    if not any(block.effective_length for block in blocks):
        raise ValueError("No observable validation proposal positions")


def _position_metrics(blocks, temperatures, position, num_bins):
    predictions, labels = [], []
    for block in blocks:
        if block.effective_length <= position:
            continue
        _, prefix = calibrated_probabilities(block.confidence_logits[:position+1], temperatures)
        predictions.append(prefix[-1])
        labels.append(int(position < block.accepted_prefix_length))
    return reliability_metrics(predictions, labels, num_bins=num_bins)


def fit_sts(blocks, *, block_size, identity, temperature_grid=DEFAULT_TEMPERATURE_GRID,
            num_bins=20, sampling_mode='stochastic'):
    """Freeze each earlier T while minimizing the current prefix ECE on a grid.

    The explicit validation split and collection regime are boundary checks,
    not proof of provenance. Caller must bind records/checkpoint/protocol hashes.
    Returned fit metrics use the fitting population and are not held-out claims.
    """
    required = ('checkpoint_sha256', 'development_records_sha256', 'rollout_protocol_sha256')
    if not isinstance(identity, dict) or any(
        not isinstance(identity.get(key), str) or len(identity[key]) != 64 or
        any(c not in '0123456789abcdef' for c in identity[key]) for key in required):
        raise ValueError("Frozen checkpoint/data/rollout-protocol SHA256 identity required")
    if not isinstance(identity.get('probability_policy'), str) or not identity['probability_policy'].strip():
        raise ValueError("Explicit target/proposal probability and sampling policy required")
    identity = {key: identity[key] for key in (*required, 'probability_policy')}
    blocks, grid = tuple(blocks), tuple(temperature_grid)
    _validate(blocks, block_size, sampling_mode)
    if not grid or any(not _positive(t) for t in grid) or 1.0 not in grid:
        raise ValueError("Temperature grid must be finite, strictly positive and include baseline T=1")
    grid = tuple(sorted(set(grid)))
    # Validate even if an individual tail position has no observations.
    reliability_metrics([], [], num_bins=num_bins)
    temperatures = [1.0] * block_size
    baseline, fitted, searches = [], [], []
    for position in range(block_size):
        baseline.append(_position_metrics(blocks, [1.0]*block_size, position, num_bins))
        if baseline[-1]['count'] == 0:
            fitted.append(dict(position=position, temperature=1.0, fitted=False, **baseline[-1]))
            searches.append([])
            continue
        trials = []
        for temperature in grid:
            candidate = temperatures.copy()
            candidate[position] = temperature
            metrics = _position_metrics(blocks, candidate, position, num_bins)
            trials.append(dict(temperature=temperature, ece=metrics['ece']))
        # Local deterministic tie convention: closest to unscaled in log space,
        # then smaller T. No floating tolerance changes the objective ordering.
        best = min(trials, key=lambda trial: (trial['ece'], abs(math.log(trial['temperature'])), trial['temperature']))
        temperatures[position] = best['temperature']
        fitted.append(dict(position=position, temperature=best['temperature'], fitted=True,
            **_position_metrics(blocks, temperatures, position, num_bins)))
        searches.append(trials)
    return dict(method='sequential_temperature_scaling', identity=identity, temperatures=temperatures,
        temperature_grid=list(grid), num_bins=num_bins, sampling_mode=sampling_mode,
        split='validation', collection_policy='full_proposal', blocks=len(blocks),
        unique_prompts=len({b.prompt_id for b in blocks}), per_position=fitted,
        uncalibrated_per_position=baseline, grid_search=searches,
        labels='realized accepted-prefix event, excluding positions after accepted EOS',
        scope='Fitting-population calibration only; no independent evaluation or decoding integration')
