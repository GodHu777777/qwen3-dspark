"""Batched target-only control with the existing FP64 categorical sampling law.

This session owns no draft model/cache and runs no proposal/shadow work. Explicit
target strategies provide the same prefill/verify/commit/release lifecycle as the
speculative session. CPU correctness is not a native performance result.
"""
import torch

from .packed_sampling import RequestSpec
from .tensor_sampling import (PROBABILITY_POLICY, logits_to_probabilities,
    sample_categorical, validate_stops, validate_temperature)


class PackedTargetOnlySession:
    """One target query and one original categorical draw per active request.

    The committed cache always excludes the newest emitted token, including for
    finished requests. An execution failure permanently invalidates this owner;
    consumed RNG or device writes are not rolled back.
    """
    def __init__(self, target, *, target_strategy, temperature=1., eos_ids=()):
        if target.lengths:
            raise ValueError('Session requires an initially empty target pool')
        if getattr(target_strategy, 'target', None) is not target:
            raise ValueError('Explicit target strategy must own the session target')
        if target.device.type == 'cuda':
            from .rocm_varlen import BACKEND
            if getattr(getattr(target, '_varlen_kernel', None), 'backend_name', None) != BACKEND:
                raise ValueError('GPU target-only session requires explicit pinned target backend')
        self.target, self.target_strategy = target, target_strategy
        self.temperature = validate_temperature(temperature)
        self.eos_ids = validate_stops(eos_ids, target.model.config.vocab_size)
        self.requests = {}; self.failed = False; self._incarnation = 0
        self.last_prefill_work = None

    def _healthy(self):
        if self.failed:
            raise RuntimeError('Session invalidated by prior execution failure; create a new session')

    def _invalidate(self):
        self.failed = True
        try:self.target_strategy.invalidate()
        except Exception as error:self.cleanup_error = repr(error)

    def _check_lengths(self):
        expected = {r:s['prompt_length']+len(s['output'])-1 for r,s in self.requests.items()}
        if self.target.lengths != expected:
            raise RuntimeError('Committed target prefixes must exclude the newest emitted anchor')

    @torch.no_grad()
    def admit(self, requests):
        self._healthy()
        if not requests:raise ValueError('Need new requests')
        for request, spec in requests.items():
            if (not isinstance(request, str) or not request or request in self.requests or
                    not isinstance(spec, RequestSpec)):
                raise ValueError('Fresh request names/specs required')
            ids = spec.input_ids
            if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or
                    ids.dtype != torch.long or ids.device != self.target.device or
                    not bool(((ids >= 0) & (ids < self.target.model.config.vocab_size)).all()) or
                    type(spec.max_new_tokens) is not int or spec.max_new_tokens < 1 or
                    not callable(getattr(spec.rng, 'uniform', None))):
                raise ValueError('Valid prompt, positive budget and per-request RNG required')
        self._check_lengths()
        self.target_strategy.preflight('prefill', tuple(s.input_ids.shape[1] for s in requests.values()),
                                       (0,)*len(requests))
        try:
            for request in requests:self.target.add_request(request)
            features = self.target_strategy.prefill({r:s.input_ids for r,s in requests.items()})
            # Exactly R final rows reach the head; no whole-prompt projection.
            final_hidden = torch.cat([features.for_request(r).last[:,-1:] for r in requests], dim=1)
            logits = self.target.model.get_output_embeddings()(final_hidden)[0]
            additions = {}
            for row,(request,spec) in enumerate(requests.items()):
                law = logits_to_probabilities(logits[row], self.temperature)
                token = int(sample_categorical(law, spec.rng).item())
                additions[request] = dict(prompt_length=spec.input_ids.shape[1], budget=spec.max_new_tokens,
                    rng=spec.rng, output=[token], finished=token in self.eos_ids or spec.max_new_tokens == 1,
                    incarnation=self._incarnation)
                self._incarnation += 1
            self.target_strategy.release_features(features)
            self.requests.update(additions); self._check_lengths()
            self.last_prefill_work = dict(target=features.work, prefill_lm_head_calls=1,
                prefill_lm_head_rows=len(requests), categorical_draws=len(requests),
                draft_forward_calls=0, projection_scope='One final hidden row per admitted request')
            return {r:list(self.requests[r]['output']) for r in requests}
        except Exception:
            self._invalidate(); raise

    @torch.no_grad()
    def step(self, active):
        """Batch one anchor per unique unfinished request; order defines draw order."""
        self._healthy(); active = tuple(active)
        if (not active or len(set(active)) != len(active) or
                any(r not in self.requests or self.requests[r]['finished'] for r in active)):
            raise ValueError('Unique unfinished active requests required')
        self._check_lengths(); before = self.target.lengths
        self.target_strategy.preflight('verification', (1,)*len(active), tuple(before[r] for r in active))
        try:
            anchors = {r:torch.tensor([[self.requests[r]['output'][-1]]], dtype=torch.long,
                                      device=self.target.device) for r in active}
            features = self.target_strategy.verify(anchors)
            logits = self.target.predict(features)
            probabilities = {}; tokens = {}; record = {}
            for request in active:
                state = self.requests[request]
                probabilities[request] = logits_to_probabilities(logits[request][0,0], self.temperature)
                token = int(sample_categorical(probabilities[request], state['rng']).item())
                tokens[request] = token
                reason = ('eos' if token in self.eos_ids else
                          'budget' if len(state['output'])+1 == state['budget'] else 'round_complete')
                record[request] = dict(token=token, committed=1, stop_reason=reason,
                    cache_before=before[request], cache_after=before[request]+1)
            # All request decisions precede the single target commit. The old
            # anchor is committed; the just-sampled token is excluded from KV.
            self.target_strategy.commit(features, {r:1 for r in active})
            self.target_strategy.release_features(features)
            for request in active:
                self.requests[request]['output'].append(tokens[request])
                self.requests[request]['finished'] = record[request]['stop_reason'] in ('eos','budget')
            self._check_lengths()
            return dict(requests=record, target_probs=probabilities, probability_policy=PROBABILITY_POLICY,
                work=dict(target=features.work, target_lm_head_calls=1, target_lm_head_rows=len(active),
                          categorical_draws=len(active), draft_forward_calls=0, shadow_proposal_positions=0))
        except Exception:
            self._invalidate(); raise

    def remove(self, request):
        self._healthy()
        if request not in self.requests:raise KeyError(request)
        try:
            self.target.remove_request(request); del self.requests[request]; self._check_lengths()
        except Exception:
            self._invalidate(); raise

    def outputs(self):
        self._healthy()
        return {r:list(state['output']) for r,state in self.requests.items()}
