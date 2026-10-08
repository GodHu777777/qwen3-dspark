"""Cached stochastic integration: exact toy laws and actual tiny Qwen CPU KV."""
from collections import defaultdict
from fractions import Fraction as F
import itertools
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dspark_qwen.cached_decode import DraftContextCache
from dspark_qwen.cached_sampling import cached_speculative_sample, cached_target_sample
from dspark_qwen.cached_target import CachedFeatures, CachedTarget
from dspark_qwen.canonical_target import CanonicalTarget
from dspark_qwen.config import DraftConfig
from dspark_qwen.model import DSparkDraft
from dspark_qwen.tensor_sampling import (
    TensorProposal, TensorRandom, logits_to_probabilities, sample_categorical,
)
from test_sampling import categorical_branches, exact_target_sequence, proposal_paths, verification_paths
from test_tensor_sampling import TensorTape, row


def sequence_tapes(p, q, gamma, budget, stops, prefix=(), mass=F(1), tape=()):
    """Enumerate full cached-decoder RNG streams including target first token."""
    if len(prefix) == budget or (prefix and prefix[-1] in stops):
        yield prefix, mass, tape
        return
    if not prefix:
        for token, probability, midpoint in categorical_branches(p(())):
            yield from sequence_tapes(p, q, gamma, budget, stops, (token,),
                                      mass * probability, tape + (midpoint,))
        return
    remaining = budget - len(prefix)
    for tokens, rows, proposal_mass, proposal_tape in proposal_paths(
            lambda draft: q(prefix + draft), min(gamma, remaining), lambda _: True):
        target = tuple(p(prefix + tokens[:j]) for j in range(len(tokens) + 1))
        for output, probability, verify_tape in verification_paths(tokens, rows, target, remaining, stops):
            yield from sequence_tapes(p, q, gamma, budget, stops, prefix + output.tokens,
                mass * proposal_mass * probability, tape + proposal_tape + verify_tape)


class ToyTarget:
    """Prefix-sensitive law; stored context features expose cache token identity."""
    def __init__(self, p):
        self.p = p
        self.layer_ids = (0,)
        self.model = SimpleNamespace(config=SimpleNamespace(vocab_size=2))
        self.reset()

    def reset(self): self.history = []
    @property
    def length(self): return len(self.history)
    def _validate_ids(self, ids):
        assert ids.ndim == 2 and ids.shape[0] == 1 and ids.shape[1] > 0
    def prefill(self, ids):
        self.reset()
        return self.append(ids)
    def append(self, ids):
        start = self.length
        self.history.extend(ids[0].tolist())
        features = ids[:, :, None].double()
        return CachedFeatures(start, features, features)
    def predict(self, features, last_only=False):
        logits = torch.stack([row(self.p(tuple(self.history[1:features.start + j + 1]))).log()
                              for j in range(features.last.shape[1])])[None]
        return logits[:, -1] if last_only else logits
    def crop(self, length): self.history = self.history[:length]


class ToyDraft:
    def __init__(self, q, gamma):
        self.q = q
        self.spec = SimpleNamespace(layer_ids=(0,), block_size=max(1, gamma))
    def eval(self): return self
    def project_context_kv(self, features, start):
        return [(features[:, None].clone(), features[:, None].clone())]
    def propose_stochastic_cached(self, anchor, kv, length, *, temperature, rng, max_draft_tokens):
        context = tuple(int(x) for x in kv[0][0][0, 0, :, 0].tolist())
        prefix = context[1:] + (int(anchor.item()),)
        tokens, rows = [], []
        for _ in range(max_draft_tokens):
            q = row(self.q(prefix + tuple(tokens)))
            token = int(sample_categorical(q, rng))
            tokens.append(token)
            rows.append(q)
        return TensorProposal(torch.tensor(tokens, dtype=torch.long),
            torch.stack(rows) if rows else torch.empty((0, 2), dtype=torch.float64), torch.zeros(len(tokens)))


class CachedToySamplingTests(unittest.TestCase):
    def test_exact_full_cached_output_law_and_cache_prefix(self):
        p = lambda prefix: (F(1, 4), F(3, 4)) if sum(prefix) % 2 else (F(3, 4), F(1, 4))
        q = lambda prefix: (F(1, 2), F(1, 2)) if len(prefix) % 2 else (F(1, 4), F(3, 4))
        prompt = torch.tensor([[0]])
        for gamma, budget, stops in itertools.product((0, 1, 2), (0, 1, 3), ((), (1,))):
            law = defaultdict(F)
            for expected, probability, tape in sequence_tapes(p, q, gamma, budget, stops):
                target, rng = ToyTarget(p), TensorTape(tape)
                actual, records = cached_speculative_sample(target, ToyDraft(q, gamma), prompt,
                    budget, rng=rng, eos_ids=stops, max_draft_tokens=gamma)
                rng.exhausted()
                self.assertEqual(tuple(actual), expected)
                self.assertEqual(target.history, [0] + actual[:-1] if actual else [])
                for record in records:
                    self.assertEqual(record['verified_proposal_length'], record['proposed'])
                    self.assertEqual(record['verified_cache_end'],
                                     record['cache_before'] + record['proposed'] + 1)
                law[tuple(actual)] += probability
            self.assertEqual(dict(law), exact_target_sequence(p, budget, stops))
            self.assertEqual(sum(law.values()), 1)


class BoundaryRandom(TensorRandom):
    """Force a chosen rejection index for one-hot q on positive tiny-model p."""
    def __init__(self, reject_at):
        self.reject_at, self.attempt, self.draws = reject_at, 0, 0
    def uniform(self, reference):
        if reference.ndim == 2:  # Acceptance uniform: verifier's p matrix.
            value = math.nextafter(1, 0) if self.attempt == self.reject_at else 0.0
            self.attempt += 1
        else:  # First token, proposal, correction or bonus categorical.
            value = 0.0 if self.draws == 0 else 0.5
            self.attempt = 0
        self.draws += 1
        return reference.new_tensor(value)


class CachedQwenSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(127)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, attention_dropout=0.0)
        config._attn_implementation = 'sdpa'
        self.model = Qwen3ForCausalLM(config).eval()
        self.spec = DraftConfig(layer_ids=(0, 2), num_layers=2, block_size=3,
                                markov_rank=8, mask_token_id=31)
        self.draft = DSparkDraft(self.model, self.spec).eval()
        self.target = CachedTarget(self.model, self.spec.layer_ids)
        self.prompt = torch.tensor([[1, 2, 3, 4]])

    def controlled(self):
        base = self.draft
        class Controlled:
            spec = base.spec
            def eval(self): return self
            def project_context_kv(self, features, start): return base.project_context_kv(features, start)
            def propose_stochastic_cached(self, anchor, kv, length, *, temperature, rng, max_draft_tokens):
                tokens = torch.arange(10, 10 + max_draft_tokens)
                q = torch.zeros((max_draft_tokens, 32), dtype=torch.float64)
                if max_draft_tokens:
                    q.scatter_(1, tokens[:, None], 1)
                for r, token in zip(q, tokens):
                    assert int(sample_categorical(r, rng)) == int(token)
                return TensorProposal(tokens, q, torch.arange(max_draft_tokens).float())
        return Controlled()

    def assert_cache_content(self, output, draft_cache):
        if not output:
            self.assertEqual(self.target.length, 0)
            return
        prefix = torch.cat((self.prompt, torch.tensor([output[:-1]], dtype=torch.long)), dim=1)
        fresh = CachedTarget(self.model, self.spec.layer_ids)
        features = fresh.prefill(prefix)
        self.assertEqual(self.target.length, prefix.shape[1])
        for actual, expected in zip(self.target.cache.layers, fresh.cache.layers):
            torch.testing.assert_close(actual.keys[:, :, :prefix.shape[1]], expected.keys, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(actual.values[:, :, :prefix.shape[1]], expected.values, atol=1e-6, rtol=1e-5)
        if draft_cache.layers:
            self.assertEqual(draft_cache.length, prefix.shape[1])
            expected = self.draft.project_context_kv(features.context, 0)
            for (k, v), (ek, ev) in zip(draft_cache.layers, expected):
                torch.testing.assert_close(k, ek, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(v, ev, atol=1e-6, rtol=1e-5)

    def decode_tracked(self, draft, count, rng, **kwargs):
        caches = []
        class Tracked(DraftContextCache):
            def __init__(self, draft):
                super().__init__(draft)
                caches.append(self)
        with patch('dspark_qwen.cached_sampling.DraftContextCache', Tracked):
            output, records = cached_speculative_sample(self.target, draft, self.prompt, count, rng=rng, **kwargs)
        if caches:
            self.assert_cache_content(output, caches[-1])
        else:
            self.assertFalse(output)
        return output, records

    def test_actual_proposal_one_backbone_real_q_and_pre_token_confidence(self):
        features = self.target.prefill(self.prompt)
        cache = DraftContextCache(self.draft)
        cache.append(features.context)
        anchor = torch.tensor([[5]])
        hidden = self.draft.backbone_cached(anchor, cache.layers, self.target.length)
        base = self.draft.lm_head(hidden)
        with patch.object(self.draft, 'backbone_cached', wraps=self.draft.backbone_cached) as backbone:
            proposal = self.draft.propose_stochastic_cached(anchor, cache.layers, self.target.length,
                temperature=0.7, rng=TensorRandom(torch.Generator().manual_seed(8)))
            self.assertEqual(backbone.call_count, 1)
        prev = anchor[0]
        for j, token in enumerate(proposal.tokens):
            emb = self.draft.markov_embedding(prev)
            expected_q = logits_to_probabilities(base[j:j+1] + self.draft.markov_projection(emb), 0.7)[0]
            expected_confidence = self.draft.confidence(torch.cat((hidden[j:j+1], emb), -1)).flatten()[0]
            torch.testing.assert_close(proposal.draft_probs[j], expected_q, atol=0, rtol=0)
            torch.testing.assert_close(proposal.confidence_logits[j], expected_confidence, atol=0, rtol=0)
            self.assertGreater(float(proposal.draft_probs[j, token]), 0)
            prev = token.reshape(1)
        with patch.object(self.draft, 'backbone_cached', wraps=self.draft.backbone_cached) as backbone:
            empty = self.draft.propose_stochastic_cached(anchor, cache.layers, self.target.length,
                temperature=1, rng=TensorTape(()), max_draft_tokens=0)
            self.assertEqual(backbone.call_count, 0)
            self.assertEqual(empty.draft_probs.shape, (0, 32))

    def test_partial_rejection_all_accept_and_budget_cache_contents(self):
        for reject_at, count in itertools.product((0, 1, 2, None), (2, 3, 4, 5, 9)):
            with self.subTest(rejection=reject_at, count=count):
                output, records = self.decode_tracked(self.controlled(), count, BoundaryRandom(reject_at))
                self.assertEqual(len(output), count)
                if reject_at is not None and reject_at < min(3, count - 1):
                    self.assertEqual(records[0]['accepted'], reject_at)
                    self.assertEqual(records[0]['extra_token_kind'], 'residual')
                else:
                    self.assertEqual(records[0]['accepted'], min(3, count - 1))
                self.assertEqual(records[-1]['termination'], 'budget')
                for record in records:
                    self.assertEqual(record['verified_proposal_length'], record['proposed'])
                    self.assertIsNone(record['accepted_eos_position'])
                    self.assertNotIn('target_probs', record)

    def test_accepted_residual_bonus_eos_and_initial_eos(self):
        for reject_at in (0, 1, 2, None):
            reference, records = self.decode_tracked(self.controlled(), 6, BoundaryRandom(reject_at), trace_tokens=True)
            first = records[0]
            extra = first['committed_tokens'][-1]
            for eos in ({reference[0]}, {10}, {11}, {extra}):
                output, stopped = self.decode_tracked(self.controlled(), 6, BoundaryRandom(reject_at), eos_ids=eos)
                expected_stop = next((i + 1 for i, token in enumerate(reference) if token in eos), len(reference))
                self.assertEqual(output, reference[:expected_stop])
                if stopped and output[-1] in eos:
                    last = stopped[-1]
                    self.assertEqual(last['termination'], 'eos')
                    if last['extra_token_kind'] is None:
                        self.assertEqual(last['accepted_eos_position'], last['accepted'] - 1)
                    else:
                        self.assertIsNone(last['accepted_eos_position'])

    def test_real_model_seeded_state_isolation_observer_and_canonical_control(self):
        observations = []
        def run(seed):
            return self.decode_tracked(self.draft, 10, TensorRandom(torch.Generator().manual_seed(seed)),
                temperature=0.8, observer=observations.append, trace_probabilities=True)
        global_rng = torch.random.get_rng_state().clone()
        first, records = run(99)
        run(3)  # A different request must not pollute either cache or RNG stream.
        second, _ = run(99)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
        self.assertTrue(observations)
        for record in records:
            self.assertEqual(record['target_probs'].device.type, 'cpu')
            self.assertEqual(record['target_probs'].shape[0], record['proposed'] + 1)
            self.assertEqual(record['draft_probs'].dtype, torch.float64)
        for observation in observations:
            self.assertEqual(observation.verified_proposal_length, observation.proposal.tokens.numel())
            self.assertEqual(observation.temperature, 0.8)
            self.assertEqual(observation.probability_policy, 'float64_softmax_normalize_cdf_v1')
        for _ in range(2):
            self.decode_tracked(self.draft, 6, TensorRandom(torch.Generator().manual_seed(77)), max_draft_tokens=0)
        canonical = CanonicalTarget(self.model, self.spec.layer_ids, capacity=64, query_width=4)
        a, rounds = cached_speculative_sample(canonical, self.draft, self.prompt, 7,
            rng=TensorRandom(torch.Generator().manual_seed(7)))
        b, _ = cached_speculative_sample(canonical, self.draft, self.prompt, 7,
            rng=TensorRandom(torch.Generator().manual_seed(7)))
        self.assertEqual(a, b)
        canonical.validate_cache()
        self.assertTrue(all(r['physical_verification_rows'] == 4 for r in rounds))

    def test_target_baseline_same_adapter_and_failure_resets_state(self):
        generator = lambda: TensorRandom(torch.Generator().manual_seed(5))
        baseline = CachedTarget(self.model, ())  # Fair baseline has no context hooks.
        first = cached_target_sample(baseline, self.prompt, 7, temperature=0.6, rng=generator())
        second = cached_target_sample(baseline, self.prompt, 7, temperature=0.6, rng=generator())
        self.assertEqual(first, second)
        self.assertEqual(baseline.length, self.prompt.shape[1] + len(first) - 1)
        # n=0 consumes exactly the target-only RNG sequence on dynamic target.
        actual, _ = cached_speculative_sample(self.target, self.draft, self.prompt, 7,
            temperature=0.6, rng=generator(), max_draft_tokens=0)
        self.assertEqual(actual, first)
        def fail(_): raise RuntimeError('observer failure')
        with self.assertRaisesRegex(RuntimeError, 'observer failure'):
            cached_speculative_sample(self.target, self.draft, self.prompt, 7, rng=generator(), observer=fail)
        self.assertEqual(self.target.length, 0)
        again = cached_target_sample(self.target, self.prompt, 7, temperature=0.6, rng=generator())
        self.assertEqual(again, first)
        self.assertEqual(cached_speculative_sample(self.target, self.draft, self.prompt, 0), ([], []))
        self.assertEqual(self.target.length, 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
