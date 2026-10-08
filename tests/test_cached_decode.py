import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dspark_qwen.cached_decode import DraftContextCache, cached_speculative_greedy
from dspark_qwen.cached_target import CachedTarget
from dspark_qwen.config import DraftConfig
from dspark_qwen.eval_cached_decode import diagnose_same_prefix
from dspark_qwen.model import DSparkDraft
from dspark_qwen.target import FrozenTarget


class CachedDecodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=256, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        self.model = Qwen3ForCausalLM(config).eval()
        self.spec = DraftConfig(layer_ids=(0, 2), num_layers=2, block_size=3,
            markov_rank=8, mask_token_id=63)
        self.full = FrozenTarget(self.model, self.spec)
        self.target = CachedTarget(self.model, self.spec.layer_ids)
        self.draft = DSparkDraft(self.model, self.spec).eval()
        self.prompt = torch.tensor([[1, 2, 3, 4]])

    def controlled_draft(self, expected, reject_at=None):
        base, prompt_length = self.draft, self.prompt.shape[1]
        class ControlledDraft:
            spec = base.spec
            def eval(self): return self
            def project_context_kv(self, features, start):
                return base.project_context_kv(features, start)
            def propose_greedy_cached(self, _anchor, _kv, length):
                offset = length - prompt_length + 1
                proposal = expected[offset:offset + self.spec.block_size]
                if reject_at is not None:
                    proposal[reject_at] = (proposal[reject_at] + 1) % 64
                return proposal, [0.5] * len(proposal)
        return ControlledDraft()

    def assert_invariants(self, tokens, rounds):
        length = self.prompt.shape[1]
        if not tokens:
            self.assertEqual(self.target.length, 0)
            return
        emitted = 1
        for r in rounds:
            self.assertEqual(r["cache_before"], length + emitted - 1)
            self.assertEqual(r["verified_length"], r["cache_before"] + r["proposed"] + 1)
            emitted += r["committed"]
            self.assertEqual(r["cache_after"], length + emitted - 1)
            self.assertEqual(r["draft_cache_after"], r["cache_after"])
        self.assertEqual(emitted, len(tokens))
        self.assertEqual(self.target.length, length + len(tokens) - 1)
        # Cache contains the committed prefix: continuing its unprocessed last
        # anchor must agree with a fresh full target forward on that prefix.
        ids = torch.cat((self.prompt, self.prompt.new_tensor([tokens])), dim=1)
        next_chunk = self.target.append(self.prompt.new_tensor([[tokens[-1]]]))
        expected = self.full.capture(ids).last[:, -1:]
        torch.testing.assert_close(next_chunk.last, expected, atol=1e-6, rtol=1e-5)

    def test_projected_draft_cache_matches_full_backbone_after_append_and_crop(self):
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        context = self.full.capture(ids[:, :-1]).context
        cache = DraftContextCache(self.draft)
        cache.append(context[:, :3]); cache.append(context[:, 3:])
        for length in (7, 4, 0):
            cache.crop(length)
            if length < 7:
                cache.append(context[:, length:])
            actual = self.draft.backbone_cached(ids[:, -1:], cache.layers, 7)
            expected = self.draft.backbone(ids, context, torch.tensor([7]))[0, 0]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            proposals, _ = self.draft.propose_greedy_cached(ids[:, -1:], cache.layers, 7)
            self.assertEqual(proposals, self.draft.propose_greedy(ids, context)[0])
        with self.assertRaises(ValueError): cache.crop(8)

    def test_real_draft_matches_cached_sequential_target(self):
        expected = self.target.greedy(self.prompt, 17)
        actual, rounds = cached_speculative_greedy(self.target, self.draft, self.prompt, 17)
        self.assertEqual(actual, expected)
        self.assert_invariants(actual, rounds)

    def test_rejection_at_every_position_and_all_accepted_bonus(self):
        expected = self.target.greedy(self.prompt, 48)
        for rejection in (0, 1, 2, None):
            with self.subTest(rejection=rejection):
                draft = self.controlled_draft(expected, rejection)
                actual, rounds = cached_speculative_greedy(self.target, draft, self.prompt, 17)
                self.assertEqual(actual, expected[:17])
                self.assertEqual(rounds[0]["accepted"], 3 if rejection is None else rejection)
                self.assertEqual(rounds[0]["committed"], 4 if rejection is None else rejection + 1)
                self.assert_invariants(actual, rounds)

    def test_same_prefix_diagnostic_replays_prior_accepted_cache_history(self):
        expected = self.target.greedy(self.prompt, 48)
        actual, rounds = cached_speculative_greedy(self.target,
            self.controlled_draft(expected), self.prompt, 17, trace_tokens=True)
        for index in (5, 7, 9):
            diagnostic = diagnose_same_prefix(self.target, self.full, self.prompt, actual, index, rounds)
            self.assertTrue(diagnostic["available"])
            self.assertEqual(diagnostic["prefix_length"], self.prompt.shape[1] + index)
            for path in diagnostic["paths"].values():
                self.assertEqual(path["argmax"], actual[index])
            for field in ("current_chunk_max_logit_difference", "history_chunking_max_logit_difference",
                          "sequential_vs_full_max_logit_difference"):
                self.assertLess(diagnostic[field], 1e-5)

    def test_budget_crops_even_an_accepted_last_token_and_discards_bonus(self):
        expected = self.target.greedy(self.prompt, 48)
        for count in (0, 1, 2, 3, 4, 5, 8, 9):
            with self.subTest(count=count):
                actual, rounds = cached_speculative_greedy(self.target,
                    self.controlled_draft(expected), self.prompt, count)
                self.assertEqual(actual, expected[:count])
                if rounds:
                    self.assertEqual(rounds[-1]["termination"], "max_new_tokens")
                    self.assertLessEqual(rounds[-1]["committed_accepted"], rounds[-1]["committed"])
                self.assert_invariants(actual, rounds)

    def test_eos_anchor_accepted_correction_and_bonus_boundaries(self):
        expected = self.target.greedy(self.prompt, 48)
        for index in (0, 1, 2, 3, 4, 8):
            eos = {expected[index]}
            stop = next(i for i, token in enumerate(expected) if token in eos) + 1
            for rejection in (None, 0, 1, 2):
                with self.subTest(index=index, rejection=rejection):
                    actual, rounds = cached_speculative_greedy(self.target,
                        self.controlled_draft(expected, rejection), self.prompt, 20, eos)
                    self.assertEqual(actual, expected[:stop])
                    if rounds:
                        self.assertEqual(rounds[-1]["termination"], "eos")
                    self.assert_invariants(actual, rounds)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
