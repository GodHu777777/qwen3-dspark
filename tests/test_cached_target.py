import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dspark_qwen.cached_target import CachedTarget
from dspark_qwen.bench_cached_target import check_hf_cached, check_recompute


class CachedTargetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        self.model = Qwen3ForCausalLM(config).eval()
        self.target = CachedTarget(self.model, layer_ids=(0, 2))
        self.ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

    @torch.no_grad()
    def full(self, ids):
        return self.model.model(ids, use_cache=False, output_hidden_states=True, return_dict=True)

    def assert_chunk(self, chunk, prefix):
        expected = self.full(prefix)
        torch.testing.assert_close(chunk.last, expected.last_hidden_state[:, chunk.start:],
                                   atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(self.target.logits(chunk.last),
            self.target.logits(expected.last_hidden_state[:, chunk.start:]), atol=1e-6, rtol=1e-5)
        context = torch.cat((expected.hidden_states[1], expected.hidden_states[3]), -1)
        torch.testing.assert_close(chunk.context, context[:, chunk.start:], atol=1e-6, rtol=1e-5)
        self.assertEqual(chunk.end, prefix.shape[1])
        self.assertEqual(self.target.length, prefix.shape[1])
        self.assertFalse(chunk.last.requires_grad)

    def test_prefill_incremental_and_block_logits_match_full_recompute(self):
        self.assert_chunk(self.target.prefill(self.ids[:, :3]), self.ids[:, :3])
        self.assert_chunk(self.target.append(self.ids[:, 3:4]), self.ids[:, :4])
        self.assert_chunk(self.target.append(self.ids[:, 4:]), self.ids)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))

    def test_crop_rejected_suffix_and_replacement_at_every_boundary(self):
        # Verification processed positions 3..8. Keep none, some or all of that
        # block, then process a correction/bonus. Also test a complete rollback.
        for keep in range(10):
            with self.subTest(keep=keep):
                self.target.prefill(self.ids[:, :3])
                self.target.append(self.ids[:, 3:])
                self.target.crop(keep)
                replacement = torch.tensor([[41, 42]])
                prefix = torch.cat((self.ids[:, :keep], replacement), -1)
                self.assert_chunk(self.target.append(replacement), prefix)

    def test_repeated_rollbacks_and_prefill_reset(self):
        self.target.prefill(self.ids)
        for keep in (7, 3, 0):
            self.target.crop(keep)
            self.assert_chunk(self.target.append(self.ids[:, keep:]), self.ids)
        self.assert_chunk(self.target.prefill(self.ids[:, :1]), self.ids[:, :1])
        without_features = CachedTarget(self.model)
        self.assertIsNone(without_features.prefill(self.ids).context)

    @torch.no_grad()
    def full_greedy(self, ids, count, eos=()):
        tokens = []
        for _ in range(count):
            hidden = self.full(ids).last_hidden_state[:, -1]
            token = self.target.logits(hidden).argmax(-1)
            tokens.append(token.item())
            ids = torch.cat((ids, token[:, None]), -1)
            if tokens[-1] in eos:
                break
        return tokens

    def test_greedy_exact_tokens_eos_and_output_limits(self):
        for prompt_length in (1, 3, 9):
            prompt = self.ids[:, :prompt_length]
            expected = self.full_greedy(prompt, 12)
            for count in (0, 1, 2, 12):
                with self.subTest(prompt_length=prompt_length, count=count):
                    self.assertEqual(self.target.greedy(prompt, count), expected[:count])
                    self.assertEqual(self.target.length, prompt_length + count - 1 if count else 0)
            for eos in ({expected[0]}, {expected[4], 63}):
                self.assertEqual(self.target.greedy(prompt, 12, eos), self.full_greedy(prompt, 12, eos))

    def test_hf_cached_policy_and_full_recompute_diagnostic(self):
        saved_config = self.model.generation_config
        saved_config.repetition_penalty = 2.0
        for eos in (set(), {self.full_greedy(self.ids, 1)[0]}):
            self.assertTrue(check_hf_cached(self.target, self.ids, 12, eos)["equal"])
            self.assertTrue(check_recompute(self.target, self.ids, 12, eos)["equal"])
        self.assertIs(self.model.generation_config, saved_config)
        self.assertEqual(saved_config.repetition_penalty, 2.0)

    def test_invalid_operations_do_not_mutate_valid_cache(self):
        with self.assertRaises(ValueError): self.target.append(self.ids)
        with self.assertRaises(ValueError): self.target.crop(0)
        self.target.prefill(self.ids)
        for length in (-1, 10, True, 2.5):
            with self.assertRaises(ValueError): self.target.crop(length)
            self.assertEqual(self.target.length, 9)
        for ids in (self.ids[:, :0], self.ids.expand(2, -1), self.ids.float()):
            with self.assertRaises(ValueError): self.target.append(ids)
            self.assertEqual(self.target.length, 9)
        for count in (-1, 1.5, True):
            with self.assertRaises(ValueError): self.target.greedy(self.ids, count)
        with self.assertRaises(ValueError): CachedTarget(self.model, (2, 0))
        with self.assertRaises(ValueError): CachedTarget(self.model, (4,))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
