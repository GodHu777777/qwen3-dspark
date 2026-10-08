import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dspark_qwen.cached_decode import cached_speculative_greedy
from dspark_qwen.cached_target import CachedTarget
from dspark_qwen.canonical_target import CanonicalTarget
from dspark_qwen.config import DraftConfig
from dspark_qwen.model import DSparkDraft


class CanonicalTargetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(79)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=256, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        self.model = Qwen3ForCausalLM(config).eval()
        self.spec = DraftConfig(layer_ids=(0, 2), num_layers=2, block_size=7,
            markov_rank=8, mask_token_id=63)
        self.target = CanonicalTarget(self.model, self.spec.layer_ids, capacity=64)
        self.draft = DSparkDraft(self.model, self.spec).eval()
        self.ids = torch.tensor([[1, 2, 3, 4]])

    def test_padded_hidden_head_shape_and_only_valid_context(self):
        head_shapes = []
        hook = self.model.lm_head.register_forward_pre_hook(lambda _m, inputs: head_shapes.append(inputs[0].shape[1]))
        try:
            prefill = self.target.prefill(self.ids)
            self.target.predict(prefill, last_only=True)
            chunk = self.target.append(torch.tensor([[5]]))
            one = self.target.predict(chunk, last_only=True)
            all_valid = self.target.predict(chunk)
        finally:
            hook.remove()
        self.assertEqual(head_shapes, [4, 8, 8])
        self.assertEqual(chunk.last.shape[1], 1)
        self.assertEqual(chunk.context.shape[1], 1)
        self.assertEqual(chunk.padded_last.shape[1], 8)
        self.assertEqual(chunk.end, 5)
        torch.testing.assert_close(one, all_valid[:, 0], rtol=0, atol=0)
        self.target.validate_cache()
        with self.assertRaises(ValueError): self.target.logits(chunk.last)
        # Dynamic cache keeps the original one-row projection behavior.
        ordinary = CachedTarget(self.model)
        features = ordinary.prefill(self.ids)
        torch.testing.assert_close(ordinary.predict(features, last_only=True),
                                   ordinary.logits(features.last[:, -1]), rtol=0, atol=0)

    def test_dummy_and_rejected_suffix_cannot_leak(self):
        outputs, contexts = [], []
        for dummy, poison in ((0, -1000), (63, 1000)):
            target = CanonicalTarget(self.model, self.spec.layer_ids, capacity=64, dummy_token_id=dummy)
            target.prefill(self.ids)
            target.append(torch.arange(5, 13)[None])
            pointers = [(l.keys.data_ptr(), l.values.data_ptr()) for l in target.cache.layers]
            target.crop(5)
            for layer in target.cache.layers:
                layer.keys[:, :, 5:].fill_(poison)
                layer.values[:, :, 5:].fill_(poison)
            features = target.append(torch.tensor([[41, 42, 43]]))
            outputs.append(target.predict(features))
            contexts.append(features.context)
            self.assertEqual(pointers, [(l.keys.data_ptr(), l.values.data_ptr()) for l in target.cache.layers])
            self.assertEqual(target.length, 8)
            target.validate_cache()
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        torch.testing.assert_close(*contexts, rtol=0, atol=0)

    def test_crop_boundaries_full_context_and_capacity_headroom(self):
        for keep in range(13):
            self.target.prefill(self.ids)
            self.target.append(torch.arange(5, 13)[None])
            self.target.crop(keep)
            prefix = torch.cat((self.ids, torch.arange(5, 13)[None]), -1)[:, :keep]
            replacement = torch.tensor([[41]])
            features = self.target.append(replacement)
            reference = self.model.model(torch.cat((prefix, replacement), -1), use_cache=False,
                                         output_hidden_states=True, return_dict=True)
            torch.testing.assert_close(features.last, reference.last_hidden_state[:, -1:], atol=2e-6, rtol=1e-5)
            expected_context = torch.cat((reference.hidden_states[1][:, -1:], reference.hidden_states[3][:, -1:]), -1)
            torch.testing.assert_close(features.context, expected_context, atol=2e-6, rtol=1e-5)
            self.target.validate_cache()
        tiny = CanonicalTarget(self.model, capacity=8)
        tiny.prefill(self.ids)
        with self.assertRaises(ValueError): tiny.append(torch.tensor([[5]]))
        self.assertEqual(tiny.length, 4)

    def controlled(self, expected, reject_at):
        base, prompt_length = self.draft, self.ids.shape[1]
        class Controlled:
            spec = base.spec
            def eval(self): return self
            def project_context_kv(self, features, start): return base.project_context_kv(features, start)
            def propose_greedy_cached(self, anchor, kv, length):
                offset = length - prompt_length + 1
                proposal = expected[offset:offset + 7]
                if reject_at is not None: proposal[reject_at] = (proposal[reject_at] + 1) % 64
                return proposal, [0.5] * len(proposal)
        return Controlled()

    def test_actual_draft_and_all_rejection_boundaries(self):
        expected = self.target.greedy(self.ids, 40)
        real, _ = cached_speculative_greedy(self.target, self.draft, self.ids, 23)
        self.assertEqual(real, expected[:23])
        for rejection in list(range(7)) + [None]:
            actual, rounds = cached_speculative_greedy(self.target, self.controlled(expected, rejection), self.ids, 23)
            self.assertEqual(actual, expected[:23])
            self.assertEqual(self.target.length, self.ids.shape[1] + len(actual) - 1)
            self.assertEqual(rounds[0]["accepted"], 7 if rejection is None else rejection)
            self.target.validate_cache()

    def test_eos_limits_and_initial_reset(self):
        expected = self.target.greedy(self.ids, 40)
        for count in (0, 1, 2, 7, 8, 9, 16):
            for eos in (set(), {expected[0]}, {expected[5]}, {expected[8]}):
                wanted = expected[:count]
                stop = next((i for i, t in enumerate(wanted) if t in eos), None)
                if stop is not None: wanted = wanted[:stop + 1]
                actual, _ = cached_speculative_greedy(self.target,
                    self.controlled(expected, None), self.ids, count, eos)
                self.assertEqual(actual, wanted)
                self.assertEqual(self.target.length, self.ids.shape[1] + len(actual) - 1 if actual else 0)
                self.target.validate_cache()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
