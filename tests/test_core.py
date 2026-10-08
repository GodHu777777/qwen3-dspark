import tempfile
import importlib.util
import unittest
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dspark_qwen.checkpoint import load_checkpoint, save_checkpoint
from dspark_qwen.config import DraftConfig
from dspark_qwen.data import select_anchors
from dspark_qwen.decode import speculative_greedy, target_greedy
from dspark_qwen.losses import objective
from dspark_qwen.model import DSparkDraft, block_mask
from dspark_qwen.target import FrozenTarget


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        c = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, attention_dropout=0.0)
        c._attn_implementation = "sdpa"
        self.model = Qwen3ForCausalLM(c)
        self.spec = DraftConfig(layer_ids=(0, 2), num_layers=2, block_size=3,
            markov_rank=8, mask_token_id=63, num_anchors=2)
        self.target = FrozenTarget(self.model, self.spec)
        self.draft = DSparkDraft(self.model, self.spec)
        self.ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])

    def test_target_capture_matches_hf_and_is_frozen(self):
        f = self.target.capture(self.ids)
        expected = self.model.model(self.ids, output_hidden_states=True, return_dict=True)
        torch.testing.assert_close(f.context, torch.cat((expected.hidden_states[1], expected.hidden_states[3]), -1))
        torch.testing.assert_close(f.last, expected.last_hidden_state)
        self.assertFalse(f.context.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))

    def test_mask_truth_table_and_answer_anchor_boundary(self):
        m = block_mask(torch.tensor([2, 4]), 3, 7)[0, 0]
        self.assertEqual(m[0].nonzero().flatten().tolist(), [0, 1, 7, 8, 9])
        self.assertEqual(m[4].nonzero().flatten().tolist(), [0, 1, 2, 3, 10, 11, 12])
        a = select_anchors(torch.tensor([[False, False, False, True, True]]), 9,
                           torch.Generator().manual_seed(1))
        self.assertEqual(a.tolist(), [2, 3])

    def test_mask_matches_pinned_nemo_reference(self):
        path = Path(__file__).resolve().parents[1] / "references/nemo-2d365eda/dflash_mask.py"
        loader = importlib.util.spec_from_file_location("nemo_reference_mask", path)
        reference = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(reference)
        anchors = torch.tensor([[0, 2, 6]])
        expected = reference.create_dflash_sdpa_mask(anchors, torch.ones_like(anchors, dtype=torch.bool),
            ctx_len=7, block_size=3, device=torch.device("cpu"), dtype=torch.float32)
        self.assertTrue(torch.equal(block_mask(anchors[0], 3, 7), expected == 0))

    def test_future_context_and_other_block_cannot_leak(self):
        f = self.target.capture(self.ids)
        anchors = torch.tensor([2, 4])
        original = self.draft.backbone(self.ids, f.context, anchors)
        changed_context = f.context.clone()
        changed_context[:, 2:] += torch.randn_like(changed_context[:, 2:]) * 100
        changed_ids = self.ids.clone(); changed_ids[:, 4] = 42
        perturbed = self.draft.backbone(changed_ids, changed_context, anchors)
        torch.testing.assert_close(original[:, 0], perturbed[:, 0], atol=1e-6, rtol=1e-6)
        visible = f.context.clone(); visible[:, :2] += torch.randn_like(visible[:, :2]) * 100
        different = self.draft.backbone(self.ids, visible, anchors)
        self.assertGreater((original[:, 0] - different[:, 0]).abs().max().item(), 1e-4)

    def test_labels_loss_and_gradient_paths(self):
        f = self.target.capture(self.ids)
        out = self.draft(self.ids, f.context, torch.tensor([2, 5]))
        self.assertEqual(out["labels"][0, 0].tolist(), [4, 5, 6])
        self.assertEqual(out["valid"][0, 1].tolist(), [True, False, False])
        teacher = self.target.logits(f.last[:, out["label_positions"] - 1])
        loss, metrics = objective(out, teacher, torch.ones_like(self.ids, dtype=torch.bool), self.spec)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["supervised_tokens"], 4)
        for module in [self.draft.fc, self.draft.layers, self.draft.markov_embedding,
                       self.draft.markov_projection, self.draft.confidence]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_acceptance_target_matches_discrete_overlap(self):
        p = torch.tensor([0.7, 0.3]); q = torch.tensor([0.2, 0.8])
        out = dict(logits=q.log().reshape(1, 1, 1, 2).requires_grad_(),
            confidence=torch.zeros(1, 1, 1, requires_grad=True), labels=torch.zeros(1, 1, 1, dtype=torch.long),
            label_positions=torch.tensor([[0]]), valid=torch.ones(1, 1, 1, dtype=torch.bool))
        spec = DraftConfig(block_size=1, ce_alpha=0, l1_alpha=0, confidence_alpha=1)
        loss, metrics = objective(out, p.log().reshape(1, 1, 1, 2), torch.tensor([[True]]), spec)
        self.assertAlmostEqual(metrics["teacher_forced_accept"], torch.minimum(p, q).sum().item(), places=6)
        loss.backward()
        # Confidence labels must not backpropagate into draft probabilities.
        self.assertEqual(out["logits"].grad.abs().sum().item(), 0)

    def test_checkpoint_roundtrip_and_resume_optimizer(self):
        opt = torch.optim.AdamW([p for p in self.draft.parameters() if p.requires_grad])
        rng = torch.Generator().manual_seed(123)
        # Populate optimizer state, not just an empty optimizer shell.
        f = self.target.capture(self.ids)
        self.draft(self.ids, f.context, torch.tensor([2]))["logits"].sum().backward()
        opt.step(); opt.zero_grad()
        expected = self.draft.trainable_state()
        with tempfile.TemporaryDirectory() as d:
            cp = save_checkpoint(d, 1, self.draft, opt, rng, {"identity": {"test": 1}})
            with torch.no_grad(): self.draft.fc.weight.add_(1)
            load_checkpoint(cp, self.draft, {"test": 1}, opt, rng)
            for name, tensor in expected.items():
                torch.testing.assert_close(self.draft.trainable_state()[name], tensor, rtol=0, atol=0)
            def update_once():
                opt.zero_grad()
                self.draft(self.ids, f.context, torch.tensor([2]))["logits"].sum().backward()
                opt.step()
                return self.draft.trainable_state()
            uninterrupted = update_once()
            load_checkpoint(cp, self.draft, {"test": 1}, opt, rng)
            resumed = update_once()
            for name in uninterrupted:
                torch.testing.assert_close(uninterrupted[name], resumed[name], rtol=0, atol=0)
            with self.assertRaises(ValueError): load_checkpoint(cp, self.draft, {"test": 2})

    def test_greedy_verify_rejection_all_accept_bonus_eos_and_limit(self):
        prompt = self.ids[:, :3]
        expected = target_greedy(self.target, prompt, 9, set())
        actual, _ = speculative_greedy(self.target, self.draft, prompt, 9, set())
        self.assertEqual(actual, expected)
        target = self.target
        class ControlledDraft:
            def __init__(self, reject): self.reject = reject
            def eval(self): return self
            def propose_greedy(self, ids, _context):
                tokens = target_greedy(target, ids, 3, set())
                if self.reject: tokens[0] = (tokens[0] + 1) % 64
                return tokens, [0.5] * 3
        for reject in [False, True]:
            actual, rounds = speculative_greedy(target, ControlledDraft(reject), prompt, 9, set())
            self.assertEqual(actual, expected)
            self.assertEqual(rounds[0]["accepted"], 0 if reject else 3)
        eos = expected[2]
        actual, _ = speculative_greedy(target, ControlledDraft(False), prompt, 9, {eos})
        self.assertEqual(actual, expected[:expected.index(eos) + 1])
        self.assertEqual(speculative_greedy(target, self.draft, prompt, 0, set())[0], [])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
