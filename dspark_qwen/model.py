"""Qwen3 DSpark architecture adapted from NVIDIA NeMo AutoModel (Apache-2.0).

See THIRD_PARTY_NOTICES.md. Single-sequence dense SDPA implementation, vanilla
Markov head only. Frozen embedding/head modules alias the target in memory.
"""
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding


def block_mask(anchors, block_size, context_length):
    """True means visible: context j < anchor; own draft block is bidirectional."""
    n = anchors.numel()
    q_block = torch.arange(n * block_size, device=anchors.device) // block_size
    ctx = torch.arange(context_length, device=anchors.device)
    visible_context = ctx[None, :] < anchors[q_block, None]
    visible_block = q_block[:, None] == q_block[None, :]
    return torch.cat((visible_context, visible_block), dim=-1)[None, None]


def rotate(x, cos, sin):
    left, right = x.chunk(2, dim=-1)
    return x * cos[:, None] + torch.cat((-right, left), -1) * sin[:, None]


class DraftAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = config.head_dim
        self.q_proj = nn.Linear(h, self.heads * self.dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(h, self.kv_heads * self.dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(h, self.kv_heads * self.dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.heads * self.dim, h, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.dim, eps=config.rms_norm_eps)

    def forward(self, z, context, cos, sin, mask):
        b, length, _ = z.shape
        q = self.q_norm(self.q_proj(z).view(b, length, self.heads, self.dim)).transpose(1, 2)
        # Same learned K/V projections for context and block, as in NeMo.
        both = torch.cat((context, z), dim=1)
        k = self.k_norm(self.k_proj(both).view(b, -1, self.kv_heads, self.dim)).transpose(1, 2)
        v = self.v_proj(both).view(b, -1, self.kv_heads, self.dim).transpose(1, 2)
        q = rotate(q, cos[:, -length:], sin[:, -length:])
        k = rotate(k, cos, sin)
        groups = self.heads // self.kv_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        return self.o_proj(out.transpose(1, 2).reshape(b, length, -1))


class DraftBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = DraftAttention(config)
        self.post_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, z, context, cos, sin, mask):
        z = z + self.attention(self.input_norm(z), context, cos, sin, mask)
        return z + self.mlp(self.post_norm(z))


class DSparkDraft(nn.Module):
    def __init__(self, target, spec):
        super().__init__()
        c = target.config
        spec.validate(c)
        self.spec = spec
        self.fc = nn.Linear(len(spec.layer_ids) * c.hidden_size, c.hidden_size, bias=False)
        self.context_norm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.layers = nn.ModuleList([DraftBlock(c) for _ in range(spec.num_layers)])
        self.norm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.rotary = Qwen3RotaryEmbedding(c)
        self.markov_embedding = nn.Embedding(c.vocab_size, spec.markov_rank)
        self.markov_projection = nn.Linear(spec.markov_rank, c.vocab_size, bias=False)
        self.confidence = nn.Linear(c.hidden_size + spec.markov_rank, 1)
        # Initialize only new weights, before attaching the frozen target modules.
        def init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=c.initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(init)
        self.embedding = target.get_input_embeddings().requires_grad_(False)
        self.lm_head = target.get_output_embeddings().requires_grad_(False)

    def backbone(self, input_ids, features, anchors):
        if input_ids.shape[0] != 1 or features.shape[0] != 1 or anchors.ndim != 1:
            raise ValueError("Single sequence and a 1D anchor list required")
        if not anchors.numel() or torch.any(anchors < 0) or torch.any(anchors >= input_ids.shape[1]):
            raise ValueError("Invalid anchor positions")
        n, length = anchors.numel(), self.spec.block_size
        noise = torch.full((1, n, length), self.spec.mask_token_id, device=input_ids.device, dtype=torch.long)
        noise[0, :, 0] = input_ids[0, anchors]
        z = self.embedding(noise.flatten(1))
        context = self.context_norm(self.fc(features.detach()))
        context_positions = torch.arange(features.shape[1], device=input_ids.device)
        draft_positions = (anchors[:, None] + torch.arange(length, device=anchors.device)).flatten()
        positions = torch.cat((context_positions, draft_positions))[None]
        # Keep model trainable weights FP32 and use autocast; never downcast RoPE buffers.
        cos, sin = self.rotary(z, positions)
        mask = block_mask(anchors, length, features.shape[1])
        for layer in self.layers:
            z = layer(z, context, cos, sin, mask)
        return self.norm(z).reshape(1, n, length, -1)

    def forward(self, input_ids, features, anchors):
        hidden = self.backbone(input_ids, features, anchors)
        offsets = torch.arange(self.spec.block_size, device=anchors.device)
        label_positions = anchors[:, None] + offsets + 1
        valid = label_positions < input_ids.shape[1]
        safe = label_positions.clamp(max=input_ids.shape[1] - 1)
        previous = input_ids[:, (anchors[:, None] + offsets).clamp(max=input_ids.shape[1] - 1)]
        prev_emb = self.markov_embedding(previous)
        logits = self.lm_head(hidden) + self.markov_projection(prev_emb)
        confidence = self.confidence(torch.cat((hidden, prev_emb), -1)).squeeze(-1)
        return dict(logits=logits, confidence=confidence, labels=input_ids[:, safe],
                    label_positions=safe, valid=valid[None], hidden=hidden)

    @torch.no_grad()
    def propose_greedy(self, input_ids, context):
        anchor = torch.tensor([input_ids.shape[1] - 1], device=input_ids.device)
        hidden = self.backbone(input_ids, context, anchor)[0, 0]
        base = self.lm_head(hidden)
        prev = input_ids[0, -1:]
        tokens, confidences = [], []
        for k in range(self.spec.block_size):
            emb = self.markov_embedding(prev)
            # Confidence is computed before sampling current token; no look-ahead.
            confidences.append(self.confidence(torch.cat((hidden[k:k+1], emb), -1)).sigmoid().item())
            prev = (base[k:k+1] + self.markov_projection(emb)).argmax(-1)
            tokens.append(prev.item())
        return tokens, confidences

    def trainable_state(self):
        names = {name for name, p in self.named_parameters() if p.requires_grad}
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items() if name in names}

    def load_trainable_state(self, state):
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError("Checkpoint trainable parameter names do not match architecture")
        self.load_state_dict(state, strict=False)  # Frozen weights reconstructed from fingerprinted target.
