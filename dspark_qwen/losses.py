"""NeMo-style CE + probability L1 + detached acceptance-target BCE."""
import torch
from torch.nn import functional as F


def objective(output, target_logits, answer_mask, spec):
    logits = output["logits"].float()
    teacher = target_logits.detach().float()
    valid = output["valid"] & answer_mask[:, output["label_positions"]]
    positions = torch.arange(spec.block_size, device=logits.device)
    weights = valid.float() * torch.exp(-positions.float() / spec.loss_decay_gamma)
    denominator = weights.sum()
    if denominator.item() == 0:
        raise ValueError("No assistant tokens supervised in these blocks")
    ce = F.cross_entropy(logits.flatten(0, -2), output["labels"].flatten(), reduction="none").reshape_as(weights)
    p, q = teacher.softmax(-1), logits.softmax(-1)
    l1 = (p - q).abs().sum(-1)
    accept = (1 - 0.5 * l1).clamp(0, 1).detach()
    conf = F.binary_cross_entropy_with_logits(output["confidence"].float(), accept, reduction="none")
    mean = lambda value: (value * weights).sum() / denominator
    ce_loss, l1_loss, conf_loss = mean(ce), mean(l1), mean(conf)
    loss = spec.ce_alpha * ce_loss + spec.l1_alpha * l1_loss + spec.confidence_alpha * conf_loss
    with torch.no_grad():
        metrics = {"loss": loss.item(), "ce": ce_loss.item(), "l1": l1_loss.item(),
                   "confidence_bce": conf_loss.item(), "teacher_forced_accept": mean(accept).item(),
                   "confidence_mae": mean((output["confidence"].sigmoid() - accept).abs()).item(),
                   "supervised_tokens": int(valid.sum())}
    return loss, metrics
