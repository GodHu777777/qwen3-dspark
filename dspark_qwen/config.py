from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DraftConfig:
    layer_ids: tuple[int, ...] = (1, 7, 14, 21, 26)
    num_layers: int = 5
    block_size: int = 7
    markov_rank: int = 256
    mask_token_id: int = 151669
    num_anchors: int = 4
    ce_alpha: float = 0.1
    l1_alpha: float = 0.9
    confidence_alpha: float = 1.0
    loss_decay_gamma: float = 4.0

    def validate(self, target):
        if tuple(sorted(set(self.layer_ids))) != tuple(self.layer_ids) or not self.layer_ids:
            raise ValueError("layer_ids must be nonempty, strictly increasing")
        if not all(0 <= i < target.num_hidden_layers - 1 for i in self.layer_ids):
            raise ValueError("This implementation captures intermediate block outputs only")
        if not 0 <= self.mask_token_id < target.vocab_size:
            raise ValueError("mask_token_id outside target vocabulary")
        if min(self.num_layers, self.block_size, self.markov_rank, self.num_anchors) < 1:
            raise ValueError("Draft sizes must be positive")
        if self.loss_decay_gamma <= 0:
            raise ValueError("loss_decay_gamma must be positive")
        if min(self.ce_alpha, self.l1_alpha, self.confidence_alpha) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if getattr(target, "use_sliding_window", False):
            raise ValueError("Sliding-window targets are not supported by this reference")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if "layer_ids" in data:
            data["layer_ids"] = tuple(data["layer_ids"])
        return cls(**data)
