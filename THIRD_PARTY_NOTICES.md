# Third-party sources

Architecture and loss implementation adapted from NVIDIA NeMo AutoModel,
copyright (c) 2026 NVIDIA CORPORATION, licensed under Apache License 2.0.
License copy: `references/NeMo-LICENSE`.

Reference commit: `2d365eda1050dd80ee9bd3bfc651e9ae9bbe8c68`.
Repository: https://github.com/NVIDIA-NeMo/Automodel

Relevant source paths:

- `nemo_automodel/components/speculative/dspark/draft_qwen3.py`
- `nemo_automodel/components/speculative/dspark/target.py`
- `nemo_automodel/components/speculative/dspark/markov_head.py`
- `nemo_automodel/components/speculative/dspark/loss.py`
- `nemo_automodel/components/attention/dflash_mask.py`
- `examples/speculative/dspark/qwen3_0.6b_dspark.yaml`

This implementation separates target capture, masking, model, losses, data and
training for readability. It uses single-sequence dense PyTorch SDPA and FP32
trainable parameters under BF16 autocast, shares frozen target embeddings/head
in memory, and saves only trainable draft parameters. It is not a NeMo/SGLang
checkpoint-format implementation. Target weights retain their original license.
Transformers Qwen3 MLP, RMSNorm and RoPE are imported from the installed
Hugging Face Transformers package, not vendored here.
