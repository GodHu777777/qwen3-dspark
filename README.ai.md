# DSpark Qwen training lab

Scope: pilot data regeneration, single-GPU DSpark draft training and a reference
greedy verifier for frozen dense Qwen3. This directory is an independent Git
repository with origin git@github.com:GodHu777777/qwen3-dspark.git. Read AGENTS.md.
The enclosing workspace, if present, has its own README.ai.md.

Local source is authoritative. Remote scripts are explicitly copied to
`amd:/home/nju/services/dspark-qwen`; data and runs live under
`/home/nju/data/dspark-qwen`. Model files and existing Python environments are
read-only dependencies. Never stop unrelated GPU processes without authorization.

- `configs/pilot.json`: pinned source, target path, sampling and size limits.
- `scripts/prepare.py`: sample unique first-user prompts from one pinned Parquet
  shard; split before regeneration. Discard original assistant answers.
- `scripts/generate.py`: batched Transformers generation with exact token IDs,
  fingerprints, per-batch seeds, output validation and atomic batch resume.
- `scripts/audit.py`: stdlib audit of prompt identity, disjoint splits, EOS,
  template alignment, length bounds and exported hashes.
- `output/`: copied run evidence and generated pilot files.
- `dspark_qwen/`: target capture, mask/backbone/heads, losses, data, checkpoint,
  training and full-recompute greedy decoding. No NeMo runtime dependency.
- `configs/train-pilot.json`: current short real-device training configuration.
- `tests/test_core.py`: standard unittest suite, including pinned NeMo mask parity.
- `references/nemo-2d365eda/`: exact upstream source snapshot and hashes; reference
  mask is executed as a CPU oracle. Preserve Apache-2.0 attribution.

Run modules from remote /home/nju/services/dspark-qwen with the existing Python.
Only copy the intended package/config/tests; do not modify base environments or
target weights. Keep large trainable checkpoints and optimizer states remote.

Training invariants: selected layer IDs are zero-based decoder block outputs;
context KV visibility is strictly position < anchor; own draft block is
bidirectional, other blocks invisible. Anchor+6 masks predict the NEXT 7 tokens.
Final normalized target features supply detached teacher distributions, not
draft context. Target inference is no_grad. Trainables remain FP32 under BF16 AMP,
with FP32 RoPE frequencies. Frozen target embedding/head are aliased in memory.
Each micro-step has one unpadded sequence; only assistant output/EOS is supervised.

Checkpoint identity binds source, target/tokenizer hashes, data, config and runtime.
Do not mutate a completed run manifest to get past resume checks. New source or
config requires a new run; checkpoint formats are local, not NeMo-compatible.
decode.py supports exact greedy acceptance logic only, full-prefix recomputation;
stochastic sampling, KV cache lifecycle and hardware scheduling remain future work.

The first-shard pilot is not a representative full-dataset reproduction. Clean
train/validation exports contain only EOS-completed nonempty answers fitting the
training sequence budget. Other outcomes are kept in rejected.jsonl. Do not
filter factual mistakes: the goal is to reproduce the target's behavior.

Resume requires identical prompt bytes, configuration, script, model fingerprints,
and runtime versions. It restarts whole unfinished batches; batching is part of
the generation configuration. Exact GPU bit reproducibility is not promised.
Tokenizer training must use the same non-thinking template and special tokens;
raw prompt/output token IDs are retained to audit tokenization alignment.

Verified 2026-10-08 pilot: 64 inputs, 49 training + 7 validation completions,
8 length-capped rejects. Local evidence: output/pilot-20261008. Remote run:
/home/nju/data/dspark-qwen/pilot-20261008. Torch 2.12.0+rocm7.2,
Transformers 5.17.0; no environments or model weights were modified.
Tokenize=True now returns BatchEncoding by default in that Transformers version;
keep explicit return_dict=False when a token list is required.

Real training verified 2026-10-08: current run is
`/home/nju/data/dspark-qwen/train-pilot-20261008-v2`, 8 optimizer steps with a
step-4 process restart. Local evidence in output/train-pilot-20261008-v2.
8 unit tests passed; a real 2-prompt, 30-token greedy comparison matched target
exactly but accepted ZERO draft tokens. Teacher-forced overlap after 8 steps
was ~1.29%, not a serving acceptance estimate. Do not claim trained quality or
speedup. v2 final weights match the directly decoded previous run byte-for-byte;
greedy-equivalence.json records this and identical decoding source/target hashes.
The train config output path already exists: choose a new directory for a new run.

Public checkout: configs/*.example.json are portable templates; existing
pilot.json and train-pilot.json are ignored machine-specific launch configs.
output/ is ignored (contains generated text). Publish aggregate evidence only in
reports/ and document source identity/scope. docs/experiment-plan.md defines
learnability, held-out quality, cached correctness, cost and scheduling gates.
Use git -C dspark-qwen when working from the parent workspace.
