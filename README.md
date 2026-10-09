# Qwen3 DSpark

A readable, single-GPU reproduction of the Qwen3 DSpark draft architecture,
with auditable data regeneration, training, and an exact greedy reference verifier.
Based on [NVIDIA NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding),
pinned to commit `2d365eda1050dd80ee9bd3bfc651e9ae9bbe8c68`.

**Research implementation in progress. No inference speedup has been demonstrated.**
The full-prefix greedy verifier is a correctness reference. Incremental target
and draft caches are also implemented; their native BF16 equality gate has known
failures described below.

A [packed multi-request decoder](docs/packed-sampling.md) and a
[two-step capacity reference](docs/async-capacity.md) now run together on tiny CPU
models, with per-request KV reconstruction and confidence-source checks. Native
GPU integration, measured hardware profiles and execution overlap remain open.

## Architecture

```mermaid
flowchart LR
    X[Prefix tokens] --> T[Frozen Qwen3 target]
    T --> H[Block outputs 1 / 7 / 14 / 21 / 26]
    H --> P[Concat + projection + RMSNorm]
    P --> KV[Context K/V in each draft layer]
    A[Anchor token + 6 MASK tokens] --> D[5-layer parallel draft]
    KV --> D
    D --> M[Shared frozen LM head + rank-256 Markov head]
    D --> C[Confidence head]
    M --> V[Target verification]
    V --> O[Accept prefix / correct rejection / EOS]
```

Layer indices are zero-based. Draft queries attend to context strictly before
that block's anchor and to every position in their own draft block. Blocks cannot
see one another. The target's final normalized hidden states provide detached
teacher probabilities for training; they are distinct from the five-layer draft
context. Frozen embedding and LM-head parameters are shared in memory.

## Quick start

Install a PyTorch build appropriate to your CUDA/ROCm device first. The tested
GPU environment is PyTorch 2.12.0+rocm7.2 and Transformers 5.17.0.

```bash
python -m pip install -e '.[data]'
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
cp configs/pilot.example.json configs/pilot.local.json
cp configs/train-pilot.example.json configs/train-pilot.local.json
```

Download Qwen3-0.6B separately and set `model` to its local directory in both
configs. Set data and output paths for your machine. Model and dataset licenses
remain applicable; neither weights nor generated examples are included here.
The pinned source dataset shard identity is in [references/source.json](references/source.json).

```bash
python scripts/prepare.py --config configs/pilot.local.json \
  --parquet /path/to/train-00000-of-00006.parquet --output data/pilot/input
python scripts/generate.py --config configs/pilot.local.json \
  --prompts data/pilot/input/prompts.jsonl --output data/pilot/generated
python scripts/audit.py --run data/pilot
python -m dspark_qwen.train --config configs/train-pilot.local.json
python -m dspark_qwen.eval_decode --checkpoint runs/train-pilot/step-000008 \
  --examples 2 --max-new-tokens 24
```

Generation resumes complete batches with identical inputs/config/runtime. Training
supports `--stop-after N` followed by `--resume`; source, configuration, model,
data and runtime identities must match. Use a new directory for a new experiment.
Checkpoint optimizer files must come from a trusted local run.

## Evidence and limits

The [initial execution experiment](reports/pilot-20261008/README.md) used 49 training
and 7 held-out completed responses. Eight optimizer steps with a process restart
verified the training/checkpoint path. Eight CPU tests covered mask parity with
NeMo, information isolation, gradients, exact optimizer resume and greedy branches.
Two real prompts produced 30 tokens identical to target-only greedy, with **zero
accepted draft tokens**. These are execution/correctness checks, not quality or
performance results.

A subsequent [aligned single-record learning diagnostic](reports/diagnostic-aligned-20261008/README.md)
used one target-greedy trajectory and all possible rollout anchors. At steps
64, 96 and 128, all seven draft positions matched the teacher and all seven
proposals per round were accepted (32 output tokens exactly matched the target).
The [earlier sampled-trajectory diagnostic](reports/diagnostic-20261008/README.md)
exposed a confound: its training and rollout prefixes did not match. The aligned
result establishes learnability on seen prefixes, **not held-out quality or speed**.

Run the bounded diagnostic after generating the pilot data:

```bash
python -m dspark_qwen.diagnose_learning --config configs/train-pilot.local.json \
  --output runs/diagnostic-aligned --steps 128 --eval-every 32 \
  --rollout-tokens 32 --trajectory target-greedy --anchor-coverage contiguous
```

The [cached target baseline](reports/cached-target-20261008/README.md) on AMD
Radeon AI PRO R9700 measured about 30 decode tokens/s in eager Transformers
(BF16, batch 1). It matched Hugging Face cached greedy on 117 checked tokens.
This measures the target alone, without draft or intermediate-feature costs.
One full-recompute BF16 trace differed near a tie; different kernel shapes are
not guaranteed to produce identical floating-point argmaxes. Six CPU cache
regressions cover chunking, rollback, EOS and the HF reference configuration.
The [cached speculative prototype](reports/cached-decode-gate-20261008/README.md)
failed its real BF16 equality gate on two of three prompts, despite passing
CPU cache invariants; block-versus-sequential numerical fidelity remains unresolved. These measurements do not establish speculative speedup.

An optional [fixed-shape numerical control](docs/canonical-target.md) passed a
[real-draft comparison](reports/canonical-real-draft-gate-20261008/README.md)
against its own sequential execution on 3/3 prompts (70 tokens), but matched
stock decoding on only 2/3. Both held-out prompts accepted zero draft tokens.
This separate execution policy does not resolve the native BF16 gate or prove speedup.

The five-layer draft has 161,692,161 trainable parameters. Training uses single
unpadded sequences, dense SDPA, FP32 trainables and BF16 autocast. Four anchors and
two accumulation microsteps keep the pilot small; this is not paper-scale training.
The local checkpoint format is not directly compatible with NeMo, vLLM or SGLang.

The [reproduction scope](docs/dspark-reproduction-scope.md) separates the paper's
algorithm, public training code and production system. Independent CPU modules
now cover [stochastic verification](docs/stochastic-sampling.md), global prefix
allocation and [sequential confidence calibration](docs/confidence-calibration.md).
Their mathematical tests do not establish trained quality or serving performance.
The [cached stochastic path](docs/cached-stochastic.md) retains actual Markov
proposal probabilities and passes CPU distribution/cache checks. Its
[real BF16 gate](reports/stochastic-gate-20261009/README.md) passed 9 executions
and 6 same-path repeat checks, but 47 of 48 same-prefix probability comparisons
differed (maximum TV 0.04162). Execution reproducibility does not establish
distribution equivalence to sequential decoding. A
[packed target reference](docs/packed-target.md) verifies variable-length chunks
in one model call, with independent request caches; its attention mask remains
dense and still awaits real GPU validation.
Real rollout calibration, efficient multi-request execution, hardware capacity
profiles and asynchronous scheduling remain unfinished. The earlier
[experiment plan](docs/experiment-plan.md) preserves the initial stage gates.

The [three-way data pipeline](docs/data-pipeline.md) adds pilot exclusions,
immutable selection/resume, and a separate final-test export for expanded training.
Its [completed expansion](reports/data-expansion-20261009/README.md) produced
932 train, 119 dev and 119 final-test responses from 1,280 inputs; 110 were rejected.
Structural audits passed. Final test remains excluded from development and tuning.

[Expanded training through step1280](reports/expanded-training-step1280-20261009/README.md)
uses 32 anchors and accumulation 8, with verified checkpoint/optimizer resumes and
a frozen target. On the same 119 dev rows, teacher-forced overlap increased from
0.086% initially to 37.34%; this is not measured rollout acceptance or speedup.
The [resource and first-resume report](reports/expanded-training-20261009/README.md)
records the two-cycle memory gate and earlier steps.

[Fixed-panel stochastic rollouts](reports/expanded-quality512-20261009/README.md)
compare step128 and step512 on the same 32 prompts, source and sampling protocol.
Accepted draft tokens per round increased from 0.1778 to 0.4958, improving on
31 of 32 prompts. This is acceptance evidence, not a speedup measurement;
BF16 block/sequential probability differences remain, and STS is not fitted.

A [six-call ROCm diagnostic](reports/native-varlen-diagnostic-20261009/README.md)
identified a causal-window alignment mismatch in the tested public varlen path.
An explicit private-ATen control produced the required alignment on that shape.
The original public-wrapper gate remains failed. The explicit pinned backend
[passed all original small-tensor cases](reports/pinned-varlen-tensor-gate-20261009/README.md);
whole pretrained-model/KV validation is still pending.

## Experiment journal

[实验复盘日志](docs/lab-notebook.md)记录问题、假设、处理方法、验证结果、
失败尝试和下一步决策，并关联提交与证据。历史结果与进行中的实验分开标注。

## Code map

| Module | Responsibility |
| --- | --- |
| `target.py`, `model.py` | Frozen target capture, draft attention and heads |
| `losses.py`, `data.py` | CE/L1/confidence supervision and exact token alignment |
| `train.py`, `checkpoint.py` | Training, evaluation, state and identity checks |
| `decode.py`, `eval_decode.py` | Full-recompute greedy reference and token comparison |
| `cached_target.py`, `bench_cached_target.py` | Incremental target cache and isolated cost measurement |
| `cached_decode.py`, `eval_cached_decode.py` | Experimental draft KV lifecycle and numerical fidelity gate |
| `canonical_target.py` | Optional fixed-shape numerical control |
| `sampling.py` | CPU stochastic probability reference and exact-distribution tests |
| `tensor_sampling.py`, `cached_sampling.py` | Experimental random Markov proposals, residual sampling and cache commits |
| `scheduler.py`, `calibration.py` | Global prefix planning and sequential confidence temperature fitting |
| `packed_target.py` | One-forward variable-query target with per-request KV and dense marker mask |
| `scripts/` | Prompt selection, target regeneration and data auditing |
| `tests/` | Tensor-level correctness and regression checks |

Development instructions: [README.ai.md](README.ai.md). Apache-2.0;
[third-party attribution](THIRD_PARTY_NOTICES.md).
