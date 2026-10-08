# Frozen expanded-development rollouts

`rollout_protocol.py` freezes the development panel independently of checkpoint.
`collect_rollout.py` collects quality32 from any compatible expanded checkpoint.
The collector is a native stochastic protocol measurement, not a performance
benchmark or a proof of sequential-target distribution losslessness. Native BF16
block probabilities can differ from fresh sequential probabilities.

## Freeze once, reuse across checkpoints

Run from an independently deployed source checkout (use `git archive`, not macOS
`tar`, which can introduce AppleDouble Python sidecars). Choose private, new paths.
These manifest/dry-run commands use only the Python standard library and never
load Torch or touch CUDA:

```sh
python -m dspark_qwen.rollout_protocol --config /private/train-expanded.json --output /private/panel.json
python -m dspark_qwen.collect_rollout --checkpoint /private/step-000128 --manifest /private/panel.json --output /private/quality128-dry --dry-run
python -m dspark_qwen.collect_rollout --checkpoint /private/step-000512 --manifest /private/panel.json --output /private/quality512-dry --dry-run
```

The manifest requires exactly 119 accepted validation rows. The first 32 in the
immutable exported record order are quality. The remaining 87 are sorted by
SHA256 of `dspark-expanded-dev-sts-v1`, a NUL separator and record ID; the first
44 are fit, remaining 43 eval. IDs, exact token prompt identities and NFKC/whitespace
normalized first-user text identities must be unique across development records.
Test/unknown splits, missing rows, changed records and silent replacements fail.

Each row binds its complete record hash, prompt-token hash and seed. Seed is
`(20261009 + integer(SHA256(UTF8(record_id)))) modulo 2**63` by default; it does
not depend on checkpoint. Manifest identity also binds development file bytes,
generation manifest/summary bytes and target/tokenizer fingerprints. A checkpoint
must match data, target and block size; its weights and metadata are hashed
separately in each run. Fixed per-prompt seeds support paired checkpoint panels,
not token equality between different sampling algorithms.

The frozen protocol is temperature 1, no filtering, actual float64 q, 128 output
tokens and full block proposals capped only by remaining budget. Target is native
BF16/SDPA; draft parameters are FP32 under BF16 autocast. Only quality rows 0 and
1, round 0 are preselected for same-prefix TV probes. EOS before that round makes
the probe unreached; a replacement case/round is never selected from outcomes.

## Coordinated execution

After checkpoint and GPU-window review, omit `--dry-run` and choose a new output
directory. Execution creates a source snapshot, hashes all package modules and
runs a separate snapshot worker. The worker rechecks bound inputs before loading
the GPU. It requires 8 GiB free and caps process allocation at 6 GiB; default
worker timeout is 3600 seconds (configurable up to 7200). Existing output paths
are refused. Neither training source snapshots nor checkpoint identities change.

`run.json` holds private binding/prompt identities. `result.json` and
`aggregate.json` are atomically updated after each observed block, completed
prompt and numerical row; a failure marks partial evidence as failed.
`private-blocks.jsonl` contains `RolloutBlock`-compatible confidence/label inputs.
`private-rounds.jsonl` preserves proposal/committed token IDs, selected actual p/q,
normalization diagnostics, finite/cache checks and denominator evidence.
`private-outputs.jsonl` retains actual sampled outputs. Only the two preselected
probes retain bounded full tensors, on CPU, with partial sequential replay
preserved on failure. `stdout.log`, failure traces and `worker-exit.json` preserve
process evidence; timeout/launcher failure records are distinct from completion.
All files are private by default; review aggregate evidence before publication.

## Denominators and labels

Four denominators are recorded separately: proposed positions, target-verified
proposal positions, attempted acceptance uniforms and effective prefix labels.
After rejection at position j, every later verified cumulative-prefix event is
known zero even though no acceptance uniform is attempted there. Accepted draft
EOS includes its own position and removes later positions. Residual/bonus EOS
never counts as accepted draft EOS. Unproposed/unscored budget tails are absent.
A first-token EOS or one-token budget creates zero blocks, not fake zero labels.

Quality reports expose actual accepted draft token counts and per-position
prefix-event counts. Conditional overlap is not substituted for binary labels.
Cache audits verify target and projected draft shapes, finiteness, commit/crop
boundaries and positive residual support. Numerical TV remains a separate result
with no automatic losslessness verdict.

## STS follow-up boundary

This milestone implements quality32 only. The frozen fit44/eval43 groups are
reserved for a later collector/fitting workflow after quality selects and freezes
one checkpoint. Fit and eval must share that checkpoint and protocol. Compare
unscaled head, frozen STS and per-position prefix prevalence estimated **only on
fit44**, applying the same constants on eval43 for ECE/Brier. Do not substitute a
constant-zero baseline or report fitting ECE as held-out performance. All 119 dev
prompts already participated in teacher-forced monitoring: eval43 is prompt-held-out
from STS fit, not untouched model-selection data. Final test stays locked.

## CPU verification

Nine tests in `tests/test_rollout_collection.py` cover exact grouping/leakage,
seed and identity stability, rejection-tail/EOS denominators, stdlib dry-run,
source tamper detection, timeout evidence, tiny real Qwen repeatability/actual q,
bounded TV tensors, budget-one/EOS-first and injected partial probe failure.
They passed with HIP/CUDA/ROCR devices hidden, OMP threads 2. Real expanded
step128/step512 binding dry-runs passed on the same frozen manifest. No real
quality GPU rollout or fitted STS result is claimed by these checks.
