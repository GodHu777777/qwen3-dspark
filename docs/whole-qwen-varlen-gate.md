# Whole pretrained Qwen/KV gate: frozen CPU-prepared protocol

`scripts/probe_qwen_varlen.py` is prepared for a separately authorized GPU gate.
Its CPU tests use tiny BF16 Qwen models solely as fixtures. The formal CLI accepts
only the fingerprinted real dense Qwen3-0.6B architecture:28 layers, hidden1024,
intermediate3072, Hq16/Hkv8/D128, vocab151936 and default RoPE. It verifies the
model/tokenizer files against the supplied generation manifest before loading.
There is no tiny-model CLI mode, weight substitution, generation or final-test read.
Transformers5.17.0 and the explicit pinned private ROCm backend are required.

```sh
python scripts/probe_qwen_varlen.py --dry-run --model /private/Qwen3-0.6B --generation-manifest /private/generated/manifest.json
```

The default/dry-run is standard-library-only. Real `--execute --output NEW_DIR`
is not authorized by this document. It requires a fresh source archive, external
pre/post process/KFD/desktop/ASR checks and a separately coordinated GPU window.
The internal controller launches its own worker process group with a fixed300s
limit, preserves stdout/stderr and records worker exit/timeout plus separate
result states. It never retries or changes a backend. It requires8 GiB free and
caps process allocation at6 GiB before loading three copies of the actual target.

## Fixed forwards and synthetic tokens

All exact token sequences are literals in `PROTOCOL`, bound by its hash. Requests
use A100..121, B200..230 and C300..313, all within the real vocabulary. A and B
continue their fixed token cursors after crop; reused C gets token313. The active
B poison branch uses the predeclared alternate sequence500..507. These are
synthetic attention/cache fixtures, not a model-quality evaluation.

| Normal stage | Operation and appended query lengths | Resulting active K lengths |
| --- | --- | --- |
| prime | A16/B21/C13 | 16/21/13 |
| cached_tail | A1/B8; C13 inactive | 17/29 |
| crop_exit_readd | Crop A8/B3, remove/re-add C with fresh marker, append A3/B2/C1 | 11/5/1 |
| crop_zero | Crop A0, append A2; B5/C1 inactive | 2 |

The native packed target, separate dense packed model and independent per-request
SDPA model follow this exact normal schedule. Model instances are not shared
across these three paths; the independent requests share only their immutable
SDPA model weights, with separate caches. Each native packed append requires one
whole-model forward and one native flash attention operator per actual Qwen layer.

Two additional control/poison pairs restore the exact native post-prime snapshot:
append A1 with inactive C KV poisoned K=100/V=−100, and append A1/B8 with B history
KV poisoned plus B incoming tokens replaced by500..507. Query shape/order for
control and poison are unchanged. There are8 native model forwards in total
(4 normal+4 paired branches), hence224 native layer calls for the28-layer model;
112 normal calls receive mandatory same-QKV numerical checks. Dense packed has4
normal forwards; per-request SDPA has9 normal forwards. No adaptive case selection,
additional native retry or poisoned-amplitude numerical acceptance rule is used.

## Three independent result states

### 1. Structural/content status

Exact checks cover actual model `input_ids` and RoPE `position_ids`, request
markers, local positions, spans/lengths and independent gather mapping. Per-layer
QKV must be finite and gathered K/V must equal the physical request-local slices
selected independently from request-marker metadata. Using the layout's own
indices as its only oracle would be insufficient; the script also reconstructs
them from the expected request order and observed physical markers.

Every append preserves the previous request KV prefix byte-exactly; inactive
requests remain unchanged. Crop is checked against snapshots of actual per-layer
KV and the exact marker/position subset. Removal eliminates old-marker rows;
re-add creates a fresh marker at local position0. Initial uninitialized cache
layers represent an empty prefix, not a layer-count failure.

For each poison pair, A's actual per-layer Q/K/V, final normalized hidden,
logits and all per-layer cached K/V must be byte-identical between control and
poison. B/C artificial amplitudes do not invoke the normal numerical threshold.
The proof includes saved private A attention inputs and output/cache tensors.
Any failed structural assertion stops with partial evidence and failed/incomplete
states; no later shape-only check can turn it into a pass.

### 2. Normal-layer attention numerical status

Every normal attention callback compares its **actual BF16 Q/K/V and native
output** against an independent FP32 MATH SDPA oracle with the explicit
bottom-right mask and actual model scale. No sequential model's different Q/K/V
is substituted for this local mathematical reference.

The fixed original limits apply to every layer, both pooled over requests and to
each request separately: all elements satisfy `abs(error) <= .02 + .02*abs(ref)`
and RMS≤.005. Reductions use FP64 to preserve serializable errors even for very
large finite BF16 failures. Shapes, finite status, per-request errors and exact
gather checks are recorded. A numerical mismatch saves Q/K/V, output, cumulative
lengths and the FP32 oracle before proceeding. The finite matrix continues so all
normal layers/stages can be measured; the final layer gate remains failed if any
required comparison failed. Nonfinite/shape/operator exceptions stop execution
and preserve the failing inputs and previously completed observations.

No thresholds are inferred from the measured whole-model result. A passing
small-tensor backend gate does not waive this normal-layer gate.

### 3. End-to-end numerical comparison status

For all67 normal query rows, per request/stage, the script reports final hidden,
selected features at layers1/7/14/21/26, per-layer cached K/V, and actual pretrained
LM-head logit differences against both dense packed and independent per-request
SDPA. It records max absolute/RMS/equality counts, argmax changes and per-row
unfiltered temperature1 float64 TV/max probability difference using the existing
probability policy. All comparison tensors are saved privately before metrics
are computed, so a metric failure retains its input evidence.

These endpoint comparisons quantify BF16 path differences; they do not acquire a
post-hoc pass threshold. `numerical_comparison_status=completed` means the required
comparisons were recorded, not numerical equivalence, output losslessness or speed.

## Completion, identity and evidence

The report separates `structural_status`, `layer_attention_numerical_status` and
`numerical_comparison_status`. A fully measured finite layer mismatch yields
`completed_with_failed_layer_attention_gate` and nonzero worker exit. The
controller can record that execution completed while the mandatory gate failed;
it never asserts a blanket system pass. `system_pass_claimed` remains false.

Model/generation file hashes, architecture, complete protocol and script/package
source hashes are bound before worker launch and checked before model load.
The source identity is checked again after execution. Atomic result updates occur
throughout layer/stage checking. Private artifacts include failed layer tensors,
all normal endpoint comparison tensors, poison-pair A tensors and failure cache
state. Tensor names identify synthetic request/stage/layer only; all payloads
remain private by default. No full-vocabulary GPU traces accumulate across steps;
probability differences are computed one output row at a time.

CPU validation currently covers guarded stdlib binding and tiny-model refusal,
weight tamper refusal, owned-group timeout, complete tiny BF16 normal/poison flow,
fixed-limit numerical failure retention, independent gather corruption, actual
model position corruption and a second-layer backend exception. These tests do
not execute the private GPU backend or replace the real pretrained gate.
