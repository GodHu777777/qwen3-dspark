# DSpark Qwen training lab

Scope: Qwen3-0.6B data regeneration, bounded DSpark training, audited stochastic
reference decoding and CPU calibration. The full serving/scheduling system remains
incomplete. This directory is an independent Git
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
sampling.py now supplies an independent CPU stochastic probability reference,
with exact rational distribution oracles (11 tests). tensor_sampling.py and
cached_sampling.py now add a separate float64-law tensor/cached reference with
real draft Markov sampling, actual q and raw confidence retention, fixed admission,
strict cache commit/crop and a target-only baseline. The model forward dtype is
unchanged. 76 CPU tests pass in the integration snapshot. The old pilot aligned128
real GPU gate completed 9 runs/6 repeat checks, while 47/48 same-prefix rows had
nonzero TV (max 0.0416194062); see reports/stochastic-gate-20261009. Same seed across different algorithms does not imply
identical sampled tokens; no performance or calibrated scheduling claim. See
docs/cached-stochastic.md and docs/stochastic-sampling.md. scheduler.py is a pure CPU global prefix
planner; hardware profiling, multi-request execution and asynchronous scheduling
remain unfinished (docs/dspark-reproduction-scope.md).
calibration.py implements CPU sequential temperature scaling (8 algorithm tests).
calibrate_rollout.py adds a strict stdlib fit44/eval43 CLI with completed-worker
evidence, same-checkpoint/data/protocol/source/runtime checks, disjoint prompt
identities, frozen artifact integrity, and fit-only prefix-prevalence constants.
It compares unscaled/STS/constants with 20-bin ECE/Brier and coverage. Explicit
fit/eval collector groups require configs/sts-step1280-selection.json; quality
remains the default. The frozen034064b fit44/eval43 collections and default-grid
CPU STS fit/eval completed; results are mixed rather than universal improvement.
See reports/sts-step1280-20261009 and docs/confidence-calibration.md.
packed_target.py executes variable-length request chunks in one Qwen forward,
with per-request positions, marker-based causal isolation and independent KV
crop/removal. Five CPU tests check actual KV contents and lifecycle against
independent targets. It uses a dense Q*K mask, with explicit work-domain counts;
this dense oracle is not an efficient varlen kernel. The separate packed_sampling
module now supplies a CPU-validated multi-request decoder; serving remains incomplete.
See docs/packed-target.md. Native target GPU evidence is reported separately
below; the integrated trained packed decoder now has a bounded native GPU result
(see reports/packed-decoder-gate-20261009), with endpoint differences preserved.
CachedTarget has six CPU FP32 tests and a real same-dtype HF cached greedy
check (117 tokens on seven pilot prompts). Reports/cached-target-20261008 records
a BF16 full-recompute/cached near-tie divergence and isolated target costs.
Cached speculative block vs sequential BF16 equality is still unresolved.
Do not claim bit-identical cross-kernel behavior or speculative speedup.

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

Experiment logging: maintain docs/lab-notebook.md with problems, hypotheses,
methods, failed attempts, observed results, limits and next decisions. Link source
commits/configurations and aggregate evidence; do not reconstruct undocumented
history as fact. User requested a dedicated 6.1 Sol recording role. The existing
sol_data agent currently fulfills that dedicated role and maintains the notebook
at each experiment milestone; attempts to create or restore a separate
experiment_journal agent were rejected by the system agent thread limit. Reuse
sol_data for follow-ups while that limit persists; do not describe a separate
experiment_journal as the current owner. Root reviews and commits the evidence.
Astra owns correctness/training/result review and periodic direction review.

Checkpoint integrity update (96b8fd3): new checkpoints hash both trainable weights
and optimizer/RNG state, and check step consistency before loading parameters.
Legacy checkpoints without resume checksums support weights-only loading; do not
add a fabricated checksum to bypass the legacy resume refusal.

2026-10-08 bounded learning diagnostic: reports/diagnostic-aligned-20261008
records one greedy32 trajectory with all31 rollout anchor positions; at steps
64/96/128, 4 rounds each accept7, yielding exactly32 target tokens. This verifies
learnability on seen prefixes only. The earlier sampled/spaced diagnostic had
no complete rollout-prefix overlap and is preserved separately. Use
diagnose_learning --trajectory target-greedy --anchor-coverage contiguous for
the aligned protocol. Execution source hashes and historical script are retained.

Expanded data: scripts/data_pipeline.py centralizes normalized exclusions, split
identity, atomic batches and final-test export. configs/expand.example.json selects
1024train/128dev/128test, excluding all64 pilot prompts. Generation binds the
explicit source commit plus actual script/helper hashes. final-test is omitted
from development records, but permissions are a same-user convention, not a
security boundary; underlying batches contain test. Read docs/data-pipeline.md.

Cached speculative prototype: cached_decode.py maintains both caches as committed
prefix excluding the latest emitted anchor. Only committed verification features
are appended to draft per-layer projected K/V. 30 CPU tests pass; independent
42-case/429-round content audit found no rollback pollution. Real BF16 gate fails
2/3 prompts (reports/cached-decode-gate-20261008); do not advertise exact BF16
output or speedup. eval_cached_decode has FP32 control and failed-prefix path
decomposition, but rerunning that native evaluator with the new four-path
diagnostic field remains pending, as does its FP32 cached-block control.
reports/static-shape-candidate-20261008 is a separate canonical same-prefix
probe. That target-only probe also invoked the dynamic four-path helper on the
saved failures; this is not a new native draft rollout or a passing native gate.

Optional canonical numerical control: canonical_target.py uses fixed query
width, fixed StaticCache capacity and explicit causal/valid masks. Features retain
the full padded hidden tensor; predict(features,last_only) projects all padded
rows before selecting valid outputs. Ordinary CachedTarget keeps its original
requested-row projection; cached_decode uses the shared predict interface.
This is a pinned HF 5.17 StaticLayer adapter, not a default engine or CUDA graph.
39 CPU tests passed at integration. The real aligned draft gate passed 3/3
canonical comparisons (70 tokens), but only 2/3 matched stock BF16 cached output;
both dev examples accepted zero draft tokens. Preserve the native dynamic failure
report. See docs/canonical-target.md and reports/canonical-real-draft-gate-20261008.
Padding is an optional control and must not replace the paper's variable global
verification budget or hide physical work/SPS(B) boundaries. No speedup is measured.

Next training plan: configs/train-expanded.example.json and
docs/expanded-training-plan.md. Use an immutable source checkout for training,
because strict resume identity hashes every package module including unused ones.
The 1280-input expansion completed with generation/audit/runner exit 0 and an
independent structural re-audit: 932 train, 119 dev, 119 final test, 110 rejects.
See reports/data-expansion-20261009/full-*.json. Summary training_tokens includes
rendered templates across all accepted splits, not actual train-only input IDs.
The training preflight counts 424267 exact train sequence tokens, maximum 2224;
template metadata includes a trailing newline after EOS. Final test is locked.
Recheck current training/evaluation jobs before any conflicting GPU timing work.

Expanded training preflight: memory_gate.py probes non-dominated real
(sequence length, actual anchor count) shapes for two full accumulation/update
cycles, including Adam state already resident. It is an empirical resource gate,
not a formal worst-case guarantee. eval_dev_tf.py uses a fixed validation panel
and per-position denominators; experiment_inputs rejects final test, duplicate
identities and mismatched data/model fingerprints before model loading. Expanded
data CPU preflight and the real two-cycle memory gate passed; peak allocated was
5,636,591,616 bytes at the observed (2224 tokens,32 anchors) shape. Fresh training
through step32 passed identity/checkpoint/frozen-target checks. See notebook
T04/T05 for evidence and timing boundaries; recheck live state for later segments.

Bounded stochastic gate CLI: eval_stochastic_gate.py binds the original prior
train/two-dev panel and checkpoint, target/data hashes and a fresh source snapshot.
--dry-run is stdlib-only and does not import torch or touch CUDA. A real execution
uses a separate snapshot worker, native BF16 forward/float64 probability law,
same-path seeded repeats, cache/finite/EOS checks, and independently reported
same-prefix block-vs-sequential probability/logit differences. All traces are
private except reviewed scalar evidence. reports/stochastic-gate-20261009 records
execution/reproduction pass, max TV 0.0416194062, max logit difference 0.5, and one
argmax change. This uses the original pilot aligned128 checkpoint, not expanded
training weights, and does not establish distribution losslessness or speedup.
See docs/stochastic-gate.md; GPU windows still require
coordination with generation, memory gates and training.

Expanded development rollout collection: rollout_protocol.py freezes exactly
119 accepted dev prompts into quality32 (export order), fit44/eval43 (fixed salted
ID hash), checkpoint-independent per-prompt seeds, data/target/generation identity
and two preselected first-round TV probes. collect_rollout.py provides stdlib-only
dry-run and a separate bound source worker for native BF16 stochastic
collection (quality32 by default; explicit fit44/eval43 require frozen selection) (temp1, no filtering, float64 actual q,128 tokens, full proposals).
It records four distinct position denominators, rejection-tail prefix zeros,
accepted-EOS truncation, cache/finite checks and private partial evidence. Nine
CPU tests and real step128/512 binding dry-runs pass. Expanded step128 quality32
completed with worker/launcher/controller exit 0: 3,268 rounds, 581 accepted draft
tokens (0.1778/round), 3,874 output tokens, and longest accepted prefix 3. All 16
preselected same-prefix TV rows differ (max 0.04576414), with no argmax change;
see reports/expanded-quality128-20261009. This is neither a speed benchmark nor
a sequential-distribution equivalence result. Step512 quality32 completed on the exact same source/panel/protocol: 2,620 rounds,
1,299 draft tokens (0.4958/round), 3,941 outputs; 31/32 prompt ratios improved.
One accepted-EOS block excludes five verified tail labels, so effective positions
are 17,955 versus 17,960 verified. See reports/expanded-quality512-20261009.
The original1280-step training plan is complete; see reports/expanded-training-step1280-20261009.
Quality1280 also completed on unchanged a278e5a/panel/protocol: 2,270 rounds,
1,573 accepted draft tokens (0.692952/round), 3,856 outputs, 28/32 prompt ratios
improved versus512. Root froze step1280 for STS research in
configs/sts-step1280-selection.json; see reports/expanded-quality1280-20261009.
All16 numerical probe rows still have nonzero TV (max0.0927705), so this is not
a sequential-target-law or serving-speed guarantee. CPU fit/eval workflow now
completed on034064b: fit44 yielded3157blocks, eval43 yielded3280blocks. All group
OS exits and CPU fit/eval exits were0. STS ECE worsens versus unscaled at positions
1/2/5, Brier worsens at1–6; the fit-only constant has lower ECE at1–5 but higher
Brier at all7 positions. Eval tail events at6/7 number19/9. No eval-driven tuning;
retain frozen STS as the paper-method branch and both baselines. See
reports/sts-step1280-20261009 for per-position counts/coverage and independent checks. See docs/rollout-collection.md; keep final test locked and coordinate
GPU windows separately. This does not modify completed old-pilot gate evidence.

Experimental native varlen: varlen_target.py adds active-request KV gathering and
one HF native-attention callback per layer, retaining packed_target.py as a dense
oracle. Ten CPU tensor/model tests and three probe-controller tests passed.
The first real gfx1201 probe failed numerical comparison on 17841/18432 elements;
see reports/native-varlen-probe-20261009 and docs/varlen-target.md. Native output
must not be used for quality/performance claims until this correctness failure
is resolved. The failure does not affect the separate SDPA quality collector.

The independent six-call diagnostic completed on cd317f7: four public-wrapper
outputs match top-left causality, while two private no-window ATen controls
match bottom-right; GQA and repeated-KV public outputs are bit-identical. See
reports/native-varlen-diagnostic-20261009. This supports explicit-window mapping
as the cause for this pinned runtime/shape. It neither passes the original failed
gate nor establishes whole-Qwen correctness. An explicit pinned backend has a separate full-tensor protocol; whole-model/KV
validation remains required before any broader correctness claim.

Explicit private ROCm backend preparation: rocm_varlen.py adds the opt-in
rocm_aten_no_window_pinned_v1 reference kernel with exact observed runtime/schema
pinning and BF16 Hq16/Hkv8/D128/input/no-autograd guards. Public remains default;
no fallback or production-verification claim. Runtime checks are cached at
construction, layout checks handle normal/no_grad/inference_mode explicitly.
probe_varlen.py can bind this backend while preserving the original public
PROTOCOL, cases and thresholds; new raw evidence is saved before assertions.
25 CPU tests pass, including an11-call CPU substitute of the entire tensor gate.
The explicit backend passed the full three-case/11-call tensor gate on source
5281a6a, with original thresholds and exact isolation; see
reports/pinned-varlen-tensor-gate-20261009. The public failure remains unchanged.
docs/pinned-rocm-varlen.md defines actual-pretrained-Qwen/KV requirements; whole-model structural
checks and BF16 numerical reporting stay separate, with any numerical acceptance
threshold requiring a pre-execution decision. Do not edit the frozen training
snapshot or reinterpret the old public failure as a pass.

Whole-pretrained-Qwen gate: scripts/probe_qwen_varlen.py and
docs/whole-qwen-varlen-gate.md define the fixed synthetic-input, three-result-state
protocol. Eight CPU tests and real-target fingerprint dry-run passed. Root reports
the first formal 7d1fcdb run exited and released the GPU with a layer26 RMS threshold
failure; see reports/whole-qwen-varlen-gate-20261009 for its independent report. This is not a full native
correctness pass and does not affect the separate SDPA collector. Preserve the
300-second bound and original per-layer thresholds; endpoint numerical differences
remain separate from structural acceptance.

STS workflow CPU verification: 29 tests (8 algorithm,10 collection,11 workflow)
passed with CUDA/HIP/ROCR devices hidden on the existing remote runtime; real
step1280 selection-bound dry-runs completed all44 fit and43 eval bindings without
GPU execution. Private evidence: output/sts-cpu-preflight-20261009-v2.

Packed draft backbone: packed_draft.py reuses existing draft modules for one
flattened multi-request backbone with fully visible own blocks and separate
projected-context KV. Native noncausal ROCm selection is explicit; the bounded
trained GPU result is in reports/packed-decoder-gate-20261009. Ten CPU tests cover actual KV, lifecycle/isolation, batched module calls,
transaction failures and BF16 AMP boundaries. Projected K/V may have different
dtypes; preserve their representation and cast at the attention boundary, not
by downcasting trainables/RoPE. See docs/packed-draft.md.
Packed Markov sampling and multi-request verification are now connected in
packed_sampling.py (docs/packed-sampling.md). Eleven new CPU tests, plus ten draft
and five target tests, passed with GPUs hidden. One batch invokes one backbone,
batched heads, one target verification and one committed-feature projection;
actual q and original pre-token confidence remain session-owned. Prefill projects
only R final hidden rows. GPU sessions require the explicit pinned causal target
and noncausal draft backends; neither this entry guard nor CPU tests resolve the
whole-Qwen RMS failure. The bounded trained GPU gate on8378ea7 passed75 structural checks and all15
pooled/35 request draft-layer comparisons. Same-input endpoint TV is nonzero
on18/22 verification rows (max0.02938953), so the earlier target failure and
distribution limitations remain. See reports/packed-decoder-gate-20261009.
Measured capacity and actual asynchronous overlap remain unverified.

Two-step capacity CPU reference: async_capacity.py freezes absolute K from exactly
t-2 history and allocates using current calibrated prefix scores; scheduler.py's
literal synchronous early-stop algorithm remains unchanged. Full shadow drafting
runs during cold start and zero allocation. Historical search crosses all SPS
cliffs; profile/calibration/request incarnations and private confidence provenance
are bound. Output-budget marginal benefit stops at remaining-1 because the last
accepted token cannot produce a bonus. async_round.CapacityRoundDriver enforces
freeze -> propose -> bind -> verify/commit -> finish and charges synchronous host
copies, calibration and hashing. Twenty hidden-GPU tests passed, including real
three-round tiny-Qwen KV reconstruction. Profiles are synthetic; CPU ordering is
not evidence of GPU overlap, graph replay or measured SPS. See docs/async-capacity.md.

Target-only engine preparation: probe_vllm_offline.py and guard_vllm_smoke.py
provide a bounded, identity-aware offline smoke. The first e0f5ce8 run stopped
before engine initialization because distribution and module versions were
incorrectly equated; reports/vllm-offline-smoke-20261009-e0f5ce8 preserves that
failure. The03b80a2 correction pins both fields separately and completed an
actual35-prompt-token/128-output-token smoke with worker/controller OS0 and
verified release. The engine reported FULL_AND_PIECEWISE graph capture and
ROCM_ATTN with its internal Triton fallback. Replay was not independently
instrumented; this single request is not a benchmark. See
reports/vllm-offline-smoke-20261009-03b80a2. Reuse this tested
supervisor for future experiments; the already-executed packed gate used an older
external bare-PID controller and cannot establish PID-reuse-safe supervision.

CPU-prepared performance tools: `performance_workloads.py` and
`configs/performance-workloads.example.json` define shared exact synthetic
R=1/2/4, prompt64/256, output128 workloads. `benchmark_vllm_offline.py` measures
one-engine fixed batches with separate diagnostic event timing;
`guard_vllm_benchmark.py` reuses identity-aware supervision.
`profile_packed_decoder.py` separately measures all64 R2/C128 full-shadow
allocations from restored real caches/RNG. Its local round rates explicitly
exclude planner/history cost and are ineligible as full CapacityRoundDriver
profiles. Both default to CPU binding/preflight; preparation is not a GPU
benchmark result. Read docs/vllm-offline-benchmark.md and
docs/packed-performance-plan.md before execution. Do not mix output tokens/s,
local rounds/s and end-to-end serving capacity.

The first formal vLLM fixed-batch baseline completed all 54 warmup/primary/diagnostic
batches on 700bfa6 with one engine and worker/controller OS0. Pooled output
throughput is 127–131 tok/s at R1, 247–252 at R2 and 475–516 at R4 over the two
synthetic prompt lengths. All five primary repeats per cell remain public in
reports/vllm-offline-benchmark-20261009-700bfa6. Actual default ROCM_ATTN used
its internal Triton path; graph capture completed with sizes 1/2/4/8, but replay
was not independently traced. This target-only measurement is not DSpark
speedup or an arrival-load frontier. Immutable 700bfa6 also passed the complete
204-test CPU suite with GPUs hidden (output/integration-cpu-700bfa6).

The first native local cost profile completed all 64 allocations on 700bfa6:
128 warmups, 320 primary samples and 64 diagnostic samples. Primary round
median across the exact frozen R2/C128 domain was 94.978 ms. See
reports/packed-profile-20261009-700bfa6 for every sample and same-B contrasts.
Target append and proposal outside the backbone are coarse investigation targets;
nested host spans/GPU event intervals do not isolate kernel cost. The measured
local rate still excludes real planner/history and cannot become a full capacity
profile by relabeling. Growing-context prediction, matched native trajectories,
actual graph replay and scheduling overlap remain unverified.

Native matched-workload preparation: scripts/benchmark_packed_decoder.py uses
one target/draft load and a single native target adapter, resetting request
caches inside each timed batch. It follows the shared six-case/54-batch protocol
with eager full-shadow fixed maximum prefixes, preserving the original FP64
sampling law. Eight CPU tests cover the actual production factory, unequal
completion with retained inactive KV, and a round deadline preserving earlier
samples. Memory records explicitly include inherited pre-reset target KV. The
1800-second worker budget is fixed before the first run; this is not evidence
that the native end-to-end comparison has completed. Read
docs/native-offline-benchmark.md. No capacity planner, graphs or overlap are
integrated in this baseline.

Persistent target KV candidate: persistent_target_kv.py separates fixed-address
resident KV from speculative scratch and registered attention staging buffers.
Eight local CPU tests include real tiny-Qwen KV prefix commits, stale-handle
rejection, metadata-failure recovery and allocated-device alias normalization.
These first ran on Torch 2.11 / Transformers 5.4; immutable 2010503 then passed
all eight on pinned AMD Torch 2.12.0+rocm7.2 / Transformers 5.17.0 with GPUs
hidden (3.664 s, SSH OS0; output/persistent-target-kv-cpu-2010503-amd). This
storage prototype is not connected to the decoder and proves no native graph,
overlap or performance improvement. Capacity-tail native support and graph-safe
attention still require a separate probe; see docs/persistent-target-kv.md.

Native end-to-end result: immutable 0c36b03 passed the 212-test CPU suite and
completed all 54 shared batches with worker/controller/SSH OS0 and independent
resource-release checks. Root independently matched its archive to Git and
recomputed all six primary rates: 16.092, 13.928, 23.686, 24.032, 36.408 and
33.848 output tok/s in canonical R1/2/4 × C64/256 order. This is 7.06–12.26%
of the frozen vLLM baseline throughput; no speedup. See
reports/native-offline-benchmark-20261009-0c36b03. The raw 54-batch trace is
private; public batch scalars and aggregate work preserve all repeats. This is
eager full-shadow fixed-prefix execution, without real capacity scheduling,
graph replay or overlap. It compares execution stacks and does not isolate
speculation overhead; the previous target RMS failure remains unchanged.

Native capacity-tail/capture preparation: scripts/probe_native_capacity_graph.py
binds real target QKV and a fixed R2/Q(1,4) bucket at three growing contexts.
It tests exact/fixed-max/capacity attention and finite/NaN unused-tail isolation
before attempting gather+attention capture/replay. Six local CPU contract tests
pass, including V-only resident contamination and partial-failure evidence;
the CPU replay emulator is not a GPU graph. Read docs/native-capacity-graph-probe.md
before execution. Immutable f5d03d4 subsequently passed 226 CPU tests and its
one bounded GPU run completed all 15 eager variants and three real graph replays.
Independent CPU inspection of 42 saved tensor artifacts reconstructed 108 input
comparisons and 63 pooled/request output comparisons; all output comparisons
were bit-identical. Worker/controller/outer OS0 and independent release passed.
See reports/native-capacity-graph-20261009-f5d03d4. Actual graph scope is only
gather+attention at fixed Q=(1,4), with growing C and fixed K capacity. Resident
isolation and current input-pointer stability are executed assertions, not an
offline reconstruction of unsaved snapshots. Full-model graphs, selection from
t-2 capacity K, sampling integration, overlap and speed gains remain unverified.

Persistent full-target candidate: persistent_qwen_target.py reuses the complete
HF Qwen3 forward with a custom scratch-only Cache.update and request-local CPU
attention callback. Its prefill/verify/commit/abort interface separates speculative
KV from committed state; selected raw block outputs remain distinct from final
normalized features. Four local tiny-Qwen tests cover all-layer KV, features,
logits, partial commits, failed-forward recovery and inactive-request isolation.
The initial f46e63f candidate was CPU-only and required an explicit session
interface migration. The subsequent session and graph implementations are
described below; read docs/persistent-qwen-target.md for the current interface.
Immutable f46e63f also passed the complete 230-test CPU suite on pinned AMD
Torch 2.12.0+rocm7.2 / Transformers 5.17.0 with GPUs hidden (13.647 s, actual
SSH/test exit 0); all 334 archive files were verified. Evidence is under
output/persistent-qwen-pinned-cpu-20261009-f46e63f. No tested HF API incompatibility
was observed, and this CPU result makes no native or full-model graph claim.

Persistent session integration: `target_strategy.py` adds an explicit optional
strategy to `PackedSpeculativeSession`; default append/crop remains unchanged.
Finite declared ordered-Q buckets select the smallest compatible K capacity;
missing shapes fail explicitly. Persistent verification computes all sampling
decisions before one target commit, then projects committed draft features before
releasing the feature lease. Admission retains the single R-row LM-head call.
Nine real tiny-Qwen tests exercise actual q/p/RNG, EOS/budgets, lifecycle and
failure cleanup. Frozen f46e63f plus the four integration overlays passed 239
local CPU tests (235 passed, four Linux-only skips); source and log hashes were
independently verified. See docs/persistent-sampling.md and private evidence
output/persistent-sampling-cpu-20261009. This does not establish GPU graph leases,
full-model replay, overlap or speedup; the native graph implementation is separate.
Immutable 933ed88 subsequently passed all 239 tests on pinned AMD Torch 2.12.0
/ Transformers 5.17.0 with GPUs hidden (16.039 s, test/SSH exit 0, no skips).
The exact Git archive and eight evidence hashes were independently verified;
see output/persistent-sampling-pinned-cpu-20261009-933ed88. This is CPU evidence.

Same-backend target-only control: `packed_target_sampling.PackedTargetOnlySession`
uses the explicit target strategy with one anchor query and original FP64
categorical draw per active request, without draft/shadow work. Admission projects
only R final hidden rows. Eight real tiny-Qwen CPU tests compare actual probability
rows, RNG traces, outputs and all-layer KV against the independent cached target
reference, including EOS, budgets, inactive requests and cleanup failures.
See docs/packed-target-only.md and output/packed-target-only-cpu-20261009.
This is preparation for paired E2E attribution, not a performance result or a
replacement for the strong vLLM baseline. A subsequent ninth test exercises two
real target-only session rounds with the CPU replay emulator, matching actual
probabilities, RNG, outputs and all-layer KV; it is not GPU graph evidence.

Full-target graph implementation: `persistent_qwen_graph.py` captures the original
HF embedding, all decoder layers (including QKV/RoPE/MLP), final norm, scratch KV
and selected raw feature copies; LM head, sampling and commit stay outside.
Native eager and graph backends are explicit and pinned. CPU replay is separately
named and is not GPU evidence. Review found and repaired shared-backend cross-pool
event confusion, missing poison on cancellation-check exceptions, and undercounted
private graph-pool memory. Events now bind exact receipts and registered programs;
retained memory charges private-pool segment sizes, including inactive blocks.
Feature leases survive commit until same-stream consumers finish enqueueing reads.
Finite bucket counts and byte reservations are enforced without capture-on-miss.
Frozen 21ce2bb plus seven overlays passed 260 local CPU tests (256 passed, four
Linux-only skips), including real speculative-session/emulator composition.
Evidence: output/persistent-qwen-graph-cpu-20261009/repair and the independent
review under output/persistent-qwen-graph-review-20261009. Independent focused
repair checks reject the original event/cancellation counterexamples and verify
inactive private-pool accounting; no blocker remains in that CPU review scope.
Pinned Torch 2.12 API
inspection confirms pool-scoped memory_snapshot exists, but actual GPU pool
accounting, full-model capture/replay and speed remain unverified. See
docs/persistent-qwen-graph.md. Exact ordered-Q buckets are a bounded first step;
general finite physical-B families, calibrated capacity and overlap remain open.
Immutable c2d11a0 then passed all 261 tests in the pinned AMD CPU environment
(17.615 s, no skips, test/SSH exit 0, three GPU visibility variables empty).
All 343 source files were checked before/after; root independently matched its
Git archive and eight evidence hashes. See
output/persistent-qwen-graph-pinned-cpu-20261009-c2d11a0. This includes both
speculative and target-only emulator composition, not actual GPU graph execution.
