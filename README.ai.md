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
history as fact. User requested a dedicated 6.1 Sol recording role. The dedicated
experiment_journal agent owns the notebook at experiment milestones when available.
Earlier agent-limit failures required sol_data to cover this role; on 2026-10-09
experiment_journal was restored and sol_data handed over sole editing ownership
without pending notebook edits. For S32, waking experiment_journal failed again
with the agent thread limit, so root assigned sol_data sole temporary notebook
editing ownership. Coordinate handoffs before changing the writer. Root reviews
and commits the evidence.
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

Full-target GPU probe preparation: scripts/probe_full_target_graph.py runs the
complete pretrained native eager phase before attempting one full-HF capture and
three growing-context replays at ordered Q=(1,4). It compares selected raw layers,
final norm, logits and all-layer KV at fixed limits, checks partial commits/abort
and inactive isolation, and requires unchanged Python model/layer counters during
real replay. Scoped private-pool bytes and raw failure evidence are retained.
The identity-aware supervisor enforces one 300-second window and preserves ASR.
Eight targeted CPU tests passed; this preparation has no GPU or speed result.
Read docs/full-target-graph-probe.md before execution. The same-native eager
reference is not a stock attention oracle and cannot overturn the earlier
whole-Qwen cross-backend numerical failure. No additional full core suite was
run for these three additive files; immutable c2d11a0's 261-test result remains
the core baseline. Private preparation evidence:
output/full-target-graph-preparation-20261009.

Full-target GPU probe result (execution source 3829d53): three native eager states
and three real full-target graph replays passed for ordered Q=(1,4), including
partial commit, abort and inactive isolation. Replay left model/decoder Python
counters unchanged. The private graph pool retained 2 MiB with no active tensor
bytes, confirming why allocated/global deltas cannot substitute for pool accounting.
Worker/controller/SSH exited 0 and release was independently checked. Complete
archive recovery and root's CPU re-audit verified 30 tensor artifacts and 1,239
numerical comparisons (all torch.equal and storage-byte equal; frozen limits
unchanged). See reports/full-target-graph-20261009-3829d53 and notebook S30.
This supersedes the preparation-only status for this finite same-backend family;
cross-backend RMS/TV failures, E2E performance, capacity and overlap remain open.

Paired R1 attribution preparation: scripts/benchmark_paired_r1.py compares the
same persistent target-only stack with trained fixed-gamma7 full-shadow decoding
on the original C64/C256 requests, 128 output tokens, 36 ordered batches. Both
arms share resident target/draft weights and Q1/Q8 graphs; Q2..7 tails explicitly
run native eager inside the timer. Startup validates the new graph shapes at
unchanged numerical limits and preserves failure tensors. Original FP64 sampling,
the frozen vLLM baseline and prior numerical limitations remain unchanged.
Eleven focused CPU tests and independent host/sampling composition checks passed;
these are not GPU throughput evidence. Read docs/paired-r1-benchmark.md for the
fixed 1800-second supervision and memory bounds before execution. This two-case
attribution experiment does not complete multi-request capacity or overlap.

Paired R1 GPU result (immutable 192a357): all 36 batches completed, worker,
controller and SSH exited 0, and independent release preserved the ASR service.
Root verified all 4,059 rounds, original source archive/352 files, complete
evidence archive, output budgets, graph coverage and pooled rates. At C64/C256,
target-only achieved 44.8049/44.1817 tok/s versus speculative 23.1037/19.2743;
the latter is 51.6%/43.6% of the same-stack control and 17.6%/15.1% of vLLM.
Speculative primary graph coverage was 435/455 and 500/530 rounds; explicit eager
tails remain charged. Accepted draft tokens/round were 0.3956/0.1981 on these
synthetic inputs, not the natural quality32 population. See
reports/paired-r1-benchmark-20261009-192a357 and notebook S31. Q1/Q8 raw startup
artifacts independently matched native eager at the frozen limits. This resolves
the missing same-stack attribution measurement, not per-operation bottleneck
attribution, numerical-law equivalence, multi-request capacity or actual overlap.

Paired diagnostic profile (immutable f0b0268): the diagnostic-only observer and
independent pidfd resource monitor passed 15 pinned Linux CPU tests, including
native GIL-held sampling/termination. Independent review closed invalid trace-ID
attribution and GIL-blocked thread-monitor defects; see notebook S32. The single
GPU run then preserved 32 samples (8 warmup, 20 primary, 4 event diagnostics) but
failed exporting its first full-session trace: observed 282942823 bytes exceeded
the frozen 268435456-byte limit; final partial bytes were 298398246 after sampled
overshoot. Worker -9, controller/SSH 1, no timeout or retry; independent release
preserved ASR. Post-run source hashes and controller input-integrity checks passed;
the killed worker did not write its post-input check. Primary rates were
45.0407/44.6040 target-only and 24.3167/20.7376 speculative tok/s for C64/C256.
See reports/paired-profile-20261009-f0b0268. Event intervals include dispatch gaps
and host waits overlap earlier GPU work; neither proves kernel-active attribution.
The full CPU/runtime/GPU timeline remains unavailable, and historical vLLM,
quality/calibration and numerical limitations remain unchanged.

Model-signature CPU investigation: scripts/benchmark_model_signature_cpu.py
measures the original signature and isolated components on a same-size BF16 meta
Qwen model, loading config only. One focused equivalence test and a single pinned
AMD CPU window passed; no weights, forward or GPU initialization were used.
See reports/model-signature-cpu-20261009. AMD original/candidate medians were
1601.134/1560.543 microseconds with broadly overlapping repeat ranges. Keep the
production implementation unchanged: component timings are not additive and do
not establish end-to-end gains. Independent CPU counterexamples also reject
cross-boundary signature caching: public model aliases can mutate before prepare,
submit, finish or during a synchronous wait. Preserve all three current checks;
this does not imply detection of writes that already bypass version tracking.
Return next to finite physical-B families and multi-request current-confidence
admission, retaining prior capacity/RNG ordering, actual work counts and explicit
eager fallback costs. No new lease framework or GPU profiling retry is required
by these CPU findings.

Finite physical-B CPU foundation: `persistent_target_kv.QueryFamily` separates
fixed R/B/maxQ/Kcapacity/maxK from transaction actual ordered Q. One explicit
`register_families` call allocates shared metadata/gather backing with stable
per-family views. Exact `Bucket` remains compatible; graph I/O and private-pool
reservations remain separately charged per captured program. The real CPU
`PackedSpeculativeSession` composition matches exact eager q/p, RNG, outputs and
target/draft KV while varying actual Q and B, with graph-emulator and explicit
eager families. See docs/query-families.md and output/query-family-cpu-20261009.
The later CPU bridge below establishes synchronous t-2/current-confidence
composition; generalized native family capture, speed gain and overlap remain open.
Keep all three fresh model-signature checks and receipt/feature lease boundaries; never pad actual physical B to a
reservation or infer native support from the emulator or launch-argument spy.

Full-shadow capacity/family bridge: five real tiny-Qwen CPU tests now compose
`CapacityRoundDriver` with persistent `QueryFamily` selection, using a full-shadow
exact-eager control rather than fixed-budget draws. Synthetic B5 and rising-zero
profiles exercise nonuniform Q, cold starts, historical t-2, removal/reincarnation,
remaining1 and K8 reservation with actual B2. q/p, decisions, RNG, outputs and both
KV caches match; committed caches also match a separate full-prefix HF reference
within declared tolerances. Concrete driver profiles must map physical B to actual
B, checked at construction and step entry; pure planner mappings stay general.
Current-score-dependent missing family/provenance rejection happens after proposal
RNG but before target/verification RNG/commit, and the driver cannot retry. See
docs/query-families.md and output/capacity-query-family-cpu-20261009. This is CPU
correctness evidence with synthetic SPS and untrained heads, not hardware-aware
performance or confidence quality. No remote deployment or GPU run was performed.


Variable-Q native gate preparation: `scripts/probe_query_family_graph.py` is an
additive bounded protocol; the old fixed-Q probe remains unchanged. Six states
use R2 plus one inactive resident, shared B3/B5/E2 families, changed per-request Q
and one reversed roster. All six same-family native eager states must pass before
two captures; the mixed phase contains five graph replays and one explicit eager
state. Metadata, raw selected layers/final norm/logits, all-layer scratch and
committed KV, exact own commit/abort/isolation, lease use and Python counters are
checked. Original numerical thresholds stay fixed. Two independent 512 MiB graph
reservations, 1280 MiB combined budget, 64 MiB workspace cap, 6 GiB allocator cap
and 300-second worker limit are explicit. Raw tensors are saved before assertions.
Read docs/query-family-graph-probe.md; CPU preparation is not native support,
measured SPS, sampled acceptance or speed evidence. A device run requires reviewed
immutable source, fresh live resource checks and coordinated execution.


Variable-Q native result (immutable a4942af): after 18 targeted pinned AMD CPU
tests passed with GPUs hidden (3.628 s, exit 0), the single device run completed
all six native eager states, two captures, five actual graph replays and one
explicit eager state. Every replay left original model/28 decoder Python counters
unchanged. Worker/controller/SSH exited 0; independent identity/KFD/ASR release
and all 375 source hashes passed. See reports/query-family-graph-20261009-a4942af.
The independent CPU raw audit passed 5,584 checks and 3,612 numerical reductions
over 65 tensor artifacts (no missing/failed checks); max absolute error and RMS
were 0. Byte equality is descriptive under the original limits. Distinct private
pools retained 2 MiB each; workspace 6,967,476 bytes includes shared arena
3,605,722, with graph reservations plus separate I/O 1,073,840,192 bytes. This
proves only the declared same-native R2 finite family domain; it does not execute
trained draft/confidence/capacity scheduling, measure SPS/speed or demonstrate
overlap. Prior numerical-law limits, same-stack slowdown and separate vLLM
baseline remain unchanged. Next cost measurements must charge actual shapes and
the full end-to-end round, including eager and host work.


R2 complete-session benchmark preparation: `scripts/benchmark_paired_r2.py` adds
a single original R2/C256 workload with target-only, fixed-gamma7 full shadow and
zero-admission full shadow arms. It reuses optional R1 runner interfaces; original
R1 defaults and observer entry points remain intact. See docs/paired-r2-benchmark.md.
Declare R2/B2..16 and R1/B1..8 families; capture only R2/B2 and R2/B16. All other
verification work, including R1 tails, remains explicitly eager and timed. Validate
both hot graphs at initial C256/256 and committed C368/375, with pooled and
per-request raw-layer, norm, logits and all-layer scratch-KV limits unchanged.
The 27 complete batches use original seeds and 128 outputs per request.
The zero arm consumes proposal RNG and measures whole-path overhead, not isolated
draft latency. Fixed draft cost cannot be cancelled when maximizing E/(D+V+H).
No new Profile/SPS, confidence admission, training or overlap is introduced.
Related local CPU tests passed 32 with two Linux-only skips; device execution and
performance remain pending. Deploy only reviewed immutable source with fresh guards.


R2 actual result (immutable 17da0b4): one bounded native run completed 27 batches,
following all 34 pinned AMD GPU-hidden CPU tests (26.201 s, exit 0). See
reports/paired-r2-benchmark-20261009-17da0b4. Primary output tokens/s: target-only
80.666075, gamma7 full shadow 34.986799, zero-admission full shadow 31.565220;
gamma7/native target-only=0.433724. Separate strong vLLM R2/C256=246.566727.
Gamma7 accepted 220/7080 selected proposals, 0.415094/global round or
0.209524/request-round, on this synthetic workload; 7350 shadow positions,
490 graph/40 eager primary rounds including 10 R1 tails. Zero selected no
proposals but generated 8890 shadow positions; accepted/selected is null.
The zero arm follows a different RNG trajectory; its gap is whole-path overhead.
Independent scalar audit checked all 3240 rounds and 380 source hashes; CPU raw
audit passed 2203 checks and 756 reductions with max absolute error/RMS zero
at initial and grown contexts under unchanged thresholds. Prefill/growth token arrays
were not separately saved; the audit does not independently replay prefix execution. Worker/controller/SSH
exited 0 without timeout/retry; independent live release preserved ASR.
No measured SPS/Profile, learned confidence admission, capacity policy or overlap
was exercised, and no inference speedup has been demonstrated.
Next decision: defer capacity/SPS expansion and prepare a bounded R2 coarse
critical-path diagnostic (two arms, six complete batches, 300-second window).
Preserve plain controls, output/RNG/work invariants and original primary rates;
measure observer perturbation. No Kineto trace, retry or new speed claim. This
is a proposed follow-up, not implemented or executed by the R2 result commit.


R2 coarse diagnostic development: `scripts/profile_paired_r2.py` is an additive
observer of the original complete-session runner. Its fixed six-batch plan uses
target-only and full-shadow gamma7, each with warmup, plain and coarse-sync calls.
Outer actual-call intervals are nonoverlapping: shadow includes its internal
heads/probability work, and prefill includes its internal commit. Record pre-fence
drain separately from body plus post-fence service, retain unassigned remainder,
and keep signature spans nested rather than adding them to parent totals.
Timer-external semantic evidence must compare actual outputs, final RNG state and
complete work/decisions between plain and observed calls. The diagnostic is
bounded to 300 seconds with original GPU limits and no trace export. See
docs/paired-r2-profile.md. Preparation/testing and native results must be recorded
separately; no primary speedup can be obtained by subtracting diagnostic spans.


R2 coarse diagnostic result (immutable ab4e3a5): pinned AMD GPU-hidden40 CPU
tests passed (34.599 s, exit0), then the single300-second-bounded native run
completed all6 batches. See reports/paired-r2-coarse-20261009-ab4e3a5. Plain/coarse
wall seconds: target3.165032/3.216314, gamma7 7.350081/7.478367; one pair per arm
is insufficient to isolate observer overhead from run variation. Actual output
tokens, final RNG bytes and complete round work/decisions match exactly. Root's
independent audit checked699 rounds,2087 disjoint stage records and386 source
files. Setup raw audit passed2203 checks/756 reductions with maxabs/RMS0 under
unchanged limits; prefill/growth prefix execution is not independently replayed.
Gamma7 synchronized services: full shadow3.412737s, target2.590843s, outside-shadow
FP64 law/sampling0.834723s, commit/projection/release0.518412s. Pre-drains0.015334s
and unassigned remainder0.106318s complete the7.478367s observed batch. Signature
host spans (target381/.758828s, gamma294/.607391s) remain nested inside target
service; retain all three fresh checks and the previous unsafe-caching veto.
Worker/controller/SSH and independent release passed, preserving ASR, without
timeout/retry/trace export. Full shadow is the largest observed gamma category
but its inner cause and kernel-active/CPU utilization are still unresolved.
No optimization, new primary rate, SPS, learned admission or scheduler benefit
was established; preserve the S40 same-backend control and separate vLLM rates.
The accepted next investigation is a bounded full-shadow inner-stage diagnostic:
separate backbone, base head, Markov/confidence, FP64 law/draw and issuance/copy
remainder with complete semantic controls and reversed plain/observed pairs.
The proposed hypothesis is FP64 law/draw accounting for at least50% of propose
on both orders; this is not yet a result or a demonstrated optimization.


Full-shadow inner-service preparation (notebook S43):
`scripts/profile_shadow_inner_r2.py` adds a fixed six-batch diagnostic; see
`docs/shadow-inner-r2-profile.md`. Within each owning propose, disjoint child
services, prior drains and residual close the parent service; never add parent
and child ledgers or nested signatures. Runtime guards prove at most127 rounds
and127×46=5842 R2 child records under the original8192 cap. Reversed plain/inner
pairs use raw token/RNG/work identity and the predeclared three-state rule:
both fractions at least0.50 supported, both below not supported, mixed sides or
any wall ratio outside[0.95,1.05] inconclusive. Semantic/coverage failure aborts.
The5% band is diagnostic, not a confidence interval or a speedup estimate.
Local first six tests passed. The initial47-test run failed only because a new
test incorrectly required completed committed KV to be cleared. Correcting that
test to persisted final lengths left runtime/doc bytes unchanged; frozen v2
passed45 with two existing Linux-only skips (11.549 s, OS0). Independent source,
classification, failure-cleanup and alias review approved the393-file snapshot.
This is CPU preparation only; native results remain pending. Root deploys an
exact source commit archive and coordinates the single bounded GPU run. Preserve
original laws/checks, budgets and prior primary/vLLM results; no training, new
Profile, optimization or stage-subtraction speedup is established.

Full-shadow inner-service result (notebook S44, immutable db9d61c):
the pinned AMD GPU-hidden suite passed all47 tests (42.099 s; wrapper44.1717 s,
OS0), and the single native run completed6 batches/657 rounds/9720 child records.
Four non-warmup batches preserve actual tokens, RNG bytes and full round work
exactly. Reversed observed/plain ratios are1.104047 and1.135000, both above the
predeclared1.05 ceiling; FP64 law+draw/propose fractions are.416480 and.411749.
The formal classification is inconclusive, not not_supported or an identified
unperturbed bottleneck. See reports/shadow-inner-r2-20261009-db9d61c.
Parent service equals disjoint child services, child pre-drains and residual;
never add parent and children, and retain all three fresh signature checks.
Root scalar and raw setup audits passed (2203 checks/756 reductions, maxabs/RMS0);
prefill/growth prefix execution was not independently replayed. Execution,
collection and independent release passed without timeout/retry/trace.
Independent Direction result review passed; root accepted stopping the profiler
chain. Next prepare only a CPU-gated fixed-shape categorical fallback candidate,
preserving zero tails/CDF overflow and mutable-RNG no-support/shape exceptions.
After CPU equivalence and implementation review, propose20 observer-free complete
target/gamma A/B batches under the original300 s and memory bounds. Proposed
screening requires EACH arm pooled B/A throughput>=1.02 and at least3/4 pairs
faster; otherwise retain the original. Freeze this rule before execution.
This is a future direction, not an implementation, frozen protocol or A/B result.
Preserve S40 native primary and separate vLLM results and prior numerical/law/TV
limits; this perturbed diagnostic establishes no optimization or speedup.


Categorical fallback CPU gate (notebook S45):
`dspark_qwen/experimental_categorical.py` is opt-in; production sampling remains
unchanged as the exact oracle. Only callback-after nonempty1D fallback uses a
fixed-shape integer maximum; metadata/no-support cases retain original nonzero
behavior and errors. FP64 CDF/right-searchsorted, one draw and validation order
remain intact. The scalar positivity guard can synchronize; no speed is inferred.
First6-test run had one incorrect remaining1 coverage assertion; adding an explicit
budget2 real shadow session changed only tests. Revised7 passed, then frozen402-file
base ba87fc4 plus3 additive files passed71 tests in4.787 s/OS0; candidate-bound
existing Fraction oracle passed5 in.118 s. Independent review passed1874 oracle
pairs,8 mutable-RNG cases and151936 strided-vocab fallback. Real tiny-Qwen CPU
sessions preserve per-round laws/confidence, populated KV, tokens/RNG/work exactly.
Core may now prepare a separately reviewed/frozen20-batch observer-free R2 A/B
runner; no native or performance evidence exists yet. Keep production defaults,
original budgets and both-arm screening; retain S44 inconclusive and prior primary,
vLLM and numerical-law limits. See docs/categorical-fixed-shape.md.


Observer-free categorical A/B runner preparation (same S45):
`scripts/benchmark_categorical_ab_r2.py` now freezes20 original-R2 complete batches
(4 warmups+16 primary), with direct timer-external sampler alias binding/restoration,
no observer/stage fence/signature wrapper and unchanged fresh checks/session sync.
First6 runner tests exposed two injected-extra-field rejection failures; union-key
semantic comparison fixes added/removed fields (existing nested RNG was compared).
The13-case revision passed6; final15-case native-gate design adds151936 dense and
strided zero-tail cases with private actual A/B tensors saved/hashed. Frozen77 CPU
tests passed in8.749 s/OS0, with404 source files unchanged before/after.
Protocol SHA30dea383c399f51bc46d233510d81d3e64700ad73297465d4f68e557f85e3608
predeclares EACH arm pooled1024 outputs/summed4 walls B/A>=1.02 and at least3/4
strictly faster pairs. Complete semantic/native gates precede any performance
result; valid misses retain A, partials do not pool. Original300/290 s and memory
limits, production default and prior benchmark/numerical limits remain intact.
CPU preparation is not native execution or a measured gain; use an exact committed
source archive, excluding recorder working-tree docs, for any reviewed deployment.

Independent runner review approved one bounded native attempt subject to root live
launch guards; large-only gate failure and pooled-ratio/favorable-pair counterexamples
passed on CPU. Root committed/pushed the five tested overlays as c0d55d2 and began
exact-archive preparation/GPU-hidden CPU deployment; actual device results are pending.


Categorical native A/B result (S46, immutable c0d55d2):
all20 observer-free batches completed (2330 rounds,5120 outputs). AMD GPU-hidden77
CPU tests passed (33.472 s log;41.4752 s wrapper). Actual-device15 categorical cases,
independent retained2 large-tensor gate audit, scalar audit and startup raw audit
(2203 checks/756 reductions, maxabs/RMS0) passed. All10 same-arm batches including
warmups preserve actual tokens/RNG/work exactly; prior cross-backend law limits remain.
Target A/B pooled rates were81.261790/80.510255 tok/s (B/A.990752,2/4 B wins);
gamma35.170647/34.495830 (B/A.980813,1/4 wins). BOTH arms miss the frozen>=1.02
and>=3/4 practical screen. Retain the original default and close this candidate
branch; no tuning/retry, theoretical-structure speed claim or statistical significance.
See reports/categorical-ab-r2-20261009-c0d55d2. Native/controller/collection/audits
and independent release exited0;142.1928 s controller elapsed is whole-run time,
not throughput. Independent Direction result review passed and confirmed closing
the candidate branch while retaining A; root accepted the next read-only CPU task.
Historical primary/vLLM and numerical-law/TV bounds remain. No production integration
or useful E2E gain shown.

Next only verify/recover existing quality32 step128/512/1280 round artifacts and
reanalyze first selected p/q and min(1,p/q), with fixed output-progress bins
0-31/32-63/64-95/96-127 and all32 prompts/EOS/budget/missing coverage. Missing raw
or provenance means a recorded gap, not replacement generation/training/GPU/final-test.
Existing natural1280 first-prefix843/2270=.371366 and26/32 below.5 describe broad
early rejection only; do not equate TF.373430 or transfer synthetic3.1% acceptance.
This accepted direction has not been executed; selected-token risk is not full-law
overlap, and different checkpoint trajectories are not matched-prefix causal controls.


Existing quality32 first-risk CPU reanalysis (S47):
`scripts/analyze_natural_first_risk.py` is a stdlib-only reader of recovered historical
step128/512/1280 artifacts. All8158 first selected p/q pairs were valid and all32
ordinals retained per step. Pooled alpha=min(1,p0/q0) was.158279/.308461/.372736;
equal-prompt alpha.155778/.324532/.397361. Step1280 has26/32 prompt means below.5
and all four fixed progress-bin pooled means below.40;25 ordinals remain below.5
across all checkpoints. 512→1280 alpha changed26 up/6 down; observed first-prefix
changed27 up/4 down/1 tie (correcting an earlier oral5-down report).
See reports/natural-first-risk-20261009. Five frozen tests passed (.661 s, OS0);
one actual analysis exited0 with36 input hashes unchanged. Independent98691 raw
assertions and110698 per-value checks/8158 rows passed; aggregates were shared
before core finished, so no fully blinded double-analysis claim. No analysis/test
failure occurred; an overbroad publication term check was fixed without changing
analysis content. Current recovered raw hashes are not historical precommit hashes;
original a278e5a source/binding/panel/aggregate/exit provenance reconciles.
These are distinct sampled trajectories, not matched-prefix causal effects. Selected
alpha is not full-vocabulary overlap or a calibration result; no data-mismatch/
exposure-bias attribution, independent-round CI, new rollout or speed claim follows.
Root accepted only designing a bounded matched-prefix quality/objective diagnostic;
no new execution/training/generation/GPU/final-test or policy search is authorized by
these results. Keep original sampler and frozen step1280/STS state and prior limits.


Matched-prefix probe design only (S48): docs/matched-prefix-diagnostic.md and
reports/matched-prefix-design-20261010 record32/32 common initial states across
historical128/512/1280 collections. Atk0 both Markov previous-token paths use the
anchor; direct recorded/sampled substitution starts atk>=1, so first-position
mismatch cannot be assigned to within-block exposure bias from this source fact.
Root accepted only six existing probes: preselected ordinals0/1,round0,three steps,
first row. Fix each state's p_ref to128 sequential row0; report full-law O_ref,
O_seq/O_block and TV/reference sensitivity, preserving per-state disagreements.
Valid unequal sequential vectors are a numerical control, not an automatic gap;
missing/broken identity/payload/law/gather is a gap, with no replacement collection.
Remote existence/bytes/hashes remain unverified; estimated197MB is schema-based,
not a transferred payload. No probe transfer/load, implementation, tests or model
execution ran. Two states cannot establish general acceptance/speed, dense/cache q
equality or training causality; keep original sampler and prior numerical-law limits.

Final source/design review approved this design only (document SHA327b4c0a…163355b),
with no new analyzer/runner/tests or probe execution. See the public independent
review; source byte alignment does not establish numerical equality. Overlap uses
O=sum min and the finite-mass identity O=(sum p+sum q-L1)/2; normalized1-O rejection
interpretation and saved-law mass errors are separate. Native paths remain controls.


Six historical matched-probe CPU result (S49):
`scripts/analyze_matched_prefix_probes.py` analyzes six recovered original ordinal0/1 round0
payloads across128/512/1280 (196953054 bytes). All138 law rows validated; quality
comparison used six first rows with exact prefix/proposal/selected p/q reconciliation.
First11 and final12 tests passed (.071/.070 s); root's one actual CPU execution
exited0 (outer1.691160 s, supervisor1.302382 s), with source/input hashes unchanged.
A core connection interruption resumed from saved code; no test/analysis failure or
repeated recovery occurred. CPU timing is not inference performance.
Fixed128 sequential-reference O_ref was.000434/.003433/.001763 for ordinal0 and
.047161/.973534/.981109 for ordinal1. From512→1280 one state worsens and one improves;
no two-state mean hides this. Each state's sequential and block teachers separately
match byte-for-byte across checkpoints, while block/seq remain different laws.
Independent six-row180-check reductions preceded freeze and withheld values until
formal completion; JSON-only337-check comparison passed (maxdiff6.66e-16<=1e-12).
See reports/matched-prefix-probes-20261010. Current recovery hashes are not historical
precommit hashes; two states cannot establish quality32 generality, dense/cache q
equality, training causes or speed. Close this analysis, retain original sampler and
frozen1280/STS state. Root accepted only designing a32-state initial-prefix
512/1280 panel with frozen coverage/reference/numerical/resource bounds; no new
implementation or model/GPU/generation/training/final-test execution occurred.


Original32 matched-initial-state diagnostic design (S50, no implementation):
See [design](docs/matched-initial32-diagnostic.md) and
[preparation evidence](reports/matched-initial32-design-20261010/README.md).
All32 saved prompts/first-target outputs match across512/1280; lengths32–326,
total2972, with64 original seven-proposal round0s. Read-only AMD hashing confirmed
both metadata and draft files (646774924 bytes each) against original bindings,
OS0 in9.397773 s; installed Torch/Transformers metadata matches. No torch import,
target/optimizer read, model/GPU init or new tests occurred. Future target/tokenizer,
loaded HIP/driver/device/backend, deployed source and coordinated window remain gaps.
Final design SHA177b91cc…5492b1 passed the [independent design-only review](reports/matched-initial32-design-20261010/independent-review.json).
Four weights-only loads run512 anchors0/1,1280 anchors0/1,then30 remaining ordinals
per checkpoint; anchors count once. Preserve7-row draft/head, q adapter[1,V],
block adapter[8,V], fresh sequential adapter[V], no_grad/eval/original BF16 AMP.
Proposed64 draft forwards/128 prefills/64 block8/64 anchor1 appends, no warmup/retry.
Fix each state's common reference to NEW512 sequential row0;1280 seq is a numerical
repeat, not a replacement. Report complete equal32 overlap/change/sign coverage,
with teacher TV/mass/sensitivity separate; no causal, holdout, quality threshold or
speed claim. Four historical full-law anchors must pass own-path byte equality
before the remaining30 ordinals; all64 selected gathers/raw confidence must match.
Lawful replay drift means historical numerical mismatch/inconclusive and stop,
not corrupt old evidence; save raw first, no threshold relaxation/backend switch.
Proposed300/290 s,6GiB allocator,8GiB RSS,1GiB output,preload free>=8GiB and
whole-card used-VRAM rise<=8GiB over a fixed baseline, including other processes.
Missing telemetry is a gap; never reset baseline/stop ASR, reap only own worker.
No runner or model execution exists; retain original sampler/frozen1280/STS and
prior benchmark/vLLM/numerical/confidence/scheduler bounds. See notebook S50.
