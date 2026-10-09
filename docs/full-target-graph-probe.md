# Full pretrained target graph capability protocol

`scripts/probe_full_target_graph.py` prepares one bounded comparison of the full
original HF target under explicit pinned native eager execution and actual
CUDA/ROCm graph replay. It is a separate runner over the CPU-validated persistent
target; it does not change the target, sampling law, model weights or environment.
The default command only binds fingerprints and source hashes. No GPU execution
or speed result is reported here.

## Fixed scope and inputs

The formal binding checks all target/tokenizer fingerprints in the generation
manifest and the real dense Qwen3-0.6B architecture. It uses the existing shared
`r2-c256` synthetic request token arrays and their workload hash. The two active
requests start with their first 128 tokens. A third inactive resident uses tokens
200:217 of the second request. Inputs are synthetic fixtures, not held-out text.

One ordered verification bucket has Q=(1,4), context ceilings=(144,144), physical
Q=5, physical K capacity=293 and fixed max K=148. The three context vectors are
(128,128), (129,131), (130,135). Query IDs are sliced from the bound source arrays
at each current context. Artificial committed-query counts are (1,3), (1,4),
(0,0); the last round explicitly aborts. These are transaction fixtures, not
sampled acceptance decisions. Prefill uses a separate registered eager bucket.

The graph contains the original embedding, all 28 decoder blocks (QKV, QK norm,
RoPE, attention, output projection, residuals and MLP), final norm, scratch KV
writes/gathers and raw selected-output copies. The LM head, prefill, metadata,
resident commits and all sampling remain outside. Selected IDs are exactly
[1,7,14,21,26], each a zero-based raw decoder-block output. Layer 26 is not the
last decoder block; inequality with final norm is not used as identity evidence.

## Required sequence and comparisons

The entire three-state native eager phase must pass before capture begins. At
each state it saves raw selected features, final-normalized hidden states, logits
and all 28 layers' speculative K/V. Independent forward hooks identify every
selected raw output and the original final norm exactly. It aborts the first
verification and checks every resident key/value byte is unchanged, then fills
only the inactive resident with finite K=100/V=-100 and repeats verification.
All output and scratch tensors must remain bit-identical to that state's first
eager result. It commits the declared prefix, checks exact old-prefix-plus-scratch
contents, restores the inactive resident and saves complete resident tensors.

The fixture then resets ownership, explicitly zeros retained resident storage
and performs the same native eager prefill. This setup restores the complete
initial tensor state; ordinary request reset alone intentionally retains old
unaddressable bytes. Capture must preserve resident bytes and fixed addresses.
The captured program has two side-stream warmups and one capture, followed by
exactly three useful replays with refreshed token IDs and positions.

Every selected raw layer, final norm, logits, and each layer's K and V compare
against that state's saved native eager tensors. The fixed numerical limits are
elementwise `abs_error <= 0.02 + 0.02 * abs(reference)` and RMS <= 0.005, both
pooled and for each request separately. Shape/dtype and finite values are required.
Committed active prefixes compare per layer/request with the same limits; the
inactive resident and each path's own commit operation compare exactly. No
threshold changes are made after execution.

The graph feature lease must survive target commit/abort. Attempting another
prepare while leased must fail. An actual LM-head read after commit must equal
the same replay's saved pre-commit logits exactly. This lease check does not
impose bit equality between native eager and graph outputs. Release follows the
consumer on the target owner stream.

Model-forward and every decoder-hook Python counters are sampled before/after
capture and each replay. Setup must increment them three times. On real GPU,
every useful replay must leave all counters unchanged while input IDs/positions
change and paired outputs/KV pass. CPU tests instead require the emulator's
counter increments and reject it as real graph evidence. The counters establish
that these Python boundaries did not rerun; they do not measure GPU performance.

The eager reference uses the same `PersistentQwenTarget` and pinned attention
backend. Its eager cache path is compared with the separate captured cache path.
Raw hooks independently witness layer identity; they are not an independent
attention oracle. Passing this probe would establish same-backend full-target
graph fidelity for one fixed family, not sequential/cross-backend distribution
equivalence. The earlier [whole-Qwen layer-26 RMS failure](../reports/whole-qwen-varlen-gate-20261009/README.md)
and BF16 endpoint differences remain unchanged. It also does not establish a
t−2 capacity-selected graph family, overlap, end-to-end throughput or speedup.

## Budgets, supervision and evidence

The graph private-pool reservation is 512 MiB. The combined registered workspace,
fixed graph input/output and graph-reservation budget is 768 MiB, with a separate
256 MiB workspace cap and one graph. Actual retained graph bytes come from scoped
private-pool segment `total_size`, including inactive blocks; global reserved
before/after/delta remain separate. A reservation failure durably saves the full
pool snapshot and measured/reservation accounting before propagating failure.
This does not measure transient allocator peaks or all process/driver memory.

The formal controller reuses `guard_vllm_smoke` and its identity-aware supervisor
and ASR guard. It requires 8 GiB pre-launch free memory, applies the existing
6 GiB allocator-fraction cap, and allows one 300-second worker window with a
290-second cooperative boundary. There are no retries or other GPU experiments
inside this runner. An actual timeout retains partial evidence; it is not a pass.
Independent release checks and the outer controller exit remain operational
requirements. Execution requires a separately reviewed immutable Git archive
and an authorized GPU window.

```sh
python scripts/probe_full_target_graph.py --model MODEL --generation-manifest GENERATION_JSON
# In the separately authorized window only:
python scripts/probe_full_target_graph.py --model MODEL --generation-manifest GENERATION_JSON \
  --execute --output FRESH_DIR --asr-pid PID --asr-url URL
```

The binding includes the target, generation manifest, shared workloads and actual
runner/dependency hashes; the worker and controller recheck it at completion.
Private `.pt` files preserve request tokens, actual prepared metadata, independent
raw-layer witnesses, all eager/isolation/replay outputs and scratch K/V, initial
and post-commit resident tensors. Input evidence is saved before useful replay;
capture metadata is saved from the actual prepared fixed buffers, not the prior
case's positions. Numeric outputs are saved before assertions. Failure handling
also attempts a full resident/scratch snapshot and, for capture failures, the
prepared buffer state; a device error preventing that write is explicitly recorded.
Only aggregate scalar reports should be published; raw tensors stay private.

The report retains six ordered observations (three eager, three graph), separate
stage states and structural comparisons. Failed eager validation cannot advance
to capture. A failed replay retains its raw output and earlier passing cases.
The controller rejects incomplete domains, failed checks, CPU-only accounting
and any claimed real replay whose Python counters changed.

## Local preparation evidence

Eight targeted CPU tests exercise real four-layer tiny Qwen tensors, the actual
persistent target and the explicitly labeled CPU emulator. They cover the full
three-state protocol, selected-layer identity, resident-value contamination,
inactive isolation, partial commits/rollback/lease use, prepared capture metadata,
budget failure with retained pool snapshot, numerical replay failure, deadline
partial evidence and rejection of Python-executing pseudo-replay. Binding is
tested with heavyweight imports forbidden. The first seven-test run passed;
a subsequent test-helper edit briefly failed import due to a mistaken default
argument and was fixed, with all eight tests passing. Logs are retained under
`output/full-target-graph-preparation-20261009`.

The local runtime is Torch 2.11 / Transformers 5.4 with GPUs hidden. Existing
core commit `c2d11a0` separately passed all 261 tests on the pinned AMD CPU runtime;
this runner adds only its script, documentation and targeted tests. No duplicate
full core suite, GPU execution or new numerical result is implied by preparation.
