# Native capacity-tail and capture capability probe

`scripts/probe_native_capacity_graph.py` is a bounded executable probe. The
[first real execution](../reports/native-capacity-graph-20261009-f5d03d4/README.md)
on immutable `f5d03d4` passed all 18 observations, including three actual GPU
graph replays, with bit-identical output comparisons and verified release.
It answers whether the pinned private ROCm attention operator can consume real
pretrained Qwen QKV through a fixed-capacity K/V buffer and replay a fixed-address
**gather + attention** subgraph as the committed context grows. It does not capture
QKV projections, RoPE, a whole decoder layer/model, sampling, commit or scheduling.

## Frozen inputs and applicability

The formal CLI binds the real dense Qwen3-0.6B architecture and all target/tokenizer
fingerprints from the generation manifest. It loads the full pretrained target
once, BF16 with Transformers 5.17.0. A temporary model-owned attention callback
captures layer zero's actual normalized, RoPE-applied Q/K and projected V, while
delegating the forward to the installed stock SDPA function. The previous model
backend setting is restored in `finally`; existing adapters/defaults and source
oracles are not changed. CPU tests use a separately labeled tiny Qwen fixture.

Token inputs are the exact shared `r2-c256` request sequences. Each requested
context is extracted from a real full-prefix target forward. The experiment then
retains its own earlier committed first-layer KV and appends each subsequent
captured query suffix, like a cache transaction. The native reference uses those
same committed bytes plus new captured KV, not a numerically different full-prefix
recomputation. This is a same-QKV native capability check; it does not repair or
replace the earlier whole-Qwen numerical gate.

One explicitly registered bucket has R=2, ordered Q=(1,4), context ceilings
(144,144), physical Q=5, K capacity=293, max Q=4 and fixed max K=148.
The three ordered context states are (128,128), (129,131), (130,135). Artificial
committed-query counts are respectively (1,3), (1,4), (0,0). They are a KV
transaction fixture, **not sampled draft acceptance events**. Source positions
change with the real prefix. The captured subgraph receives their effect through
externally refreshed real QKV; it does not contain RoPE or use a position tensor
as a substitute for recomputing QKV.

The exact ordered-Q shape is not determined solely by a t−2 global capacity K.
A passing result would establish this one operator shape family, not zero-overhead
scheduling or a graph selectable from K before current confidence is known. The
later R/physical-B/max-Q family and double-buffer/event ownership questions in
[persistent-target-kv.md](persistent-target-kv.md) remain open.

## Independent stages and fixed numerical decisions

The report stores separate `real_qkv`, `eager_tail` and `capture_replay` states.
Real-QKV extraction saves each request as soon as it completes, preserving prior
requests if a later target forward fails. Raw tensors are saved before checking
numerical results. Operator inputs are
saved **before calling** the operator, so a rejected shape or backend exception
still preserves the attempted QKV/cumulative lengths/positions. There are 18
expected observations in one immutable order: five eager variants for each of
three contexts, followed by three replays.

The eager stage applies these five variants at each context:

1. Exact active K/V tensor lengths and the actual maximum request K length.
2. The same exact K/V tensors with fixed max K=148.
3. Capacity-length K/V with zero-filled unused tail and fixed max K=148.
4. The same capacity buffers with unused key tail=100 and value tail=−100.
5. The same capacity buffers with NaN in both unused tails.

All calls use the pinned private ATen schema, causal bottom-right attention,
zero dropout and no attention window. Cumulative K terminates at actual active
K (261, 265 and 270 rows), below the 293-row capacity. Exact input tensors do not
contain capacity padding. Records separately report actual operator Q/K shapes,
logical active K, padding, max-Q/max-K arguments, and storage/gather work; fixed
storage capacity is never presented as observed logical attention work.

Variants 1–3 compare against variant 1 using preregistered BF16 limits:
`abs_error <= 0.02 + 0.02 * abs(reference)` elementwise and RMS≤0.005, both pooled
and per request. Every compared tensor must have matching shape/dtype/device and
finite output. Variants 4 and 5 must be **bit-identical** to variant 3, with finite
outputs. NaN-tail rejection or contamination is a required-gate failure, not an
exception to be removed after seeing the result. It concerns an unreachable tail;
it makes no promise about native attention with NaNs in valid context.

Only after all 15 eager observations pass does the tool create a new persistent
pool, restore the initial prefix, allocate its fixed query buffer and attempt
capture. Two warmup calls run on a side stream, followed by one graph capture.
The body includes fixed-size `index_select` source gathers, `where` selection,
invalid-tail zeroing, and the private attention operator. Python transaction
validation, metadata construction, host copies, disk writes, QKV generation and
commits remain outside capture. This experimental path uses the persistent
module's source-bound workspace directly without changing that module.

Each of the three replays updates real query/scratch content and metadata in
place. It compares against that state's exact eager native reference using the
same limits, and against the eager zero-tail capacity output bit-for-bit. It
checks all resident/scratch/workspace/query addresses, the captured output address,
and both unchanged committed keys and values before the artificial commit. Committed bytes are then
compared exactly with an independently concatenated old-prefix plus chosen
scratch prefix. Later contexts use those committed bytes. A rejected or partial
probe never becomes a completed capture result.

The private operator is called directly only in this experimental probe. The
existing production reference wrapper still rejects capacity-tail shapes via
its original exact-K contract. Its runtime/schema pin is validated before formal
execution; no fallback, backend preference change, package patch or threshold
retuning is performed.

## Bounded execution and evidence

Default invocation only hashes/binds sources, actual model fingerprints and the
shared manifest. It does not import Torch/Transformers or initialize a GPU:

```sh
python scripts/probe_native_capacity_graph.py --model MODEL --generation-manifest GENERATION_JSON
```

Formal execution additionally requires `--execute --output FRESH_DIR --asr-pid PID
--asr-url URL`, from a reviewed immutable source archive in a separately authorized
GPU window. It reuses the identity-aware `guard_vllm_smoke` supervisor and ASRGuard,
with a 300-second worker bound, a 290-second cooperative boundary, no retries,
8 GiB pre-launch free-memory requirement and 6 GiB allocator fraction cap. Health,
KFD ownership, actual worker wait status, cleanup and released-process evidence
remain mandatory. An independent post-release check and outer controller OS exit
are still required operationally; worker JSON alone is insufficient.

The stage result is rewritten after every completed comparison. A backend error,
failed comparison or deadline preserves completed observations and the full
expected domain, and prevents advancing to capture. A supervisor kill cannot be
caught by Python; already-written files remain partial evidence. Source/model/
workload bindings are rechecked at worker completion and by the controller.
Controller validation requires the exact complete ordered observation domain,
passed numerical/identity/isolation fields and all three completed stages.

This is a capability probe, not a throughput benchmark or a whole-model graph
implementation. No speed estimate should be derived from disk-heavy evidence
capture, warmups or wall-clock completion. Even a successful result still needs
integration with deterministic full-target execution, a bounded graph-bucket cache,
correct sampling/commit ownership, and the intended CPU/GPU scheduling overlap.

## Local CPU preparation

Contract tests execute actual tiny Qwen QKV extraction, independent per-request
SDPA over cumulative active lengths, finite/NaN unused-tail isolation, fixed-buffer
growing-context bookkeeping and an explicitly labeled CPU replay emulator. The
emulator executes Python tensor work and copies into a fixed output allocation;
it is not CUDA/ROCm graph execution. Tests also inject values-only resident contamination in eager and replay paths
(the outputs can still numerically match), and force a finite-tail mismatch, a
NaN rejection, a capture error and a deadline, confirming durable partial evidence
and that failed eager validation never enters capture. The formal binding test
forbids heavyweight imports and validates the exact real architecture contract.

Local preparation uses the existing Torch 2.11.0 / Transformers 5.4.0 CPU runtime,
not the pinned native environment. The formal worker requires Torch/ROCm runtime
identity from `PinnedRocmVarlenKernel`, real GPU BF16 Hq16/Hkv8/D128 tensors and
Transformers 5.17.0. No GPU capability is inferred from local tests.

The immutable formal source also passed all 226 CPU tests in the pinned AMD
environment with GPUs hidden (13.806 seconds, OS0). GPU capability evidence
comes from the separate real execution linked above, not that CPU suite.
