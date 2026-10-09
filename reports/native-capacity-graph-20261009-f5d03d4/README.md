# Native capacity-tail and gather + attention graph probe — 2026-10-09

The pinned ROCm operator passed this fixed-shape capability probe with real
pretrained Qwen3-0.6B first-layer QKV. All **18 observations completed**: five
eager variants at each of three growing contexts, followed by three actual GPU
graph replays. Every compared output was identical, including its BF16 storage
bytes. The finite and NaN unused tails did not contaminate attention output.

This establishes **gather + native attention capture/replay for one registered
ordered-Q bucket**. QKV projection, RoPE, full-layer/model execution, commit,
sampling and scheduler overlap remain outside the graph. There is no throughput
measurement or speedup claim, and the earlier whole-Qwen RMS failure is unchanged.

## Frozen shape and observed results

Two requests use ordered query lengths (1,4), physical Q=5, context ceilings
(144,144), K buffer capacity=293 and fixed maximum K=148. Actual active K is
smaller than capacity. The initial prefixes and new QKV come from real stock-SDPA
forwards using the exact shared `r2-c256` synthetic token sequences. Later
operator inputs use the probe's retained committed bytes plus new captured KV,
not newly recomputed full-prefix bytes as a substitute for the retained cache.

| Context lengths | Actual active K | Capacity tail rows | Actual max K | Fixed max K | Eager observations | Graph replay |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| (128,128) | 261 | 32 | 132 | 148 | 5/5 passed | passed |
| (129,131) | 265 | 28 | 135 | 148 | 5/5 passed | passed |
| (130,135) | 270 | 23 | 139 | 148 | 5/5 passed | passed |

The five eager variants are exact-length K/V with actual max K, exact-length K/V
with fixed max K, capacity K/V with zero tail, capacity K/V with key tail=100 and
value tail=−100, and capacity K/V with NaN tails. Cumulative K always terminates
at the active rows. The finite/NaN tails are unreachable padding, not valid
context; this result says nothing about NaNs in reachable KV.

Original thresholds remained `abs_error <= 0.02 + 0.02*abs(reference)` and
RMS≤0.005, pooled and per request. Finite/NaN-tail outputs additionally had to
match the zero-tail output bitwise. Replays had to meet the exact-length limits
and match the eager zero-tail capacity output bitwise. All observed errors were
zero. The independent CPU audit additionally compared output storage bytes,
which distinguishes signed-zero representations as well as numerical equality.
See [result.json](result.json) for every original scalar comparison and
[verification.json](verification.json) for the independent recomputation.

The storage module reports a 293-row output workspace and 586 gather rows per
K/V per layer for this bucket. Exact eager calls actually receive 261/265/270 K
rows; capacity calls receive 293. These physical shapes, padding and logical
active lengths remain distinct in each observation. Capacity is not relabeled
as measured logical attention work.

## Real graph boundary and transaction evidence

After all 15 eager observations passed, a fresh persistent pool restored the
initial prefixes. Two side-stream warmups preceded one `torch.cuda.CUDAGraph`
capture on the ROCm runtime. Its body performs fixed-size `index_select` gathers,
`where` selection, unused-tail zeroing and the pinned private attention operator.
There are three replays with in-place updates to real query/scratch contents and
metadata. Context positions advance through (128,128), (129,131), (130,135);
real QKV is recomputed externally, so the graph does not merely reuse stale RoPE.

The running probe compared 18 input allocation addresses—query, resident K/V,
scratch K/V, indices/masks, cumulative lengths, positions and workspace K/V—with
their initial addresses at each replay. All stability assertions passed. The
actual output address recorded for each replay also remained constant. The
public input address dictionaries are the initial snapshot: current input
stability is supported by the executed comparison flags, rather than separate
saved current-address dictionaries.

Artificial commit counts (1,3), (1,4), then (0,0) exercise growing KV transactions.
They are a deterministic fixture, not draft acceptance events. Each running
pre-commit check compares both full resident keys and values with saved in-memory
copies. Each commit compares every request's resulting K/V exactly with its old
prefix plus the selected scratch prefix. All checks passed. Full pre-commit
resident snapshots were not written to disk; their whole-pool isolation cannot
be independently recomputed offline from absent tensors.

The independent audit does reconstruct both committed K and V across contexts
from saved initial prefixes and captured new rows. All saved eager active inputs
and all replay exact-reference inputs match that reconstruction, including query,
cumulative lengths and positions. This supplies independent evidence for the
saved active-prefix path without claiming an offline whole-pool snapshot audit.

## Source, runtime and release

One execution used immutable source
`f5d03d4ca82482a65963429e94e4068896f5a77a`, archive SHA-256
`7d2714bcb9f40a7270243e11f313d0e1641e01a123e173ebeb004d5ef8257b9c`.
All 324 archived files, the actual target/tokenizer and shared workload bindings
were verified before and after execution. The complete CPU suite passed 226
tests in 13.806 seconds with GPUs hidden on the pinned runtime, before the GPU
probe. The fixed 300-second bound, 290-second cooperative boundary, 8 GiB free
memory guard and 6 GiB allocator cap were preserved. No retry, threshold change,
backend fallback or reduced observation domain occurred.

The actual device runtime was Torch 2.12.0+rocm7.2, HIP 7.2.53211,
Transformers 5.17.0, gfx1201, with the originally pinned private ATen schema and
AOTriton preference. [Source identity](source-identity.json) preserves protocol,
model, code and evidence hashes. Raw QKV stays in private evidence; public files
contain scalar results and hashes only.

Worker, controller and outer execution all exited 0. Supervisor lifetime was
19.940416 seconds, including model loading, evidence serialization and cleanup;
this is not a kernel or graph performance measurement. No timeout or cleanup
signal occurred. Independent live verification found the worker, controller and
shell absent; the same preserved ASR was ready, idle and the only KFD owner.
VRAM returned exactly to the prior 8,967,499,776 bytes used. Four health samples
observed at most 10,742,800,384 bytes of global VRAM use, including ASR; this is
not a continuous or per-process peak. The configured allocator cap is not a
measured peak. See [runtime audit](runtime-audit.json).

## Independent audit and next boundary

The [independent verifier](independent-verify.py) imports no probe/backend code
and loads all 42 original tensor files onto CPU. It verifies 108 exact input
tensor comparisons, all finite/NaN tails, 21 output comparison groups containing
63 pooled/per-request records, source/input integrity and release evidence.
Maximum absolute error and RMS are both 0; all compared output storage bytes
match. Root separately checked ordered scalar completeness, addresses, exits and
release, and matched the deployed archive byte-for-byte against Git; see
[root verification](root-verification.json).

The earlier [native end-to-end baseline](../native-offline-benchmark-20261009-0c36b03)
remains only 7.06%–12.26% of the measured vLLM throughput. This capability result
does not change that measurement. It removes one obstacle to integrating a
persistent target path: the pinned operator accepts this capacity-tail layout
and participates in a replayable gather/attention subgraph.

Actual target integration still needs deterministic layer/model execution,
input/position updates, bounded graph-bucket ownership and correct commit
and sampling lifetimes. Ordered Q=(1,4) is not determined solely by a historical
t−2 global capacity K, so this result does not establish graph selection from K,
asynchronous CPU/GPU overlap or a complete capacity scheduler.
