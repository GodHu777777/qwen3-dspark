# R1 paired complete-session attribution

`scripts/benchmark_paired_r1.py` prepares a separate attribution experiment on
the original `r1-c64` and `r1-c256` cases. Each request completes all 128 output
tokens. The two arms are persistent native target-only sampling and trained
step1280 fixed-γ7 speculative sampling with all seven shadow positions computed
every round, including when the selected prefix is zero. This is a two-case
complete-session experiment. The final multi-request objective remains open;
this panel is not the old six-case/54-batch benchmark or a serving frontier.

The [original native benchmark](native-offline-benchmark.md), shared six-case
manifest and core sampling/cache/graph code are unchanged. The new runner first
validates the complete original manifest, binds its bytes, and only then derives
the explicitly named R1 view. It keeps the original temperature 1, empty EOS
set, dedicated seeded RNG per request, exact FP64 probability law, rejection
sampling and checks. Equal seeds across algorithms need not yield equal tokens.

## Explicit execution and memory plan

One persistent target adapter and one loaded trained draft serve both arms.
Both arms capture the same selected raw layers `[1,7,14,21,26]`, retain the same
resident KV/scratch allocations, and share the same Q=1 graph. The draft weights
and both graph private pools remain resident during target-only batches; their
resident cost is reported, though target-only executes no draft backbone or
projected draft cache. The speculative arm constructs a fresh packed draft cache
inside each batch. This comparison attributes execution overhead in this stack;
it does not describe a separately optimized target-only deployment.

Exactly ten workspaces are declared: eager prefill Q=64 and Q=256 at C=0, and
verification Q=1…8 with C ceiling `383-Q`. Verification K capacity is therefore
383 for every Q. Slots=1, resident context capacity=384 and scratch query capacity
256. For the real BF16 target, these workspaces require 39.7172 MiB and resident
plus scratch storage requires 70 MiB, excluding target/draft weights, graph
intermediates, LM-head/probability tensors and allocator overhead.

Only Q=1 and Q=8 have explicitly registered graphs. Setup primes the exact shared
256-token prompt, captures each graph using its declared input slice, aborts the
capture transaction and validates its first replay before resetting request state.
For each new Q shape, same-input native eager and first replay compare every raw
selected layer, final norm, logits and every layer's scratch K/V at fixed
atol=.02, rtol=.02 and RMS≤.005. With R=1 the pooled and request reduction coincide.
Resident KV must remain unchanged through capture and both aborted verifications,
and actual GPU replay must not increment model/decoder Python counters. Original
comparison tensors are saved before numerical assertions, including failure
artifacts. `setup_validation_seconds` separately charges these checks and artifact
writes; they are in this E2E startup, not a separate GPU gate or primary sample.
The reference remains same-native eager, so this adds no independent attention
numerical oracle or distribution-equivalence claim.
Setup performs no sampling and never
captures on a miss. Q=2…7 intentionally use the same persistent native eager
target. These complete tail rounds remain inside batch wall time and appear in
per-round and per-batch coverage counts; they are not dropped or labeled graph
replays. Prefill always runs eager. Physical Q is never padded.

The reservation is unchanged from the reviewed bound: 512 MiB per graph, two
graphs, combined graph/workspace budget 1280 MiB, workspace cap 64 MiB, 8 GiB
free-memory guard and 6 GiB process allocator cap. Private pools retain independent
accounting including inactive allocator blocks. A larger measured pool is a
failure, not permission to shrink the panel or increase budgets. Capture-progress
records preserve completed captures and current setup stage; private-pool budget
failure preserves its measured accounting and raw pool snapshot.

## Complete paired schedule and timing

The panel has 36 batches: two warmup, five primary and two diagnostic repeats
per case per arm, or 18 batches per arm. The original two-case schedule determines
case order (including rotated primary case order). Within every case/repeat pair,
even repeats run target-only then speculative; odd repeats reverse the arms.
Every paired batch uses fresh request state and original per-request seed.

The primary timer starts before cache reset and session construction. It includes
admission and the first sampled token, every verification round, all speculative
shadow work and actual-q validation, original FP64 calculations and draws, host
metadata, graph submission/completion waits, eager tails, commit, draft projection,
lease release and final device synchronization. Preparing input tensors/RNG objects
is separately timed outside the batch, as in the prior native baseline. Serialization
and output hashing follow the timer. Full-vocabulary observations and q/p results
are released after each round. Model loading, priming and capture are setup costs
reported separately; no cold-start performance claim is made.

Primary TTFT/completion fields remain null. Separate diagnostic batches record
host-observed admission and completion times. These include all session overhead
and are not isolated kernel measurements. Pre-reset allocated/reserved bytes and
whole-operation peaks include common loaded weights, both graph pools, prepared
inputs and inherited target KV. They are not isolated per-case memory costs.

Each round stores its actual execution kind, logical/physical Q, complete existing
work record, sampling decisions and model/all-decoder Python count deltas. Actual
GPU Q=1/8 replay must leave those counters unchanged. Eager tail calls increment
all counters once. CPU tests explicitly use the CPU replay emulator, count its
Python execution and report zero actual GPU graph rounds. Actual graph-covered
rows and declared graph-plan rows are distinct fields.

Worker and controller require all 36 identities in exact order and independently
recompute arm-level pooled throughput `sum(output tokens)/sum(batch wall seconds)`.
Each arm/case also retains all five primary durations and summed primary execution
coverage, including eager-tail rounds and rows and actual GPU graph rounds and rows.
A missing final diagnostic is incomplete even if all primary samples exist.
Completed samples survive failures/deadlines; incomplete batches are not samples.
The supervisor bound is 1800 seconds, with a 1790-second cooperative boundary
including loading and setup. There is no automatic retry or alternate protocol.

## Baselines, fingerprints and interpretation

The default CLI performs stdlib-only binding of the real target/generation
fingerprints, exact selected step1280 checkpoint, shared manifest, protocol and
runner/package dependencies. It also binds the frozen vLLM source identity and
scalar evidence. The matched five-repeat rates are independently recomputed from
that report's raw scalar samples: 131.2641 output tok/s for R1/C64 and 127.3399
for R1/C256. They appear alongside both new native arms; the strong baseline is
not replaced by the slower same-stack control. The old vLLM timing scope and this
native FP64 implementation differ, so this remains an execution-stack comparison.

The earlier [whole-Qwen numerical failure](../reports/whole-qwen-varlen-gate-20261009/README.md)
and BF16 endpoint differences remain. The same-native graph fidelity gate cannot
establish cross-backend target-law equivalence, and paired speed measurements do
not establish it either. No calibrated capacity scheduler or CPU–GPU overlap is
integrated. Extending to the full multi-request goal requires finite physical-B
families and a bounded graph/workspace policy; exact ordered-Q enumeration is
not a scalable substitute.

Preparation uses focused CPU tests through the actual production factory, with
real tiny BF16 Qwen and real draft modules. It covers full 128-token sessions,
same-path repeat hashes, all six eager tail shapes, Q=1 full-shadow work, shared
adapter/weights and fresh resets, exact schedule validation, baseline binding,
real replay counter rejection, failed startup numerical validation with raw
outputs preserved, and partial deadline preservation. CPU emulation
does not establish GPU performance. This document describes preparation until
a separately reviewed immutable GPU run is recorded.
