# Native fixed-output offline baseline

`scripts/benchmark_packed_decoder.py` measures the shared six-case workload.
The [first real execution](../reports/native-offline-benchmark-20261009-0c36b03/README.md)
completed all 54 batches on immutable `0c36b03`; its throughput was only
7.06–12.26% of the matched vLLM baseline. It is separate from the frozen-cache
local profile in `profile_packed_decoder.py`. The original protocol fixes an
1800-second worker deadline and a 1790-second cooperative boundary. At roughly95 ms for the measured R2/C128
local round,54 batches times127 rounds already gives about651 seconds before
admission/setup and other R/context costs. This is budget-planning arithmetic,
not a transfer guarantee or timing prediction. The larger bound is fixed before
any end-to-end execution; never shrink the panel during a run.

The method is deliberately explicit: eager native target plus trained step1280
draft, complete seven-position shadow proposals for every unfinished request,
and fixed maximum prefix `max(0,min(7,remaining−1))` selected before the proposal
draw. There is no integrated capacity scheduler, fitted SPS, asynchronous overlap
or graph replay. Existing FP64 q/p laws, finite/normalization checks, RNG ownership,
confidence/capability validation, rejection logic and cache operations remain
unchanged. The whole-Qwen target numerical failure remains a limitation of the
path; comparing its speed does not establish target-law equivalence.

## Shared workload and model lifetime

Both this runner and the strong vLLM runner consume
`configs/performance-workloads.example.json`: exact materialized token IDs and
per-request seeds for R∈{1,2,4}, supplied prompt lengths64/256,128 output tokens,
temperature1, no filtering/penalties, ignore EOS and no chat templating. No
dataset or final-test records are read. Equal seeds across algorithms do not
imply equal tokens. Output hashes permit same-path repeat checks without public
token traces.

One target and one trained draft load serve the full54-batch schedule:12 warmup,
30 primary and12 diagnostic batches. The native target adapter is constructed and
registered once. Fresh request/cache state is created inside every measured batch;
target cache reset and new projected-draft cache construction are charged.
Model/kernel/allocator caches remain naturally warm, but request prefixes are
never reused between batches. The shared production `make_session_factory` is
used by both the worker and injected CPU tests; the tests do not duplicate a
separate corrected factory. The native adapter must not be reconstructed from
an already-native model configuration as if it were the original SDPA model.

Admission emits the first of128 output tokens. Subsequent rounds emit exactly
the remaining budget, with each request's committed prefix excluding its latest
anchor. The `remaining−1` prefix rule ensures the final round need not verify
unusable proposal positions. It does not skip full shadow computation: when only
one output remains, ell=0 still pays the complete draft block and verifies the
anchor. Finished requests become inactive but remain resident until batch end;
later copy/gather/validation work includes their actual resident cache costs.

## Timed boundary and evidence

Preparing token tensors and dedicated RNG objects occurs before the batch timer,
analogous to the vLLM runner's token-list/SamplingParams preparation, and its cost
is separately recorded. The primary timer starts before fresh native request/cache
construction and includes batched admission, first-token sampling, all complete
shadow rounds, private confidence copying, capability validation, target
verification, q/p checks, acceptance, crop/projection and final synchronization.
This method's fixed allocation does not include the real capacity planner/history;
`capacity_scheduler_integrated=false` remains explicit.

Every round retains scalar allocation, actual logical/physical B, resident and
active counts, accepted/proposed/committed counts and the existing physical work
records. Serialization, output hashes and duplicate final-output validation occur
after timing. Transient proposal observations and verification q/p tensors are
released once verification and scalar recording finish; the measurement observer
must not keep a prior round's full vocabulary tensors alive through the next
round. This changes neither sampling law nor the engine's private actual-q
ownership before commit.

The factory retains the target's final KV until the next batch reset. Memory
records therefore include explicit `pre_reset_allocated_bytes` and
`pre_reset_reserved_bytes`, followed by `whole_operation_peak_*` from before
reset through completion. Those peaks include inherited prior-batch target KV,
prepared inputs and model allocations. They must not be interpreted as an
independent steady-state peak of the current cell; reset remains inside wall time.

Primary per-request TTFT/completion fields are null rather than inferred from
batch latency. Separate diagnostic batches record host observations of the first
batched admission return and each request's finishing round. Their common start
is the native batch submission boundary. These times include native request
setup, sampling, projected-context work and observation; they are not isolated
prefill kernels or individually queued asynchronous requests. The vLLM diagnostic
uses individual `add_request` boundaries, so report the differing API observation
scopes explicitly rather than treating them as identical service timestamps.
Diagnostic timings never replace primary timings.

The shared summarizer reports every primary duration and pooled output throughput
`sum(output tokens)/sum(batch wall seconds)`, not the average of per-batch rates.
Worker and controller independently require all54 expected identities in exact
schedule order and recompute aggregates from JSONL. Missing diagnostics cannot
be hidden behind complete primary averages. Incomplete batches are not timed
samples; completed prior samples and the full declared panel survive failure or
deadline. No local-profile rate is inferred from this end-to-end throughput.

## Execution and preparation evidence

The default CLI is stdlib-only binding of actual model/tokenizer, selected trained
checkpoint, shared workload, exact protocol and script/package/supervisor hashes.
Formal execution reuses the identity-aware `guard_vllm_smoke.supervise` and
`ASRGuard`, with no new bare-PID controller, no automatic retry or fallback.
Source/input identities are checked again after execution, and actual process
exit and release are separate from worker JSON. The preparation retains an8 GiB
free-memory guard and6 GiB process allocator cap. GPU execution requires a fresh
immutable archive and separate authorization.

Eight CPU tests cover stdlib binding, all54 schedule identities and raw aggregate
recomputation, missing final diagnostic failure, deadline preservation, real tiny
BF16 consecutive batches through the production factory with one adapter and
fresh caches, exact first-token/budget accounting, full shadow work at ell0 and
separate diagnostic observations. A deterministic unequal-completion fixture
finishes B one round before A and checks that B's15 KV rows remain resident and
charged to the later round's physical context. A deadline inside the next actual
batch preserves the previously completed sample and the full54 expected panel.

The initial six-test run failed its two real fixtures because it reconstructed
the native adapter on an already-native model config. That failed log is retained.
Reusing one registered adapter and resetting its cache resolved it. Review then
identified two preparation defects: tests had copied the corrected factory rather
than calling production code, and memory peaks were mislabeled without disclosing
inherited prior-batch KV. The current factory tests and explicit memory fields
resolve those issues before any GPU benchmark. CPU tests use hidden GPUs and
cannot establish native end-to-end performance.
