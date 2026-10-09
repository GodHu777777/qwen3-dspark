# Native variable-Q full-target capability protocol

`scripts/probe_query_family_graph.py` is an additive bounded probe over
`QueryFamily`. It reuses the existing full-target probe's binding, raw tensor
saving, numerical comparison, forward witnesses and resource supervisor. It
does not edit the old fixed-Q protocol or its historical result. Its default
command is stdlib-only input/source binding; GPU execution belongs to a reviewed
immutable source and coordinated runtime window.

This is a capability and same-backend graph-fidelity check, not a speed benchmark,
sampling/quality experiment, independent attention oracle or stock-HF numerical
law guarantee. It does not change the trained draft, calibration or final test.

## Frozen six-state domain

The target is the pinned pretrained BF16 Qwen3-0.6B under Transformers 5.17.0 and
the existing explicit ROCm backend. Selected raw decoder layers remain
`[1,7,14,21,26]`. Inputs are the original shared `r2-c256` synthetic arrays. Active
requests start at C=(128,128); a third inactive request retains tokens 200:217 of
the second array. The same eager prefill restores the full resident initial state
between the reference and captured paths.

All families share one fixed metadata/gather arena, with R=2 and maxK=148:

| Family | Actual physical B | Fixed maxQ | K capacity | Execution |
| --- | ---: | ---: | ---: | --- |
| S3 | 3 | 3 | 291 | Registered graph |
| L5 | 5 | 5 | 293 | Registered graph |
| E2 | 2 | 2 | 290 | Explicit eager |

Fixed maxQ and maxK are strictly greater than the actual per-request maxima in
every declared verification state. No query padding fills B. Unused key capacity
is neutral storage excluded by cumulative key lengths.

| Case | Family | Ordered requests | Actual Q | Contexts in that order | Artificial commit |
| --- | --- | --- | --- | --- | --- |
| 0 | S3 | r0,r1 | 1,2 | 128,128 | 1,1 |
| 1 | L5 | r0,r1 | 1,4 | 129,129 | 1,3 |
| 2 | E2 | r0,r1 | 1,1 | 130,132 | 1,1 |
| 3 | L5 | r0,r1 | 2,3 | 131,133 | 1,2 |
| 4 | L5 | r1,r0 | 4,1 | 135,132 | 2,1 |
| 5 | S3 | r0,r1 | 2,1 | 133,137 | Abort 0,0 |

The final canonical active lengths are r0=133 and r1=137; idle remains 17. These
commits are transaction fixtures, not acceptance events. Middle E2 overwrites
shared metadata before L5 replay resumes; the late request reversal changes the
slot/gather order as well as Q. S3→L5→S3 verifies small-large-small arena reuse.

## Required execution and comparisons

All six native eager reference states must pass before any capture begins. Each
state saves raw selected features, final norm, logits and every layer's scratch
K/V before checking them. Independent hooks witness the selected raw block
outputs and original final norm. Eager abort must leave all resident K/V exact;
a repeat with only the inactive resident filled with K=100/V=−100 must produce
bit-identical finite outputs and scratch. Each path's own artificial commit is
checked exactly against old resident bytes plus the selected scratch prefixes.

After reset, retained resident storage is explicitly zeroed and eager prefill
reconstructs the complete initial state. S3 is captured with Q=(1,2), then L5
with Q=(1,4), both at C=(128,128). Each graph has two warmups and one capture.
Model and every decoder-hook Python counters must increment exactly three times
per GPU capture. Capture cannot modify resident bytes or fixed addresses.

The second six-state sequence performs exactly five useful graph replays and one
explicit eager call. Actual prepared token IDs, cumulative Q/K, positions,
resident/scratch gather indices, source selection and valid masks are saved.
Independent host construction checks those buffers against each ordered case.
Graph input/output and all shared view addresses must remain stable. Real replay
must leave model and every decoder Python counter unchanged; E2 must increment
each once. The CPU emulator instead increments once per replay and is explicitly
rejected as real GPU evidence. The report verifier recomputes the counter and
physical-work checks from saved fields rather than trusting success labels.

Same-backend eager versus replay comparisons retain the original limits:
elementwise `abs_error <= 0.02 + 0.02 * abs(reference)` and RMS <= 0.005, both
pooled and for each ordered request. Every selected raw layer, final norm, logits
and each layer's scratch K/V is checked. Every committed active K/V prefix is
checked per layer/request with the same limits. Bit equality is descriptive for
these cross-path comparisons; it is required only for structural own-state,
inactive-isolation and lease-consumer checks. No post-run threshold adjustment is
part of this protocol.

Graph features remain leased after commit or abort. A valid E2 verification must
be refused while the lease is held; an actual post-commit LM-head read must equal
that same execution's pre-commit logits exactly. Release follows the consumer on
the owner stream. This uses the existing receipt and lease contract and adds no
ownership framework or asynchronous overlap.

## Resource and evidence bounds

Two independent private graph pools each reserve 512 MiB. The combined budget is
1280 MiB and the separate workspace cap is 64 MiB. Shared metadata/gather backing
is charged once; graph I/O and private-pool reservation remain charged per graph.
Actual retained graph bytes use the existing scoped allocator segment accounting,
including inactive blocks. Shared workspaces do not authorize sharing private
graph pools. A second-capture failure preserves the first successful capture and
the failing pool's measured/reserved bytes and raw snapshot.

The existing ASR-preserving supervisor retains the 8 GiB pre-launch free-memory
requirement, 6 GiB allocator-fraction cap, 300-second outer window and 290-second
cooperative boundary. There are no retries or secondary experiments in the probe.
The allocator cap is not a measurement of all process/driver allocations. Finite
model shapes and six states bound raw tensor evidence; this runner exports no
profiler trace. Worker/controller input hashes and actual exits remain required,
with independent process/resource release checked by the coordinating runner.

`result.json` preserves twelve ordered observations, two capture records, stage
status and each attempted case/family. Raw numeric output is written before its
numerical assertions. Prepared replay input is written before submission.
Failures preserve prior cases and attempt resident/scratch plus prepared-buffer
snapshots; a device error preventing snapshot writing is separately recorded.
Incomplete or timed-out domains cannot pass `verify_report`.

```sh
python scripts/probe_query_family_graph.py --model MODEL --generation-manifest GENERATION_JSON
# With reviewed frozen source in the coordinated window:
python scripts/probe_query_family_graph.py --model MODEL --generation-manifest GENERATION_JSON \
  --execute --output FRESH_DIR --asr-pid PID --asr-url URL
```

## Local preparation boundary

Seven new CPU contract tests cover the complete real tiny-Qwen six-state sequence,
stdlib binding, unchanged old protocol, raw-layer/value corruption, stale same-B
cumulative Q before submission, second-capture budget failure, retained numeric
failure output, Python-executing pseudo-replay rejection and deadline evidence.
The eight original full-target probe tests are also retained as focused regression.
Local results and source identities are in
`output/query-family-graph-preparation-20261009/`. These tests use the explicit
CPU replay emulator, not ROCm or pretrained weights; no duplicate full core suite
is required by this additive preparation. Native generalized-Q correctness,
capture support, source/runtime fidelity and resource release remain facts for
the separately coordinated execution, not conclusions from this document.


## Completed native execution

Execution commit `a4942af` subsequently passed the complete fixed domain, with
worker/controller/SSH exit0 and independently verified release. The
[result report](../reports/query-family-graph-20261009-a4942af/README.md) records
all twelve observations, two captures and the independent raw CPU audit. The
preparation history above remains CPU evidence; the subsequent run supplies
native evidence only for the declared finite same-backend target domain.
