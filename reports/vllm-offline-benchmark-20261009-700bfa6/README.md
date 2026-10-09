# Qwen3-0.6B vLLM offline fixed-batch benchmark — 2026-10-09

One real engine completed all 54 preregistered batches: 12 warmups, 30 primary
measurements and 12 separate diagnostic batches. Worker and controller exited 0;
the saved independent release check found all seven observed owned PIDs absent,
the preserved ASR ready and idle, and device memory restored. This report measures
the target-only vLLM engine. It does not yet establish a DSpark speedup.

## Primary results

Each request generated exactly 128 tokens with ignore-EOS. Each cell retains all
five primary repeats. Output throughput is total completed output tokens divided
by total primary batch wall time, excluding warmups and diagnostics.

| Requests R | Supplied prompt tokens C | Pooled output tok/s | Median batch s | Min–max batch s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 131.264 | 0.972095 | 0.968560–0.984229 |
| 1 | 256 | 127.340 | 1.001198 | 0.999815–1.018282 |
| 2 | 64 | 252.165 | 1.010095 | 1.006178–1.036311 |
| 2 | 256 | 246.567 | 1.048369 | 1.011746–1.060691 |
| 4 | 64 | 515.998 | 0.986125 | 0.982235–1.021410 |
| 4 | 256 | 475.260 | 1.068880 | 1.028510–1.119473 |

The clock covers synchronous `LLM.generate` entry through return: request
submission, prefill, decode and completion. Input-list and SamplingParams setup
precede it; output validation and hashing follow it. Batch wall time is not
per-request latency. Primary TTFT and per-request completion latency are **null**
because the installed stats-disabled API does not provide them. Do not divide
batch wall time by R. See [primary-metrics.json](primary-metrics.json) for all
five times, sums, mean, median, minimum and maximum. No samples were discarded
as outliers; five repeats do not establish high-percentile serving tails.

## Separate diagnostic observations

Two additional batches per cell used explicit `add_request`/`step` events, after
all primary measurements. Times start immediately before each request submission.
The first-output clock ends on its first nonempty host-observed chunk. It includes
submission, queueing, first-token work, IPC and observation; **TTFT is not pure
prefill time**. Pure prefill remains null. Completion latency likewise includes
observer overhead. These observations do not replace primary timings.

| Case | Request observations | Median diagnostic TTFT ms | Median diagnostic completion s |
| --- | ---: | ---: | ---: |
| r1-c64 | 2 | 45.251 | 0.986084 |
| r1-c256 | 2 | 56.563 | 1.014466 |
| r2-c64 | 4 | 43.774 | 0.979081 |
| r2-c256 | 4 | 80.853 | 1.045437 |
| r4-c64 | 8 | 91.848 | 1.040385 |
| r4-c256 | 8 | 80.182 | 1.062275 |

Requests within a batch are correlated. The table pools their observations only
for a compact descriptive summary; [diagnostic-metrics.json](diagnostic-metrics.json)
contains mean/min/max as well. [scalar-samples.jsonl](scalar-samples.jsonl) preserves
all 54 original scalar records, every per-request observation and output hash,
including warmups. It contains no prompt or generated token lists.

## Frozen workload and engine

Execution source: `700bfa60746f1315baaeec5ead28cc5ee553ae52`. The shared
[workload manifest](../../configs/performance-workloads.example.json) SHA-256 is
`3f2faab15e641ba0703effafb0099a0a4a73913a513137645def9d6f562e3a12`. It materializes exact synthetic token IDs for
R∈{1,2,4}, C∈{64,256}; request i uses seed `20261009+i`. There is no template
expansion or prefix caching. Sampling uses temperature 1, top_p 1, top_k 0,
min_p 0 and neutral penalties. These synthetic inputs measure fixed-output
throughput, not natural-language quality. Equal seeds across implementations do
not establish equal sampled outputs or float64 probability-law fidelity.

The single BF16 engine used `generation_config=vllm`, tensor parallel size 1,
`max_num_seqs=4`, model/batched-token limits 4096, memory utilization 0.18,
`enforce_eager=False` and disabled statistics. The six-case primary order rotated
by repeat. Engine initialization took 98.778713 s
outside batch timing; imports/version checks took 20.506031 s.
Torch is 2.12.0+rocm7.2, Transformers 5.17.0, installed vLLM distribution
0.30.0+rocm723 and module version 0.30.0, on gfx1201 with 34,208,743,424 bytes
reported device memory. The preserved ASR remained resident and idle.

Default `ROCM_ATTN` logged its internal Triton paged-attention fallback. The
engine logged completed `FULL_AND_PIECEWISE` graph capture with configured sizes
`[1,2,4,8]`; graph replay was not independently instrumented. No manual eager or
backend fallback was applied. The existing `OMP_NUM_THREADS=2` warning is retained;
these settings were not tuned after seeing the results. Normal compilation/JIT
cache effects remain part of this single-engine warm measurement.

## Audit and scope

The worker and controller required all ordered 54 identities. A separate local
stdlib verifier independently reconstructed the schedule, checked request counts
and 128-token outputs, diagnostic clock bounds, primary null fields, and all six
rates and summary statistics. Root independently repeated the scalar checks and
compared the source archive byte-for-byte with `git archive` of the source commit.
See [verification.json](verification.json), [root-verification.json](root-verification.json)
and [independent-verify.py](independent-verify.py). The verifier takes the original
private evidence directory and imports no benchmark implementation.

[Source identity](source-identity.json) records archive/workload/binding, critical
source and model/tokenizer hashes plus original evidence hashes. The execution
bound 288 archived source files, 3,147 installed vLLM files and eight model/tokenizer
files; all 3,456 bound files passed the saved post-run integrity check. Private
machine paths and original logs remain outside the public report.

[Runtime audit](runtime-audit.json) records actual OS exits, setup boundaries,
backend/graph evidence and resource release. The supervisor waited for the worker;
the outer shell waited for the controller. Descendant absence does not assert
that every child separately exited 0. There was no timeout, retry or supervisor
cleanup signal; vLLM's own shutdown SIGTERM is separately identified. The
35 health samples reached a maximum observed global VRAM use of
15,068,602,368 bytes, including the resident ASR;
this is not a process allocator peak or a continuously measured maximum.

This is offline fixed-batch output throughput, not speculative verification SPS,
an arbitrary verification-budget B profile, an arrival-load frontier, HTTP service
capacity or a distribution-equivalence result. Native R2/C128/all64 allocation
profiling is a separate experiment; its full-round timing and supplied end-to-end
C256 workload must not be conflated. A matched native trajectory measurement is
still needed before a DSpark performance comparison.
