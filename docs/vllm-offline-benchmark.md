# Fixed-batch strong-engine measurement protocol

This is a CPU-prepared experiment, not a completed benchmark. The functional
[03b80a2 smoke](../reports/vllm-offline-smoke-20261009-03b80a2/README.md) established
that the installed default-graph engine can initialize and complete one request.
Its single generation duration is not a speed result. The measured native
execution-domain plan remains separate in [packed-performance-plan.md](packed-performance-plan.md).

Both implementations consume the exact materialized token IDs and per-request
seeds in `configs/performance-workloads.example.json`. Its six end-to-end cells
are R∈{1,2,4}, supplied prompt C∈{64,256}, fixed 128 output tokens, temperature 1,
no filtering, no penalties, ignore EOS, no chat-template expansion and no prefix
caching. These are new synthetic throughput inputs, not natural-language quality
examples or final-test records. All supplied IDs count toward C. The model config
SHA and vocabulary are pinned. Equal seeds do not imply equal vLLM/reference
outputs or prove float64-law fidelity.

The same manifest separately defines native frozen-cache profiling at active
and resident R=2, committed cache C=(128,128), gamma=7 and all 64 ordered prefix
vectors in {0,…,7}², covering logical target B=2…16. This committed C excludes the
latest anchor and differs from supplied end-to-end prompt length. Its local
snapshot uses each r2-c256 request's first 128 supplied IDs and original seed;
admission leaves committed C128 and a separately emitted latest anchor. Native local
`SPS=1/T_full_round` is not vLLM output tokens/s. Normal target-only decode usually
has one query per active request; the baseline does not claim to measure arbitrary
speculative verification B or provide the native planner's capacity curve.

`scripts/benchmark_vllm_offline.py` defaults to a stdlib-only preflight. After
review and separate GPU authorization, one engine is loaded once and reused for
all cases. It keeps BF16, `generation_config=vllm`, memory ratio 0.18,
`enforce_eager=False`, prefix cache disabled and stats disabled. Maximum active
requests increases to 4 to cover the preregistered panel; max_model_len and
max_num_batched_tokens remain 4096. This new configuration still needs its own
execution validation. Record the actual graph mode/capture shapes and attention
path. The smoke's default `ROCM_ATTN` internally used Triton paged attention;
retain that fact and any changed actual dispatch in the benchmark report.
There is no manual eager/backend fallback, automatic retry or package change.

The fixed schedule is 54 batches: two discarded warmup batches per cell (12),
five primary batches per cell (30), then two diagnostic batches per cell (12).
Primary repetitions rotate the six-case order by replicate index. Every sample,
including warmups and outliers, remains in JSONL. Fresh request state is required
between batches; prefix caching remains off, while engine/kernel/allocator caches
remain naturally warm. Model construction, compilation/capture and setup are
reported outside sample timings. Token-list preparation and SamplingParams
construction occur before timing; real request submission/input processing,
prefill, decode and completion remain inside it. No reference replay or tensor
oracle runs inside measured batches.

Primary timing measures the synchronous `LLM.generate` call from entry through
return. That API's completion boundary includes the engine's execution/IPC work;
there is no separate client-device synchronization or per-token observer added.
Request counts, prompt IDs, exact 128-token outputs and length-stop completion are
checked after timing, and output hashes are computed afterward. The primary
record's per-request completion latency and TTFT are explicitly null/unsupported:
with stats disabled, the installed RequestOutput metrics are unavailable.
Never divide batch latency by R and call it request latency.

For each cell, report all five primary batch times, mean/median/min/max and pooled
output throughput `sum(completed output tokens) / sum(batch wall seconds)`.
Warmups and diagnostic timings are excluded from that aggregate. Do not average
per-batch rates, infer high-percentile service tails from five samples, or publish
an incomplete cell as a completed measurement. Partial samples survive a failure
or deadline; they do not invent unmeasured cells.

The separate diagnostic pass uses the same initialized engine with explicit
`add_request`/`step` output events and DELTA output chunks. For each request it
records the host time immediately before submission, first nonempty output chunk
and final completion. Its TTFT includes submission, queueing, prefill, first-token
sampling, IPC and observation; it is not pure prefill kernel time. Completion
latency has the same request-specific start. Observer iteration, token counting
and output collection are inside the diagnostic batch wall time. These results
are labeled diagnostic and never replace primary timings. Pure prefill time
remains null/unsupported without a separately instrumented engine pass. Arrival
rates, TTFT/TPOT under load, queueing frontiers and HTTP serving are later studies.

`scripts/guard_vllm_benchmark.py` reuses the tested Linux identity-aware supervisor
from the smoke. Build a fresh binding from an immutable committed source archive,
all installed-source files, full model/tokenizer files and the workload manifest.
The binding's `request` path names that shared workload file. Include both guard
scripts in the file map. Use the existing preserved-ASR/fresh KFD/VRAM guards,
one 600-second worker deadline plus the documented bounded health/cleanup cost,
no retries and independent post-release verification. The worker validates both
pinned distribution/module version identities. The wrapper requires all 54
samples and complete primary cells, unchanged workload/source bindings, actual
worker/controller OS success and released owned processes. Hitting the deadline
leaves a partial experiment, never a silent reduced panel.

CPU tests validate the fixed domain, schedule, exact output-count checks,
unsupported primary request metrics, request-specific diagnostic clock boundaries,
pooled-rate arithmetic, duplicate/partial sample detection and malformed domains.
They use fake engine objects and do not establish that the new multi-request
configuration or diagnostic API path works on the actual engine. That remains a
bounded separately authorized execution gate.

Installed source signatures were inspected without importing vLLM: LLMEngine
`has_unfinished_requests`, `add_request(request_id,prompt,params,...)`, and `step`
return the expected request-output interface. RequestState stores the token-prompt
IDs; its `_new_request_output` asserts they are present and passes them even for
DELTA (only prompt logprobs are popped). Our pinned tokens-only path rejects a
missing/mismatching echo rather than silently assuming it belongs to the request.
This source evidence and fake tests do not replace actual diagnostic execution.
The specification persists all 54 expected (phase,repeat,case) identities. Worker
and controller independently require their exact complete order and recompute
aggregates from JSONL; complete primary cells with a missing diagnostic batch
cannot claim overall success.
