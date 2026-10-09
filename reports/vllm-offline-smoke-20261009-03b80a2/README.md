# vLLM offline smoke: initialization and 128-token request completed

The separately authorized `03b80a2` run initialized Qwen3-0.6B in BF16 with the
installed engine's default graph configuration and completed one synthetic
35-token prompt → 128-token output request. Worker and controller actual OS exits
were **0**, with no timeout or automatic retry. This establishes a working
strong-engine baseline configuration, not a measured speedup or sampling-fidelity
result. The [earlier version-contract failure](../vllm-offline-smoke-20261009-e0f5ce8/README.md)
remains separately recorded.

The corrected runner independently verified distribution metadata
`0.30.0+rocm723` and module version `0.30.0`. The original frozen request SHA, seed
20261009, BF16, temperature 1, no filtering, fixed 128-token ignore-EOS budget,
`generation_config=vllm`, memory ratio 0.18, one active request and
`enforce_eager=False` were retained. No final-test records were used.

The engine reported `FULL_AND_PIECEWISE`, capture sizes [1, 2], and completed
PIECEWISE/FULL graph capture. Its default `ROCM_ATTN` path reported an internal
fallback from the ROCm custom paged-attention kernel to the Triton implementation.
This actual dispatch is retained explicitly; no manual backend switch or eager
rerun occurred. Capture logs support graph initialization; graph replay was not
independently instrumented. Normal engine initialization compiled/JIT-cached its
kernels; no package installation or environment patch was performed.

All 266 archived source files, 3,147 installed vLLM files, eight model/tokenizer
files and 3,434 total bound files were verified before and after execution.
The supervisor waited the worker's actual OS exit; an outer shell waited and
captured the controller's actual OS exit. No separately waited shell exit is
claimed. Independent inspection found all 13 owned shell/controller/worker/
descendant PIDs absent. KFD returned to preserved ASR alone, ready and not busy.
All 25 health samples remained ready/not busy. The supervisor issued no cleanup
signals; the engine's own normal teardown sent SIGTERM to EngineCore.

VRAM returned to 8,967,499,776 bytes used and 25,241,243,648 bytes free. Five-second
guard samples observed at most 15,065,853,952 bytes used; this is a sampled global
value, not exact allocator peak or isolated engine memory. Diagnostic stage times
are preserved in the audit, but a single smoke's generation duration is not a
throughput benchmark. Matched synthetic workloads, warmups, repetitions and
latency definitions are required for the next performance experiment.

See [summary.json](summary.json), [backend-evidence.json](backend-evidence.json),
[execution-audit.json](execution-audit.json), [source-identity.json](source-identity.json)
and [verification.json](verification.json). Raw tokens, installed/local paths and
full logs remain private. This result does not establish reference float64-law
fidelity, multi-request serving capacity, arrival-load behavior or DSpark speedup.
