# vLLM offline smoke: runner version-contract failure

The single authorized `e0f5ce8` attempt stopped after import, before engine
initialization. The runner compared the module's version against the distribution
metadata version. This installed package exposes **0.30.0** from its already
fingerprinted `_version.py`, while distribution metadata is **0.30.0+rocm723**.
The runner incorrectly treated these distinct fields as identical. This is a
runner contract failure, not evidence that BF16, the GPU, attention or graphs fail.

Worker and controller actual OS exits were both **1**; no timeout, retry or
fallback occurred. The failure was emitted 13.21 seconds after the worker began;
controller lifecycle time was 15.90 seconds. These are diagnostic timings, not
performance measurements. No engine was constructed and no tokens were generated.

The intended workload was a newly authored public synthetic non-thinking prompt
(35 tokens, seed 20261009), 128 fixed output tokens, Qwen3-0.6B BF16, temperature 1,
no filtering, `generation_config=vllm`, memory ratio 0.18 and default graph settings
(`enforce_eager=False`). Final-test records were not used. Import emitted an
optional torch-c-dlpack JIT warning; the fatal exception was the explicit version
guard. No package installation or environment fix was attempted.

The immutable archive contains 257 files. The binding covers 3,147 installed
vLLM files, all eight model/tokenizer files and 3,425 total files; all remained
unchanged after execution. A subsequent stdlib AST read of the already-bound
`_version.py` established the distinct module value without another vLLM import.
The original traceback, worker log, stages and process evidence remain private;
public scalar identities and evidence hashes are retained here.

The supervisor waited the worker's OS exit; an outer shell waited and captured
the controller's OS exit. No separately waited shell exit is claimed. Independent
post-release inspection found all four owned shell/controller/worker/descendant
PIDs absent. KFD belonged only to preserved ASR, which remained ready and not busy.
VRAM before and after was 8,967,499,776 bytes used and 25,241,243,648 bytes free.
No cleanup signals were required.

See [summary.json](summary.json), [version-diagnosis.json](version-diagnosis.json),
[execution-audit.json](execution-audit.json), [source-identity.json](source-identity.json)
and [verification.json](verification.json). Engine initialization, actual graph
capture/replay, a complete request, sampling fidelity and a strong performance
baseline remain unverified. A corrected runner needs its own review, immutable
source and separately authorized experiment; this record is not overwritten.
