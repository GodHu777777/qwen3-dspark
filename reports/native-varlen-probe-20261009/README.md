# Native varlen small-tensor probe: numerical gate failed

One authorized native attempt from committed source `d45442722aecb0f555df35b1f66feb8c7b44c01f` completed with exit1 and no timeout. No retry, fallback or threshold change was performed. The first native call returned finite BF16 output with the required shape, but failed comparison against the FP32 MATH per-request oracle.

| Observation | Result |
| --- | --- |
| Runtime | Torch 2.12.0+rocm7.2; HIP 7.2.53211; gfx1201 |
| ROCm flash preference | AOTriton; preference only, not an independent device-kernel trace |
| Heads / head dimension | Hq16 / Hkv8 / D128 |
| First-case query / key lengths | [1,8] / [17,29], plus13 inactive resident keys |
| Fixed limits | atol0.02, rtol0.02; additional RMS≤0.005 |
| Elementwise mismatch | 17,841 / 18,432 = 96.8% |
| Maximum absolute error | 3.7750649452 |
| Maximum relative error | 33742.2461 |
| Process exit / timeout | 1 / false |
| Controller wall time | 11.595825s, not a benchmark |

Before the failure, the native output shape/dtype/device and finiteness checks,
the exactly-one native ATen flash operator assertion, and agreement between the
independent per-request and packed-mask FP32 MATH oracles passed. The raw profiler
event map was not persisted before the assertion; this dispatch statement follows
the source control flow and recorded traceback. The scalar error comes directly
from that traceback. RMS and peak allocation were not persisted. Native versus
packed comparison, causal ramp sentinel, poison checks and both crop cases were
not reached. Failure magnitude alone does not diagnose its cause.

The private worker result retains its initial `running` status and empty cases
because the failure preceded case append. The explicit failure and controller
completion records establish terminated failure; it is not an ongoing job.

An initial preflight rejected existing desktop render/card handles before any
native worker was created. That guard was corrected with authorization to reject
additional compute-device owners while retaining the desktop inventory. This was
not another native attempt. Desktop activity and the existing ASR stayed running;
the gate is not a timing benchmark. After exit only ASR held the compute-device
handle, ASR remained ready and not busy, the worker PID was absent, and all157
archived source files were unchanged. No further GPU run or training resume was
started.

`summary.json` contains sanitized scalar evidence, exact source/archive/protocol
hashes, limits and hashes of private raw records. `protocol.json` preserves the
predeclared protocol. Raw stdout/stderr, process inventory, machine paths and full
failure records stay private. This failure does not establish full-model fidelity,
distribution losslessness, speedup, calibration or output quality.
