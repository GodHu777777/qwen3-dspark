# Strong target-only engine: bounded offline smoke protocol

This is the next execution proposal, not a completed GPU result. Source inventory
found vLLM 0.30.0+rocm723 with offline `LLM.generate`, Qwen3 registration and graph
configuration support. Hidden-GPU `vllm --help` timed out after 25 seconds; no
process remained. That observation does not establish a kernel/engine failure.
No installation, environment change, Podman action or eager fallback is proposed.

`scripts/probe_vllm_offline.py` defaults to a stdlib-only preflight. It validates
frozen private token IDs and writes the exact configuration, input/config/source
hashes and intended fixed workload without importing vLLM or Torch. `--execute`
is reserved for one separately authorized GPU window. It refuses to overwrite
an output directory, and there is no automatic retry.

The initial smoke uses a newly authored public synthetic prompt tokenized with
Qwen3-0.6B `apply_chat_template(..., enable_thinking=False, add_generation_prompt=True)`,
128 output tokens, BF16, temperature=1, top_p=1, top_k=0, min_p=0, no penalties,
`generation_config=vllm`, and an explicit seed. It ignores EOS for fixed work,
which differs from an EOS-stopping quality evaluation. `scripts/prepare_vllm_smoke_request.py` performs this step with GPUs hidden and
records the complete local model/tokenizer file hashes. It does not read any
dataset or final-test record. The prompt and seed must be frozen and hashed before the window, with both parties using the same prompt
for later comparisons. Raw prompt/output tokens stay private. A successful
smoke proves only that this configuration initializes and completes one request.

Engine settings: TP=1, max_model_len=4096, max_num_seqs=1,
max_num_batched_tokens=4096, gpu_memory_utilization=0.18, no prefix caching,
`enforce_eager=False`, no explicit compilation/attention override, and no log
stats/profiling/per-token observer. Preserve the installed engine's default graph
configuration; initialization or graph failure is retained as a failure, not
silently rerun eager. The worker records configured graph mode and engine logs.
`enforce_eager=False` alone does not prove graph capture or replay; those need
separate evidence. The 0.18 setting is an engine budget ratio, not a hard total
process/VRAM limit, and model/graphs/workspaces must be checked live.

`scripts/guard_vllm_smoke.py` provides the actual Linux stdlib supervisor. Its
default validates a frozen binding without starting work. The binding records
worker/controller/request/model/tokenizer hashes and the Python/model/request
paths, preserved ASR identity and health endpoint. Include the committed archive
and installed-source inventory files in its `files` mapping as well. Before its
explicit `--execute`, the root must:

1. Freeze the committed runner/source archive SHA and validate the installed
   source inventory/version, model file manifest, private request SHA and exact
   intended settings. Never deploy mutable working-tree code.
2. Obtain the single agreed GPU window. Fresh `/proc`/KFD/VRAM checks must show no
   unowned GPU workload besides the preserved ASR PID 1208354. Probe ASR on port
   8768 and require ready/not busy. Record identity/start time; never kill it.
3. Record fresh total/free VRAM and account for the estimated 0.18 engine budget
   plus headroom. The initial guard requires at least 10 GiB free VRAM;
   the running guard aborts owned work below 2 GiB free. Decline launch if headroom cannot safely accommodate this
   configuration; do not solve that by changing settings during the run.
4. The supervisor starts exactly one worker using `/home/nju/vllm-env/bin/python`, in a dedicated
   process group. Capture worker/controller PIDs and start times before
   work, and collect its descendant process tree. No GPU invisibility variables
   are set on this explicitly authorized worker.

The worker deadline is **600 seconds** from launch, including import, engine
initialization/compilation/capture and single generation. The supervisor scans
child status/tree every 0.1 seconds and ASR health every 5 seconds. A blocking
health/KFD check can add up to 6 seconds before deadline handling; teardown adds
at most 10 seconds TERM grace plus 2 seconds KILL/reap observation, then a bounded
post-release guard. These costs are outside the 600-second worker deadline and
remain in controller wall time. Any completion first observed after the deadline
is conservatively a timeout. Stage/log evidence is preserved. Only still-live
owned PIDs with matching start times receive TERM/KILL. A Linux child-subreaper adopts orphaned descendants, including a grandchild
escaping its original process group. Descendant discovery checks live ancestor
start times; a recycled parent PID or group number cannot establish ownership.
There are no broad process-name kills.
No retries, reduced-memory reruns, eager fallback, package fix or backend switch
are authorized by this protocol. An execution timeout is inconclusive about the
specific backend unless a concrete earlier traceback identifies a cause.

The supervisor directly launches and waits the worker's actual OS exit; there
is no intermediate launcher whose exit could be confused with worker success.
An outer shell must capture the controller's actual OS exit separately. Verify all owned parent/worker/descendant PIDs are gone,
KFD returns to ASR only, ASR is ready/not busy, and model/source/input fingerprints
are unchanged. Stop after release. A worker-written `completed` is insufficient
without successful OS exits and clean release. If cleanup cannot be proven, mark
release unverified and escalate rather than starting another workload.

Private evidence must include the frozen specification, archive/installed/model
identities, worker stdout/stderr, stage JSONL, failure traceback when present,
observed process tree/start times, pre/post ASR/KFD/VRAM snapshots, deadline action,
worker/controller OS exit records and final release check. Public evidence is sanitized scalar
status/hashes/length/configuration and explicit failed or unverified conditions.

The smoke reports no throughput or speedup. Later engine comparisons need
separate warmup, repeated synchronized measurement, matched lengths/load and a
context/request frontier (e.g. 128/512/2048 input lengths and 1/4/8 requests).
BF16/T1/no-filter settings do not prove identical reference float64 sampling
arithmetic or losslessness. Keep fidelity and performance conclusions separate.

CPU fixtures test exact graph-preserving settings, invalid token inputs, actual
zero/nonzero OS exit, timeout cleanup across process groups, orphan adoption,
ASR-guard failure and the recycled-parent PID ownership counterexample. They
never import vLLM or start GPU work.

The initial CPU preparation produced 35 prompt tokens (seed 20261009), request
SHA256 `8b676c697dd4abf77294f61c8b8bb7b2a7ad70ad92c26c7b5ce41863e8625237`,
and eight model/tokenizer file hashes. The first preparation failed because the
installed Transformers returned a BatchEncoding by default; explicitly setting
`return_dict=False` fixed the serialization contract. No GPU launch occurred.
Seven Linux CPU lifecycle/input tests and both real-input stdlib preflights passed.
