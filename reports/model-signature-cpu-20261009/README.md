# Model-signature CPU/meta cost decomposition, 2026-10-09

Decision: keep the production signature unchanged. Reusing dtype/device strings
within one call reduces the median by only 1.02% locally and 2.54% on AMD; the
seven-repeat ranges overlap broadly. This bounded measurement does not establish
a dependable optimization or an end-to-end speed gain. Cross-call signature
caching is excluded because model/config/tensors remain publicly mutable and
prepare/submit/finish must retain fresh mutation detection.

Execution source: Git `d9e282d0e7ce5982a5d14bf08804f5a8c9948c6f` plus exactly the
benchmark and focused-test overlays identified in [identity.json](identity.json).
All 364 source files matched before and after the single pinned AMD window.
The original `FullTargetExecution._model_signature` implementation was measured
directly and was not edited.

## Method and results

A BF16 Qwen3-0.6B model is constructed on the meta device: 28 layers, 310 unique
parameter tensors, two buffers and 596,049,920 parameter elements. No model
weights, forward, training, or GPU initialization is used. Each row has 50 calls
per repeat, seven repeats in fixed-seed randomized order, and ten warmup calls.
The measurement has a 60-second bound inside a 90-second supervisor window.
Saved [local](local-meta.json) and [AMD](amd-meta.json) JSON retain every repeat.
Values below are median microseconds per call.

| Operation | Local Mac | Pinned AMD CPU |
| --- | ---: | ---: |
| `config_to_dict` | 11.192 | 87.650 |
| `config_repr_prebuilt` | 3.691 | 8.932 |
| `config_to_dict_repr` | 15.148 | 100.717 |
| `named_parameters_tuple` | 184.706 | 550.345 |
| `named_buffers_tuple` | 124.296 | 286.599 |
| `named_tensor_enumeration` | 311.877 | 867.150 |
| `metadata_and_version_prescanned` | 88.346 | 274.086 |
| `metadata_without_version_prescanned` | 82.258 | 249.252 |
| `version_only_prescanned` | 8.863 | 26.054 |
| `full_original` | 422.633 | 1601.134 |
| `full_recomposed` | 422.712 | 1588.138 |
| `full_local_string_candidate` | 418.322 | 1560.543 |

The measured enumeration operation is the largest isolated component. Config
serialization and fresh tensor metadata/version inspection remain material on
AMD. Individual rows have separate call/allocation overhead and are **not an
additive partition** of the full-signature row. They cannot be used to assign
percentages of GPU kernel time or explain the complete E2E throughput gap.

AMD original full signature spans 1522.485–1804.966 µs across repeats; the local
string candidate spans 1359.685–1771.143 µs. Candidate/original median ratios are
0.989798 locally and 0.974649 on AMD. The within-call candidate remains only in
the measurement script; no production optimization was adopted.

## Mutation safety

An independent eight-case CPU counterexample review found that the original
implementation rejects mutations before prepare, between prepare and submit,
between submit and finish, and during a synchronous backend wait. A test-local
cross-call cached signature admits all four. Public references, frozen autograd
parameters and stream identity do not enforce model immutability. Preserve all
three validation boundaries; no new lease/cache infrastructure is introduced.
See [safety-review.json](safety-review.json). This preserves existing refusal
semantics, not comprehensive detection of arbitrary writes (for example `.data`
can bypass version tracking). Two collector-only mistakes were recorded before
the final eight-case run exited 0; they are not hidden as successful runs.

## Validation and execution identity

- Local runtime: Python 3.14.3, Torch 2.11.0, Transformers 5.4.0, ARM64 macOS.
- AMD runtime: Python 3.12.14, Torch 2.12.0+rocm7.2, Transformers 5.17.0, x86_64 Linux.
- AMD reads only the pinned pretrained config JSON (SHA in the result), while
  local uses an architecture-matched synthetic config. Config repr sizes are
  1490 and 1463 bytes respectively. Neither config is claimed identical to a
  fully loaded pretrained runtime model after all loader modifications.
- Exact original/decomposed/candidate signature equality is checked before and
  after measurement. The one focused test also covers aliases, mixed dtype and
  CPU/meta devices, in-place buffer mutation, config mutation and training mode.
- AMD measurement, focused test, worker and SSH all exited 0. The test passed in
  5.437 s; measurement elapsed 2.642 s; supervision elapsed 16.785 s. No retry,
  full suite or GPU experiment was run in this window.
- Source and config remained unchanged. All owned process identities disappeared;
  preserved ASR identity and health passed, KFD remained owned only by ASR, and
  available VRAM was 25,252,777,984 bytes. Three GPU visibility variables were empty.

Meta tensor data pointers are zero and no GPU storage/runtime is exercised.
Machine/dependency/config differences prevent pooling the local and AMD samples.
No full CPU/runtime/GPU timeline or serving-speed attribution is supplied by this
microbenchmark. Earlier numerical-law failures and paired E2E limitations remain.

Private raw evidence: `output/signature-cpu-20261009/`, including
`pinned-run/evidence/`, source archive/overlays, actual OS exits, supervision,
release checks, and recomputed scalar audit. The public identity file hashes
these records. The worker log retains four `(null): No such file or directory`
lines emitted during dependency startup; both commands subsequently completed
successfully, and no cause is inferred from those lines.
