# Explicit pinned ROCm private-ATen reference backend

`rocm_aten_no_window_pinned_v1` is an explicit, still-unverified adapter option.
The default remains the public `native_varlen` wrapper. The
[public failure](../reports/native-varlen-probe-20261009/README.md) and
[six-call alignment diagnosis](../reports/native-varlen-diagnostic-20261009/README.md)
remain separate historical results. The diagnostic covered one synthetic shape;
it did not pass the full adapter.

```python
VarlenPackedTarget(model, layer_ids, native_backend="rocm_aten_no_window_pinned_v1")
```

`dspark_qwen/rocm_varlen.py` refuses unsupported identities before native execution:
Torch2.12.0+rocm7.2, Torch git7661cd9c6b841b62b7f411aa52ec51f05457263b,
HIP7.2.53211, gfx1201, AOTriton preference, unset `TORCH_ROCM_FA_PREFER_CK`,
and the exact observed private operator schema. This is a version-scoped
experiment; the runtime pin is not a universal API compatibility promise or an
independent attestation of the loaded device-kernel binary.

Runtime checks occur once at kernel construction. Runtime/backend settings must
remain unchanged throughout its lifetime. The entry accepts only contiguous
BF16 THD tensors on the selected GPU, Hq16/Hkv8/D128, nonempty query suffixes,
int32 cumulative lengths and scale1/sqrt(128), without autograd. It calls
`aten::_flash_attention_forward` with dropout0, is_causal=True,
return_debug_mask=False, both windows=None and remaining optional controls=None.
There is no split tuning, backend-preference mutation, exception retry or fallback.
CPU test injection and explicit private-backend selection are mutually exclusive.

Input metadata checks remain on every call. Cumulative values synchronize once
per new immutable layout, reused across layers only while normal tensor version
counters are unchanged. Inference tensors have no version counter, so their
cumulative values are checked each call. Only the most recent layout is retained.
These checks are reference instrumentation, not a production timing claim.

## Full tensor gate before a whole-model gate

The original gate gains an explicit choice:

```sh
python scripts/probe_varlen.py --dry-run --backend rocm_aten_no_window_pinned_v1
```

This is standard-library-only and does not touch a GPU. Real `--execute` needs a
separately coordinated window and fresh output directory. The original public
`PROTOCOL` remains byte-for-byte unchanged as a Python literal, with JSON identity
`08c501349bb59d30a888f1a5669a6c5bcb0bdfbf5f9e894c409fb2e702d49ff9`.
The private protocol explicitly records its backend, runtime pin, None windows,
private entry and original protocol hash. All original random draw order, three
cases, shapes, thresholds, causal sentinel and poison/isolation checks stay fixed.
There are11 native calls if all original assertions pass; failure stops the gate.

Evidence handling changed: JSON writes are atomic, each native call saves QKV/cu
and raw output before finite/shape/operator/numerical assertions, and oracle
outputs plus comparison scalar errors are saved before numerical assertions.
These are control/persistence changes, not relaxed numerical criteria. Old
reports and source archives are not rewritten.

CPU verification:10 new tests plus5 target,3 controller and7 diagnostic tests
pass with HIP/CUDA/ROCR hidden. Tests include runtime-pin refusal, exact private
arguments, CPU refusal, shape/dtype/scale/cumulative/autograd guards, mutation
checks in normal/no_grad/inference_mode, unchanged public protocol, an11-call
whole tensor-gate CPU substitute and first-failure evidence preservation. A CPU
substitute is not a real private-kernel execution. The full native tensor gate
and whole pretrained Qwen/KV gate have not yet run for this new adapter.

## Proposed whole pretrained Qwen/KV gate contract (not yet executed)

Freeze this plan and a concrete script/source/config manifest before GPU
execution. Use the actual fingerprinted Qwen3-0.6B target weights/config/tokenizer,
native BF16, default RoPE and the observed Hq16/Hkv8/D128 architecture; refuse a
changed fingerprint or incompatible config. Do not replace the target with a
random tiny model. A tiny model is only for CPU control-flow tests. No development
or final-test examples, draft checkpoint, training config or generation are
needed: use predeclared in-vocabulary synthetic token sequences, persisted with
hashes. There is no token-quality or timing claim.

### Fixed request schedule

Use requests A, B, C and monotonically fresh request markers. Token IDs are
predeclared consecutive ranges within the target vocabulary (A starts100,
B starts200, C starts300); no RNG-dependent prompt selection. The script must
record exact token IDs privately, lengths and hashes publicly.

1. Prime A16, B21, C13 in one model forward (all three q=k prefixes).
2. Append A1 and B8, retaining inactive C13: q=[1,8], k=[17,29].
3. Crop A to8/B to3; remove C and re-add its external ID with a fresh marker;
   append A3/B2/C1: q=[3,2,1], k=[11,5,1].
4. Crop A to0, retain B5/C1 inactive, append A2: q=[2], k=[2], inactive6.

Execute the same chunk/position schedule on a separate BF16 dense `PackedTarget`
and on independent BF16 SDPA per-request targets. They are numerical references,
not assumed bit-identical. Never share a model instance whose attention config
is mutated between live adapters. Require one whole-model forward per packed
append and one native attention call per actual Qwen layer, with no dense fallback.

### Structural/content gates: exact checks

For every step/layer, verify integer request markers, local positions, spans and
cumulative lengths against the fixed schedule. Verify target/cache finiteness,
shape and dtype. Snapshot each request's actual KV before mutation. Crop/remove
must preserve the retained slices byte-exactly, leave unrelated requests' KV
byte-exactly unchanged and eliminate removed-marker rows. Re-added C must begin
at local position0 with a fresh marker. Every append must preserve its previous
cached prefix exactly. Instrument actual per-layer gathered KV and check it is
the expected request-local gather of physical KV, with no inactive rows; aggregate
length agreement alone is insufficient.

Predeclare two additional paired branches restored from the same pre-step2
state, including model/cache/request metadata:

- Inactive-only: append A1, compare unmodified state with C KV poisoned to
  K=100/V=−100 on every layer. Active A gathered inputs, resulting hidden/logits
  and A KV must be byte-identical.
- Other-active: append A1/B8, compare unmodified state with B history KV poisoned
  and B incoming token IDs changed to a fixed alternate range. A inputs/positions,
  hidden/logits and KV must be byte-identical. Query shape/order stay the same.

Persist raw evidence before assertions. Any failed exact invariant marks the
structural gate failed; do not relax it afterward. Instrumented metadata/gather
checks do not by themselves prove correct attention math.

### BF16 numerical comparisons: separate, predeclared reporting

Compare all query rows, per request and per stage, for final normalized hidden
states, selected layer features and per-layer K/V against the separate dense and
per-request SDPA references. Record max absolute difference, RMS, equality counts
and finite status; retain bounded private tensors. Project all compared rows
through the actual pretrained LM head. Report logit max/RMS and argmax changes,
then temperature1 unfiltered float64 normalization and per-row TV/max probability
difference using the established probability policy. Report distributions and
worst rows, not only one global maximum.

**Do not invent a BF16 agreement threshold from the observed outcome.** This
proposal gates structural invariants exactly and records numerical comparisons
without a numerical pass flag. Its terminal statuses are `structural_passed` (or
failed) and `numerical_comparison_completed` (or failed), with no blanket adapter
pass or losslessness verdict. If an overall numerical acceptance gate is wanted,
root must approve and freeze its hidden/KV/logit/TV criteria before execution;
that remains an outstanding decision. A complete tensor gate plus these reports
is necessary evidence, not automatic production promotion.

An outer controller must bind Git archive, target fingerprints, protocol and
input hashes; record live pre/post process/KFD/desktop/ASR state; enforce a
predeclared bounded timeout; preserve worker/controller exits and partial evidence.
Do not run alongside training or quality collection. No GPU authorization is
implied by this document or the explicit backend option.
