# Optional canonical target execution

`CanonicalTarget` is an experimental numerical control alongside the ordinary
`CachedTarget`. It does not replace the default DynamicCache path or DSpark's
variable global verification-budget scheduler. Mathematical acceptance rules,
floating-point agreement with a particular target runtime, and speed are separate
claims; see `dspark-reproduction-scope.md`.

## Interface and invariants

- Construct with a fixed request capacity and query width, for example
  `CanonicalTarget(model, layer_ids, capacity=256, query_width=8)`.
- `prefill` processes the full prompt once. Both canonical sequential and
  speculative routes use that same forward and full-prompt LM-head projection.
- `append` accepts 1 through `query_width` real tokens, computes a full padded
  query, and returns only the real rows in `features.last` and `features.context`.
  `features.padded_last` retains every computed row for the output projection.
- Use `predict(features, last_only=...)`. Canonical prediction projects the full
  padded hidden tensor before selecting real rows. Direct `logits(hidden)` is
  rejected to prevent accidental M=1 versus M=8 changes. Ordinary CachedTarget
  prediction keeps its existing requested-row projection and has no padding.
- Returned `start/end`, logical cache length and draft features exclude dummy
  rows. Absolute `crop` excludes rejected real suffixes. The latest emitted
  anchor remains unprocessed at the speculative-loop boundary.
- Every key position is masked by both absolute causality and logical validity.
  Stale physical KV beyond the retained prefix cannot be used as context.
- Allocate physical headroom for a complete final query even near EOS/output
  limits. Capacity is fixed for the request; the adapter does not silently grow
  or switch buckets. Invalid capacity is an error, not truncation.

HF 5.17 StaticLayer has no crop method. This pinned adapter updates its
`cumulative_length` tensors in place while preserving KV backing addresses.
`validate_cache()` checks counters and capacity explicitly. Static KV does not by
itself make the implementation a CUDA-graph engine: masks, inputs and other
buffers still need a graph-ready lifecycle, and graph execution is unimplemented.

## Evidence and limits

`reports/static-shape-candidate-20261008` records the earlier target-only oracle
probe and original failing-prefix replay. `reports/canonical-real-draft-gate-20261008`
records the actual aligned draft checkpoint: 3/3 canonical comparisons passed
(70 tokens), but only 2/3 matched stock BF16 cached output. Both dev examples
accepted zero draft tokens. The native dynamic BF16 failure report remains valid.

These small checks support an optional execution contract, not a guarantee for
arbitrary prefixes, capacity buckets, batch sizes, devices or kernel versions.
The next fidelity panel must include longer held-out inputs and boundary cases.
Do not silently change greedy argmax, discard failing examples or treat an FP32
control as proof that BF16 is resolved.

## Reproduce the private real-draft gate

Use an existing private `eval_cached_decode` result and a fresh output directory:

```sh
python -m dspark_qwen.eval_canonical_decode \
  --prior-gate /path/to/prior/result.json \
  --output /path/to/new/canonical-gate \
  --capacity 256 --max-new-tokens 32
```

`--dry-run` checks example coverage, headroom and target/data identity without
opening a GPU. The GPU gate retains all prior examples, including failures, loads
the real checkpoint, saves private token traces, and compares canonical and stock
targets separately. It is not a performance benchmark.

For performance, compare canonical speculative execution with both canonical
sequential and stock target-only execution at the same precision and workload.
Report prefill, padded physical work, draft and feature-cache costs, and global
logical/physical verification budgets. Permanent maximum padding can erase the
SPS(B) cost boundaries the paper's scheduler exploits; this control must not
silently become a substitute for that scheduler.
