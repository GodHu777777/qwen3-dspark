# Trained packed decoder: CPU-prepared GPU gate

`scripts/probe_packed_decoder.py` binds the actual Qwen3-0.6B target and selected
trained step1280 draft before a separately authorized GPU execution. This is a
correctness experiment, not a quality, scheduling-overlap or performance test.
The earlier whole-Qwen target RMS gate remains **failed and unchanged**. Even a
passing draft gate cannot establish whole-system numerical equivalence.

## Binding and execution boundary

The default and `--dry-run` paths use only the standard library. They verify the
real target architecture and model/tokenizer fingerprints against the generation
manifest, then require the exact trained checkpoint:

- Draft weights SHA256: `d6f21ab3af5187ed181efdbde54118a9ada21b458d9957515a17d6b3693d0de4`.
- Metadata SHA256: `8bb0c5113d02decac5b8a64c585284dd1664f1a5f88f13014713ec1facc7ecc5`.
- Step1280, five draft layers, block7, selected target layers1/7/14/21/26.

```sh
python scripts/probe_packed_decoder.py --dry-run \
  --model /private/Qwen3-0.6B \
  --generation-manifest /private/generated/manifest.json \
  --checkpoint /private/step-001280
```

No dataset or final-test content is read. The model generation manifest is used
only for identity. Formal execution requires Transformers5.17.0, the pinned
`rocm_aten_no_window_pinned_v1` target backend and the separate
`rocm_aten_no_window_noncausal_pinned_v1` draft backend. There is no formal CLI
tiny-model mode, backend fallback or automatic retry. Source/package hashes,
protocol and all input identities are rebound inside the worker.

The worker loads two real target model copies: the native packed target and an
independent SDPA reference. The trained draft aliases the native target's frozen
embedding/head; trainable parameters and RoPE remain FP32 under BF16 autocast.
The target is BF16. Execution requires8 GiB free memory, caps process allocation
at6 GiB, and kills only its owned worker process group after300 seconds. A fresh
source archive, live process/KFD/ASR/memory checks and a separately coordinated
GPU window remain required; this document grants no execution authorization.

## Fixed matrix

Synthetic initial prompts have lengths A16/B21/C13. Their exact token IDs and
seeds are frozen in `PROTOCOL`; C is removed after the first round and re-admitted
with a one-token prompt and a new seed/incarnation. Temperature is1, EOS stopping
is disabled and each request has output budget64. Allocations are declared before
proposal draws and do not use confidence or sampled tokens to select prefixes.

| Round | Draft mode | Verification allocation | Target query rows |
| --- | --- | --- | --- |
| shadow_one | Full seven-position shadow for A/B/C | A1/B7/C0 | 11 |
| shadow_readd | Full seven-position shadow for A/B/C | A0/B1/C3 | 7 |
| fixed_zero | Skip A's draft; B's full backbone | A0/B2; C inactive | 4 |

Zero allocation still verifies the current anchor. Initial admission and C's
re-admission add50 and1 target query rows, respectively:73 normal target query
rows in five native forwards. Each native target forward must expose28 private
flash attention events, totaling140. The three normal draft forwards require
five noncausal callbacks each. Two control/poison pairs add four draft forwards,
so there are seven draft forwards and35 draft native events overall; only the15
normal draft layer calls receive the numerical acceptance test.

The poison pairs restore the same projected-cache snapshot before each branch.
The inactive-C pair runs A alone; the active-B pair runs A/B and replaces B's
anchor as well as B's projected context. Victim K/V are set to100/−100. A's
actual layer Q/K/V/output, final hidden, base-head logits and cached KV must be
exactly unchanged. Artificial poison amplitudes do not enter the normal gate.

## Independent result states

`structural_status` requires exact old-prefix and inactive-cache preservation,
fresh C incarnation/target/draft markers and absence of removed markers. Every
projected context chunk must equal the actual committed native target features;
rejected tails cannot be projected. Actual resident target and projected draft
KV are compared across all requests, including inactive requests.

`draft_layer_numerical_status` checks each normal layer's actual native Q/K/V
against FP32 noncausal MATH SDPA with the same scale and GQA. The oracle explicitly
disables enclosing autocast and asserts FP32 output. For every pooled layer and
every request, all elements must satisfy `abs(error) <= .02 + .02*abs(reference)`
and RMS must be at most.005. The script rejects drift between these protocol
limits and its shared numerical helper. Finite threshold failures save evidence
and continue the full matrix; a failure remains failed. Nonfinite values,
structural failures or operator exceptions stop with partial evidence. Nearest
BF16 rounding error and native-versus-rounded error are separate diagnostics and
do not relax the original thresholds.

`endpoint_comparison_status=completed` means comparison data were recorded. The
independent SDPA target replays exactly the native path's actual input chunks and
committed lengths; it never creates an independently sampled trajectory. Hidden
states and selected features are compared at admission and verification. Each
verification compares logits and float64 probability TV. Resident target KV and
projected draft KV are compared at every stage; reference draft KV comes from
the original `project_context_kv` using independent target features. No endpoint
acceptance threshold is invented from observations.

The public status fields remain separate. A finite draft numerical failure
returns `completed_with_failed_draft_layer_gate` and a nonzero worker exit while
the controller may record that execution completed. `system_pass_claimed` is
always false, including when all new checks pass. Instrumented wall time is not
a benchmark.

## Evidence and CPU preparation

Private evidence includes every normal layer's Q/K/V/output/FP32 oracle and
cumulative lengths, all endpoint/cache comparison tensors, both poison branches,
and actual proposal tokens/q/confidence, target probabilities, committed tokens
and verified inputs. Shadow proposals are retained independently from selected
prefixes. Failed execution also saves available last inputs/features and cache
state. These files and machine-specific binding paths must not be published.

CPU validation uses tiny BF16 models solely as test fixtures. The final five-test
suite passed in5.948 seconds with GPUs hidden. It covers stdlib binding/refusal,
helper-threshold consistency, owned-group timeout, complete lifecycle/poison flow
and a deliberate finite attention bias that must retain failed numerical status
while completing the matrix. The saved oracle tensors are checked to be FP32.
The real target/checkpoint standard-library dry-run also passed with GPUs hidden.
Neither result exercises the private ROCm operator.

The initial four-test run passed before review found an oracle mistake:
`.float()` inputs and `sdpa_kernel(MATH)` alone do not disable enclosing BF16
autocast. The corrected oracle explicitly enters `torch.autocast(...,
enabled=False)` and the final suite adds persisted-dtype proof. The original log
is retained as development history, not presented as valid FP32-oracle proof.
CPU preparation evidence and hashes live privately under
`output/packed-decoder-gate-cpu-20261009/`. No real GPU result exists at this
preparation milestone.


## Subsequent execution

The immutable8378ea7 GPU matrix completed with worker/launcher/controller exit0,
75 structural checks and all15 pooled/35 request draft comparisons passing.
Same-input verification probabilities still differ on18/22 rows, max TV0.02938953.
See [the complete aggregate report](../reports/packed-decoder-gate-20261009/README.md)
for saved-tensor audits, physical work counts and the external controller's
PID-reuse limitation. The prior target RMS failure remains unchanged; this was
not a performance benchmark or a whole-system equivalence result.
