# Matched initial32 diagnostic: CPU preparation

The [runner](../../scripts/diagnose_matched_initial32.py) implements the
[S50 design](../../docs/matched-initial32-diagnostic.md). This phase adds code and
CPU validation, with no deployment or real model execution.

The diagnostic compares checkpoints 512 and 1280 on the same 32 saved initial
states. Four historical full-law anchors must pass before the remaining states
are processed. It preserves the seven-row draft head, eight-row block target
adapter, separate sequential target pass, and the fixed step512 sequential
reference. It makes no new acceptance-rate or throughput claim.

## Validation

- **19 synthetic CPU tests passed**, OS exit 0. Tests cover shape and autocast
  scope, no-grad behavior, cache reset, four-load execution order, complete
  coverage, numerical mismatch retention, invalid laws, input identity,
  dependency-read boundaries, and resource supervision with owned child cleanup.
- The first 14-test run had one fixture failure: macOS canonicalized `/var` to
  `/private/var`. Resolving the expected path fixed the assertion. The subsequent
  18-test run and final 19-test run passed; their evidence hashes are retained in
  [validation.json](validation.json).
- The actual local `--evidence-only` dry run passed for **32 states / 64
  checkpoint-state pairs**. It hashes existing evidence without loading the
  historical probe tensors. Its `dependencies_verified` value remains **false**.
- Review also found a completion-reporting defect: a nonzero worker exit after
  writing a completed summary could leave that summary authoritative. The
  supervisor now retains it privately as unconfirmed and reports failure. Tests
  cover this case and failure of final telemetry validation.

See the [test log](tests-final.log), [source identities](source-freeze.json),
[binding summary](evidence-only.json), and [independent review](independent-review.json).

## Remaining execution gates

The real AMD runtime must expose a PCI identifier matching the physical device
monitored through sysfs. Missing identity, driver telemetry, or required memory
headroom stops preflight. Availability of those runtime properties is untested.
Deployment identity, current target/checkpoint hashes, a coordinated live GPU
window, and native historical numerical replay still need verification.

CPU checks do not establish GPU readiness, full-panel overlap, natural rollout
acceptance, or speed. Existing negative same-backend timing results and the
separate vLLM strong baseline remain unchanged. No confidence admission or
hardware-aware scheduling benefit has been demonstrated.
