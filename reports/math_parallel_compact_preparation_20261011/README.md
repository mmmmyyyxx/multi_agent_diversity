# Parallel compact preparation

Current method: `unified_team_prompt_search_v2_6_parallel_compact_v1`.
Default Solver concurrency increases from 8 to 16; independent Gradient workers
increase from 2 to 8. Reflection, clustering, per-example recovery, opportunity
selection and atomic team admission retain their dependency order. Fresh
sixteen-reservation accounting retains phase token ceilings and fail-closed
durability. Original freezes, stopped-run evidence and consumed scopes are unchanged.

Optimizer input encoding replaces a duplicated written-solution view by a
smaller exact Unicode interval and SHA256 into the complete ordinary response.
Full private observations, reference solutions and provenance remain available.
Gradient and Reflection use this encoding; Solver messages and aggregation do not.
Gradient gains explicit JSON-only formatting guidance with unchanged output
limits, strict parsers, guards and three-draw recovery. New method, binding,
cache, input and prompt identities prevent reuse of old execution evidence.

Private timing records measure semaphore waiting/occupancy, each physical
transport and retry backoff. Success/error timings persist in receipts before
ledger reconciliation. Supplier internal queue time is not inferred.

Offline reencoding of sealed Optimize request messages gave the following
character counts, including the new reference/Gradient formatting instruction:

| Role | Messages | Original characters | Compact characters | Reduction |
|---|---:|---:|---:|---:|
| Gradient | 429 | 4,167,309 | 3,360,965 | 19.35% |
| Reflection | 36 | 872,876 | 622,072 | 28.73% |

Every packet decoded to its original JSON value. This compares representations
of the same historical evidence, not new model outputs. Character savings do
not establish billed-token savings or latency. Formatting guidance is not yet
shown to reduce generated-output rejection. Higher concurrency is measured
with synthetic transport only; supplier behavior requires a fresh experiment.

Verification status is in `verification.json`. Fake-provider concurrency checks
measure actual peak requests, ordered results, per-example serial recovery,
lossless packet decoding, durable receipts and reservation drainage. These are
engineering observations, not real-provider speed, token billing or efficacy.

The new manifest is DRAFT with no source freeze, execution binding or API grant.
Validation remains unauthorized and Test sealed. No real API is called.
Historical/private replay tests remain explicitly excluded by the current suite;
no claim of complete historical replay is made.

Final current coverage: **721 passed, 2 skipped**.
The full campaign initially passed 710 checks and skipped two; eleven
registration tests failed because the preceding preparation still carried the
active flag. Switching the sole active preparation to the new DRAFT corrected
this inconsistency. All eleven exact failed test identities passed on targeted
recheck; no guard or test expectation was relaxed. Final coverage combines
the full campaign with that exact recheck, rather than claiming a second full
campaign was run.
The entrypoint subset passed 112 checks; final capacity/timing regression
passed 19 checks. Compileall, current repository governance, strict DRAFT
manifest preflight, sanitization and diff checks passed. All 3,784 sealed files
matched original bytes. The explicit omitted inventory contains 220 historical
or private modules; these are not counted as passing current replay.
