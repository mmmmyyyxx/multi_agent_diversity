# RG-GEPA durable ledger and audit repair

This zero-API engineering repair resolves the retry3 accounting defect without changing the frozen scientific design. Retry3 remains a preserved `HOLD_INCOMPLETE_OPTIMIZER_LEDGER` attempt and is not repaired, resumed, or treated as mechanism evidence.

The executor now uses one append-only, flush-and-fsync ledger writer for Solver logical invocations, optimizer provider attempts, and explicit minibatch cache reuses. Reflection attempts are persisted from a `finally` block, including failed attempts when the logical call ultimately raises. Solver stage attribution is also cleared in `finally` blocks around parent initialization and candidate profiling.

Ledger schema `rg_gepa_execution_ledger_v2` distinguishes `logical_call_id`, `attempt_index`, `record_kind`, provider-attempt counts, successful-provider counts, and cache reuse. Solver records remain logical-invocation rows with physical attempt counts; optimizer records remain one row per physical provider attempt.

The production audit now requires exactly 12 logical reflection groups (`6 cases x B_0/B_1`), exactly one successful and final-success attempt per group, while allowing and charging failed retries. It strictly checks reflection attribution and reconciles runtime Solver/optimizer accounting against the durable disk ledger.

No provider, validation, or Test50 call was made. No retry4 execution was started. Historical retry roots and reports were not modified. A future API execution requires a new explicit authorization, fresh roots, and a new source freeze.

## Verification

- Focused RG-GEPA durability/audit suite: 21 passed.
- Canonical repository suite (`pytest tests`): 945 passed; one pre-existing historical-artifact test remains failed because its immutable Solver cache does not cover the exact old probe.
- Compileall: PASS.
- Deterministic production-audit fixtures: PASS, including 12 logical/12 physical and 12 logical/13 physical retry cases.
- API calls: 0.
- Validation calls: 0.
- Test50 calls: 0.
- Retry4 started: false.
