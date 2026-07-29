# Seed44 Source-Revision Delta Audit

`v6_seed44_control` used `5ad0fb9`; `v6_seed44_memory` used `50bbc70`.
The source delta is baseline-only for the Full setting, but the paired runs
also have different initial train states. The comparison is therefore labelled
**unmatched**; it is not source-matched or near-matched.

## Changed files

- `scripts/preflight_member_aware.py`
- `scripts/run_task_level_accuracy.py`
- `tests/test_preflight_member_aware.py`
- `tests/test_task_runner.py`

## Semantic assessment

- Responsibility assignment, target scheduler, candidate acceptance, Solver contract, and Full proposal-memory implementation: unchanged by this revision delta.
- The runner and run-specific preflight now force `shared_baseline` to `proposal_memory_mode=off`; `shared_member_aware_full` retains its requested mode.
- This fixes the baseline launch path. The Seed44 control Full run already requested `off`, so the diff does not change its Full optimization logic.

The accompanying `matched_pair_manifest.json` independently records normalized
runtime-config and initial-train-state comparisons. No conclusion should treat
Seed44 as a matched efficacy comparison unless both fields are consistent.
