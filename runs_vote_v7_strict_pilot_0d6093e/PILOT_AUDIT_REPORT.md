# Vote-Oriented v7 Strict Pilot Audit Report

## 1. Executive conclusion

Recommendation: **GO** for the next four-task, multi-seed experiment.
All 20 matched pilot runs completed. Integrity conclusions are strong; method-effect and statistical conclusions remain pilot-level because only two seeds were used.

## 2. Exact commit and environment

- Commit: `0d6093e5944c56fc9873a5296b6ee154f8469b65` (`Add matched static Vote-Pareto baseline`)
- Pre-run repository state: clean
- Tests: 187 passed, 0 failed, 0 errors
- Models: deepseek-chat for solver, optimizer, and evaluator

## 3. Strict split audit

Both tasks used the fixed task-manifest splits. Optimization/validation/test file hashes differ, all three pairwise question overlaps are zero, and leakage_warning is false for all 20 runs.

## 4. Matched configuration audit

Matched configuration pass: **True**. Relative to the static setting, observed semantic differences exactly match the intended cumulative ablations; operational `out_dir` differences were ignored.

## 5. TCS integrity

All 20 `audit_tcs_run.py` checks report `problems=false`; invalid candidate metadata count is zero. Total optimizer candidates: 1632.

## 6. Main accuracy results

See `pilot_metric_summary.md` and `pilot_metric_summary.csv`. The largest pilot vote gains over static occur on disambiguation_qa, while sports_understanding is mixed across ablations.

## 7. Shared-error and pivotal analysis

All optimizer candidates in the three error-aware settings contain the required paired fields. Paired/evidence question hashes are optimization-only. Aggregate dependence-guard rejection rate is 0.355, below the 0.80 halt threshold.

## 8. Residual specialization analysis

Profile contamination audit pass: **True**. Per agent, persisted profile update counts never exceed eligible active top-1 changes plus accepted existing-beam refreshes. No validation/test hash appears in capability evidence.

## 9. Cycle/trust guard analysis

Cycle/trust settings generated candidates without systematic depletion. Cycle guard directly rejected 0 candidates, so this pilot does not demonstrate cycle-guard activation; trust region rejected 4. Aggregate forced-current fallback rate across v7 settings is 0.000.

## 10. Candidate generation and fallback behavior

Aggregate v7 active-prompt update rate is 0.643; optimizer underfill rate is 0.094; forced-current fallback rate is 0.000. These remain below the specified degeneration thresholds.

## 11. Cost and cache savings

Across 20 runs: 83620 API calls, 44075651 tokens, and 247382 candidate-evaluation calls saved versus naive rollout. See `pilot_cost_summary.csv`.

## 12. Paired statistical analysis

Paired results use identical task/seed/question hashes, exact McNemar tests, and 10,000-draw paired bootstrap intervals. With two seeds, these are pilot evidence rather than final significance claims.

- disambiguation_qa / shared_vote_pareto_tcs_boundary_selector: delta=+0.0833, win/loss/tie=15/5/100, p=0.0414, 95% CI [+0.0167, +0.1583]
- disambiguation_qa / shared_vote_error_pareto_tcs: delta=+0.0250, win/loss/tie=7/4/109, p=0.5488, 95% CI [-0.0250, +0.0833]
- disambiguation_qa / shared_vote_error_pareto_tcs_residual_specialization: delta=+0.0417, win/loss/tie=7/2/111, p=0.1797, 95% CI [-0.0083, +0.0917]
- disambiguation_qa / shared_vote_error_pareto_tcs_residual_cycle_guard: delta=+0.0750, win/loss/tie=12/3/105, p=0.0352, 95% CI [+0.0167, +0.1417]
- sports_understanding / shared_vote_pareto_tcs_boundary_selector: delta=-0.0167, win/loss/tie=4/6/110, p=0.7539, 95% CI [-0.0667, +0.0333]
- sports_understanding / shared_vote_error_pareto_tcs: delta=+0.0083, win/loss/tie=5/4/111, p=1.0000, 95% CI [-0.0417, +0.0583]
- sports_understanding / shared_vote_error_pareto_tcs_residual_specialization: delta=-0.0333, win/loss/tie=3/7/110, p=0.3438, 95% CI [-0.0833, +0.0167]
- sports_understanding / shared_vote_error_pareto_tcs_residual_cycle_guard: delta=+0.0000, win/loss/tie=5/5/110, p=1.0000, 95% CI [-0.0500, +0.0500]

## 13. Failures or anomalies

- The pilot was interrupted once by an account-status/overdue-payment API error and resumed from checkpoint after service restoration.
- A later capacity-429 burst exhausted one validation call's three retries; the same frozen command resumed at epoch 1 cursor 60 without repeating training.
- Other capacity 429 and connection errors recovered through retry. No run metadata, split, TCS, or checkpoint integrity failure remained.

## 14. Go/No-Go recommendation for formal experiment

**GO**, with the frozen matched design and strict split protocol retained. Integrity and search-health gates pass, and at least one non-destructive improvement trend is present. Do not treat this two-seed pilot as proof of a final accuracy advantage; the formal experiment must use more tasks and seeds.

### Conclusion types

- Run integrity: 20/20 complete; strict split, matched configuration, TCS metadata, test-once, and profile contamination checks pass.
- Method trend: disambiguation_qa shows encouraging vote-oriented gains; sports_understanding is mixed, so no universal superiority claim is supported.
- Statistical significance: not established by this pilot; inspect `pilot_paired_statistics.csv` and confirm with the formal multi-seed experiment.
