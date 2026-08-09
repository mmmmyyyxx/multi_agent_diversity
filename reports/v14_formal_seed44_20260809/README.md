# v14 Disambiguation QA Formal Seed44 Report

- Seed44 execution: **COMPLETE**
- Protocol gate: **PASS**
- Module 1 gate: **PASS**
- Module 2 gate: **PASS**
- Module 3 gate: **PASS**
- Full-method health gate: **PASS**
- Source commit: `1a6270ae62a899efb3c722b3a31384c96cb6f50b`
- Source-tree hash: `57598daaf6b6e0528df4f1f033a658e576f84b6b3683ec0cab416f0421b4e8ab`

Seed44 completed all five settings. Each optimized setting completed 32 updates;
Static completed with zero optimization updates. All settings used the same
frozen initialization, and the cache-chain, exact-observation, setting-isolation,
test-isolation, and source-identity checks passed.

| Setting | Updates | Accepted | Final train member counts | Train vote | Final test vote |
|---|---:|---:|---|---:|---:|
| Static | 0 | 0 | `[61,61,61,61,61]` | 61/75 | 92/125 |
| S0 | 32 | 5 | `[68,65,65,65,61]` | 67/75 | 101/125 |
| S1 | 32 | 5 | `[65,63,63,65,63]` | 67/75 | 102/125 |
| S2 | 32 | 4 | `[61,64,61,63,65]` | 63/75 | 100/125 |
| S3 | 32 | 7 | `[64,66,61,61,64]` | 64/75 | 99/125 |

The final-test results are report-only and were not used for gating, model
selection, diagnosis, scheduling, or candidate acceptance. A single seed is not
an efficacy or generalization claim.

The S3 RCRU path evaluated 43 candidates: 11 passed Layer 1, 10 passed Layer 2,
10 passed Layer 3, and 7 updates were committed. One real two-branch winner
competition occurred. No objective regression, multiple-commit, non-finite
value, cache-path, or accepted-candidate safety violation was found.

`seed44_results_summary.json` is the compact result index and
`seed44_gate_summary.json` is the complete machine-readable health-gate report.
Each setting directory contains analysis-ready numerical/hash artifacts only.
Prompts, questions, gold labels, model answers, raw provider output,
credentials, endpoints, SQLite caches, checkpoints, logs, and absolute paths
are excluded.

The broader Seed44/45/46 formal experiment was subsequently stopped by the user
during Seed45. This report contains only the completed Seed44 evidence and does
not characterize the partial Seed45 run.
