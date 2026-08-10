# v14 Qwen3-14B Seed46 Pilot and Formal Report

- Method: `member_aware_peer_state_v14`
- Checkpoint: `23`
- Source commit: `e5bdc9f27f7a5594072aafd828c7c6053297c03c`
- Task: `disambiguation_qa`
- Seed: `46`
- Base model: `qwen3-14b`
- Thinking: disabled
- Pilot protocol gate: **PASS**
- Pilot empirical gate: **PASS**
- Formal protocol gate: **PASS**
- Pre-registered additional-seed gate: **HOLD**

The authoritative Pilot completed Static and S0-S3 without a final test. It
used 1,620,104 tokens and accepted 14 updates across the optimized settings.
The S3 RCRU flow was:

```text
24 evaluated -> 8 Layer 1 -> 7 Layer 2 -> 7 Layer 3
             -> 6 branch-selected -> 4 global commits
```

The authoritative Formal run completed all five settings. Static used zero
optimization updates; S0-S3 each completed 32 updates. Every setting selected
the final active state, evaluated the test set exactly once after training, and
made zero validation calls. The formal run used 6,096,304 tokens.

| Setting | Accepted | Final train vote /75 | Final test vote /125 | Test member counts |
|---|---:|---:|---:|---|
| Static | 0 | 50 | 85 | `[85,85,85,85,85]` |
| S0 | 4 | 55 | 90 | `[85,85,86,92,84]` |
| S1 | 6 | 55 | 90 | `[91,98,85,82,85]` |
| S2 | 9 | 60 | 93 | `[88,91,94,91,88]` |
| S3 | 7 | 58 | 87 | `[91,85,87,88,85]` |

The pre-registered expansion rule required S3 to beat Static on train and test,
have at least one accepted update, pass all safety gates, and satisfy
`S3 test vote >= S2 test vote`. All conditions passed except the last:
`87 < 93`. Therefore Seed44 and Seed45 were not started for this pipeline.
This is a mixed single-seed result, not an efficacy or generalization claim.

`seed46_pipeline_summary.json` is the compact machine-readable result and
expansion decision. `pilot/` and `formal/` contain the protocol gates and
analysis-ready numerical/hash artifacts for every setting. Prompts, questions,
gold labels, model answers, raw provider output, credentials, endpoints,
SQLite caches, checkpoints, process logs, and absolute local paths are excluded.
