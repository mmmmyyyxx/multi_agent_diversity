# v14 Seed46 Pilot Report

- Gate: **PASS**
- Complete runs: 5 / 5
- BLOCKER: 0
- MAJOR: 0
- Total tokens: 4000324
- Audit mode: `frozen_source_execution`
- Run source commit: `1a6270ae62a899efb3c722b3a31384c96cb6f50b`
- Auditor commit: `1a6270ae62a899efb3c722b3a31384c96cb6f50b`

The five settings used one frozen Seed46 initialization with member-correct
counts `[63, 63, 63, 63, 63]`. All cache-chain, exact-observation, setting
isolation, test-isolation, and source-identity checks passed. No final test was
run.

S3 produced the decisive v14 mechanism witness:

```text
14 generated -> 1 Layer1 -> 1 Layer2 -> 1 Layer3 -> 1 accepted
```

The selected role-only candidate had target gain `+1`, vote gain `0`, one
positive support, no negative support, and bootstrap LCB `0.0`. This supports
lifting the formal-experiment HOLD, but it is not a claim of vote-accuracy
improvement: every setting finished with vote-correct count `63/75`.

`pilot_results_summary.json` is the compact result index. Each setting
subdirectory contains analysis-ready numerical/hash artifacts only. Prompts,
questions, labels, model answers, raw provider output, credentials, endpoints,
SQLite caches, checkpoints, logs, and absolute paths are excluded.
