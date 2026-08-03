# Final Method Code Audit

- Method: `member_aware_peer_state_v8`
- Audit gate: **PASS**
- BLOCKER: 0 unresolved
- MAJOR: 0 unresolved
- Offline tests after post-run strictness repair: 223 passed
- Real API calls during audit: 0

The audit and post-run completion review found and resolved four
experiment-path defects:

1. no-test baseline runs previously evaluated test unconditionally;
2. the task runner previously shared one mutable cache across settings and did
   not enforce one frozen initialization per task-seed;
3. two `ruin_names` strict-split rows had comma-split option structures.
4. independent setting caches were cloned before the baseline test, so
   unchanged prompts could receive different test observations across settings.

Defect 4 invalidates strict interpretation of the 36 already completed runs;
their published aggregates are explicitly exploratory. Future runs clone each
setting's independent mutable cache from a cumulative task-seed observation
reference and merge new exact-request entries back before the next setting;
the v2 gate requires proof of that chain.

The fixes do not change compact responsibility, TCS semantics, Stage A/B,
candidate acceptance, parser strictness, or the immutable Solver contract.
