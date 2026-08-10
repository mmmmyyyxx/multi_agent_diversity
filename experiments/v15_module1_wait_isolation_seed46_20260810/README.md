# v15 Module 1 Wait-Placement Isolation

- Task status: **COMPLETE**
- API calls: **0**
- Baseline replay: **PASS**
- Recommended design: **W1 only**
- Persistent realizability: **not supported**

This directory performs the requested two-stage fixed-trajectory isolation:

- `R0 -> W1` isolates placement of the wait term inside state repairability;
- `W1 -> W2` tests persistent Beta feasible rate;
- `W1 -> W3` tests persistent consecutive-failure discount.

W1 retains every observed feasible branch and commit target. W2/W3 introduce
retention losses and substantial ranking churn. No API, new candidate, rollout,
validation, test rerun, or formal method modification occurred.

See `recommendation.md` and `module1_wait_isolation_summary.json`.
