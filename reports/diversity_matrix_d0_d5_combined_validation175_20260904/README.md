# D0-D5 Combined Development Validation175

This is a post-hoc development evaluation of already-frozen final states.
The original Validation50 and FormerTest125 retain separate provenance, then
correct counts are summed and divided by 175. The two accuracies are not
equally averaged. FormerTest125 is no longer an untouched held-out test.

- Training trajectories: 18/18
- Validation50 evaluations: 18/18
- FormerTest125 evaluations: 18/18
- Training rerun: false
- Checkpoint selection: none; final active state only
- Untouched held-out test remaining: false

## Aggregate Combined175 results

| Arm | VoteAcc | OracleAcc | Oracle-Vote gap |
|---|---:|---:|---:|
| D0 | 0.6705 | 0.6705 | 0.0000 |
| D1 | 0.7314 | 0.9162 | 0.1848 |
| D2 | 0.7333 | 0.9238 | 0.1905 |
| D3 | 0.7181 | 0.9105 | 0.1924 |
| D4 | 0.7295 | 0.8933 | 0.1638 |
| D5 | 0.6914 | 0.8648 | 0.1733 |
