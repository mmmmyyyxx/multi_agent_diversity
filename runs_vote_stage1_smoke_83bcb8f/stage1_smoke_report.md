# Stage 1 Smoke Report

## Status

Stage 1 passed on Git commit `83bcb8f90fd4b60344c283b050f5fe4e099a7727`
with protocol `vote_oriented_v3`. The tracked working tree was clean when the
runs started, and all 153 tests passed.

## Final test metrics

| Setting | Vote acc | Mean individual acc | Best individual acc | Oracle acc | Mean vote margin | Tie rate |
|---|---:|---:|---:|---:|---:|---:|
| shared_baseline | 0.475 | 0.470 | 0.500 | 0.600 | -0.060 | 0.000 |
| shared_scalar_tcs_vote_first | 0.575 | 0.545 | 0.650 | 0.750 | 0.090 | 0.000 |
| shared_vote_pareto_tcs | 0.575 | 0.555 | 0.600 | 0.775 | 0.110 | 0.000 |

This smoke is an integrity and signal-density check, not a performance claim.

## Candidate-level audit

| Metric | Scalar | Vote Pareto |
|---|---:|---:|
| Candidate rows | 12 | 12 |
| Feasible candidates | 12 | 12 |
| Optimizer / existing-beam candidates | 8 / 4 | 8 / 4 |
| Nonzero vote-delta rate | 25.0% | 25.0% |
| Nonzero margin-delta rate | 58.3% | 66.7% |
| Positive boundary-gain rate | 0.0% | 0.0% |
| Accuracy-guard rejection rate | 0.0% | 0.0% |
| Invalid-guard rejection rate | 0.0% | 0.0% |
| Active prompt changes / attempts | 4 / 4 | 4 / 4 |
| Selected optimizer candidates / attempts | 4 / 4 | 4 / 4 |
| Optimizer underfilled rate | 0.0% | 0.0% |
| Pareto front-0 mean size | N/A | 1.25 |
| Pareto front-0 candidate ratio | N/A | 41.7% |

Margin is materially denser than exact vote flips in both optimized runs. All
eight update attempts selected a newly generated optimizer candidate as the
active prompt.

## Integrity checks

- All three runs completed validation restore and final test evaluation.
- `run_meta.json` records the expected Git commit, clean tracked state,
  `checkpoint_version=2`, protocol v3, train-only candidate evaluation, and
  zero overlap among optimization, validation, and test questions.
- Candidate budgets are the requested `pool=20`, `unique=20`, `repeats=2`.
- `vote_delta` equals both candidate-minus-baseline vote accuracy and
  vote-gain-rate minus vote-loss-rate for all 24 candidate rows.
- `reward_total` equals the sum of logged reward components for all rows.
- No negative boundary reward component or clipping inconsistency occurred.
  This smoke produced no raw negative boundary delta, so the negative clipping
  branch is verified by regression tests rather than direct smoke observation.
- TCS audit found 8/8 completed groups, 16 optimizer candidates, no invalid
  metadata, no failed groups, and no delta inconsistency.

## Gate decision

Passed. Runs completed, TCS and data-role audits passed, candidates and active
prompt changes were observed, candidate evaluation was real and correctly
budgeted, numeric identities held, and margin supplied a denser signal than
vote flips. Stage 2 is authorized.
