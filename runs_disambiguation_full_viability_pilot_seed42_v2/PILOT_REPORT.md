# Full Viability Pilot V2

## Status

Passed the Stage-1 full-viability criteria.

Both settings use the same clean source identity:

```text
f13dd7712ec8961550f9e4168c815c68c245f733
```

Configuration:

```text
task=disambiguation_qa
settings=shared_baseline,shared_peer_state_full
seed=42
epochs=1
opt/val/test=75/50/125
candidates_per_update=3
stage_b_budget=2
teacher/critic/student_max_tokens=1800/4000/4000
```

The total call and total token budgets were disabled. Actual cost is recorded
below.

## Final Metrics

| Setting | Vote accuracy | Mean individual accuracy | Minimum individual accuracy | Invalid rate | C0 |
|---|---:|---:|---:|---:|---:|
| Shared baseline | 0.4400 | 0.4400 | 0.4400 | 0.0000 | 70 |
| Peer-state full | 0.4400 | 0.4400 | 0.4400 | 0.0000 | 70 |

The equal final metrics are expected in this run. Search accepted five training
updates, but validation rejected the evolved state and restored the initial
shared prompts before final test.

## TCS Stability

| Role | Calls | Schema-valid | Rate |
|---|---:|---:|---:|
| Teacher | 11 | 11 | 100% |
| Critic | 11 | 11 | 100% |
| Student | 8 | 8 | 100% |

Critic approved 8 of 11 valid proposals, an approval rate of 72.7%. This is
neither zero nor a near-unconditional pass rate.

No TCS call failed because of malformed or truncated JSON.

## Search Funnel

| Metric | Count |
|---|---:|
| Updates | 8 |
| Requested/raw/schema-valid candidates | 24 / 24 / 24 |
| Stage A evaluations | 24 |
| Stage B evaluations | 16 |
| Feasible candidates | 7 |
| Acceptable candidates | 7 |
| Accepted updates | 5 |

Stage B guard diagnostics:

```text
rejected_local_accuracy=2
rejected_invalid=5
rejected_vote_loss=6
rejected_pivotal_loss=6
```

The counts can overlap because a candidate may violate more than one guard.

## Responsibility And Isolation

- Selected target agents: `0, 1, 4, 4, 4, 3, 3, 3`.
- Target selection counts: `0:1, 1:1, 2:0, 3:3, 4:3`.
- Final owner distribution: `0:0, 1:6, 2:7, 3:8, 4:5`.
- Sample-memorizing candidates: 0.
- Forbidden context-field violations: 0.

Responsibility moved across four target agents and did not remain concentrated
on one agent.

## Validation Selection

The evolved validation state was:

```text
vote_acc=0.7000
mean_individual_acc=0.6520
min_individual_acc=0.5400
invalid_rate=0.0040
```

Validation marked this state infeasible. In particular, one missing
`FINAL_ANSWER` increased invalid rate above the zero-invalid initial state.
`best_prompts.json` therefore contains the restored shared initial prompts.
This confirms that validation selection and rollback operated as designed.

## Cost And Transport

Full-run cost:

```text
total_llm_calls=1793
successful_llm_calls=1779
failed_llm_attempts=14
total_tokens=1475862
```

The 14 failed attempts were solver-side HTTP 429 responses and were recovered.
No optimizer or evaluator transport attempt failed.

## Outcome

V2 passes the full-viability protocol:

- non-degenerate Critic approval;
- stable Teacher/Critic/Student JSON;
- sufficient Stage A/B traffic;
- feasible and accepted candidates;
- explainable guard rejection;
- non-degenerate responsibility movement;
- no sample memorization or context leakage;
- functioning validation selection and rollback;
- auditable, finite realized cost.

The next experiment may expand seeds or tasks without changing reward or adding
method components.
