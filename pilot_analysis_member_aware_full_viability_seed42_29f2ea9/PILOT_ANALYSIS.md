# 8-Update Full Viability Pilot

## Verdict

The `shared_member_aware_full` pipeline is operationally viable on the
`disambiguation_qa` seed-42 pilot at commit
`29f2ea9faf422c3c3277f03594af4a5f2f0042a6`.

This is a viability result, not an efficacy result. Six of eight updates
completed Teacher, Critic, Student, Stage A, and Stage B; four updates committed
prompt-team transitions. Validation still selected epoch 0, so the selected
test team remained the initial team and test accuracy did not change.

## Exact run identity

```text
method_version: member_aware_peer_state_v2
tcs_protocol_version: aggregated_small_model_tcs_v2
checkpoint_version: 7
experiment_setting: shared_member_aware_full
task: disambiguation_qa
benchmark: BBH
seed: 42
train / validation / test: 75 / 50 / 125
git commit: 29f2ea9faf422c3c3277f03594af4a5f2f0042a6
git dirty: false
config fingerprint: 1d20bf358b0bdbc92948bf86fcea3fc8b28d0abf822b9e2e1e05d73ba961d111
```

The recorded opt/validation/test overlap counts are zero. The fixed-probe
context used all 75 optimization examples programmatically while exposing at
most three aggregated patterns and three representative cases to the role
models.

## Viability gate

| Check | Result |
|---|---:|
| Updates | 8 |
| Distinct targeted members | 5 |
| Transport failures | 0 |
| Provider truncations | 0 |
| Updates with valid Student candidates | 6 |
| Updates reaching both Stage A and Stage B | 6 |
| Required Stage A/B updates | 6 |
| Accepted updates | 4 |
| Responsibility refreshes | 5 |

The responsibility lifecycle invariant passed:

```text
responsibility refreshes = 1 initial refresh + 4 accepted transitions = 5
```

The target sequence was:

```text
0, 4, 1, 1, 2, 3, 0, 4
```

Every member was selected at least once.

## Role and candidate funnel

| Metric | Count |
|---|---:|
| Teacher calls / schema-valid | 13 / 11 |
| Critic calls / schema-valid | 11 / 11 |
| Student calls / schema-valid | 6 / 6 |
| Truncated role rounds | 0 |
| Raw candidates | 12 |
| Schema-valid candidates | 12 |
| Stage A candidates | 12 |
| Stage B candidates | 12 |
| Constraint-feasible candidates | 5 |
| Acceptable candidates | 5 |
| Accepted updates | 4 |

Two updates did not reach Student or empirical evaluation:

- update 0 ended with `teacher_schema_exhausted`;
- update 2 ended with `critic_semantic_rejection_exhausted`.

Neither was a transport failure or provider truncation. The remaining six
updates each produced two valid candidates and evaluated both in Stage A and
Stage B. Updates 3 and 7 reached Stage B but produced no feasible/acceptable
candidate; those are legitimate empirical outcomes rather than pipeline
failures.

## Per-update audit

| Update | Target | Terminal class | Valid | Stage A | Stage B | Feasible | Acceptable | Accepted |
|---:|---:|---|---:|---:|---:|---:|---:|:---:|
| 0 | 0 | `teacher_schema_exhausted` | 0 | 0 | 0 | 0 | 0 | no |
| 1 | 4 | `none` | 2 | 2 | 2 | 1 | 1 | yes |
| 2 | 1 | `critic_semantic_rejection_exhausted` | 0 | 0 | 0 | 0 | 0 | no |
| 3 | 1 | `none` | 2 | 2 | 2 | 0 | 0 | no |
| 4 | 2 | `none` | 2 | 2 | 2 | 2 | 2 | yes |
| 5 | 3 | `none` | 2 | 2 | 2 | 1 | 1 | yes |
| 6 | 0 | `none` | 2 | 2 | 2 | 1 | 1 | yes |
| 7 | 4 | `none` | 2 | 2 | 2 | 0 | 0 | no |

The two pre-evaluation terminal failures are kept separate from empirical
candidate rejection through `empirical_feedback_available=false`.

## Validation and test interpretation

Validation selected epoch 0:

```text
selected_epoch: 0
selection_changed: false
```

Consequently, initial and selected test results are identical:

```text
vote correct: 61 -> 61
vote accuracy: 0.488 -> 0.488
per-agent correct counts: [61, 61, 61, 61, 61] -> [61, 61, 61, 61, 61]
member gain counts: [0, 0, 0, 0, 0]
```

This does not negate the viability result. It means the four training-time
transitions did not produce a validation-selected team under the current hard
guards and validation key. Efficacy requires a matched comparison on the same
final commit; it cannot be inferred from this single-setting viability run.

## Cost

```text
total calls: 1130
failed attempts: 0
total tokens: 877741
solver tokens: 825429
teacher tokens: 25203
critic tokens: 22179
student tokens: 4930
tokens per successful candidate: 73145.0833
tokens per Stage A candidate: 73145.0833
tokens per accepted update: 219435.25
```

These values are post-hoc accounting only and did not terminate or alter the
experiment.

## What should happen next

The next experiment should be a matched same-commit comparison of:

```text
shared_baseline
shared_member_aware_full
```

Only after that comparison should the study add
`shared_peer_state_member_pareto` and
`shared_member_aware_responsibility` to separate the effects of member-aware
Pareto selection, responsibility attribution, and responsibility-conditioned
proposal context.

The raw runtime directory, SQLite cache, full LLM call log, response excerpts,
and per-question rows remain local and ignored. The adjacent
`pilot_summary.json` contains the compact metrics and SHA-256 hashes of the
source artifacts used for this report.
