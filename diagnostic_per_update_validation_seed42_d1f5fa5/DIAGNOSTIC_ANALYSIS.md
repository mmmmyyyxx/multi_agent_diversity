# Per-Update Validation Diagnostic

## Verdict

The two requested diagnostic steps are complete.

The current result should be classified as:

```text
operationally_viable = true
method_efficacy_evaluated = false
nonzero_validation_checkpoint_found = false
matched_efficacy_pilot_ready = false
```

The 8-update per-update-validation run did not find a nonzero checkpoint that
passed the existing validation guards. This rules out the narrow hypothesis
that the original one-epoch run merely missed an early feasible team because
validation was too coarse.

The dominant problem is validation generalization under the strict solver
output contract, not pipeline execution. Early accepted transitions improved
member correct counts without reducing vote count, but one missing
`FINAL_ANSWER` made those states infeasible. A later accepted transition also
regressed one member by four validation questions.

## Run identity

```text
git commit: d1f5fa5c0cbf06b3971466567d38302bea62509f
git dirty: false
method: member_aware_peer_state_v2
TCS protocol: aggregated_small_model_tcs_v2
setting: shared_member_aware_full
task: disambiguation_qa
seed: 42
train / validation / test: 75 / 50 / 125
epochs: 8
update_every: 75
updates per epoch: 1
total updates: 8
```

The manifest, dataset files, question sets, model request identity, solver
token limit, candidate budgets, and method semantics match the preceding
viability pilot except for the intended validation schedule.

## Step 1: Offline diagnosis of the original pilot

### Validation reconstruction

The original run validated only after all eight updates. Reconstructing the
four accepted intermediate teams from the local persistent solver cache gives:

| State | Member correct counts | Vote correct | Invalid rate | C0 | Ties | Feasible |
|---|---|---:|---:|---:|---:|:---:|
| Initial | 32, 32, 32, 32, 32 | 32 | 0.000 | 18 | 0 | yes |
| After update 1 | 32, 32, 32, 32, 33 | 32 | 0.004 | 15 | 0 | no |
| After update 4 | 32, 32, 36, 32, 33 | 32 | 0.004 | 10 | 0 | no |
| After update 5 | 32, 32, 36, 40, 33 | 34 | 0.008 | 4 | 0 | no |
| After update 6 | 40, 32, 36, 40, 33 | 36 | 0.008 | 3 | 2 | no |

All four intermediate teams satisfied the member competence floor and vote
floor. Every one failed only because the initial invalid rate was zero and the
candidate team produced at least one `missing_final_answer`.

Therefore, the original pilot does not support the explanation that later
updates destroyed an earlier feasible validation state. No accepted
intermediate state was feasible under the declared guards.

### Accepted optimization transitions

| Update | Target | Target gain | Vote gain | g_min | g_sum | Assigned repairs | Unique / pivotal losses |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | +17 | 0 | 0 -> 0 | 0 -> 17 | 3 | 0 / 0 |
| 4 | 2 | +24 | 0 | 0 -> 0 | 17 -> 41 | 9 | 0 / 0 |
| 5 | 3 | +32 | +15 | 0 -> 0 | 41 -> 73 | 12 | 0 / 0 |
| 6 | 0 | +32 | +8 | 0 -> 0 | 73 -> 105 | 11 | 0 / 0 |

The fixed optimization probe showed strong cumulative gains and no competence
or preservation violation for accepted candidates. The gap is between
optimization-probe acceptance and validation behavior.

### Stage-B rejections

Update 3 evaluated two candidates. Both passed the local and initial accuracy,
vote-loss, unique-correct, and pivotal-correct checks. They were rejected only
by the invalid-output guard, with three and one invalid outputs respectively.

Update 7 evaluated two candidates. Both passed local accuracy, initial
accuracy, invalid, and unique-correct checks. Both lost two pivotal-correct
cases and two vote-correct cases, so they were correctly rejected by the
vote-loss and pivotal-correct guards.

### TCS failures

Update 0 produced a valid first Teacher plan. Critic rejected it for
`actionable_specificity`. Both Teacher revision attempts then exceeded the
`repair_rule` character limit. This is a revision-stability/schema issue, not
a transport or provider-truncation failure.

Update 2 received two schema-valid Teacher plans. The first Critic rejection
used `evidence_mismatch`; the second used `actionable_specificity`. The
feedback was not identical. Critic moved from objecting to a vague fallback
criterion to demanding concrete executable disambiguation tests. This is some
evidence of an overly demanding or unstable specificity threshold, but one
failure in eight updates is not sufficient to justify changing Critic
semantics.

## Step 2: Per-update validation run

### Validation trajectory

| Epoch | Accepted this epoch | Member correct counts | Vote correct | Invalid rate | C0 | Feasible |
|---:|:---:|---|---:|---:|---:|:---:|
| 1 | no | 32, 32, 32, 32, 32 | 32 | 0.000 | 18 | yes |
| 2 | yes | 32, 32, 32, 32, 34 | 32 | 0.004 | 3 | no |
| 3 | yes | 32, 40, 32, 32, 34 | 32 | 0.004 | 2 | no |
| 4 | no | 32, 40, 32, 32, 34 | 32 | 0.004 | 2 | no |
| 5 | yes | 32, 40, 32, 28, 34 | 32 | 0.004 | 2 | no |
| 6 | no | 32, 40, 32, 28, 34 | 32 | 0.004 | 2 | no |
| 7 | no | 32, 40, 32, 28, 34 | 32 | 0.004 | 2 | no |
| 8 | no | 32, 40, 32, 28, 34 | 32 | 0.004 | 2 | no |

Epoch 1 is identical to the initial team because update 0 ended with
`teacher_schema_exhausted`. The validation key prefers the earlier initial
state, so epoch 1 is not a nonzero checkpoint.

Epochs 2 and 3 show useful cross-split competence gains:

```text
epoch 2 member gains: [0, 0, 0, 0, +2]
epoch 3 member gains: [0, +8, 0, 0, +2]
vote gain: 0
regressed members: 0
```

They nevertheless fail the exact validation protocol because one solver output
has no valid `FINAL_ANSWER` line.

Epoch 5 adds an optimization-accepted transition for member 3. On validation,
that member changes from 32 to 28 correct answers. The state therefore fails
both the invalid guard and member competence floor.

No later update changes the active team. Validation selects epoch 0, and the
selected test team is exactly the initial team:

```text
selected_epoch: 0
selection_changed: false
test vote correct: 61 -> 61
test vote accuracy: 0.488 -> 0.488
```

These test values are a consequence of validation fallback and are not an
efficacy estimate.

### Search viability

```text
updates: 8
targets: 0, 4, 1, 2, 3, 0, 4, 1
distinct targets: 5
updates reaching Stage A and Stage B: 7
raw / Stage-A / Stage-B candidates: 14 / 14 / 14
constraint-feasible candidates: 5
accepted updates: 3
terminal failures: 1 teacher_schema_exhausted
provider truncations: 1 solver finish_reason=length
transient failed attempts: 3 APIConnectionError, all recovered
responsibility refreshes: 4 = 1 initial + 3 accepted transitions
```

The TCS failure rate is 1/8, below the threshold that would justify reopening
Teacher-Critic protocol design. The pipeline remains operationally viable.

## Interpretation

The two diagnostics support three conclusions.

First, the validation schedule was not the main cause of epoch-0 selection.
Per-update validation exposed every intermediate team, but no nonzero state
passed all guards.

Second, there is genuine partial generalization. The first two accepted
transitions improved validation member counts by +2 and +8 without vote or
member regression. Their only blocker was a single missing final-answer line.
This should not be described as complete candidate overfitting.

Third, optimization-probe feasibility does not reliably predict validation
member preservation. The third accepted transition improved the training
objective but reduced its target member by four correct answers on validation.
This is a real cross-split generalization failure, not a validation granularity
artifact.

## Matched efficacy decision

A formal matched efficacy pilot should not start yet under the current
interpretation. Baseline versus Full would mostly compare the baseline with a
Full run that validation maps back to the same initial team.

Do not relax the validation competence or invalid-output guards. They are
detecting real failures and are part of the declared method.

The next focused question should be:

```text
Can responsibility-conditioned Student prompts preserve the strict
FINAL_ANSWER contract and target-member competence on unseen validation
examples while retaining their optimization-probe gains?
```

If the user elects to run the matched comparison now, it should be labeled a
proposal-stability or negative-result diagnostic, not a completed method
efficacy evaluation.

Raw SQLite caches, LLM logs, response excerpts, checkpoints, and per-question
rows remain local and ignored. `diagnostic_summary.json` contains the compact
metrics and source artifact hashes.
