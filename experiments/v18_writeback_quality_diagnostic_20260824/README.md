# V18 Common-Safe / Write-Back Quality Diagnostic — Frozen Design

## Scope and units

This is a zero-API, validation-labeled but train-evidence-only diagnostic of the
completed V18 trajectories. It changes no method, selector, gate, ranking,
candidate, or artifact and does not access test.

The units are frozen before classification:

- 18 accepted commits across both arms;
- 11 accepted Hybrid commits;
- 7 Hybrid validation collateral-loss **events**, concentrated in 2 commits;
- 5 Hybrid validation gain **events**, concentrated in 3 commits;
- gain-bearing and loss-bearing commits may overlap.

The event counts must not be described as 7 collateral commits versus 5
beneficial commits.

## Train-side evidence

For every accepted candidate, the audit reconstructs only existing train-side
fields:

- Common-Safe guard outcomes;
- target gain;
- train vote gain, loss, and net counts;
- pivotal and unique-correct losses;
- coverage loss;
- soft-utility delta;
- assigned-residual repair count;
- candidate stage;
- feasible-pool size and branch-winner count.

The audit verifies the persisted `common_monotone_safe` ranking within each
branch and the persisted cross-branch winner. It never evaluates an uncommitted
candidate on validation and makes no counterfactual quality claim about one.

## Frozen risk signals

The following existing train-side binary signals are evaluated without fitting
weights or thresholds to validation:

1. `train_vote_loss_positive`;
2. `train_pivotal_loss_positive`;
3. `train_unique_loss_positive`;
4. `train_coverage_loss_positive`;
5. `train_vote_gain_and_loss_cooccur`;
6. `train_soft_utility_negative`;
7. `train_target_only_progress`;
8. `generic_revision_stage`;
9. `assigned_residual_repair_positive`.

For each signal, the report gives counts only: flagged commits, validation-loss
commits, negative-net commits, precision, sensitivity, and false positives.
No significance, generalization, or tuned cutoff claim is allowed.

## Gate-versus-ranking classifier

- `COMMON_SAFE_RISK_ADMISSION_SUPPORTED` when every Hybrid negative-net commit
  passed all current Common-Safe guards and had positive train vote losses.
- `SAFER_FEASIBLE_ALTERNATIVE_AVAILABLE` requires a candidate in the same
  frozen update with strictly fewer train vote losses than the committed
  candidate; zero-loss availability is reported separately.
- `RANKING_MISSELECTION_SUPPORTED` requires a Hybrid negative-net commit for
  which the selected candidate was not minimum-loss and such a lower-loss
  feasible alternative existed.
- `FEASIBLE_SET_QUALITY_GAP_SUPPORTED` requires at least one Hybrid negative-net
  commit with no zero-train-vote-loss feasible alternative.

This does not prove that a zero-loss guard is the correct method. It only asks
whether the completed candidate pool contained an observable risk signal and
whether ranking could have avoided it without changing the feasible set.

## M2F / compatibility availability

The signal is classified as available only if the frozen V18 registry enabled
compatibility repair, compatibility events exist, and a candidate-level
compatibility or responsibility-contribution value was persisted before
write-back. Missing evidence is reported as unavailable, never inferred or
backfilled.

## Hard exclusions

- API/model/Solver/Teacher/Critic/Student/evaluator calls;
- test access or new validation evaluation;
- candidate replay, trajectory replay, or counterfactual commit;
- method, Common-Safe, ranking, selector, M2F, or compatibility changes;
- raw prompts, questions, answers, responses, endpoints, credentials, caches,
  checkpoints, or absolute paths in tracked output.
