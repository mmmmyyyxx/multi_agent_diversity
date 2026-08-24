# V18 Harmful-Commit M2F Repair Pilot

## Gate result

```text
Phase A gate = STOP_INELIGIBLE_UNDER_FROZEN_M2F
Phase B gate = NOT_RUN_PHASE_A_STOP
Final diagnosis = M2F_NOT_SUPPORTED
Interpretation = not_evaluated_ineligible_under_frozen_m2f
```

Phase A reconstructed all seven Common-Safe feasible candidates with positive
train Vote loss from the two frozen V18 harmful pools. Source prompt hashes,
parent team hashes, targets, responsibility membership, Common-Safe outcomes,
and the historical raw artifact tree were verified.

Phase B was not run. This is a frozen applicability stop, not evidence that an
executed repair failed.

## Why eligibility is empty

Existing M2F repair requires all three conditions:

1. responsibility gain is positive;
2. candidate-specific loss evidence is positive;
3. the source was rejected for target, team-Vote, or terminal-invalid
   regression.

All 7 sources satisfy the first condition and have at least one pivotal/unique
loss evidence item. All 7 fail only the third condition: they are Common-Safe
feasible and have empty rejection-reason lists.

The requested pool filter (`feasible AND train_vote_loss > 0`) therefore finds
7 sources, while unchanged M2F eligibility finds 0. Eligibility was not widened
to force API execution.

## Source inventory

```text
source candidates = 7
historically committed harmful sources = 2
source target gain total = 39
source train Vote gains = 43
source train Vote losses = 15
source train Vote net = 28
frozen M2F eligible = 0
```

The two historically committed sources account for 7 validation loss events
and validation net -4. Validation results for the other five uncommitted
sources remain `NA`; they were not evaluated counterfactually.

## Repair metrics

```text
valid repair outputs = 0
feasible repairs = 0
zero-loss repairs = 0
lower-loss repairs = 0
targeting retained = 0 (not evaluated)
harmful validation cases before = 2
harmful validation cases after = NA
MODEL_CALLS = 0
SOLVER_CALLS = 0
EVALUATOR_CALLS = 0
NEW_VALIDATION_CALLS = 0
NEW_TEST_CALLS = 0
```

Zero repair counts mean `not attempted`, not empirical repair failures.

## Answer to the study question

The existing M2F implementation cannot currently be evaluated on these
train-visible harmful write-back cases without changing its eligibility
semantics. It is designed to repair candidates already rejected for a frozen
collateral-regression reason, whereas the V18 cases are accepted candidates
whose aggregate train gains mask per-example Vote losses.

Accordingly, this task identifies an M2F applicability gap at the boundary
between rejected-candidate repair and accepted-update quality control. It does
not authorize changing that boundary in this task.
