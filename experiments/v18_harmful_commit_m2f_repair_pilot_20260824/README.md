# V18 Harmful-Commit M2F Repair Pilot — Frozen Protocol

## Question

Can the existing M2F candidate-specific compatibility repair reduce the
train-visible collateral risk in the two harmful V18 Hybrid commits while
retaining target repair?

## Frozen cases

Only these historical candidate pools are in scope:

- Seed59 update 3;
- Seed61 update 5.

Every Common-Safe feasible source with `train_vote_loss > 0` is reconstructed.
No other parent, candidate, seed, trajectory, validation state, or test state
may be added.

## Existing M2F eligibility is normative

The experiment must call the unchanged
`multi_dataset_diverse_rl.compatibility_repair.repair_eligible()` function.
Its frozen conditions are:

```text
responsibility_gain_count > 0
loss_evidence_count > 0
source rejection reasons contain at least one of:
  target_regression
  team_vote_regression
  terminal_invalid_regression
```

The requested external pool filter (`source feasible` and
`train_vote_loss > 0`) may narrow the source inventory, but it may not replace
or widen M2F eligibility. If the existing M2F rule yields zero eligible sources,
Phase A records `STOP_INELIGIBLE_UNDER_FROZEN_M2F` and Phase B makes zero API,
Solver, evaluator, validation, and test calls.

## Frozen metrics if Phase B is admitted

For each eligible Source→Repair pair:

- target gain and retention;
- train Vote gains, losses, and net;
- Common-Safe feasibility;
- validation Vote gains, losses, net, target delta, and Oracle delta, evaluated
  only after repair generation and train-side decisions freeze.

Validation cannot enter repair generation, eligibility, Common-Safe, ranking,
or pair selection. Test is forbidden.

Target-gain retention is frozen as:

```text
sum(max(0, repair_target_gain)) / sum(max(1, source_target_gain))
```

The high-retention threshold is `>= 0.8`, matching the existing M2F analysis
convention.

## Frozen diagnosis

- `M2F_WRITEBACK_RISK_REDUCTION_SUPPORTED`: repair train Vote loss total is
  lower, target-gain retention is at least 0.8, and validation negative-net
  harmful cases decrease;
- `M2F_TRAIN_COLLATERAL_ONLY`: train loss is lower with retention at least 0.8,
  but validation harmful cases do not decrease;
- `M2F_HARMFUL`: target-gain retention is below 0.5 or validation negative-net
  cases increase;
- otherwise `M2F_NOT_SUPPORTED`.

When Phase B is not admitted because existing M2F eligibility is empty, the
only allowed label is `M2F_NOT_SUPPORTED` with interpretation
`not_evaluated_ineligible_under_frozen_m2f`.

## Hard exclusions

- changing selector, W1, Hybrid, Common-Safe, ranking, M2F prompt, retry, or
  eligibility;
- adding a train-vote-loss hard gate or risk model;
- regenerating source candidates or rerunning V18 trajectories;
- validation leakage or any test call;
- raw prompts, questions, answers, responses, credentials, endpoints, caches,
  checkpoints, or absolute paths in tracked output.
