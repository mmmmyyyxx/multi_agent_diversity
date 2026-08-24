# V18 Trajectory Gain/Loss Decomposition — Frozen Design

## Scope

This is a zero-API, validation-only, read-only decomposition of the completed
V18 `W1_TOP2` and `HYBRID_BASE` trajectories for seeds 59, 60, and 61.

It does not change a selector, prompt, candidate, acceptance rule, method,
trajectory, or frozen artifact. It does not access the test split. The four
additional Hybrid commits are an aggregate count difference, not four matched
counterfactual updates. Commits after arm divergence are never paired.

## Frozen transition reconstruction

An accepted commit is included only when its `update_lineage.jsonl` row is
committed, validation was evaluated, and its before/after validation state
indices resolve uniquely in `validation_states.jsonl`.

For each accepted transition and validation example:

- `gain`: `vote_correct` changes from false to true;
- `loss`: `vote_correct` changes from true to false;
- `net_delta = gain_count - loss_count`.

Three independent checks must agree:

1. row-level `gain_count - loss_count`;
2. the persisted `validation_vote_delta`;
3. the before/after validation metric delta.

For every seed and arm, the telescoping identity is mandatory:

```text
sum(accepted-transition net_delta)
    = final validation Vote count - initial validation Vote count
```

## Frozen persistence classes

Every gain is classified by its later validation correctness sequence:

- `retained_to_final`: never becomes wrong later;
- `overwritten_then_recovered_to_final`: becomes wrong, later becomes correct,
  and is correct at the final state;
- `overwritten_then_recovered_but_not_final`: becomes wrong, is recovered at
  least once, but is wrong at the final state;
- `overwritten_not_recovered`: becomes wrong and never becomes correct again.

Every loss is classified by the source of the current correct spell:

- `new_collateral_regression`: correctness traces continuously to the initial
  state without an intervening recovery;
- `prior_conversion_overwritten`: the current correct spell began at an
  earlier accepted validation gain.

Losses also record whether they remain wrong to the final state or are later
recovered. A commit with at least one gain and at least one loss is explicitly
marked as simultaneous local gain and collateral loss.

## Frozen transfer and quality rules

Train-to-validation transfer is evaluated per accepted commit, without
matching commits across arms:

- `positive`: validation net delta is positive;
- `neutral`: validation net delta is zero;
- `negative`: validation net delta is negative;
- `train_vote_progress_not_transferred`: train vote delta is positive while
  validation net delta is non-positive.

Commit quality is summarized per arm as positive/zero/negative validation-net
counts and proportions, mean gains, mean losses, and mean net delta per commit.

The diagnostic flags are frozen as follows:

- `collateral_regression`: Hybrid has validation losses and either more losses
  per commit than W1 or more negative-net commits;
- `transfer_failure`: Hybrid contains at least one train-vote-positive commit
  with non-positive validation net;
- `beneficial_conversion_later_overwritten`: at least one Hybrid validation
  gain is wrong at a later state;
- `higher_throughput_lower_average_quality`: Hybrid has more commits than W1
  and a lower mean validation net delta per commit.

The final diagnosis is the ordered list of all supported flags. No seed-specific
threshold or classifier is permitted. Seed61 is highlighted descriptively only.

## Hard exclusions

- API, model, Solver, Teacher, Critic, Student, optimizer, and evaluator calls;
- test files or test evaluation;
- validation selection;
- raw prompts, questions, answers, model responses, endpoints, credentials,
  SQLite/cache contents, checkpoints, or absolute paths in tracked output;
- matched-causal claims between diverged trajectories;
- significance or generalization claims from three seeds.
