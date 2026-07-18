# V8 Competence-Depth Method

## Scope

V8 is an opt-in extension of Vote-Oriented v7. Existing v7 settings keep their original reward, Pareto objectives, prompt limits, best-state selector, and defaults. V8 keeps five-agent majority voting and changes only prompt evolution.

## Coverage Depth

For each example, `K` is the number of correct agents. `Ck = P(K >= k)` for `k=1..5`. Candidate transitions use paired baseline and candidate correctness matrices already produced by candidate evaluation:

- gain at depth `k`: baseline `K < k`, candidate `K >= k`
- loss at depth `k`: baseline `K >= k`, candidate `K < k`
- net delta: gain rate minus loss rate

Depth 1 is oracle coverage. With five agents and no vote tie, depth 3 is majority-vote accuracy. This adds no solver calls.

## Competence Schedule

Epoch 1 uses `specialization_strength = 0`. At the end of each complete optimization epoch, V8 computes the mean accuracy of the two weakest agents and sets the next epoch's strength to:

```text
clip((bottom2_mean_acc - 0.55) / (0.65 - 0.55), 0, 1)
```

Validation and test never control this schedule. Schedule state is checkpointed and restored exactly.

## Selection And Reward

The update selector retains v7 boundary/error pressure and adds an early competence-deficit bonus that vanishes at strength 1.

`competence_depth_pareto` uses four objectives: maximize vote gain, minimize vote loss, maximize target-agent accuracy, and maximize `(1-s)*depth2_net_delta + s*boundary_shared_error_net_gain`.

The competence reward emphasizes target accuracy, `K=1 -> K=2`, and vote gains early. It blends smoothly into the unchanged v7 vote-useful-diversity reward as `s` approaches 1. Accuracy guard tolerance is `s * configured_epsilon`.

## Progressive Residual Specialization

Only the full v8 setting scales capability affinity, coverage-gap pressure, and residual guidance by strength. Evidence collection remains active from epoch 1, while early profiles receive extra support shrinkage.

## Prompt Integrity

V8 never character-truncates candidate prompts. Prompts up to 1100 characters are accepted normally; 1101-1400 are accepted and logged; longer prompts are rejected. Candidates must end at a sentence boundary. Existing v7 settings retain legacy behavior.

## Best State And Settings

`vote_competence_first` ranks by vote, bottom-2 accuracy, C2, smaller best-minus-bottom2 gap, vote margin, mean accuracy, invalid rate, and earlier epoch.

- `shared_legacy_coverage_useful_tcs_strict`
- `shared_vote_tcs_competence_schedule`
- `shared_vote_tcs_competence_depth2`
- `shared_vote_tcs_competence_depth2_progressive_residual`

The matched v7 baseline is `shared_vote_error_pareto_tcs_residual_cycle_guard`.
