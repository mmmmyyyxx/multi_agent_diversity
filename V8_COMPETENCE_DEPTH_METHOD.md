# V8 Competence-Depth Method

## Scope

V8 is an opt-in extension of Vote-Oriented v7. Existing v7 settings keep their original reward, Pareto objectives, prompt limits, best-state selector, and defaults. The team uses plurality voting, not a strict-majority threshold.

## Coverage Depth

For each example, `K` is the number of correct agents. `Ck = P(K >= k)` for `k=1..5`. Candidate transitions use paired baseline and candidate correctness matrices already produced by candidate evaluation:

- gain at depth `k`: baseline `K < k`, candidate `K >= k`
- loss at depth `k`: baseline `K >= k`, candidate `K < k`
- net delta: gain rate minus loss rate

Depth 1 is oracle support, depth 2 is two-agent support/correct redundancy,
depth 3 is three-agent support, depth 4 is four-agent support, and depth 5 is
unanimous correctness. C3 is diagnostic and is not the general vote-accuracy
metric: two correct votes can win a plurality when wrong votes are dispersed.
Coverage-depth transitions add no solver calls.

## V8.1 Competence Schedule

Epoch 1 uses `specialization_strength = 0`. Before any update, V8.1 selects one fixed, seeded probe from the optimization split and evaluates the initial active prompts. After every epoch-end beam refresh, it evaluates the same questions with the current active prompts. Online train accuracy mixes multiple prompt generations and is diagnostic only; validation selects the best state and test evaluates that state, but neither controls the schedule.

For probe size `N`, the effective gain thresholds are `max(0.01, 1/N)` and `max(0.06, 4/N, low+1e-8)`. Raw strength linearly maps the bottom-2 gain relative to the initial probe into `[0,1]`. Strength may increase only when probe mean accuracy, C1, and C2 remain within 0.01 of their initial values. The gated target is smoothed with EMA 0.5, limited to a 0.35 increase per epoch, and is monotonic by default. The schedule version is `competence_depth_v2_opt_snapshot_c1_guard`.

A candidate-level C1 guard additionally requires `depth1_net_delta >= 0` on the paired candidate batch. It is a feasibility guard, not a reward component or fifth Pareto objective, and adds no solver calls.

Validation and test never control this schedule. Schedule state is checkpointed and restored exactly.

## Selection And Reward

The update selector retains v7 boundary/error pressure and adds an early competence-deficit bonus that vanishes at strength 1.

`competence_depth_pareto` uses four objectives: maximize actual plurality vote gain, minimize actual plurality vote loss, maximize target-agent accuracy, and maximize `(1-s)*depth2_net_delta + s*plurality_boundary_shared_error_net_gain`.

The competence reward emphasizes target accuracy, `K=1 -> K=2`, and vote gains early. It blends smoothly into the unchanged v7 vote-useful-diversity reward as `s` approaches 1. Accuracy guard tolerance is `s * configured_epsilon`.

Progressive specialization is considered exercised only when a nonzero strength is actually used by a later training epoch. A nonzero strength computed after the final epoch is reported as `activation_after_final_epoch`, not as an exercised stage.

## Progressive Residual Specialization

Only the full v8 setting scales capability affinity, coverage-gap pressure, and residual guidance by strength. Evidence collection remains active from epoch 1, while early profiles receive extra support shrinkage.

## Prompt Integrity

V8 never character-truncates candidate prompts. Prompts up to 1100 characters are accepted normally; 1101-1400 are accepted and logged; longer prompts are rejected. Candidates must end at a sentence boundary. Existing v7 settings retain legacy behavior.

## Best State And Settings

`vote_competence_first` ranks by actual plurality vote accuracy, bottom-2 accuracy, C2, smaller best-minus-bottom2 gap, plurality margin, mean accuracy, invalid rate, and earlier epoch. All vote transitions and pivotal diagnostics call the same plurality aggregator with the configured tie-break and question hash.

- `shared_legacy_coverage_useful_tcs_strict`
- `shared_vote_tcs_competence_schedule`
- `shared_vote_tcs_competence_depth2`
- `shared_vote_tcs_competence_depth2_progressive_residual`

The matched v7 baseline is `shared_vote_error_pareto_tcs_residual_cycle_guard`.
