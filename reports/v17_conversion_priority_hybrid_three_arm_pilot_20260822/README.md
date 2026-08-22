# V17 Conversion-Priority Hybrid Three-Arm Pilot

## Scope and provenance

This report records the preregistered fixed-parent comparison of three Target-2
allocation rules:

- `HYBRID_BASE`: round-robin exploration;
- `HYBRID_BREADTH_PRIORITY`: conversion-responsibility breadth, with the frozen
  round-robin rule as the tie-breaker;
- `HYBRID_DIRECT_FLIP_PRIORITY`: direct-flip opportunity, with the frozen
  round-robin rule as the tie-breaker.

Target-1 remained the same W1 rank-1 member in every arm. The five parent
states, source-candidate budget, loss-blind revision policy, Common-Safe gate,
ranking, max-one rule, validation policy, and classifiers were frozen before
the first API call. The execution source was commit
`160e6d7777f7606e0c2fa783332d479a41352781`.

This was a no-commit fixed-parent pilot, not an online trajectory experiment.
It made no test calls, committed no prompt, and mutated no trajectory.

## Protocol result

The canonical audit passed with no blocker:

| Item | Result |
| --- | ---: |
| Parents | 5 |
| Cells | 15 |
| Conceptual branches | 30 |
| Unique branches | 15 |
| Reused branches | 15 |
| Candidate records | 44 |
| New test calls | 0 |
| Actual prompt commits | 0 |
| Trajectory mutations | 0 |

The selector intervention was real: Breadth and Direct each changed Target-2
relative to Base in 3/5 parents, and the two priority rules selected different
members in 2/5 parents.

## Aggregate outcome

| Arm | Feasible branches | Feasible candidates | WOULD_COMMIT | Deeper-support gains | Vote conversions | Vote regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 3 | 6 | 3 | 6 | 2 | 1 |
| Breadth | 3 | 6 | 3 | 6 | 2 | 1 |
| Direct-Flip | 3 | 6 | 3 | 6 | 2 | 1 |

Every preregistered parent-level comparison was a tie: the Breadth and Direct
arms each had 0 wins, 5 ties, and 0 losses against Base on their primary local
signals. The frozen diagnosis is therefore:

```text
NO_CLEAR_PRIORITY_SIGNAL
```

## Why the interventions did not change the transition

Across the three parents where at least one priority rule changed Target-2,
the eight unique intervention-relevant Target-2 branches produced no feasible
branch winner. Some branches stopped at the frozen Critic semantic gate; the
remaining generated candidates failed Common-Safe through target regression
and no target-or-vote progress, sometimes with team-vote regression.

Consequently, all three arms either selected the same feasible shared branch
or made no update. The three arms therefore had identical hypothetical
transitions even though their selectors differed.

This distinguishes the result from the earlier binary-filter pilot: the
current policies did create an allocation intervention, but that intervention
did not survive the downstream generation and feasibility pipeline.

## Local diagnosis and limits

The pilot provides no evidence that Breadth-Priority or Direct-Flip-Priority is
better than Base Hybrid on these five frozen parents. It instead exposes a
local realizability bottleneck after Target-2 allocation: the newly selected
conversion opportunities did not yield a feasible candidate under the frozen
proposal and safety pipeline.

This is a small fixed-parent mechanism study. It does not estimate online
trajectory efficacy, held-out test performance, or population-level effects,
and it does not justify changing the frozen method by itself.

## Published artifacts

- `parent_level.csv`: parent-level selection, funnel, transfer, and conversion
  outcomes;
- `branch_level.csv`: analysis-safe branch identities and feasibility counts;
- `candidate_level.csv`: analysis-safe candidate deltas and rejection classes;
- `conversion_structure.csv`: support-depth transition counts by arm;
- `summary.json`: canonical aggregate facts and call/token accounting;
- `classifier.json`: frozen decision rules and diagnosis;
- `source_freeze.json`: execution-source and registry provenance;
- `test_report.txt`: verification commands and outcomes;
- `sanitization_report.txt`: publication-safety scan result;
- `sha256_manifest.json`: hashes of the published report files.

Raw prompts, questions, answers, responses, provider details, caches,
checkpoints, and local paths are intentionally excluded.
