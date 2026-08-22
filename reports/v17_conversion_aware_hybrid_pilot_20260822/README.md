# V17 Conversion-Aware Hybrid Pilot — Phase A Inventory HOLD

## Objective

This task preregistered a minimal fixed-parent comparison between the existing
Hybrid selector and a conversion-aware exploration filter. Phase A was required
to identify six new historical V17 S2 parents, exactly two from each of Seeds
56, 57, and 58, before any API call.

## Outcome

```text
PHASE_A_GATE = HOLD
PHASE_B_STARTED = false
PHASE_A_API_CALLS = 0
PHASE_B_API_CALLS = 0
NEW_TEST_CALLS = 0
ACTUAL_PROMPT_COMMIT = 0
TRAJECTORY_MUTATION = 0
```

The deterministic train-state inventory found only five eligible new parents:

| Seed | Eligible update indices | Required | Available |
| ---: | --- | ---: | ---: |
| 56 | 4, 7 | 2 | 2 |
| 57 | 6, 7 | 2 | 2 |
| 58 | 6 | 2 | 1 |

Seed58 therefore fails the frozen per-seed quota. The task explicitly forbids
relaxing parent eligibility, reusing historical development parents, or
changing the 2-per-seed design. Phase B was not run.

## Deterministic eligibility audit

The inventory used only pre-selection train-state information:

- seed and update index;
- reconstructable historical V17 S2 state;
- W1 total-order replay;
- deterministic responsibility-constrained round-robin order;
- responsibility-eligible member count;
- current `G` and `H` values;
- conversion residual count (`0 < G <= H`);
- conversion-eligible alternatives after excluding W1 Rank-1.

It did not use candidate feasibility, historical candidate success,
`WOULD_COMMIT`, validation outcomes, or test outcomes.

All six parents from the earlier 2x2/W1 development study and all six parents
from the Hybrid prospective pilot were excluded. The three deduplicated
conversion-audit transitions derive from the latter Hybrid parents and were
therefore excluded as well.

## Seed58 shortfall

After exclusions, the reconstructable Seed58 candidates were updates 0, 1, 3,
5, and 6. Updates 0, 1, 3, and 5 had no current conversion residual. Update 6
was the only eligible parent: it had 12 conversion residuals and four
conversion-eligible alternative members after excluding W1 Rank-1.

The shortfall is an experimental-design sampling constraint, not an API,
transport, parser, persistence, validation, or test failure.

## Decision

No selector source was frozen and no low-API execution was launched. A new
task must prospectively choose one of the following before this pilot can run:

1. provide an additional independent Seed58 S2 historical trajectory;
2. preregister a different seed-balanced sample;
3. explicitly revise the per-seed quota or parent-exclusion rule.

This report does not recommend one option and does not alter the frozen method.
