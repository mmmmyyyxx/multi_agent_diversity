# Paper-Method Comparison

| Method | Progress semantics | Responsibility role | Tuned weights | Complexity | Decision activity |
|---|---|---|---|---|---|
| S2 | Common target-or-vote progress | None in candidate ranking | None | Low | Reference |
| v14 RCRU | Vote-or-lane progress | Primary feasibility, Pareto, ranking and hard support layers | No fitted scalar weights | High | 4/32 different from S2; includes lane-only progress |
| M3-B | Common progress | Secondary RCRU tie-break | None | Medium | 1/32 raw change; no aggregate responsibility improvement |
| Scalar family | Common progress | Equal-weight responsibility component | Fixed 1:1:1 | Low-to-medium | 1/32 raw change, 0/32 responsibility-attributable changes |

## Evaluation

The scalar family is easier to explain and ablate than v14 RCRU, and common
feasibility removes lane-only acceptance. However, all three responsibility
signals select exactly the same global winners as M3-B. Their sole difference
from S2 occurs where the selected and S2 candidates have identical lane,
support, and coalition responsibility values. The change is produced by later
non-responsibility tie-breaks, not by the scalar responsibility component.

Consequently the Seed46 fixed-pool evidence does not support presenting the
scalar family or M3-B as an independently active third paper module. S2 is the
supported method body for this evidence. This is not a general multi-seed
efficacy claim.
