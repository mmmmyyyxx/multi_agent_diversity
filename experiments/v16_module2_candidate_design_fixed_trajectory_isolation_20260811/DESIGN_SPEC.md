# Frozen v16 Module2 Candidate Design Study

Status: `DESIGN_VARIANTS_FROZEN = true` before fixed-trajectory analysis.

This document defines a design study, not a runtime method. It preserves v15 Module1, selected targets, dual branches, candidate budget, common-safe acceptance, ranking, and max-one commit. It introduces no target-negative tradeoff, RCRU, scalar reward, or new hard preservation rejection rule.

## Design objective

Module1 already discovers complementary expertise. The isolated Module2 problem is to expose the most useful unresolved responsibility to a selected member while warning generation not to destroy existing vote-critical or stable competence.

The conceptual decomposition is:

```text
Responsibility-Conditioned Evolution
  = Repair Responsibilities
  + Preservation Responsibilities
```

## Fixed capacity

- `REPAIR_SET_MAX = 6`
- `PRESERVATION_SET_MAX = 6`

These are set capacities, not a sweep. Six repair items allow a compact boundary-first portfolio while testing whether the historical two-case context was too restrictive. Six preservation items create a symmetric but bounded guardrail set. The complexity audit must decide whether this is reasonable; it may not tune the numbers after observing results.

## Repair responsibilities

Only a selected target's frozen historical branch assignments are eligible. An item must have target wrong and team vote wrong. Exact repository plurality and enumerated repair distance determine priority:

1. `r=1, G>0`;
2. `G=1` fragmented/orphan not already tier 1;
3. `r=2`;
4. remaining assigned residuals.

Ties use question hash. Vote-correct items are never propagated as repair responsibilities. This retains unique service and minimal necessary redundancy rather than teaching every residual to every member.

## Preservation responsibilities

Only target-currently-correct examples are eligible.

- P1 vote-critical: with peers fixed, marking the target invalid changes a correct vote to incorrect.
- P2 coalition-support: the parent vote remains correct after removal, but the exact plurality margin decreases.
- P3 stable competence: remaining items for which the target was correct at every accepted state observed up to the parent.

The repository already supports validity-aware plurality, so P1 removal is not fabricated. Preservation is generation guidance only; it does not alter common-safe acceptance.

## Variants

- C0: exact historical v15 context reference.
- C1: deterministic boundary-first repair set.
- C2: C1 repair set plus deterministic P1/P2/P3 preservation set.
- C3: identical C2 membership plus compact `G`, exact `r`, boundary class, target role, and preservation-tier labels.

Because C2 and C3 have identical items, fixed-pool efficacy cannot distinguish them. C3 is eligible for recommendation only if its metadata adds meaningful, nonredundant design information sufficient to justify its character/token cost.

## Frozen F decomposition

Rules are mutually exclusive and ordered:

1. `F_LOCAL_GAIN_GLOBAL_COLLATERAL` if at least one C1 repair item improves.
2. `F_TARGET_DEGRADATION` if no repair item improves and global target gain is negative.
3. `F_LOCAL_NO_PROGRESS` if no repair item improves, target gain is zero, and vote gain is non-positive.
4. `F_OTHER` otherwise.

## Evidence boundary

Exact reconstruction can assess membership, repair distance, preservation coverage, historical deltas, and context size. Fixed-pool retrospective analysis can assess whether C0-generated candidates align with the signals. It cannot establish that C1–C3 would generate better candidates or improve accuracy. Those claims remain `REQUIRES_PILOT`.
