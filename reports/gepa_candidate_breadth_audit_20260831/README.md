# GEPA-style Candidate Breadth Audit

## Scope

This report is a zero-API retrospective audit of the two frozen V18 harmful
Common-Safe feasible candidate pools. It changes no method, target selector,
ranking, Common-Safe rule, M20/M2F mechanism, trajectory, or historical
artifact, and it does not access test.

The train-only selector first keeps the current Common-Safe Top-2. For
each parent residual it computes candidate-local `(DeltaVote, DeltaMargin,
target-correct)` and counts best-in-class residual coverage. The local frontier
keeps the maximum best-in-class count, then maximum repaired-residual count;
the unchanged historical quality key is only a deterministic tie-break. The
choice is frozen before existing validation labels are read.

## Phase A result

```text
parents = 2
historical candidates = 7
Top-K candidates audited = 4
winner changes = 0
aggregate validation Vote improvement = 0
aggregate validation Oracle improvement = 0
diagnosis = CANDIDATE_SELECTION_NOT_PRIMARY
```

The GEPA-style residual frontier does not provide evidence that the historical
winner-selection rule was the primary bottleneck in these frozen pools.

## Phase B status

Phase B is `NOT_RUN_PHASE_A_STOP`. The task's explicit Phase-A stop rule is
applied, and this turn also contains no authorization for new model/API calls.
Accordingly, `phase_b_candidate_pool.csv` contains only its frozen schema.
No claim is made about N=2 versus N=4 proposal breadth.
