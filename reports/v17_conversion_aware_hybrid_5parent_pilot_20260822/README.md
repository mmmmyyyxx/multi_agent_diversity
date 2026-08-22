# V17 Conversion-Aware Hybrid — Five-Parent Paired Pilot

## Status

This is a new prospective five-parent experiment. It does not amend or replace
the earlier six-parent preregistration, whose status remains `HOLD` with Phase B
not started.

```text
PHASE_A2_GATE = HOLD_AT_PRE_API_DIFFERENTIATION_GATE
PHASE_B_GATE = NOT_STARTED
TARGET2_DIFFERENT_COUNT = 0
TARGET2_SAME_COUNT = 5
API_CALLS = 0
NEW_TEST_CALLS = 0
ACTUAL_PROMPT_COMMIT = 0
TRAJECTORY_MUTATION = 0
```

## Five-parent all-eligible sample

The experiment deterministically reconstructed all five naturally eligible,
previously unseen historical V17 S2 parents identified by the earlier
inventory:

- Seed56 updates 4 and 7;
- Seed57 updates 6 and 7;
- Seed58 update 6.

The seed distribution is 2/2/1. No parent was added, removed, duplicated, or
weighted, and no candidate, validation, or test outcome was used.

## Selector replay

For every parent, Target 1 was the unchanged W1 Rank-1 member. Base Target 2
was the first remaining responsibility-eligible member in the deterministic RR
order. Conversion-Aware Target 2 was the first member in that same deterministic
order after filtering to members with legal responsibility for at least one
current conversion residual (`0 < G <= H`).

| Parent | W1 Target 1 | Base Target 2 | Conversion-Aware Target 2 | Different |
| --- | ---: | ---: | ---: | --- |
| Seed56 update 4 | 2 | 4 | 4 | no |
| Seed56 update 7 | 2 | 0 | 0 | no |
| Seed57 update 6 | 4 | 0 | 0 | no |
| Seed57 update 7 | 3 | 1 | 1 | no |
| Seed58 update 6 | 0 | 1 | 1 | no |

In all five states, Base Hybrid's RR exploration target was already in the
conversion-eligible pool. The proposed filter therefore produced no actual
intervention.

## Gate decision

The preregistration requires immediate STOP when
`TARGET2_DIFFERENT_COUNT = 0`. Running the ten conceptual cells would duplicate
the same two target branches within every paired parent and could not identify
an allocation effect. Consequently:

- no execution source was frozen;
- no role, Solver, or evaluator API call was made;
- no train or validation candidate rollout was run;
- no candidate, branch, or `WOULD_COMMIT` result exists;
- no scientific classifier or final diagnosis was evaluated.

This is an intervention-support failure, not evidence that conversion-aware
allocation is effective, ineffective, or harmful.

## Historical integrity

All seven tracked files in the earlier HOLD report were hashed before and after
this task. Their SHA256 values remained unchanged. The old HOLD report was not
modified.

## Reproducibility and limitations

The replay uses only historical train-state `G/H`, frozen responsibility
assignments, W1 priorities, and deterministic RR ordering. The prospective
sample remains seed-imbalanced (2/2/1), although the more immediate limitation
is that the selector contrast is identically zero on all five parents.
