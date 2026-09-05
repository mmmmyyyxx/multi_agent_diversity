# Vote-Aligned Static/P0/P1 Confirmatory Replication

The frozen classifier is `CONFIRMATORY_REPLICATION_NOT_SUPPORTED`. Seed76
satisfied every strict requirement, but Seed77 did not: the P1-minus-P0
Validation ensemble-gain contrast was negative and P1 placed fewer rows at
`G>=3` than P0. Seed75 is contextual prior evidence and is excluded from this
two-seed confirmatory decision.

## Validation50 results

| Seed | Arm | Vote | MeanMember | Vote-Mean | Oracle | G>=1 | G>=3 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 76 | Static | 0.560 | 0.560 | 0.000 | 0.560 | 28 | 28 |
| 76 | P0 | 0.600 | 0.584 | 0.016 | 0.840 | 42 | 30 |
| 76 | P1 | 0.660 | 0.580 | 0.080 | 0.920 | 46 | 33 |
| 77 | Static | 0.560 | 0.560 | 0.000 | 0.560 | 28 | 28 |
| 77 | P0 | 0.680 | 0.608 | 0.072 | 0.860 | 43 | 34 |
| 77 | P1 | 0.640 | 0.636 | 0.004 | 0.920 | 46 | 32 |

## Frozen contrasts

| Seed | P1-P0 Vote-Mean | P1-Static Mean | P1-Static Vote | P1-P0 G>=3 | P1-P0 G>=1 | All requirements |
|---:|---:|---:|---:|---:|---:|---|
| 76 | +0.064 | +0.020 | +0.100 | +3 | +4 | PASS |
| 77 | -0.068 | +0.076 | +0.080 | -2 | +3 | FAIL |
| Mean | -0.002 | +0.048 | +0.090 | +0.5 | +3.5 | — |

P1 improved absolute Validation Vote and MeanMember over Static on both new
seeds. It also increased Oracle and `G>=1` relative to P0 on both seeds. The
plurality-effective redundancy result did not replicate consistently: Seed76
moved three additional rows to `G>=3`, whereas Seed77 moved two fewer rows to
`G>=3` and P1 underperformed P0 in Vote despite stronger MeanMember and Oracle.
The evidence therefore supports a repeatable coverage/competence effect, not a
repeatable conversion of that coverage into plurality-effective redundancy.

## Protocol and audit

- Solver: `qwen3-8b`, thinking disabled.
- Teacher/Critic/Student/Evaluator: `qwen3.7-flash`; semantic Critic retained.
- Four trainable trajectories completed by the frozen six-no-commit early
  stop; commits were 5/9 for Seed76 P0/P1 and 7/6 for Seed77 P0/P1.
- Static made no training update, target selection, candidate generation, or
  commit.
- Shadow50 and Validation50 were evaluated only after all four trainable final
  states were frozen. Test50 calls were zero.
- The first post-run audit HOLD is preserved. It was caused by the reused
  Seed75 inventory validator retaining its import-time Seed75 default scope.
  The offline scope-binding repair made no model call and the corrected audit
  passed against the unchanged execution artifacts.

This is a two-seed prospective confirmatory result. It does not justify tuning
the scheduler, adding another seed, or rewriting the frozen classifier.
