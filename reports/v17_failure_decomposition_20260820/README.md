# V17 Frozen Failure Decomposition

This is a zero-API observational audit of the frozen V17 trajectories. It does not rerun training, validation, or test evaluation and does not change the method. All 45 arm-by-seed-by-split cells were reconstructed exactly from existing final-state evidence.

`PRIMARY_DIAGNOSIS = TARGET_CONCENTRATION_ASSOCIATED_WITH_MEMBER_TRANSFER_REGRESSION`

`SECONDARY_DIAGNOSIS = BROAD_COMPETENCE_DEGRADATION_WITH_LOWER_UPDATE_THROUGHPUT`

The central result is not a stable optimization-residual overfit signature and not complementarity without vote conversion. S2 loses aggregate test oracle coverage (`-21` rows) as well as aggregate test votes (`-5` rows), with the vote loss occurring in two of three seeds. This is consistent with broad competence/generalization degradation. S2 also accepts `8` updates versus S1's `13`, supporting lower update throughput. S2 targeting is more concentrated in every seed; high-target members have worse transfer than low-target members in Seeds 57 and 58. This is an association only.

S3 and S4 recover three aggregate plurality-correct rows relative to S2: S2-to-S3 contributes one net row and S3-to-S4 contributes two. However, S2-to-S3 loses 18 aggregate oracle-covered rows, while S3-to-S4 restores two. The vote recovery is therefore dominated by conversion of existing coverage and changes in wrong-coalition structure, not stable recovery of broad coverage. Seed56 reverses under S3. The aggregate recovery remains two rows short of S1, and S4 remains 37 oracle-covered rows below S1. These are descriptive frozen-trajectory associations, not causal attribution to W1, R-M20, or M2F.

Historical Module1 specialization/coverage evidence remains valid, but V17 does not show an incremental final-test advantage over Generic. The two facts are compatible only as historical structural evidence plus unstable generalization; V17 itself does not support a clean complementarity-conversion explanation. Historical fixed-parent Module2 evidence also remains unchanged; this audit only characterizes its V17 transfer.

## Table 1 - Split-level performance

| Seed | Arm | Train vote | Validation vote | Test vote | Test oracle | Test oracle-vote gap |
|---|---|---|---|---|---|---|
| 56 | S0 | 0.6667 | 0.6600 | 0.6880 | 0.6880 | 0.0000 |
| 56 | S1 | 0.6667 | 0.6400 | 0.6800 | 0.8880 | 0.2080 |
| 56 | S2 | 0.7600 | 0.6800 | 0.7360 | 0.8560 | 0.1200 |
| 56 | S3 | 0.7600 | 0.6000 | 0.6880 | 0.7840 | 0.0960 |
| 56 | S4 | 0.7200 | 0.6000 | 0.6960 | 0.7840 | 0.0880 |
| 57 | S0 | 0.6800 | 0.6600 | 0.6800 | 0.6800 | 0.0000 |
| 57 | S1 | 0.8267 | 0.6400 | 0.7360 | 0.9120 | 0.1760 |
| 57 | S2 | 0.6800 | 0.6200 | 0.6800 | 0.8480 | 0.1680 |
| 57 | S3 | 0.7467 | 0.6000 | 0.7040 | 0.7680 | 0.0640 |
| 57 | S4 | 0.7600 | 0.5800 | 0.7120 | 0.8320 | 0.1200 |
| 58 | S0 | 0.6667 | 0.6200 | 0.6800 | 0.6800 | 0.0000 |
| 58 | S1 | 0.7467 | 0.5200 | 0.7200 | 0.8960 | 0.1760 |
| 58 | S2 | 0.6667 | 0.6400 | 0.6800 | 0.8240 | 0.1440 |
| 58 | S3 | 0.7467 | 0.6400 | 0.7120 | 0.8320 | 0.1200 |
| 58 | S4 | 0.7467 | 0.6400 | 0.7120 | 0.7840 | 0.0720 |

## Table 2 - S1 to S2 test transitions

| Seed | Wrong-to-correct | Correct-to-wrong | Net | Oracle lost | Coverage retained | Wrong coalition |
|---|---|---|---|---|---|---|
| 56 | 8 | 1 | 7 | 0 | 1 | 1 |
| 57 | 8 | 15 | -7 | 2 | 13 | 13 |
| 58 | 5 | 10 | -5 | 0 | 10 | 10 |

## Table 3 - Complementarity conversion

| Seed | Oracle gain | Oracle loss | Oracle gain to vote | Unconverted oracle gain | New pivotal to vote |
|---|---|---|---|---|---|
| 56 | 4 | 8 | 0 | 4 | 8 |
| 57 | 1 | 9 | 0 | 1 | 3 |
| 58 | 2 | 11 | 0 | 2 | 0 |

## Table 4 - Target/update concentration

| Seed | Arm | Target entropy | Target Gini | Commit entropy | Commit Gini | Accepted |
|---|---|---|---|---|---|---|
| 56 | S1 | 0.9954 | 0.0500 | 0.3955 | 0.6667 | 3 |
| 56 | S2 | 0.9159 | 0.2750 | 0.6460 | 0.5000 | 4 |
| 57 | S1 | 0.9954 | 0.0500 | 1.0000 | 0.0000 | 5 |
| 57 | S2 | 0.9822 | 0.1250 | 0.4307 | 0.6000 | 2 |
| 58 | S1 | 0.9954 | 0.0500 | 1.0000 | 0.0000 | 5 |
| 58 | S2 | 0.9724 | 0.1500 | -0.0000 | 0.8000 | 2 |

## Table 5 - Member transfer

| Seed | Agent | Targets | Commits | Train delta | Val delta | Test delta | Unique delta | Pivotal delta |
|---|---|---|---|---|---|---|---|---|
| 56 | 0 | 1 | 0 | -0.0800 | 0.0000 | -0.0640 | -7.0000 | 4.0000 |
| 56 | 1 | 2 | 0 | -0.1200 | 0.0800 | -0.0800 | -7.0000 | 4.0000 |
| 56 | 2 | 5 | 2 | 0.0800 | -0.1000 | 0.0320 | 0.0000 | 7.0000 |
| 56 | 3 | 5 | 1 | 0.0667 | 0.1400 | 0.0880 | 6.0000 | 5.0000 |
| 56 | 4 | 3 | 1 | 0.0667 | -0.0400 | -0.0400 | 2.0000 | 4.0000 |
| 57 | 0 | 3 | 0 | -0.1067 | 0.1000 | -0.0320 | 0.0000 | -3.0000 |
| 57 | 1 | 4 | 0 | -0.0933 | -0.0800 | -0.0160 | -1.0000 | -5.0000 |
| 57 | 2 | 2 | 1 | -0.0133 | 0.1400 | 0.0640 | 3.0000 | -7.0000 |
| 57 | 3 | 4 | 1 | -0.0400 | 0.1600 | -0.0720 | 3.0000 | -9.0000 |
| 57 | 4 | 3 | 0 | -0.0133 | 0.0200 | -0.0560 | -2.0000 | -3.0000 |
| 58 | 0 | 5 | 2 | -0.0133 | 0.1200 | -0.0960 | 14.0000 | -8.0000 |
| 58 | 1 | 3 | 0 | -0.0533 | 0.0800 | -0.0320 | -1.0000 | -4.0000 |
| 58 | 2 | 2 | 0 | -0.0933 | 0.0800 | -0.0240 | -2.0000 | -6.0000 |
| 58 | 3 | 3 | 0 | -0.0400 | 0.1200 | -0.0160 | -2.0000 | -4.0000 |
| 58 | 4 | 3 | 0 | -0.0667 | -0.0800 | 0.0240 | -6.0000 | -2.0000 |

## Table 6 - Module2 recovery

| Seed | S2-to-S3 net | S3-to-S4 net | S2-to-S4 net | R-M20 profile | M2F profile |
|---|---|---|---|---|---|
| 56 | -6 | 1 | -5 | NO_VOTE_GAIN | MIXED |
| 57 | 3 | 1 | 4 | MIXED | MIXED |
| 58 | 4 | 0 | 4 | MIXED | NO_VOTE_GAIN |

## Table 7 - Frozen hypothesis verdicts

| Hypothesis | Seed56 | Seed57 | Seed58 | Supporting | Status | Detail |
|---|---|---|---|---|---|---|
| H1 | SUPPORT | NO_SUPPORT | NO_SUPPORT | 1 | NOT_SUPPORTED |  |
| H2 | NO_SUPPORT | NO_SUPPORT | NO_SUPPORT | 0 | NOT_SUPPORTED |  |
| H3 | NO_SUPPORT | SUPPORT | SUPPORT | 2 | SUPPORTED |  |
| H4A | NO_SUPPORT | SUPPORT | SUPPORT | 2 | SUPPORTED |  |
| H4B | NO_SUPPORT | NO_SUPPORT | NO_SUPPORT | 0 | NOT_SUPPORTED |  |
| H5 | NO_SUPPORT | NO_SUPPORT | NO_SUPPORT | 0 | NOT_SUPPORTED |  |

## Evidence limitations

The frozen row evidence supports correctness, plurality structure, member coverage, unique/pivotal roles, target schedules, and commit logs. No independently frozen D/R/C/B/U role-profile definition was available, so that optional sub-analysis is marked evidence-insufficient rather than redefined post hoc. No low-API escalation is required.

## Next scientific question

Determine whether the broad-competence loss originates during candidate generation or during common-safe train-only selection, using a separately preregistered frozen-parent analysis; do not change the frozen V17 method in this audit.
