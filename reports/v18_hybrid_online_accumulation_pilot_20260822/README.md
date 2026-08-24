# V18 Hybrid Online Accumulation Pilot -- Scientific Analysis

## Gate provenance

```text
Original frozen execution audit: FAIL / HOLD
Independent zero-API revision-parity semantics audit: PASS
post_hoc_corrected_gate_v1: PASS
Scientific analysis admitted: true
```

The original auditor incorrectly required every revision attempt to produce an
evaluable row. Four invalid revision outputs legally consumed their frozen
revision opportunities but produced no evaluable rows. The independent audit
established compute-matched attempt budgets. The original frozen HOLD remains
unchanged; it was not rewritten as a pass.

The scientific analysis uses the original unchanged V18 trajectories. The
experiment itself was not rerun or repaired. No revision, candidate, update,
trajectory, validation model output, or test evaluation was rerun.

## Frozen result

```text
ONLINE_ACCUMULATION_SUPPORTED = true
ONLINE_VOTE_CONVERSION_SIGNAL = true
HYBRID_THROUGHPUT_RECOVERY_REPRODUCED = true
PERSISTENT_SINGLETON_REDUCED = true
FINAL_VALIDATION_VOTE_SIGNAL = neutral
FINAL_DIAGNOSIS = LONGITUDINAL_ACCUMULATION_WITH_VOTE_CONVERSION
```

Hybrid-recovered singleton coverage did deepen across subsequent online
responsibility updates, and some of that deeper support converted into correct
plurality decisions. This is a mechanism signal from a small three-seed,
validation-only pilot under a post-hoc corrected gate--not a formal test or
generalization claim.

## Per-seed evidence

| Seed | Arm | Commits | Feasible branches | 0->1 | 0->1->2+ | Persistent singleton | Cross-member deepening | Coverage->Vote | Final Val Vote | Final Val Oracle |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 59 | W1_TOP2 | 2 | 3 | 8 | 2 | 6 | 2 | 0 | 0.660 | 0.820 |
| 59 | HYBRID_BASE | 5 | 5 | 12 | 4 | 8 | 4 | 2 | 0.680 | 0.880 |
| 60 | W1_TOP2 | 2 | 2 | 8 | 5 | 3 | 5 | 0 | 0.660 | 0.820 |
| 60 | HYBRID_BASE | 2 | 2 | 7 | 4 | 3 | 4 | 0 | 0.660 | 0.800 |
| 61 | W1_TOP2 | 3 | 3 | 7 | 2 | 5 | 2 | 1 | 0.660 | 0.800 |
| 61 | HYBRID_BASE | 4 | 4 | 7 | 3 | 3 | 3 | 1 | 0.600 | 0.780 |

## Paired Hybrid - W1 comparisons

| Seed | Deepening | Vote conversion | Commits | Final Vote | Final Oracle |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 59 | +2 | +2 | +3 | +0.020 | +0.060 |
| 60 | -1 | +0 | +0 | +0.000 | -0.020 |
| 61 | +1 | +0 | +1 | -0.060 | -0.020 |

Deepening differences were `[2, -1, 1]`
(mean `0.667`, W/T/L
`{'wins': 2, 'ties': 0, 'losses': 1}`). Vote-conversion
differences were `[2, 0, 0]`
(W/T/L `{'wins': 1, 'ties': 2, 'losses': 0}`).

## Mechanism hierarchy

1. **Throughput.** Hybrid produced 11
   feasible branches and 11 commits,
   versus 8 and
   7 for W1.
2. **Coverage recovery.** Hybrid produced
   26 validation 0->1 recoveries,
   versus 23.
3. **Longitudinal accumulation.** Hybrid produced
   11 recovered-then-deepened
   cases, versus 9.
4. **Cross-member accumulation.** The corresponding counts were
   11 versus
   9.
5. **Vote conversion.** Recovered-coverage conversions were
   3 versus
   1.
6. **Final validation.** Mean VoteAcc was
   0.647 for Hybrid and
   0.660 for W1; mean
   OracleAcc was 0.820
   versus 0.813. The
   frozen final-vote classifier is `neutral`.

## Revision accounting

Revision attempt count and evaluable revision row count are reported
separately. W1 had 24
attempts and 24
evaluable rows. Hybrid had
30 attempts,
26
evaluable rows, and 4
invalid outputs. Each invalid output has `attempted=true`, `valid_output=false`,
`evaluable_row=false`, and `opportunity_consumed=true` in `revision_attempts.csv`.

## Transfer and limitations

`TARGET_TRANSFER_GAP` is reported descriptively as validation target delta
minus train target delta for each accepted update. Full update-level evidence
is in `update_lineage.csv`; arm summaries are in `summary.json`.

This report uses exact counts, paired seed differences, means, and W/T/L only.
It makes no significance claim, adds no seed, uses no test split, and does not
implement a new selector or method.
