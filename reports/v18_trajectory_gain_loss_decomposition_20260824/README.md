# V18 Trajectory-Level Gain/Loss Decomposition

## Conclusion

```text
COLLATERAL_REGRESSION+TRANSFER_FAILURE+HIGHER_THROUGHPUT_LOWER_AVERAGE_QUALITY
```

Hybrid's extra throughput did not fail because early beneficial validation
conversions were later overwritten: all five Hybrid validation gains remained
correct to the final state. The net gap instead came from collateral losses and
weak train-to-validation transfer concentrated in a small number of accepted
commits. Hybrid made 11 commits versus 7 for W1, but its mean validation net
delta per commit was -0.182 versus 0.000.

This is a validation-only, zero-API decomposition of already completed V18
trajectories. It does not access test, change the selector or method, or treat
the aggregate four-commit difference as four matched causal updates. The two
arms diverged after early commits; comparisons below are arm-level and
seed-level structural summaries only.

## Commit quality

| Arm | Commits | Positive | Zero | Negative | Gains | Losses | Gain/commit | Loss/commit | Net/commit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1_TOP2 | 7 | 0 | 7 | 0 | 1 | 1 | 0.143 | 0.143 | 0.000 |
| HYBRID_BASE | 11 | 1 | 8 | 2 | 5 | 7 | 0.455 | 0.636 | -0.182 |

All 7 W1 commits were validation-net neutral. Hybrid had 1 positive, 8 neutral,
and 2 negative commits. Two Hybrid commits had positive train vote progress but
non-positive validation net, compared with one W1 commit. Simultaneous
validation gains and losses occurred in 2 Hybrid commits and 1 W1 commit.

## Per-trajectory telescope

| Seed | Arm | Commits | Gains | Losses | Initial-to-final Vote | Positive | Zero | Negative | Train vote not transferred |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 59 | W1_TOP2 | 2 | 0 | 0 | +0 | 0 | 2 | 0 | 0 |
| 59 | HYBRID_BASE | 5 | 4 | 3 | +1 | 1 | 3 | 1 | 1 |
| 60 | W1_TOP2 | 2 | 0 | 0 | +0 | 0 | 2 | 0 | 0 |
| 60 | HYBRID_BASE | 2 | 0 | 0 | +0 | 0 | 2 | 0 | 0 |
| 61 | W1_TOP2 | 3 | 1 | 1 | +0 | 0 | 3 | 0 | 1 |
| 61 | HYBRID_BASE | 4 | 1 | 4 | -3 | 0 | 3 | 1 | 1 |

For all 6 trajectories:

```text
sum(accepted-transition validation net deltas)
    == final validation Vote count - initial validation Vote count
```

The identity passed 6/6. Aggregate initial-to-final Vote-count change was 0 for
W1 and -2 for Hybrid.

## Gain persistence and loss provenance

- W1: 1 gain, retained to the final state; 1 loss, an initial-competence
  collateral regression.
- Hybrid: 5 gains, all retained to the final state; 7 losses, all
  initial-competence collateral regressions.
- Hybrid overwritten-gain count: 0.
- Hybrid prior-conversion-overwritten loss count: 0.
- Two of the three Seed59 Hybrid collateral losses were later recovered; the
  remaining Seed59 loss and all four Seed61 losses remained wrong at the end.

Thus the data do not support `beneficial_conversion_later_overwritten` as the
trajectory bottleneck in this pilot. The harmful transitions introduced new
validation regressions while their local gains themselves persisted.

## Seed61 focus under the same classifier

| Arm | Update | Train Vote delta | Validation gains | Validation losses | Validation net | Transfer class |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| W1_TOP2 | 1 | +0 | 0 | 0 | +0 | neutral |
| W1_TOP2 | 4 | +0 | 0 | 0 | +0 | neutral |
| W1_TOP2 | 7 | +6 | 1 | 1 | +0 | neutral |
| HYBRID_BASE | 1 | +0 | 0 | 0 | +0 | neutral |
| HYBRID_BASE | 2 | +0 | 0 | 0 | +0 | neutral |
| HYBRID_BASE | 4 | +0 | 0 | 0 | +0 | neutral |
| HYBRID_BASE | 5 | +9 | 1 | 4 | -3 | negative |

Seed61 W1 ended at net 0: its only validation-changing commit produced one gain
and one loss. Seed61 Hybrid ended at -3: update 5 produced train Vote +9 but
validation gains 1, losses 4, net -3. The one validation gain persisted; the
four losses were new collateral regressions and remained wrong. This uses the
same frozen classifier as seeds 59 and 60.

## Bottleneck classification

- `collateral_regression = true`: Hybrid incurred 7 validation losses (0.636
  per commit) versus W1's 1 (0.143 per commit), including two negative-net
  commits.
- `transfer_failure = true`: two Hybrid train-vote-positive commits had
  non-positive validation net. Seed59 update 3 transferred +2 train Vote into
  validation net -1; Seed61 update 5 transferred +9 into validation net -3.
- `beneficial_conversion_later_overwritten = false`: none of the five Hybrid
  validation gains became wrong later.
- `higher_throughput_lower_average_quality = true`: Hybrid committed more often
  (11 vs 7) but had lower average validation net quality (-0.182 vs 0.000).

The most precise interpretation is therefore: Hybrid's recovered throughput
created real and persistent local validation gains, but a few accepted commits
combined those gains with larger collateral losses and poor train-to-validation
transfer. The final net-efficacy gap is not explained by later overwriting of
the beneficial conversions.

## Files

- `accepted_commit_quality.csv`: one row per accepted transition;
- `validation_gain_persistence.csv`: every validation Vote gain and its later
  persistence class;
- `validation_loss_provenance.csv`: every validation Vote loss, origin, and
  later recovery;
- `trajectory_decomposition.csv`: per-seed/arm telescoping and quality totals;
- `source_artifact_hashes.csv`: hashes of the three whitelisted source artifact
  roles for each trajectory;
- `summary.json`, `classifier.json`, `fact_assertions.json`: machine-readable
  conclusions and invariants.
