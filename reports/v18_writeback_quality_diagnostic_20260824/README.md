# V18 Common-Safe / Write-Back Quality Diagnostic

## Result

```text
COMMON_SAFE_FEASIBLE_SET_QUALITY_GAP_WITH_EXISTING_TRAIN_VOTE_LOSS_RISK_SIGNAL
```

The present bottleneck is accepted-update quality rather than opportunity
discovery. The diagnostic supports a Common-Safe feasible-set quality gap and
identifies existing train-side vote loss as a local risk signal. It does not
support the narrower claim that the current ranking selected an obviously
more risky candidate while a lower-loss feasible alternative was available.

This is a zero-API, validation-labeled but train-evidence-only analysis. It did
not access test, evaluate an uncommitted candidate on validation, change the
selector, gate, or ranking, or replay a candidate or trajectory.

## Unit correction

The earlier event totals are not independent commit counts:

- 7 Hybrid validation collateral-loss events occurred in 2 accepted commits;
- 5 Hybrid validation gain events occurred in 3 accepted commits;
- 2 commits contained both validation gains and losses;
- only 1 Hybrid commit had positive validation net.

Therefore this is not a comparison of seven collateral commits against five
beneficial commits.

## Why Common-Safe accepted the harmful updates

| Seed | Update | Stage | Train target | Train Vote gains | Train Vote losses | Train Vote net | Val gains | Val losses | Val net | Feasible | Zero-loss feasible |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 59 | 3 | loss_blind_generic_revision | +6 | 4 | 2 | +2 | 2 | 3 | -1 | 4 | 0 |
| 61 | 5 | m20_source | +7 | 10 | 1 | +9 | 1 | 4 | -3 | 3 | 0 |

Both candidates passed target non-regression, team-Vote non-regression,
target-or-vote strict progress, and terminal-invalid non-regression. Both had
strict target and strict train-Vote progress, so the issue is not vote-only or
target-only admission. Common-Safe constrains aggregate train Vote count but
permits per-example vote losses when larger train gains make the net
non-negative.

## Gate versus ranking

At Seed59 update 3, all 4 feasible candidates had 2-3 train Vote losses. At
Seed61 update 5, all 3 feasible candidates had 1-3 train Vote losses. The
committed candidate was tied for minimum train Vote loss in both updates, no
zero-loss feasible candidate existed, and only one branch winner reached
write-back competition.

The persisted `common_monotone_safe` branch rankings were reconstructed exactly.
Train Vote loss already appears as a late ranking component, but the earlier
Vote/target/soft-utility terms and the absence of a zero-loss candidate leave
the current ranking no clearly safe alternative. Consequently:

```text
COMMON_SAFE_RISK_ADMISSION_SUPPORTED = true
FEASIBLE_SET_QUALITY_GAP_SUPPORTED = true
RANKING_MISSELECTION_SUPPORTED = false
```

This does not establish that a zero-loss hard guard would be correct. It shows
only that the harmful accepted pools were uniformly loss-bearing and that the
current gate allowed them because their train net was positive.

## Existing train-side risk signals

| Signal | Flagged | Negative flagged | Loss-bearing flagged | Precision | Sensitivity | False positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_vote_loss_positive | 2 | 2 | 2 | 1.0 | 1.0 | 0 |
| train_pivotal_loss_positive | 2 | 2 | 2 | 1.0 | 1.0 | 0 |
| train_unique_loss_positive | 2 | 0 | 0 | 0.0 | 0.0 | 2 |
| train_coverage_loss_positive | 2 | 0 | 0 | 0.0 | 0.0 | 2 |
| train_vote_gain_and_loss_cooccur | 2 | 2 | 2 | 1.0 | 1.0 | 0 |
| train_soft_utility_negative | 1 | 0 | 0 | 0.0 | 0.0 | 1 |
| train_target_only_progress | 8 | 0 | 0 | 0.0 | 0.0 | 8 |
| generic_revision_stage | 8 | 1 | 1 | 0.125 | 0.5 | 7 |
| assigned_residual_repair_positive | 8 | 2 | 2 | 0.25 | 1.0 | 6 |

Within the 11 Hybrid commits, `train_vote_loss_positive` and
`train_pivotal_loss_positive` each flagged exactly the 2 negative-net commits
and no others. `train_vote_gain_and_loss_cooccur` was equivalent in this small
sample. By contrast, unique-correct loss and coverage loss each flagged two
commits but neither was validation-negative; the sole positive-net commit had
both signals. Target-only progress flagged eight commits and none was
validation-negative.

The vote-loss signal is therefore observable before write-back and locally
discriminative here, but its apparent precision/sensitivity is based on only
two harmful commits across two seeds. It is a prospective quality-control
candidate, not a validated new acceptance rule.

## M2F / compatibility evidence

```text
registry compatibility repair enabled = false
compatibility events = 0
Module2 context diagnostic events = 0
non-null candidate responsibility-contribution records = 0
M2F_COMPATIBILITY_SIGNAL_AVAILABLE_IN_V18 = false
```

V18 did not compute or persist an M2F/compatibility score in the write-back
path, and the winner key did not use one. The completed artifacts therefore
cannot show that an available compatibility signal was ignored. A future
prospective diagnostic may test such a signal, but it cannot be retroactively
reconstructed as if it had governed these candidates.

## Research interpretation

Hybrid improved opportunity realization and longitudinal accumulation, but it
also admitted more low-transfer updates. The two observed harmful commits had
strong train gains while producing validation collateral. Current evidence
shifts the research focus from target allocation to transfer-safe write-back
quality control, while leaving the method unchanged.

## Published files

- `accepted_commit_train_evidence.csv`: all 18 commits and frozen train-side
  signals;
- `feasible_candidate_pool.csv`: sanitized train evidence for every feasible
  candidate, with no counterfactual validation result;
- `risk_signal_diagnostics.csv`: fixed signal contingency counts;
- `summary.json`, `classifier.json`, and `fact_assertions.json`: units,
  diagnosis, and machine-readable checks;
- `source_artifact_hashes.csv` and `sha256_manifest.json`: provenance hashes.
