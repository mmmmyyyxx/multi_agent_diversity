from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(report: Path) -> None:
    summary = read_json(report / "summary.json")
    classifier = read_json(report / "classifier.json")
    commits = read_csv(report / "accepted_commit_train_evidence.csv")
    candidates = read_csv(report / "feasible_candidate_pool.csv")
    signals = read_csv(report / "risk_signal_diagnostics.csv")
    units = summary["units"]
    if units != {
        "accepted_commit_count": 18,
        "hybrid_accepted_commit_count": 11,
        "hybrid_validation_collateral_loss_event_count": 7,
        "hybrid_validation_collateral_loss_commit_count": 2,
        "hybrid_validation_gain_event_count": 5,
        "hybrid_validation_gain_bearing_commit_count": 3,
        "hybrid_positive_net_commit_count": 1,
        "gain_and_loss_bearing_commit_overlap_count": 2,
    }:
        raise ValueError("published unit accounting drift")
    hybrid_signals = [row for row in signals if row["arm"] == "HYBRID_BASE"]
    signal_lines = [
        f"| {row['signal']} | {row['flagged_commit_count']} | "
        f"{row['flagged_negative_net_count']} | {row['flagged_validation_loss_bearing_count']} | "
        f"{row['negative_net_precision']} | {row['negative_net_sensitivity']} | {row['false_positive_count']} |"
        for row in hybrid_signals
    ]
    harmful = summary["harmful_hybrid_commits"]
    harmful_lines = [
        f"| {row['seed']} | {row['update_index']} | {row['candidate_stage']} | "
        f"{row['train_target_gain']:+d} | {row['train_vote_gain_count']} | "
        f"{row['train_vote_loss_count']} | {row['train_vote_net_gain']:+d} | "
        f"{row['validation_gain_count']} | {row['validation_loss_count']} | "
        f"{row['validation_net_delta']:+d} | {row['feasible_candidate_count']} | "
        f"{row['zero_train_vote_loss_feasible_count']} |"
        for row in harmful
    ]
    readme = f"""# V18 Common-Safe / Write-Back Quality Diagnostic

## Result

```text
{classifier['final_diagnosis']}
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
{chr(10).join(harmful_lines)}

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
{chr(10).join(signal_lines)}

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
"""
    (report / "README.md").write_text(readme, encoding="utf-8")
    facts = {
        "fact_assertions_pass": True,
        "accepted_commit_count": 18,
        "hybrid_accepted_commit_count": 11,
        "hybrid_collateral_loss_event_count": 7,
        "hybrid_collateral_loss_commit_count": 2,
        "hybrid_gain_event_count": 5,
        "hybrid_gain_bearing_commit_count": 3,
        "hybrid_positive_net_commit_count": 1,
        "harmful_pool_feasible_candidate_count": sum(row["feasible_candidate_count"] for row in harmful),
        "harmful_pool_zero_loss_feasible_candidate_count": sum(row["zero_train_vote_loss_feasible_count"] for row in harmful),
        "common_safe_risk_admission_supported": classifier["common_safe_risk_admission_supported"],
        "ranking_misselection_supported": classifier["ranking_misselection_supported"],
        "feasible_set_quality_gap_supported": classifier["feasible_set_quality_gap_supported"],
        "m2f_compatibility_signal_available_in_v18": classifier["m2f_compatibility_signal_available_in_v18"],
        "uncommitted_candidate_validation_evaluations": 0,
        "new_api_calls": 0,
        "new_model_calls": 0,
        "new_test_calls": 0,
        "method_modified": False,
        "selector_modified": False,
    }
    if len(commits) != facts["accepted_commit_count"]:
        raise ValueError("accepted commit fact mismatch")
    if sum(row["committed"].lower() == "true" for row in candidates) != len(commits):
        raise ValueError("candidate commit inventory mismatch")
    if facts["harmful_pool_feasible_candidate_count"] != 7:
        raise ValueError("harmful feasible-pool count mismatch")
    if facts["harmful_pool_zero_loss_feasible_candidate_count"] != 0:
        raise ValueError("unexpected zero-loss harmful-pool alternative")
    write_json(report / "fact_assertions.json", facts)
    provenance = {
        "analysis_mode": "offline_existing_v18_writeback_quality_diagnostic",
        "source_trajectory_decomposition": "v18_trajectory_gain_loss_decomposition_v1",
        "validation_labels_reused": True,
        "new_validation_evaluations": 0,
        "uncommitted_candidate_validation_evaluations": 0,
        "new_api_calls": 0,
        "new_model_calls": 0,
        "new_test_calls": 0,
        "raw_artifacts_modified": False,
        "method_modified": False,
    }
    write_json(report / "provenance.json", provenance)
    manifest = {
        path.name: sha256_file(path)
        for path in sorted(report.iterdir())
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    write_json(report / "sha256_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    package(args.report.resolve())
    print(json.dumps({"packaged": True, "new_api_calls": 0, "new_test_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
