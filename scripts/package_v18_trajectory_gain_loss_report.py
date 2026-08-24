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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(report: Path) -> None:
    summary = read_json(report / "summary.json")
    classifier = read_json(report / "classifier.json")
    trajectories = read_csv(report / "trajectory_decomposition.csv")
    commits = read_csv(report / "accepted_commit_quality.csv")
    if summary["telescoping_identity_pass_count"] != 6:
        raise ValueError("all six telescoping identities must pass")
    if summary["scope"] != {
        "validation_only": True,
        "new_api_calls": 0,
        "new_model_calls": 0,
        "new_test_calls": 0,
        "method_modified": False,
        "selector_modified": False,
        "cross_arm_commit_matching": False,
        "diverged_trajectories_treated_as_matched": False,
    }:
        raise ValueError("analysis scope drift")
    aggregate = summary["aggregate"]
    w1 = aggregate["W1_TOP2"]
    hybrid = aggregate["HYBRID_BASE"]
    rows = []
    for row in trajectories:
        rows.append(
            f"| {row['seed']} | {row['arm']} | {row['accepted_commit_count']} | "
            f"{row['validation_gain_count']} | {row['validation_loss_count']} | "
            f"{int(row['initial_to_final_vote_delta']):+d} | {row['positive_net_commits']} | "
            f"{row['zero_net_commits']} | {row['negative_net_commits']} | "
            f"{row['train_vote_progress_not_transferred_commits']} |"
        )
    seed61_commits = [row for row in commits if int(row["seed"]) == 61]
    seed61_lines = [
        f"| {row['arm']} | {row['update_index']} | {int(row['train_vote_delta']):+d} | "
        f"{row['validation_gain_count']} | {row['validation_loss_count']} | "
        f"{int(row['validation_net_delta']):+d} | {row['validation_transfer_class']} |"
        for row in seed61_commits
    ]
    readme = f"""# V18 Trajectory-Level Gain/Loss Decomposition

## Conclusion

```text
{classifier['final_diagnosis']}
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
| W1_TOP2 | {w1['commit_count']} | {w1['positive_net_count']} | {w1['zero_net_count']} | {w1['negative_net_count']} | {w1['validation_gain_count']} | {w1['validation_loss_count']} | {w1['mean_gain_count_per_commit']:.3f} | {w1['mean_loss_count_per_commit']:.3f} | {w1['mean_net_delta_per_commit']:.3f} |
| HYBRID_BASE | {hybrid['commit_count']} | {hybrid['positive_net_count']} | {hybrid['zero_net_count']} | {hybrid['negative_net_count']} | {hybrid['validation_gain_count']} | {hybrid['validation_loss_count']} | {hybrid['mean_gain_count_per_commit']:.3f} | {hybrid['mean_loss_count_per_commit']:.3f} | {hybrid['mean_net_delta_per_commit']:.3f} |

All 7 W1 commits were validation-net neutral. Hybrid had 1 positive, 8 neutral,
and 2 negative commits. Two Hybrid commits had positive train vote progress but
non-positive validation net, compared with one W1 commit. Simultaneous
validation gains and losses occurred in 2 Hybrid commits and 1 W1 commit.

## Per-trajectory telescope

| Seed | Arm | Commits | Gains | Losses | Initial-to-final Vote | Positive | Zero | Negative | Train vote not transferred |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

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
{chr(10).join(seed61_lines)}

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
"""
    (report / "README.md").write_text(readme, encoding="utf-8")
    facts = {
        "analysis_version": summary["analysis_version"],
        "fact_assertions_pass": True,
        "trajectory_count": 6,
        "accepted_transition_count": 18,
        "telescoping_identity_pass_count": 6,
        "w1_commit_count": 7,
        "hybrid_commit_count": 11,
        "aggregate_commit_difference_not_matched_pairs": True,
        "hybrid_gain_count": 5,
        "hybrid_gain_retained_to_final_count": 5,
        "hybrid_gain_overwritten_later_count": 0,
        "hybrid_loss_count": 7,
        "hybrid_new_collateral_loss_count": 7,
        "hybrid_prior_conversion_overwritten_loss_count": 0,
        "seed61_hybrid_initial_to_final_vote_delta": -3,
        "new_api_calls": 0,
        "new_model_calls": 0,
        "new_test_calls": 0,
        "method_modified": False,
        "selector_modified": False,
    }
    actual = {
        "trajectory_count": summary["trajectory_count"],
        "accepted_transition_count": summary["accepted_transition_count"],
        "telescoping_identity_pass_count": summary["telescoping_identity_pass_count"],
        "w1_commit_count": w1["commit_count"],
        "hybrid_commit_count": hybrid["commit_count"],
        "hybrid_gain_count": hybrid["validation_gain_count"],
        "hybrid_gain_retained_to_final_count": hybrid["gain_retained_to_final_count"],
        "hybrid_gain_overwritten_later_count": hybrid["gain_overwritten_later_count"],
        "hybrid_loss_count": hybrid["validation_loss_count"],
        "hybrid_new_collateral_loss_count": hybrid["loss_new_collateral_regression_count"],
        "hybrid_prior_conversion_overwritten_loss_count": hybrid["loss_prior_conversion_overwritten_count"],
        "seed61_hybrid_initial_to_final_vote_delta": summary["seed61_focus"]["HYBRID_BASE"]["initial_to_final_vote_delta"],
    }
    for key, value in actual.items():
        if facts[key] != value:
            raise ValueError(f"fact assertion failed: {key}")
    write_json(report / "fact_assertions.json", facts)
    provenance = {
        "original_frozen_audit_status": summary["source_gate"]["original_frozen_audit_status"],
        "post_hoc_corrected_gate_status": summary["source_gate"]["post_hoc_corrected_gate_status"],
        "raw_artifact_identity": summary["source_gate"]["raw_artifact_identity"],
        "analysis_mode": "offline_existing_artifact_trajectory_decomposition",
        "validation_only": True,
        "new_api_calls": 0,
        "new_model_calls": 0,
        "new_test_calls": 0,
        "experiment_rerun": False,
        "raw_artifacts_modified": False,
        "cross_arm_commit_matching": False,
    }
    write_json(report / "gate_provenance.json", provenance)
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
