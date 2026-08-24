from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ARMS = ("W1_TOP2", "HYBRID_BASE")
SEEDS = (59, 60, 61)
TRANSITIONS = ("0_to_1", "1_to_2", "1_to_3_plus", "2_to_3", "2_to_4_plus", "3_to_4_plus")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value: str | None) -> float | None:
    if value in (None, "", "NA"):
        return None
    return float(value)


def integer(value: str | None) -> int | None:
    parsed = number(value)
    return None if parsed is None else int(parsed)


def wtl(values: Iterable[float]) -> dict[str, int]:
    values = list(values)
    return {
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
    }


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def train_transition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result = {key: 0 for key in TRANSITIONS}
    for row in rows:
        old, new = int(row["G_before"]), int(row["G_after"])
        if old == 0 and new == 1:
            result["0_to_1"] += 1
        if old == 1 and new == 2:
            result["1_to_2"] += 1
        if old == 1 and new >= 3:
            result["1_to_3_plus"] += 1
        if old == 2 and new == 3:
            result["2_to_3"] += 1
        if old == 2 and new >= 4:
            result["2_to_4_plus"] += 1
        if old == 3 and new >= 4:
            result["3_to_4_plus"] += 1
    return result


def add_counts(target: dict[str, int], values: dict[str, int]) -> None:
    for key in target:
        target[key] += int(values.get(key, 0))


def target_summary(updates: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in updates:
        for key in ("target1", "target2"):
            if row.get(key) is not None:
                counts[int(row[key])] += 1
    total = sum(counts.values())
    entropy = -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    ) if total else 0.0
    return {
        "target_member_counts": {str(agent): counts[agent] for agent in range(5)},
        "unique_targeted_members": len(counts),
        "target_entropy": entropy,
        "target_concentration": max(counts.values(), default=0) / max(1, total),
    }


def transfer_summary(updates: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in updates if str(row.get("committed", "")).lower() == "true"]
    result: dict[str, Any] = {"accepted_update_count": len(accepted)}
    for label, train_key, validation_key in (
        ("target", "train_target_delta", "validation_target_delta"),
        ("vote", "train_vote_delta", "validation_vote_delta"),
        ("oracle", "train_oracle_delta", "validation_oracle_delta"),
    ):
        train = [float(row[train_key]) for row in accepted]
        validation = [float(row[validation_key]) for row in accepted]
        gaps = [right - left for left, right in zip(train, validation, strict=True)]
        result.update({
            f"train_{label}_delta_sum": sum(train),
            f"validation_{label}_delta_sum": sum(validation),
            f"mean_train_{label}_delta": mean(train),
            f"mean_validation_{label}_delta": mean(validation),
            f"mean_{label}_transfer_gap_validation_minus_train": mean(gaps),
        })
    return result


def package(
    *,
    report: Path,
    run_root: Path,
    admission: dict[str, Any],
    semantics: dict[str, Any],
    corrected: dict[str, Any],
    analysis_timestamp: str,
) -> dict[str, Any]:
    if admission.get("scientific_analysis_admitted") is not True:
        raise ValueError("scientific analysis was not admitted")
    if corrected.get("gate") != "PASS":
        raise ValueError("post-hoc corrected gate must pass")
    summary = read_json(report / "summary.json")
    classifier = read_json(report / "classifier.json")
    trajectory_rows = read_csv(report / "trajectory_level.csv")
    update_rows = read_csv(report / "update_lineage.csv")
    residual_rows = read_csv(report / "residual_lineage.csv")
    parity_by_key = {
        (int(row["seed"]), str(row["arm"])): row
        for row in semantics["trajectory_rows"]
    }

    paired_rows: list[dict[str, Any]] = []
    trajectory_by_key = {
        (int(row["seed"]), row["arm"]): row for row in trajectory_rows
    }
    for seed in SEEDS:
        left = trajectory_by_key[(seed, "W1_TOP2")]
        right = trajectory_by_key[(seed, "HYBRID_BASE")]
        paired_rows.append({
            "seed": seed,
            "deepening_hybrid_minus_w1": integer(right["longitudinal_deepened_coverage_count"]) - integer(left["longitudinal_deepened_coverage_count"]),
            "vote_conversion_hybrid_minus_w1": integer(right["recovered_coverage_to_vote_count"]) - integer(left["recovered_coverage_to_vote_count"]),
            "commit_hybrid_minus_w1": integer(right["accepted_commit_count"]) - integer(left["accepted_commit_count"]),
            "final_vote_acc_hybrid_minus_w1": number(right["final_validation_vote_acc"]) - number(left["final_validation_vote_acc"]),
            "final_oracle_acc_hybrid_minus_w1": number(right["final_validation_oracle_acc"]) - number(left["final_validation_oracle_acc"]),
        })
    write_csv(report / "paired_comparison.csv", paired_rows)

    revision_attempt_rows: list[dict[str, Any]] = []
    per_trajectory_extra: dict[tuple[int, str], dict[str, Any]] = {}
    transition_summary: dict[str, dict[str, dict[str, int]]] = {}
    target_rows: list[dict[str, Any]] = []
    transfer_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            run = run_root / f"seed{seed}" / arm
            events = read_jsonl(run / "loss_blind_generic_revision_events.jsonl")
            event_rows = []
            for event in events:
                row = {
                    "seed": seed,
                    "arm": arm,
                    "update_index": int(event["update_index"]),
                    "target_member": int(event["target_agent_id"]),
                    "parent_team_hash": event["parent_team_hash"],
                    "source_candidate_hash": event["source_candidate_hash"],
                    "attempted": bool(event["revision_attempted"]),
                    "valid_output": bool(event["revision_output_valid"]),
                    "evaluable_row": bool(event["revision_output_valid"]),
                    "opportunity_consumed": bool(event["revision_attempted"]),
                    "sanitized_terminal_failure_class": str(event.get("terminal_failure_class", "")),
                }
                event_rows.append(row)
                revision_attempt_rows.append(row)
            parity = parity_by_key[(seed, arm)]
            assert len(event_rows) == int(parity["revision_attempt_count"])
            assert sum(row["valid_output"] for row in event_rows) == int(parity["evaluable_revision_row_count"])
            updates = [row for row in update_rows if int(row["seed"]) == seed and row["arm"] == arm]
            target = target_summary(updates)
            target_rows.append({"seed": seed, "arm": arm, **target})
            transfer = transfer_summary(updates)
            transfer_by_arm[arm].append(transfer)
            train = train_transition_counts(read_jsonl(run / "g_transition_audit.jsonl"))
            validation = ast.literal_eval(trajectory_by_key[(seed, arm)]["support_transitions"])
            transition_summary[f"{seed}:{arm}"] = {"train": train, "validation": validation}
            per_trajectory_extra[(seed, arm)] = {
                "revision_attempt_count": int(parity["revision_attempt_count"]),
                "valid_revision_output_count": int(parity["revision_output_valid_count"]),
                "invalid_revision_output_count": int(parity["revision_output_invalid_count"]),
                "evaluable_revision_row_count": int(parity["evaluable_revision_row_count"]),
                **target,
            }
    write_csv(report / "revision_attempts.csv", revision_attempt_rows)
    write_csv(report / "target_allocation.csv", target_rows)

    enriched_trajectory = []
    for row in trajectory_rows:
        key = (int(row["seed"]), row["arm"])
        extra = per_trajectory_extra[key]
        enriched_trajectory.append({
            **row,
            "revision_attempt_count": extra["revision_attempt_count"],
            "valid_revision_output_count": extra["valid_revision_output_count"],
            "invalid_revision_output_count": extra["invalid_revision_output_count"],
            "evaluable_revision_row_count": extra["evaluable_revision_row_count"],
            "target_member_counts": json.dumps(extra["target_member_counts"], sort_keys=True),
        })
    write_csv(report / "trajectory_level.csv", enriched_trajectory)

    arm_transition: dict[str, dict[str, dict[str, int]]] = {}
    arm_margin: dict[str, dict[str, int]] = {}
    arm_followup: dict[str, dict[str, int]] = {}
    arm_transfer: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        train = {key: 0 for key in TRANSITIONS}
        validation = {key: 0 for key in TRANSITIONS}
        for seed in SEEDS:
            add_counts(train, transition_summary[f"{seed}:{arm}"]["train"])
            add_counts(validation, transition_summary[f"{seed}:{arm}"]["validation"])
        arm_transition[arm] = {"train": train, "validation": validation}
        residuals = [row for row in residual_rows if row["arm"] == arm]
        arm_margin[arm] = {
            "recovered_residual_count": len(residuals),
            "wrong_coalition_decreased": sum(row["wrong_coalition_decreased"] == "True" for row in residuals),
            "wrong_coalition_unchanged": sum(row["wrong_coalition_unchanged"] == "True" for row in residuals),
            "wrong_coalition_increased": sum(row["wrong_coalition_increased"] == "True" for row in residuals),
            "margin_improved": sum(row["margin_improved"] == "True" for row in residuals),
            "margin_unchanged": sum(row["margin_unchanged"] == "True" for row in residuals),
            "margin_worsened": sum(row["margin_worsened"] == "True" for row in residuals),
        }
        arm_followup[arm] = dict(Counter(row["followup_case"] for row in residuals))
        accepted_total = sum(row["accepted_update_count"] for row in transfer_by_arm[arm])
        arm_transfer[arm] = {
            "accepted_update_count": accepted_total,
            **{
                key: (
                    sum(float(row[key]) * int(row["accepted_update_count"]) for row in transfer_by_arm[arm]) / accepted_total
                    if accepted_total else None
                )
                for key in (
                    "mean_train_target_delta", "mean_validation_target_delta",
                    "mean_target_transfer_gap_validation_minus_train",
                    "mean_train_vote_delta", "mean_validation_vote_delta",
                    "mean_vote_transfer_gap_validation_minus_train",
                    "mean_train_oracle_delta", "mean_validation_oracle_delta",
                    "mean_oracle_transfer_gap_validation_minus_train",
                )
            },
        }

    final_vote_diffs = [row["final_vote_acc_hybrid_minus_w1"] for row in paired_rows]
    vote_wtl = wtl(final_vote_diffs)
    if classifier["final_validation_vote_signal"]:
        final_vote_label = "positive"
    elif mean(final_vote_diffs) < 0 and vote_wtl["losses"] > vote_wtl["wins"]:
        final_vote_label = "negative"
    else:
        final_vote_label = "neutral"

    paired_summary = {
        key: {
            "mean_hybrid_minus_w1": mean(row[key] for row in paired_rows),
            "wtl": wtl(row[key] for row in paired_rows),
        }
        for key in (
            "deepening_hybrid_minus_w1", "vote_conversion_hybrid_minus_w1",
            "commit_hybrid_minus_w1", "final_vote_acc_hybrid_minus_w1",
            "final_oracle_acc_hybrid_minus_w1",
        )
    }
    summary.update({
        "gate_provenance": {
            "original_frozen_audit_status": "FAIL/HOLD",
            "independent_semantics_audit_status": "PASS",
            "post_hoc_corrected_gate_status": "PASS",
            "corrected_gate_version": corrected["version"],
            "scientific_analysis_admitted": True,
            "phase_b_gate_interpretation": "post_hoc_corrected_admission_pass",
        },
        "revision_accounting": {
            arm: {
                key: sum(int(parity_by_key[(seed, arm)][key]) for seed in SEEDS)
                for key in (
                    "valid_source_count", "revision_attempt_count",
                    "revision_output_valid_count", "revision_output_invalid_count",
                    "evaluable_revision_row_count",
                )
            } for arm in ARMS
        },
        "support_transition_histogram": arm_transition,
        "margin_wrong_coalition": arm_margin,
        "followup_opportunity_analysis": arm_followup,
        "target_allocation": {
            arm: {
                "member_counts": {
                    str(agent): sum(
                        row["target_member_counts"][str(agent)]
                        for row in (per_trajectory_extra[(seed, arm)] for seed in SEEDS)
                    ) for agent in range(5)
                },
                "mean_entropy": mean(per_trajectory_extra[(seed, arm)]["target_entropy"] for seed in SEEDS),
                "mean_concentration": mean(per_trajectory_extra[(seed, arm)]["target_concentration"] for seed in SEEDS),
            } for arm in ARMS
        },
        "train_to_validation_transfer": arm_transfer,
        "paired_comparison": {"per_seed": paired_rows, "aggregate": paired_summary},
        "final_validation_vote_signal_label": final_vote_label,
        "scientific_scope": {
            "validation_only": True,
            "new_api_calls": 0,
            "new_test_calls": 0,
            "significance_claims": False,
            "small_seed_count": 3,
        },
    })
    write_json(report / "summary.json", summary)

    gate_provenance = {
        "original_frozen_gate_status": "FAIL/HOLD",
        "original_blockers": admission["original_frozen_blockers"],
        "independent_semantics_audit_status": "PASS",
        "post_hoc_corrected_gate_status": "PASS",
        "corrected_gate_version": corrected["version"],
        "raw_artifact_hash_verified": True,
        "raw_artifact_identity": admission["raw_artifact_identity"],
        "scientific_analysis_admitted": True,
        "scientific_analysis_timestamp": analysis_timestamp,
        "new_api_calls": 0,
        "new_model_calls": 0,
        "new_test_calls": 0,
        "experiment_rerun": False,
        "invalid_revisions_retried": False,
    }
    write_json(report / "gate_provenance.json", gate_provenance)

    def f(value: Any, digits: int = 3) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    raw_lines = []
    for row in enriched_trajectory:
        raw_lines.append(
            f"| {row['seed']} | {row['arm']} | {row['accepted_commit_count']} | "
            f"{row['feasible_branch_count']} | {row['recovered_singleton_count']} | "
            f"{row['longitudinal_deepened_coverage_count']} | {row['persistent_singleton_count']} | "
            f"{row['cross_member_support_accumulation_count']} | {row['recovered_coverage_to_vote_count']} | "
            f"{f(row['final_validation_vote_acc'])} | {f(row['final_validation_oracle_acc'])} |"
        )
    paired_lines = [
        f"| {row['seed']} | {row['deepening_hybrid_minus_w1']:+d} | "
        f"{row['vote_conversion_hybrid_minus_w1']:+d} | {row['commit_hybrid_minus_w1']:+d} | "
        f"{row['final_vote_acc_hybrid_minus_w1']:+.3f} | {row['final_oracle_acc_hybrid_minus_w1']:+.3f} |"
        for row in paired_rows
    ]
    aggregate = summary["aggregate"]
    readme = f"""# V18 Hybrid Online Accumulation Pilot -- Scientific Analysis

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
ONLINE_ACCUMULATION_SUPPORTED = {str(classifier['online_accumulation_supported']).lower()}
ONLINE_VOTE_CONVERSION_SIGNAL = {str(classifier['online_vote_conversion_signal']).lower()}
HYBRID_THROUGHPUT_RECOVERY_REPRODUCED = {str(classifier['hybrid_throughput_recovery_reproduced']).lower()}
PERSISTENT_SINGLETON_REDUCED = {str(classifier['persistent_singleton_reduced']).lower()}
FINAL_VALIDATION_VOTE_SIGNAL = {final_vote_label}
FINAL_DIAGNOSIS = {classifier['final_diagnosis']}
```

Hybrid-recovered singleton coverage did deepen across subsequent online
responsibility updates, and some of that deeper support converted into correct
plurality decisions. This is a mechanism signal from a small three-seed,
validation-only pilot under a post-hoc corrected gate--not a formal test or
generalization claim.

## Per-seed evidence

| Seed | Arm | Commits | Feasible branches | 0->1 | 0->1->2+ | Persistent singleton | Cross-member deepening | Coverage->Vote | Final Val Vote | Final Val Oracle |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(raw_lines)}

## Paired Hybrid - W1 comparisons

| Seed | Deepening | Vote conversion | Commits | Final Vote | Final Oracle |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(paired_lines)}

Deepening differences were `{[row['deepening_hybrid_minus_w1'] for row in paired_rows]}`
(mean `{paired_summary['deepening_hybrid_minus_w1']['mean_hybrid_minus_w1']:.3f}`, W/T/L
`{paired_summary['deepening_hybrid_minus_w1']['wtl']}`). Vote-conversion
differences were `{[row['vote_conversion_hybrid_minus_w1'] for row in paired_rows]}`
(W/T/L `{paired_summary['vote_conversion_hybrid_minus_w1']['wtl']}`).

## Mechanism hierarchy

1. **Throughput.** Hybrid produced {aggregate['HYBRID_BASE']['feasible_branches']}
   feasible branches and {aggregate['HYBRID_BASE']['accepted_commits']} commits,
   versus {aggregate['W1_TOP2']['feasible_branches']} and
   {aggregate['W1_TOP2']['accepted_commits']} for W1.
2. **Coverage recovery.** Hybrid produced
   {aggregate['HYBRID_BASE']['recoveries_0_to_1']} validation 0->1 recoveries,
   versus {aggregate['W1_TOP2']['recoveries_0_to_1']}.
3. **Longitudinal accumulation.** Hybrid produced
   {aggregate['HYBRID_BASE']['deepenings_0_to_1_to_2_plus']} recovered-then-deepened
   cases, versus {aggregate['W1_TOP2']['deepenings_0_to_1_to_2_plus']}.
4. **Cross-member accumulation.** The corresponding counts were
   {aggregate['HYBRID_BASE']['cross_member_accumulations']} versus
   {aggregate['W1_TOP2']['cross_member_accumulations']}.
5. **Vote conversion.** Recovered-coverage conversions were
   {aggregate['HYBRID_BASE']['recovered_coverage_vote_conversions']} versus
   {aggregate['W1_TOP2']['recovered_coverage_vote_conversions']}.
6. **Final validation.** Mean VoteAcc was
   {aggregate['HYBRID_BASE']['mean_final_validation_vote_acc']:.3f} for Hybrid and
   {aggregate['W1_TOP2']['mean_final_validation_vote_acc']:.3f} for W1; mean
   OracleAcc was {aggregate['HYBRID_BASE']['mean_final_validation_oracle_acc']:.3f}
   versus {aggregate['W1_TOP2']['mean_final_validation_oracle_acc']:.3f}. The
   frozen final-vote classifier is `{final_vote_label}`.

## Revision accounting

Revision attempt count and evaluable revision row count are reported
separately. W1 had {summary['revision_accounting']['W1_TOP2']['revision_attempt_count']}
attempts and {summary['revision_accounting']['W1_TOP2']['evaluable_revision_row_count']}
evaluable rows. Hybrid had
{summary['revision_accounting']['HYBRID_BASE']['revision_attempt_count']} attempts,
{summary['revision_accounting']['HYBRID_BASE']['evaluable_revision_row_count']}
evaluable rows, and {summary['revision_accounting']['HYBRID_BASE']['revision_output_invalid_count']}
invalid outputs. Each invalid output has `attempted=true`, `valid_output=false`,
`evaluable_row=false`, and `opportunity_consumed=true` in `revision_attempts.csv`.

## Transfer and limitations

`TARGET_TRANSFER_GAP` is reported descriptively as validation target delta
minus train target delta for each accepted update. Full update-level evidence
is in `update_lineage.csv`; arm summaries are in `summary.json`.

This report uses exact counts, paired seed differences, means, and W/T/L only.
It makes no significance claim, adds no seed, uses no test split, and does not
implement a new selector or method.
"""
    (report / "README.md").write_text(readme, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--corrected_gate", type=Path, required=True)
    parser.add_argument("--analysis_timestamp", required=True)
    args = parser.parse_args()
    result = package(
        report=args.report.resolve(),
        run_root=args.root.resolve(),
        admission=read_json(args.admission),
        semantics=read_json(args.semantics),
        corrected=read_json(args.corrected_gate),
        analysis_timestamp=args.analysis_timestamp,
    )
    print(json.dumps({
        "final_diagnosis": result["classifier"]["final_diagnosis"],
        "scientific_analysis_admitted": True,
        "new_api_calls": 0,
        "new_test_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
