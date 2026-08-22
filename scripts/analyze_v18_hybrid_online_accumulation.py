from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v18_hybrid_online_accumulation_support import (
    ARMS,
    SEEDS,
    build_residual_lineage,
    classify,
    summarize_trajectory,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NA" if value is None else value for key, value in row.items()})


def _followup(
    lineage: dict[str, Any], states: list[dict[str, Any]], updates: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    recovery = int(lineage["first_0_to_1_state"])
    state_rows = {
        int(state["state_index"]): {
            row["example_id_hash"]: row for row in state["examples"]
        }
        for state in states
    }
    opportunity = targeted = feasible = False
    for update in updates:
        if int(update["validation_state_index_before"]) < recovery:
            continue
        example = state_rows[int(update["validation_state_index_before"])][lineage["example_id_hash"]]
        eligible = set(map(int, example["responsibility_eligible_member_ids"]))
        if not eligible:
            continue
        opportunity = True
        selected = {target for target in (update["target1"], update["target2"]) if target is not None}
        intersection = eligible & set(map(int, selected))
        if intersection:
            targeted = True
            feasible = feasible or any(
                row["update_index"] == update["update_index"]
                and row["target_member"] in intersection
                and row["feasible"]
                for row in candidates
            )
    successful = lineage["later_1_to_2_state"] is not None
    if successful:
        case = "D_SUCCESSFUL"
    elif targeted and feasible:
        case = "C_FEASIBLE_NO_DEEPENING"
    elif targeted:
        case = "B_TARGETED_INFEASIBLE"
    elif opportunity:
        case = "A_OPPORTUNITY_NOT_TARGETED"
    else:
        case = "NO_FOLLOWUP_OPPORTUNITY"
    return {
        "followup_opportunity_present": opportunity,
        "followup_targeted": targeted,
        "followup_successful": successful,
        "followup_case": case,
    }


def analyze(root: Path, audit: dict[str, Any], source_freeze: dict[str, Any], out: Path) -> dict[str, Any]:
    if audit["gate"] != "PASS":
        raise ValueError("canonical Phase B audit must pass before analysis")
    if out.exists():
        raise FileExistsError("fresh report directory required")
    out.mkdir(parents=True)
    trajectories = []
    update_rows = []
    validation_rows = []
    residual_rows = []
    candidate_rows = []
    total_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "successful_llm_calls": 0}
    for seed in SEEDS:
        for arm in ARMS:
            run = root / f"seed{seed}" / arm
            states = read_jsonl(run / "validation_states.jsonl")
            updates = read_jsonl(run / "update_lineage.jsonl")
            candidates = read_jsonl(run / "candidate_level_sanitized.jsonl")
            summary = read_json(run / "online_run_summary.json")
            trajectory = summarize_trajectory(
                states=states,
                accepted_commit_count=summary["accepted_commit_count"],
                feasible_branch_count=sum(row["feasible_branch_count"] for row in updates),
                feasible_candidate_count=sum(row["feasible_candidate_count"] for row in updates),
                update_opportunities=len(updates),
            )
            trajectory.update({
                "seed": seed,
                "arm": arm,
                "planned_update_opportunities": summary["planned_update_count"],
                "completed_update_opportunities": summary["completed_update_count"],
                "early_stop_reason": summary.get("early_stop_reason", ""),
                "valid_source_count": sum(row["candidate_stage"] == "source" and row["valid"] for row in candidates),
                "valid_revision_count": sum(row["candidate_stage"] == "revision" and row["valid"] for row in candidates),
                "unique_target_members": len(summary["target_members"]),
            })
            target_counts = {
                agent: sum(agent in (row["target1"], row["target2"]) for row in updates)
                for agent in range(5)
            }
            total_targets = sum(target_counts.values())
            import math
            trajectory["target_entropy"] = -sum(
                (count / total_targets) * math.log(count / total_targets)
                for count in target_counts.values() if count
            ) if total_targets else 0.0
            trajectory["target_concentration"] = max(target_counts.values(), default=0) / max(1, total_targets)
            trajectories.append(trajectory)
            for row in updates:
                update_rows.append(dict(row))
            lineages = build_residual_lineage(states)
            by_hash = {row["example_id_hash"]: row for row in lineages}
            for row in lineages:
                residual_rows.append({
                    "seed": seed,
                    "arm": arm,
                    **row,
                    **_followup(row, states, updates, candidates),
                })
            for state in states:
                for example in state["examples"]:
                    lineage = by_hash.get(example["example_id_hash"])
                    state_index = int(state["state_index"])
                    validation_rows.append({
                        "experiment_seed": seed,
                        "arm": arm,
                        "state_index": state_index,
                        "after_update_index": state["after_update_index"],
                        "example_id_hash": example["example_id_hash"],
                        "G": example["G"],
                        "H": example["H"],
                        "M": example["M"],
                        "vote_correct": example["vote_correct"],
                        "oracle_covered": example["oracle_covered"],
                        "newly_correct_member_count": example["newly_correct_member_count"],
                        "correct_member_count": example["correct_member_count"],
                        "first_recovery_state": lineage["first_0_to_1_state"] if lineage else None,
                        "deepened_after_recovery": bool(lineage and lineage["later_1_to_2_state"] is not None and state_index >= lineage["later_1_to_2_state"]),
                        "vote_converted_after_recovery": bool(lineage and lineage["first_vote_conversion_state"] is not None and state_index >= lineage["first_vote_conversion_state"]),
                    })
            candidate_rows.extend(candidates)
            for key in total_cost:
                total_cost[key] += int(summary["cost"].get(key, 0))
    classifier = classify(trajectories)
    aggregate = {}
    for arm in ARMS:
        rows = [row for row in trajectories if row["arm"] == arm]
        aggregate[arm] = {
            "accepted_commits": sum(row["accepted_commit_count"] for row in rows),
            "feasible_branches": sum(row["feasible_branch_count"] for row in rows),
            "recoveries_0_to_1": sum(row["recovered_singleton_count"] for row in rows),
            "deepenings_0_to_1_to_2_plus": sum(row["longitudinal_deepened_coverage_count"] for row in rows),
            "persistent_singletons": sum(row["persistent_singleton_count"] for row in rows),
            "cross_member_accumulations": sum(row["cross_member_support_accumulation_count"] for row in rows),
            "recovered_coverage_vote_conversions": sum(row["recovered_coverage_to_vote_count"] for row in rows),
            "mean_final_validation_vote_acc": sum(row["final_validation_vote_acc"] for row in rows) / len(rows),
            "mean_final_validation_oracle_acc": sum(row["final_validation_oracle_acc"] for row in rows) / len(rows),
        }
    summary = {
        "summary_version": "v18_hybrid_online_accumulation_summary_v1",
        "phase_a_gate": source_freeze["phase_a_gate"]["gate"],
        "phase_b_gate": audit["gate"],
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "trajectory_count": len(trajectories),
        "online_horizon": 8,
        "new_test_calls": audit["new_test_calls"],
        "experimental_prompt_commits": audit["experimental_prompt_commits"],
        "experimental_trajectory_transitions": audit["experimental_trajectory_transitions"],
        "conceptual_branch_count": audit["conceptual_branch_count"],
        "actual_branch_evaluation_count": audit["actual_branch_evaluation_count"],
        "actual_candidate_record_count": audit["actual_candidate_record_count"],
        "aggregate": aggregate,
        "cost": total_cost,
        "classifier": classifier,
    }
    write_csv(out / "trajectory_level.csv", trajectories)
    write_csv(out / "update_lineage.csv", update_rows)
    write_csv(out / "validation_longitudinal.csv", validation_rows)
    write_csv(out / "residual_lineage.csv", residual_rows)
    write_csv(out / "candidate_level.csv", candidate_rows)
    write_json(out / "summary.json", summary)
    write_json(out / "classifier.json", classifier)
    write_json(out / "source_freeze.json", {
        "freeze_version": source_freeze["freeze_version"],
        "execution_commit": source_freeze["execution_commit"],
        "registry_content_hash": source_freeze["registry_content_hash"],
        "file_count": source_freeze["file_count"],
        "source_freeze_status": source_freeze["source_freeze_status"],
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source_freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.root.resolve(), read_json(args.audit), read_json(args.source_freeze), args.out.resolve()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
