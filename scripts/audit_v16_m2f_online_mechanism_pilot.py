from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


SETTING = "experimental_v16_m2f_online_compatibility_repair"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit(run: Path, freeze: Path) -> dict[str, Any]:
    meta = read_json(run / "run_meta.json")
    final = read_json(run / "final_summary.json")
    decisions = read_jsonl(run / "candidate_decisions.jsonl")
    branches = read_jsonl(run / "dual_target_branch_decisions.jsonl")
    commits = read_jsonl(run / "dual_target_commit_decisions.jsonl")
    events = read_jsonl(run / "online_compatibility_repair_events.jsonl")
    frozen = read_json(freeze)
    blockers: list[str] = []
    if meta.get("run_identity", {}).get("git_commit") != frozen.get("git_head"):
        blockers.append("source_identity")
    if meta.get("run_identity", {}).get("experiment_setting") != SETTING:
        blockers.append("setting_identity")
    if int(meta.get("planned_update_count", -1)) != 8 or int(meta.get("completed_update_count", -1)) != 8:
        blockers.append("update_count")
    selection = final.get("selection_summary", {})
    if (
        bool(selection.get("validation_used"))
        or int(selection.get("validation_evaluation_count", -1)) != 0
        or int(selection.get("test_evaluation_count", -1)) != 0
        or bool(meta.get("final_test_enabled"))
    ):
        blockers.append("train_only")
    if len(decisions) != 8 or len(commits) != 8 or len(branches) > 16:
        blockers.append("trajectory_inventory")
    decision_candidates = {
        (int(decision["update_index"]), str(row["prompt_hash"])): row
        for decision in decisions for row in decision.get("candidates", [])
    }
    decision_branches = {
        (int(decision["update_index"]), int(branch["target_agent_id"])): branch
        for decision in decisions for branch in decision.get("branches", [])
    }
    commit_hashes = {
        (int(row["update_index"]), str(row.get("committed_prompt_hash", "")))
        for row in commits if row.get("committed_prompt_hash")
    }
    source_feasible = 0
    repair_attributable = 0
    for event in events:
        update = int(event["update_index"])
        source_key = (update, str(event["source_candidate_hash"]))
        source = decision_candidates.get(source_key)
        if source is None or source.get("candidate_stage") != "m20_source":
            blockers.append("source_join")
            continue
        branch = decision_branches.get(
            (update, int(event["target_agent_id"]))
        )
        if branch is None:
            blockers.append("branch_join")
        else:
            expected_parent = next(
                (str(row.get("parent_team_hash", "")) for row in commits
                 if int(row["update_index"]) == update),
                "",
            )
            if str(event.get("parent_team_hash", "")) != expected_parent:
                blockers.append("parent_hash")
        source_passed = bool((source.get("constraint") or {}).get("passed"))
        source_feasible += int(source_passed)
        if source_passed != bool(event["source_common_safe"]):
            blockers.append("source_common_safe_mismatch")
        if event["repair_attempted"] != event["repair_eligible"]:
            blockers.append("attempt_budget")
        if event["repair_feasible"] and not event["repair_output_valid"]:
            blockers.append("invalid_feasible")
        repaired_hash = str(event.get("repaired_candidate_hash", ""))
        if repaired_hash:
            repaired = decision_candidates.get((update, repaired_hash))
            if repaired is None or repaired.get("candidate_stage") != "compatibility_repair":
                blockers.append("repair_join")
            elif str(repaired.get("repair_plan_hash")) != str(event["source_candidate_hash"]):
                blockers.append("source_child_mismatch")
        expected_committed = bool(repaired_hash and (update, repaired_hash) in commit_hashes)
        if bool(event["repair_committed"]) != expected_committed:
            blockers.append("commit_join")
        if expected_committed:
            if source_passed:
                blockers.append("attribution_source_feasible")
            repair_attributable += 1
    summary = {
        "gate": "PASS" if not blockers else "FAIL",
        "blockers": sorted(set(blockers)),
        "seed": 52,
        "planned_updates": 8,
        "completed_updates": len(decisions),
        "m20_source_candidates": len(events),
        "m20_feasible": source_feasible,
        "repair_eligible": sum(bool(row["repair_eligible"]) for row in events),
        "repair_attempted": sum(bool(row["repair_attempted"]) for row in events),
        "repair_output_valid": sum(bool(row["repair_output_valid"]) for row in events),
        "repair_feasible": sum(bool(row["repair_feasible"]) for row in events),
        "repair_committed": sum(bool(row["repair_committed"]) for row in events),
        "repair_attributable_accepted_updates": repair_attributable,
        "validation_evaluations": int(selection.get("validation_evaluation_count", -1)),
        "test_evaluations": int(selection.get("test_evaluation_count", -1)),
        "critical_net_analysis_metric_only": True,
        "oracle_delta_analysis_metric_only": True,
        "vote_net_analysis_metric_only": True,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh audit output required")
    value = audit(args.run, args.freeze)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    raise SystemExit(0 if value["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
