from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v18_safety_only_critic_pilot_support import read_json, sha256_file, write_json
from scripts.v18_shadow_raw_critic_support import ARMS


def audit(raw: Path, registry_path: Path, freeze_path: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError("fresh gate root required")
    registry, freeze = read_json(registry_path), read_json(freeze_path)
    blockers: list[str] = []
    if sha256_file(registry_path) != freeze["registry_sha256"]:
        blockers.append("registry_hash")
    cases = []
    for expected in registry["cases"]:
        path = raw / expected["case_id"] / "case_result.json"
        if not path.exists():
            blockers.append(expected["case_id"] + ":missing")
            continue
        case = read_json(path)
        cases.append(case)
        if case["parent_hash"] != expected["parent_team_hash"] or int(case["target_member"]) != int(expected["target_agent_id"]):
            blockers.append(expected["case_id"] + ":identity")
        if set(case["arms"]) != set(ARMS):
            blockers.append(expected["case_id"] + ":arms")
        if not case.get("first_teacher_plan_shared"):
            blockers.append(expected["case_id"] + ":teacher_not_shared")
        if case["team_prompt_commit_count"] or case["trajectory_mutation_count"] or case["test_calls"]:
            blockers.append(expected["case_id"] + ":isolation")
        control, shadow = case["arms"]["canonical_control"], case["arms"]["shadow_raw"]
        if not control["teacher_plan_sequence"] or not shadow["teacher_plan_sequence"] or control["teacher_plan_sequence"][0] != shadow["teacher_plan_sequence"][0]:
            blockers.append(expected["case_id"] + ":first_plan_mismatch")
        if shadow["critic_api_calls"] != 0 or shadow["teacher_api_calls"] != 0:
            blockers.append(expected["case_id"] + ":shadow_replay_api")
        for arm in ARMS:
            branch = case["arms"][arm]
            if branch["state_hash_before"] != branch["state_hash_after"]:
                blockers.append(expected["case_id"] + ":state_mutation")
            if branch["strict_valid_candidates"] > 4:
                blockers.append(expected["case_id"] + ":candidate_budget")
            if branch["test_calls"] or branch["team_prompt_commit_count"] or branch["trajectory_mutation_count"]:
                blockers.append(expected["case_id"] + ":branch_isolation")
        for event in shadow["shadow_events"]:
            if not event["original_rejected"] or not event["shadow_effective_approved"] or not event["original_failed_checks"]:
                blockers.append(expected["case_id"] + ":invalid_shadow_event")
            if event["teacher_plan_hash"] != shadow["teacher_plan_sequence"][0]:
                blockers.append(expected["case_id"] + ":shadow_plan_mismatch")
        if shadow["shadow_intervention_count"] and not shadow["student_reached"]:
            blockers.append(expected["case_id"] + ":intervention_did_not_reach_student")
    if len(cases) != 6:
        blockers.append("case_inventory")
    summary = read_json(raw / "pilot_summary.json") if (raw / "pilot_summary.json").exists() else {}
    if summary.get("branch_count") != 12 or summary.get("test_calls") or summary.get("team_prompt_commit_count") or summary.get("trajectory_mutation_count"):
        blockers.append("summary_contract")
    gate = {
        "gate": "PASS" if not blockers else "HOLD",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "case_count": len(cases),
        "branch_count": len(cases) * 2,
        "shadow_intervention_count": sum(case["arms"]["shadow_raw"]["shadow_intervention_count"] for case in cases),
        "shadow_critic_api_calls": sum(case["arms"]["shadow_raw"]["critic_api_calls"] for case in cases),
        "test_calls": 0,
    }
    out.mkdir(parents=True)
    write_json(out / "audit_summary.json", gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.raw.resolve(), args.registry.resolve(), args.freeze.resolve(), args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()

