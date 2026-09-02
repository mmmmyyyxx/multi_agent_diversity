from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from scripts.v18_safety_only_critic_pilot_support import ARMS, ROOT, read_json, sha256_file, write_json


def audit(raw: Path, registry_path: Path, freeze_path: Path, out: Path) -> dict:
    if out.exists(): raise FileExistsError("fresh gate root required")
    registry, freeze = read_json(registry_path), read_json(freeze_path)
    blockers = []
    if sha256_file(registry_path) != freeze["registry_sha256"]: blockers.append("registry_hash")
    rows = []
    for case in registry["cases"]:
        path = raw / case["case_id"] / "case_result.json"
        if not path.exists(): blockers.append(case["case_id"] + ":missing"); continue
        row = read_json(path); rows.append(row)
        if row["parent_hash"] != case["parent_team_hash"] or int(row["target_member"]) != int(case["target_agent_id"]): blockers.append(case["case_id"] + ":identity")
        if set(row["arms"]) != set(ARMS): blockers.append(case["case_id"] + ":arms")
        if row["safety_only_critic_api_calls"] != 0: blockers.append(case["case_id"] + ":safety_api")
        if row["team_prompt_commit_count"] or row["trajectory_mutation_count"] or row["test_calls"]: blockers.append(case["case_id"] + ":isolation")
        for arm in ARMS:
            branch = row["arms"][arm]
            if branch["state_hash_before"] != branch["state_hash_after"]: blockers.append(case["case_id"] + ":state_mutation")
            if branch["strict_valid_candidates"] > 4: blockers.append(case["case_id"] + ":candidate_budget")
            if arm == "deterministic_safety_only" and branch["critic_api_calls"]: blockers.append(case["case_id"] + ":critic_api")
    if len(rows) != 6: blockers.append("case_inventory")
    summary = read_json(raw / "pilot_summary.json")
    if summary["branch_count"] != 12 or summary["test_calls"] or summary["team_prompt_commit_count"]: blockers.append("summary_contract")
    gate = {"gate": "PASS" if not blockers else "HOLD", "blocker_count": len(blockers), "blockers": blockers, "case_count": len(rows), "branch_count": len(rows) * 2, "safety_only_critic_api_calls": sum(row["safety_only_critic_api_calls"] for row in rows), "test_calls": 0}
    out.mkdir(parents=True); write_json(out / "audit_summary.json", gate); return gate


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--raw",type=Path,required=True); parser.add_argument("--registry",type=Path,required=True); parser.add_argument("--freeze",type=Path,required=True); parser.add_argument("--out",type=Path,required=True); args=parser.parse_args()
    print(json.dumps(audit(args.raw.resolve(),args.registry.resolve(),args.freeze.resolve(),args.out.resolve()),indent=2))


if __name__ == "__main__": main()
