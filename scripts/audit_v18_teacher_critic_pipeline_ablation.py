from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v18_safety_only_critic_pilot_support import canonical_hash, read_json, sha256_file, write_json
from scripts.v18_teacher_critic_pipeline_support import ARMS


def audit(raw: Path, registry_path: Path, freeze_path: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh audit root required")
    registry, freeze = read_json(registry_path), read_json(freeze_path)
    blockers: list[str] = []
    if sha256_file(registry_path) != freeze["registry_sha256"]:
        blockers.append("registry_hash_mismatch")
    if canonical_hash({key: value for key, value in registry.items() if key != "registry_content_hash"}) != registry["registry_content_hash"]:
        blockers.append("registry_content_mismatch")
    for item in freeze["files"]:
        if not (ROOT / item["path"]).is_file() or sha256_file(ROOT / item["path"]) != item["sha256"]:
            blockers.append(f"source_freeze_mismatch:{item['path']}")
    summary_path = raw / "pilot_summary.json"
    if not summary_path.is_file():
        blockers.append("missing_pilot_summary")
    branch_count = 0
    candidate_count = 0
    validation_count = 0
    for case in registry["cases"]:
        case_root = raw / case["case_id"]
        freeze_file = case_root / "train_freeze.json"
        result_file = case_root / "case_result.json"
        if not freeze_file.is_file() or not result_file.is_file():
            blockers.append(f"missing_case_artifact:{case['case_id']}")
            continue
        train_mtime = freeze_file.stat().st_mtime_ns
        result = read_json(result_file)
        if result.get("parent_hash") != case["parent_team_hash"]:
            blockers.append(f"parent_hash_mismatch:{case['case_id']}")
        if set(result.get("arms", {})) != set(ARMS):
            blockers.append(f"arm_inventory:{case['case_id']}")
        for arm in ARMS:
            path = case_root / arm / "branch_result.json"
            if not path.is_file():
                blockers.append(f"missing_branch:{case['case_id']}:{arm}")
                continue
            branch_count += 1
            if path.stat().st_mtime_ns < train_mtime:
                blockers.append(f"validation_before_train_freeze:{case['case_id']}:{arm}")
            row = read_json(path)
            candidates = list(row.get("candidate_rows", []))
            candidate_count += len(candidates)
            validation_count += int(bool(row.get("winner_hash")))
            if row.get("parent_hash") != case["parent_team_hash"]:
                blockers.append(f"branch_parent:{case['case_id']}:{arm}")
            if int(row.get("target_member", -1)) != int(case["target_agent_id"]):
                blockers.append(f"branch_target:{case['case_id']}:{arm}")
            if row.get("state_hash_before") != row.get("state_hash_after"):
                blockers.append(f"state_mutation:{case['case_id']}:{arm}")
            if row.get("team_prompt_commit_count") or row.get("trajectory_mutation_count"):
                blockers.append(f"commit_or_mutation:{case['case_id']}:{arm}")
            if row.get("test_calls"):
                blockers.append(f"test_access:{case['case_id']}:{arm}")
            if len(candidates) > 4:
                blockers.append(f"candidate_budget:{case['case_id']}:{arm}")
            source_count = sum(item.get("candidate_stage") == "source" for item in candidates)
            revision_count = sum(item.get("candidate_stage") == "revision" for item in candidates)
            if source_count > 2 or revision_count > source_count:
                blockers.append(f"source_revision_budget:{case['case_id']}:{arm}")
            hard = list(row.get("hard_gate_decisions", []))
            if arm in {"A_CANONICAL", "B_TEACHER_CLEAN"} and hard:
                blockers.append(f"unexpected_hard_gate:{case['case_id']}:{arm}")
            if arm in {"C_NO_SEMANTIC_CRITIC", "D_ADVISORY_CRITIC"}:
                for decision in hard:
                    if not decision.get("teacher_plan_hash"):
                        blockers.append(f"hard_gate_missing_plan:{case['case_id']}:{arm}")
                    if decision.get("category") not in {"none", "schema", "anti_cheating", "output_contract"}:
                        blockers.append(f"hard_gate_category:{case['case_id']}:{arm}")
            if arm == "A_CANONICAL" and not row.get("canonical_passthrough"):
                blockers.append(f"canonical_not_passthrough:{case['case_id']}")
            if arm == "C_NO_SEMANTIC_CRITIC":
                if row.get("semantic_critic_used") or row.get("advisory_api_calls"):
                    blockers.append(f"semantic_critic_used_in_c:{case['case_id']}")
            if arm == "D_ADVISORY_CRITIC":
                if any(item.get("effective_block") for item in row.get("semantic_critic_decisions", [])):
                    blockers.append(f"advisory_blocked:{case['case_id']}")
            if arm in {"A_CANONICAL", "B_TEACHER_CLEAN"} and not row.get("semantic_critic_used"):
                blockers.append(f"semantic_critic_missing:{case['case_id']}:{arm}")
    if summary_path.is_file():
        summary = read_json(summary_path)
        if int(summary.get("case_count", -1)) != len(registry["cases"]):
            blockers.append("summary_case_count")
        if int(summary.get("branch_count", -1)) != len(registry["cases"]) * len(ARMS):
            blockers.append("summary_branch_count")
        if summary.get("test_calls") or summary.get("team_prompt_commit_count") or summary.get("trajectory_mutation_count"):
            blockers.append("summary_isolation")
    audit_summary = {
        "gate": "PASS" if not blockers else "HOLD",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "case_count": len(registry["cases"]),
        "arm_count": len(ARMS),
        "branch_count": branch_count,
        "candidate_count": candidate_count,
        "winner_only_validation_count": validation_count,
        "test_calls": 0,
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
    }
    out.mkdir(parents=True)
    write_json(out / "audit_summary.json", audit_summary)
    return audit_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.raw.resolve(), args.registry.resolve(), args.freeze.resolve(), args.out.resolve()]
    if any(ROOT.resolve() not in path.parents for path in paths):
        raise SystemExit("project-local paths required")
    result = audit(*paths)
    print(json.dumps(result, indent=2))
    if result["gate"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
