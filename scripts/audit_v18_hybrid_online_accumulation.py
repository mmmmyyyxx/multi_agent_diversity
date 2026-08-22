from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from scripts.v18_hybrid_online_accumulation_support import (
    ARMS,
    HYBRID,
    SEEDS,
    UPDATES,
    W1,
    hybrid_targets,
    sha256_json,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_source(freeze: dict[str, Any]) -> list[str]:
    problems = []
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != freeze["execution_commit"]:
        problems.append("execution_commit")
    if dirty:
        problems.append("tracked_worktree_dirty")
    for row in freeze["files"]:
        path = ROOT / row["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            problems.append(f"source_file:{row['path']}")
    return problems


def audit(root: Path, registry: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    blockers = verify_source(freeze)
    if registry["registry_content_hash"] != freeze["registry_content_hash"]:
        blockers.append("registry_freeze_identity")
    payload = {key: value for key, value in registry.items() if key != "registry_content_hash"}
    if sha256_json(payload) != registry["registry_content_hash"]:
        blockers.append("registry_content_hash")
    execution = read_json(root / "execution_summary.json")
    if execution.get("trajectory_count") != 6:
        blockers.append("trajectory_inventory")
    if execution.get("new_test_calls") != 0:
        blockers.append("new_test_call")
    if execution.get("infrastructure_failure_count") != 0:
        blockers.append("infrastructure_failure")
    if execution.get("initialization_matched_within_seed") is not True:
        blockers.append("initialization_mismatch")

    trajectory_rows = []
    conceptual_branches = conceptual_sources = 0
    actual_branch_evaluations = 0
    actual_candidate_records = 0
    for seed in SEEDS:
        hashes = set()
        for arm in ARMS:
            run = root / f"seed{seed}" / arm
            try:
                summary = read_json(run / "online_run_summary.json")
                updates = read_jsonl(run / "update_lineage.jsonl")
                states = read_jsonl(run / "validation_states.jsonl")
                candidates = read_jsonl(run / "candidate_level_sanitized.jsonl")
                decisions = read_jsonl(run / "candidate_decisions.jsonl")
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                blockers.append(f"artifact:{seed}:{arm}:{type(exc).__name__}")
                continue
            trajectory_rows.append(summary)
            hashes.add(summary["initialization_snapshot_hash"])
            if summary["seed"] != seed or summary["arm"] != arm:
                blockers.append(f"run_identity:{seed}:{arm}")
            if summary["planned_update_count"] != UPDATES:
                blockers.append(f"planned_updates:{seed}:{arm}")
            completed = int(summary["completed_update_count"])
            if completed != len(updates) or completed > UPDATES:
                blockers.append(f"completed_updates:{seed}:{arm}")
            if completed != UPDATES and summary.get("early_stop_reason") != "no_actionable_responsibility":
                blockers.append(f"unexpected_early_stop:{seed}:{arm}")
            if summary["new_test_calls"] != 0:
                blockers.append(f"test_call:{seed}:{arm}")
            commits = sum(bool(row["committed"]) for row in updates)
            if summary["accepted_commit_count"] != commits:
                blockers.append(f"commit_count:{seed}:{arm}")
            if summary["validation_evaluation_count"] != 1 + commits:
                blockers.append(f"validation_schedule:{seed}:{arm}")
            if len(states) != 1 + commits:
                blockers.append(f"validation_state_count:{seed}:{arm}")
            if any(
                row["validation_evaluated"] != row["committed"] for row in updates
            ):
                blockers.append(f"validation_changed_state_only:{seed}:{arm}")
            if any(
                row["committed"]
                and row["responsibility_refresh_count_after"]
                <= row["responsibility_refresh_count_before"]
                for row in updates
            ):
                blockers.append(f"responsibility_refresh:{seed}:{arm}")
            if any(
                (row["parent_team_hash"] == row["successor_team_hash"])
                == bool(row["committed"])
                for row in updates
            ):
                blockers.append(f"transition_identity:{seed}:{arm}")
            if any(len(row.get("selected_target_ids", [])) > 2 for row in decisions):
                blockers.append(f"target_budget:{seed}:{arm}")
            for update, decision in zip(updates, decisions, strict=True):
                selected = tuple(map(int, decision.get("selected_target_ids", ())))
                priorities = sorted(
                    decision.get("agent_target_priorities", ()),
                    key=lambda row: (int(row["selection_rank"]), int(row["agent_id"])),
                )
                order = [int(row["agent_id"]) for row in priorities]
                if len(order) >= 2:
                    expected = (
                        tuple(order[:2]) if arm == W1 else hybrid_targets(
                            seed=seed,
                            update_index=int(update["update_index"]),
                            w1_order=order,
                            responsibility_eligible=order,
                        )
                    )
                    if selected != expected:
                        blockers.append(f"selector:{seed}:{arm}:{update['update_index']}")
                conceptual_branches += len(selected)
                conceptual_sources += 2 * len(selected)
                actual_branch_evaluations += len(decision.get("branches", ()))
            valid_sources = sum(
                row["candidate_stage"] == "source" and row["valid"] for row in candidates
            )
            revisions = sum(row["candidate_stage"] == "revision" for row in candidates)
            if revisions != valid_sources:
                blockers.append(f"revision_parity:{seed}:{arm}")
            if any(row["candidate_stage"] not in {"source", "revision"} for row in candidates):
                blockers.append(f"candidate_stage:{seed}:{arm}")
            actual_candidate_records += len(candidates)
        if len(hashes) != 1:
            blockers.append(f"matched_initialization:seed{seed}")
    if len(trajectory_rows) != 6:
        blockers.append("complete_run_count")
    return {
        "audit_version": "v18_hybrid_online_accumulation_audit_v1",
        "gate": "PASS" if not blockers else "FAIL",
        "blockers": sorted(set(blockers)),
        "complete_run_count": len(trajectory_rows),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "planned_trajectory_count": 6,
        "planned_update_opportunities": 6 * UPDATES,
        "conceptual_branch_count": conceptual_branches,
        "conceptual_source_candidate_count": conceptual_sources,
        "actual_branch_evaluation_count": actual_branch_evaluations,
        "actual_candidate_record_count": actual_candidate_records,
        "new_test_calls": sum(row.get("new_test_calls", 0) for row in trajectory_rows),
        "experimental_prompt_commits": sum(
            row.get("experimental_prompt_commits", 0) for row in trajectory_rows
        ),
        "experimental_trajectory_transitions": sum(
            row.get("experimental_trajectory_transitions", 0) for row in trajectory_rows
        ),
        "infrastructure_failure_count": sum(
            row.get("infrastructure_failure_count", 0) for row in trajectory_rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source_freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.root.resolve(), read_json(args.registry), read_json(args.source_freeze)
    )
    write = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(write, encoding="utf-8")
    print(write)
    raise SystemExit(0 if result["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
