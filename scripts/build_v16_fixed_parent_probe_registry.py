from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT / "runs" / "v16p51" / "disambiguation_qa"
    / "experimental_v16_c3_coalition_aware_preservation_seed51"
)
VARIANTS = (
    "c0_current_v15",
    "c2_boundary_plus_preservation",
    "c3_coalition_aware_preservation",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_registry(run_dir: Path) -> dict[str, Any]:
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
    prompts = json.loads((run_dir / "best_prompts.json").read_text(encoding="utf-8"))
    states = read_jsonl(run_dir / "peer_state_history.jsonl")
    assignments = read_jsonl(run_dir / "responsibility_assignments.jsonl")
    scores = read_jsonl(run_dir / "repairability_adjusted_target_scores.jsonl")
    if len(prompts) != 5 or len(states) % 75:
        raise ValueError("Seed51 final prompt/profile inventory is incomplete")
    state_blocks = [states[index:index + 75] for index in range(0, len(states), 75)]
    final_state = state_blocks[-1]
    initial_state = state_blocks[0]
    final_assignment = assignments[-1]
    final_version = int(final_assignment["team_state_version"])
    latest_update = max(int(row["update_index"]) for row in scores)
    latest_scores = sorted(
        (row for row in scores if int(row["update_index"]) == latest_update),
        key=lambda row: int(row["selection_rank"]),
    )
    selected = latest_scores[:3]
    if len(selected) != 3 or any(int(row["active_lane_size"]) <= 0 for row in selected):
        raise ValueError("final state does not contain three actionable targets")
    train_path = Path(meta["config"]["train_path"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    with train_path.open(encoding="utf-8-sig", newline="") as handle:
        dataset = list(csv.DictReader(handle))
    if len(dataset) != 75:
        raise ValueError("fixed probe must contain exactly 75 training examples")
    questions = []
    by_hash = {row["question_hash"]: row for row in final_state}
    for row in dataset:
        question = str(row["question"])
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if question_hash not in by_hash:
            raise ValueError("dataset and persisted final profile do not match")
        questions.append({"question": question, "answer": row["answer"], "question_hash": question_hash})
    service = final_assignment["service_assignment_by_question"]
    cases = []
    for case_index, score in enumerate(selected):
        target = int(score["agent_id"])
        lane = str(score["active_lane"])
        assigned = sorted(
            question_hash for question_hash, row in service.items()
            if int(row["service_agent_id"]) == target and str(row["repair_lane"]) == lane
        )
        if len(assigned) != int(score["active_lane_size"]):
            raise ValueError("active slice does not reconcile to service routing")
        cases.append({
            "case_id": f"final_state_rank_{case_index + 1}_target_{target}",
            "selection_rule": "final_state_actionable_target_selection_rank_first_three_v1",
            "parent_team_hash": str(score["team_prompt_state_hash"]),
            "team_state_version": final_version,
            "source_update_index": latest_update,
            "target_agent_id": target,
            "target_selection_rank": int(score["selection_rank"]),
            "active_lane": lane,
            "assigned_question_hashes": assigned,
            "cell_order": [VARIANTS[(case_index + offset) % 3] for offset in range(3)],
        })
    stable = {}
    for agent in range(5):
        stable[str(agent)] = sorted(
            row["question_hash"] for row in final_state
            if all(bool(block[final_state.index(row)]["team_correctness"][agent]) for block in state_blocks)
        )
    payload = {
        "registry_version": "v16_fixed_parent_generation_probe_registry_v1",
        "source_runtime_commit": str(meta["run_identity"]["git_commit"]),
        "source_setting": str(meta["canonical_experiment_setting"]),
        "source_seed": 51,
        "case_selection_uses_candidate_outcomes": False,
        "case_selection_pool": "immutable_seed51_c3_final_active_state",
        "candidate_count_per_cell": 2,
        "commit_enabled": False,
        "validation_enabled": False,
        "final_test_enabled": False,
        "variants": list(VARIANTS),
        "cases": cases,
        "base_config": meta["config"],
        "parent_prompts": prompts,
        "questions": questions,
        "initial_profiles": initial_state,
        "active_profiles": final_state,
        "stable_correct_question_hashes_by_agent": stable,
        "accepted_state_count": len(state_blocks),
        "source_final_summary_hash": sha256_json(summary),
    }
    payload["registry_content_hash"] = sha256_json(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out = args.out.resolve()
    if ROOT.resolve() not in out.parents:
        raise SystemExit("registry output must remain under the project root")
    payload = build_registry(run_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "api_calls": 0, "case_count": len(payload["cases"]),
        "cell_count": len(payload["cases"]) * 3, "registry_content_hash": payload["registry_content_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()
