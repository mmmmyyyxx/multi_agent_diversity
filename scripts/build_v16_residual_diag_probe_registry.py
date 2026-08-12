from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "m20_current_v15",
    "m2a_residual_diagnosis",
    "m2b_diagnosis_minimal_edit",
    "m2c_diagnosis_minimal_edit_relevance_critic",
)
SOURCE_RUNS = {
    48: ROOT / "runs/v15f48/disambiguation_qa/shared_responsibility_conditioned_dual_target_seed48",
    49: ROOT / "runs/v15f49/disambiguation_qa/shared_responsibility_conditioned_dual_target_seed49",
    50: ROOT / "runs/v15f50/disambiguation_qa/shared_responsibility_conditioned_dual_target_seed50",
    51: ROOT / "runs/v16p51/disambiguation_qa/experimental_v16_c0_current_v15_seed51",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def initial_prompts(meta: dict[str, Any]) -> list[str]:
    config = meta["config"]
    supplied = config.get("provided_prompts_json")
    if supplied:
        prompts = json.loads(supplied)
    else:
        prompts = [str(config["shared_prompt"])] * 5
    if len(prompts) != 5:
        raise ValueError("initial prompt inventory must contain exactly five prompts")
    return prompts


def load_questions(meta: dict[str, Any], state: list[dict[str, Any]]) -> list[dict[str, str]]:
    path = Path(meta["config"]["train_path"])
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_hash = {str(row["question_hash"]): row for row in state}
    result = []
    for row in rows:
        question = str(row["question"])
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if question_hash not in by_hash:
            raise ValueError("training data and initial persisted profile do not match")
        result.append({"question": question, "answer": str(row["answer"]), "question_hash": question_hash})
    if len(result) != 75:
        raise ValueError("fixed training probe must contain 75 examples")
    return result


def build_case(seed: int, run_dir: Path, case_index: int) -> dict[str, Any]:
    meta = read_json(run_dir / "run_meta.json")
    decisions = read_jsonl(run_dir / "candidate_decisions.jsonl")
    states = read_jsonl(run_dir / "peer_state_history.jsonl")
    if len(states) < 75:
        raise ValueError(f"Seed{seed}: missing initial peer state")
    if len(states) % 75:
        raise ValueError(f"Seed{seed}: peer-state rows do not form complete states")
    state_blocks = [states[index:index + 75] for index in range(0, len(states), 75)]
    initial = state_blocks[0]
    prompts = initial_prompts(meta)
    prompt_hashes = [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts]
    selected_case = None
    selected_prompts: list[str] | None = None
    selected_state: list[dict[str, Any]] | None = None
    selected_state_index = -1
    current_prompts = list(prompts)
    state_index = 0
    previous_outcomes: dict[int, dict[str, Any]] = {}
    for decision in sorted(decisions, key=lambda row: int(row["update_index"])):
        prompt_hashes = [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in current_prompts]
        reconstructed_hash = hashlib.sha256(
            json.dumps(prompt_hashes, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if str(decision["parent_team_hash"]) != reconstructed_hash:
            raise ValueError(f"Seed{seed}: historical parent prompt reconstruction failed")
        active_state = state_blocks[state_index]
        by_hash = {str(row["question_hash"]): row for row in active_state}
        for branch in sorted(decision["branches"], key=lambda row: (int(row["target_selection_rank"]), int(row["target_agent_id"]))):
            target = int(branch["target_agent_id"])
            assigned = sorted(map(str, branch["assigned_question_hashes"]))
            legal = []
            for question_hash in assigned:
                row = by_hash[question_hash]
                target_wrong = not bool(row["team_correctness"][target])
                peer_correct = any(
                    bool(correct) for agent, correct in enumerate(row["team_correctness"])
                    if agent != target
                )
                if target_wrong and peer_correct:
                    legal.append(question_hash)
            if assigned and legal:
                selected_case = (decision, branch, legal)
                selected_prompts = list(current_prompts)
                selected_state = active_state
                selected_state_index = state_index
                break
        if selected_case:
            break
        winner_target = decision.get("target_agent_id")
        winner_hash = str(decision.get("accepted_prompt_hash") or "")
        for branch in decision["branches"]:
            agent = int(branch["target_agent_id"])
            evaluated = int(branch["funnel"].get("stage_a_evaluated", 0)) > 0
            branch_winner = str(branch.get("branch_winner_hash") or "")
            previous_outcomes[agent] = {
                "attempted": True,
                "empirical_evaluation_completed": evaluated,
                "accepted": bool(winner_hash and agent == winner_target),
                "target_correct_delta": 0,
                "vote_correct_delta": 0,
                "minimum_member_gain_delta": 0,
                "total_member_gain_delta": 0,
                "assigned_repair_count": 0,
                "rejection_reasons": (
                    ["cross_branch_competition_loser"]
                    if branch_winner and not (winner_hash and agent == winner_target)
                    else sorted({
                        reason
                        for row in decision["candidates"]
                        if int(row["target_agent_id"]) == agent
                        for reason in (row.get("constraint") or {}).get("rejection_reasons", [])
                    })
                ),
            }
        accepted_hash = str(decision.get("accepted_prompt_hash") or "")
        if accepted_hash:
            accepted = next(
                row for row in decision["candidates"]
                if str(row["prompt_hash"]) == accepted_hash
            )
            current_prompts[int(accepted["target_agent_id"])] = str(accepted["evaluation"]["prompt"])
            state_index += 1
            if state_index >= len(state_blocks):
                raise ValueError(f"Seed{seed}: accepted transition lacks persisted state")
    if selected_case is None:
        raise ValueError(f"Seed{seed}: no reconstructible structural case")
    decision, branch, legal = selected_case
    assert selected_prompts is not None and selected_state is not None
    return {
        "case_id": f"seed{seed}_earliest_structural_parent_target{branch['target_agent_id']}",
        "source_seed": seed,
        "selection_rule": "earliest_reconstructible_parent_then_target_rank_agent_id_v1",
        "source_runtime_commit": str(meta["run_identity"]["git_commit"]),
        "source_setting": str(meta["canonical_experiment_setting"]),
        "source_update_index": int(decision["update_index"]),
        "parent_team_hash": str(decision["parent_team_hash"]),
        "team_state_version": selected_state_index,
        "target_agent_id": int(branch["target_agent_id"]),
        "target_selection_rank": int(branch["target_selection_rank"]),
        "active_lane": str(branch["active_lane"]),
        "assigned_question_hashes": sorted(map(str, branch["assigned_question_hashes"])),
        "structurally_qualifying_question_hashes": legal,
        "cell_order": [VARIANTS[(case_index + offset) % 4] for offset in range(4)],
        "base_config": meta["config"],
        "parent_prompts": selected_prompts,
        "questions": load_questions(meta, selected_state),
        "initial_profiles": initial,
        "active_profiles": selected_state,
        "stable_correct_question_hashes_by_agent": {
            str(agent): sorted(
                str(row["question_hash"]) for row_index, row in enumerate(selected_state)
                if all(bool(block[row_index]["team_correctness"][agent]) for block in state_blocks[:selected_state_index + 1])
            ) for agent in range(5)
        },
        "accepted_state_count": selected_state_index + 1,
        "previous_update_outcome_by_agent": previous_outcomes,
    }


def build_registry() -> dict[str, Any]:
    cases = [build_case(seed, SOURCE_RUNS[seed], index) for index, seed in enumerate((48, 49, 50, 51))]
    payload = {
        "registry_version": "v16_residual_diag_fixed_parent_registry_v1",
        "case_selection_uses_candidate_outcomes": False,
        "case_selection_fully_reconstructible_only": True,
        "candidate_count_per_cell": 2,
        "commit_enabled": False,
        "validation_enabled": False,
        "final_test_enabled": False,
        "model": "qwen3-14b",
        "thinking": False,
        "variants": list(VARIANTS),
        "cases": cases,
    }
    payload["registry_content_hash"] = digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if ROOT.resolve() not in out.parents:
        raise SystemExit("output must remain under project root")
    if out.exists():
        raise SystemExit("registry output must be fresh")
    payload = build_registry()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "api_calls": 0, "case_count": 4, "cell_count": 16,
        "membership_mismatch": 0, "registry_content_hash": payload["registry_content_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()
