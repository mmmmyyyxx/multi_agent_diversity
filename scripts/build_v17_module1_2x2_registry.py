from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from build_v16_residual_diag_probe_registry import (
    ROOT,
    initial_prompts,
    load_questions,
    read_json,
    read_jsonl,
)


DESIGN_ROOT = ROOT / "experiments/v17_module1_2x2_causal_isolation_20260820"
SOURCE_ROOT = ROOT / "runs/v17_formal_5arm_3seed_20260813"
SOURCE_SETTING = "experimental_v16_efficacy_g_matched"
CELLS = ("A", "B", "C", "D")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def source_run(seed: int) -> Path:
    return (
        SOURCE_ROOT / f"seed{seed}" / "disambiguation_qa"
        / f"{SOURCE_SETTING}_seed{seed}"
    )


def prompt_team_hash(prompts: list[str]) -> str:
    hashes = [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts]
    return hashlib.sha256(json.dumps(
        hashes, ensure_ascii=False, separators=(",", ":")
    ).encode()).hexdigest()


def rr_targets(seed: int, update_index: int) -> list[int]:
    start = seed % 5
    return [
        (start + 2 * update_index) % 5,
        (start + 2 * update_index + 1) % 5,
    ]


def w1_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["expected_update_value"]),
        -float(row["opportunity_value"]),
        -float(row["normalized_direct_fix"]),
        -float(row["normalized_support_margin"]),
        -float(row["normalized_uplift_deficit"]),
        -float(row["normalized_wait"]),
        str(row["seeded_rank"]),
        int(row["agent_id"]),
    )


def reconstruct_case(spec: dict[str, Any]) -> dict[str, Any]:
    seed = int(spec["seed"])
    wanted_update = int(spec["update_index"])
    run = source_run(seed)
    meta = read_json(run / "run_meta.json")
    decisions = sorted(
        read_jsonl(run / "candidate_decisions.jsonl"),
        key=lambda row: int(row["update_index"]),
    )
    states = read_jsonl(run / "peer_state_history.jsonl")
    if len(states) % 75:
        raise ValueError(f"Seed{seed}: incomplete 75-row state blocks")
    blocks = [states[index:index + 75] for index in range(0, len(states), 75)]
    prompts = initial_prompts(meta)
    state_index = 0
    previous: dict[int, dict[str, Any]] = {}
    selected: dict[str, Any] | None = None
    selected_prompts: list[str] | None = None
    selected_state: list[dict[str, Any]] | None = None
    for decision in decisions:
        update = int(decision["update_index"])
        if prompt_team_hash(prompts) != str(decision["parent_team_hash"]):
            raise ValueError(f"Seed{seed} update{update}: parent reconstruction mismatch")
        if update == wanted_update:
            selected = decision
            selected_prompts = list(prompts)
            selected_state = blocks[state_index]
            break
        for branch in decision["branches"]:
            agent = int(branch["target_agent_id"])
            evaluated = int(branch["funnel"].get("stage_a_evaluated", 0)) > 0
            previous[agent] = {
                "attempted": True,
                "empirical_evaluation_completed": evaluated,
                "accepted": bool(
                    decision.get("accepted_prompt_hash")
                    and int(decision.get("target_agent_id", -1)) == agent
                ),
                "target_correct_delta": 0,
                "vote_correct_delta": 0,
                "minimum_member_gain_delta": 0,
                "total_member_gain_delta": 0,
                "assigned_repair_count": 0,
                "rejection_reasons": [],
            }
        accepted_hash = str(decision.get("accepted_prompt_hash") or "")
        if accepted_hash:
            accepted = next(
                row for row in decision["candidates"]
                if str(row["prompt_hash"]) == accepted_hash
            )
            prompts[int(accepted["target_agent_id"])] = str(
                accepted["evaluation"]["prompt"]
            )
            state_index += 1
    if selected is None or selected_prompts is None or selected_state is None:
        raise ValueError(f"Seed{seed}: update {wanted_update} is unavailable")

    priority = next(
        row for row in read_jsonl(run / "target_priority_audit.jsonl")
        if int(row["update_index"]) == wanted_update
    )
    ordered = sorted(priority["priorities"], key=w1_sort_key)
    independently_selected_w1 = [int(row["agent_id"]) for row in ordered[:2]]
    persisted_w1 = list(map(int, priority["selected_target_ids"]))
    if independently_selected_w1 != persisted_w1:
        raise ValueError(f"Seed{seed}: W1 total-order replay mismatch")

    responsibility = next(
        row for row in read_jsonl(run / "responsibility_assignments.jsonl")
        if int(row["team_state_version"]) == state_index
    )
    priority_by_agent = {int(row["agent_id"]): row for row in ordered}
    active_lane = {
        agent: str(priority_by_agent[agent]["active_lane"])
        for agent in priority_by_agent
    }
    active_hashes: dict[int, list[str]] = {agent: [] for agent in range(5)}
    for question_hash, assignment in responsibility[
        "service_assignment_by_question"
    ].items():
        agent = int(assignment["service_agent_id"])
        if str(assignment["repair_lane"]) == active_lane.get(agent):
            active_hashes[agent].append(str(question_hash))
    for agent in set(rr_targets(seed, wanted_update) + persisted_w1):
        if not active_hashes[agent]:
            raise ValueError(
                f"Seed{seed} update{wanted_update}: target {agent} lacks active slice"
            )

    return {
        **spec,
        "source_seed": seed,
        "source_update_index": wanted_update,
        "source_runtime_commit": str(meta["run_identity"]["git_commit"]),
        "source_setting": str(meta["canonical_experiment_setting"]),
        "parent_team_hash": str(selected["parent_team_hash"]),
        "team_state_version": state_index,
        "round_robin_target_ids": rr_targets(seed, wanted_update),
        "w1_target_ids": persisted_w1,
        "w1_independent_replay_target_ids": independently_selected_w1,
        "w1_priority_rows": ordered,
        "active_lane_by_agent": {str(k): v for k, v in active_lane.items()},
        "active_residual_hashes_by_agent": {
            str(agent): sorted(hashes) for agent, hashes in active_hashes.items()
        },
        "base_config": meta["config"],
        "parent_prompts": selected_prompts,
        "questions": load_questions(meta, selected_state),
        "initial_profiles": blocks[0],
        "active_profiles": selected_state,
        "accepted_state_count": state_index + 1,
        "stable_correct_question_hashes_by_agent": {
            str(agent): sorted(
                str(row["question_hash"])
                for row_index, row in enumerate(selected_state)
                if all(
                    bool(block[row_index]["team_correctness"][agent])
                    for block in blocks[:state_index + 1]
                )
            )
            for agent in range(5)
        },
        "previous_update_outcome_by_agent": previous,
        "historical_diagnostic_only": {
            "branch_stage_b_evaluated": sum(
                int(row["funnel"].get("stage_b_evaluated", 0))
                for row in selected["branches"]
            ),
            "branch_feasible": sum(
                int(row["funnel"].get("constraint_feasible", 0))
                for row in selected["branches"]
            ),
            "committed": bool(selected.get("accepted_prompt_hash")),
        },
    }


def build_registry(execution_commit: str = "PHASE_A_UNCOMMITTED") -> dict[str, Any]:
    selection = read_json(DESIGN_ROOT / "parent_selection.json")
    cases = [reconstruct_case(row) for row in selection["parents"]]
    hashes = [row["parent_team_hash"] for row in cases]
    if len(set(hashes)) != 6:
        raise ValueError("selected parent team hashes must be pairwise distinct")
    payload = {
        "registry_version": "v17_module1_2x2_fixed_parent_registry_v1",
        "execution_commit": execution_commit,
        "source_runtime_commits": sorted({
            row["source_runtime_commit"] for row in cases
        }),
        "case_count": 6,
        "cell_count": 24,
        "branch_count": 48,
        "source_candidate_budget": 96,
        "source_candidates_per_target": 2,
        "loss_blind_revision_per_valid_source": 1,
        "model": "qwen3-14b",
        "thinking": False,
        "cells": list(CELLS),
        "commit_enabled": False,
        "trajectory_mutation_enabled": False,
        "phase_a_api_calls": 0,
        "phase_a_validation_calls": 0,
        "final_test_enabled": False,
        "proposal_memory_mode": "off",
        "cases": cases,
    }
    payload["registry_content_hash"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execution_commit", default="PHASE_A_UNCOMMITTED")
    args = parser.parse_args()
    out = args.out.resolve()
    if ROOT.resolve() not in out.parents:
        raise SystemExit("output must be project-local")
    if out.exists():
        raise SystemExit("registry output must be fresh")
    payload = build_registry(args.execution_commit)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "api_calls": 0, "validation_calls": 0,
        "test_calls": 0, "case_count": 6, "cell_count": 24,
        "registry_content_hash": payload["registry_content_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()
