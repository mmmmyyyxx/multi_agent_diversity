from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_v16_residual_diag_probe_registry import read_jsonl
from build_v17_module1_2x2_registry import reconstruct_case, source_run
from v17_conversion_priority_hybrid_support import (
    ARMS,
    BASE,
    BREADTH,
    CLASSIFIER_VERSION,
    DIRECT,
    arm_specs,
    branch_key,
    canonical_hash,
    generation_config,
)


PARENTS = ((56, 4), (56, 7), (57, 6), (57, 7), (58, 6))


def _responsibility(seed: int, version: int) -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(source_run(seed) / "responsibility_assignments.jsonl")
        if int(row["team_state_version"]) == int(version)
    ]
    if len(rows) != 1:
        raise ValueError("responsibility state is not uniquely reconstructable")
    return rows[0]


def reconstruct_priority_case(seed: int, update_index: int) -> dict[str, Any]:
    case = reconstruct_case({
        "case_id": f"prospective_seed{seed}_u{update_index}",
        "seed": seed,
        "update_index": update_index,
    })
    case.pop("historical_diagnostic_only", None)
    eligible = [int(row["agent_id"]) for row in case["w1_priority_rows"]]
    conversion = {
        str(row["question_hash"]): row
        for row in case["active_profiles"]
        if 0 < int(row["gold_vote_count"]) <= int(row["largest_wrong_vote_count"])
    }
    responsibility = _responsibility(seed, int(case["team_state_version"]))
    service: dict[int, dict[str, dict[str, Any]]] = {agent: {} for agent in range(5)}
    for question_hash, assignment in responsibility["service_assignment_by_question"].items():
        service[int(assignment["service_agent_id"])][str(question_hash)] = assignment
    scores: dict[str, dict[str, int]] = {}
    for agent in eligible:
        hashes = set(service[agent]) & set(conversion)
        scores[str(agent)] = {
            "conversion_responsibility_count": len(hashes),
            "direct_vote_flip_count": sum(
                str(service[agent][question_hash]["repair_lane"]) == "direct_flip"
                for question_hash in hashes
            ),
        }
    case["responsibility_eligible_ids"] = sorted(eligible)
    case["conversion_residual_count"] = len(conversion)
    case["selector_scores_by_agent"] = scores
    case["arm_specs"] = arm_specs(case)
    case["generation_config"] = generation_config(case)
    case["generation_config_hash"] = canonical_hash(case["generation_config"])
    targets = sorted({
        int(row["target_member"])
        for rows in case["arm_specs"].values()
        for row in rows
    })
    case["unique_branch_keys"] = {
        str(target): branch_key(case, target) for target in targets
    }
    return case


def build_registry(execution_commit: str = "PHASE_A_UNCOMMITTED") -> dict[str, Any]:
    cases = [reconstruct_priority_case(seed, update) for seed, update in PARENTS]
    unique = sum(len(case["unique_branch_keys"]) for case in cases)
    breadth_different = sum(
        case["arm_specs"][BREADTH][1]["target_member"]
        != case["arm_specs"][BASE][1]["target_member"]
        for case in cases
    )
    direct_different = sum(
        case["arm_specs"][DIRECT][1]["target_member"]
        != case["arm_specs"][BASE][1]["target_member"]
        for case in cases
    )
    cross_different = sum(
        case["arm_specs"][DIRECT][1]["target_member"]
        != case["arm_specs"][BREADTH][1]["target_member"]
        for case in cases
    )
    payload = {
        "registry_version": "v17_conversion_priority_three_arm_registry_v1",
        "execution_commit": execution_commit,
        "source_runtime_commits": sorted({case["source_runtime_commit"] for case in cases}),
        "parent_selection": {
            "selection": "all_five_naturally_eligible_unseen_historical_parents",
            "parents": [{"seed": seed, "update_index": update} for seed, update in PARENTS],
            "candidate_outcomes_used": False,
            "validation_used": False,
            "test_used": False,
        },
        "case_count": 5,
        "cell_count": 15,
        "conceptual_branch_count": 30,
        "deduplicated_branch_count": unique,
        "conceptual_source_slot_count": 60,
        "actual_source_slot_budget": unique * 2,
        "source_candidates_per_target": 2,
        "loss_blind_revision_per_valid_source": 1,
        "arms": list(ARMS),
        "breadth_target2_different_count": breadth_different,
        "direct_target2_different_count": direct_different,
        "breadth_vs_direct_target2_different_count": cross_different,
        "model": "qwen3-14b",
        "thinking": False,
        "context_mode": "member_aware_residual_search",
        "proposal_mode": "generic_evolution",
        "candidate_acceptance": "unchanged_common_safe",
        "candidate_ranking": "unchanged_common_safe_ranking",
        "cell_selection": "unchanged_max_one",
        "commit_enabled": False,
        "trajectory_mutation_enabled": False,
        "validation_after_decision_only": True,
        "final_test_enabled": False,
        "proposal_memory_mode": "off",
        "phase_a_api_calls": 0,
        "phase_a_validation_calls": 0,
        "phase_a_test_calls": 0,
        "classifier_version": CLASSIFIER_VERSION,
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
    if out.exists() or ROOT.resolve() not in out.parents:
        raise SystemExit("registry output must be fresh and project-local")
    payload = build_registry(args.execution_commit)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "parents": payload["case_count"],
        "cells": payload["cell_count"],
        "conceptual_branches": payload["conceptual_branch_count"],
        "unique_branches": payload["deduplicated_branch_count"],
        "breadth_different": payload["breadth_target2_different_count"],
        "direct_different": payload["direct_target2_different_count"],
        "breadth_vs_direct_different": payload["breadth_vs_direct_target2_different_count"],
        "api_calls": 0,
        "test_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
