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

from build_v16_residual_diag_probe_registry import read_json, read_jsonl
from build_v17_module1_2x2_registry import reconstruct_case, source_run
from v17_hybrid_target_allocation_support import (
    ARMS,
    arm_specs,
    branch_key,
    canonical_hash,
    generation_config,
    rr_eligible_order,
)


SOURCE_ROOT = ROOT / "runs/v17_formal_5arm_3seed_20260813"
OLD_SELECTION = ROOT / "experiments/v17_module1_2x2_causal_isolation_20260820/parent_selection.json"
TRACKED_SELECTION = ROOT / "experiments/v17_hybrid_target_allocation_pilot_20260821/prospective_parent_selection.json"
SEEDS = (56, 57, 58)


def old_parent_pairs() -> set[tuple[int, int]]:
    payload = read_json(OLD_SELECTION)
    return {(int(row["seed"]), int(row["update_index"])) for row in payload["parents"]}


def eligible_inventory(seed: int) -> list[dict[str, Any]]:
    run = source_run(seed)
    priorities = {
        int(row["update_index"]): row
        for row in read_jsonl(run / "target_priority_audit.jsonl")
    }
    decisions = {
        int(row["update_index"]): row
        for row in read_jsonl(run / "candidate_decisions.jsonl")
    }
    excluded = old_parent_pairs()
    result = []
    for update in sorted(set(priorities) & set(decisions)):
        rows = priorities[update].get("priorities", [])
        if (seed, update) in excluded or len(rows) < 2:
            continue
        result.append({
            "seed": int(seed),
            "update_index": int(update),
            "eligible_member_count": len(rows),
            "parent_team_hash": str(decisions[update]["parent_team_hash"]),
        })
    return result


def _nearest_unused(inventory: list[dict[str, Any]], position: int, used: set[int]) -> int:
    candidates = [index for index in range(len(inventory)) if index not in used]
    if not candidates:
        raise ValueError("prospective inventory is too small")
    return min(candidates, key=lambda index: (abs(index - position), index))


def select_parent_specs() -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for seed in SEEDS:
        inventory = eligible_inventory(seed)
        if len(inventory) < 2:
            raise ValueError(f"Seed{seed} has fewer than two eligible unused parents")
        positions = [round((len(inventory) - 1) / 3), round(2 * (len(inventory) - 1) / 3)]
        used: set[int] = set()
        for ordinal, position in enumerate(positions, start=1):
            index = _nearest_unused(inventory, position, used)
            used.add(index)
            row = inventory[index]
            selected.append({
                "case_id": f"prospective_seed{seed}_p{ordinal}_u{row['update_index']}",
                "stratum": f"prospective_quantile_{ordinal}",
                "seed": seed,
                "update_index": int(row["update_index"]),
                "eligible_sequence_length": len(inventory),
                "eligible_sequence_index": index,
                "target_fraction": "1/3" if ordinal == 1 else "2/3",
                "selection_outcomes_used": False,
            })
    if len(selected) != 6 or {row["seed"] for row in selected} != set(SEEDS):
        raise ValueError("prospective selection inventory mismatch")
    return selected


def selection_payload() -> dict[str, Any]:
    rows = select_parent_specs()
    return {
        "selection_version": "v17_hybrid_prospective_parent_selection_v1",
        "source_setting": "experimental_v16_efficacy_g_matched",
        "selection_rule": "per_seed_sorted_eligible_sequence_nearest_one_third_two_thirds_v1",
        "excluded_diagnostic_parent_count": 6,
        "old_diagnostic_parents_excluded_from_primary": True,
        "selection_uses_validation": False,
        "selection_uses_test": False,
        "selection_uses_branch_or_commit_outcomes": False,
        "parents": rows,
    }


def reconstruct_hybrid_case(spec: dict[str, Any]) -> dict[str, Any]:
    case = reconstruct_case(spec)
    case.pop("historical_diagnostic_only", None)
    ordered = case["w1_priority_rows"]
    eligible = [int(row["agent_id"]) for row in ordered]
    if len(eligible) < 2:
        raise ValueError("parent lacks two responsibility-eligible members")
    case["responsibility_eligible_ids"] = sorted(eligible)
    case["rr_eligible_order"] = rr_eligible_order(
        case["source_seed"], case["source_update_index"], eligible
    )
    case["arm_specs"] = arm_specs(case)
    case["generation_config"] = generation_config(case)
    case["generation_config_hash"] = canonical_hash(case["generation_config"])
    case["unique_branch_keys"] = {
        str(target): branch_key(case, target)
        for target in sorted({
            row["target_member"]
            for arm in case["arm_specs"].values()
            for row in arm
        })
    }
    return case


def build_registry(execution_commit: str = "PHASE_A_UNCOMMITTED") -> dict[str, Any]:
    selection = selection_payload()
    if TRACKED_SELECTION.is_file():
        tracked = read_json(TRACKED_SELECTION)
        if tracked != selection:
            raise ValueError("tracked prospective selection does not match deterministic replay")
    cases = [reconstruct_hybrid_case(row) for row in selection["parents"]]
    old = old_parent_pairs()
    if any((int(row["source_seed"]), int(row["source_update_index"])) in old for row in cases):
        raise ValueError("old diagnostic parent entered prospective primary sample")
    unique_branches = sum(len(row["unique_branch_keys"]) for row in cases)
    payload = {
        "registry_version": "v17_hybrid_target_allocation_registry_v1",
        "execution_commit": execution_commit,
        "source_runtime_commits": sorted({row["source_runtime_commit"] for row in cases}),
        "parent_selection": selection,
        "case_count": 6,
        "cell_count": 18,
        "conceptual_branch_count": 36,
        "deduplicated_branch_count": unique_branches,
        "conceptual_source_slot_count": 72,
        "actual_source_slot_budget": unique_branches * 2,
        "source_candidates_per_target": 2,
        "loss_blind_revision_per_valid_source": 1,
        "arms": list(ARMS),
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
        "classifier_version": "v17_hybrid_target_allocation_classifier_v1",
        "cases": cases,
    }
    payload["registry_content_hash"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execution_commit", default="PHASE_A_UNCOMMITTED")
    parser.add_argument("--write_selection", type=Path)
    args = parser.parse_args()
    if args.write_selection:
        target = args.write_selection.resolve()
        if target.exists() or ROOT.resolve() not in target.parents:
            raise SystemExit("selection output must be fresh and project-local")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(selection_payload(), indent=2) + "\n", encoding="utf-8")
    out = args.out.resolve()
    if out.exists() or ROOT.resolve() not in out.parents:
        raise SystemExit("registry output must be fresh and project-local")
    payload = build_registry(args.execution_commit)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "case_count": 6, "cell_count": 18,
        "conceptual_branch_count": 36,
        "deduplicated_branch_count": payload["deduplicated_branch_count"],
        "api_calls": 0, "validation_calls": 0, "test_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
