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

from multi_dataset_diverse_rl.tcs import PeerStateDiagnosisContext
from build_v17_hybrid_target_allocation_registry import old_parent_pairs, select_parent_specs
from v17_hybrid_target_allocation_support import (
    ARMS,
    HYBRID,
    RR,
    W1,
    arm_specs,
    branch_key,
    context_hashes,
    immutable_state_hash,
    probe_system,
    rr_eligible_order,
)


def preflight(registry: dict[str, Any], scratch: Path) -> dict[str, Any]:
    errors: list[str] = []
    context_rows: list[dict[str, Any]] = []
    cases = registry.get("cases", [])
    expected_selection = select_parent_specs()
    if registry.get("parent_selection", {}).get("parents") != expected_selection:
        errors.append("prospective_parent_selection_replay")
    if len(cases) != 6:
        errors.append("case_count")
    if any((int(row["source_seed"]), int(row["source_update_index"])) in old_parent_pairs() for row in cases):
        errors.append("old_diagnostic_parent_in_primary")
    if sorted(int(row["source_seed"]) for row in cases) != [56, 56, 57, 57, 58, 58]:
        errors.append("seed_balance")
    deduplicated = 0
    for case in cases:
        try:
            arms = arm_specs(case)
            if arms != case["arm_specs"]:
                errors.append(f"arm_replay:{case['case_id']}")
                continue
            eligible = set(map(int, case["responsibility_eligible_ids"]))
            if len(eligible) < 2:
                errors.append(f"eligible_pool:{case['case_id']}")
            w1_rank1 = int(case["w1_priority_rows"][0]["agent_id"])
            rr_order = rr_eligible_order(case["source_seed"], case["source_update_index"], eligible)
            if arms[HYBRID][0]["target_member"] != w1_rank1:
                errors.append(f"hybrid_exploit:{case['case_id']}")
            if arms[HYBRID][1]["target_member"] != next(row for row in rr_order if row != w1_rank1):
                errors.append(f"hybrid_explore_rr:{case['case_id']}")
            unique_targets = sorted({
                row["target_member"] for arm in arms.values() for row in arm
            })
            deduplicated += len(unique_targets)
            for arm in ARMS:
                targets = [int(row["target_member"]) for row in arms[arm]]
                if len(targets) != 2 or len(set(targets)) != 2 or not set(targets).issubset(eligible):
                    errors.append(f"target_contract:{case['case_id']}:{arm}")
            for target in unique_targets:
                system = probe_system(
                    case, target=target,
                    out_dir=scratch / case["case_id"] / str(target), cache_path="",
                )
                before = immutable_state_hash(system)
                hashes = context_hashes(case, target)
                context, _ = system._proposal_context(
                    target, system.agents[target].current_prompt, hashes
                )
                if not isinstance(context, PeerStateDiagnosisContext):
                    errors.append(f"context_type:{case['case_id']}:{target}")
                if not hashes:
                    errors.append(f"empty_context:{case['case_id']}:{target}")
                if immutable_state_hash(system) != before:
                    errors.append(f"state_mutation:{case['case_id']}:{target}")
                if system.team_prompt_state_hash() != case["parent_team_hash"]:
                    errors.append(f"parent_hash:{case['case_id']}:{target}")
                if branch_key(case, target) != case["unique_branch_keys"][str(target)]:
                    errors.append(f"branch_key:{case['case_id']}:{target}")
                context_rows.append({
                    "case_id": case["case_id"], "target_member": target,
                    "context_type": type(context).__name__,
                    "assigned_hash_count": len(hashes),
                })
        except Exception as exc:
            errors.append(f"case_exception:{case.get('case_id')}:{type(exc).__name__}:{exc}")
    for key, expected in (
        ("case_count", 6), ("cell_count", 18),
        ("conceptual_branch_count", 36), ("conceptual_source_slot_count", 72),
        ("source_candidates_per_target", 2),
        ("loss_blind_revision_per_valid_source", 1),
    ):
        if int(registry.get(key, -1)) != expected:
            errors.append(key)
    if int(registry.get("deduplicated_branch_count", -1)) != deduplicated:
        errors.append("deduplicated_branch_count")
    if registry.get("arms") != list(ARMS):
        errors.append("arm_inventory")
    if registry.get("classifier_version") != "v17_hybrid_target_allocation_classifier_v1":
        errors.append("classifier_not_frozen")
    if any(int(registry.get(key, -1)) for key in (
        "phase_a_api_calls", "phase_a_validation_calls", "phase_a_test_calls"
    )):
        errors.append("phase_a_isolation")
    if registry.get("final_test_enabled") is not False:
        errors.append("test_policy")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "prospective_parents_frozen": len(cases),
        "old_diagnostic_parents_excluded_from_primary": "old_diagnostic_parent_in_primary" not in errors,
        "api_calls": 0, "model_calls": 0, "validation_calls": 0, "test_calls": 0,
        "cell_count": 18, "conceptual_branch_count": 36,
        "deduplicated_branch_count": deduplicated,
        "context_checks": context_rows,
        "selector_invariants": {
            "hybrid_target1_w1_rank1": not any("hybrid_exploit" in row for row in errors),
            "hybrid_target2_rr_remaining_eligible": not any("hybrid_explore_rr" in row for row in errors),
            "all_targets_legal_distinct": not any("target_contract" in row for row in errors),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = preflight(json.loads(args.registry.read_text(encoding="utf-8")), args.scratch.resolve())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
