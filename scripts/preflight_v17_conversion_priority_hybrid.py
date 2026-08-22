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
from build_v17_conversion_priority_hybrid_registry import PARENTS, build_registry
from v17_conversion_priority_hybrid_support import (
    ARMS,
    BASE,
    BREADTH,
    CLASSIFIER_VERSION,
    DIRECT,
    arm_specs,
    branch_key,
    context_hashes,
    immutable_state_hash,
    probe_system,
)


def preflight(registry: dict[str, Any], scratch: Path) -> dict[str, Any]:
    errors: list[str] = []
    cases = registry.get("cases", [])
    if [(int(row["source_seed"]), int(row["source_update_index"])) for row in cases] != list(PARENTS):
        errors.append("parent_inventory")
    unique = 0
    context_checks = 0
    for case in cases:
        try:
            arms = arm_specs(case)
            if arms != case["arm_specs"]:
                errors.append(f"selector_replay:{case['case_id']}")
            unique_targets = sorted({
                int(row["target_member"]) for rows in arms.values() for row in rows
            })
            unique += len(unique_targets)
            if len({rows[0]["target_member"] for rows in arms.values()}) != 1:
                errors.append(f"target1_not_shared:{case['case_id']}")
            for target in unique_targets:
                system = probe_system(
                    case,
                    target=target,
                    out_dir=scratch / case["case_id"] / str(target),
                    cache_path="",
                )
                before = immutable_state_hash(system)
                hashes = context_hashes(case, target)
                context, _ = system._proposal_context(
                    target, system.agents[target].current_prompt, hashes
                )
                if not isinstance(context, PeerStateDiagnosisContext):
                    errors.append(f"context_type:{case['case_id']}:{target}")
                if immutable_state_hash(system) != before:
                    errors.append(f"state_mutation:{case['case_id']}:{target}")
                if branch_key(case, target) != case["unique_branch_keys"][str(target)]:
                    errors.append(f"branch_key:{case['case_id']}:{target}")
                context_checks += 1
        except Exception as exc:
            errors.append(f"case_exception:{case.get('case_id')}:{type(exc).__name__}:{exc}")
    checks = {
        "case_count": int(registry.get("case_count", -1)) == 5,
        "cell_count": int(registry.get("cell_count", -1)) == 15,
        "conceptual_branches": int(registry.get("conceptual_branch_count", -1)) == 30,
        "unique_branches": int(registry.get("deduplicated_branch_count", -1)) == unique == 15,
        "source_budget": int(registry.get("actual_source_slot_budget", -1)) == 30,
        "arms": registry.get("arms") == list(ARMS),
        "breadth_intervention": int(registry.get("breadth_target2_different_count", -1)) == 3,
        "direct_intervention": int(registry.get("direct_target2_different_count", -1)) == 3,
        "cross_intervention": int(registry.get("breadth_vs_direct_target2_different_count", -1)) == 2,
        "classifier": registry.get("classifier_version") == CLASSIFIER_VERSION,
        "zero_api": all(int(registry.get(key, -1)) == 0 for key in (
            "phase_a_api_calls", "phase_a_validation_calls", "phase_a_test_calls"
        )),
        "no_test": registry.get("final_test_enabled") is False,
        "no_commit": registry.get("commit_enabled") is False,
        "no_mutation": registry.get("trajectory_mutation_enabled") is False,
    }
    errors.extend(key for key, passed in checks.items() if not passed)
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "checks": checks,
        "parents": len(cases),
        "cells": 15,
        "conceptual_branches": 30,
        "unique_branches": unique,
        "reused_branches": 30 - unique,
        "context_checks": context_checks,
        "breadth_target2_different_count": registry.get("breadth_target2_different_count"),
        "direct_target2_different_count": registry.get("direct_target2_different_count"),
        "breadth_vs_direct_target2_different_count": registry.get("breadth_vs_direct_target2_different_count"),
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    registry = (
        json.loads(args.registry.read_text(encoding="utf-8"))
        if args.registry else build_registry()
    )
    result = preflight(registry, args.scratch.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
