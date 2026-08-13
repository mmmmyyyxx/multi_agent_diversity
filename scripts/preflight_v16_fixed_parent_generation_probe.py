from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.responsibility import RepairLane
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.tcs import ExperimentalModule2DiagnosisContext, SingleLaneDiagnosisContext
from scripts.run_v16_fixed_parent_generation_probe import SETTING, profile


def preflight(registry: dict) -> dict:
    errors: list[str] = []
    cell_rows = []
    for case in registry["cases"]:
        memberships = {}
        for variant in registry["variants"]:
            flat = dict(registry["base_config"])
            flat.setdefault("module2_evolution_variant", "m20_current_v15")
            flat.update({
                "experiment_setting": SETTING[variant], "module2_context_variant": variant,
                "initialization_mode": "provided_prompt_set",
                "provided_prompts_json": json.dumps(registry["parent_prompts"]),
                "out_dir": str(ROOT / "runs" / "v16_fixed_parent_probe_preflight_only"),
                "shared_solver_cache_path": "", "resume_from_checkpoint": False,
                "final_test_enabled": False, "proposal_memory_mode": "off", "seed": 51,
            })
            defaults = Config().to_flat_dict()
            cfg = Config.from_flat(**{
                key: flat.get(key, default) for key, default in defaults.items()
            })
            system = PromptEnsembleOptimizationSystem(cfg)
            data = [{"question": row["question"], "answer": row["answer"]} for row in registry["questions"]]
            system.fixed_probe = system.build_probe(data)
            system.initial_profiles = [profile(registry["initial_profiles"], agent) for agent in range(5)]
            system.active_profiles = [profile(registry["active_profiles"], agent) for agent in range(5)]
            system.accepted_state_count = int(registry["accepted_state_count"])
            system.stable_correct_question_hashes_by_agent = {
                int(agent): set(hashes)
                for agent, hashes in registry["stable_correct_question_hashes_by_agent"].items()
            }
            system.team_state_version = int(case["team_state_version"])
            target = int(case["target_agent_id"])
            system.cached_active_lane_by_agent[target] = RepairLane(str(case["active_lane"]))
            context, _ = system._proposal_context(
                target, system.agents[target].current_prompt, set(case["assigned_question_hashes"])
            )
            if system.team_prompt_state_hash() != case["parent_team_hash"]:
                errors.append(f"parent_hash:{case['case_id']}:{variant}")
            if variant == "c0_current_v15":
                if not isinstance(context, SingleLaneDiagnosisContext):
                    errors.append(f"c0_context_type:{case['case_id']}")
            else:
                if not isinstance(context, ExperimentalModule2DiagnosisContext):
                    errors.append(f"experimental_context_type:{case['case_id']}:{variant}")
                else:
                    memberships[variant] = {
                        "repair": [row.question_hash for row in context.repair_cases],
                        "preservation": [row.question_hash for row in context.preservation_cases],
                    }
            cell_rows.append({"case_id": case["case_id"], "variant": variant, "context_type": type(context).__name__})
        if memberships.get("c2_boundary_plus_preservation") != memberships.get("c3_coalition_aware_preservation"):
            errors.append(f"c2_c3_membership:{case['case_id']}")
    return {
        "preflight_version": "v16_fixed_parent_generation_probe_preflight_v1",
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "api_calls": 0, "case_count": len(registry["cases"]), "cell_count": len(cell_rows),
        "commit_enabled": False, "validation_enabled": False, "final_test_enabled": False,
        "cells": cell_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = preflight(json.loads(args.registry.read_text(encoding="utf-8")))
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
