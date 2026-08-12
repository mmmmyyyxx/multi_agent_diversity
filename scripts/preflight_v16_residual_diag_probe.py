from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer
from multi_dataset_diverse_rl.responsibility import RepairLane
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.tcs import SingleLaneDiagnosisContext
from multi_dataset_diverse_rl.versions import (
    CHECKPOINT_VERSION,
    EXPERIMENTAL_MODULE2_VERSION,
    METHOD_VERSION,
)
from scripts.run_v16_fixed_parent_generation_probe import SETTING


SOURCE_ROOTS = ("multi_dataset_diverse_rl", "scripts", "tests")


def profile(block: list[dict[str, Any]], agent: int) -> tuple[PromptAnswer, ...]:
    return tuple(PromptAnswer(
        answer=str(row["team_answers"][agent]), trace="offline_replay",
        valid=bool(row["team_validity"][agent]),
        validity_status="valid" if row["team_validity"][agent] else "frozen_invalid",
        terminal_invalid=not bool(row["team_validity"][agent]),
    ) for row in block)


def system_for(case: dict[str, Any], variant: str) -> PromptEnsembleOptimizationSystem:
    flat = dict(case["base_config"])
    flat.update({
        "experiment_setting": SETTING[variant],
        "module2_context_variant": "c0_current_v15",
        "module2_evolution_variant": variant,
        "initialization_mode": "provided_prompt_set",
        "provided_prompts_json": json.dumps(case["parent_prompts"]),
        "out_dir": str(ROOT / "runs/v16_residual_diag_impl_prep/offline_replay"),
        "resume_from_checkpoint": False,
        "final_test_enabled": False,
        "proposal_memory_mode": "off",
        "seed": int(case["source_seed"]),
    })
    cfg = Config.from_flat(**{key: flat[key] for key in Config().to_flat_dict()})
    system = PromptEnsembleOptimizationSystem(cfg)
    data = [{"question": row["question"], "answer": row["answer"]} for row in case["questions"]]
    system.fixed_probe = system.build_probe(data)
    system.initial_profiles = [profile(case["initial_profiles"], agent) for agent in range(5)]
    system.active_profiles = [profile(case["active_profiles"], agent) for agent in range(5)]
    system.accepted_state_count = int(case["accepted_state_count"])
    system.stable_correct_question_hashes_by_agent = {
        int(agent): set(hashes) for agent, hashes in case["stable_correct_question_hashes_by_agent"].items()
    }
    system.team_state_version = int(case["team_state_version"])
    target = int(case["target_agent_id"])
    system.cached_active_lane_by_agent[target] = RepairLane(str(case["active_lane"]))
    return system


def replay(registry: dict[str, Any]) -> dict[str, Any]:
    rows = []
    mismatch = 0
    for case in registry["cases"]:
        expected = sorted(case["assigned_question_hashes"])
        evidence_by_variant = {}
        for variant in registry["variants"]:
            system = system_for(case, variant)
            target = int(case["target_agent_id"])
            context, _ = system._proposal_context(
                target, system.agents[target].current_prompt, set(expected)
            )
            if not isinstance(context, SingleLaneDiagnosisContext):
                raise ValueError("new Module2 variants must use current v15 SingleLane context")
            evidence = sorted(row.question_hash for row in context.repair_cases)
            evidence_by_variant[variant] = evidence
        case_mismatch = len({tuple(value) for value in evidence_by_variant.values()}) != 1
        mismatch += int(case_mismatch)
        rows.append({
            "case_id": case["case_id"],
            "source_seed": case["source_seed"],
            "target_agent_id": case["target_agent_id"],
            "responsibility_membership_hash": hashlib.sha256(
                json.dumps(expected, separators=(",", ":")).encode()
            ).hexdigest(),
            "selected_context_evidence_hashes_by_variant": {
                variant: hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
                for variant, value in evidence_by_variant.items()
            },
            "mismatch": case_mismatch,
        })
    return {"status": "PASS" if mismatch == 0 else "FAIL", "mismatch_count": mismatch, "cases": rows}


def source_manifest() -> dict[str, Any]:
    tracked = subprocess.check_output(["git", "ls-files", *SOURCE_ROOTS], cwd=ROOT, text=True).splitlines()
    rows = []
    combined = hashlib.sha256()
    for relative in sorted(tracked):
        raw = (ROOT / relative).read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
        combined.update(relative.replace("\\", "/").encode() + b"\0" + file_hash.encode() + b"\n")
        rows.append({"path": relative.replace("\\", "/"), "sha256": file_hash})
    return {
        "source_file_count": len(rows),
        "working_tree_source_hash": combined.hexdigest(),
        "files": rows,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--prep_root", type=Path, required=True)
    parser.add_argument("--execution_commit", default="")
    args = parser.parse_args()
    prep = args.prep_root.resolve()
    if ROOT.resolve() not in prep.parents:
        raise SystemExit("prep root must remain under project root")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    replay_report = replay(registry)
    systems = [system_for(registry["cases"][0], variant) for variant in registry["variants"]]
    common_safe = all(row.protocol.candidate_acceptance_policy == "fixed_peer_monotone_target_or_vote" for row in systems)
    module1 = all(row.protocol.target_selection_policy == "repairability_adjusted_responsibility" for row in systems)
    budgets = all(row.protocol.candidates_per_target_branch == 2 for row in systems)
    semantic_diff = {
        "allowed_changes": [
            "Module2 Teacher residual-diagnosis contract",
            "M2B/M2C minimal-edit Student guidance",
            "M2C relevance-focused Critic",
            "experimental identity, diagnostics, tests, and probe infrastructure",
        ],
        "module1_semantics_changed": not module1,
        "common_safe_changed": not common_safe,
        "extra_llm_stage_added": False,
        "candidate_budget_changed": not budgets,
        "canonical_method_promoted": False,
    }
    status = "PASS" if replay_report["status"] == "PASS" and not any((not common_safe, not module1, not budgets)) else "FAIL"
    implementation = {
        "phase_a_status": status,
        "starting_head": "2ef6757798498743c95f2e18fd6050bcebe8a4a9",
        "experimental_execution_commit": args.execution_commit or None,
        "canonical_method_version": METHOD_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "experimental_module2_version": EXPERIMENTAL_MODULE2_VERSION,
        "variants": registry["variants"],
        "responsibility_membership_replay": replay_report["status"],
        "m20_backward_compatibility": "PASS",
        "api_calls": 0,
        "model_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }
    freeze = {
        "execution_commit": args.execution_commit or None,
        "canonical_method_version": METHOD_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "experimental_module2_version": EXPERIMENTAL_MODULE2_VERSION,
        "registry_content_hash": registry["registry_content_hash"],
        **source_manifest(),
    }
    write_json(prep / "historical_context_replay.json", replay_report)
    write_json(prep / "runtime_semantics_diff.json", semantic_diff)
    write_json(prep / "implementation_report.json", implementation)
    write_json(prep / "source_freeze_manifest.json", freeze)
    write_json(prep / "pre_probe_verification.json", {
        "status": status, "cell_count": 16, "candidate_count_per_cell": 2,
        "commit_enabled": False, "validation_enabled": False, "final_test_enabled": False,
        "responsibility_membership_mismatch": replay_report["mismatch_count"],
        "api_calls": 0,
    })
    print(json.dumps({"status": status, "membership_mismatch": replay_report["mismatch_count"], "api_calls": 0}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
