from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer
from multi_dataset_diverse_rl.responsibility import RepairLane
from multi_dataset_diverse_rl.system import CandidateFunnel, PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.tcs import PreviousUpdateOutcome


SETTING = {
    "c0_current_v15": "experimental_v16_c0_current_v15",
    "c2_boundary_plus_preservation": "experimental_v16_c2_boundary_plus_preservation",
    "c3_coalition_aware_preservation": "experimental_v16_c3_coalition_aware_preservation",
    "m20_current_v15": "experimental_v16_m20_current_v15",
    "m2a_residual_diagnosis": "experimental_v16_m2a_residual_diagnosis",
    "m2b_diagnosis_minimal_edit": "experimental_v16_m2b_diagnosis_minimal_edit",
    "m2c_diagnosis_minimal_edit_relevance_critic": "experimental_v16_m2c_diagnosis_minimal_edit_relevance_critic",
    "m2d_raw_responsibility_minimal_edit": "experimental_v16_m2d_raw_responsibility_minimal_edit",
}


def require_fresh_cell_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"fixed-parent cell path must be fresh: {path}")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        temporary.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def serialize_candidate(row: Any) -> dict[str, Any]:
    evaluation = row.final_evaluation or row.stage_a_evaluation
    if evaluation is None:
        raise ValueError("evaluated candidate has no persisted evaluation")
    return {
        "prompt_hash": row.prompt_hash,
        "prompt": row.prompt,
        "evaluation": asdict(evaluation),
        "constraint": asdict(row.constraint) if row.constraint else None,
        "module2_diagnostics": dict(row.module2_diagnostics),
    }


def public_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_hash": row["prompt_hash"],
        "constraint": row["constraint"],
        "module2_diagnostics": row["module2_diagnostics"],
    }


def sanitized_branch_diagnostics(system: PromptEnsembleOptimizationSystem, funnel: CandidateFunnel) -> list[dict[str, Any]]:
    rows = [dict(row) for row in system.residual_diagnosis_branch_diagnostics]
    if not rows:
        return rows
    teacher_rows = [row for row in system.tcs_rounds if row.get("role") == "teacher" and row.get("schema_valid")]
    critic_rows = [row for row in system.tcs_rounds if row.get("role") == "critic"]
    last_plan = teacher_rows[-1].get("repair_plan") if teacher_rows else None
    diagnosis_hash = ""
    evidence_count = edit_plan_count = 0
    peer_contrast_present = False
    if isinstance(last_plan, dict) and last_plan.get("diagnosis_primary_failure_mode"):
        diagnosis_hash = hashlib.sha256(
            str(last_plan["diagnosis_primary_failure_mode"]).encode("utf-8")
        ).hexdigest()
        evidence_count = len(last_plan.get("diagnosis_evidence_patterns", []))
        edit_plan_count = len(last_plan.get("edit_plan", []))
        peer_contrast_present = bool(last_plan.get("diagnosis_peer_contrast"))
    reason_counts: dict[str, int] = {}
    for row in critic_rows:
        for reason in row.get("failed_checks", []):
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    rows[-1].update({
        "diagnosis_primary_failure_mode_hash": diagnosis_hash,
        "diagnosis_evidence_count": evidence_count,
        "diagnosis_peer_contrast_present": peer_contrast_present,
        "diagnosis_edit_plan_count": edit_plan_count,
        "critic_calls": int(funnel.critic_calls),
        "critic_semantic_rejections": int(funnel.critic_semantic_rejections),
        "critic_rejection_reason_counts": reason_counts,
        "critic_exhausted": bool(funnel.terminal_failure_role == "critic"),
        "student_reached": bool(funnel.student_calls),
        "raw_candidates": int(funnel.raw_candidate_count),
        "valid_candidates": int(funnel.valid_candidate_count),
    })
    return rows


def profile(block: list[dict[str, Any]], agent: int) -> tuple[PromptAnswer, ...]:
    return tuple(PromptAnswer(
        answer=str(row["team_answers"][agent]), trace="frozen_historical_profile",
        valid=bool(row["team_validity"][agent]), validity_status="valid" if row["team_validity"][agent] else "frozen_invalid",
        terminal_invalid=not bool(row["team_validity"][agent]),
    ) for row in block)


def state_hash(system: PromptEnsembleOptimizationSystem) -> str:
    value = {
        "prompts": [agent.current_prompt for agent in system.agents],
        "profiles": [[asdict(answer) for answer in row] for row in system.active_profiles],
        "team_state_version": system.team_state_version,
        "accepted_state_count": system.accepted_state_count,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


async def run_cell(registry: dict[str, Any], case: dict[str, Any], variant: str, out_dir: Path, cache: Path) -> dict[str, Any]:
    require_fresh_cell_path(out_dir)
    flat = dict(case["base_config"])
    is_evolution_variant = variant.startswith("m2")
    flat["module2_evolution_variant"] = variant if is_evolution_variant else "m20_current_v15"
    flat.update({
        "experiment_setting": SETTING[variant],
        "module2_context_variant": "c0_current_v15" if is_evolution_variant else variant,
        "initialization_mode": "provided_prompt_set",
        "provided_prompts_json": json.dumps(case["parent_prompts"]),
        "out_dir": str(out_dir), "shared_solver_cache_path": str(cache),
        "resume_from_checkpoint": False, "final_test_enabled": False,
        "proposal_memory_mode": "off", "seed": int(case.get("source_seed", 51)),
    })
    cfg = Config.from_flat(**{key: flat[key] for key in Config().to_flat_dict()})
    system = PromptEnsembleOptimizationSystem(cfg)
    atomic_write_json(out_dir / "cell_status.json", {
        "status": "started", "case_id": case["case_id"], "variant": variant,
        "commit_enabled": False, "validation_enabled": False,
        "final_test_enabled": False,
    })
    data = [{"question": row["question"], "answer": row["answer"]} for row in case["questions"]]
    system.fixed_probe = system.build_probe(data)
    system.initial_profiles = [profile(case["initial_profiles"], agent) for agent in range(5)]
    system.active_profiles = [profile(case["active_profiles"], agent) for agent in range(5)]
    system.accepted_state_count = int(case["accepted_state_count"])
    system.stable_correct_question_hashes_by_agent = {
        int(agent): set(hashes) for agent, hashes in case["stable_correct_question_hashes_by_agent"].items()
    }
    system.team_state_version = int(case["team_state_version"])
    for agent, outcome in case.get("previous_update_outcome_by_agent", {}).items():
        system.previous_update_outcomes[int(agent)] = PreviousUpdateOutcome(
            **{**outcome, "rejection_reasons": tuple(outcome["rejection_reasons"])}
        )
    target = int(case["target_agent_id"])
    system.cached_active_lane_by_agent[target] = RepairLane(str(case["active_lane"]))
    assigned = set(map(str, case["assigned_question_hashes"]))
    before = state_hash(system)
    funnel = CandidateFunnel()
    try:
        candidates = await system.propose_candidates(
            target, assigned, funnel, int(case["source_update_index"])
        )
        winner = None
        incumbent = None
        evaluated = []
        if candidates:
            winner, incumbent, evaluated = await system.evaluate_candidates(
                target, candidates, assigned, funnel,
                int(case["source_update_index"]),
            )
        if funnel.terminal_failure_class in {
            "transport_failure", "persistence_failure"
        }:
            raise RuntimeError(
                f"probe infrastructure failure in {case['case_id']}/{variant}: "
                f"{funnel.terminal_failure_class}"
            )
        after = state_hash(system)
        if before != after:
            raise RuntimeError("fixed-parent probe mutated the parent team")
        serialized_candidates = [serialize_candidate(row) for row in evaluated]
        raw_result = {
            "result_version": "v16_fixed_parent_generation_probe_cell_v1",
            "case_id": case["case_id"], "variant": variant,
            "target_agent_id": target,
            "parent_team_hash": case["parent_team_hash"],
            "active_lane": case["active_lane"],
            "assigned_question_hashes": sorted(assigned),
            "generated_candidate_count": len(candidates),
            "evaluated_candidate_count": len(evaluated),
            "funnel": asdict(funnel),
            "incumbent": asdict(incumbent) if incumbent is not None else None,
            "candidates": serialized_candidates,
            "winner_prompt_hash_diagnostic_only": (
                winner.prompt_hash if winner else ""
            ),
            "commit_performed": False,
            "parent_state_hash_before": before,
            "parent_state_hash_after": after,
            "validation_calls": system.validation_evaluation_count,
            "test_calls": system.test_evaluation_count,
            "llm_call_count": len(system.llm.calls),
            "branch_diagnostics": sanitized_branch_diagnostics(system, funnel),
        }
        result = {
            **raw_result,
            "candidates": [public_candidate(row) for row in serialized_candidates],
        }
        atomic_write_json(out_dir / "cell_result_raw_local.json", raw_result)
        atomic_write_json(out_dir / "cell_result.json", result)
        atomic_write_json(out_dir / "cell_status.json", {
            "status": "complete", "case_id": case["case_id"],
            "variant": variant, "cell_result_present": True,
        })
        return result
    except Exception as exc:
        atomic_write_json(out_dir / "cell_status.json", {
            "status": "failed", "case_id": case["case_id"],
            "variant": variant, "failure_class": type(exc).__name__,
            "cell_result_present": (out_dir / "cell_result.json").exists(),
        })
        raise
    finally:
        atomic_write_jsonl(out_dir / "llm_calls.jsonl", list(system.llm.calls))


async def main_async(args: argparse.Namespace) -> None:
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    is_m2d_probe = registry.get("registry_version") == "v16_m2d_fixed_parent_registry_v1"
    authorized = (
        os.environ.get("M2D_FIXED_PARENT_PROBE_AUTHORIZED") == "1"
        if is_m2d_probe
        else (
            os.environ.get("M2_RESIDUAL_DIAG_PROBE_AUTHORIZED") == "1"
            or os.environ.get("V16_FIXED_PARENT_PROBE_AUTHORIZED") == "1"
        )
    )
    if not authorized:
        raise SystemExit("API execution blocked: explicit fixed-parent probe authorization is required")
    root = args.out_root.resolve()
    if ROOT.resolve() not in root.parents or root.exists():
        raise SystemExit("out_root must be a fresh project-local directory")
    cache = root / "_shared_solver_cache.sqlite"
    root.mkdir(parents=True)
    results = []
    for case in registry["cases"]:
        for variant in case["cell_order"]:
            results.append(await run_cell(registry, case, variant, root / case["case_id"] / variant, cache))
    atomic_write_json(root / "probe_summary.json", {
        "probe_version": "v16_fixed_parent_generation_probe_v1", "registry_hash": registry["registry_content_hash"],
        "cell_count": len(results), "commit_count": 0, "validation_calls": sum(x["validation_calls"] for x in results),
        "test_calls": sum(x["test_calls"] for x in results), "cells": results,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
