from __future__ import annotations

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
from multi_dataset_diverse_rl.tcs import PreviousUpdateOutcome


G0 = "g0_fixed_target_generic"
M20 = "m20_current_v15"
VARIANTS = (G0, M20)
GENERATION_SETTING = {
    G0: "shared_generic_evolution",
    M20: "experimental_v16_m20_current_v15",
}
EVALUATION_SETTING = "experimental_v16_m20_current_v15"
AUTHORIZATION_ENV = "GENERIC_M20_FIXED_PARENT_PROBE_AUTHORIZED"
SOURCE_ROOTS = ("multi_dataset_diverse_rl", "scripts", "tests")
FROZEN_DEFINITION_SHA256 = {
    "DESIGN_SPEC.md": (
        "f669bd8ab36db3c20fc477047389883443fd9f477f60460f9510e72018f6350f"
    ),
    "analysis_spec.json": (
        "1975718907226277c68d4215b122519e782b4cdb4f4a1f8185f4084d0074e168"
    ),
    "probe_preregistration.json": (
        "6f57309c4465b64dd81cd2b9d9759bef0116f3ebf1a59a82dd7d37d7768db0b6"
    ),
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def responsibility_evidence_hash(hashes: set[str] | list[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(map(str, hashes)), separators=(",", ":")).encode()
    ).hexdigest()


def profile(
    block: list[dict[str, Any]], agent: int
) -> tuple[PromptAnswer, ...]:
    return tuple(
        PromptAnswer(
            answer=str(row["team_answers"][agent]),
            trace="frozen_historical_profile",
            valid=bool(row["team_validity"][agent]),
            validity_status=(
                "valid" if row["team_validity"][agent] else "frozen_invalid"
            ),
            terminal_invalid=not bool(row["team_validity"][agent]),
            response_hash=hashlib.sha256(
                (
                    f"frozen_historical_profile:{row['question_hash']}:"
                    f"{agent}:{row['team_answers'][agent]}:"
                    f"{int(bool(row['team_validity'][agent]))}"
                ).encode()
            ).hexdigest(),
            created_at=1.0,
        )
        for row in block
    )


def _flat_config(
    case: dict[str, Any],
    *,
    setting: str,
    out_dir: Path,
    cache_path: Path | str,
) -> dict[str, Any]:
    flat = dict(case["base_config"])
    flat.update({
        "experiment_setting": setting,
        "module2_context_variant": "c0_current_v15",
        "module2_evolution_variant": "m20_current_v15",
        "initialization_mode": "provided_prompt_set",
        "provided_prompts_json": json.dumps(case["parent_prompts"]),
        "out_dir": str(out_dir),
        "shared_solver_cache_path": str(cache_path),
        "resume_from_checkpoint": False,
        "final_test_enabled": False,
        "proposal_memory_mode": "off",
        "seed": int(case["source_seed"]),
        "agent_model": "qwen3-14b",
        "optimizer_model": "qwen3-14b",
        "evaluator_model": "qwen3-14b",
    })
    defaults = Config().to_flat_dict()
    return {key: flat[key] for key in defaults}


def system_for(
    case: dict[str, Any],
    *,
    setting: str,
    out_dir: Path,
    cache_path: Path | str,
) -> PromptEnsembleOptimizationSystem:
    cfg = Config.from_flat(
        **_flat_config(
            case,
            setting=setting,
            out_dir=out_dir,
            cache_path=cache_path,
        )
    )
    system = PromptEnsembleOptimizationSystem(cfg)
    data = [
        {"question": row["question"], "answer": row["answer"]}
        for row in case["questions"]
    ]
    system.fixed_probe = system.build_probe(data)
    system.initial_profiles = [
        profile(case["initial_profiles"], agent) for agent in range(5)
    ]
    system.active_profiles = [
        profile(case["active_profiles"], agent) for agent in range(5)
    ]
    system.accepted_state_count = int(case["accepted_state_count"])
    system.stable_correct_question_hashes_by_agent = {
        int(agent): set(map(str, hashes))
        for agent, hashes in case[
            "stable_correct_question_hashes_by_agent"
        ].items()
    }
    system.team_state_version = int(case["team_state_version"])
    for agent, outcome in case.get(
        "previous_update_outcome_by_agent", {}
    ).items():
        system.previous_update_outcomes[int(agent)] = PreviousUpdateOutcome(
            **{
                **outcome,
                "rejection_reasons": tuple(outcome["rejection_reasons"]),
            }
        )
    target = int(case["target_agent_id"])
    system.cached_active_lane_by_agent[target] = RepairLane(
        str(case["active_lane"])
    )
    return system


def generation_system(
    case: dict[str, Any],
    variant: str,
    *,
    out_dir: Path,
    cache_path: Path | str,
) -> PromptEnsembleOptimizationSystem:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported generic/M20 variant: {variant}")
    return system_for(
        case,
        setting=GENERATION_SETTING[variant],
        out_dir=out_dir,
        cache_path=cache_path,
    )


def evaluation_system(
    case: dict[str, Any],
    *,
    out_dir: Path,
    cache_path: Path | str,
) -> PromptEnsembleOptimizationSystem:
    return system_for(
        case,
        setting=EVALUATION_SETTING,
        out_dir=out_dir,
        cache_path=cache_path,
    )


def generation_hashes(variant: str, frozen: set[str]) -> set[str]:
    if variant == G0:
        return set()
    if variant == M20:
        return set(frozen)
    raise ValueError(f"unsupported generic/M20 variant: {variant}")


def state_payload(system: PromptEnsembleOptimizationSystem) -> dict[str, Any]:
    responsibility = system.responsibility_state
    return {
        "prompts": [agent.current_prompt for agent in system.agents],
        "profiles": [
            [asdict(answer) for answer in row]
            for row in system.active_profiles
        ],
        "initial_profiles": [
            [asdict(answer) for answer in row]
            for row in system.initial_profiles
        ],
        "team_state_version": system.team_state_version,
        "accepted_state_count": system.accepted_state_count,
        "anchors": {
            str(key): value.value if value is not None else None
            for key, value in system.responsibility_state.
            specialization_anchor_by_agent.items()
        },
        "active_lanes": {
            str(key): value.value if value is not None else None
            for key, value in system.cached_active_lane_by_agent.items()
        },
        "responsibility_state": asdict(responsibility),
        "cached_service_assignments": asdict(
            system.cached_service_assignments
        ) if hasattr(system.cached_service_assignments, "__dataclass_fields__") else {
            str(key): asdict(value)
            if hasattr(value, "__dataclass_fields__") else value
            for key, value in system.cached_service_assignments.items()
        },
        "cached_active_responsibility_assignments": {
            str(agent): [asdict(row) for row in rows]
            for agent, rows in system.cached_active_responsibility_assignments.items()
        },
        "cached_responsibility_assignments": {
            str(agent): [asdict(row) for row in rows]
            for agent, rows in system.cached_responsibility_assignments.items()
        },
        "cached_service_portfolios": {
            str(agent): [asdict(row) for row in rows]
            for agent, rows in system.cached_service_portfolios.items()
        },
        "cached_responsibility_eligibility": {
            str(question_hash): list(agent_ids)
            for question_hash, agent_ids in
            system.cached_responsibility_eligibility.items()
        },
        "cached_repair_lane_by_question": {
            str(question_hash): lane.value
            for question_hash, lane in
            system.cached_repair_lane_by_question.items()
        },
        "cached_member_opportunities": {
            str(question_hash): [asdict(row) for row in rows]
            for question_hash, rows in system.cached_member_opportunities.items()
        },
        "cached_active_lane_by_agent": {
            str(agent): lane.value if lane is not None else None
            for agent, lane in system.cached_active_lane_by_agent.items()
        },
        "previous_update_outcomes": {
            str(agent): asdict(row)
            for agent, row in system.previous_update_outcomes.items()
        },
        "selected_target_ids": list(system.selected_target_ids),
        "team_state_version": system.team_state_version,
        "responsibility_state_version": system.responsibility_state_version,
        "repairability_state_team_hash": (
            responsibility.repairability_state_team_hash
        ),
        "branch_failure_count_by_agent": dict(
            responsibility.branch_failure_count_by_agent
        ),
        "branch_attempt_count_by_agent": dict(
            responsibility.branch_attempt_count_by_agent
        ),
        "branch_feasible_count_by_agent": dict(
            responsibility.branch_feasible_count_by_agent
        ),
        "repairability_reset_count": responsibility.repairability_reset_count,
        "agent_selection_counts": dict(system.agent_selection_counts),
        "responsibility_refresh_count": system.responsibility_refresh_count,
        "proposal_memory_entries": {
            str(key): asdict(value)
            for key, value in system.proposal_memory_entries.items()
        },
        "proposal_memory_run_id": system.proposal_memory_run_id,
        "planned_update_count": system.planned_update_count,
        "completed_update_count": system.completed_update_count,
        "early_stop_reason": system.early_stop_reason,
        "training_completed": system.training_completed,
        "final_state_selection": system.final_state_selection,
        "validation_state_cache": system.validation_state_cache,
        "validation_reuse_count": system.validation_reuse_count,
        "compat_validation_selection_completed": (
            system._compat_validation_selection_completed
        ),
        "test_used_for_selection": system.test_used_for_selection,
        "test_used_for_training": system.test_used_for_training,
        "test_called_before_training_complete": (
            system.test_called_before_training_complete
        ),
        "selected_test_metrics": system.selected_test_metrics,
        "final_test_differentiation": system.final_test_differentiation,
        "optimizer_state_history_lengths": {
            "training_dynamics": len(system.training_dynamics),
            "team_differentiation_trajectory": len(
                system.team_differentiation_trajectory
            ),
            "update_transition_decomposition": len(
                system.update_transition_decomposition
            ),
            "target_priority_audit": len(system.target_priority_audit),
            "repairability_failure_events": len(
                system.repairability_failure_events
            ),
            "repairability_reset_events": len(system.repairability_reset_events),
            "dual_target_branch_decisions": len(
                system.dual_target_branch_decisions
            ),
            "dual_target_commit_decisions": len(
                system.dual_target_commit_decisions
            ),
            "service_routing_audit": len(system.service_routing_audit),
            "specialization_anchor_trajectory": len(
                system.specialization_anchor_trajectory
            ),
            "peer_state_history": len(system.peer_state_history),
            "responsibility_assignments": len(system.responsibility_assignments),
            "member_opportunities": len(system.member_opportunities),
            "g_transition_audit": len(system.g_transition_audit),
            "specialization_trajectory": len(system.specialization_trajectory),
            "candidate_decisions": len(system.candidate_decisions),
            "rcru_candidate_decisions": len(system.rcru_candidate_decisions),
            "repairability_adjusted_target_scores": len(
                system.repairability_adjusted_target_scores
            ),
            "responsibility_portfolio_trajectory": len(
                system.responsibility_portfolio_trajectory
            ),
            "target_responsibility_context_alignment": len(
                system.target_responsibility_context_alignment
            ),
            "proposal_memory_events": len(system.proposal_memory_events),
            "proposal_rotation_trajectory": len(
                system.proposal_rotation_trajectory
            ),
        },
        "validation_calls": system.validation_evaluation_count,
        "test_calls": system.test_evaluation_count,
    }


def state_hash(system: PromptEnsembleOptimizationSystem) -> str:
    return sha256_json(state_payload(system))


def diagnostic_payload(system: PromptEnsembleOptimizationSystem) -> dict[str, Any]:
    """Snapshot append-only/call-local observations excluded from parent state."""
    evaluator = system.prompt_question_evaluator
    shared_cache = evaluator.shared_cache
    return {
        "tcs_context_history": len(system.tcs_context_history),
        "module2_context_diagnostics": len(system.module2_context_diagnostics),
        "residual_diagnosis_branch_diagnostics": len(
            system.residual_diagnosis_branch_diagnostics
        ),
        "tcs_rounds": len(system.tcs_rounds),
        "student_recovery_observations": len(
            system.student_recovery_observations
        ),
        "student_recovery_state_hash": sha256_json(system.student_recovery_state),
        "solver_invalid_outputs": len(system.solver_invalid_outputs),
        "solver_recovery_observations": len(system.solver_recovery_observations),
        "observed_solver_keys": len(system._observed_solver_keys),
        "audited_invalid_keys": len(system._audited_invalid_keys),
        "proposal_memory_attempts": len(system._proposal_memory_attempts),
        "module2_context_sets": len(system._module2_context_sets),
        "solver_history": len(system.history),
        "llm_calls": len(system.llm.calls),
        "prompt_question_cache_entries": len(evaluator.cache),
        "prompt_question_cache_hits": int(evaluator.cache_hits),
        "prompt_question_cache_misses": int(evaluator.cache_misses),
        "prompt_question_inflight": len(evaluator.inflight),
        "shared_cache_hits": int(shared_cache.hits) if shared_cache else 0,
        "shared_cache_misses": int(shared_cache.misses) if shared_cache else 0,
        "shared_cache_waits": int(shared_cache.waits) if shared_cache else 0,
    }


def team_prompt_hash(system: PromptEnsembleOptimizationSystem) -> str:
    return system.team_prompt_state_hash()


def source_manifest() -> dict[str, Any]:
    tracked = subprocess.check_output(
        ["git", "ls-files", *SOURCE_ROOTS],
        cwd=ROOT,
        text=True,
    ).splitlines()
    combined = hashlib.sha256()
    rows = []
    for relative in sorted(tracked):
        path = ROOT / relative
        raw = path.read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
        normalized = relative.replace("\\", "/")
        combined.update(
            normalized.encode() + b"\0" + file_hash.encode() + b"\n"
        )
        rows.append({"path": normalized, "sha256": file_hash})
    return {
        "source_file_count": len(rows),
        "working_tree_source_hash": combined.hexdigest(),
        "files": rows,
    }


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def tracked_source_dirty() -> list[str]:
    lines = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    return sorted(line for line in lines if line.strip())


def project_local(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents
