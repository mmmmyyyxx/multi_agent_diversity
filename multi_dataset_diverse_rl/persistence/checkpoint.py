from __future__ import annotations

import base64
import json
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ..evaluation.fixed_probe import PromptAnswer
from ..evaluation.mutable_prompt_contract import validate_mutable_decision_procedure
from ..persistence.identity import validate_run_identity
from ..responsibility import (
    RepairLane,
    ResidualServiceAssignment,
    ResponsibilityState,
    compute_repair_eligibility_sets,
)
from ..proposal_memory import entry_from_dict, entry_to_dict
from ..system import METHOD_VERSION
from ..tcs import PreviousUpdateOutcome
from ..versions import (
    CANDIDATE_PROTOCOL_FILTER_VERSION,
    CHECKPOINT_VERSION,
    COALITION_CONTRIBUTION_VERSION,
    COMMON_UPDATE_POLICY_VERSION,
    DUAL_TARGET_SEARCH_VERSION,
    EXPERIMENT_MATRIX_VERSION,
    MINIMAL_EDIT_VERSION,
    MUTABLE_PROMPT_CONTRACT_VERSION,
    PROTOCOL_RESOLUTION_VERSION,
    REPAIRABILITY_VERSION,
    RCRU_VERSION,
    RESPONSIBILITY_UTILITY_VERSION,
    ROBUST_SUPPORT_VERSION,
    SERVICE_ROUTING_VERSION,
    STUDENT_PROMPT_CONTRACT_VERSION,
)


_LEGACY_FREEZE_STATE_FIELDS = {
    "consecutive_failed_updates_by_agent",
    "last_failed_portfolio_signature_by_agent",
    "frozen_by_agent",
    "frozen_portfolio_signature_by_agent",
    "frozen_residual_hashes_by_agent",
    "frozen_direct_fix_count_by_agent",
    "frozen_margin_gain_sum_by_agent",
    "other_accepted_updates_since_freeze_by_agent",
    "freeze_count_by_agent",
}


def _responsibility_state_payload(system) -> dict[str, Any]:
    payload = asdict(system.responsibility_state)
    if not system.protocol.repairability_freeze_enabled:
        for field in _LEGACY_FREEZE_STATE_FIELDS:
            payload.pop(field, None)
    return payload


def _random_state_payload() -> str:
    return base64.b64encode(pickle.dumps(random.getstate())).decode("ascii")


def _service_assignment_payload(
    value,
    *,
    include_legacy_freeze: bool,
) -> dict[str, Any]:
    payload = {
        **asdict(value),
        "repair_lane": value.repair_lane.value,
        "eligible_agent_ids": list(value.eligible_agent_ids),
        "active_eligible_agent_ids": list(value.active_eligible_agent_ids),
    }
    if not include_legacy_freeze:
        payload.pop("service_blocked_by_freeze", None)
    return payload


def build_checkpoint(
    system,
    *,
    epoch_index: int,
    update_index: int,
    training_state: Mapping[str, Any] | None = None,
    best_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    system.validate_active_prompts()
    if system.fixed_probe is None:
        raise RuntimeError("cannot checkpoint before fixed probe initialization")
    if system.run_identity is None:
        raise RuntimeError("cannot checkpoint without run identity")
    if training_state is None:
        training_state = dict(best_state or {})
    last_context = system.tcs_context_history[-1] if system.tcs_context_history else {}
    last_teacher = next(
        (
            row for row in reversed(system.tcs_rounds)
            if row.get("role") == "teacher" and row.get("schema_valid")
        ),
        {},
    )
    last_critic = next(
        (
            row for row in reversed(system.tcs_rounds)
            if row.get("role") == "critic" and row.get("schema_valid")
        ),
        {},
    )
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "method_version": METHOD_VERSION,
        "mutable_prompt_contract_version": MUTABLE_PROMPT_CONTRACT_VERSION,
        "student_prompt_contract_version": STUDENT_PROMPT_CONTRACT_VERSION,
        "candidate_protocol_filter_version": CANDIDATE_PROTOCOL_FILTER_VERSION,
        "canonical_experiment_setting": system.protocol.name,
        "module2_context_variant": system.protocol.module2_context_variant,
        "module2_evolution_variant": system.protocol.module2_evolution_variant,
        "compatibility_repair_enabled": system.protocol.compatibility_repair_enabled,
        "experiment_matrix_version": EXPERIMENT_MATRIX_VERSION,
        "protocol_resolution_version": PROTOCOL_RESOLUTION_VERSION,
        "common_update_policy_version": COMMON_UPDATE_POLICY_VERSION,
        "module_vector": system.protocol.module_vector,
        "resolved_candidate_acceptance_policy": (
            system.protocol.candidate_acceptance_policy
        ),
        "resolved_candidate_ranking_policy": (
            system.protocol.candidate_ranking_policy
        ),
        "resolved_stage_a_policy": system.protocol.stage_a_policy,
        "rcru_versions": {
            "rcru": RCRU_VERSION,
            "responsibility_utility": RESPONSIBILITY_UTILITY_VERSION,
            "coalition_contribution": COALITION_CONTRIBUTION_VERSION,
            "robust_support": ROBUST_SUPPORT_VERSION,
            "minimal_edit": MINIMAL_EDIT_VERSION,
        },
        "run_identity": system.run_identity.to_dict(),
        "probe_version": system.fixed_probe.version,
        "probe_hash": system.fixed_probe.probe_hash,
        "epoch_index": int(epoch_index),
        "update_index": int(update_index),
        "training_state": dict(training_state),
        "prompts": [agent.current_prompt for agent in system.agents],
        "previous_active_prompts": [agent.previous_active_prompt for agent in system.agents],
        "active_profiles": [[asdict(row) for row in profile] for profile in system.active_profiles],
        "initial_profiles": [[asdict(row) for row in profile] for profile in system.initial_profiles],
        "accepted_state_count": int(system.accepted_state_count),
        "stable_correct_question_hashes_by_agent": {
            str(agent_id): sorted(values)
            for agent_id, values in (
                system.stable_correct_question_hashes_by_agent.items()
            )
        },
        "member_gain_state": system.current_team_member_gain_state(),
        "team_state_version": system.team_state_version,
        "responsibility_state_version": system.responsibility_state_version,
        "responsibility_refresh_count": system.responsibility_refresh_count,
        "responsibility_state": _responsibility_state_payload(system),
        "repairability_version": REPAIRABILITY_VERSION,
        "dual_target_search_version": DUAL_TARGET_SEARCH_VERSION,
        "candidate_budget_contract": asdict(
            system.protocol.candidate_budget_contract
        ),
        "cached_responsibility_eligibility": {
            str(key): list(value) for key, value in system.cached_responsibility_eligibility.items()
        },
        "service_routing_version": SERVICE_ROUTING_VERSION,
        "cached_repair_lane_by_question": {
            key: value.value
            for key, value in system.cached_repair_lane_by_question.items()
        },
        "cached_service_assignments": {
            key: _service_assignment_payload(
                value,
                include_legacy_freeze=(
                    system.protocol.repairability_freeze_enabled
                ),
            )
            for key, value in system.cached_service_assignments.items()
        },
        "cached_service_portfolios": {
            str(agent_id): sorted(row.question_hash for row in rows)
            for agent_id, rows in system.cached_service_portfolios.items()
        },
        "cached_active_lane_by_agent": {
            str(agent_id): lane.value if lane is not None else None
            for agent_id, lane in system.cached_active_lane_by_agent.items()
        },
        "cached_active_residual_hashes": {
            str(agent_id): sorted(row.question_hash for row in rows)
            for agent_id, rows in (
                system.cached_active_responsibility_assignments.items()
            )
        },
        "previous_update_outcomes": {
            str(agent_id): asdict(row)
            for agent_id, row in system.previous_update_outcomes.items()
        },
        "completed_tcs_state": {
            "selected_pattern_ids": list(last_context.get("selected_pattern_ids", [])),
            "selected_case_ids": list(last_context.get("selected_case_ids", [])),
            "teacher_repair_plan": last_teacher.get("repair_plan"),
            "critic_decision": {
                "approved": last_critic.get("effective_approved"),
                "failed_checks": list(last_critic.get("failed_checks", [])),
                "risk_case_ids": list(last_critic.get("risk_case_ids", [])),
                "feedback": last_critic.get("feedback", ""),
            } if last_critic else None,
            "role_retry_state": dict(system.student_recovery_state),
        },
        "student_recovery_state": dict(system.student_recovery_state),
        "student_recovery_observations": list(
            system.student_recovery_observations
        ),
        "planned_update_count": int(system.planned_update_count),
        "completed_update_count": int(system.completed_update_count),
        "early_stop_reason": system.early_stop_reason,
        "training_completed": bool(system.training_completed),
        "final_state_selection": dict(system.final_state_selection),
        "training_dynamics": list(system.training_dynamics),
        "team_differentiation_trajectory": list(
            system.team_differentiation_trajectory
        ),
        "update_transition_decomposition": list(
            system.update_transition_decomposition
        ),
        "final_test_differentiation": dict(
            system.final_test_differentiation
        ),
        "test_evaluation_count": int(system.test_evaluation_count),
        "test_used_for_selection": bool(system.test_used_for_selection),
        "test_used_for_training": bool(system.test_used_for_training),
        "test_called_before_training_complete": bool(
            system.test_called_before_training_complete
        ),
        "selected_test_metrics": dict(system.selected_test_metrics),
        "agent_selection_counts": dict(system.agent_selection_counts),
        "selected_target_ids": list(system.selected_target_ids),
        "target_priority_audit": list(system.target_priority_audit),
        "repairability_adjusted_target_scores": list(
            system.repairability_adjusted_target_scores
        ),
        "dual_target_branch_decisions": list(
            system.dual_target_branch_decisions
        ),
        "dual_target_commit_decisions": list(
            system.dual_target_commit_decisions
        ),
        "repairability_failure_events": list(
            system.repairability_failure_events
        ),
        "repairability_reset_events": list(
            system.repairability_reset_events
        ),
        "branch_lifecycle_status": {
            "failure_count_by_agent": dict(
                system.responsibility_state.branch_failure_count_by_agent
            ),
            "attempt_count_by_agent": dict(
                system.responsibility_state.branch_attempt_count_by_agent
            ),
            "feasible_count_by_agent": dict(
                system.responsibility_state.branch_feasible_count_by_agent
            ),
            "repairability_state_team_hash": (
                system.responsibility_state.repairability_state_team_hash
            ),
            "repairability_reset_count": int(
                system.responsibility_state.repairability_reset_count
            ),
        },
        "service_routing_audit": list(system.service_routing_audit),
        "specialization_anchor_trajectory": list(
            system.specialization_anchor_trajectory
        ),
        "responsibility_portfolio_trajectory": list(
            system.responsibility_portfolio_trajectory
        ),
        "target_responsibility_context_alignment": list(
            system.target_responsibility_context_alignment
        ),
        "proposal_memory_run_id": system.proposal_memory_run_id,
        "proposal_memory_entries": {
            key: entry_to_dict(entry)
            for key, entry in system.proposal_memory_entries.items()
        },
        "proposal_memory_events": list(system.proposal_memory_events),
        "proposal_rotation_trajectory": list(system.proposal_rotation_trajectory),
        "history": list(system.history),
        "peer_state_history": list(system.peer_state_history),
        "responsibility_assignments": list(system.responsibility_assignments),
        "member_opportunities": list(system.member_opportunities),
        "g_transition_audit": list(system.g_transition_audit),
        "specialization_trajectory": list(system.specialization_trajectory),
        "candidate_decisions": list(system.candidate_decisions),
        "rcru_candidate_decisions": list(system.rcru_candidate_decisions),
        "tcs_context_history": list(system.tcs_context_history),
        "module2_context_diagnostics": list(
            system.module2_context_diagnostics
        ),
        "compatibility_repair_events": list(system.compatibility_repair_events),
        "tcs_rounds": list(system.tcs_rounds),
        "solver_invalid_outputs": list(system.solver_invalid_outputs),
        "solver_recovery_observations": list(system.solver_recovery_observations),
        "llm_calls": list(system.llm.calls),
        "fixed_probe": system.fixed_probe.to_dict(),
        "shared_solver_cache_audit": {
            "path": str(system.cfg.persistence.shared_solver_cache_path or ""),
            "ready_entries": (
                system.shared_solver_cache.ready_entry_count()
                if system.shared_solver_cache is not None
                else len(system.prompt_question_evaluator.cache)
            ),
            "content_hash": (
                system.shared_solver_cache.ready_content_hash()
                if system.shared_solver_cache is not None
                else ""
            ),
        },
        "random_state": _random_state_payload(),
    }


def validate_checkpoint(payload: Mapping[str, Any], system) -> None:
    if "checkpoint_version" not in payload or "method_version" not in payload or "run_identity" not in payload:
        raise ValueError("Legacy checkpoint lacks exact run identity and cannot be resumed")
    if int(payload["checkpoint_version"]) != CHECKPOINT_VERSION:
        raise ValueError(
            "checkpoint_version_mismatch: "
            f"expected={CHECKPOINT_VERSION}, "
            f"observed={payload['checkpoint_version']}"
        )
    if str(payload["method_version"]) != METHOD_VERSION:
        raise ValueError(
            "checkpoint_method_version_mismatch: "
            f"expected={METHOD_VERSION}, observed={payload['method_version']}"
        )
    required_member_state = {
        "training_state",
        "member_gain_state",
        "target_priority_audit",
        "responsibility_state",
        "cached_responsibility_eligibility",
        "team_state_version",
        "responsibility_state_version",
        "responsibility_refresh_count",
        "previous_update_outcomes",
        "completed_tcs_state",
        "solver_recovery_observations",
        "student_recovery_state",
        "student_recovery_observations",
        "planned_update_count",
        "completed_update_count",
        "early_stop_reason",
        "training_completed",
        "final_state_selection",
        "training_dynamics",
        "team_differentiation_trajectory",
        "update_transition_decomposition",
        "member_opportunities",
        "g_transition_audit",
        "specialization_trajectory",
        "final_test_differentiation",
        "test_evaluation_count",
        "test_used_for_selection",
        "test_used_for_training",
        "test_called_before_training_complete",
        "selected_test_metrics",
        "responsibility_portfolio_trajectory",
        "target_responsibility_context_alignment",
        "selected_target_ids",
        "repairability_adjusted_target_scores",
        "dual_target_branch_decisions",
        "dual_target_commit_decisions",
        "repairability_failure_events",
        "repairability_reset_events",
        "branch_lifecycle_status",
        "repairability_version",
        "dual_target_search_version",
        "candidate_budget_contract",
        "service_routing_version",
        "cached_repair_lane_by_question",
        "cached_service_assignments",
        "cached_service_portfolios",
        "cached_active_lane_by_agent",
        "cached_active_residual_hashes",
        "service_routing_audit",
        "specialization_anchor_trajectory",
        "proposal_memory_run_id",
        "proposal_memory_entries",
        "proposal_memory_events",
        "proposal_rotation_trajectory",
        "mutable_prompt_contract_version",
        "student_prompt_contract_version",
        "candidate_protocol_filter_version",
        "canonical_experiment_setting",
        "experiment_matrix_version",
        "protocol_resolution_version",
        "common_update_policy_version",
        "module_vector",
        "resolved_candidate_acceptance_policy",
        "resolved_candidate_ranking_policy",
        "resolved_stage_a_policy",
        "module2_context_variant",
        "accepted_state_count",
        "stable_correct_question_hashes_by_agent",
        "module2_context_diagnostics",
    }
    if not required_member_state <= set(payload):
        raise ValueError(f"Checkpoint is incompatible with {METHOD_VERSION}")
    expected_contract_versions = {
        "mutable_prompt_contract_version": MUTABLE_PROMPT_CONTRACT_VERSION,
        "student_prompt_contract_version": STUDENT_PROMPT_CONTRACT_VERSION,
        "candidate_protocol_filter_version": CANDIDATE_PROTOCOL_FILTER_VERSION,
        "experiment_matrix_version": EXPERIMENT_MATRIX_VERSION,
        "protocol_resolution_version": PROTOCOL_RESOLUTION_VERSION,
        "common_update_policy_version": COMMON_UPDATE_POLICY_VERSION,
        "repairability_version": REPAIRABILITY_VERSION,
        "dual_target_search_version": DUAL_TARGET_SEARCH_VERSION,
    }
    if any(
        str(payload[key]) != expected
        for key, expected in expected_contract_versions.items()
    ):
        raise ValueError(f"Checkpoint is incompatible with {METHOD_VERSION}")
    if str(payload.get("canonical_experiment_setting", system.protocol.name)) != (
        system.protocol.name
    ):
        raise ValueError("Checkpoint candidate protocol setting is incompatible")
    if str(payload["module2_context_variant"]) != (
        system.protocol.module2_context_variant
    ):
        raise ValueError("Checkpoint Module2 context variant is incompatible")
    if str(payload.get("module2_evolution_variant", "m20_current_v15")) != (
        system.protocol.module2_evolution_variant
    ):
        raise ValueError("Checkpoint Module2 evolution variant is incompatible")
    if bool(payload.get("compatibility_repair_enabled", False)) != (
        system.protocol.compatibility_repair_enabled
    ):
        raise ValueError("Checkpoint compatibility repair setting is incompatible")
    expected_acceptance = system.protocol.candidate_acceptance_policy
    expected_ranking = system.protocol.candidate_ranking_policy
    if payload.get("module_vector") != system.protocol.module_vector:
        raise ValueError("Checkpoint module vector is incompatible")
    if payload.get("candidate_budget_contract") != asdict(
        system.protocol.candidate_budget_contract
    ):
        raise ValueError("Checkpoint candidate budget contract is incompatible")
    if str(
        payload.get("resolved_candidate_acceptance_policy", expected_acceptance)
    ) != expected_acceptance or str(
        payload.get("resolved_candidate_ranking_policy", expected_ranking)
    ) != expected_ranking or str(
        payload.get("resolved_stage_a_policy", system.protocol.stage_a_policy)
    ) != system.protocol.stage_a_policy:
        raise ValueError("Checkpoint candidate decision policy is incompatible")
    if expected_acceptance == "responsibility_robust_contribution":
        expected_rcru_versions = {
            "rcru": RCRU_VERSION,
            "responsibility_utility": RESPONSIBILITY_UTILITY_VERSION,
            "coalition_contribution": COALITION_CONTRIBUTION_VERSION,
            "robust_support": ROBUST_SUPPORT_VERSION,
            "minimal_edit": MINIMAL_EDIT_VERSION,
        }
        if payload.get("rcru_versions") != expected_rcru_versions:
            raise ValueError("Checkpoint RCRU versions are incompatible")
    if system.run_identity is None:
        raise RuntimeError("run identity must be set before checkpoint validation")
    validate_run_identity(system.run_identity, payload["run_identity"])
    if system.fixed_probe is None:
        raise RuntimeError("fixed probe must exist before checkpoint restore")
    if str(payload["probe_version"]) != system.fixed_probe.version or str(payload["probe_hash"]) != system.fixed_probe.probe_hash:
        raise ValueError("Fixed probe cache version or hash mismatch. Start a new run.")


def restore_checkpoint(system, payload: Mapping[str, Any]) -> tuple[int, int, dict[str, Any]]:
    validate_checkpoint(payload, system)
    prompts = payload["prompts"]
    previous_prompts = payload["previous_active_prompts"]
    if len(prompts) != 5 or len(previous_prompts) != 5:
        raise ValueError("checkpoint must contain exactly five agent prompts")
    normalized_prompts = [str(prompt) for prompt in prompts]
    normalized_previous = [
        None if prompt is None else str(prompt) for prompt in previous_prompts
    ]
    for prompt in normalized_prompts:
        validate_mutable_decision_procedure(prompt)
    for prompt in normalized_previous:
        if prompt is not None:
            validate_mutable_decision_procedure(prompt)
    for agent, prompt, previous in zip(
        system.agents, normalized_prompts, normalized_previous, strict=True
    ):
        agent.current_prompt = prompt
        agent.previous_active_prompt = previous
    system.active_profiles = [
        tuple(PromptAnswer(**row) for row in profile) for profile in payload["active_profiles"]
    ]
    system.initial_profiles = [
        tuple(PromptAnswer(**row) for row in profile) for profile in payload["initial_profiles"]
    ]
    system.accepted_state_count = int(payload["accepted_state_count"])
    system.stable_correct_question_hashes_by_agent = {
        int(agent_id): set(map(str, values))
        for agent_id, values in payload[
            "stable_correct_question_hashes_by_agent"
        ].items()
    }
    if set(system.stable_correct_question_hashes_by_agent) != set(range(5)):
        raise ValueError("Checkpoint stable-correct state must contain five agents")
    if json.dumps(
        payload["member_gain_state"],
        sort_keys=True,
    ) != json.dumps(
        system.current_team_member_gain_state(),
        sort_keys=True,
    ):
        raise ValueError("Checkpoint member gain state does not match restored profiles")
    raw_state = dict(payload["responsibility_state"])
    if not system.protocol.repairability_freeze_enabled:
        if _LEGACY_FREEZE_STATE_FIELDS & set(raw_state):
            raise ValueError("v15 checkpoint contains legacy freeze state")
        defaults = ResponsibilityState()
        for field in _LEGACY_FREEZE_STATE_FIELDS:
            raw_state[field] = getattr(defaults, field)
    for field in (
        "updates_since_selected_by_agent",
        "accepted_updates_by_agent",
        "target_attempt_count_by_agent",
        "consecutive_failed_updates_by_agent",
        "frozen_direct_fix_count_by_agent",
        "frozen_margin_gain_sum_by_agent",
        "other_accepted_updates_since_freeze_by_agent",
        "freeze_count_by_agent",
        "branch_failure_count_by_agent",
        "branch_attempt_count_by_agent",
        "branch_feasible_count_by_agent",
    ):
        raw_state[field] = {int(key): int(value) for key, value in raw_state[field].items()}
    for field in (
        "last_failed_portfolio_signature_by_agent",
        "frozen_portfolio_signature_by_agent",
    ):
        raw_state[field] = {
            int(key): str(value) for key, value in raw_state[field].items()
        }
    raw_state["frozen_by_agent"] = {
        int(key): bool(value) for key, value in raw_state["frozen_by_agent"].items()
    }
    raw_state["frozen_residual_hashes_by_agent"] = {
        int(key): tuple(map(str, value))
        for key, value in raw_state["frozen_residual_hashes_by_agent"].items()
    }
    raw_state["seeded_rank_by_agent"] = {
        int(key): str(value) for key, value in raw_state["seeded_rank_by_agent"].items()
    }
    raw_state["eligible_agents_by_question"] = {
        str(key): tuple(int(agent) for agent in value)
        for key, value in raw_state["eligible_agents_by_question"].items()
    }
    raw_state["specialization_anchor_by_agent"] = {
        int(key): (RepairLane(value) if value is not None else None)
        for key, value in raw_state[
            "specialization_anchor_by_agent"
        ].items()
    }
    system.responsibility_state = ResponsibilityState(**raw_state)
    lifecycle = dict(payload["branch_lifecycle_status"])
    expected_lifecycle = {
        "failure_count_by_agent": dict(
            system.responsibility_state.branch_failure_count_by_agent
        ),
        "attempt_count_by_agent": dict(
            system.responsibility_state.branch_attempt_count_by_agent
        ),
        "feasible_count_by_agent": dict(
            system.responsibility_state.branch_feasible_count_by_agent
        ),
        "repairability_state_team_hash": (
            system.responsibility_state.repairability_state_team_hash
        ),
        "repairability_reset_count": int(
            system.responsibility_state.repairability_reset_count
        ),
    }
    normalized_lifecycle = {
        **lifecycle,
        "failure_count_by_agent": {
            int(key): int(value)
            for key, value in lifecycle["failure_count_by_agent"].items()
        },
        "attempt_count_by_agent": {
            int(key): int(value)
            for key, value in lifecycle["attempt_count_by_agent"].items()
        },
        "feasible_count_by_agent": {
            int(key): int(value)
            for key, value in lifecycle["feasible_count_by_agent"].items()
        },
        "repairability_reset_count": int(
            lifecycle["repairability_reset_count"]
        ),
    }
    if normalized_lifecycle != expected_lifecycle:
        raise ValueError("Checkpoint branch lifecycle state is inconsistent")
    system.team_state_version = int(payload["team_state_version"])
    system.responsibility_state_version = int(payload["responsibility_state_version"])
    system.responsibility_refresh_count = int(payload["responsibility_refresh_count"])
    system.cached_responsibility_eligibility = {
        str(key): tuple(int(agent) for agent in value)
        for key, value in payload["cached_responsibility_eligibility"].items()
    }
    system.previous_update_outcomes = {
        int(key): PreviousUpdateOutcome(
            **{
                **row,
                "rejection_reasons": tuple(row.get("rejection_reasons", ())),
            }
        )
        for key, row in payload["previous_update_outcomes"].items()
    }
    system.student_recovery_state = dict(payload["student_recovery_state"])
    system.student_recovery_observations = list(
        payload["student_recovery_observations"]
    )
    system.planned_update_count = int(payload["planned_update_count"])
    system.completed_update_count = int(payload["completed_update_count"])
    system.early_stop_reason = str(payload["early_stop_reason"])
    system.training_completed = bool(payload["training_completed"])
    system.final_state_selection = dict(payload["final_state_selection"])
    system.training_dynamics = list(payload["training_dynamics"])
    system.team_differentiation_trajectory = list(
        payload["team_differentiation_trajectory"]
    )
    system.update_transition_decomposition = list(
        payload["update_transition_decomposition"]
    )
    system.final_test_differentiation = dict(
        payload["final_test_differentiation"]
    )
    system.test_evaluation_count = int(payload["test_evaluation_count"])
    system.test_used_for_selection = bool(payload["test_used_for_selection"])
    system.test_used_for_training = bool(payload["test_used_for_training"])
    system.test_called_before_training_complete = bool(
        payload["test_called_before_training_complete"]
    )
    system.selected_test_metrics = dict(payload["selected_test_metrics"])
    system.agent_selection_counts = {
        int(key): int(value) for key, value in payload["agent_selection_counts"].items()
    }
    system.target_priority_audit = list(payload["target_priority_audit"])
    system.selected_target_ids = [
        int(agent_id) for agent_id in payload["selected_target_ids"]
    ]
    system.repairability_adjusted_target_scores = list(
        payload["repairability_adjusted_target_scores"]
    )
    system.dual_target_branch_decisions = list(
        payload["dual_target_branch_decisions"]
    )
    system.dual_target_commit_decisions = list(
        payload["dual_target_commit_decisions"]
    )
    system.repairability_failure_events = list(
        payload["repairability_failure_events"]
    )
    system.repairability_reset_events = list(
        payload["repairability_reset_events"]
    )
    system.repairability_freeze_events = []
    system.repairability_unfreeze_events = []
    system.service_routing_audit = list(payload["service_routing_audit"])
    system.specialization_anchor_trajectory = list(
        payload["specialization_anchor_trajectory"]
    )
    system.responsibility_portfolio_trajectory = list(
        payload["responsibility_portfolio_trajectory"]
    )
    system.target_responsibility_context_alignment = list(
        payload["target_responsibility_context_alignment"]
    )
    if str(payload["proposal_memory_run_id"]) != system.proposal_memory_run_id:
        raise ValueError("Checkpoint proposal memory run identity mismatch")
    system.proposal_memory_entries = {
        str(key): entry_from_dict(value)
        for key, value in payload["proposal_memory_entries"].items()
    }
    system.proposal_memory_events = list(payload["proposal_memory_events"])
    system.proposal_rotation_trajectory = list(payload["proposal_rotation_trajectory"])
    for name in (
        "history",
        "peer_state_history",
        "responsibility_assignments",
        "member_opportunities",
        "g_transition_audit",
        "specialization_trajectory",
        "candidate_decisions",
        "tcs_context_history",
        "module2_context_diagnostics",
        "tcs_rounds",
        "solver_invalid_outputs",
        "solver_recovery_observations",
    ):
        setattr(system, name, list(payload[name]))
    system.rcru_candidate_decisions = list(
        payload.get("rcru_candidate_decisions", [])
    )
    system.compatibility_repair_events = list(
        payload.get("compatibility_repair_events", [])
    )
    system._audited_invalid_keys = {
        (str(row["prompt_hash"]), str(row["question_hash"]))
        for row in system.solver_invalid_outputs
    }
    system.solver_recovery_observations = list(payload["solver_recovery_observations"])
    system._observed_solver_keys = {
        (str(row["prompt_hash"]), str(row["question_hash"]))
        for row in system.solver_recovery_observations
    }
    system.llm.calls = list(payload["llm_calls"])
    system.fixed_probe.restore(payload["fixed_probe"])
    states, _, opportunities = system.current_states_and_opportunities()
    recomputed_eligibility, recomputed_assignments, _ = (
        compute_repair_eligibility_sets(
            team_states={row.question_hash: row for row in states},
            opportunities=opportunities,
            state=system.responsibility_state,
        )
    )
    if recomputed_eligibility != system.cached_responsibility_eligibility:
        raise ValueError(
            "Checkpoint responsibility eligibility does not match "
            "the restored active team"
        )
    system.cached_responsibility_assignments = {
        agent_id: list(rows)
        for agent_id, rows in recomputed_assignments.items()
    }
    if str(payload["service_routing_version"]) != SERVICE_ROUTING_VERSION:
        raise ValueError("Checkpoint service routing version is incompatible")
    if system.protocol.service_routing_enabled:
        saved_assignments = {
            str(key): ResidualServiceAssignment(
                **{
                    **row,
                    "service_blocked_by_freeze": bool(
                        row.get("service_blocked_by_freeze", False)
                    ),
                    "repair_lane": RepairLane(row["repair_lane"]),
                    "eligible_agent_ids": tuple(row["eligible_agent_ids"]),
                    "active_eligible_agent_ids": tuple(
                        row["active_eligible_agent_ids"]
                    ),
                }
            )
            for key, row in payload["cached_service_assignments"].items()
        }
        for question_hash, assignment in saved_assignments.items():
            if assignment.eligible_agent_ids != recomputed_eligibility.get(
                question_hash, ()
            ):
                raise ValueError(
                    "Checkpoint service assignment violates restored eligibility"
                )
            if (
                assignment.service_agent_id is not None
                and assignment.service_agent_id
                not in assignment.eligible_agent_ids
            ):
                raise ValueError("Checkpoint service agent is not legally eligible")
        legal_rows = {
            agent_id: {row.question_hash: row for row in rows}
            for agent_id, rows in recomputed_assignments.items()
        }
        service_portfolios = {
            int(agent_id): []
            for agent_id in payload["cached_service_portfolios"]
        }
        for question_hash, assignment in saved_assignments.items():
            if assignment.service_agent_id is not None:
                service_portfolios.setdefault(assignment.service_agent_id, [])
                service_portfolios[assignment.service_agent_id].append(
                    legal_rows[assignment.service_agent_id][question_hash]
                )
        restored_service_hashes = {
            str(agent_id): sorted(row.question_hash for row in rows)
            for agent_id, rows in service_portfolios.items()
        }
        if restored_service_hashes != payload["cached_service_portfolios"]:
            raise ValueError("Checkpoint service portfolios are incompatible")
        active_hashes = {
            int(agent_id): tuple(map(str, hashes))
            for agent_id, hashes in payload[
                "cached_active_residual_hashes"
            ].items()
        }
        active_slices = {
            agent_id: [
                legal_rows[agent_id][question_hash]
                for question_hash in hashes
            ]
            for agent_id, hashes in active_hashes.items()
        }
        active_lanes = {
            int(agent_id): (RepairLane(value) if value is not None else None)
            for agent_id, value in payload[
                "cached_active_lane_by_agent"
            ].items()
        }
        for agent_id, rows in active_slices.items():
            lane = active_lanes[agent_id]
            if any(
                saved_assignments[row.question_hash].repair_lane != lane
                for row in rows
            ):
                raise ValueError("Checkpoint active slice mixes repair lanes")
        system.cached_service_assignments = saved_assignments
        system.cached_repair_lane_by_question = {
            key: RepairLane(value)
            for key, value in payload[
                "cached_repair_lane_by_question"
            ].items()
        }
        system.cached_service_portfolios = service_portfolios
        system.cached_active_lane_by_agent = active_lanes
        system.cached_active_responsibility_assignments = active_slices
    else:
        system.cached_service_assignments = {}
        system.cached_repair_lane_by_question = {}
        system.cached_service_portfolios = {}
        system.cached_active_lane_by_agent = {}
        system.cached_active_responsibility_assignments = {}
    system.cached_member_opportunities = dict(opportunities)
    random.setstate(pickle.loads(base64.b64decode(str(payload["random_state"]))))
    return (
        int(payload["epoch_index"]),
        int(payload["update_index"]),
        dict(payload["training_state"]),
    )


def load_checkpoint(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))
