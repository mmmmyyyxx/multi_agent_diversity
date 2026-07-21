import asyncio
import hashlib
import json
import os
import random
import re
import subprocess
import time
import uuid
from collections import Counter
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from openai import AsyncOpenAI

from .answer_formats import canonical_answer as canonical_answer_format
from .answer_formats import extract_prediction as extract_prediction_format
from .answer_formats import match_answer as match_answer_format
from .config import Config
from .policy import (
    BEHAVIOR_CONTEXT_NAMES,
    CAPABILITY_RESIDUAL_FAMILY_NAMES,
    AgentState,
    BehaviorContext,
    BehaviorFingerprintEntry,
    BehaviorStateSummary,
    CapabilityResidualFamily,
    RejectedBehaviorSummary,
    empty_capability_profile,
    uniform_vote_context_profile,
)
from .tasks import TaskSpec, get_task_spec, infer_option_count
from .behavior_profiles import behavior_distance, build_prompt_static_profile, build_team_behavior_profiles
from .diagnostics.candidate_funnel import (
    candidate_funnel_identity,
    empty_candidate_channel_funnel,
    normalize_candidate_channel,
    record_candidate_classification,
    record_candidate_stage,
    record_funnel_event,
    restore_funnel_seen,
    serialize_funnel_seen,
    validate_candidate_channel_funnel,
)
from .lineage import update_lineage_state
from .mechanisms import mechanism_niche_key, normalize_mechanism_representation
from .metrics.vote_conversion import (
    question_vote_conversion_diagnostics,
    summarize_vote_conversion,
)
from .quality_diversity import (
    QUALITY_KEYS,
    enumerate_joint_teams,
    select_quality_diversity_archive,
    select_stable_joint_team,
    team_quality_metrics,
    team_prompt_hashes,
)
from .qd.quality_anchors import build_quality_anchor, update_quality_anchor_archive
from .search_archive import (
    candidate_quality_bucket,
    cheap_prescreen,
    mechanism_is_novel,
    refill_requirements,
    retained_archive_requirements,
    representative_requirements,
    search_space_requirements,
    select_joint_representatives,
    select_reproduction_parent,
    select_safe_archive,
)
from .rollout_diversity import (
    accuracy_rollout_team_key,
    candidate_reward as rollout_candidate_reward,
    candidate_transition_metrics,
    enumerate_rollout_teams,
    is_rollout_qd_method,
    is_vote_ready_rollout_method,
    quality_guard as rollout_quality_guard,
    rollout_distance,
    rollout_signature,
    rollout_team_key,
    rollout_quality_key,
    select_rollout_archive,
    select_rollout_representatives,
    vote_ready_candidate_key,
    wrong_diversity_is_useful,
)
from .state_conditioned import (
    STATE_CONDITIONED_CHECKPOINT_VERSION,
    candidate_row_state_fields,
    coverage_case_assignees,
    is_state_conditioned_method,
    paired_c0_metrics,
    select_state_conditioned_archive,
    select_state_conditioned_representatives,
    select_state_conditioned_team,
    state_conditioned_candidate_key,
    state_conditioned_transition_metrics,
    state_conditioned_validation_key,
    state_dataset_metrics,
    state_quality_guard,
    state_team_metrics,
)
from .utils import (
    canonical_aggregation_mode,
    compute_gold_vote_diagnostics,
    ensure_dir,
    extract_json_obj,
    infer_task_type,
    normalize_spaces,
    plurality_vote_with_diagnostics,
    set_seed,
)


PARETO_EPSILON = 1e-12
TCS_AUDIT_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("tcs_audit_context", default={})
EXPERIMENT_PROTOCOL_VERSION = "vote_oriented_v7_residual_specialization"
CHECKPOINT_VERSION = 6
PLURALITY_BOUNDARY_VERSION = "plurality_boundary_v1"
VOTE_CONTEXT_WEIGHTS = {
    BehaviorContext.TEAM_WRONG_PIVOTAL_FIX.value: 4.0,
    BehaviorContext.TARGET_CORRECT_PIVOTAL_HOLD.value: 4.0,
    BehaviorContext.TEAM_CORRECT_DOMINANT_WRONG_REDUNDANCY.value: 2.0,
    BehaviorContext.TEAM_WRONG_NONPIVOTAL.value: 1.0,
    BehaviorContext.TEAM_CORRECT_TARGET_WRONG_OTHER.value: 0.5,
    BehaviorContext.TARGET_CORRECT_ROBUST.value: 0.5,
    BehaviorContext.INVALID.value: 1.0,
}


def compute_candidate_metric_deltas(
    *,
    baseline_target_accuracy: float,
    candidate_target_accuracy: float,
    baseline_team_accuracy: float,
    candidate_team_accuracy: float,
    baseline_oracle_accuracy: float,
    candidate_oracle_accuracy: float,
    baseline_embedding_diversity: float,
    candidate_embedding_diversity: float,
    baseline_invalid_rate: float,
    candidate_invalid_rate: float,
) -> Dict[str, float]:
    """Build the one canonical set of baseline-relative candidate deltas."""
    return {
        "accuracy_delta": float(candidate_target_accuracy) - float(baseline_target_accuracy),
        "vote_delta": float(candidate_team_accuracy) - float(baseline_team_accuracy),
        "coverage_delta": float(candidate_oracle_accuracy) - float(baseline_oracle_accuracy),
        "diversity_delta": float(candidate_embedding_diversity) - float(baseline_embedding_diversity),
        "invalid_delta": float(candidate_invalid_rate) - float(baseline_invalid_rate),
    }


def tcs_metadata_applicable(candidate_metadata: Mapping[str, Any]) -> bool:
    return (
        str(candidate_metadata.get("optimizer_architecture", "") or "").lower() == "teacher_critic_student"
        and str(candidate_metadata.get("candidate_pool_source", "") or "") == "optimizer"
    )


def validate_tcs_candidate_metadata(candidate_metadata: Mapping[str, Any]) -> List[str]:
    """Return provenance errors for a real TCS optimizer candidate, if applicable."""
    if not tcs_metadata_applicable(candidate_metadata):
        return []
    errors: List[str] = []
    if not str(candidate_metadata.get("teacher_question", "") or "").strip():
        errors.append("missing_teacher_question")
    if int(candidate_metadata.get("teacher_critic_rounds", 0) or 0) < 1:
        errors.append("zero_teacher_critic_rounds")
    if str(candidate_metadata.get("candidate_source", "") or "") != "teacher_critic_student":
        errors.append("invalid_tcs_candidate_source")
    if not str(candidate_metadata.get("tcs_call_group_id", "") or "").strip():
        errors.append("missing_tcs_call_group_id")
    if "teacher_question_forced_best_score" not in candidate_metadata:
        errors.append("missing_forced_best_flag")
    approved = bool(candidate_metadata.get("teacher_question_approved", False))
    forced_best = bool(candidate_metadata.get("teacher_question_forced_best_score", False))
    if not approved and not forced_best:
        errors.append("unapproved_without_forced_best")
    if approved and forced_best:
        errors.append("approved_and_forced_best")
    rounds = int(candidate_metadata.get("teacher_critic_rounds", 0) or 0)
    forced_round = int(candidate_metadata.get("teacher_question_forced_best_round", 0) or 0)
    if forced_best and not (1 <= forced_round <= rounds):
        errors.append("invalid_forced_best_round")
    rewrite_count = int(candidate_metadata.get("teacher_rewrite_count", 0) or 0)
    if rewrite_count < 0 or rewrite_count > max(0, rounds - 1):
        errors.append("rewrite_count_exceeds_critic_rounds")
    raw_count = int(candidate_metadata.get("student_candidate_count_raw", 0) or 0)
    final_count = int(candidate_metadata.get("student_candidate_count_final", 0) or 0)
    if raw_count < 1:
        errors.append("missing_student_raw_count")
    if final_count < 1:
        errors.append("missing_student_final_count")
    if final_count > raw_count:
        errors.append("inconsistent_student_counts")
    return errors


def compute_oracle_coverage_transitions(
    baseline_correctness: Sequence[Sequence[bool]],
    candidate_correctness: Sequence[Sequence[bool]],
) -> Dict[str, float]:
    """Compute candidate-level oracle coverage gains and losses without new rollouts."""
    if len(baseline_correctness) != len(candidate_correctness):
        raise ValueError("baseline_correctness and candidate_correctness must have equal length")
    batch_size = len(baseline_correctness)
    baseline_covered = [any(bool(value) for value in row) for row in baseline_correctness]
    candidate_covered = [any(bool(value) for value in row) for row in candidate_correctness]
    gain_count = sum(int((not baseline) and candidate) for baseline, candidate in zip(baseline_covered, candidate_covered))
    loss_count = sum(int(baseline and (not candidate)) for baseline, candidate in zip(baseline_covered, candidate_covered))
    denominator = float(batch_size) if batch_size else 1.0
    gain_rate = float(gain_count) / denominator if batch_size else 0.0
    loss_rate = float(loss_count) / denominator if batch_size else 0.0
    return {
        "coverage_gain_count": int(gain_count),
        "coverage_gain_rate": float(gain_rate),
        "coverage_loss_count": int(loss_count),
        "coverage_loss_rate": float(loss_rate),
        "net_coverage_count": int(gain_count - loss_count),
        "net_coverage_delta": float(gain_rate - loss_rate),
        "baseline_oracle_accuracy": float(sum(baseline_covered) / denominator) if batch_size else 0.0,
        "candidate_oracle_accuracy": float(sum(candidate_covered) / denominator) if batch_size else 0.0,
    }


def compute_vote_transitions(
    baseline_vote_correct: Sequence[bool],
    candidate_vote_correct: Sequence[bool],
) -> Dict[str, float]:
    """Compute vote flips on the already-evaluated candidate batch."""
    if len(baseline_vote_correct) != len(candidate_vote_correct):
        raise ValueError("baseline_vote_correct and candidate_vote_correct must have equal length")
    batch_size = len(baseline_vote_correct)
    gains = sum(int((not bool(base)) and bool(candidate)) for base, candidate in zip(baseline_vote_correct, candidate_vote_correct))
    losses = sum(int(bool(base) and (not bool(candidate))) for base, candidate in zip(baseline_vote_correct, candidate_vote_correct))
    denominator = float(batch_size) if batch_size else 1.0
    gain_rate = float(gains) / denominator if batch_size else 0.0
    loss_rate = float(losses) / denominator if batch_size else 0.0
    return {
        "vote_gain_count": int(gains),
        "vote_gain_rate": gain_rate,
        "vote_loss_count": int(losses),
        "vote_loss_rate": loss_rate,
        "net_vote_count": int(gains - losses),
        "net_vote_delta": float(gain_rate - loss_rate),
    }


def compute_coverage_depth_transitions(
    baseline_correctness: Sequence[Sequence[bool]],
    candidate_correctness: Sequence[Sequence[bool]],
    max_depth: int = 5,
) -> Dict[str, float]:
    """Compute K>=depth gains/losses from paired, already-recorded rollouts."""
    if len(baseline_correctness) != len(candidate_correctness):
        raise ValueError("baseline_correctness and candidate_correctness must have equal length")
    if int(max_depth) < 1:
        raise ValueError("max_depth must be positive")
    size = len(baseline_correctness)
    denominator = float(size) if size else 1.0
    baseline_k = [sum(bool(value) for value in row) for row in baseline_correctness]
    candidate_k = [sum(bool(value) for value in row) for row in candidate_correctness]
    result: Dict[str, float] = {}
    for depth in range(1, int(max_depth) + 1):
        baseline_met = [value >= depth for value in baseline_k]
        candidate_met = [value >= depth for value in candidate_k]
        gains = sum(int(not before and after) for before, after in zip(baseline_met, candidate_met))
        losses = sum(int(before and not after) for before, after in zip(baseline_met, candidate_met))
        result.update(
            {
                f"baseline_coverage_depth_c{depth}": sum(baseline_met) / denominator if size else 0.0,
                f"candidate_coverage_depth_c{depth}": sum(candidate_met) / denominator if size else 0.0,
                f"depth{depth}_gain_count": int(gains),
                f"depth{depth}_gain_rate": gains / denominator if size else 0.0,
                f"depth{depth}_loss_count": int(losses),
                f"depth{depth}_loss_rate": losses / denominator if size else 0.0,
                f"depth{depth}_net_count": int(gains - losses),
                f"depth{depth}_net_delta": (gains - losses) / denominator if size else 0.0,
            }
        )
    return result


def competence_specialization_strength(bottom2_mean_acc: float, low: float = 0.55, high: float = 0.65) -> float:
    if float(high) <= float(low):
        raise ValueError("high must be greater than low")
    return float(np.clip((float(bottom2_mean_acc) - float(low)) / (float(high) - float(low)), 0.0, 1.0))


def normalize_mechanism_signature(steps: Sequence[Any]) -> List[str]:
    aliases = {
        "enumerate": "enumerate_candidates", "candidate": "enumerate_candidates",
        "constraint": "extract_constraints", "eliminate": "hard_elimination",
        "score": "weighted_scoring", "weight": "weighted_scoring",
        "compare": "pairwise_comparison", "counterfactual": "counterfactual_check",
        "timeline": "timeline_construction", "binding": "binding_resolution",
        "semantic": "semantic_role_check", "discourse": "discourse_distance_check",
        "contradiction": "contradiction_minimization", "verify": "final_consistency_check",
        "consistency": "final_consistency_check",
    }
    signature = []
    for step in steps or []:
        tokens = re.findall(r"[a-z0-9_]+", str(step).lower())
        operation = next((aliases[token] for token in tokens if token in aliases), "_".join(tokens[:5]))
        if operation and (not signature or signature[-1] != operation):
            signature.append(operation)
    return signature


def mechanism_signature_distance(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = list(left or []), list(right or [])
    if not a and not b:
        return 0.0
    common = sum(x == y for x, y in zip(a, b))
    return 1.0 - common / max(1, max(len(a), len(b)))


def competence_relative_specialization_strength(
    *,
    initial_metrics: Dict[str, Any],
    snapshot_metrics: Dict[str, Any],
    probe_size: int,
    current_strength: float,
    low_delta: float = 0.01,
    high_delta: float = 0.06,
    ema: float = 0.50,
    max_step: float = 0.35,
    monotonic: bool = True,
    mean_guard_epsilon: float = 0.01,
    c1_guard_epsilon: float = 0.01,
    c2_guard_epsilon: float = 0.01,
) -> Dict[str, Any]:
    """Compute the V8.1 schedule from paired static optimization probes."""
    size = max(1, int(probe_size))
    effective_low = max(float(low_delta), 1.0 / size)
    effective_high = max(float(high_delta), 4.0 / size, effective_low + 1e-8)
    initial_bottom2 = float(initial_metrics.get("bottom2_mean_acc", 0.0) or 0.0)
    snapshot_bottom2 = float(snapshot_metrics.get("bottom2_mean_acc", 0.0) or 0.0)
    initial_mean = float(initial_metrics.get("mean_individual_acc", 0.0) or 0.0)
    snapshot_mean = float(snapshot_metrics.get("mean_individual_acc", 0.0) or 0.0)
    initial_c1 = float(initial_metrics.get("coverage_depth_c1", 0.0) or 0.0)
    snapshot_c1 = float(snapshot_metrics.get("coverage_depth_c1", 0.0) or 0.0)
    initial_c2 = float(initial_metrics.get("coverage_depth_c2", 0.0) or 0.0)
    snapshot_c2 = float(snapshot_metrics.get("coverage_depth_c2", 0.0) or 0.0)
    bottom2_gain = snapshot_bottom2 - initial_bottom2
    raw_strength = float(np.clip(
        (bottom2_gain - effective_low) / (effective_high - effective_low), 0.0, 1.0
    ))
    mean_passed = snapshot_mean >= initial_mean - float(mean_guard_epsilon)
    c1_passed = snapshot_c1 >= initial_c1 - float(c1_guard_epsilon)
    c2_passed = snapshot_c2 >= initial_c2 - float(c2_guard_epsilon)
    reasons = []
    if not mean_passed:
        reasons.append("mean_regression")
    if not c1_passed:
        reasons.append("c1_regression")
    if not c2_passed:
        reasons.append("c2_regression")
    gated_raw = raw_strength if not reasons else 0.0
    previous = float(np.clip(current_strength, 0.0, 1.0))
    ema_target = (1.0 - float(ema)) * previous + float(ema) * gated_raw
    step_limited = min(previous + float(max_step), ema_target)
    next_strength = max(previous, step_limited) if bool(monotonic) else step_limited
    next_strength = float(np.clip(next_strength, 0.0, 1.0))
    return {
        "source": "optimization_static_probe",
        "probe_size": int(probe_size),
        "initial_bottom2_mean_acc": initial_bottom2,
        "snapshot_bottom2_mean_acc": snapshot_bottom2,
        "bottom2_gain": bottom2_gain,
        "initial_mean_individual_acc": initial_mean,
        "snapshot_mean_individual_acc": snapshot_mean,
        "initial_c1": initial_c1,
        "snapshot_c1": snapshot_c1,
        "initial_c2": initial_c2,
        "snapshot_c2": snapshot_c2,
        "effective_low_delta": effective_low,
        "effective_high_delta": effective_high,
        "mean_guard_passed": bool(mean_passed),
        "c1_guard_passed": bool(c1_passed),
        "c2_guard_passed": bool(c2_passed),
        "gate_failure_reasons": reasons,
        "raw_specialization_strength": raw_strength,
        "gated_raw_specialization_strength": gated_raw,
        "previous_specialization_strength": previous,
        "ema_target_specialization_strength": float(ema_target),
        "step_limited_specialization_strength": float(step_limited),
        "next_specialization_strength": next_strength,
        "ema": float(ema),
        "max_step": float(max_step),
        "monotonic": bool(monotonic),
    }


def _pareto_value(candidate: Dict[str, Any], key: str) -> float:
    metrics = candidate.get("metrics", {}) if isinstance(candidate.get("metrics", {}), dict) else {}
    return float(candidate.get(key, metrics.get(key, 0.0)) or 0.0)


def pareto_dominates(a: Dict[str, Any], b: Dict[str, Any], eps: float = PARETO_EPSILON) -> bool:
    """Return whether candidate a dominates b on vote gain/loss/target accuracy."""
    a_gain, b_gain = _pareto_value(a, "vote_gain_rate"), _pareto_value(b, "vote_gain_rate")
    a_loss, b_loss = _pareto_value(a, "vote_loss_rate"), _pareto_value(b, "vote_loss_rate")
    a_acc, b_acc = _pareto_value(a, "candidate_target_accuracy"), _pareto_value(b, "candidate_target_accuracy")
    no_worse = a_gain >= b_gain - eps and a_loss <= b_loss + eps and a_acc >= b_acc - eps
    strictly_better = a_gain > b_gain + eps or a_loss < b_loss - eps or a_acc > b_acc + eps
    return bool(no_worse and strictly_better)


def error_pareto_dominates(a: Dict[str, Any], b: Dict[str, Any], eps: float = PARETO_EPSILON) -> bool:
    """Four-objective v7 dominance; the legacy three-objective function stays unchanged."""
    if not pareto_dominates(a, b, eps) and not (
        abs(_pareto_value(a, "vote_gain_rate") - _pareto_value(b, "vote_gain_rate")) <= eps
        and abs(_pareto_value(a, "vote_loss_rate") - _pareto_value(b, "vote_loss_rate")) <= eps
        and abs(_pareto_value(a, "candidate_target_accuracy") - _pareto_value(b, "candidate_target_accuracy")) <= eps
    ):
        return False
    a_boundary = _pareto_value(a, "boundary_shared_error_net_gain")
    b_boundary = _pareto_value(b, "boundary_shared_error_net_gain")
    legacy_no_worse = (
        _pareto_value(a, "vote_gain_rate") >= _pareto_value(b, "vote_gain_rate") - eps
        and _pareto_value(a, "vote_loss_rate") <= _pareto_value(b, "vote_loss_rate") + eps
        and _pareto_value(a, "candidate_target_accuracy") >= _pareto_value(b, "candidate_target_accuracy") - eps
    )
    strictly_better = pareto_dominates(a, b, eps) or a_boundary > b_boundary + eps
    return bool(legacy_no_worse and a_boundary >= b_boundary - eps and strictly_better)


def competence_depth_dominates(a: Dict[str, Any], b: Dict[str, Any], eps: float = PARETO_EPSILON) -> bool:
    objectives_a = (
        _pareto_value(a, "vote_gain_rate"),
        -_pareto_value(a, "vote_loss_rate"),
        _pareto_value(a, "candidate_target_accuracy"),
        _pareto_value(a, "stage_aux_objective"),
    )
    objectives_b = (
        _pareto_value(b, "vote_gain_rate"),
        -_pareto_value(b, "vote_loss_rate"),
        _pareto_value(b, "candidate_target_accuracy"),
        _pareto_value(b, "stage_aux_objective"),
    )
    return all(x >= y - eps for x, y in zip(objectives_a, objectives_b)) and any(
        x > y + eps for x, y in zip(objectives_a, objectives_b)
    )


def competence_non_dominated_sort(candidates: Sequence[Dict[str, Any]]) -> List[List[int]]:
    remaining = set(range(len(candidates)))
    fronts: List[List[int]] = []
    while remaining:
        front = [
            index for index in remaining
            if not any(competence_depth_dominates(candidates[other], candidates[index]) for other in remaining if other != index)
        ]
        front.sort(key=lambda index: str(candidates[index].get("candidate_id", "")))
        fronts.append(front)
        remaining.difference_update(front)
    return fronts


def non_dominated_sort(
    candidates: Sequence[Dict[str, Any]],
    eps: float = PARETO_EPSILON,
    include_boundary_error: bool = False,
) -> List[List[int]]:
    """Deterministic non-dominated sorting; returned indices reference the input sequence."""
    remaining = set(range(len(candidates)))
    fronts: List[List[int]] = []
    while remaining:
        front = [
            index
            for index in remaining
            if not any(
                (error_pareto_dominates if include_boundary_error else pareto_dominates)(
                    candidates[other], candidates[index], eps
                )
                for other in remaining if other != index
            )
        ]
        front.sort(key=lambda index: str(candidates[index].get("candidate_id", "")))
        fronts.append(front)
        remaining.difference_update(front)
    return fronts


def compute_crowding_distances(
    candidates: Sequence[Dict[str, Any]],
    front_indices: Sequence[int],
    include_boundary_error: bool = False,
    include_competence_depth: bool = False,
) -> Dict[int, float]:
    """Compute normalized NSGA-style crowding distances for one Pareto front."""
    distances = {index: 0.0 for index in front_indices}
    if len(front_indices) <= 2:
        return {index: float("inf") for index in front_indices}
    objectives = [
        lambda item: _pareto_value(item, "vote_gain_rate"),
        lambda item: -_pareto_value(item, "vote_loss_rate"),
        lambda item: _pareto_value(item, "candidate_target_accuracy"),
    ]
    if include_competence_depth:
        objectives.append(lambda item: _pareto_value(item, "stage_aux_objective"))
    elif include_boundary_error:
        objectives.append(lambda item: _pareto_value(item, "boundary_shared_error_net_gain"))
    for objective in objectives:
        ordered = sorted(front_indices, key=lambda index: (objective(candidates[index]), str(candidates[index].get("candidate_id", ""))))
        low, high = objective(candidates[ordered[0]]), objective(candidates[ordered[-1]])
        if abs(high - low) <= PARETO_EPSILON:
            continue
        distances[ordered[0]] = float("inf")
        distances[ordered[-1]] = float("inf")
        for position in range(1, len(ordered) - 1):
            index = ordered[position]
            if np.isinf(distances[index]):
                continue
            previous_value = objective(candidates[ordered[position - 1]])
            next_value = objective(candidates[ordered[position + 1]])
            distances[index] += (next_value - previous_value) / (high - low)
    return distances



__all__ = [name for name in globals() if not name.startswith('__')]
