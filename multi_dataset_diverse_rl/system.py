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
from .tasks import TaskSpec, get_task_spec
from .behavior_profiles import build_prompt_static_profile, build_team_behavior_profiles
from .lineage import update_lineage_state
from .mechanisms import mechanism_niche_key, normalize_mechanism_representation
from .quality_diversity import (
    QUALITY_KEYS,
    enumerate_joint_teams,
    select_quality_diversity_archive,
    select_stable_joint_team,
    team_quality_metrics,
)
from .search_archive import (
    candidate_quality_bucket,
    cheap_prescreen,
    mechanism_is_novel,
    refill_requirements,
    select_joint_representatives,
    select_reproduction_parent,
    select_safe_archive,
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
CHECKPOINT_VERSION = 4
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


class TraceBeamSearchSystem:
    GENERIC_DISTINCT_PROCEDURE = (
        "Use a distinct decision procedure: first state which reasoning route you will use, "
        "then approach the problem through boundary checks, reverse validation, or an alternative representation. "
        "If that procedure is not useful, fall back to direct reasoning with one explicit verification step."
    )

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.beam_refresh_each_epoch = bool(int(self.cfg.beam_refresh_each_epoch))
        self.cfg.transient_retry_forever = bool(int(self.cfg.transient_retry_forever))
        self.cfg.llm_call_logging = bool(int(self.cfg.llm_call_logging))
        self.cfg.invalid_binary = bool(int(self.cfg.invalid_binary))
        self.cfg.use_joint_trace_diversity_evaluator = bool(int(self.cfg.use_joint_trace_diversity_evaluator))
        self.cfg.candidate_reuse_recorded_rollouts = bool(int(getattr(self.cfg, "candidate_reuse_recorded_rollouts", 1)))
        self.cfg.solver_rollout_singleflight = bool(int(getattr(self.cfg, "solver_rollout_singleflight", 1)))
        self.cfg.candidate_eval_prompt_dedup = bool(int(getattr(self.cfg, "candidate_eval_prompt_dedup", 1)))
        self.cfg.candidate_eval_cache_logging = bool(int(getattr(self.cfg, "candidate_eval_cache_logging", 1)))
        self.cfg.use_baseline_relative_reward = bool(int(getattr(self.cfg, "use_baseline_relative_reward", 1)))
        for name in (
            "boundary_selector_enabled",
            "shared_error_metrics_enabled",
            "residual_specialization_enabled",
            "error_dependence_guard_enabled",
            "residual_cycle_guard_enabled",
            "mechanism_trust_region_enabled",
            "competence_depth_enabled",
            "competence_depth2_aux_enabled",
            "competence_progressive_residual_enabled",
            "competence_schedule_monotonic",
            "competence_depth1_candidate_guard_enabled",
        ):
            setattr(self.cfg, name, bool(int(getattr(self.cfg, name, 0))))
        self.cfg.behavior_cycle_guard_enabled = bool(int(getattr(self.cfg, "behavior_cycle_guard_enabled", 1)))
        self.cfg.prompt_trust_region_enabled = bool(int(getattr(self.cfg, "prompt_trust_region_enabled", 1)))
        self.task_spec = self._build_task_spec()

        self.homogeneity_window = max(1, int(self.cfg.update_every))
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        solver_key_env = str(self.cfg.solver_api_key_env or "").strip()
        solver_base_env = str(self.cfg.solver_base_url_env or "").strip()
        evaluator_key_env = str(self.cfg.evaluator_api_key_env or "").strip()
        evaluator_base_env = str(self.cfg.evaluator_base_url_env or "").strip()
        solver_key = os.getenv(solver_key_env) if solver_key_env else api_key
        solver_base = os.getenv(solver_base_env) if solver_base_env else base_url
        evaluator_key = os.getenv(evaluator_key_env) if evaluator_key_env else api_key
        evaluator_base = os.getenv(evaluator_base_env) if evaluator_base_env else base_url
        self.solver_client = AsyncOpenAI(api_key=solver_key or api_key, base_url=solver_base)
        self.evaluator_client = AsyncOpenAI(api_key=evaluator_key or api_key, base_url=evaluator_base)

        ensure_dir(self.cfg.out_dir)
        self.previous_execution_session_id = self._read_previous_execution_session_id()
        # A resumed process is a distinct provenance session, even for a repeated step.
        self.execution_session_id = uuid.uuid4().hex[:12]
        open(os.path.join(self.cfg.out_dir, "llm_calls.jsonl"), "a", encoding="utf-8").close()
        set_seed(int(self.cfg.seed))

        self.initial_prompt_bank = self._default_prompt_bank()
        self.initial_agent_prompts = self._build_initial_prompts()
        self.initial_agent_prompt_hashes = [self._hash(p) for p in self.initial_agent_prompts]
        self.agents = [AgentState(p, homogeneity_window=self.homogeneity_window) for p in self.initial_agent_prompts]
        self._initialize_prompt_beams()

        self.history: List[Dict[str, Any]] = []
        self.update_logs: List[Dict[str, Any]] = []
        self.trajectory_events: List[Dict[str, Any]] = []
        self.train_step_logs: List[Dict[str, Any]] = []
        self.train_trace_history_logs: List[Dict[str, Any]] = []
        self.test_trace_history_logs: List[Dict[str, Any]] = []
        self.recent_window_records: List[Dict[str, Any]] = []
        self.prompt_history = self._init_prompt_history()
        self.joint_diversity_cache: Dict[str, Dict[str, Any]] = {}
        self.solver_rollout_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.solver_rollout_inflight: Dict[str, asyncio.Future] = {}
        self.solver_rollout_inflight_lock = asyncio.Lock()
        self.optimizer_generation_diagnostics: Dict[str, Dict[str, Any]] = {}
        self.no_effective_evolution_counter = 0
        self.no_effective_evolution_stopped = False
        self.no_effective_evolution_reason = ""
        self.specialization_strength = 0.0
        self.effective_residual_strength = 0.0
        self.previous_epoch_per_agent_acc: List[float] = []
        self.previous_epoch_bottom2_mean_acc = 0.0
        self.competence_phase_epoch = 1
        self.competence_schedule_version = str(getattr(self.cfg, "competence_schedule_version", "competence_depth_v1"))
        self.specialization_strength_history: List[float] = []
        self.competence_probe_indices: List[int] = []
        self.competence_probe_question_hashes: List[str] = []
        self.initial_competence_probe_metrics: Dict[str, Any] = {}
        self.latest_competence_probe_metrics: Dict[str, Any] = {}
        self.competence_probe_history: List[Dict[str, Any]] = []
        self.initial_active_prompt_hashes: List[str] = list(self.initial_agent_prompt_hashes)
        self.first_nonzero_specialization_epoch: Optional[int] = None
        self.effective_specialization_epoch_count = 0
        self.depth1_guard_rejection_count = 0
        self.catastrophic_accuracy_guard_rejection_count = 0
        self.soft_error_dependence_penalty_count = 0
        self.soft_cycle_penalty_count = 0
        self.soft_mechanism_shift_penalty_count = 0
        self.exploration_candidate_count = 0
        self.exploration_slot_occupancy_count = 0
        self.exploration_to_active_conversion_count = 0
        self.hybrid_selector_history: List[Dict[str, Any]] = []
        self.mechanism_signature_history: List[Dict[str, Any]] = []
        self.mechanism_signature_by_prompt_hash: Dict[str, List[str]] = {}
        self.beam_slot_state: Dict[str, Any] = {}
        self.exploration_slot_candidates: List[Dict[str, Any]] = []
        self.mechanism_embedding_cache: Dict[str, List[float]] = {}
        self.prompt_probe_cache: Dict[str, Dict[str, Any]] = {}
        self.mechanism_embedding_cache_hit_count = 0
        self.mechanism_embedding_cache_miss_count = 0
        self.full_probe_cache_hit_count = 0
        self.full_probe_missing_pair_evaluation_count = 0
        self.behavior_profile_by_prompt_hash: Dict[str, Dict[str, Any]] = {}
        self.joint_team_selection_history: List[Dict[str, Any]] = []
        self.lineage_history: List[Dict[str, Any]] = []
        self.quality_diversity_archive_history: List[Dict[str, Any]] = []
        self.behavior_profile_history: List[Dict[str, Any]] = []
        self.total_agent_update_count = 0
        self.task_repair_niche_occupancy_count = 0
        self.mechanism_niche_occupancy_count = 0
        self.peer_collapse_soft_count = 0
        self.peer_collapse_hard_rejection_count = 0
        self.latest_joint_team_metrics: Dict[str, Any] = {}
        self.qd_no_diversification_epochs = 0
        self.qd_change_limit_relaxed_epoch = -1
        self.qd_previous_active_niche_count = 0
        self.probation_to_safe_conversion_count = 0
        self.probation_expired_count = 0
        self.candidate_starvation_count = 0
        self.mechanism_starvation_count = 0
        self.search_branch_starvation_count = 0
        self.refill_requirements_unmet_count = 0
        self.per_agent_optimizer_update_count: Dict[str, int] = {}
        self.prompt_overlength_rejection_count = 0
        self.truncated_prompt_count = 0
        self.llm_call_logs: List[Dict[str, Any]] = []
        self.cost_summary: Dict[str, Any] = self._empty_cost_summary()
        self.embedding_model = None
        self.embedding_cache: Dict[str, List[float]] = {}
        self.solver_call_limit = max(1, int(getattr(self.cfg, "eval_solver_call_concurrency", 225) or 225))
        self.solver_call_semaphore = asyncio.Semaphore(self.solver_call_limit)

        self._load_recorded_solver_rollouts()
        self.write_run_meta()
        resume_existing = bool(int(getattr(self.cfg, "resume_from_checkpoint", False) or False))
        if not (resume_existing and os.path.exists(os.path.join(self.cfg.out_dir, "prompt_history.json"))):
            self.flush_prompt_history()
        if not (resume_existing and os.path.exists(os.path.join(self.cfg.out_dir, "cost_summary.json"))):
            self.write_cost_summary()

    def _build_task_spec(self) -> TaskSpec:
        answer_format = str(getattr(self.cfg, "answer_format", "") or "").strip()
        if not answer_format:
            return get_task_spec(self.cfg.task_type)
        return TaskSpec(
            name=f"{self.cfg.task_type}:{answer_format}",
            parse_gold=lambda answer, question=None: canonical_answer_format(answer, answer_format),
            extract_pred=lambda text, question=None: extract_prediction_format(text, answer_format),
            match_answer=lambda pred, gold: match_answer_format(pred, gold, answer_format),
        )

    def _is_accuracy_only_mode(self) -> bool:
        return str(getattr(self.cfg, "reward_mode", "")).lower() == "accuracy_only"

    def _is_guarded_reward_mode(self) -> bool:
        return str(getattr(self.cfg, "reward_mode", "")).lower() == "guarded_diversity"

    def _is_vote_useful_diversity_mode(self) -> bool:
        return str(getattr(self.cfg, "reward_mode", "")).lower() == "vote_useful_diversity"

    def _is_coverage_useful_diversity_mode(self) -> bool:
        return str(getattr(self.cfg, "reward_mode", "")).lower() == "coverage_useful_diversity"

    def _is_competence_depth_reward_mode(self) -> bool:
        return str(getattr(self.cfg, "reward_mode", "")).lower() == "competence_depth_schedule"

    def _uses_competence_depth_pareto_selection(self) -> bool:
        return str(getattr(self.cfg, "candidate_selection_mode", "")).lower() == "competence_depth_pareto"

    def _is_v82_hybrid(self) -> bool:
        return str(getattr(self.cfg, "method_version", "legacy")) in {"v8_2_hybrid_progressive", "v8_stable_qd_lineage"}

    def _is_stable_qd_lineage(self) -> bool:
        return str(getattr(self.cfg, "method_version", "legacy")) == "v8_stable_qd_lineage"

    def _apply_competence_depth1_candidate_guard(self, metrics: Dict[str, Any]) -> bool:
        enabled = bool(getattr(self.cfg, "competence_depth1_candidate_guard_enabled", False))
        epsilon = float(getattr(self.cfg, "competence_depth1_candidate_guard_epsilon", 0.0) or 0.0)
        passed = (not enabled) or float(metrics.get("depth1_net_delta", 0.0) or 0.0) >= -epsilon
        metrics.update({
            "competence_depth1_guard_enabled": enabled,
            "competence_depth1_guard_epsilon": epsilon,
            "competence_depth1_guard_passed": bool(passed),
        })
        if not passed and not str(metrics.get("rejection_reason", "")):
            metrics["rejection_reason"] = "competence_depth1_guard"
        return bool(passed)

    def complete_competence_epoch(
        self,
        per_agent_acc: Optional[Sequence[float]] = None,
        epoch: int = 0,
        *,
        snapshot_metrics: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Advance competence scheduling; v2 accepts only a static probe snapshot."""
        if not bool(getattr(self.cfg, "competence_depth_enabled", False)):
            return float(self.specialization_strength)
        mode = str(getattr(self.cfg, "competence_schedule_mode", "absolute_legacy") or "absolute_legacy")
        if mode == "baseline_relative_opt_snapshot":
            if not isinstance(snapshot_metrics, dict):
                raise ValueError("baseline_relative_opt_snapshot requires snapshot_metrics from the optimization probe")
            if not self.initial_competence_probe_metrics:
                raise ValueError("initial_competence_probe_metrics is required before advancing the v2 schedule")
            record = competence_relative_specialization_strength(
                initial_metrics=self.initial_competence_probe_metrics,
                snapshot_metrics=snapshot_metrics,
                probe_size=int(snapshot_metrics.get("probe_size", len(self.competence_probe_indices)) or len(self.competence_probe_indices)),
                current_strength=float(self.specialization_strength),
                low_delta=float(getattr(self.cfg, "competence_relative_low_delta", 0.01)),
                high_delta=float(getattr(self.cfg, "competence_relative_high_delta", 0.06)),
                ema=float(getattr(self.cfg, "competence_schedule_ema", 0.50)),
                max_step=float(getattr(self.cfg, "competence_schedule_max_step", 0.35)),
                monotonic=bool(getattr(self.cfg, "competence_schedule_monotonic", True)),
                mean_guard_epsilon=float(getattr(self.cfg, "competence_mean_guard_epsilon", 0.01)),
                c1_guard_epsilon=float(getattr(self.cfg, "competence_c1_guard_epsilon", 0.01)),
                c2_guard_epsilon=float(getattr(self.cfg, "competence_c2_guard_epsilon", 0.01)),
            )
            record.update({"epoch": int(epoch), "version": str(self.competence_schedule_version)})
            self.previous_epoch_per_agent_acc = [float(value) for value in snapshot_metrics.get("per_agent_acc", [])]
            self.previous_epoch_bottom2_mean_acc = float(snapshot_metrics.get("bottom2_mean_acc", 0.0) or 0.0)
            self.latest_competence_probe_metrics = dict(snapshot_metrics)
            self.specialization_strength = float(record["next_specialization_strength"])
            self.competence_phase_epoch = int(epoch) + 1
            self._recompute_effective_residual_strength()
            return record
        values = [float(value) for value in (per_agent_acc or [])]
        ordered = sorted(values)
        bottom2 = float(np.mean(ordered[: min(2, len(ordered))])) if ordered else 0.0
        self.previous_epoch_per_agent_acc = values
        self.previous_epoch_bottom2_mean_acc = bottom2
        self.specialization_strength = competence_specialization_strength(
            bottom2,
            float(getattr(self.cfg, "competence_floor_low", 0.55)),
            float(getattr(self.cfg, "competence_floor_high", 0.65)),
        )
        self.competence_phase_epoch = int(epoch) + 1
        self._recompute_effective_residual_strength()
        return float(self.specialization_strength)

    def _recompute_effective_residual_strength(self, qd_ready: Optional[bool] = None) -> float:
        if not self._is_stable_qd_lineage():
            self.effective_residual_strength = float(self.specialization_strength)
            return self.effective_residual_strength
        latest = dict(getattr(self, "latest_joint_team_metrics", {}) or {})
        ready = bool(latest.get("qd_readiness_passed", False)) if qd_ready is None else bool(qd_ready)
        self.effective_residual_strength = max(
            float(self.specialization_strength),
            float(self.cfg.residual_specialization_qd_floor) if ready else 0.0,
        )
        latest.update({
            "competence_schedule_strength": float(self.specialization_strength),
            "qd_residual_floor_applied": bool(ready and self.effective_residual_strength > self.specialization_strength),
            "effective_residual_strength": float(self.effective_residual_strength),
        })
        if latest:
            self.latest_joint_team_metrics = latest
        return self.effective_residual_strength

    def _effective_progressive_weight(self, configured: float) -> float:
        if bool(getattr(self.cfg, "competence_progressive_residual_enabled", False)):
            return float(configured) * float(getattr(self, "effective_residual_strength", self.specialization_strength))
        return float(configured)

    def _effective_support_shrinkage(self) -> float:
        base = float(getattr(self.cfg, "specialization_support_shrinkage", 3.0) or 3.0)
        if not bool(getattr(self.cfg, "competence_progressive_residual_enabled", False)):
            return base
        extra = float(getattr(self.cfg, "competence_extra_support_shrinkage", 3.0) or 0.0)
        return base + (1.0 - float(self.specialization_strength)) * extra

    def _uses_baseline_candidate_metrics(self) -> bool:
        return self._is_guarded_reward_mode() or self._is_vote_useful_diversity_mode() or self._is_coverage_useful_diversity_mode() or self._is_competence_depth_reward_mode()

    def _uses_vote_pareto_selection(self) -> bool:
        return str(getattr(self.cfg, "candidate_selection_mode", "scalar_reward") or "scalar_reward").lower() in {
            "vote_pareto", "vote_error_pareto", "competence_depth_pareto"
        }

    def _uses_vote_error_pareto_selection(self) -> bool:
        cfg = getattr(self, "cfg", None)
        return str(getattr(cfg, "candidate_selection_mode", "scalar_reward") or "scalar_reward").lower() == "vote_error_pareto"

    def _residual_specialization_enabled(self) -> bool:
        return bool(getattr(self.cfg, "residual_specialization_enabled", False))

    def _v7_residual_protocol_enabled(self) -> bool:
        cfg = getattr(self, "cfg", None)
        return bool(
            self._uses_vote_error_pareto_selection()
            or any(bool(getattr(cfg, name, False)) for name in (
                "boundary_selector_enabled",
                "shared_error_metrics_enabled",
                "residual_specialization_enabled",
                "error_dependence_guard_enabled",
                "residual_cycle_guard_enabled",
                "mechanism_trust_region_enabled",
            ))
        )

    def _experiment_protocol_version(self) -> str:
        if self._is_stable_qd_lineage():
            return "vote_oriented_v8_stable_qd_lineage"
        if bool(getattr(self.cfg, "competence_depth_enabled", False)):
            return "vote_oriented_v8_competence_depth"
        return EXPERIMENT_PROTOCOL_VERSION

    def _normalized_prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(self._prompt_signature(prompt).encode("utf-8")).hexdigest()

    def prompt_change_ratio(self, parent_prompt: str, candidate_prompt: str) -> float:
        parent = self._prompt_signature(parent_prompt)
        candidate = self._prompt_signature(candidate_prompt)
        return self._clip01(1.0 - SequenceMatcher(None, parent, candidate).ratio())

    def _behavior_context_for_baseline(
        self,
        *,
        agent_id: int,
        answers: Sequence[str],
        gold: str,
        rollout: Dict[str, Any],
        question_hash: str = "",
    ) -> str:
        invalids = list(rollout.get("invalid_flags", []))
        if agent_id >= len(answers) or (agent_id < len(invalids) and int(invalids[agent_id]) > 0):
            return BehaviorContext.INVALID.value
        target_correct = bool(self._safe_agent_correct(rollout, agent_id))
        team_correct = bool(rollout.get("vote_correct", 0))
        gold_count = int(rollout.get("gold_vote_count", 0) or 0)
        largest_wrong = int(rollout.get("largest_wrong_vote_count", 0) or 0)
        if target_correct:
            if bool(getattr(self.cfg, "competence_depth_enabled", False)) and team_correct:
                without_target = list(answers)
                without_target[agent_id] = ""
                counterfactual = self._vote_with_diagnostics(without_target, question_hash=question_hash)
                if not self.task_spec.match_answer(str(counterfactual.get("vote_answer", "")), gold):
                    return BehaviorContext.TARGET_CORRECT_PIVOTAL_HOLD.value
                return BehaviorContext.TARGET_CORRECT_ROBUST.value
            if gold_count - largest_wrong <= 1:
                return BehaviorContext.TARGET_CORRECT_PIVOTAL_HOLD.value
            return BehaviorContext.TARGET_CORRECT_ROBUST.value

        if not team_correct:
            counterfactual_answers = list(answers)
            counterfactual_answers[agent_id] = str(gold)
            counterfactual = self._vote_with_diagnostics(counterfactual_answers, question_hash=question_hash)
            pivotal = self.task_spec.match_answer(str(counterfactual.get("vote_answer", "")), gold)
            return (
                BehaviorContext.TEAM_WRONG_PIVOTAL_FIX.value
                if pivotal else BehaviorContext.TEAM_WRONG_NONPIVOTAL.value
            )

        target_answer = str(answers[agent_id] or "").strip()
        wrong_counts = Counter(
            str(answer or "").strip()
            for answer in answers
            if str(answer or "").strip() and not self.task_spec.match_answer(str(answer), gold)
        )
        is_dominant_wrong = bool(
            target_answer
            and wrong_counts.get(target_answer, 0) == max(wrong_counts.values(), default=0)
            and (largest_wrong > 0 or gold_count - largest_wrong <= 1)
        )
        return (
            BehaviorContext.TEAM_CORRECT_DOMINANT_WRONG_REDUNDANCY.value
            if is_dominant_wrong or gold_count - largest_wrong <= 1
            else BehaviorContext.TEAM_CORRECT_TARGET_WRONG_OTHER.value
        )

    def _candidate_behavior_metrics(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        values = {context: [] for context in BEHAVIOR_CONTEXT_NAMES}
        context_counts = {context: 0 for context in BEHAVIOR_CONTEXT_NAMES}
        fingerprint: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            context = str(row.get("behavior_context", BehaviorContext.INVALID.value))
            if context not in values:
                context = BehaviorContext.INVALID.value
            vote_gain = int(not bool(row.get("baseline_vote_correct", 0)) and bool(row.get("candidate_vote_correct", 0)))
            vote_loss = int(bool(row.get("baseline_vote_correct", 0)) and not bool(row.get("candidate_vote_correct", 0)))
            margin_delta = float(row.get("candidate_mean_vote_margin", -1.0)) - float(row.get("baseline_mean_vote_margin", -1.0))
            wrong_to_correct = int(not bool(row.get("baseline_target_correct", 0)) and bool(row.get("target_agent_correct", 0)))
            correct_to_wrong = int(bool(row.get("baseline_target_correct", 0)) and not bool(row.get("target_agent_correct", 0)))
            transition = 2.0 * vote_gain - 2.0 * vote_loss + max(0.0, margin_delta) - max(0.0, -margin_delta) + 0.5 * wrong_to_correct - 0.5 * correct_to_wrong
            values[context].append(float(transition))
            context_counts[context] += 1
            question_hash = str(row.get("question_hash", ""))
            if question_hash:
                answer_signature = self._prompt_signature(str(row.get("target_answer", "")))
                fingerprint[question_hash] = {
                    "target_correct": bool(row.get("target_agent_correct", 0)),
                    "target_answer_hash": hashlib.sha256(answer_signature.encode("utf-8")).hexdigest(),
                    "team_vote_correct": bool(row.get("candidate_vote_correct", 0)),
                    "vote_margin_bucket": int(round(10.0 * float(row.get("candidate_mean_vote_margin", -1.0)))),
                    "behavior_context": context,
                }
        return {
            "behavior_context_counts": context_counts,
            "candidate_transition_vector": {
                context: float(np.mean(context_values)) if context_values else 0.0
                for context, context_values in values.items()
            },
            "candidate_transition_support": context_counts,
            "behavior_fingerprint": fingerprint,
        }

    def _candidate_boundary_error_metrics(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        count = len(rows)
        denominator = float(count) if count else 1.0
        individual_fix = individual_regression = 0
        pivotal_rescue = pivotal_loss = 0
        plurality_opportunity = plurality_fix = plurality_loss = 0
        same_wrong_break = same_wrong_create = 0
        shared_rescue = shared_creation = 0.0
        for row in rows:
            baseline_correct = bool(row.get("baseline_target_correct", False))
            candidate_correct = bool(row.get("candidate_target_correct", row.get("target_agent_correct", False)))
            baseline_vote = bool(row.get("baseline_vote_correct", False))
            candidate_vote = bool(row.get("candidate_vote_correct", False))
            fixed = not baseline_correct and candidate_correct
            regressed = baseline_correct and not candidate_correct
            individual_fix += int(fixed)
            individual_regression += int(regressed)
            pivotal_rescue += int(fixed and not baseline_vote and candidate_vote)
            pivotal_loss += int(regressed and baseline_vote and not candidate_vote)
            plurality_opportunity += int(bool(row.get("plurality_pivotal_fix_opportunity", False)))
            plurality_fix += int(bool(row.get("plurality_pivotal_fix", False)))
            plurality_loss += int(bool(row.get("plurality_pivotal_loss", False)))
            peer_wrong_count = int(row.get("peer_wrong_count", 0) or 0)
            shared_weight = float(peer_wrong_count) / max(1, len(self.agents) - 1)
            shared_rescue += shared_weight * int(fixed)
            shared_creation += shared_weight * int(regressed)
            baseline_cluster = bool(row.get("baseline_target_in_dominant_wrong_cluster", False))
            candidate_cluster = bool(row.get("candidate_target_in_dominant_wrong_cluster", False))
            same_wrong_break += int(baseline_cluster and not candidate_cluster)
            same_wrong_create += int(not baseline_cluster and candidate_cluster)
        pivotal_rescue_rate = float(pivotal_rescue) / denominator if count else 0.0
        pivotal_loss_rate = float(pivotal_loss) / denominator if count else 0.0
        shared_rescue_score = float(shared_rescue) / denominator if count else 0.0
        shared_creation_score = float(shared_creation) / denominator if count else 0.0
        same_wrong_break_rate = float(same_wrong_break) / denominator if count else 0.0
        same_wrong_create_rate = float(same_wrong_create) / denominator if count else 0.0
        legacy_net_gain = (
            4.0 * pivotal_rescue_rate
            - 4.0 * pivotal_loss_rate
            + shared_rescue_score
            - 1.5 * shared_creation_score
            + 0.5 * same_wrong_break_rate
            - 0.5 * same_wrong_create_rate
        )
        plurality_opportunity_rate = float(plurality_opportunity) / denominator if count else 0.0
        plurality_fix_rate = float(plurality_fix) / denominator if count else 0.0
        plurality_loss_rate = float(plurality_loss) / denominator if count else 0.0
        plurality_net_gain = (
            4.0 * plurality_fix_rate
            - 4.0 * plurality_loss_rate
            + shared_rescue_score
            - 1.5 * shared_creation_score
            + 0.5 * same_wrong_break_rate
            - 0.5 * same_wrong_create_rate
        )
        active_net_gain = plurality_net_gain if bool(getattr(self.cfg, "competence_depth_enabled", False)) else legacy_net_gain
        return {
            "individual_fix_count": int(individual_fix),
            "individual_regression_count": int(individual_regression),
            "pivotal_rescue_count": int(pivotal_rescue),
            "pivotal_rescue_rate": pivotal_rescue_rate,
            "pivotal_loss_count": int(pivotal_loss),
            "pivotal_loss_rate": pivotal_loss_rate,
            "shared_error_rescue_score": shared_rescue_score,
            "shared_error_creation_score": shared_creation_score,
            "same_wrong_cluster_break_count": int(same_wrong_break),
            "same_wrong_cluster_create_count": int(same_wrong_create),
            "same_wrong_cluster_break_rate": same_wrong_break_rate,
            "same_wrong_cluster_create_rate": same_wrong_create_rate,
            "plurality_pivotal_fix_opportunity_count": int(plurality_opportunity),
            "plurality_pivotal_fix_opportunity_rate": plurality_opportunity_rate,
            "plurality_pivotal_fix_count": int(plurality_fix),
            "plurality_pivotal_fix_rate": plurality_fix_rate,
            "plurality_pivotal_loss_count": int(plurality_loss),
            "plurality_pivotal_loss_rate": plurality_loss_rate,
            "plurality_boundary_shared_error_net_gain": float(plurality_net_gain),
            "boundary_shared_error_net_gain": float(active_net_gain),
            "pivotal_definition": (
                "actual_plurality_counterfactual"
                if bool(getattr(self.cfg, "competence_depth_enabled", False))
                else "legacy_vote_boundary"
            ),
        }

    def _candidate_residual_metrics(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        tau = self._effective_support_shrinkage()
        support = {family: 0 for family in CAPABILITY_RESIDUAL_FAMILY_NAMES}
        weighted_gain = {family: 0.0 for family in CAPABILITY_RESIDUAL_FAMILY_NAMES}
        weighted_loss = {family: 0.0 for family in CAPABILITY_RESIDUAL_FAMILY_NAMES}
        weighted_sum = {family: 0.0 for family in CAPABILITY_RESIDUAL_FAMILY_NAMES}
        for row in rows:
            family = str(row.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value))
            if family not in support:
                family = CapabilityResidualFamily.UNKNOWN.value
            vote_gain = int(not bool(row.get("baseline_vote_correct", 0)) and bool(row.get("candidate_vote_correct", 0)))
            vote_loss = int(bool(row.get("baseline_vote_correct", 0)) and not bool(row.get("candidate_vote_correct", 0)))
            margin_delta = float(row.get("candidate_mean_vote_margin", -1.0)) - float(row.get("baseline_mean_vote_margin", -1.0))
            fixed = int(not bool(row.get("baseline_target_correct", 0)) and bool(row.get("candidate_target_correct", row.get("target_agent_correct", 0))))
            regressed = int(bool(row.get("baseline_target_correct", 0)) and not bool(row.get("candidate_target_correct", row.get("target_agent_correct", 0))))
            raw = 2.0 * vote_gain - 2.0 * vote_loss + margin_delta + 0.5 * fixed - 0.5 * regressed
            context = str(row.get("behavior_context", BehaviorContext.INVALID.value))
            context_weight = float(VOTE_CONTEXT_WEIGHTS.get(context, 1.0))
            weighted = context_weight * raw
            support[family] += 1
            weighted_sum[family] += weighted
            weighted_gain[family] += max(0.0, weighted)
            weighted_loss[family] += max(0.0, -weighted)
            row["raw_transition_value"] = float(raw)
            row["vote_context_weight"] = context_weight
        shrunk = {}
        reliability = {}
        for family in CAPABILITY_RESIDUAL_FAMILY_NAMES:
            reliability[family] = float(support[family] / (support[family] + tau)) if support[family] else 0.0
            shrunk[family] = float(weighted_sum[family] / (support[family] + tau)) if support[family] else 0.0
        for row in rows:
            family = str(row.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value))
            if family not in support:
                family = CapabilityResidualFamily.UNKNOWN.value
            row["support_reliability"] = reliability[family]
            row["shrunk_transition_value"] = shrunk[family]
        return {
            "capability_transition_support": support,
            "capability_weighted_gain": weighted_gain,
            "capability_weighted_loss": weighted_loss,
            "capability_support_reliability": reliability,
            "capability_shrunk_transition": shrunk,
            "capability_evidence_rows": [
                {
                    "question_hash": str(row.get("question_hash", "")),
                    "capability_residual_family": str(row.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value)),
                    "raw_transition_value": float(row.get("raw_transition_value", 0.0) or 0.0),
                    "vote_context_weight": float(row.get("vote_context_weight", 0.0) or 0.0),
                    "support_reliability": float(row.get("support_reliability", 0.0) or 0.0),
                    "shrunk_transition_value": float(row.get("shrunk_transition_value", 0.0) or 0.0),
                }
                for row in rows
            ],
        }

    @staticmethod
    def _candidate_v7_log_fields(metrics: Mapping[str, Any]) -> Dict[str, Any]:
        count_fields = (
            "individual_fix_count",
            "individual_regression_count",
            "pivotal_rescue_count",
            "pivotal_loss_count",
            "same_wrong_cluster_break_count",
            "same_wrong_cluster_create_count",
        )
        rate_fields = (
            "pivotal_rescue_rate",
            "pivotal_loss_rate",
            "shared_error_rescue_score",
            "shared_error_creation_score",
            "same_wrong_cluster_break_rate",
            "same_wrong_cluster_create_rate",
            "boundary_shared_error_net_gain",
            "capability_alignment",
        )
        fields: Dict[str, Any] = {
            key: int(metrics.get(key, 0) or 0) for key in count_fields
        }
        fields.update({
            key: float(metrics.get(key, 0.0) or 0.0) for key in rate_fields
        })
        fields.update({
            "error_dependence_guard_passed": bool(metrics.get("error_dependence_guard_passed", True)),
            "paired_boundary_transition_rows": metrics.get("paired_boundary_transition_rows", []),
            "capability_transition_support": metrics.get("capability_transition_support", {}),
            "capability_weighted_gain": metrics.get("capability_weighted_gain", {}),
            "capability_weighted_loss": metrics.get("capability_weighted_loss", {}),
            "capability_support_reliability": metrics.get("capability_support_reliability", {}),
            "capability_shrunk_transition": metrics.get("capability_shrunk_transition", {}),
            "capability_evidence_rows": metrics.get("capability_evidence_rows", []),
        })
        return fields

    def capability_alignment(self, agent: AgentState, metrics: Dict[str, Any]) -> float:
        positive = np.array([
            max(0.0, float(metrics.get("capability_shrunk_transition", {}).get(family, 0.0)))
            for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
        ], dtype=float)
        profile = np.array([
            max(0.0, float(agent.capability_profile.get(family, 0.0)))
            for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
        ], dtype=float)
        denominator = float(np.linalg.norm(positive) * np.linalg.norm(profile))
        return self._clip01(float(np.dot(positive, profile) / denominator)) if denominator > 0.0 else 0.0

    def _accumulate_capability_evidence(self, agent: AgentState, metrics: Dict[str, Any], epoch_id: int) -> None:
        agent.pending_capability_evidence.append({
            "epoch": int(epoch_id),
            "support": dict(metrics.get("capability_transition_support", {})),
            "weighted_gain": dict(metrics.get("capability_weighted_gain", {})),
            "weighted_loss": dict(metrics.get("capability_weighted_loss", {})),
        })
        agent.pending_capability_update_count += 1

    def _flush_capability_profile(self, agent: AgentState, epoch_id: int, force: bool = False) -> bool:
        period = max(1, int(getattr(self.cfg, "specialization_update_period", 2) or 2))
        if not agent.pending_capability_evidence or (not force and agent.pending_capability_update_count < period):
            return False
        tau = self._effective_support_shrinkage()
        loss_weight = float(getattr(self.cfg, "capability_loss_weight", 1.5) or 1.5)
        for pending in agent.pending_capability_evidence:
            for family in CAPABILITY_RESIDUAL_FAMILY_NAMES:
                evidence = agent.capability_evidence[family]
                evidence.support += int(pending.get("support", {}).get(family, 0) or 0)
                evidence.weighted_gain += float(pending.get("weighted_gain", {}).get(family, 0.0) or 0.0)
                evidence.weighted_loss += float(pending.get("weighted_loss", {}).get(family, 0.0) or 0.0)
                reliability = float(evidence.support / (evidence.support + tau)) if evidence.support else 0.0
                evidence.posterior_value = reliability * (
                    evidence.weighted_gain - loss_weight * evidence.weighted_loss
                )
                evidence.last_updated_epoch = int(epoch_id)
        positive = {
            family: max(0.0, agent.capability_evidence[family].posterior_value)
            for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
        }
        total_positive = sum(positive.values())
        changed = False
        if total_positive > 0.0:
            target = {family: value / total_positive for family, value in positive.items()}
            old = dict(agent.capability_profile or empty_capability_profile())
            mu = float(getattr(self.cfg, "specialization_ema", 0.20))
            updated = {
                family: (1.0 - mu) * float(old.get(family, 0.0)) + mu * target[family]
                for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
            }
            total = sum(updated.values())
            agent.capability_profile = {family: value / total for family, value in updated.items()}
            changed = True
        agent.pending_capability_evidence.clear()
        agent.pending_capability_update_count = 0
        if changed:
            agent.capability_profile_update_count += 1
        return changed

    def _update_vote_context_profile(self, agent: AgentState, metrics: Dict[str, Any]) -> bool:
        transition = metrics.get("candidate_transition_vector", {})
        positive = {context: max(0.0, float(transition.get(context, 0.0))) for context in BEHAVIOR_CONTEXT_NAMES}
        total = sum(positive.values())
        if total <= 0.0:
            return False
        target = {context: value / total for context, value in positive.items()}
        mu = float(getattr(self.cfg, "specialization_ema", 0.20))
        old = dict(agent.vote_context_profile or uniform_vote_context_profile())
        updated = {context: (1.0 - mu) * old.get(context, 0.0) + mu * target[context] for context in BEHAVIOR_CONTEXT_NAMES}
        norm = sum(updated.values())
        agent.vote_context_profile = {context: value / norm for context, value in updated.items()}
        return True

    @staticmethod
    def behavior_fingerprint_similarity(
        candidate: Mapping[str, Any],
        history: Mapping[str, Any],
    ) -> Tuple[float, int]:
        overlap = sorted(set(candidate).intersection(history))
        if not overlap:
            return 0.0, 0
        correctness_matches = 0
        answer_matches = 0
        for key in overlap:
            current = candidate[key]
            previous = history[key]
            current_correct = current.target_correct if isinstance(current, BehaviorFingerprintEntry) else bool(current.get("target_correct", False))
            previous_correct = previous.target_correct if isinstance(previous, BehaviorFingerprintEntry) else bool(previous.get("target_correct", False))
            current_answer = current.target_answer_hash if isinstance(current, BehaviorFingerprintEntry) else str(current.get("target_answer_hash", ""))
            previous_answer = previous.target_answer_hash if isinstance(previous, BehaviorFingerprintEntry) else str(previous.get("target_answer_hash", ""))
            correctness_matches += int(current_correct == previous_correct)
            answer_matches += int(current_answer == previous_answer)
        count = len(overlap)
        return 0.7 * correctness_matches / count + 0.3 * answer_matches / count, count

    @staticmethod
    def behavior_fingerprint_utility(fingerprint: Mapping[str, Any]) -> Dict[str, float]:
        utility: Dict[str, float] = {}
        for key, entry in fingerprint.items():
            target_correct = entry.target_correct if isinstance(entry, BehaviorFingerprintEntry) else bool(entry.get("target_correct", False))
            team_correct = entry.team_vote_correct if isinstance(entry, BehaviorFingerprintEntry) else bool(entry.get("team_vote_correct", False))
            margin_bucket = entry.vote_margin_bucket if isinstance(entry, BehaviorFingerprintEntry) else int(entry.get("vote_margin_bucket", 0) or 0)
            utility[str(key)] = 2.0 * float(team_correct) + float(target_correct) + 0.5 * float(margin_bucket) / 10.0
        return utility

    @staticmethod
    def paired_utility_improvement(
        candidate: Mapping[str, float], history: Mapping[str, float]
    ) -> Tuple[float, int]:
        overlap = sorted(set(candidate).intersection(history))
        if not overlap:
            return 0.0, 0
        return float(np.mean([float(candidate[key]) - float(history[key]) for key in overlap])), len(overlap)

    def _append_bounded_archive(self, archive: List[Any], value: Any) -> None:
        archive.append(value)
        limit = max(0, int(getattr(self.cfg, "behavior_archive_size", 16)))
        if limit == 0:
            archive.clear()
        elif len(archive) > limit:
            del archive[:-limit]

    def _candidate_trajectory_feasibility(
        self,
        agent: AgentState,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        prompt = str(item.get("prompt", ""))
        parent_prompt = str(item.get("parent_prompt", agent.current_prompt))
        prompt_hash = self._normalized_prompt_hash(prompt)
        parent_hash = self._normalized_prompt_hash(parent_prompt)
        change_ratio = self.prompt_change_ratio(parent_prompt, prompt)
        diagnostics: Dict[str, Any] = {
            "prompt_hash": prompt_hash,
            "parent_prompt_hash": parent_hash,
            "prompt_change_ratio": change_ratio,
            "max_behavior_cycle_similarity": 0.0,
            "behavior_cycle_overlap": 0,
            "matched_behavior_state_id": "",
            "exact_prompt_cycle": False,
            "behavior_cycle_guard_passed": True,
            "prompt_trust_region_passed": True,
            "rejection_reason": "",
        }
        if not (
            bool(getattr(self.cfg, "residual_cycle_guard_enabled", False))
            or bool(getattr(self.cfg, "mechanism_trust_region_enabled", False))
        ):
            return diagnostics

        source = self._candidate_pool_source(item)
        current_hash = self._normalized_prompt_hash(agent.current_prompt)
        if prompt_hash == current_hash and source in {"existing_beam", "current_active_fallback"}:
            return diagnostics

        _, _, original_guards_passed = self._vote_pareto_feasibility(metrics)
        if not original_guards_passed:
            return diagnostics

        proposal = item.get("proposal", {}) if isinstance(item.get("proposal", {}), dict) else {}
        if source == "optimizer" and bool(getattr(self.cfg, "mechanism_trust_region_enabled", False)):
            preserved = proposal.get("preserved_mechanisms", [])
            modified = proposal.get("modified_mechanism", proposal.get("new_or_modified_mechanism", ""))
            change_summary = str(proposal.get("change_summary", "")).strip()
            mechanism_contract_passed = bool(
                isinstance(preserved, list)
                and any(str(value).strip() for value in preserved)
                and isinstance(modified, str)
                and bool(modified.strip())
                and bool(change_summary)
            )
            diagnostics["mechanism_contract_passed"] = mechanism_contract_passed
            if not mechanism_contract_passed:
                diagnostics["prompt_trust_region_passed"] = False
                diagnostics["rejection_reason"] = "mechanism_contract_missing"
                return diagnostics

        historic_hashes = {self._normalized_prompt_hash(value) for value in agent.history}
        historic_hashes.update(state.prompt_hash for state in agent.accepted_behavior_archive)
        historic_hashes.update(state.prompt_hash for state in agent.rejected_behavior_archive)
        exact_cycle = bool(prompt_hash in historic_hashes)
        diagnostics["exact_prompt_cycle"] = exact_cycle
        if exact_cycle:
            diagnostics["rejection_reason"] = "exact_prompt_cycle"
            return diagnostics

        candidate_fingerprint = metrics.get("behavior_fingerprint", {})
        best_similarity = 0.0
        best_overlap = 0
        best_state_id = ""
        residual_cycle = bool(getattr(self.cfg, "residual_cycle_guard_enabled", False))
        candidate_utility = self.behavior_fingerprint_utility(candidate_fingerprint) if isinstance(candidate_fingerprint, dict) else {}
        matched_kind = ""
        utility_improvement = 0.0
        if bool(getattr(self.cfg, "behavior_cycle_guard_enabled", True)) and isinstance(candidate_fingerprint, dict):
            for state in agent.accepted_behavior_archive:
                similarity, overlap = self.behavior_fingerprint_similarity(candidate_fingerprint, state.behavior_fingerprint)
                if (similarity, overlap, state.state_id) > (best_similarity, best_overlap, best_state_id):
                    best_similarity, best_overlap, best_state_id = similarity, overlap, state.state_id
                    matched_kind = "accepted"
                    historical_utility = state.paired_behavior_utility or self.behavior_fingerprint_utility(state.behavior_fingerprint)
                    utility_improvement, _ = self.paired_utility_improvement(candidate_utility, historical_utility)
            if residual_cycle:
                for state in agent.rejected_behavior_archive:
                    similarity, overlap = self.behavior_fingerprint_similarity(candidate_fingerprint, state.behavior_fingerprint)
                    if (similarity, overlap, state.state_id) > (best_similarity, best_overlap, best_state_id):
                        best_similarity, best_overlap, best_state_id = similarity, overlap, state.state_id
                        matched_kind = "rejected"
                        utility_improvement, _ = self.paired_utility_improvement(
                            candidate_utility, state.paired_behavior_utility
                        )
        diagnostics.update(
            {
                "max_behavior_cycle_similarity": float(best_similarity),
                "behavior_cycle_overlap": int(best_overlap),
                "matched_behavior_state_id": best_state_id,
                "matched_behavior_archive": matched_kind,
                "paired_behavior_utility_improvement": float(utility_improvement),
            }
        )
        meaningful_improvement = bool(
            float(metrics.get("vote_delta", 0.0) or 0.0) > float(getattr(self.cfg, "behavior_cycle_improvement_epsilon", 0.01))
            or float(metrics.get("accuracy_delta", 0.0) or 0.0) > float(getattr(self.cfg, "behavior_cycle_improvement_epsilon", 0.01))
            or float(metrics.get("vote_margin_delta", 0.0) or 0.0) > float(getattr(self.cfg, "behavior_cycle_margin_epsilon", 0.05))
        )
        if residual_cycle:
            meaningful_improvement = bool(
                utility_improvement > float(getattr(self.cfg, "behavior_cycle_improvement_epsilon", 0.01))
            )
        behavior_cycle = bool(
            bool(getattr(self.cfg, "behavior_cycle_guard_enabled", True))
            and best_overlap >= int(getattr(self.cfg, "behavior_cycle_min_overlap", 16))
            and best_similarity >= float(getattr(self.cfg, "behavior_cycle_similarity_threshold", 0.95))
            and not meaningful_improvement
        )
        diagnostics["behavior_cycle_guard_passed"] = not behavior_cycle
        if behavior_cycle:
            diagnostics["rejection_reason"] = (
                "rejected_failure_cycle" if residual_cycle and matched_kind == "rejected"
                else "accepted_state_cycle" if residual_cycle
                else "behavior_cycle"
            )
            return diagnostics

        large_shift = bool(
            source == "optimizer"
            and
            bool(getattr(self.cfg, "prompt_trust_region_enabled", True))
            and agent.accept_count >= int(getattr(self.cfg, "prompt_large_shift_warmup_accepts", 2))
            and change_ratio > float(getattr(self.cfg, "prompt_max_change_ratio", 0.45))
        )
        large_shift_supported = bool(
            float(metrics.get("vote_delta", 0.0) or 0.0) >= float(getattr(self.cfg, "prompt_large_shift_min_vote_delta", 0.02))
            and float(metrics.get("accuracy_delta", 0.0) or 0.0) >= 0.0
            and float(metrics.get("vote_loss_rate", 0.0) or 0.0) <= float(getattr(self.cfg, "baseline_allowed_vote_loss", 0.0))
        )
        if bool(getattr(self.cfg, "mechanism_trust_region_enabled", False)):
            large_shift_supported = bool(
                large_shift_supported
                and float(metrics.get("pivotal_loss_rate", 0.0) or 0.0) <= 0.0
                and float(metrics.get("shared_error_creation_score", 0.0) or 0.0)
                <= float(metrics.get("shared_error_rescue_score", 0.0) or 0.0)
            )
        diagnostics["prompt_trust_region_passed"] = bool(not large_shift or large_shift_supported)
        if large_shift and not large_shift_supported:
            diagnostics["rejection_reason"] = "unsupported_large_prompt_shift"
        return diagnostics

    def _trajectory_event(
        self,
        *,
        agent_id: int,
        epoch_id: int,
        step_id: int,
        item: Dict[str, Any],
        accepted: bool,
        profile_before: Dict[str, float],
        profile_after: Dict[str, float],
    ) -> Dict[str, Any]:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        return {
            **self._base_log_fields(),
            "event": "trajectory_evolution",
            "epoch": int(epoch_id),
            "step": int(step_id),
            "agent_id": int(agent_id),
            "candidate_id": str(item.get("candidate_id", "")),
            "candidate_pool_source": self._candidate_pool_source(item),
            "candidate_source": self._candidate_generation_source(item),
            "accepted": bool(accepted),
            "behavior_context_counts": metrics.get("behavior_context_counts", {}),
            "candidate_transition_vector": metrics.get("candidate_transition_vector", {}),
            "candidate_transition_support": metrics.get("candidate_transition_support", {}),
            "capability_profile_before": profile_before,
            "capability_profile_after": profile_after,
            "vote_context_profile": dict(self.agents[agent_id].vote_context_profile),
            "capability_profile": dict(self.agents[agent_id].capability_profile),
            "capability_transition_support": metrics.get("capability_transition_support", {}),
            "capability_support_reliability": metrics.get("capability_support_reliability", {}),
            "capability_shrunk_transition": metrics.get("capability_shrunk_transition", {}),
            "capability_alignment": float(metrics.get("capability_alignment", 0.0) or 0.0),
            **{key: metrics.get(key) for key in (
                "prompt_hash", "parent_prompt_hash", "prompt_change_ratio",
                "max_behavior_cycle_similarity", "behavior_cycle_overlap", "matched_behavior_state_id",
                "exact_prompt_cycle", "behavior_cycle_guard_passed", "prompt_trust_region_passed", "rejection_reason",
                "matched_behavior_archive", "paired_behavior_utility_improvement", "mechanism_contract_passed",
            )},
        }

    def _vote_pareto_feasibility(self, metrics: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        weights = self._effective_reward_weights()
        baseline_target = float(metrics.get("baseline_target_accuracy", 0.0) or 0.0)
        candidate_target = float(metrics.get("candidate_target_accuracy", metrics.get("target_agent_accuracy", 0.0)) or 0.0)
        baseline_invalid = float(metrics.get("baseline_invalid_rate", 0.0) or 0.0)
        candidate_invalid = float(metrics.get("candidate_invalid_rate", metrics.get("invalid_rate", 0.0)) or 0.0)
        guard_epsilon = float(weights.get("accuracy_guard_epsilon", 0.0))
        if self._is_v82_hybrid():
            guard_epsilon = float(getattr(self.cfg, "catastrophic_target_accuracy_loss_epsilon", 0.05))
        elif self._uses_competence_depth_pareto_selection():
            guard_epsilon = float(self.specialization_strength) * float(getattr(self.cfg, "accuracy_guard_epsilon", guard_epsilon))
        accuracy_guard_passed = candidate_target >= baseline_target - guard_epsilon
        metrics["effective_accuracy_guard_epsilon"] = guard_epsilon
        invalid_guard_passed = candidate_invalid <= baseline_invalid + float(getattr(self.cfg, "invalid_guard_epsilon", 0.0) or 0.0)
        dependence_guard_passed = bool(self._is_v82_hybrid() or (
            not bool(getattr(self.cfg, "error_dependence_guard_enabled", False))
            or (
                float(metrics.get("pivotal_loss_rate", 0.0) or 0.0)
                <= float(getattr(self.cfg, "pivotal_loss_guard_epsilon", 0.0) or 0.0)
                and float(metrics.get("shared_error_creation_score", 0.0) or 0.0)
                <= float(metrics.get("shared_error_rescue_score", 0.0) or 0.0)
                + float(getattr(self.cfg, "shared_error_creation_epsilon", 0.02) or 0.0)
            )
        ))
        metrics["error_dependence_guard_passed"] = dependence_guard_passed
        return bool(accuracy_guard_passed), bool(invalid_guard_passed), bool(
            accuracy_guard_passed and invalid_guard_passed and dependence_guard_passed
        )

    def _vote_pareto_active_sort_key(self, item: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float, float, int, str]:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        rank = item.get("pareto_rank")
        normalized_rank = int(rank) if isinstance(rank, int) and rank >= 0 else 10**9
        return (
            -float(metrics.get("vote_delta", 0.0) or 0.0),
            float(metrics.get("vote_loss_rate", 0.0) or 0.0),
            -float(metrics.get("vote_gain_rate", 0.0) or 0.0),
            -float(metrics.get("vote_margin_delta", 0.0) or 0.0),
            -float(metrics.get("candidate_target_accuracy", metrics.get("target_agent_accuracy", 0.0)) or 0.0),
            -float(
                metrics.get("boundary_shared_error_net_gain", 0.0)
                if self._uses_vote_error_pareto_selection()
                else metrics.get("boundary_useful_diversity_delta", 0.0)
                or 0.0
            ),
            float(metrics.get("candidate_invalid_rate", metrics.get("invalid_rate", 0.0)) or 0.0),
            normalized_rank,
            str(item.get("candidate_id", "")),
        )

    def _vote_pareto_crowding_sort_key(self, item: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float, float, float, str]:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        distance = float(item.get("pareto_crowding_distance", 0.0) or 0.0)
        return (
            -distance,
            -float(metrics.get("vote_delta", 0.0) or 0.0),
            float(metrics.get("vote_loss_rate", 0.0) or 0.0),
            -float(metrics.get("vote_gain_rate", 0.0) or 0.0),
            -float(metrics.get("vote_margin_delta", 0.0) or 0.0),
            -float(metrics.get("candidate_target_accuracy", metrics.get("target_agent_accuracy", 0.0)) or 0.0),
            -float(
                metrics.get("boundary_shared_error_net_gain", 0.0)
                if self._uses_vote_error_pareto_selection()
                else metrics.get("boundary_useful_diversity_delta", 0.0)
                or 0.0
            ),
            float(metrics.get("candidate_invalid_rate", metrics.get("invalid_rate", 0.0)) or 0.0),
            str(item.get("candidate_id", "")),
        )

    def _competence_depth_sort_key(self, item: Dict[str, Any]) -> Tuple[Any, ...]:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        strength = float(self.specialization_strength)
        early = (
            -float(metrics.get("candidate_target_accuracy", 0.0) or 0.0),
            -float(metrics.get("depth2_gain_rate", 0.0) or 0.0),
            -float(metrics.get("depth2_net_delta", 0.0) or 0.0),
        )
        late = (
            -float(metrics.get("vote_gain_rate", 0.0) or 0.0),
            float(metrics.get("vote_loss_rate", 0.0) or 0.0),
            -float(metrics.get("boundary_shared_error_net_gain", 0.0) or 0.0),
        )
        order = late + early if strength >= 0.5 else early + late
        return order + (-float(metrics.get("reward", item.get("reward", 0.0)) or 0.0), str(item.get("candidate_id", "")))

    def _select_vote_pareto_beam(
        self,
        evaluated: List[Dict[str, Any]],
        beam_size: int,
        current_prompt: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Select a retained beam using feasibility, Pareto fronts, then deterministic crowding."""
        feasible: List[Dict[str, Any]] = []
        for item in evaluated:
            metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
            accuracy_passed, invalid_passed, is_feasible = self._vote_pareto_feasibility(metrics)
            metrics.update(
                {
                    "accuracy_guard_passed": accuracy_passed,
                    "invalid_guard_passed": invalid_passed,
                    "error_dependence_guard_passed": bool(metrics.get("error_dependence_guard_passed", True)),
                }
            )
            item["metrics"] = metrics
            item["pareto_feasible"] = is_feasible
            item["pareto_rank"] = None
            item["pareto_crowding_distance"] = None
            item["pareto_selected"] = False
            item["pareto_forced_fallback"] = False
            if is_feasible:
                feasible.append(item)

        original_feasible_count = len(feasible)
        forced_fallback = False
        if not feasible:
            current_hash = self._hash(current_prompt)
            fallback = next((item for item in evaluated if self._hash(str(item.get("prompt", ""))) == current_hash), None)
            if fallback is None:
                raise RuntimeError("Vote Pareto selection requires the current active prompt in the candidate pool")
            fallback["pareto_feasible"] = True
            fallback["pareto_forced_fallback"] = True
            feasible = [fallback]
            forced_fallback = True

        include_competence = self._uses_competence_depth_pareto_selection()
        include_boundary_error = self._uses_vote_error_pareto_selection()
        fronts_by_item = competence_non_dominated_sort(feasible) if include_competence else non_dominated_sort(
            feasible, include_boundary_error=include_boundary_error
        )
        retained: List[Dict[str, Any]] = []
        for rank, front_indices in enumerate(fronts_by_item):
            distances = compute_crowding_distances(
                feasible, front_indices, include_boundary_error=include_boundary_error,
                include_competence_depth=include_competence,
            )
            front = []
            for index in front_indices:
                item = feasible[index]
                item["pareto_rank"] = rank
                item["pareto_crowding_distance"] = distances.get(index, 0.0)
                front.append(item)
            slots = beam_size - len(retained)
            if slots <= 0:
                continue
            if len(front) <= slots:
                retained.extend(sorted(front, key=lambda item: str(item.get("candidate_id", ""))))
            else:
                if include_competence:
                    retained.extend(sorted(front, key=lambda item: (
                        -float(item.get("pareto_crowding_distance", 0.0) or 0.0),
                        *self._competence_depth_sort_key(item),
                    ))[:slots])
                else:
                    retained.extend(sorted(front, key=self._vote_pareto_crowding_sort_key)[:slots])
                break

        if not retained:
            raise RuntimeError("Vote Pareto selection produced an empty beam")
        retained.sort(key=self._competence_depth_sort_key if include_competence else self._vote_pareto_active_sort_key)
        for item in retained:
            item["pareto_selected"] = True
        for item in evaluated:
            metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
            metrics["pareto_rank"] = item.get("pareto_rank")
            metrics["pareto_crowding_distance"] = item.get("pareto_crowding_distance")
            metrics["pareto_feasible"] = bool(item.get("pareto_feasible", False))
            metrics["pareto_selected"] = bool(item.get("pareto_selected", False))
            item["metrics"] = metrics
        return retained, {
            "num_pareto_feasible": int(original_feasible_count),
            "num_pareto_infeasible": int(len(evaluated) - original_feasible_count),
            "num_pareto_fronts": int(len(fronts_by_item)),
            "pareto_front0_size": int(len(fronts_by_item[0])) if fronts_by_item else 0,
            "pareto_forced_current_fallback": bool(forced_fallback),
        }

    def _apply_hybrid_soft_guards(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        raw_reward = float(metrics.get("reward", 0.0) or 0.0)
        pivotal_excess = max(
            0.0,
            float(metrics.get("pivotal_loss_rate", 0.0) or 0.0)
            - float(getattr(self.cfg, "pivotal_loss_guard_epsilon", 0.0) or 0.0),
        )
        shared_excess = max(
            0.0,
            float(metrics.get("shared_error_creation_score", 0.0) or 0.0)
            - float(metrics.get("shared_error_rescue_score", 0.0) or 0.0)
            - float(getattr(self.cfg, "shared_error_creation_epsilon", 0.02) or 0.0),
        )
        error_dependence_excess = pivotal_excess + shared_excess
        cycle_excess = 0.0
        if int(metrics.get("behavior_cycle_overlap", 0) or 0) >= int(getattr(self.cfg, "behavior_cycle_min_overlap", 16)):
            cycle_excess = max(
                0.0,
                float(metrics.get("max_behavior_cycle_similarity", 0.0) or 0.0)
                - float(getattr(self.cfg, "behavior_cycle_similarity_threshold", 0.95) or 0.95),
            )
        mechanism_shift_excess = max(
            0.0,
            float(metrics.get("prompt_change_ratio", 0.0) or 0.0)
            - float(getattr(self.cfg, "prompt_max_change_ratio", 0.45) or 0.45),
        )
        if metrics.get("mechanism_contract_passed") is False:
            mechanism_shift_excess = max(mechanism_shift_excess, 1.0)
        if self._is_stable_qd_lineage():
            cycle_excess = 0.0
            mechanism_shift_excess = 0.0
        mild_accuracy_regression = min(
            float(getattr(self.cfg, "catastrophic_target_accuracy_loss_epsilon", 0.05) or 0.05),
            max(0.0, -float(metrics.get("accuracy_delta", 0.0) or 0.0)),
        )
        components = {
            "soft_error_dependence_penalty": float(getattr(self.cfg, "soft_guard_error_dependence_weight", 0.5)) * error_dependence_excess,
            "soft_cycle_penalty": float(getattr(self.cfg, "soft_guard_cycle_weight", 0.2)) * cycle_excess,
            "soft_mechanism_shift_penalty": float(getattr(self.cfg, "soft_guard_mechanism_shift_weight", 0.2)) * mechanism_shift_excess,
            "soft_accuracy_regression_penalty": float(getattr(self.cfg, "soft_guard_accuracy_regression_weight", 0.5)) * mild_accuracy_regression,
        }
        penalty = sum(components.values())
        soft_reasons = []
        if error_dependence_excess > 0.0:
            soft_reasons.append("error_dependence")
        if cycle_excess > 0.0:
            soft_reasons.append("residual_cycle")
        if mechanism_shift_excess > 0.0:
            soft_reasons.append("mechanism_shift")
        if mild_accuracy_regression > 0.0:
            soft_reasons.append("mild_accuracy_regression")
        trajectory_reason = str(metrics.get("rejection_reason", ""))
        trajectory_soft_reasons = {
            "behavior_cycle", "accepted_state_cycle", "rejected_failure_cycle",
            "unsupported_large_prompt_shift", "mechanism_contract_missing",
        }
        if self._is_stable_qd_lineage():
            trajectory_soft_reasons.remove("mechanism_contract_missing")
        if trajectory_reason in trajectory_soft_reasons:
            if trajectory_reason not in soft_reasons:
                soft_reasons.append(trajectory_reason)
            metrics["rejection_reason"] = ""
        metrics.update({
            "raw_reward": raw_reward,
            "soft_guard_penalty": float(penalty),
            "penalized_reward": raw_reward - penalty,
            "soft_guard_reasons": soft_reasons,
            "hard_guard_passed": not bool(metrics.get("rejection_reason", "")),
            **components,
        })
        return metrics

    def _select_hybrid_beam(
        self,
        evaluated: List[Dict[str, Any]],
        beam_size: int,
        current_prompt: str,
        *,
        agent_id: Optional[int] = None,
        epoch_id: Optional[int] = None,
        step_id: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self._is_stable_qd_lineage():
            for item in evaluated:
                self._attach_stable_mechanism_representation(item)
                item["prompt_hash"] = self._normalized_prompt_hash(str(item.get("prompt", "")))
            retained, summary = select_quality_diversity_archive(
                evaluated, beam_size, self._normalized_prompt_hash(current_prompt), self.cfg
            )
            for item in evaluated:
                item["pareto_selected"] = item in retained
            self.quality_diversity_archive_history.append({
                "event": "quality_diversity_archive", "agent_id": agent_id,
                "epoch": epoch_id, "step": step_id, "niche_count": summary["niche_count"],
                "retained_prompt_hashes": [item.get("prompt_hash", "") for item in retained],
                "retained_sources": [item.get("beam_slot", "") for item in retained],
                "retained_niches": [item.get("qd_niche_key", item.get("metrics", {}).get("qd_niche_key", [])) for item in retained],
            })
            return retained, summary
        _, summary = self._select_vote_pareto_beam(evaluated, len(evaluated), current_prompt)
        current_hash = self._normalized_prompt_hash(current_prompt)
        safe = next(
            (item for item in evaluated if self._normalized_prompt_hash(str(item.get("prompt", ""))) == current_hash),
            None,
        )
        hard_pass = [item for item in evaluated if bool(item.get("pareto_feasible", False))]
        exploit_pool = [item for item in hard_pass if int(item.get("pareto_rank", 999) or 0) == 0 and item is not safe]
        exploit = max(
            exploit_pool,
            key=lambda item: (float(item.get("metrics", {}).get("penalized_reward", item.get("reward", 0.0)) or 0.0), str(item.get("candidate_id", ""))),
            default=None,
        )
        explore_pool = []
        for item in hard_pass:
            if item is safe or item is exploit:
                continue
            if self._candidate_pool_source(item) != "optimizer" and str(item.get("metrics", {}).get("beam_slot", "")) != "explore":
                continue
            metrics = item.get("metrics", {})
            evidence = float(metrics.get("accuracy_delta", 0.0) or 0.0) >= 0.0 or (
                float(metrics.get("depth1_net_delta", 0.0) or 0.0)
                + float(metrics.get("depth2_net_delta", 0.0) or 0.0)
            ) >= 0.0
            if evidence and float(metrics.get("mechanism_signature_distance", 0.0) or 0.0) > 0.0:
                explore_pool.append(item)
        explore = max(
            explore_pool,
            key=lambda item: (
                float(item.get("metrics", {}).get("penalized_reward", item.get("reward", 0.0)) or 0.0)
                + float(item.get("metrics", {}).get("mechanism_novelty_bonus", 0.0) or 0.0),
                str(item.get("candidate_id", "")),
            ),
            default=None,
        )
        retained: List[Dict[str, Any]] = []
        for item, slot in ((exploit, "exploit"), (safe, "safe"), (explore, "explore")):
            if item is None or item in retained:
                continue
            item["beam_slot"] = slot
            retained.append(item)
        for item in sorted(
            hard_pass,
            key=lambda row: -float(row.get("metrics", {}).get("penalized_reward", row.get("reward", 0.0)) or 0.0),
        ):
            if len(retained) >= beam_size:
                break
            if item not in retained:
                item["beam_slot"] = "exploit" if not retained else "safe_fill"
                retained.append(item)
        retained = retained[:beam_size]
        for item in evaluated:
            if item not in retained:
                item["beam_slot"] = "not_retained"
                item["pareto_selected"] = False
            else:
                item["pareto_selected"] = True
        summary.update({
            "safe_slot_occupancy": int(any(item.get("beam_slot") == "safe" for item in retained)),
            "exploit_slot_occupancy": int(any(item.get("beam_slot") == "exploit" for item in retained)),
            "explore_slot_occupancy": int(any(item.get("beam_slot") == "explore" for item in retained)),
        })
        return retained, summary

    def _attach_stable_mechanism_representation(self, item: Dict[str, Any]) -> Dict[str, Any]:
        metrics = item.setdefault("metrics", {})
        steps = metrics.get("mechanism_steps", metrics.get("mechanism_signature", []))
        representation = normalize_mechanism_representation(str(item.get("prompt", "")), steps)
        cache_key = representation["mechanism_hash"]
        vector = self.mechanism_embedding_cache.get(cache_key)
        if vector is not None:
            self.mechanism_embedding_cache_hit_count = int(getattr(self, "mechanism_embedding_cache_hit_count", 0)) + 1
        if vector is None and representation["mechanism_embedding_text"]:
            self.mechanism_embedding_cache_miss_count = int(getattr(self, "mechanism_embedding_cache_miss_count", 0)) + 1
            model = self._load_embedding_model()
            encoded = model.encode([representation["mechanism_embedding_text"]], normalize_embeddings=True)
            vector = self._normalize_vector(np.asarray(encoded)[0])
            self.mechanism_embedding_cache[cache_key] = list(vector)
        representation["mechanism_embedding"] = list(vector or [])
        metrics["mechanism_representation"] = representation
        metrics["normalized_operation_sequence"] = list(representation["normalized_operation_sequence"])
        return representation

    def _empty_cost_summary(self) -> Dict[str, Any]:
        return {
            "solver_calls": 0,
            "optimizer_calls": 0,
            "evaluator_calls": 0,
            "total_llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "latency_seconds": 0.0,
            "candidate_eval_solver_api_calls": 0,
            "candidate_eval_cache_hits": 0,
            "candidate_eval_inflight_reuses": 0,
            "candidate_eval_calls_saved_vs_naive": 0,
            "candidate_eval_prompt_dedup_savings": 0,
            "full_probe_cache_hits": 0,
            "full_probe_missing_pair_evaluations": 0,
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0,
        }

    def _client_role_from_stage(self, stage: str, client_role: str) -> str:
        role = str(client_role or "").strip().lower()
        if role in {"solver", "optimizer"}:
            return role
        if "optimizer" in str(stage or "").lower():
            return "optimizer"
        return "evaluator"

    def _estimate_tokens(self, text: str) -> int:
        text = str(text or "")
        if not text:
            return 0
        words = len(re.findall(r"\S+", text))
        chars = len(text)
        return max(1, max(words, int(chars / 4)))

    def _usage_value(self, usage: Any, key: str, default: int = 0) -> int:
        if usage is None:
            return int(default)
        value = usage.get(key, default) if isinstance(usage, dict) else getattr(usage, key, default)
        try:
            return int(value or default)
        except Exception:
            return int(default)

    def _record_llm_call(
        self,
        *,
        stage: str,
        client_role: str,
        model: str,
        temperature: float,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
        success: bool,
        error_type: str = "",
        audit_context: Optional[Mapping[str, Any]] = None,
    ):
        role = self._client_role_from_stage(stage, client_role)
        prompt_tokens = int(max(0, prompt_tokens))
        completion_tokens = int(max(0, completion_tokens))
        total_tokens = prompt_tokens + completion_tokens
        latency_seconds = float(max(0.0, latency_seconds))
        context = dict(TCS_AUDIT_CONTEXT.get() or {})
        context.update(dict(audit_context or {}))
        stage_name = str(context.get("llm_call_stage", "") or self._normalize_llm_call_stage(stage))
        model_role = str(context.get("model_role", "") or self._model_role_for_client_role(role))
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": str(stage or ""),
            "llm_call_stage": stage_name,
            "optimizer_architecture": str(context.get("optimizer_architecture", getattr(self.cfg, "optimizer_architecture", "")) or ""),
            "epoch": context.get("epoch"),
            "step": context.get("step"),
            "agent_id": context.get("agent_id"),
            "parent_id": context.get("parent_id"),
            "teacher_critic_round": context.get("teacher_critic_round"),
            "tcs_call_group_id": str(context.get("tcs_call_group_id", "") or ""),
            "execution_session_id": str(context.get("execution_session_id", getattr(self, "execution_session_id", "")) or getattr(self, "execution_session_id", "")),
            "update_attempt_id": str(context.get("update_attempt_id", "") or ""),
            "model_role": model_role,
            "model_name": str(model or ""),
            "client_role": role,
            "model": str(model or ""),
            "temperature": float(temperature),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": latency_seconds,
            "success": bool(success),
            "call_succeeded": bool(success),
            "response_empty": bool(context.get("response_empty", False)),
            "error_type": str(error_type or ""),
        }
        self.llm_call_logs.append(row)

        summary = self.cost_summary
        summary[f"{role}_calls"] = int(summary.get(f"{role}_calls", 0) or 0) + 1
        summary["total_llm_calls"] = int(summary.get("total_llm_calls", 0) or 0) + 1
        summary["prompt_tokens"] = int(summary.get("prompt_tokens", 0) or 0) + prompt_tokens
        summary["completion_tokens"] = int(summary.get("completion_tokens", 0) or 0) + completion_tokens
        summary["total_tokens"] = int(summary.get("total_tokens", 0) or 0) + total_tokens
        summary["estimated_cost"] = float(summary.get("estimated_cost", 0.0) or 0.0)
        summary["latency_seconds"] = float(summary.get("latency_seconds", 0.0) or 0.0) + latency_seconds

        if len(self.llm_call_logs) >= 20:
            self.flush_llm_call_logs()
            self.write_cost_summary()

    @staticmethod
    def _model_role_for_client_role(client_role: str) -> str:
        return {"optimizer": "optimizer", "evaluator": "evaluator", "solver": "agent"}.get(str(client_role), "evaluator")

    @staticmethod
    def _normalize_llm_call_stage(stage: str) -> str:
        lowered = str(stage or "").lower()
        if "teacher_rewrite" in lowered:
            return "teacher_rewrite"
        if "teacher_critic" in lowered:
            return "critic"
        if lowered.startswith("teacher_"):
            return "teacher"
        if "student_json_retry" in lowered:
            return "student_json_retry"
        if "student_json_repair" in lowered:
            return "student_json_repair"
        if "student_" in lowered:
            return "student"
        if "solver" in lowered:
            return "solver"
        return "one_shot_optimizer" if "optimizer" in lowered else lowered

    def _default_prompt_bank(self) -> List[str]:
        if str(self.cfg.task_type).lower() == "mmlu":
            return [
                "Use a concept-first procedure: name the tested concept, map it to the options, then choose one final answer.",
                "Use a contradiction-checking procedure: state a quick inconsistency test, apply it to plausible options, then choose one final answer.",
                "Use a boundary-and-scope procedure: inspect qualifiers, exceptions, and scope before comparing options and choosing.",
                "Use a backward-validation procedure: test what must be true if each plausible option were correct, then choose.",
                "Use an evidence-alignment procedure: tie the decision to specific clues in the stem before choosing.",
                "Use a mechanism-first procedure: explain the underlying rule or mechanism before selecting an option.",
            ]
        return [
            "Use an equation-first procedure: define variables, derive equations, solve, then check units or constraints.",
            "Use a backward-checking procedure: solve, then verify by substitution or reverse reasoning.",
            "Use a decomposition procedure: split the problem into sub-results, solve each, then combine carefully.",
            "Use a boundary-case procedure: check hidden assumptions, off-by-one cases, and impossible values before finalizing.",
            "Use a representation procedure: create a compact table, relation, or diagram in words before computing.",
            "Use an invariant procedure: track totals, conserved quantities, or repeated structure before calculating.",
        ]

    def _build_initial_prompts(self) -> List[str]:
        if self.cfg.agents <= 0:
            return []
        if str(self.cfg.init_mode).lower() == "bank":
            return [self.initial_prompt_bank[i % len(self.initial_prompt_bank)] for i in range(self.cfg.agents)]
        return [self.cfg.shared_prompt for _ in range(self.cfg.agents)]

    def _hash(self, value: str) -> str:
        return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]

    def _solver_cache_settings(self) -> Dict[str, Any]:
        return {
            "task_type": str(self.cfg.task_type),
            "agent_model": str(self.cfg.agent_model),
            "temperature": float(self.cfg.temperature),
            "max_tokens": int(self.cfg.max_tokens),
        }

    def _solver_rollout_cache_key_from_hashes(self, question_hash: str, prompt_hash: str, agent_id: int) -> str:
        settings = self._solver_cache_settings()
        return self._hash(
            "|".join(
                [
                    str(int(agent_id)),
                    str(settings["task_type"]),
                    str(settings["agent_model"]),
                    f"{float(settings['temperature']):.8g}",
                    str(settings["max_tokens"]),
                    str(question_hash),
                    str(prompt_hash),
                ]
            )
        )

    def _solver_rollout_cache_key(self, question_hash: str, prompt: str, agent_id: int) -> str:
        return self._solver_rollout_cache_key_from_hashes(question_hash, self._hash(prompt), agent_id)

    def _record_solver_rollout(
        self,
        question_hash: str,
        prompt: str,
        trace: str,
        answer: str,
        agent_id: Optional[int] = None,
        source: str = "",
        prompt_hash: Optional[str] = None,
    ):
        qh = str(question_hash or "").strip()
        ph = str(prompt_hash or self._hash(prompt)).strip()
        if agent_id is None:
            return
        try:
            aid = int(agent_id)
        except Exception:
            return
        if aid < 0 or not qh or not ph:
            return
        key = self._solver_rollout_cache_key_from_hashes(qh, ph, aid)
        row = {
            **self._solver_cache_settings(),
            "question_hash": qh,
            "prompt_hash": ph,
            "agent_id": aid,
            "trace": str(trace or ""),
            "answer": str(answer or ""),
            "source": str(source or ""),
            "cache_origin": "current_run",
        }
        self._add_solver_rollout_cache_row(row)
        self._append_solver_rollout_record(row)

    def _add_solver_rollout_cache_row(self, row: Dict[str, Any]):
        try:
            qh = str(row.get("question_hash", "")).strip()
            ph = str(row.get("prompt_hash", "")).strip()
            aid = int(row.get("agent_id", -1))
        except Exception:
            return
        if aid < 0 or not qh or not ph:
            return
        key = self._solver_rollout_cache_key_from_hashes(qh, ph, aid)
        normalized = dict(row)
        normalized.setdefault("cache_origin", "current_run")
        self.solver_rollout_cache.setdefault(key, []).append(normalized)

    def _record_solver_rollouts(
        self,
        question_hash: str,
        prompts: List[str],
        traces: List[str],
        answers: List[str],
        source: str,
    ):
        for i, prompt in enumerate(prompts):
            if i >= len(traces) or i >= len(answers):
                continue
            self._record_solver_rollout(
                question_hash=question_hash,
                prompt=str(prompt),
                trace=str(traces[i]),
                answer=str(answers[i]),
                agent_id=i,
                source=source,
            )

    def _lookup_solver_rollout(self, question_hash: str, prompt: str, agent_id: int) -> Optional[Dict[str, Any]]:
        try:
            aid = int(agent_id)
        except Exception:
            return None
        key = self._solver_rollout_cache_key(question_hash, prompt, aid)
        cached = self.solver_rollout_cache.get(key)
        if not isinstance(cached, list) or not cached:
            return None
        return dict(cached[-1])

    def _append_solver_rollout_record(self, row: Dict[str, Any]):
        if not self.cfg.candidate_reuse_recorded_rollouts:
            return
        path = os.path.join(self.cfg.out_dir, "solver_rollout_records.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            return

    def _existing_run_meta_matches_solver_cache(self) -> bool:
        meta_path = os.path.join(self.cfg.out_dir, "run_meta.json")
        if not os.path.exists(meta_path):
            return True
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return False
        cfg = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
        if not cfg:
            return False
        checks = {
            "task_type": str(self.cfg.task_type),
            "agent_model": str(self.cfg.agent_model),
            "max_tokens": int(self.cfg.max_tokens),
        }
        for key, expected in checks.items():
            if str(cfg.get(key, "")) != str(expected):
                return False
        try:
            return abs(float(cfg.get("temperature", 0.0)) - float(self.cfg.temperature)) < 1e-12
        except Exception:
            return False

    def _iter_recorded_rollout_files(self) -> List[str]:
        out_dir = str(self.cfg.out_dir)
        names = ["solver_rollout_records.jsonl", "train_trace_history.jsonl", "test_trace_history.jsonl"]
        paths = [os.path.join(out_dir, name) for name in names]
        if os.path.isdir(out_dir):
            for name in sorted(os.listdir(out_dir)):
                if name.endswith("_predictions.jsonl") or (name.startswith("val_epoch") and name.endswith("_predictions.jsonl")):
                    paths.append(os.path.join(out_dir, name))
        seen = set()
        deduped = []
        for path in paths:
            if path not in seen and os.path.exists(path):
                seen.add(path)
                deduped.append(path)
        return deduped

    def _load_recorded_solver_rollouts(self):
        if not self.cfg.candidate_reuse_recorded_rollouts:
            return
        if not self._existing_run_meta_matches_solver_cache():
            return
        loaded = 0
        for path in self._iter_recorded_rollout_files():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(row, dict):
                            continue
                        if "prompt_hash" in row and "trace" in row and "answer" in row and "agent_id" in row:
                            persisted = dict(row)
                            persisted["cache_origin"] = "persisted"
                            self._add_solver_rollout_cache_row(persisted)
                            loaded += 1
                            continue
                        qh = str(row.get("question_hash", "")).strip()
                        agents = row.get("agents", [])
                        if not qh or not isinstance(agents, list):
                            continue
                        for agent in agents:
                            if not isinstance(agent, dict):
                                continue
                            prompt_hash = str(agent.get("prompt_hash", "")).strip()
                            if not prompt_hash:
                                continue
                            try:
                                agent_id = int(agent.get("agent_id", -1))
                            except Exception:
                                continue
                            self._add_solver_rollout_cache_row(
                                {
                                    **self._solver_cache_settings(),
                                    "question_hash": qh,
                                    "prompt_hash": prompt_hash,
                                    "agent_id": agent_id,
                                    "trace": str(agent.get("trace", "")),
                                    "answer": str(agent.get("answer", "")),
                                    "source": os.path.basename(path),
                                    "cache_origin": "persisted",
                                }
                            )
                            loaded += 1
            except Exception:
                continue
        if loaded and self.cfg.llm_call_logging:
            print(f"[solver-reuse] loaded recorded rollouts={loaded} unique_keys={len(self.solver_rollout_cache)}", flush=True)

    def _initialize_prompt_beams(self):
        for agent in self.agents:
            incumbent = self._make_beam_item(agent.current_prompt, None, {}, None, 0)
            incumbent.update({"is_incumbent": True, "archive_bucket": "safe"})
            agent.safe_qd_archive = [dict(incumbent)]
            agent.prompt_beam = [incumbent]

    def _make_beam_item(
        self,
        prompt: str,
        score: Optional[float],
        metrics: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        generation: int = 0,
        candidate_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        gen = int(generation)
        prompt_text = str(prompt)
        return {
            "id": candidate_id or f"g{gen}_{self._hash(prompt_text)}",
            "prompt": prompt_text,
            "score": None if score is None else float(score),
            "metrics": dict(metrics or {}),
            "parent_id": parent_id,
            "generation": gen,
            "prompt_hash": self._normalized_prompt_hash(prompt_text),
        }

    def _refresh_joint_representatives(self, agent: AgentState) -> None:
        archive = list(getattr(agent, "safe_qd_archive", []) or [])
        if not archive:
            archive = [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]
        agent.prompt_beam = [dict(item) for item in select_joint_representatives(
            archive, self._normalized_prompt_hash(agent.current_prompt), int(self.cfg.joint_representative_beam_size),
        )] or [dict(archive[0])]

    def _expire_probation_branches(self, epoch_id: int) -> int:
        expired = 0
        for agent in self.agents:
            expired += self._expire_agent_probation_branches(agent)
        self.probation_expired_count += expired
        return expired

    def _expire_agent_probation_branches(self, agent: AgentState) -> int:
        retained = []
        update_count = sum(int(value or 0) for value in agent.optimizer_update_count_by_epoch.values())
        expired = 0
        for item in getattr(agent, "probation_archive", []):
            born = int(item.get("probation_created_update", update_count) or update_count)
            if update_count - born >= int(self.cfg.probation_archive_ttl_updates):
                expired += 1
            else:
                retained.append(item)
        agent.probation_archive = retained
        return expired

    def expire_probation_branches(self, epoch_id: int) -> int:
        """Public epoch-end hook; TTL is measured in each agent's update turns."""
        return self._expire_probation_branches(epoch_id)

    def _current_joint_change_limit(self, epoch: int) -> int:
        early = float(self.specialization_strength) < float(self.cfg.joint_team_change_limit_switch_strength)
        base = int(self.cfg.joint_team_max_active_changes_early if early else self.cfg.joint_team_max_active_changes_late)
        return base + int(self.cfg.joint_team_change_limit_relaxation) if int(getattr(self, "qd_change_limit_relaxed_epoch", -1)) == int(epoch) else base

    def _select_stable_qd_parents(self, agent: AgentState, epoch_id: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Keep active exploitation while guaranteeing archived niches reproduction turns."""
        expired = self._expire_agent_probation_branches(agent)
        self.probation_expired_count += expired
        active_hash = self._normalized_prompt_hash(agent.current_prompt)
        active = next(
            (item for item in getattr(agent, "safe_qd_archive", []) if str(item.get("prompt_hash", "")) == active_hash),
            self._make_beam_item(agent.current_prompt, None, {}, None, 0),
        )
        alternate, source, niche = select_reproduction_parent(
            active,
            getattr(agent, "safe_qd_archive", []),
            getattr(agent, "probation_archive", []),
            agent.per_niche_parent_count,
            epoch=int(epoch_id),
            min_opportunities=int(self.cfg.qd_niche_min_parent_opportunities_per_epoch),
            allow_probation=bool(self.cfg.probation_parent_enabled),
        )
        parents, sources = [active], ["active"]
        if alternate is not None and str(alternate.get("prompt_hash", "")) != str(active.get("prompt_hash", "")):
            parents.append(alternate)
            sources.append(source)
            key = f"{int(epoch_id)}:{niche}"
            agent.per_niche_parent_count[key] = int(agent.per_niche_parent_count.get(key, 0) or 0) + 1
            if source == "probation_niche":
                agent.probation_parent_count += 1
        return parents, sources

    def _mark_mechanism_novelty(
        self,
        item: Dict[str, Any],
        *,
        parent: Optional[Dict[str, Any]],
        existing: Sequence[Dict[str, Any]],
    ) -> bool:
        novel = mechanism_is_novel(
            item,
            parent,
            existing,
            near_duplicate_threshold=float(self.cfg.mechanism_near_duplicate_similarity_threshold),
        )
        item.setdefault("metrics", {})["mechanism_novel"] = bool(novel)
        return bool(novel)

    def _active_prompt_list(self) -> List[str]:
        prompts = []
        for agent in self.agents:
            beam = getattr(agent, "prompt_beam", [])
            if beam and isinstance(beam[0], dict):
                prompts.append(str(beam[0].get("prompt", agent.current_prompt)))
            else:
                prompts.append(str(agent.current_prompt))
        return prompts

    def _base_log_fields(self) -> Dict[str, Any]:
        requested_aggregation_mode = str(getattr(self.cfg, "aggregation_mode", "majority") or "majority")
        fields = {
            "execution_session_id": self._current_execution_session_id(),
            "comparison_task_id": getattr(self.cfg, "comparison_task_id", ""),
            "setting": getattr(self.cfg, "experiment_setting", ""),
            "benchmark": getattr(self.cfg, "benchmark", ""),
            "answer_format": getattr(self.cfg, "answer_format", ""),
            "task_type": self.cfg.task_type,
            "dataset_format": getattr(self.cfg, "dataset_format", ""),
            "agent_model": self.cfg.agent_model,
            "optimizer_model": self.cfg.optimizer_model,
            "evaluator_model": self.cfg.evaluator_model,
            "search_mode": self.cfg.search_mode,
            "reward_mode": self.cfg.reward_mode,
            "diversity_metric": self.cfg.diversity_metric,
            "embedding_model": self.cfg.embedding_model,
            "aggregation_mode": requested_aggregation_mode,
            "requested_aggregation_mode": requested_aggregation_mode,
            "effective_aggregation_mode": canonical_aggregation_mode(requested_aggregation_mode),
        }
        if bool(getattr(self.cfg, "competence_depth_enabled", False)):
            fields["plurality_boundary_version"] = PLURALITY_BOUNDARY_VERSION
        if self._is_v82_hybrid():
            fields.update({
                "method_version": str(getattr(self.cfg, "method_version", "legacy")),
                "competence_schedule_version": str(getattr(self.cfg, "competence_schedule_version", "legacy")),
                "target_selector_version": str(getattr(self.cfg, "target_selector_version", "legacy")),
                "beam_policy_version": str(getattr(self.cfg, "beam_policy_version", "legacy")),
                "tcs_candidate_policy_version": str(getattr(self.cfg, "tcs_candidate_policy_version", "legacy")),
                "mechanism_signature_version": str(getattr(self.cfg, "mechanism_signature_version", "legacy")),
            })
            if self._is_stable_qd_lineage():
                fields.update({
                    "active_team_selector_version": str(self.cfg.active_team_selector_version),
                    "lineage_policy_version": str(self.cfg.lineage_policy_version),
                    "mechanism_distance_version": str(self.cfg.mechanism_distance_version),
                    "joint_active_team_selection_enabled": True,
                    "quality_diversity_archive_enabled": True,
                    "probation_archive_enabled": bool(self.cfg.probation_archive_enabled),
                    "candidate_refill_enabled": bool(self.cfg.candidate_refill_enabled),
                    "early_self_drift_disabled": True,
                    "behavior_diversity_primary": True,
                    "mechanism_embedding_secondary": True,
                    "candidate_refill_version": str(self.cfg.candidate_refill_version),
                    "archive_policy_version": str(self.cfg.archive_policy_version),
                    "joint_quality_filter_version": str(self.cfg.joint_quality_filter_version),
                    "probe_stability_version": str(self.cfg.probe_stability_version),
                    "parent_selection_version": str(self.cfg.parent_selection_version),
                })
        return fields

    def _read_previous_execution_session_id(self) -> str:
        path = os.path.join(self.cfg.out_dir, "run_meta.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return ""
        return str(payload.get("execution_session_id", "") or "") if isinstance(payload, dict) else ""

    def _current_execution_session_id(self) -> str:
        return str(getattr(self, "execution_session_id", "") or "")

    def _update_attempt_id(self, epoch_id: int, step_id: int, agent_id: int) -> str:
        return f"{self._current_execution_session_id()}_e{int(epoch_id)}_s{int(step_id)}_a{int(agent_id)}"

    def _tcs_call_group_id(self, update_attempt_id: str, parent_id: str, parent_prompt: str) -> str:
        return (
            f"{update_attempt_id}_p{self._hash(str(parent_id))}_"
            f"{self._hash(str(parent_prompt))}"
        )

    @staticmethod
    def _git_provenance() -> Dict[str, Any]:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                # Generated run directories are often untracked. Dirty here
                # means tracked source/configuration changes, not fresh output.
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            return {"git_commit": commit, "git_dirty": bool(status.strip())}
        except (OSError, subprocess.SubprocessError):
            return {"git_commit": "", "git_dirty": None}

    def _split_integrity_metadata(self) -> Dict[str, Any]:
        raw = str(getattr(self.cfg, "split_integrity_json", "") or "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {"parse_error": "invalid_split_integrity_json"}
        return payload if isinstance(payload, dict) else {"parse_error": "split_integrity_json_is_not_an_object"}

    def write_run_meta(self):
        provenance = self._git_provenance()
        meta = {
            **self._base_log_fields(),
            "comparison_task_id": getattr(self.cfg, "comparison_task_id", ""),
            "benchmark": getattr(self.cfg, "benchmark", ""),
            "answer_format": getattr(self.cfg, "answer_format", ""),
            "dataset_format": getattr(self.cfg, "dataset_format", ""),
            "init_mode": self.cfg.init_mode,
            "agents": self.cfg.agents,
            "epochs": self.cfg.epochs,
            "train_size": self.cfg.train_size,
            "val_size": self.cfg.val_size,
            "test_size": self.cfg.test_size,
            "update_every": self.cfg.update_every,
            "beam_size": self.cfg.beam_size,
            "candidate_eval_batch_size": self.cfg.candidate_eval_batch_size,
            "candidate_eval_strategy": self.cfg.candidate_eval_strategy,
            "candidate_eval_data_source": str(getattr(self.cfg, "candidate_eval_data_source", "optimization_train")),
            "candidate_eval_repeats": self.cfg.candidate_eval_repeats,
            "candidate_eval_execution_mode": getattr(self.cfg, "candidate_eval_execution_mode", "legacy"),
            "solver_rollout_singleflight": bool(getattr(self.cfg, "solver_rollout_singleflight", True)),
            "candidate_eval_prompt_dedup": bool(getattr(self.cfg, "candidate_eval_prompt_dedup", True)),
            "candidate_eval_cache_logging": bool(getattr(self.cfg, "candidate_eval_cache_logging", True)),
            "candidate_selection_mode": getattr(self.cfg, "candidate_selection_mode", "scalar_reward"),
            "best_state_selection_mode": getattr(self.cfg, "best_state_selection_mode", "vote_first"),
            "method_version": str(getattr(self.cfg, "method_version", "legacy")),
            "target_selector_mode": str(getattr(self.cfg, "target_selector_mode", "legacy")),
            "target_selector_version": str(getattr(self.cfg, "target_selector_version", "legacy")),
            "beam_policy_version": str(getattr(self.cfg, "beam_policy_version", "legacy")),
            "tcs_candidate_policy_version": str(getattr(self.cfg, "tcs_candidate_policy_version", "legacy")),
            "mechanism_signature_version": str(getattr(self.cfg, "mechanism_signature_version", "legacy")),
            "optimizer_architecture": getattr(self.cfg, "optimizer_architecture", ""),
            "optimizer_fallback_mode": getattr(self.cfg, "optimizer_fallback_mode", ""),
            "teacher_critic_max_rounds": getattr(self.cfg, "teacher_critic_max_rounds", 0),
            "teacher_question_pass_threshold": getattr(self.cfg, "teacher_question_pass_threshold", 0.0),
            "teacher_critic_use_voting_failure": bool(getattr(self.cfg, "teacher_critic_use_voting_failure", False)),
            "competence_schedule_mode": str(getattr(self.cfg, "competence_schedule_mode", "absolute_legacy")),
            "competence_schedule_version": str(getattr(self.cfg, "competence_schedule_version", "competence_depth_v1")),
            "competence_probe_size": int(getattr(self.cfg, "competence_probe_size", 0) or 0),
            "competence_probe_seed_offset": int(getattr(self.cfg, "competence_probe_seed_offset", 7000)),
            "competence_probe_question_hashes": list(getattr(self, "competence_probe_question_hashes", [])),
            "initial_competence_probe_metrics": dict(getattr(self, "initial_competence_probe_metrics", {})),
            "competence_relative_low_delta": float(getattr(self.cfg, "competence_relative_low_delta", 0.01)),
            "competence_relative_high_delta": float(getattr(self.cfg, "competence_relative_high_delta", 0.06)),
            "competence_schedule_ema": float(getattr(self.cfg, "competence_schedule_ema", 0.50)),
            "competence_schedule_max_step": float(getattr(self.cfg, "competence_schedule_max_step", 0.35)),
            "competence_schedule_monotonic": bool(getattr(self.cfg, "competence_schedule_monotonic", True)),
            "competence_mean_guard_epsilon": float(getattr(self.cfg, "competence_mean_guard_epsilon", 0.01)),
            "competence_c1_guard_epsilon": float(getattr(self.cfg, "competence_c1_guard_epsilon", 0.01)),
            "competence_c2_guard_epsilon": float(getattr(self.cfg, "competence_c2_guard_epsilon", 0.01)),
            "competence_depth1_candidate_guard_enabled": bool(getattr(self.cfg, "competence_depth1_candidate_guard_enabled", False)),
            "competence_depth1_candidate_guard_epsilon": float(getattr(self.cfg, "competence_depth1_candidate_guard_epsilon", 0.0)),
            "execution_session_id": self.execution_session_id,
            "previous_execution_session_id": self.previous_execution_session_id,
            "experiment_protocol_version": self._experiment_protocol_version(),
            "checkpoint_version": CHECKPOINT_VERSION,
            **provenance,
            "split_integrity": self._split_integrity_metadata(),
            "model_role_map": {
                "agent_model": "solver rollouts for train/validation/test answering",
                "optimizer_model": (
                    "prompt-evolution generator calls: one_shot optimizer, TCS Teacher, "
                    "Teacher rewrite, Student, Student JSON retry, and Student JSON repair"
                ),
                "evaluator_model": (
                    "TCS Critic and optional joint trace diversity evaluator"
                ),
                "embedding_model": "local trace-embedding encoder for diversity diagnostics",
            },
            "initial_agent_prompts": self.initial_agent_prompts,
            "initial_agent_prompt_hashes": self.initial_agent_prompt_hashes,
            "config": asdict(self.cfg),
            "framework": "accuracy_only_evolutionary_beam" if self._is_accuracy_only_mode() else "vote_oriented_evolutionary_beam",
        }
        with open(os.path.join(self.cfg.out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _init_prompt_history(self) -> Dict[str, Any]:
        return {
            str(i): {
                "initial_prompt": agent.initial_prompt,
                "initial_prompt_hash": self._hash(agent.initial_prompt),
                "current_prompt": agent.current_prompt,
                "current_prompt_hash": self._hash(agent.current_prompt),
                "prompt_beam": agent.prompt_beam,
                "events": [],
            }
            for i, agent in enumerate(self.agents)
        }

    def _append_prompt_history_event(self, agent_id: int, epoch: int, step: int, decision: str, changed: bool):
        key = str(agent_id)
        agent = self.agents[agent_id]
        self.prompt_history.setdefault(key, {"events": []})
        self.prompt_history[key]["current_prompt"] = agent.current_prompt
        self.prompt_history[key]["current_prompt_hash"] = self._hash(agent.current_prompt)
        self.prompt_history[key]["prompt_beam"] = agent.prompt_beam
        self.prompt_history[key].setdefault("events", []).append(
            {
                "epoch": epoch,
                "step": step,
                "decision": decision,
                "changed": bool(changed),
                "current_prompt_hash": self._hash(agent.current_prompt),
            }
        )

    def sync_prompt_history_current_state(
        self,
        event: str = "sync_current_state",
        epoch: Any = "final",
        step: int = 0,
        selected_epoch: Optional[int] = None,
    ):
        for agent_id, agent in enumerate(self.agents):
            key = str(agent_id)
            row = self.prompt_history.setdefault(
                key,
                {
                    "initial_prompt": getattr(agent, "initial_prompt", ""),
                    "initial_prompt_hash": self._hash(getattr(agent, "initial_prompt", "")),
                    "events": [],
                },
            )
            row["current_prompt"] = agent.current_prompt
            row["current_prompt_hash"] = self._hash(agent.current_prompt)
            row["prompt_beam"] = agent.prompt_beam
            event_row = {
                "epoch": epoch,
                "step": step,
                "decision": event,
                "changed": self._hash(agent.current_prompt) != row.get("initial_prompt_hash", ""),
                "current_prompt_hash": self._hash(agent.current_prompt),
            }
            if selected_epoch is not None:
                event_row["selected_epoch"] = int(selected_epoch)
            row.setdefault("events", []).append(event_row)

    def _contains_task_specific_content(self, prompt: str, question: Optional[str] = None) -> bool:
        text = normalize_spaces(str(prompt)).lower()
        if "final_answer:" in text:
            return True
        if question:
            q = normalize_spaces(question).lower()
            words = [w for w in re.findall(r"[a-zA-Z0-9]{4,}", q) if len(w) >= 6]
            if len(words) >= 4:
                hits = sum(1 for w in set(words) if w in text)
                if hits >= min(4, max(2, len(set(words)) // 3)):
                    return True
        return False

    def _sanitize_prompt(self, prompt: str, agent_id: int, question: Optional[str] = None) -> Tuple[str, bool]:
        if self._contains_task_specific_content(prompt, question):
            return self.agents[agent_id].initial_prompt, True
        return str(prompt).strip(), False

    def _prompt_signature(self, prompt: str) -> str:
        return normalize_spaces(str(prompt or "")).lower()

    def _is_redundant_candidate_prompt(
        self,
        parent_prompt: str,
        candidate_prompt: str,
        seen_signatures: Optional[set] = None,
        *,
        allow_substantive_parent_extension: bool = False,
    ) -> bool:
        candidate_sig = self._prompt_signature(candidate_prompt)
        if not candidate_sig:
            return True
        if seen_signatures and candidate_sig in seen_signatures:
            return True

        parent_sig = self._prompt_signature(parent_prompt)
        stock_sig = self._prompt_signature(self.GENERIC_DISTINCT_PROCEDURE)
        stock_count = candidate_sig.count(stock_sig)
        parent_stock_count = parent_sig.count(stock_sig)

        if candidate_sig == parent_sig:
            return True
        if stock_count > 1:
            return True
        if (
            parent_sig
            and candidate_sig.startswith(parent_sig)
            and len(candidate_sig) > len(parent_sig) + 40
            and not allow_substantive_parent_extension
        ):
            return True
        if stock_count > parent_stock_count and parent_stock_count > 0:
            return True
        return False

    def _empty_optimizer_generation_diagnostics(self) -> Dict[str, Any]:
        return {
            "optimizer_architecture": str(getattr(self.cfg, "optimizer_architecture", "one_shot") or "one_shot"),
            "optimizer_raw_response_empty": 0,
            "optimizer_json_parse_failed": 0,
            "optimizer_raw_candidate_count": 0,
            "optimizer_empty_prompt_count": 0,
            "optimizer_sanitized_count": 0,
            "optimizer_redundant_filtered_count": 0,
            "optimizer_schema_filtered_count": 0,
            "optimizer_final_candidate_count": 0,
            "optimizer_underfilled": False,
            "teacher_question": "",
            "tcs_call_group_id": "",
            "execution_session_id": "",
            "update_attempt_id": "",
            "teacher_question_approved": False,
            "teacher_question_rejected": False,
            "teacher_question_rejection_reason": "",
            "teacher_question_forced_best_score": False,
            "teacher_question_forced_best_round": 0,
            "teacher_question_forced_below_threshold": False,
            "teacher_question_score": 0.0,
            "teacher_critic_rounds": 0,
            "teacher_quality_critique": "",
            "teacher_specificity_critique": "",
            "teacher_task_alignment_critique": "",
            "teacher_error_alignment_critique": "",
            "teacher_diversity_critique": "",
            "teacher_rewrite_count": 0,
            "student_candidate_count_raw": 0,
            "student_candidate_count_final": 0,
            "student_candidate_filtered_count": 0,
            "student_candidate_filter_reasons": [],
            "student_all_candidates_filtered": False,
            "student_missing_required_field_count": 0,
            "student_missing_required_fields": [],
            "student_raw_response_empty": False,
            "student_raw_response_preview": "",
            "student_json_parse_failed": False,
            "student_json_parse_error": "",
            "student_json_retry_attempted": False,
            "student_json_retry_succeeded": False,
            "student_json_retry_raw_response_preview": "",
            "student_json_repair_attempted": False,
            "student_json_repair_succeeded": False,
            "student_json_repair_raw_response_preview": "",
            "student_json_repair_failure_reason": "",
            "student_json_has_candidates_key": False,
            "student_candidates_is_list": False,
            "student_candidates_empty_list": False,
            "student_refusal_or_explanation": False,
            "student_failure_stage": "",
            "num_teacher_calls": 0,
            "num_critic_calls": 0,
            "num_teacher_rewrite_calls": 0,
            "num_student_calls": 0,
            "num_student_retry_calls": 0,
            "num_student_repair_calls": 0,
        }

    def _record_optimizer_generation_diagnostics(
        self,
        agent_id: int,
        parent_id: str,
        diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = self._empty_optimizer_generation_diagnostics()
        normalized.update(diagnostics or {})
        normalized["optimizer_final_candidate_count"] = int(normalized.get("optimizer_final_candidate_count", 0) or 0)
        normalized["optimizer_underfilled"] = bool(normalized.get("optimizer_underfilled", False))
        if not hasattr(self, "optimizer_generation_diagnostics"):
            self.optimizer_generation_diagnostics = {}
        key = f"{int(agent_id)}:{str(parent_id)}"
        self.optimizer_generation_diagnostics[key] = dict(normalized)
        return normalized

    def _optimizer_generation_diagnostics_for_parent(self, agent_id: int, parent_id: str) -> Dict[str, Any]:
        if not hasattr(self, "optimizer_generation_diagnostics"):
            self.optimizer_generation_diagnostics = {}
        key = f"{int(agent_id)}:{str(parent_id)}"
        return dict(self.optimizer_generation_diagnostics.get(key, self._empty_optimizer_generation_diagnostics()))

    def _required_optimizer_fields(self, architecture: Optional[str] = None) -> List[str]:
        arch = str(architecture or getattr(self.cfg, "optimizer_architecture", "one_shot") or "one_shot").lower()
        if arch == "teacher_critic_student":
            required = [
                "candidate_prompt",
                "student_interpretation_of_question",
                "target_error_pattern",
                "accuracy_repair_rule",
                "diversity_contribution",
                "error_correlation_reduction",
                "task_alignment_rule",
                "peer_redundancy_avoidance",
                "expected_accuracy_effect",
                "expected_diversity_effect",
                "risk_control",
                "rationale",
            ]
            if self._v7_residual_protocol_enabled():
                required.extend([
                    "preserved_mechanisms",
                    "modified_mechanism",
                    "change_summary",
                    "target_residual_family",
                    "expected_shared_error_effect",
                ])
            if self._is_v82_hybrid():
                required.extend(["candidate_type", "mechanism_steps", "target_failure_buckets", "expected_effect"])
            return required
        return ["candidate_prompt"]

    def _missing_optimizer_fields(
        self,
        item: Dict[str, Any],
        architecture: Optional[str] = None,
    ) -> List[str]:
        missing = []
        for field in self._required_optimizer_fields(architecture):
            value = item.get(field, None)
            if value is None:
                missing.append(field)
                continue
            if isinstance(value, str) and not value.strip():
                missing.append(field)
                continue
            if isinstance(value, list) and len(value) == 0:
                missing.append(field)
                continue
        return missing

    def _candidate_has_required_optimizer_fields(
        self,
        item: Dict[str, Any],
        architecture: Optional[str] = None,
    ) -> bool:
        return not self._missing_optimizer_fields(item, architecture)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            if isinstance(value, bool):
                return float(value)
            if isinstance(value, (int, float)):
                return float(value)
            text = str(value).strip()
            if not text:
                return float(default)
            try:
                return float(text)
            except Exception:
                pass
            match = re.search(r"[-+]?\d*\.?\d+", text)
            if match:
                return float(match.group(0))
            return float(default)
        except Exception:
            return float(default)

    def _is_optimizer_generated_candidate_source(self, source: Any) -> bool:
        text = str(source or "").strip().lower()
        return text in {"optimizer", "teacher_critic_student"}

    @staticmethod
    def _candidate_pool_source(item: Mapping[str, Any]) -> str:
        """Return where a candidate entered the pool, with legacy checkpoint support."""
        return str(item.get("candidate_pool_source") or item.get("source") or "").strip()

    @staticmethod
    def _candidate_generation_source(item: Mapping[str, Any]) -> str:
        """Return the mechanism that generated the candidate prompt."""
        return str(item.get("candidate_source") or "").strip()

    def _teacher_metadata_from_diagnostics(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "optimizer_architecture",
            "teacher_question",
            "teacher_question_approved",
            "teacher_question_rejected",
            "teacher_question_rejection_reason",
            "teacher_question_forced_best_score",
            "teacher_question_forced_best_round",
            "teacher_question_forced_below_threshold",
            "teacher_question_score",
            "teacher_critic_rounds",
            "teacher_quality_critique",
            "teacher_specificity_critique",
            "teacher_task_alignment_critique",
            "teacher_error_alignment_critique",
            "teacher_diversity_critique",
            "teacher_rewrite_count",
            "student_candidate_count_raw",
            "student_candidate_count_final",
            "student_candidate_filtered_count",
            "student_candidate_filter_reasons",
            "student_all_candidates_filtered",
            "student_missing_required_field_count",
            "student_missing_required_fields",
            "student_raw_response_empty",
            "student_raw_response_preview",
            "student_json_parse_failed",
            "student_json_parse_error",
            "student_json_retry_attempted",
            "student_json_retry_succeeded",
            "student_json_retry_raw_response_preview",
            "student_json_repair_attempted",
            "student_json_repair_succeeded",
            "student_json_repair_raw_response_preview",
            "student_json_repair_failure_reason",
            "student_json_has_candidates_key",
            "student_candidates_is_list",
            "student_candidates_empty_list",
            "student_refusal_or_explanation",
            "student_failure_stage",
        ]
        return {key: diagnostics.get(key, self._empty_optimizer_generation_diagnostics().get(key)) for key in keys}

    @staticmethod
    def _merge_student_diagnostics(diagnostics: Dict[str, Any], student_diagnostics: Mapping[str, Any]) -> None:
        """Student defaults must not erase Teacher/Critic provenance from the same parent."""
        for key, value in student_diagnostics.items():
            if str(key).startswith("student_"):
                diagnostics[key] = value

    def _student_failure_log_fields(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        diagnostics = diagnostics or {}
        return {
            "student_raw_response_empty": bool(diagnostics.get("student_raw_response_empty", False)),
            "student_raw_response_preview": str(diagnostics.get("student_raw_response_preview", ""))[:1000],
            "student_json_parse_failed": bool(diagnostics.get("student_json_parse_failed", False)),
            "student_json_parse_error": str(diagnostics.get("student_json_parse_error", ""))[:500],
            "student_json_retry_attempted": bool(diagnostics.get("student_json_retry_attempted", False)),
            "student_json_retry_succeeded": bool(diagnostics.get("student_json_retry_succeeded", False)),
            "student_json_retry_raw_response_preview": str(diagnostics.get("student_json_retry_raw_response_preview", ""))[:1000],
            "student_json_repair_attempted": bool(diagnostics.get("student_json_repair_attempted", False)),
            "student_json_repair_succeeded": bool(diagnostics.get("student_json_repair_succeeded", False)),
            "student_json_repair_raw_response_preview": str(diagnostics.get("student_json_repair_raw_response_preview", ""))[:1000],
            "student_json_repair_failure_reason": str(diagnostics.get("student_json_repair_failure_reason", ""))[:500],
            "student_json_has_candidates_key": bool(diagnostics.get("student_json_has_candidates_key", False)),
            "student_candidates_is_list": bool(diagnostics.get("student_candidates_is_list", False)),
            "student_candidates_empty_list": bool(diagnostics.get("student_candidates_empty_list", False)),
            "student_refusal_or_explanation": bool(diagnostics.get("student_refusal_or_explanation", False)),
            "student_failure_stage": str(diagnostics.get("student_failure_stage", "")),
        }

    def _student_candidate_schema_json(self) -> str:
        prompt_limit = int(
            getattr(self.cfg, "student_candidate_prompt_hard_max_chars", 1400)
            if bool(getattr(self.cfg, "competence_depth_enabled", False))
            else getattr(self.cfg, "student_candidate_prompt_max_chars", 900)
        )
        candidate_schema = {
                    "candidate_prompt": f"standalone complete prompt, <= {prompt_limit} chars",
                    "student_interpretation_of_question": "one short sentence",
                    "target_error_pattern": "short phrase",
                    "accuracy_repair_rule": "one short sentence",
                    "diversity_contribution": "one short sentence",
                    "error_correlation_reduction": "one short sentence",
                    "task_alignment_rule": "one short sentence",
                    "peer_redundancy_avoidance": "one short sentence",
                    "expected_accuracy_effect": "one short sentence",
                    "expected_diversity_effect": "one short sentence",
                    "risk_control": "one short sentence",
                    "rationale": "one short sentence",
                    "change_summary": "optional short local-edit summary",
                    "preserved_mechanisms": ["optional preserved mechanism"],
                    "new_or_modified_mechanism": "optional changed mechanism",
        }
        if self._v7_residual_protocol_enabled():
            candidate_schema.update({
                "modified_mechanism": "one local mechanism changed in v7",
                "target_residual_family": "task-independent residual family",
                "expected_shared_error_effect": "one short sentence",
            })
        if self._is_v82_hybrid():
            candidate_schema.update({
                "candidate_type": "task_specific_repair or mechanism_alternative",
                "mechanism_steps": ["ordered executable decision operation"],
                "target_failure_buckets": ["general_error, c1_creation, c2_creation, boundary, or residual"],
                "expected_effect": "one short sentence",
            })
        schema = {"candidates": [candidate_schema]}
        return json.dumps(schema, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _hybrid_candidate_type_rejection_reason(candidate_type: str, seen_types: set) -> str:
        normalized = str(candidate_type or "").strip().lower()
        if normalized not in {"task_specific_repair", "mechanism_alternative"}:
            return f"invalid_candidate_type:{normalized or 'missing'}"
        if normalized in seen_types:
            return f"duplicate_candidate_type:{normalized}"
        return ""

    def _student_refusal_or_explanation(self, text: str) -> bool:
        lowered = normalize_spaces(text).lower()
        refusal_markers = [
            "i cannot",
            "i can't",
            "unable to",
            "cannot comply",
            "sorry",
            "as an ai",
            "instead",
            "here is",
            "i will",
        ]
        return any(marker in lowered for marker in refusal_markers)

    def _truncate_candidate_text_fields(
        self,
        item: Dict[str, Any],
        prompt_max_chars: Optional[int] = None,
        field_max_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        prompt_max = int(prompt_max_chars or getattr(self.cfg, "student_candidate_prompt_max_chars", 900) or 900)
        field_max = int(field_max_chars or getattr(self.cfg, "student_candidate_max_chars_per_field", 320) or 320)
        out = dict(item or {})
        for key, value in list(out.items()):
            if isinstance(value, str):
                value = normalize_spaces(value)
                if key == "candidate_prompt":
                    out[key] = value[:prompt_max]
                else:
                    out[key] = value[:field_max]
        return out

    @staticmethod
    def _prompt_ends_with_sentence_boundary(prompt: str) -> bool:
        return bool(re.search(r"[.!?;:)\]}'\"]\s*$", str(prompt or "").strip()))

    def _prepare_v8_candidate_text_fields(self, item: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        out = dict(item or {})
        field_max = int(getattr(self.cfg, "student_candidate_max_chars_per_field", 320) or 320)
        for key, value in list(out.items()):
            if isinstance(value, str):
                out[key] = normalize_spaces(value) if key == "candidate_prompt" else normalize_spaces(value)[:field_max]
        prompt = str(out.get("candidate_prompt", "")).strip()
        soft = int(getattr(self.cfg, "student_candidate_prompt_soft_max_chars", 1100) or 1100)
        hard = int(getattr(self.cfg, "student_candidate_prompt_hard_max_chars", 1400) or 1400)
        audit = {
            "candidate_prompt_char_count": len(prompt),
            "candidate_prompt_over_soft_limit": len(prompt) > soft,
            "candidate_prompt_over_hard_limit": len(prompt) > hard,
            "candidate_prompt_overlength_rejected": len(prompt) > hard,
            "candidate_prompt_ends_with_sentence_boundary": self._prompt_ends_with_sentence_boundary(prompt),
        }
        if len(prompt) > hard:
            self.prompt_overlength_rejection_count += 1
            return None, audit
        if prompt and not audit["candidate_prompt_ends_with_sentence_boundary"]:
            audit["candidate_prompt_incomplete_rejected"] = True
            return None, audit
        return out, audit

    def _structured_fallback_role(self, agent_id: int, index: int, mode: str = "diversity") -> Dict[str, Any]:
        def finalize(row: Dict[str, Any]) -> Dict[str, Any]:
            if not self._v7_residual_protocol_enabled():
                return row
            result = dict(row)
            result["mechanism_name"] = str(result.pop("role_name", "local_reasoning_mechanism"))
            prompt = str(result.get("candidate_prompt", ""))
            prompt = re.sub(r"^You are (?:an?|the) [^.]+\.\s*", "", prompt, flags=re.IGNORECASE)
            result["candidate_prompt"] = "Use this solver instruction: " + prompt
            return result

        accuracy_repair_roles = [
            {
                "role_name": "constraint_verifier",
                "candidate_prompt": (
                    "You are a constraint verifier. Before answering, list the explicit constraints and qualifiers in the question. "
                    "Reject any answer that satisfies the general pattern but violates a stated constraint. "
                    "Then give exactly one final answer after a brief consistency check."
                ),
                "decision_procedure": [
                    "list explicit constraints",
                    "compare plausible answers against constraints",
                    "reject constraint violations",
                    "final consistency check",
                ],
                "when_to_use": "Use when the target agent misses qualifiers, exceptions, or stated constraints.",
                "fallback_strategy": "If no explicit constraints are visible, compare the two most plausible answers and verify the final choice.",
                "target_error_pattern": "missed_constraint",
                "accuracy_repair_rule": "force explicit constraint listing before selecting the final answer",
                "expected_accuracy_effect": "reduces premature selections that violate stated conditions",
            },
            {
                "role_name": "option_elimination_specialist",
                "candidate_prompt": (
                    "You are an option-elimination specialist. Compare the plausible answer choices one by one. "
                    "For each choice, state the strongest reason it could be correct and the strongest reason it could fail. "
                    "Choose the answer with the fewest unresolved failures, then output exactly one final answer."
                ),
                "decision_procedure": [
                    "identify plausible choices",
                    "test each choice",
                    "eliminate unsupported choices",
                    "select final answer",
                ],
                "when_to_use": "Use when the target agent jumps to a plausible answer without eliminating alternatives.",
                "fallback_strategy": "If choices are implicit, name the plausible interpretations and eliminate them as alternatives.",
                "target_error_pattern": "insufficient_option_elimination",
                "accuracy_repair_rule": "require option-by-option or interpretation-by-interpretation elimination",
                "expected_accuracy_effect": "makes the target agent compare alternatives instead of following the first plausible route",
            },
            {
                "role_name": "reverse_answer_validator",
                "candidate_prompt": (
                    "You are a reverse-answer validator. Start from the most plausible candidate answers and ask what must be true for each to be correct. "
                    "Reject candidates whose required assumptions conflict with the question. "
                    "Select the answer with the strongest support and provide exactly one final answer."
                ),
                "decision_procedure": [
                    "name plausible candidates",
                    "derive required assumptions",
                    "reject conflicting assumptions",
                    "final answer",
                ],
                "when_to_use": "Use when the target agent gives weakly supported answers or fails to verify assumptions.",
                "fallback_strategy": "If assumptions are hard to name, run a contradiction check on the selected answer before finalizing.",
                "target_error_pattern": "weak_verification",
                "accuracy_repair_rule": "validate the selected answer by checking the assumptions it requires",
                "expected_accuracy_effect": "catches unsupported selections before the final answer is emitted",
            },
            {
                "role_name": "format_and_answer_auditor",
                "candidate_prompt": (
                    "You are a format-and-answer auditor. Solve the problem normally, then audit the final answer format before responding. "
                    "Ensure the final response contains exactly one answer in the required format and no extra alternatives."
                ),
                "decision_procedure": [
                    "solve",
                    "check answer format",
                    "remove extra alternatives",
                    "emit exactly one final answer",
                ],
                "when_to_use": "Use when the target agent omits, duplicates, or malforms the final answer.",
                "fallback_strategy": "If the reasoning is uncertain, still emit one best-supported final answer in the required format.",
                "target_error_pattern": "invalid_or_missing_final_answer",
                "accuracy_repair_rule": "add a final answer audit that enforces exactly one valid answer",
                "expected_accuracy_effect": "reduces invalid outputs and missing-answer failures",
            },
        ]
        if str(mode).lower() in {"accuracy_repair", "accuracy"}:
            role = accuracy_repair_roles[(int(agent_id) + int(index)) % len(accuracy_repair_roles)]
            return finalize({
                **role,
                "anti_overlap_rule": "Use the named repair procedure because it fixes a target-agent error pattern, not because it sounds different.",
                "validity_checks": ["trace shows the repair procedure", "final answer is explicit", "no sample text is copied"],
                "accuracy_checks": ["repair rule is executed", "final answer is verified before output"],
            })

        roles = [
            {
                "role_name": "boundary_condition_checker",
                "candidate_prompt": (
                    "You are a boundary-condition checker. Before answering, list the explicit constraints, "
                    "edge cases, and qualifiers in the question. Eliminate choices or interpretations that violate "
                    "any constraint, then verify the final answer against each constraint."
                ),
                "decision_procedure": ["list constraints", "check edge cases", "eliminate violations", "verify final answer"],
                "when_to_use": "Use when errors come from missing qualifiers, edge cases, or hidden constraints.",
                "fallback_strategy": "If there are no clear constraints, switch to direct reasoning with one contradiction check.",
            },
            {
                "role_name": "reverse_validator",
                "candidate_prompt": (
                    "You are a reverse validator. Start from the strongest candidate answers and ask what would need "
                    "to be true for each one. Reject candidates whose required assumptions conflict with the question, "
                    "then choose the answer with the fewest unsupported assumptions."
                ),
                "decision_procedure": ["name strongest candidates", "derive required assumptions", "reject conflicts", "choose supported answer"],
                "when_to_use": "Use when several answers look plausible but one fails under reverse checking.",
                "fallback_strategy": "If no candidate can be reverse-checked, compare the two most plausible answers directly.",
            },
            {
                "role_name": "counterexample_tester",
                "candidate_prompt": (
                    "You are a counterexample tester. For each plausible answer, try to construct a minimal counterexample "
                    "or contradiction. Prefer the answer that survives counterexample search, and perform one final consistency check."
                ),
                "decision_procedure": ["identify plausible answers", "search counterexamples", "compare survivors", "run consistency check"],
                "when_to_use": "Use when the team is overconfident in a tempting but brittle answer.",
                "fallback_strategy": "If counterexamples are not meaningful, use explicit option elimination.",
            },
            {
                "role_name": "representation_converter",
                "candidate_prompt": (
                    "You are a representation converter. Rewrite the problem into a compact alternative form such as "
                    "a table, symbolic relation, coordinate list, or cause-effect chain. Solve from that representation "
                    "and verify that it preserves the original question."
                ),
                "decision_procedure": ["convert representation", "solve converted form", "map back to original", "verify preservation"],
                "when_to_use": "Use when direct wording is confusing or spatial, logical, or relational structure matters.",
                "fallback_strategy": "If conversion adds no clarity, return to concise direct reasoning.",
            },
            {
                "role_name": "ambiguity_resolver",
                "candidate_prompt": (
                    "You are an ambiguity resolver. Identify the key ambiguous phrase, pronoun, rule, or label. "
                    "Test each interpretation against the full context, discard interpretations that require unstated facts, "
                    "and answer using the interpretation best supported by the text."
                ),
                "decision_procedure": ["find ambiguity", "test interpretations", "discard unstated assumptions", "answer supported reading"],
                "when_to_use": "Use when mistakes come from pronoun, label, or wording ambiguity.",
                "fallback_strategy": "If no ambiguity exists, use boundary-condition checking.",
            },
        ]
        order = [0, 3, 1, 2, 4]
        role = roles[order[(int(agent_id) + int(index)) % len(order)]]
        return finalize({
            **role,
            "anti_overlap_rule": "Use the named procedure explicitly instead of repeating the default decomposition order.",
            "validity_checks": ["trace shows the named procedure", "final answer is explicit", "no sample text is copied"],
            "accuracy_checks": ["compare plausible alternatives", "verify the final choice against the question"],
        })

    async def _chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
        client_role: str = "evaluator",
        audit_context: Optional[Mapping[str, Any]] = None,
    ) -> str:
        client = self.solver_client if client_role == "solver" else self.evaluator_client
        last_err: Optional[Exception] = None
        attempt = 0
        transient_failures = 0
        timeout_sec = float(self.cfg.llm_call_timeout or 0.0)
        prompt_estimate = self._estimate_tokens(system_prompt) + self._estimate_tokens(user_prompt)
        while True:
            start_time = time.time()
            try:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if timeout_sec > 0:
                    kwargs["timeout"] = timeout_sec
                resp = await client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                self._record_llm_call(
                    stage=stage,
                    client_role=client_role,
                    model=model,
                    temperature=temperature,
                    prompt_tokens=self._usage_value(usage, "prompt_tokens", prompt_estimate),
                    completion_tokens=self._usage_value(usage, "completion_tokens", self._estimate_tokens(text)),
                    latency_seconds=time.time() - start_time,
                    success=True,
                    audit_context={**dict(audit_context or {}), "response_empty": not bool(str(text or "").strip())},
                )
                return text
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                transient = any(
                    x in msg
                    for x in [
                        "timeout",
                        "timed out",
                        "deadline",
                        "rate limit",
                        "temporarily",
                        "temporary",
                        "connection",
                        "server",
                        "overloaded",
                        "try again",
                        "503",
                        "502",
                        "504",
                    ]
                )
                if not transient:
                    if attempt >= max(1, int(self.cfg.max_retries)):
                        self._record_llm_call(
                            stage=stage,
                            client_role=client_role,
                            model=model,
                            temperature=temperature,
                            prompt_tokens=prompt_estimate,
                            completion_tokens=0,
                            latency_seconds=time.time() - start_time,
                            success=False,
                            error_type=type(e).__name__,
                            audit_context=audit_context,
                        )
                        break
                else:
                    transient_failures += 1
                    if not self.cfg.transient_retry_forever and transient_failures > int(self.cfg.max_transient_retries or self.cfg.max_retries):
                        self._record_llm_call(
                            stage=stage,
                            client_role=client_role,
                            model=model,
                            temperature=temperature,
                            prompt_tokens=prompt_estimate,
                            completion_tokens=0,
                            latency_seconds=time.time() - start_time,
                            success=False,
                            error_type=type(e).__name__,
                            audit_context=audit_context,
                        )
                        break
                sleep_sec = min(float(self.cfg.max_retry_backoff), float(self.cfg.retry_sleep) * (2 ** attempt))
                if self.cfg.llm_call_logging:
                    print(f"[LLM][retry] stage={stage} model={model} attempt={attempt + 1} sleep={sleep_sec:.2f} error={normalize_spaces(str(e))[:240]}", flush=True)
                await asyncio.sleep(sleep_sec)
                attempt += 1
        raise RuntimeError(f"LLM call failed at {stage}: {last_err}")

    async def solve_once(self, question: str, agent_id: int, prompt_text: str) -> Tuple[str, str]:
        effective_task = infer_task_type(task_type=self.cfg.task_type, question=question, answer=None)
        answer_format = str(getattr(self.cfg, "answer_format", "") or "").strip().lower()
        if answer_format == "option_letter":
            answer_hint = "<A/B/C/D>"
        elif answer_format == "boolean":
            answer_hint = "<true/false>"
        elif answer_format == "yes_no":
            answer_hint = "<yes/no>"
        elif answer_format == "valid_invalid":
            answer_hint = "<valid/invalid>"
        elif answer_format == "numeric":
            answer_hint = "<number>"
        else:
            answer_hint = "<answer>"
        if effective_task == "mmlu" or answer_format == "option_letter":
            system_prompt = (
                "You are solving a multiple-choice question. Follow the agent role faithfully and make the role's "
                "decision procedure visible in a compact trace. Do not merely name the role; execute it. "
                f"Avoid filler, avoid copying the question, and end with exactly one line: FINAL_ANSWER: {answer_hint}.\n\n"
                f"Agent role:\n{prompt_text}"
            )
        else:
            system_prompt = (
                "You are solving a reasoning problem. Follow the agent role faithfully and make the role's "
                "decision procedure visible in a compact trace. Do not merely name the role; execute it. "
                f"Avoid filler, avoid copying the question, and end with exactly one line: FINAL_ANSWER: {answer_hint}.\n\n"
                f"Agent role:\n{prompt_text}"
            )
        text = await self._chat(
            model=self.cfg.agent_model,
            system_prompt=system_prompt,
            user_prompt=f"Question:\n{question}\n\nSolve with the assigned role and keep the trace concise.",
            temperature=float(self.cfg.temperature),
            max_tokens=int(self.cfg.max_tokens),
            stage=f"solver_agent_{agent_id}",
            client_role="solver",
        )
        return text, self.task_spec.extract_pred(text, question)

    async def solve_with_prompts(self, question: str, prompts: List[str]) -> Tuple[List[str], List[str]]:
        return await self.solve_with_prompts_limited(question, prompts, self.solver_call_semaphore)

    async def solve_with_prompts_limited(
        self,
        question: str,
        prompts: List[str],
        solver_call_semaphore: asyncio.Semaphore,
    ) -> Tuple[List[str], List[str]]:
        async def solve_agent(agent_id: int):
            async with solver_call_semaphore:
                return await self.solve_once(question, agent_id, prompts[agent_id])

        outs = await asyncio.gather(*[solve_agent(i) for i in range(len(self.agents))])
        return [x[0] for x in outs], [x[1] for x in outs]

    async def get_or_create_solver_rollout(
        self,
        *,
        cache_key: str,
        lookup: Callable[[], Optional[Dict[str, Any]]],
        call_factory: Callable[[], Awaitable[Dict[str, Any]]],
    ) -> Tuple[Dict[str, Any], str]:
        """Read a rollout cache or coalesce an identical in-flight API request."""
        cached = lookup()
        if isinstance(cached, dict) and "trace" in cached and "answer" in cached:
            return cached, "persisted_cache" if cached.get("cache_origin") == "persisted" else "memory_cache"

        if not bool(getattr(self.cfg, "solver_rollout_singleflight", True)):
            return await call_factory(), "api_call"
        if not hasattr(self, "solver_rollout_inflight"):
            self.solver_rollout_inflight = {}
        if not hasattr(self, "solver_rollout_inflight_lock"):
            self.solver_rollout_inflight_lock = asyncio.Lock()

        owner = False
        async with self.solver_rollout_inflight_lock:
            future = self.solver_rollout_inflight.get(cache_key)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                # Consume exceptions if an owner fails before another waiter attaches.
                future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
                self.solver_rollout_inflight[cache_key] = future
                owner = True
        if not owner:
            return await future, "inflight_reuse"
        try:
            row = await call_factory()
            future.set_result(row)
            return row, "api_call"
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            async with self.solver_rollout_inflight_lock:
                self.solver_rollout_inflight.pop(cache_key, None)

    async def _solve_agent_rollout(
        self,
        *,
        question: str,
        question_hash: str,
        prompt: str,
        agent_id: int,
        source: str,
    ) -> Tuple[str, str, str]:
        cache_key = self._solver_rollout_cache_key(question_hash, prompt, agent_id)

        async def call_factory() -> Dict[str, Any]:
            async with self.solver_call_semaphore:
                trace, answer = await self.solve_once(question, agent_id, prompt)
            self._record_solver_rollout(
                question_hash=question_hash,
                prompt=prompt,
                trace=trace,
                answer=answer,
                agent_id=agent_id,
                source=source,
            )
            return {"trace": trace, "answer": answer, "cache_origin": "current_run"}

        if not self.cfg.candidate_reuse_recorded_rollouts:
            trace, answer = await self.solve_once(question, agent_id, prompt)
            return trace, answer, "api_call"
        row, origin = await self.get_or_create_solver_rollout(
            cache_key=cache_key,
            lookup=lambda: self._lookup_solver_rollout(question_hash, prompt, agent_id),
            call_factory=call_factory,
        )
        return str(row.get("trace", "")), str(row.get("answer", "")), origin

    async def solve_with_prompts_reusing_records(
        self,
        question: str,
        prompts: List[str],
        source: str = "candidate_eval",
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        prompts = list(prompts)
        while len(prompts) < len(self.agents):
            prompts.append(self.agents[len(prompts)].current_prompt)
        qh = self._hash(question)
        n = len(self.agents)
        outs = await asyncio.gather(
            *[
                self._solve_agent_rollout(
                    question=question, question_hash=qh, prompt=prompts[agent_id], agent_id=agent_id, source=source
                )
                for agent_id in range(n)
            ]
        )
        final_traces = [str(row[0] or "") for row in outs]
        final_answers = [str(row[1] or "") for row in outs]
        origins = [str(row[2]) for row in outs]
        reuse_hits = sum(origin in {"memory_cache", "persisted_cache", "inflight_reuse"} for origin in origins)
        api_calls = sum(origin == "api_call" for origin in origins)
        stats = {
            "solver_reuse_enabled": bool(self.cfg.candidate_reuse_recorded_rollouts),
            "solver_reuse_hits": int(reuse_hits),
            "solver_reuse_misses": int(api_calls),
            "solver_calls": int(api_calls),
            "solver_reuse_total": int(n),
            "solver_memory_cache_hits": int(sum(origin == "memory_cache" for origin in origins)),
            "solver_persisted_cache_hits": int(sum(origin == "persisted_cache" for origin in origins)),
            "solver_inflight_reuses": int(sum(origin == "inflight_reuse" for origin in origins)),
        }
        return final_traces, final_answers, stats

    async def ensure_recorded_rollouts_for_prompts(
        self,
        eval_batch: List[Dict[str, str]],
        prompts: List[str],
        source: str,
    ) -> Dict[str, Any]:
        if not self.cfg.candidate_reuse_recorded_rollouts or not eval_batch:
            return {"enabled": bool(self.cfg.candidate_reuse_recorded_rollouts), "solver_calls": 0, "solver_reuse_hits": 0, "solver_reuse_total": 0}
        totals = {"solver_calls": 0, "solver_reuse_hits": 0, "solver_reuse_total": 0}
        for ex in eval_batch:
            q = str(ex.get("question", ""))
            if not q:
                continue
            _, _, stats = await self.solve_with_prompts_reusing_records(q, prompts, source=source)
            totals["solver_calls"] += int(stats.get("solver_calls", 0) or 0)
            totals["solver_reuse_hits"] += int(stats.get("solver_reuse_hits", 0) or 0)
            totals["solver_reuse_total"] += int(stats.get("solver_reuse_total", 0) or 0)
        totals["enabled"] = True
        totals["solver_reuse_hit_rate"] = float(totals["solver_reuse_hits"] / totals["solver_reuse_total"]) if totals["solver_reuse_total"] else 0.0
        return totals

    async def _prewarm_factorized_candidate_rollouts(
        self,
        *,
        agent_id: int,
        eval_batch: List[Dict[str, str]],
        peer_prompts: List[str],
        candidate_pool: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Populate per-question rollouts for fixed peers and unique target prompts.

        Team metrics are deliberately *not* cached here. Every candidate later calls
        the normal evaluator, which recomposes a team on the current batch.
        """
        unique_prompts: Dict[str, str] = {}
        for candidate in candidate_pool:
            prompt = str(candidate.get("prompt", ""))
            if prompt:
                unique_prompts.setdefault(self._hash(normalize_spaces(prompt)), prompt)
        active_prompt = str(peer_prompts[agent_id]) if agent_id < len(peer_prompts) else str(self.agents[agent_id].current_prompt)
        unique_prompts.setdefault(self._hash(normalize_spaces(active_prompt)), active_prompt)
        requests: List[Tuple[str, int, str, str]] = []
        for ex in eval_batch:
            question = str(ex.get("question", ""))
            if not question:
                continue
            question_hash = self._hash(question)
            for peer_id, prompt in enumerate(peer_prompts):
                if peer_id != agent_id:
                    requests.append((question, peer_id, str(prompt), question_hash))
            for prompt in unique_prompts.values():
                requests.append((question, agent_id, prompt, question_hash))

        async def prewarm(row: Tuple[str, int, str, str]):
            question, row_agent_id, prompt, question_hash = row
            trace, answer, origin = await self._solve_agent_rollout(
                question=question,
                question_hash=question_hash,
                prompt=prompt,
                agent_id=row_agent_id,
                source=f"candidate_factorized_{'peer' if row_agent_id != agent_id else 'target'}_agent_{agent_id}",
            )
            return trace, answer, origin

        results = await asyncio.gather(*[prewarm(request) for request in requests])
        origins = [origin for _, _, origin in results]
        candidate_count = len(candidate_pool)
        example_count = len(eval_batch)
        naive = candidate_count * len(self.agents) * example_count
        factorized = (max(0, len(self.agents) - 1) + len(unique_prompts)) * example_count
        api_calls = sum(origin == "api_call" for origin in origins)
        memory_hits = sum(origin == "memory_cache" for origin in origins)
        persisted_hits = sum(origin == "persisted_cache" for origin in origins)
        inflight = sum(origin == "inflight_reuse" for origin in origins)
        return {
            "candidate_eval_execution_mode": "factorized_cached",
            "candidate_eval_candidate_object_count": candidate_count,
            "candidate_eval_unique_target_prompt_count": len(unique_prompts),
            "candidate_eval_duplicate_target_prompt_count": max(0, candidate_count - len(unique_prompts)),
            "candidate_eval_example_count": example_count,
            "candidate_eval_repeat_count": 1,
            "candidate_eval_naive_rollout_request_count": naive,
            "candidate_eval_factorized_rollout_request_count": factorized,
            "candidate_eval_unique_rollout_key_count": len(requests),
            "candidate_eval_memory_cache_hit_count": memory_hits,
            "candidate_eval_persisted_cache_hit_count": persisted_hits,
            "candidate_eval_inflight_reuse_count": inflight,
            "candidate_eval_solver_api_call_count": api_calls,
            "candidate_eval_rollout_failure_count": 0,
            "candidate_eval_calls_saved_vs_naive": naive - api_calls,
            "candidate_eval_cache_hit_rate": float((memory_hits + persisted_hits + inflight) / len(requests)) if requests else 0.0,
            "candidate_eval_peer_rollout_key_count": max(0, len(self.agents) - 1) * example_count,
            "candidate_eval_target_rollout_key_count": len(unique_prompts) * example_count,
            "candidate_eval_prompt_dedup_savings": max(0, candidate_count - len(unique_prompts)) * example_count,
        }

    async def solve_with_current_prompts(self, question: str) -> Tuple[List[str], List[str]]:
        return await self.solve_with_prompts(question, self._active_prompt_list())

    async def solve_with_agent_prompt_override(
        self,
        question: str,
        agent_id: int,
        prompt: str,
        peer_prompts: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        prompts = list(peer_prompts or self._active_prompt_list())
        while len(prompts) < len(self.agents):
            prompts.append(self.agents[len(prompts)].current_prompt)
        prompts[agent_id] = prompt
        return await self.solve_with_prompts(question, prompts)

    def rule_invalid_check(self, trace: str, answer: str = "") -> Dict[str, Any]:
        text = str(trace or "")
        reasons = []
        tokens = re.findall(r"\w+", text)
        if len(normalize_spaces(text)) < 40:
            reasons.append("trace_too_short")
        if "FINAL_ANSWER:" not in text:
            reasons.append("missing_final_answer")
        if len(tokens) < 12:
            reasons.append("too_few_tokens")
        if not str(answer or "").strip():
            reasons.append("empty_extracted_answer")
        if len(tokens) >= 12:
            bigrams = list(zip(tokens, tokens[1:]))
            repeated = len(bigrams) - len(set(bigrams))
            ratio = repeated / max(1, len(bigrams))
            if ratio > 0.35:
                reasons.append("bigram_repetition")
        return {"invalid": int(bool(reasons)), "reasons": reasons}

    def _load_embedding_model(self):
        if self.embedding_model is not None:
            return self.embedding_model
        from sentence_transformers import SentenceTransformer

        model_name = str(self.cfg.embedding_model)
        try:
            self.embedding_model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.embedding_model = SentenceTransformer(model_name)
        return self.embedding_model

    def _split_trace_for_embedding(self, text: str) -> List[str]:
        words = normalize_spaces(text).split()
        if not words:
            return []
        chunk_words = max(1, int(self.cfg.trace_embedding_chunk_words or 320))
        overlap = max(0, min(int(self.cfg.trace_embedding_chunk_overlap or 0), chunk_words - 1))
        if len(words) <= chunk_words:
            return [" ".join(words)]
        chunks = []
        step = max(1, chunk_words - overlap)
        for start in range(0, len(words), step):
            chunk = words[start : start + chunk_words]
            if chunk:
                chunks.append(" ".join(chunk))
            if start + chunk_words >= len(words):
                break
        return chunks

    def _normalize_vector(self, vector: Any) -> List[float]:
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm <= 0.0 or np.isnan(norm):
            return []
        return (arr / norm).astype(float).tolist()

    def _encode_trace_document(self, trace: str) -> List[float]:
        cleaned = normalize_spaces(trace)
        if not cleaned:
            return []
        cache_key = self._hash(f"{self.cfg.embedding_model}|{self.cfg.trace_embedding_chunk_words}|{self.cfg.trace_embedding_chunk_overlap}|{cleaned}")
        cached = self.embedding_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        chunks = self._split_trace_for_embedding(cleaned)
        if not chunks:
            return []
        model = self._load_embedding_model()
        embeddings = model.encode(chunks, normalize_embeddings=True)
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            pooled = arr
        else:
            pooled = np.mean(arr, axis=0)
        vector = self._normalize_vector(pooled)
        self.embedding_cache[cache_key] = vector
        return vector

    def _vector_cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 1.0
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom <= 0.0 or np.isnan(denom):
            return 1.0
        sim = float(np.dot(va, vb) / denom)
        return float(max(-1.0, min(1.0, sim)))

    def embedding_overlap_diagnostics(
        self,
        traces: List[str],
        prompts: Optional[List[str]] = None,
        invalids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        n = len(traces)
        invalids = list(invalids or [0 for _ in traces])
        embeddings = [self._encode_trace_document(trace) for trace in traces]
        pair_rows = []
        per_agent_scores = [0.0 for _ in range(n)]
        per_agent_counts = [0 for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (i < len(invalids) and int(invalids[i]) > 0) or (j < len(invalids) and int(invalids[j]) > 0):
                    sim = 1.0
                else:
                    sim = self._vector_cosine_similarity(embeddings[i], embeddings[j])
                    sim = max(0.0, sim)
                pair_rows.append({"pair": [i, j], "overlap": sim})
                per_agent_scores[i] += sim
                per_agent_scores[j] += sim
                per_agent_counts[i] += 1
                per_agent_counts[j] += 1
        per_agent_overlap = [
            float(per_agent_scores[i] / per_agent_counts[i]) if per_agent_counts[i] else 0.0
            for i in range(n)
        ]
        mean_overlap = float(np.mean([p["overlap"] for p in pair_rows])) if pair_rows else 0.0
        threshold = float(self.cfg.homogeneity_overlap_threshold)
        for row in pair_rows:
            i, j = row["pair"]
            row["invalid_pair"] = bool(
                (i < len(invalids) and int(invalids[i]) > 0)
                or (j < len(invalids) and int(invalids[j]) > 0)
            )
        high_pairs = [p for p in pair_rows if float(p["overlap"]) >= threshold]
        roles = [
            {
                "agent_id": i,
                "prompt_preview": normalize_spaces((prompts or self._active_prompt_list())[i])[:220] if i < len(prompts or []) else "",
                "trace_hash": self._hash(traces[i]),
                "trace_preview": normalize_spaces(traces[i])[:360],
                "overlap_pressure": per_agent_overlap[i],
            }
            for i in range(n)
        ]
        embedding_diversity = max(0.0, min(1.0, 1.0 - mean_overlap))
        return {
            "mean_embedding_overlap": mean_overlap,
            "embedding_diversity": embedding_diversity,
            "trace_embedding_model": str(self.cfg.embedding_model),
            "trace_embedding_chunk_words": int(self.cfg.trace_embedding_chunk_words),
            "trace_embedding_chunk_overlap": int(self.cfg.trace_embedding_chunk_overlap),
            "per_agent_overlap": per_agent_overlap,
            "pair_overlaps": pair_rows,
            "high_overlap_pairs": high_pairs,
            "homogeneity_overlap_threshold": threshold,
            "roles": roles,
        }

    def _vote_with_diagnostics(self, answers: List[str], question_hash: str = "") -> Dict[str, Any]:
        return plurality_vote_with_diagnostics(
            answers,
            tie_break_method=str(getattr(self.cfg, "vote_tie_break", "random")),
            seed=int(getattr(self.cfg, "seed", 0) or 0),
            question_hash=question_hash,
        )

    def _rollout_any_correct(self, rollout: Dict[str, Any]) -> int:
        return int(any(int(x) for x in rollout.get("individual_correct", [])))

    def _safe_agent_correct(self, rollout: Dict[str, Any], agent_id: int) -> int:
        individual = list(rollout.get("individual_correct", []))
        return int(individual[agent_id]) if 0 <= int(agent_id) < len(individual) else 0

    def _target_trace_novelty(self, traces: List[str], target_agent_id: int) -> float:
        try:
            target_agent_id = int(target_agent_id)
            if target_agent_id < 0 or target_agent_id >= len(traces):
                return 0.0
            target = self._encode_trace_document(str(traces[target_agent_id]))
            peers = [
                self._encode_trace_document(str(trace))
                for i, trace in enumerate(traces)
                if i != target_agent_id and str(trace or "").strip()
            ]
            sims = [self._vector_cosine_similarity(target, peer) for peer in peers if peer]
            if not target or not sims:
                return 0.0
            return self._clip01(1.0 - float(np.mean([max(0.0, sim) for sim in sims])))
        except Exception:
            return 0.0

    def _trace_diversity_for_indices(self, traces: List[str], indices: List[int]) -> float:
        try:
            embeddings = [self._encode_trace_document(str(traces[i])) for i in indices if 0 <= i < len(traces)]
            embeddings = [vec for vec in embeddings if vec]
            if len(embeddings) < 2:
                return 0.0
            diversities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    diversities.append(1.0 - max(0.0, self._vector_cosine_similarity(embeddings[i], embeddings[j])))
            return self._clip01(float(np.mean(diversities)) if diversities else 0.0)
        except Exception:
            return 0.0

    def _useful_trace_diversity(
        self,
        traces: List[str],
        individual_correct: List[int],
        invalid_flags: List[int],
    ) -> float:
        indices = [
            i
            for i, correct in enumerate(individual_correct)
            if int(correct) > 0 and i < len(invalid_flags) and int(invalid_flags[i]) <= 0
        ]
        return self._trace_diversity_for_indices(traces, indices)

    def _weighted_vote_with_diagnostics(
        self,
        answers: List[str],
        invalid_flags: Optional[List[int]] = None,
        per_agent_overlap: Optional[List[float]] = None,
        question_hash: str = "",
    ) -> Dict[str, Any]:
        invalid_flags = list(invalid_flags or [0 for _ in answers])
        per_agent_overlap = list(per_agent_overlap or [0.0 for _ in answers])
        scores: Dict[str, float] = {}
        agent_weights = []
        for i, raw_answer in enumerate(answers):
            answer = str(raw_answer or "").strip()
            invalid = int(invalid_flags[i]) if i < len(invalid_flags) else 0
            overlap = self._clip01(per_agent_overlap[i]) if i < len(per_agent_overlap) else 0.0
            reliability = 1.0
            validity = 0.0 if invalid else 1.0
            independence = min(max(0.0, 1.0 - overlap), 0.5)
            weight = float(reliability * validity * independence)
            agent_weights.append(
                {
                    "agent_id": i,
                    "answer": answer,
                    "reliability": reliability,
                    "validity": validity,
                    "independence": independence,
                    "weight": weight,
                }
            )
            if answer and weight > 0.0:
                scores[answer] = scores.get(answer, 0.0) + weight

        fallback = False
        if not scores:
            fallback = True
            majority = self._vote_with_diagnostics(answers, question_hash=question_hash)
            return {
                "weighted_vote_answer": str(majority.get("vote_answer", "")),
                "weighted_vote_scores": {},
                "weighted_vote_tie": bool(majority.get("vote_tie", False)),
                "weighted_tie_candidates": list(majority.get("tie_candidates", [])),
                "weighted_tie_break_method": str(majority.get("tie_break_method", "")),
                "weighted_vote_agent_weights": agent_weights,
                "weighted_vote_fallback": fallback,
            }

        max_score = max(scores.values())
        tied = [answer for answer, score in scores.items() if abs(float(score) - float(max_score)) <= 1e-12]
        tied_set = set(tied)
        method = str(getattr(self.cfg, "vote_tie_break", "random") or "random").lower()
        if len(tied) <= 1:
            selected = tied[0]
        elif method == "abstain":
            selected = ""
        elif method == "random":
            seed_material = f"{int(getattr(self.cfg, 'seed', 0) or 0)}|{question_hash}|weighted_vote"
            rng = random.Random(int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:12], 16))
            selected = rng.choice(sorted(tied))
        else:
            selected = next((str(answer or "").strip() for answer in answers if str(answer or "").strip() in tied_set), sorted(tied)[0])
        return {
            "weighted_vote_answer": selected,
            "weighted_vote_scores": {key: float(value) for key, value in scores.items()},
            "weighted_vote_tie": len(tied) > 1,
            "weighted_tie_candidates": sorted(tied),
            "weighted_tie_break_method": method,
            "weighted_vote_agent_weights": agent_weights,
            "weighted_vote_fallback": fallback,
        }

    def compute_rollout_metrics(
        self,
        traces: List[str],
        answers: List[str],
        gold: str,
        prompts: Optional[List[str]] = None,
        question_hash: str = "",
    ) -> Dict[str, Any]:
        plurality_vote = self._vote_with_diagnostics(answers, question_hash=question_hash)
        plurality_vote_answer = str(plurality_vote.get("vote_answer", ""))
        individual_correct = [int(self.task_spec.match_answer(a, gold)) for a in answers]
        plurality_vote_correct = int(self.task_spec.match_answer(plurality_vote_answer, gold))
        gold_vote_diagnostics = compute_gold_vote_diagnostics(
            answers,
            gold,
            self.task_spec.match_answer,
            len(self.agents),
        )
        plurality_margin_votes = int(
            gold_vote_diagnostics.get("gold_vote_count", 0)
            - gold_vote_diagnostics.get("largest_wrong_vote_count", 0)
        )
        gold_vote_diagnostics.update({
            "plurality_margin_votes": plurality_margin_votes,
            "normalized_plurality_margin": float(
                gold_vote_diagnostics.get("normalized_vote_margin", -1.0)
            ),
            "strict_plurality_win": bool(plurality_margin_votes > 0),
            "plurality_gold_leading": bool(plurality_margin_votes > 0),
            "plurality_gold_top_tied": bool(plurality_margin_votes == 0 and bool(answers)),
            "plurality_gold_one_vote_behind": bool(plurality_margin_votes == -1),
            "plurality_gold_far_behind": bool(plurality_margin_votes <= -2),
        })
        if self._is_accuracy_only_mode():
            n = len(traces)
            active_prompts = prompts or self._active_prompt_list()
            roles = [
                {
                    "agent_id": i,
                    "prompt_preview": normalize_spaces(active_prompts[i])[:220] if i < len(active_prompts) else "",
                    "trace_hash": self._hash(traces[i]) if i < len(traces) else "",
                    "trace_preview": self._trace_method_preview(traces[i]) if i < len(traces) else "",
                    "overlap_pressure": 0.0,
                }
                for i in range(n)
            ]
            invalids = [0 for _ in traces]
            overlap = {
                "mean_embedding_overlap": 0.0,
                "embedding_diversity": 0.0,
                "trace_embedding_model": "",
                "trace_embedding_chunk_words": int(self.cfg.trace_embedding_chunk_words),
                "trace_embedding_chunk_overlap": int(self.cfg.trace_embedding_chunk_overlap),
                "per_agent_overlap": [0.0 for _ in traces],
                "pair_overlaps": [],
                "high_overlap_pairs": [],
                "homogeneity_overlap_threshold": float(self.cfg.homogeneity_overlap_threshold),
                "roles": roles,
            }
        else:
            invalids = [self.rule_invalid_check(traces[i], answers[i] if i < len(answers) else "").get("invalid", 1) for i in range(len(traces))]
            overlap = self.embedding_overlap_diagnostics(traces, prompts, invalids=invalids)
        weighted_vote = self._weighted_vote_with_diagnostics(
            answers,
            invalid_flags=[int(x) for x in invalids],
            per_agent_overlap=list(overlap.get("per_agent_overlap", [])),
            question_hash=question_hash,
        )
        weighted_vote_answer = str(weighted_vote.get("weighted_vote_answer", ""))
        weighted_vote_correct = int(self.task_spec.match_answer(weighted_vote_answer, gold))
        requested_aggregation_mode = str(getattr(self.cfg, "aggregation_mode", "majority") or "majority").lower()
        effective_aggregation_mode = canonical_aggregation_mode(requested_aggregation_mode)
        aggregation_fallback = ""
        if effective_aggregation_mode == "weighted_vote":
            vote_answer = weighted_vote_answer
            vote_correct = weighted_vote_correct
            vote_tie = bool(weighted_vote.get("weighted_vote_tie", False))
            tie_candidates = list(weighted_vote.get("weighted_tie_candidates", []))
            tie_break_method = str(weighted_vote.get("weighted_tie_break_method", ""))
        else:
            if effective_aggregation_mode == "verifier_select":
                aggregation_fallback = "verifier_select_not_implemented_fallback_majority"
                effective_aggregation_mode = "plurality"
            elif effective_aggregation_mode != "plurality":
                effective_aggregation_mode = "plurality"
            vote_answer = plurality_vote_answer
            vote_correct = plurality_vote_correct
            vote_tie = bool(plurality_vote.get("vote_tie", False))
            tie_candidates = list(plurality_vote.get("tie_candidates", []))
            tie_break_method = str(plurality_vote.get("tie_break_method", ""))
        any_correct = int(any(individual_correct))
        pivotal_fix_opportunities = []
        pivotal_holds = []
        for agent_id, correct in enumerate(individual_correct):
            opportunity = False
            hold = False
            if not correct and not plurality_vote_correct:
                counterfactual_answers = list(answers)
                counterfactual_answers[agent_id] = gold
                counterfactual = self._vote_with_diagnostics(counterfactual_answers, question_hash=question_hash)
                opportunity = bool(self.task_spec.match_answer(str(counterfactual.get("vote_answer", "")), gold))
            if correct and plurality_vote_correct:
                without_target = list(answers)
                without_target[agent_id] = ""
                counterfactual = self._vote_with_diagnostics(without_target, question_hash=question_hash)
                hold = not bool(self.task_spec.match_answer(str(counterfactual.get("vote_answer", "")), gold))
            pivotal_fix_opportunities.append(int(opportunity))
            pivotal_holds.append(int(hold))
        useful_diversity = 0.0 if self._is_accuracy_only_mode() else self._useful_trace_diversity(traces, individual_correct, [int(x) for x in invalids])
        return {
            "vote_answer": vote_answer,
            "vote_correct": vote_correct,
            "individual_correct": individual_correct,
            "vote_tie": vote_tie,
            "tie_candidates": tie_candidates,
            "vote_counts": dict(plurality_vote.get("vote_counts", {})),
            "tie_break_method": tie_break_method,
            "aggregation_mode": requested_aggregation_mode,
            "requested_aggregation_mode": requested_aggregation_mode,
            "effective_aggregation_mode": effective_aggregation_mode,
            "aggregation_fallback": aggregation_fallback,
            "plurality_boundary_version": PLURALITY_BOUNDARY_VERSION,
            "plurality_vote_answer": plurality_vote_answer,
            "plurality_vote_correct": plurality_vote_correct,
            "plurality_vote_tie": bool(plurality_vote.get("vote_tie", False)),
            "plurality_tie_candidates": list(plurality_vote.get("tie_candidates", [])),
            "plurality_vote_counts": dict(plurality_vote.get("vote_counts", {})),
            "plurality_tie_break_method": str(plurality_vote.get("tie_break_method", "")),
            "plurality_tie_break_question_hash": str(question_hash),
            "plurality_pivotal_fix_opportunity_per_agent": pivotal_fix_opportunities,
            "plurality_pivotal_hold_per_agent": pivotal_holds,
            "plurality_pivotal_fix_opportunity_rate": float(np.mean(pivotal_fix_opportunities)) if pivotal_fix_opportunities else 0.0,
            "plurality_pivotal_hold_rate": float(np.mean(pivotal_holds)) if pivotal_holds else 0.0,
            # Historical names remain diagnostic aliases for old readers.
            "majority_vote_answer": plurality_vote_answer,
            "majority_vote_correct": plurality_vote_correct,
            "majority_vote_tie": bool(plurality_vote.get("vote_tie", False)),
            "majority_tie_candidates": list(plurality_vote.get("tie_candidates", [])),
            "majority_vote_counts": dict(plurality_vote.get("vote_counts", {})),
            "majority_tie_break_method": str(plurality_vote.get("tie_break_method", "")),
            "weighted_vote_answer": weighted_vote_answer,
            "weighted_vote_correct": weighted_vote_correct,
            "weighted_vote_tie": bool(weighted_vote.get("weighted_vote_tie", False)),
            "weighted_tie_candidates": list(weighted_vote.get("weighted_tie_candidates", [])),
            "weighted_vote_scores": dict(weighted_vote.get("weighted_vote_scores", {})),
            "weighted_vote_agent_weights": list(weighted_vote.get("weighted_vote_agent_weights", [])),
            "weighted_vote_fallback": bool(weighted_vote.get("weighted_vote_fallback", False)),
            "any_correct": any_correct,
            **gold_vote_diagnostics,
            "useful_diversity": useful_diversity,
            "invalid_rate": float(np.mean(invalids)) if invalids else 1.0,
            "invalid_score": 1.0 - (float(np.mean(invalids)) if invalids else 1.0),
            "invalid_flags": [int(x) for x in invalids],
            **overlap,
        }

    def _trace_method_preview(self, trace: str, max_chars: int = 420) -> str:
        text = re.sub(r"FINAL_ANSWER\s*:\s*.*", "", str(trace or ""), flags=re.IGNORECASE)
        text = re.sub(r"\b(answer|final answer)\b\s*[:=-]?\s*[A-Da-d0-9.+/, \\-]+", "[answer redacted]", text, flags=re.IGNORECASE)
        return normalize_spaces(text)[:max_chars]

    def _redact_optimizer_text(self, text: str, max_chars: int = 420) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"FINAL_ANSWER\s*:\s*.*", "FINAL_ANSWER: [redacted]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(final answer|answer)\b\s*[:=-]\s*[^\n.;,]+", r"\1: [redacted]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(option|choice)\s+[A-Z]\b", "option [label]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\([A-Z]\)", "([label])", cleaned)
        cleaned = re.sub(r"\b[A-D]\s*[\).]\s*", "[label]. ", cleaned)
        cleaned = re.sub(r"\b(true|false|yes|no|valid|invalid)\b", "[boolean-like]", cleaned, flags=re.IGNORECASE)
        return normalize_spaces(cleaned)[:max_chars]

    def _answer_behavior_preview(self, answer: str) -> Dict[str, Any]:
        raw = str(answer or "").strip()
        lowered = raw.lower()
        if not raw:
            kind = "missing"
        elif re.fullmatch(r"\(?[A-Za-z]\)?\.?", raw):
            kind = "option_like"
        elif lowered in {"true", "false", "yes", "no", "valid", "invalid"}:
            kind = "boolean_like"
        elif re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", raw):
            kind = "numeric_like"
        else:
            kind = "text_like"
        return {
            "present": bool(raw),
            "length": len(raw),
            "kind": kind,
            "has_multiple_tokens": len(re.findall(r"\S+", raw)) > 1,
        }

    def _peer_behavior_summary(
        self,
        peer_traces: List[str],
        peer_correct_flags: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        flags = [int(x) for x in list(peer_correct_flags or [])]
        previews = [self._redact_optimizer_text(t, max_chars=240) for t in peer_traces[:2]]
        word_counts = [len(re.findall(r"\w+", str(t or ""))) for t in peer_traces]
        verification_terms = re.compile(r"\b(check|verify|therefore|because|constraint|eliminate|assumption|contradiction)\b", re.IGNORECASE)
        return {
            "num_peer_traces": len(peer_traces),
            "num_peer_correct": int(sum(flags)) if flags else 0,
            "peer_trace_previews": previews,
            "peer_longer_than_target": False,
            "mean_peer_trace_words": float(np.mean(word_counts)) if word_counts else 0.0,
            "peer_uses_verification_terms": bool(any(verification_terms.search(str(t or "")) for t in peer_traces)),
        }

    def _infer_target_error_pattern(
        self,
        target_trace: str,
        target_answer: str,
        peer_traces: List[str],
        rollout: Dict[str, Any],
        agent_id: int,
    ) -> Dict[str, Any]:
        invalid_flags = list(rollout.get("invalid_flags", [])) if isinstance(rollout, dict) else []
        invalid = int(invalid_flags[agent_id]) if agent_id < len(invalid_flags) else int(self.rule_invalid_check(target_trace, target_answer).get("invalid", 0))
        text = normalize_spaces(str(target_trace or ""))
        lower = text.lower()
        answer_preview = self._answer_behavior_preview(target_answer)
        target_words = len(re.findall(r"\w+", text))
        peer_words = [len(re.findall(r"\w+", str(t or ""))) for t in peer_traces]
        peer_mean_words = float(np.mean(peer_words)) if peer_words else 0.0

        def result(pattern: str, hint: str, family: CapabilityResidualFamily, confidence: float) -> Dict[str, Any]:
            return {
                "error_pattern": pattern,
                "repair_hint": hint,
                "capability_residual_family": family.value,
                "confidence": self._clip01(confidence),
            }

        if invalid or not answer_preview["present"]:
            if not answer_preview["present"] or "final_answer:" not in str(target_trace or ""):
                return result(
                    "invalid_or_missing_final_answer",
                    "add a final answer audit that emits exactly one answer in the required format",
                    CapabilityResidualFamily.OUTPUT_VALIDITY,
                    1.0,
                )
            return result(
                "format_violation",
                "check answer format and remove extra alternatives before finalizing",
                CapabilityResidualFamily.OUTPUT_VALIDITY,
                1.0,
            )
        if target_words < 35:
            return result(
                "premature_answer",
                "delay the final answer until after evidence comparison and a short verification step",
                CapabilityResidualFamily.FINAL_VERIFICATION,
                0.8,
            )
        if re.search(r"\b(calculate|compute|equation|number|sum|difference|multiply|divide|symbol|formula)\b", lower):
            if not re.search(r"\b(check|verify|substitut|unit|sanity)\b", lower):
                return result(
                    "calculation_or_symbolic_slip",
                    "add a numeric or symbolic sanity check before the final answer",
                    CapabilityResidualFamily.NUMERIC_SYMBOLIC,
                    0.9,
                )
        if re.search(r"\b(option|choice|alternative|candidate)\b", lower) and not re.search(r"\b(eliminate|reject|compare|fail|against)\b", lower):
            return result(
                "insufficient_option_elimination",
                "force option-by-option elimination before selecting the final answer",
                CapabilityResidualFamily.OPTION_COMPARISON,
                0.9,
            )
        if re.search(r"\b(constraint|except|unless|only|must|not|qualifier|condition)\b", lower) and not re.search(r"\b(list|check|satisfy|violate)\b", lower):
            return result(
                "missed_constraint",
                "force the agent to list explicit constraints before selecting an answer",
                CapabilityResidualFamily.QUALIFIER_NEGATION,
                0.85,
            )
        if re.search(r"\b(before|after|earlier|later|first|last|sequence|order|timeline|simultaneous)\b", lower):
            return result(
                "temporal_order_confusion",
                "construct and verify an explicit temporal ordering before selecting the answer",
                CapabilityResidualFamily.TEMPORAL_ORDER,
                0.75,
            )
        if re.search(r"\b(entity|person|object|name|pronoun|refer|correspond|bind)\b", lower):
            return result(
                "entity_binding_confusion",
                "bind each entity and reference explicitly before propagating constraints",
                CapabilityResidualFamily.ENTITY_BINDING,
                0.7,
            )
        if re.search(r"\b(relation|left|right|above|below|inside|between|adjacent|relative)\b", lower):
            return result(
                "relation_tracking_slip",
                "track each relation in a compact normalized representation and verify composition",
                CapabilityResidualFamily.RELATION_TRACKING,
                0.75,
            )
        if not re.search(r"\b(check|verify|therefore|because|contradiction|assumption|consistent)\b", lower):
            return result(
                "weak_verification",
                "add a final consistency check against the question before output",
                CapabilityResidualFamily.FINAL_VERIFICATION,
                0.75,
            )
        if re.search(r"\b(contradiction|inconsistent|assumption|counterexample|impossible)\b", lower):
            return result(
                "contradiction_check_failure",
                "test the provisional conclusion for contradiction or a concrete counterexample",
                CapabilityResidualFamily.CONTRADICTION_CHECK,
                0.7,
            )
        if peer_mean_words >= max(45.0, float(target_words) * 1.35):
            return result(
                "peer_has_more_specific_reasoning",
                "require grounding the answer in specific clues rather than generic reasoning",
                CapabilityResidualFamily.COMMONSENSE_CONSISTENCY,
                0.55,
            )
        generic_terms = len(re.findall(r"\b(careful|think|analyze|reason|solve|answer)\b", lower))
        evidence_terms = len(re.findall(r"\b(because|therefore|constraint|eliminate|verify|assumption|example|case)\b", lower))
        if generic_terms >= 4 and evidence_terms <= 1:
            return result(
                "overly_generic_reasoning",
                "replace generic reasoning with a concrete evidence-comparison procedure",
                CapabilityResidualFamily.COMMONSENSE_CONSISTENCY,
                0.5,
            )
        return result(
            "unknown_error_pattern",
            "use a concrete compare-then-verify procedure before the final answer",
            CapabilityResidualFamily.UNKNOWN,
            0.0,
        )

    def _case_key(self, sample_hash: str, a: int, b: int) -> str:
        left, right = sorted([int(a), int(b)])
        return f"{sample_hash}:{left}-{right}"

    def _build_homogeneous_cases(
        self,
        sample_hash: str,
        traces: List[str],
        answers: List[str],
        prompts: List[str],
        metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        invalids = [int(x) for x in metrics.get("invalid_flags", [])]
        cases: List[Dict[str, Any]] = []
        for pair in metrics.get("high_overlap_pairs", []):
            if not isinstance(pair, dict):
                continue
            ids = pair.get("pair", [])
            if not isinstance(ids, list) or len(ids) != 2:
                continue
            a, b = int(ids[0]), int(ids[1])
            if a >= len(traces) or b >= len(traces):
                continue
            if (a < len(invalids) and invalids[a]) or (b < len(invalids) and invalids[b]):
                continue
            overlap = float(pair.get("overlap", 0.0))
            for target, peer in [(a, b), (b, a)]:
                cases.append(
                    {
                        "case_id": self._hash(f"{sample_hash}|{target}|{peer}|{overlap:.6f}"),
                        "case_key": self._case_key(sample_hash, target, peer),
                        "sample_hash": sample_hash,
                        "target_agent_id": target,
                        "peer_agent_id": peer,
                        "pair_overlap": overlap,
                        "target_trace_preview": self._trace_method_preview(traces[target]),
                        "peer_trace_preview": self._trace_method_preview(traces[peer]),
                        "target_answer": str(answers[target]) if target < len(answers) else "",
                        "peer_answer": str(answers[peer]) if peer < len(answers) else "",
                        "target_prompt_preview": normalize_spaces(prompts[target])[:260] if target < len(prompts) else "",
                        "peer_prompt_preview": normalize_spaces(prompts[peer])[:260] if peer < len(prompts) else "",
                        "target_valid": True,
                        "peer_valid": True,
                        "team_correct": bool(metrics.get("vote_correct", 0)),
                        "case_type": "homogeneous_valid_pair",
                    }
                )
        return cases

    def _build_validity_cases(
        self,
        sample_hash: str,
        traces: List[str],
        answers: List[str],
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        rows = []
        for agent_id, trace in enumerate(traces):
            answer = answers[agent_id] if agent_id < len(answers) else ""
            invalid = self.rule_invalid_check(trace, answer)
            if not int(invalid.get("invalid", 0)):
                continue
            rows.append(
                {
                    "case_id": self._hash(f"{sample_hash}|invalid|{agent_id}|{self._hash(trace)}"),
                    "sample_hash": sample_hash,
                    "target_agent_id": agent_id,
                    "trace_preview": self._trace_method_preview(trace),
                    "answer_present": bool(str(answer).strip()),
                    "invalid_reasons": list(invalid.get("reasons", [])),
                    "target_prompt_preview": normalize_spaces(prompts[agent_id])[:260] if agent_id < len(prompts) else "",
                    "case_type": "hard_validity_case",
                }
            )
        return rows

    def is_homogeneity_window_warmup_done(self) -> bool:
        return all(len(a.recent_homogeneity_flags) >= self.homogeneity_window for a in self.agents)

    def is_update_window_ready(self) -> bool:
        return len(self.recent_window_records) >= self.homogeneity_window

    def clear_homogeneity_windows(self):
        for agent in self.agents:
            agent.recent_homogeneity_flags.clear()
            agent.homogeneity_count = 0
        self.recent_window_records = []

    def select_agents_for_update(self, metrics: Dict[str, Any]) -> List[int]:
        if not self.is_homogeneity_window_warmup_done():
            return []
        diagnosis = self._window_update_diagnosis(self.recent_window_records)
        pressures = list(diagnosis.get("per_agent_overlap_pressure", metrics.get("per_agent_overlap", [])))
        if not pressures or all(float(x) <= 0 for x in pressures):
            return []
        case_counts = diagnosis.get("homogeneous_case_counts", [])
        invalid_rates = diagnosis.get("per_agent_invalid_rate", [])
        tie_eps = float(self.cfg.homogeneity_pressure_tie_eps)
        ids = list(range(len(self.agents)))
        random.shuffle(ids)
        max_pressure = max(float(x) for x in pressures) if pressures else 0.0
        ids.sort(
            key=lambda i: (
                1 if (i < len(invalid_rates) and float(invalid_rates[i]) >= float(self.cfg.invalid_repair_rate_threshold)) else 0,
                round(float(pressures[i]) / max(tie_eps, 1e-6)) if i < len(pressures) else 0,
                int(case_counts[i]) if i < len(case_counts) else 0,
                int(self.agents[i].homogeneity_count),
                float(pressures[i]) if i < len(pressures) else 0.0,
            ),
            reverse=True,
        )
        ids = [
            i for i in ids
            if (
                (i < len(invalid_rates) and float(invalid_rates[i]) >= float(self.cfg.invalid_repair_rate_threshold))
                or (float(pressures[i]) if i < len(pressures) else 0.0) >= max(0.0, max_pressure - tie_eps)
            )
        ]
        if not ids:
            ids = list(range(len(self.agents)))
        active = sum(1 for a in self.agents if a.homogeneity_count > 0)
        return ids[: (2 if active >= 2 else 1)]

    def select_error_agents_for_update(self) -> List[int]:
        if not self.is_update_window_ready():
            return []
        wrong_counts = [0 for _ in range(len(self.agents))]
        team_wrong_counts = [0 for _ in range(len(self.agents))]
        seen_counts = [0 for _ in range(len(self.agents))]
        for rec in self.recent_window_records:
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
            individual = list(metrics.get("individual_correct", []))
            team_correct = int(metrics.get("vote_correct", 0) or 0)
            for agent_id in range(len(self.agents)):
                if agent_id >= len(individual):
                    continue
                seen_counts[agent_id] += 1
                if not int(individual[agent_id]):
                    wrong_counts[agent_id] += 1
                    if not team_correct:
                        team_wrong_counts[agent_id] += 1
        ids = list(range(len(self.agents)))
        random.shuffle(ids)
        ids.sort(
            key=lambda i: (
                int(wrong_counts[i]),
                int(team_wrong_counts[i]),
                seen_counts[i] - wrong_counts[i],
            ),
            reverse=True,
        )
        ids = [i for i in ids if wrong_counts[i] > 0]
        return ids[: (2 if len(ids) >= 2 else 1)]

    def select_reward_agents_for_update(self, diagnosis: Dict[str, Any], metrics: Dict[str, Any]) -> List[int]:
        if str(getattr(self.cfg, "target_selector_mode", "legacy")) == "hybrid_competence_boundary":
            selected = self._select_hybrid_reward_agents(diagnosis)
            if self._is_stable_qd_lineage() and bool(self.cfg.target_selector_fairness_enabled):
                rows = diagnosis.get("hybrid_selector_diagnostics", [])
                positive = [row for row in rows if float(row.get("hybrid_target_score", 0.0) or 0.0) > 0.0]
                epoch = int(getattr(self, "competence_phase_epoch", 1))
                minimum = int(self.cfg.min_optimizer_updates_per_agent_per_epoch)
                under_minimum = [
                    row for row in positive
                    if int(self.per_agent_optimizer_update_count.get(f"{epoch}:{int(row['agent_id'])}", 0)) < minimum
                ]
                if not positive:
                    diagnosis["fairness_slot_selected"] = None
                    diagnosis["fairness_slot_skipped_no_evidence"] = True
                elif under_minimum:
                    fairness = min(
                        under_minimum,
                        key=lambda row: (
                            int(self.per_agent_optimizer_update_count.get(f"{epoch}:{int(row['agent_id'])}", 0)),
                            -float(row.get("hybrid_target_score", 0.0) or 0.0), int(row["agent_id"]),
                        ),
                    )
                    fairness_id = int(fairness["agent_id"])
                    selected = (selected[:1] + ([fairness_id] if fairness_id not in selected[:1] else selected[1:2]))[:2]
                    diagnosis["fairness_slot_selected"] = fairness_id
                    diagnosis["fairness_slot_skipped_no_evidence"] = False
                else:
                    selected = [int(row["agent_id"]) for row in sorted(
                        positive, key=lambda row: (-float(row.get("hybrid_target_score", 0.0) or 0.0), int(row["agent_id"])),
                    )[:2]]
                    diagnosis["fairness_slot_selected"] = None
                    diagnosis["fairness_slot_skipped_no_evidence"] = False
            return selected
        if bool(getattr(self.cfg, "boundary_selector_enabled", False)):
            return self._select_boundary_reward_agents(diagnosis)
        error_counts = list(diagnosis.get("per_agent_error_count", []))
        team_wrong_counts = list(diagnosis.get("per_agent_team_wrong_error_count", []))
        invalid_rates = list(diagnosis.get("per_agent_invalid_rate", []))
        pivotal_fix_counts = list(diagnosis.get("per_agent_pivotal_fix_count", []))
        dominant_wrong_counts = list(diagnosis.get("per_agent_dominant_wrong_redundancy_count", []))

        ids = list(range(len(self.agents)))
        random.shuffle(ids)

        def value(rows: List[Any], idx: int, default: float = 0.0) -> float:
            if idx >= len(rows):
                return float(default)
            try:
                return float(rows[idx])
            except Exception:
                return float(default)

        scored = []
        for agent_id in ids:
            base_score = (
                3.0 * value(error_counts, agent_id)
                + 2.0 * value(team_wrong_counts, agent_id)
                + 2.0 * value(invalid_rates, agent_id)
                + 2.0 * value(pivotal_fix_counts, agent_id)
                + 1.0 * value(dominant_wrong_counts, agent_id)
            )
            if base_score > 0.0:
                scored.append((float(base_score), agent_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [agent_id for _, agent_id in scored]
        return selected[: (2 if len(selected) >= 2 else 1)]

    def _select_hybrid_reward_agents(self, diagnosis: Dict[str, Any]) -> List[int]:
        strength = self._clip01(float(getattr(self, "specialization_strength", 0.0)))
        weights = {
            "individual_error_rate": 1.0 - 0.4 * strength,
            "weakness_score": 0.5 - 0.2 * strength,
            "c1_creation_opportunity": 1.2 - 0.4 * strength,
            "c2_creation_opportunity": 1.0,
            "plurality_pivotal_fix_opportunity": 0.25 + strength,
            "dominant_wrong_redundancy": 0.2 + 0.8 * strength,
            "shared_error_residual": 0.8 * strength,
            "capability_gap_affinity": 0.5 * strength,
        }

        def value(name: str, agent_id: int) -> float:
            values = diagnosis.get(name, [])
            return self._clip01(float(values[agent_id])) if agent_id < len(values) else 0.0

        latest = dict(getattr(self, "latest_competence_probe_metrics", {}) or {})
        probe_acc = [float(v) for v in latest.get("per_agent_acc", [])]
        probe_mean = float(latest.get("mean_individual_acc", np.mean(probe_acc) if probe_acc else 0.0) or 0.0)
        pressure = diagnosis.get("per_agent_capability_pressure", [])
        gaps = diagnosis.get("capability_coverage_gap", {})
        diagnostics: List[Dict[str, Any]] = []
        ids = list(range(len(self.agents)))
        scored: List[Tuple[float, int]] = []
        for agent_id in ids:
            family_pressure = pressure[agent_id] if agent_id < len(pressure) and isinstance(pressure[agent_id], dict) else {}
            pressure_total = sum(max(0.0, float(v)) for v in family_pressure.values())
            capability_gap = 0.0
            if pressure_total > 0.0:
                capability_gap = sum(
                    max(0.0, float(family_pressure.get(family, 0.0)))
                    * max(0.0, float(gaps.get(family, 0.0)))
                    for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
                ) / pressure_total
                capability_gap /= max(1.0, float(len(self.agents)))
            components = {
                "individual_error_rate": value("per_agent_general_error_rate", agent_id),
                "weakness_score": self._clip01(max(0.0, probe_mean - probe_acc[agent_id])) if agent_id < len(probe_acc) else 0.0,
                "c1_creation_opportunity": value("per_agent_c1_creation_opportunity", agent_id),
                "c2_creation_opportunity": value("per_agent_c2_creation_opportunity", agent_id),
                "plurality_pivotal_fix_opportunity": value("per_agent_plurality_pivotal_fix_rate", agent_id),
                "dominant_wrong_redundancy": value("per_agent_dominant_wrong_rate", agent_id),
                "shared_error_residual": value("per_agent_shared_error_rate", agent_id),
                "capability_gap_affinity": self._clip01(capability_gap),
            }
            score = sum(weights[name] * components[name] for name in weights)
            diagnostics.append({
                "agent_id": agent_id,
                "applied_specialization_strength": strength,
                **components,
                "hybrid_target_score": float(score),
                "selected": False,
            })
            if score > 0.0:
                scored.append((float(score), agent_id))
        scored.sort(key=lambda row: (-row[0], row[1]))
        selected = [agent_id for _, agent_id in scored[: (2 if len(scored) >= 2 else 1)]]
        for row in diagnostics:
            row["selected"] = int(row["agent_id"]) in selected
        diagnosis["hybrid_selector_weights"] = weights
        diagnosis["hybrid_selector_diagnostics"] = sorted(diagnostics, key=lambda row: int(row["agent_id"]))
        return selected

    def _select_boundary_reward_agents(self, diagnosis: Dict[str, Any]) -> List[int]:
        ids = list(range(len(self.agents)))
        random.shuffle(ids)

        def rate(name: str, agent_id: int) -> float:
            values = diagnosis.get(name, [])
            return float(values[agent_id]) if agent_id < len(values) else 0.0

        pressures = diagnosis.get("per_agent_capability_pressure", [])
        gaps = diagnosis.get("capability_coverage_gap", {})
        scored: List[Tuple[float, int]] = []
        for agent_id in ids:
            plurality_boundary = bool(getattr(self.cfg, "competence_depth_enabled", False))
            pivotal_fix_rate = rate(
                "per_agent_plurality_pivotal_fix_rate"
                if plurality_boundary and diagnosis.get("per_agent_plurality_pivotal_fix_rate")
                else "per_agent_pivotal_fix_rate",
                agent_id,
            )
            pivotal_hold_rate = rate(
                "per_agent_plurality_pivotal_hold_rate"
                if plurality_boundary and diagnosis.get("per_agent_plurality_pivotal_hold_rate")
                else "per_agent_pivotal_hold_rate",
                agent_id,
            )
            boundary_error_rate = rate(
                "per_agent_plurality_boundary_error_rate"
                if plurality_boundary and diagnosis.get("per_agent_plurality_boundary_error_rate")
                else "per_agent_near_boundary_error_rate",
                agent_id,
            )
            base_score = (
                4.0 * pivotal_fix_rate
                + 2.0 * boundary_error_rate
                + 0.5 * pivotal_hold_rate
                + 1.5 * rate("per_agent_dominant_wrong_rate", agent_id)
                + 1.0 * rate("per_agent_shared_error_rate", agent_id)
                + 0.5 * rate("per_agent_general_error_rate", agent_id)
                + 1.0 * rate("per_agent_invalid_rate", agent_id)
            )
            affinity = coverage_bonus = 0.0
            if agent_id < len(pressures) and isinstance(pressures[agent_id], dict):
                family_pressure = {
                    family: max(0.0, float(pressures[agent_id].get(family, 0.0)))
                    for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
                }
                total_pressure = sum(family_pressure.values())
                if total_pressure > 0.0:
                    profile = self.agents[agent_id].capability_profile
                    affinity = sum(
                        float(profile.get(family, 0.0)) * family_pressure[family] / total_pressure
                        for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
                    )
                    coverage_bonus = sum(
                        float(gaps.get(family, 0.0)) * family_pressure[family] / total_pressure
                        for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
                    ) / max(1, (len(self.agents) // 2) + 1)
            score = (
                base_score
                + self._effective_progressive_weight(float(getattr(self.cfg, "capability_affinity_weight", 0.25))) * affinity
                + self._effective_progressive_weight(float(getattr(self.cfg, "capability_coverage_gap_weight", 0.25))) * coverage_bonus
            )
            if bool(getattr(self.cfg, "competence_depth_enabled", False)):
                references = self.previous_epoch_per_agent_acc or list(diagnosis.get("per_agent_accuracy", []))
                reference = float(references[agent_id]) if agent_id < len(references) else 0.0
                deficit = max(0.0, float(getattr(self.cfg, "competence_floor_high", 0.65)) - reference)
                score += (
                    (1.0 - float(self.specialization_strength))
                    * float(getattr(self.cfg, "competence_selector_weight", 1.0))
                    * deficit
                )
            if base_score > 0.0 and score > 0.0:
                scored.append((score, agent_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [agent_id for _, agent_id in scored]
        return selected[: (2 if len(selected) >= 2 else 1)]

    def _window_accuracy_diagnosis(self, window_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        per_agent_seen = [0 for _ in range(len(self.agents))]
        per_agent_correct = [0 for _ in range(len(self.agents))]
        per_agent_team_wrong = [0 for _ in range(len(self.agents))]
        all_error_cases: List[Dict[str, Any]] = []
        target_error_cases: List[Dict[str, Any]] = []
        focus_cases: List[Dict[str, Any]] = []
        for idx, rec in enumerate(window_records):
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
            traces = list(rec.get("traces", []))
            answers = list(rec.get("answers", []))
            prompts = list(rec.get("prompts", []))
            individual = list(metrics.get("individual_correct", []))
            invalids = list(metrics.get("invalid_flags", []))
            team_correct = int(metrics.get("vote_correct", 0) or 0)
            vote_answer = str(metrics.get("vote_answer", ""))
            gold = str(rec.get("gold", ""))
            question_hash = str(rec.get("question_hash", ""))
            row_cases = []
            for agent_id in range(len(self.agents)):
                if agent_id >= len(individual):
                    continue
                per_agent_seen[agent_id] += 1
                per_agent_correct[agent_id] += int(individual[agent_id])
                target_invalid = int(invalids[agent_id]) if agent_id < len(invalids) else 0
                if int(individual[agent_id]) and not target_invalid:
                    if bool(getattr(self.cfg, "boundary_selector_enabled", False)) and gold:
                        context = self._behavior_context_for_baseline(
                            agent_id=agent_id,
                            answers=answers,
                            gold=gold,
                            rollout=metrics,
                            question_hash=question_hash,
                        )
                        if context == BehaviorContext.TARGET_CORRECT_PIVOTAL_HOLD.value:
                            target_error_cases.append({
                                "case_id": self._hash(f"{question_hash}|pivotal_correct_protection|{agent_id}|{idx}"),
                                "case_type": "pivotal_correct_protection",
                                "window_index": idx,
                                "question_hash": question_hash,
                                "sample_hash": question_hash,
                                "target_agent_id": agent_id,
                                "target_trace_preview": self._redact_optimizer_text(traces[agent_id] if agent_id < len(traces) else ""),
                                "target_answer_preview": self._answer_behavior_preview(answers[agent_id] if agent_id < len(answers) else ""),
                                "target_invalid": False,
                                "target_correct": True,
                                "team_correct": bool(team_correct),
                                "peer_correct_available": True,
                                "error_pattern": "pivotal_correct_behavior",
                                "repair_hint": "preserve the local mechanism that keeps this answer correct near the vote boundary",
                                "capability_residual_family": CapabilityResidualFamily.UNKNOWN.value,
                                "confidence": 1.0,
                                "vote_context": context,
                            })
                    continue
                if not team_correct:
                    per_agent_team_wrong[agent_id] += 1
                peer_correct_flags = [
                    int(individual[i])
                    for i in range(len(individual))
                    if i != agent_id
                ]
                peer_correct_ids = [
                    int(i)
                    for i in range(len(individual))
                    if i != agent_id and int(individual[i])
                ]
                peer_trace_candidates = [
                    str(traces[i])
                    for i in peer_correct_ids
                    if i < len(traces)
                ]
                if not peer_trace_candidates:
                    peer_trace_indices = [
                        i for i in range(len(traces))
                        if i != agent_id
                    ][:2]
                    peer_trace_candidates = [
                        str(traces[i]) for i in peer_trace_indices
                    ]
                    selected_peer_flags = [
                        int(individual[i]) for i in peer_trace_indices
                        if i < len(individual)
                    ]
                else:
                    selected_peer_flags = [1 for _ in peer_trace_candidates]
                error_info = self._infer_target_error_pattern(
                    target_trace=str(traces[agent_id]) if agent_id < len(traces) else "",
                    target_answer=str(answers[agent_id]) if agent_id < len(answers) else "",
                    peer_traces=peer_trace_candidates,
                    rollout=metrics,
                    agent_id=agent_id,
                )
                context = self._behavior_context_for_baseline(
                    agent_id=agent_id,
                    answers=answers,
                    gold=gold,
                    rollout=metrics,
                    question_hash=question_hash,
                ) if gold and (
                    bool(getattr(self.cfg, "boundary_selector_enabled", False))
                    or self._v7_residual_protocol_enabled()
                ) else BehaviorContext.INVALID.value
                peer_wrong_count = sum(1 for flag in peer_correct_flags if not flag)
                gold_count = int(metrics.get("gold_vote_count", sum(individual)) or 0)
                largest_wrong = int(metrics.get("largest_wrong_vote_count", 0) or 0)
                if bool(getattr(self.cfg, "boundary_selector_enabled", False)):
                    target_answer = str(answers[agent_id] if agent_id < len(answers) else "")
                    wrong_counts = Counter(
                        str(answer or "").strip()
                        for i, answer in enumerate(answers)
                        if i < len(individual) and not int(individual[i]) and str(answer or "").strip()
                    )
                    in_dominant_wrong = bool(
                        target_answer.strip()
                        and wrong_counts.get(target_answer.strip(), 0) == max(wrong_counts.values(), default=0)
                        and max(wrong_counts.values(), default=0) > 1
                    )
                    if target_invalid:
                        target_case_type = "target_invalid"
                    elif context == BehaviorContext.TEAM_WRONG_PIVOTAL_FIX.value:
                        target_case_type = "target_wrong_pivotal_vote_fix"
                    elif abs(gold_count - largest_wrong) <= 1:
                        target_case_type = "target_wrong_near_vote_boundary"
                    elif in_dominant_wrong:
                        target_case_type = "target_wrong_dominant_wrong_cluster"
                    elif peer_wrong_count > 0:
                        target_case_type = "target_wrong_shared_error"
                    elif any(peer_correct_flags) and not team_correct:
                        target_case_type = "target_wrong_peer_correct_nonboundary"
                    else:
                        target_case_type = "target_wrong_vote_already_correct"
                elif not int(individual[agent_id]) and any(peer_correct_flags):
                    target_case_type = "target_agent_wrong_and_peer_correct"
                elif not int(individual[agent_id]) and team_correct:
                    target_case_type = "target_agent_wrong_and_vote_correct"
                elif not int(individual[agent_id]):
                    target_case_type = "target_agent_wrong_and_vote_wrong"
                else:
                    target_case_type = "target_agent_invalid"
                peer_summary = self._peer_behavior_summary(peer_trace_candidates, peer_correct_flags=selected_peer_flags)
                target_words = len(re.findall(r"\w+", str(traces[agent_id]) if agent_id < len(traces) else ""))
                peer_summary["peer_longer_than_target"] = bool(peer_summary.get("mean_peer_trace_words", 0.0) > max(0, target_words))
                case = {
                    "case_id": self._hash(f"{rec.get('question_hash', '')}|accuracy_error|{agent_id}|{idx}"),
                    "case_type": "target_agent_answer_error",
                    "window_index": idx,
                    "sample_hash": rec.get("question_hash", ""),
                    "target_agent_id": agent_id,
                    "target_trace_preview": self._trace_method_preview(traces[agent_id]) if agent_id < len(traces) else "",
                    "target_answer": str(answers[agent_id]) if agent_id < len(answers) else "",
                    "team_vote_answer": vote_answer,
                    "team_correct": bool(team_correct),
                    "target_prompt_preview": normalize_spaces(prompts[agent_id])[:260] if agent_id < len(prompts) else "",
                }
                target_error_cases.append(
                    {
                        "case_id": self._hash(f"{rec.get('question_hash', '')}|target_error_repair|{agent_id}|{idx}"),
                        "case_type": target_case_type,
                        "window_index": idx,
                        "question_hash": rec.get("question_hash", ""),
                        "sample_hash": rec.get("question_hash", ""),
                        "target_agent_id": agent_id,
                        "target_trace_preview": self._redact_optimizer_text(traces[agent_id] if agent_id < len(traces) else ""),
                        "target_answer_preview": self._answer_behavior_preview(answers[agent_id] if agent_id < len(answers) else ""),
                        "peer_trace_preview": peer_summary.get("peer_trace_previews", []),
                        "peer_behavior_summary": peer_summary,
                        "target_invalid": bool(target_invalid),
                        "target_correct": bool(int(individual[agent_id])),
                        "team_correct": bool(team_correct),
                        "peer_correct_available": bool(any(peer_correct_flags)),
                        "error_pattern": str(error_info.get("error_pattern", "unknown_error_pattern")),
                        "repair_hint": str(error_info.get("repair_hint", "")),
                        "capability_residual_family": str(error_info.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value)),
                        "confidence": float(error_info.get("confidence", 0.0) or 0.0),
                        "peer_wrong_count": int(peer_wrong_count),
                        "baseline_correct_count": int(sum(int(value) for value in individual)),
                        "vote_context": context,
                        "target_prompt_preview": normalize_spaces(prompts[agent_id])[:260] if agent_id < len(prompts) else "",
                    }
                )
                row_cases.append(case)
                all_error_cases.append(case)
            if row_cases:
                focus_cases.append(
                    {
                        "window_index": idx,
                        "team_correct": bool(team_correct),
                        "wrong_agent_ids": [int(c.get("target_agent_id", -1)) for c in row_cases],
                        "vote_answer": vote_answer,
                    }
                )
        per_agent_accuracy = [
            float(per_agent_correct[i] / per_agent_seen[i]) if per_agent_seen[i] else 0.0
            for i in range(len(self.agents))
        ]
        current_prompts = [
            {"agent_id": i, "prompt_preview": normalize_spaces(p)[:260], "prompt_hash": self._hash(p)}
            for i, p in enumerate(self._active_prompt_list())
        ]
        return {
            "window_size": len(window_records),
            "prompt_roles": current_prompts,
            "focus_cases": focus_cases[:5],
            "error_cases": all_error_cases,
            "target_error_cases": target_error_cases,
            "per_agent_accuracy": per_agent_accuracy,
            "per_agent_error_count": [int(per_agent_seen[i] - per_agent_correct[i]) for i in range(len(self.agents))],
            "per_agent_team_wrong_error_count": per_agent_team_wrong,
            "team_accuracy": float(np.mean([int((rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}).get("vote_correct", 0) or 0) for rec in window_records])) if window_records else 0.0,
        }

    def _window_update_diagnosis(self, window_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        scored = []
        all_homogeneous_cases: List[Dict[str, Any]] = []
        all_validity_cases: List[Dict[str, Any]] = []
        num_agents = len(self.agents)
        per_agent_invalid = [0 for _ in range(num_agents)]
        per_agent_seen = [0 for _ in range(num_agents)]
        per_agent_pressure_rows = [[] for _ in range(num_agents)]
        pivotal_fix_counts = [0 for _ in range(num_agents)]
        dominant_wrong_counts = [0 for _ in range(num_agents)]
        near_boundary_error_counts = [0 for _ in range(num_agents)]
        shared_error_counts = [0 for _ in range(num_agents)]
        c1_creation_counts = [0 for _ in range(num_agents)]
        c2_creation_counts = [0 for _ in range(num_agents)]
        pivotal_hold_counts = [0 for _ in range(num_agents)]
        vote_values: List[int] = []
        vote_margin_values: List[float] = []
        boundary_diversity_values: List[float] = []
        embedding_overlap_values: List[float] = []

        for idx, rec in enumerate(window_records):
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
            individual = [int(value) for value in metrics.get("individual_correct", [])]
            invalids = [int(value) for value in metrics.get("invalid_flags", [])]
            pressures = list(metrics.get("per_agent_overlap", []))
            answers = [str(answer or "").strip() for answer in rec.get("answers", [])]
            gold = str(rec.get("gold", ""))
            question_hash = str(rec.get("question_hash", ""))
            vote_correct = int(metrics.get("vote_correct", 0) or 0)
            vote_tie = bool(metrics.get("vote_tie", False))
            gold_count = int(metrics.get("gold_vote_count", sum(individual)) or 0)
            largest_wrong = int(metrics.get("largest_wrong_vote_count", 0) or 0)
            margin = float(metrics.get("normalized_vote_margin", -1.0) if metrics.get("normalized_vote_margin") is not None else -1.0)
            boundary_diversity = float(metrics.get("boundary_useful_diversity", 0.0) or 0.0)
            invalid_rate = float(metrics.get("invalid_rate", 0.0) or 0.0)
            reward_pressure = float(1 - vote_correct) + max(0.0, -margin) + invalid_rate
            scored.append((reward_pressure, idx, rec))
            vote_values.append(vote_correct)
            vote_margin_values.append(margin)
            boundary_diversity_values.append(boundary_diversity)
            embedding_overlap_values.append(float(metrics.get("mean_embedding_overlap", 0.0) or 0.0))
            all_homogeneous_cases.extend(list(rec.get("homogeneous_cases", [])))
            all_validity_cases.extend(list(rec.get("validity_cases", [])))

            wrong_counts = Counter(
                answers[agent_id]
                for agent_id in range(min(len(answers), len(individual)))
                if answers[agent_id] and not individual[agent_id]
            )
            for agent_id in range(num_agents):
                if agent_id < len(invalids):
                    per_agent_seen[agent_id] += 1
                    per_agent_invalid[agent_id] += invalids[agent_id]
                if agent_id < len(pressures):
                    per_agent_pressure_rows[agent_id].append(float(pressures[agent_id]))
                if agent_id >= len(individual):
                    continue
                if individual[agent_id]:
                    if bool(getattr(self.cfg, "competence_depth_enabled", False)):
                        without_target = list(answers)
                        if agent_id < len(without_target):
                            without_target[agent_id] = ""
                        without_vote = self._vote_with_diagnostics(without_target, question_hash=question_hash)
                        if vote_correct and not self.task_spec.match_answer(
                            str(without_vote.get("vote_answer", "")), gold
                        ):
                            pivotal_hold_counts[agent_id] += 1
                    elif gold_count - 1 <= largest_wrong:
                        pivotal_hold_counts[agent_id] += 1
                    continue
                if gold_count == 0:
                    c1_creation_counts[agent_id] += 1
                elif gold_count == 1:
                    c2_creation_counts[agent_id] += 1
                peer_wrong_count = sum(
                    int(not individual[peer_id])
                    for peer_id in range(len(individual))
                    if peer_id != agent_id
                )
                plurality_pivotal = False
                if bool(getattr(self.cfg, "competence_depth_enabled", False)):
                    counterfactual_answers = list(answers)
                    if agent_id < len(counterfactual_answers):
                        counterfactual_answers[agent_id] = gold
                    counterfactual_vote = self._vote_with_diagnostics(
                        counterfactual_answers, question_hash=question_hash
                    )
                    plurality_pivotal = bool(
                        not vote_correct
                        and self.task_spec.match_answer(str(counterfactual_vote.get("vote_answer", "")), gold)
                    )
                    if plurality_pivotal:
                        near_boundary_error_counts[agent_id] += 1
                elif abs(gold_count - largest_wrong) <= 1:
                    near_boundary_error_counts[agent_id] += 1
                if peer_wrong_count > 0:
                    shared_error_counts[agent_id] += 1
                answer = answers[agent_id] if agent_id < len(answers) else ""
                remaining_wrong = dict(wrong_counts)
                if answer and answer in remaining_wrong:
                    remaining_wrong[answer] -= 1
                    if remaining_wrong[answer] <= 0:
                        remaining_wrong.pop(answer, None)
                counterfactual_largest_wrong = max(remaining_wrong.values(), default=0)
                if (
                    plurality_pivotal
                    if bool(getattr(self.cfg, "competence_depth_enabled", False))
                    else ((not vote_correct or vote_tie) and gold_count + 1 > counterfactual_largest_wrong)
                ):
                    pivotal_fix_counts[agent_id] += 1
                if answer and gold_count > 0 and wrong_counts.get(answer, 0) == largest_wrong and abs(gold_count - largest_wrong) <= 1:
                    dominant_wrong_counts[agent_id] += 1

        scored.sort(key=lambda item: item[0], reverse=True)
        focus_cases = []
        for score, idx, rec in scored[: min(3, max(1, len(scored)))]:
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
            individual = list(metrics.get("individual_correct", []))
            focus_cases.append(
                {
                    "window_index": idx,
                    "reward_pressure": round(score, 4),
                    "vote_correct": bool(metrics.get("vote_correct", 0)),
                    "vote_tie": bool(metrics.get("vote_tie", False)),
                    "normalized_vote_margin": float(metrics.get("normalized_vote_margin", -1.0) if metrics.get("normalized_vote_margin") is not None else -1.0),
                    "boundary_useful_diversity": float(metrics.get("boundary_useful_diversity", 0.0) or 0.0),
                    "wrong_agent_ids": [agent_id for agent_id, correct in enumerate(individual) if not int(correct)],
                    "invalid_rate": float(metrics.get("invalid_rate", 0.0) or 0.0),
                }
            )

        homogeneous_case_counts = [0 for _ in range(num_agents)]
        for case in all_homogeneous_cases:
            agent_id = int(case.get("target_agent_id", -1))
            if 0 <= agent_id < num_agents:
                homogeneous_case_counts[agent_id] += 1
        accuracy_diagnosis = self._window_accuracy_diagnosis(window_records)
        boundary_case_types = {
            "pivotal": "target_wrong_pivotal_vote_fix",
            "near": "target_wrong_near_vote_boundary",
            "shared": "target_wrong_shared_error",
            "dominant": "target_wrong_dominant_wrong_cluster",
            "hold": "pivotal_correct_protection",
        }
        boundary_counts = {
            key: [0 for _ in range(num_agents)] for key in boundary_case_types
        }
        capability_pressure = [
            {family: 0.0 for family in CAPABILITY_RESIDUAL_FAMILY_NAMES}
            for _ in range(num_agents)
        ]
        for case in accuracy_diagnosis.get("target_error_cases", []):
            if not isinstance(case, dict):
                continue
            agent_id = int(case.get("target_agent_id", -1))
            if not 0 <= agent_id < num_agents:
                continue
            case_type = str(case.get("case_type", ""))
            for key, expected in boundary_case_types.items():
                boundary_counts[key][agent_id] += int(case_type == expected)
            family = str(case.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value))
            if family not in capability_pressure[agent_id]:
                family = CapabilityResidualFamily.UNKNOWN.value
            capability_pressure[agent_id][family] += 1.0
        seen = [max(1, int(value)) for value in per_agent_seen]
        coverage_depth = {
            family: sum(
                int(
                    agent.capability_evidence[family].support > 0
                    and agent.capability_evidence[family].posterior_value > 0.0
                )
                for agent in self.agents
            )
            for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
        }
        majority_threshold = 2 if bool(getattr(self.cfg, "competence_depth_enabled", False)) else (num_agents // 2) + 1
        coverage_gap = {
            family: max(0, majority_threshold - depth)
            for family, depth in coverage_depth.items()
        }
        current_prompts = [
            {"agent_id": i, "prompt_preview": normalize_spaces(prompt)[:260], "prompt_hash": self._hash(prompt)}
            for i, prompt in enumerate(self._active_prompt_list())
        ]
        return {
            "diagnosis_type": "vote_update",
            "window_size": len(window_records),
            "focus_cases": focus_cases,
            "prompt_roles": current_prompts,
            "mean_window_overlap": float(np.mean(embedding_overlap_values)) if embedding_overlap_values else 0.0,
            "mean_embedding_overlap": float(np.mean(embedding_overlap_values)) if embedding_overlap_values else 0.0,
            "mean_reward_pressure": float(np.mean([item[0] for item in scored])) if scored else 0.0,
            "window_vote_acc": float(np.mean(vote_values)) if vote_values else 0.0,
            "window_mean_vote_margin": float(np.mean(vote_margin_values)) if vote_margin_values else -1.0,
            "window_mean_boundary_useful_diversity": self._clip01(float(np.mean(boundary_diversity_values))) if boundary_diversity_values else 0.0,
            "homogeneous_cases": sorted(all_homogeneous_cases, key=lambda case: float(case.get("pair_overlap", 0.0)), reverse=True),
            "validity_cases": all_validity_cases,
            "error_cases": list(accuracy_diagnosis.get("error_cases", [])),
            "target_error_cases": list(accuracy_diagnosis.get("target_error_cases", [])),
            "per_agent_accuracy": list(accuracy_diagnosis.get("per_agent_accuracy", [])),
            "per_agent_error_count": list(accuracy_diagnosis.get("per_agent_error_count", [])),
            "per_agent_team_wrong_error_count": list(accuracy_diagnosis.get("per_agent_team_wrong_error_count", [])),
            "team_accuracy": float(accuracy_diagnosis.get("team_accuracy", 0.0)),
            "homogeneous_case_counts": homogeneous_case_counts,
            "per_agent_invalid_rate": [float(per_agent_invalid[i] / per_agent_seen[i]) if per_agent_seen[i] else 0.0 for i in range(num_agents)],
            "per_agent_overlap_pressure": [float(np.mean(values)) if values else 0.0 for values in per_agent_pressure_rows],
            "per_agent_pivotal_fix_count": pivotal_fix_counts,
            "per_agent_dominant_wrong_redundancy_count": dominant_wrong_counts,
            "per_agent_pivotal_fix_rate": [pivotal_fix_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_plurality_pivotal_fix_rate": [pivotal_fix_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_plurality_boundary_error_rate": [near_boundary_error_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_near_boundary_error_rate": [near_boundary_error_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_shared_error_rate": [shared_error_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_c1_creation_opportunity": [c1_creation_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_c2_creation_opportunity": [c2_creation_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_dominant_wrong_rate": [dominant_wrong_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_general_error_rate": [
                float(accuracy_diagnosis.get("per_agent_error_count", [0] * num_agents)[i]) / seen[i]
                for i in range(num_agents)
            ],
            "per_agent_pivotal_hold_rate": [pivotal_hold_counts[i] / seen[i] for i in range(num_agents)],
            "per_agent_plurality_pivotal_hold_rate": [pivotal_hold_counts[i] / seen[i] for i in range(num_agents)],
            "pivotal_definition": "actual_plurality_counterfactual" if bool(getattr(self.cfg, "competence_depth_enabled", False)) else "legacy_vote_boundary",
            "per_agent_capability_pressure": capability_pressure,
            "capability_coverage_depth": coverage_depth,
            "capability_coverage_gap": coverage_gap,
            "homogeneity_overlap_threshold": float(self.cfg.homogeneity_overlap_threshold),
        }

    def _cases_for_agent(self, diagnosis: Dict[str, Any], agent_id: int) -> List[Dict[str, Any]]:
        return [
            c for c in diagnosis.get("homogeneous_cases", [])
            if isinstance(c, dict) and int(c.get("target_agent_id", -1)) == int(agent_id)
        ]

    def _validity_cases_for_agent(self, diagnosis: Dict[str, Any], agent_id: int) -> List[Dict[str, Any]]:
        return [
            c for c in diagnosis.get("validity_cases", [])
            if isinstance(c, dict) and int(c.get("target_agent_id", -1)) == int(agent_id)
        ]

    def _accuracy_cases_for_agent(self, diagnosis: Dict[str, Any], agent_id: int) -> List[Dict[str, Any]]:
        return [
            c for c in diagnosis.get("error_cases", [])
            if isinstance(c, dict) and int(c.get("target_agent_id", -1)) == int(agent_id)
        ]

    def _target_error_cases_for_agent(self, diagnosis: Dict[str, Any], agent_id: int) -> List[Dict[str, Any]]:
        if bool(getattr(self.cfg, "boundary_selector_enabled", False)):
            priority = {
                "target_wrong_pivotal_vote_fix": 0,
                "target_wrong_near_vote_boundary": 1,
                "target_wrong_shared_error": 2,
                "target_wrong_dominant_wrong_cluster": 3,
                "target_wrong_peer_correct_nonboundary": 4,
                "target_wrong_vote_already_correct": 5,
                "pivotal_correct_protection": 6,
                "target_invalid": 7,
            }
        else:
            priority = {
                "target_agent_wrong_and_peer_correct": 0,
                "target_agent_wrong_and_vote_correct": 1,
                "target_agent_wrong_and_vote_wrong": 2,
                "target_agent_invalid": 3,
            }
        cases = [
            c for c in diagnosis.get("target_error_cases", [])
            if isinstance(c, dict) and int(c.get("target_agent_id", -1)) == int(agent_id)
        ]
        cases.sort(key=lambda c: (priority.get(str(c.get("case_type", "")), 99), int(c.get("window_index", 0) or 0)))
        return cases

    def _window_random_case_summaries(self, agent_id: int, limit: int) -> List[Dict[str, Any]]:
        if limit <= 0 or not self.recent_window_records:
            return []
        records = list(self.recent_window_records)
        random.shuffle(records)
        rows = []
        for rec in records[:limit]:
            traces = list(rec.get("traces", []))
            answers = list(rec.get("answers", []))
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
            if agent_id >= len(traces):
                continue
            rows.append(
                {
                    "case_type": "random_window_case",
                    "sample_hash": rec.get("question_hash", ""),
                    "target_agent_id": agent_id,
                    "target_trace_preview": self._trace_method_preview(traces[agent_id]),
                    "target_answer": str(answers[agent_id]) if agent_id < len(answers) else "",
                    "target_overlap_pressure": float(metrics.get("per_agent_overlap", [0.0] * len(self.agents))[agent_id]) if agent_id < len(metrics.get("per_agent_overlap", [])) else 0.0,
                    "team_correct": bool(metrics.get("vote_correct", 0)),
                }
            )
        return rows

    def _build_case_generation_batches(self, agent_id: int, diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
        target_error_cases = self._target_error_cases_for_agent(diagnosis, agent_id)
        target_error_limit = max(1, int(self.cfg.max_homogeneous_cases_per_agent))
        if self._is_accuracy_only_mode():
            error_cases = self._accuracy_cases_for_agent(diagnosis, agent_id)
            random_cases = [
                c for c in self._window_random_case_summaries(agent_id, max(0, int(self.cfg.random_window_cases_per_agent)))
                if isinstance(c, dict)
            ]
            batches = [
                {
                    "batch_type": "target_error_repair",
                    "priority": -2,
                    "cases": target_error_cases[:target_error_limit],
                    "purpose": "repair target-agent observed error patterns before changing diversity",
                },
                {
                    "batch_type": "accuracy_error_cases",
                    "priority": 0,
                    "cases": error_cases[: max(1, int(self.cfg.max_homogeneous_cases_per_agent))],
                    "purpose": "repair target-agent answer mistakes observed in the current update window",
                },
                {
                    "batch_type": "mixed_window_accuracy_cases",
                    "priority": 1,
                    "cases": random_cases,
                    "purpose": "keep the revised prompt robust on nearby window examples while improving accuracy",
                },
            ]
            return [b for b in batches if b.get("cases") or str(b.get("batch_type")) in {"target_error_repair", "accuracy_error_cases"}]
        if self._is_v82_hybrid():
            def take(predicate: Any, count: int, used: set) -> List[Dict[str, Any]]:
                if count <= 0:
                    return []
                rows = []
                for case in target_error_cases:
                    case_id = str(case.get("case_id", ""))
                    if case_id in used or not predicate(case):
                        continue
                    rows.append(case)
                    used.add(case_id)
                    if len(rows) >= count:
                        break
                return rows

            used: set = set()
            c1 = take(lambda case: int(case.get("baseline_correct_count", -1)) == 0, 1, used)
            c2 = take(lambda case: int(case.get("baseline_correct_count", -1)) == 1, 1, used)
            boundary = take(
                lambda case: str(case.get("case_type", "")) in {
                    "target_wrong_pivotal_vote_fix", "target_wrong_near_vote_boundary"
                },
                1,
                used,
            )
            residual = take(
                lambda case: str(case.get("case_type", "")) in {
                    "target_wrong_shared_error", "target_wrong_dominant_wrong_cluster"
                },
                1 if float(getattr(self, "specialization_strength", 0.0)) > 0.0 else 0,
                used,
            )
            general = take(lambda case: not bool(case.get("target_correct", False)), 2, used)
            budget = max(1, int(self.cfg.max_homogeneous_cases_per_agent) + int(self.cfg.random_window_cases_per_agent))
            chosen = general + c1 + c2 + boundary + residual
            for case in target_error_cases:
                if len(chosen) >= budget:
                    break
                case_id = str(case.get("case_id", ""))
                if case_id not in used:
                    chosen.append(case)
                    used.add(case_id)
            chosen_ids = {str(case.get("case_id", "")) for case in chosen}
            buckets = [
                ("general_error", general, "repair general target-agent errors and preserve competence"),
                ("c1_c2_creation", c1 + c2, "create C1/C2 coverage even when one repair cannot yet flip plurality"),
                ("actual_plurality_boundary", boundary, "repair cases verified by the actual plurality counterfactual"),
                ("residual_shared_error", residual, "reduce shared residual errors after specialization activates"),
            ]
            batches = []
            for priority, (batch_type, rows, purpose) in enumerate(buckets):
                retained = [case for case in rows if str(case.get("case_id", "")) in chosen_ids]
                if retained:
                    batches.append({"batch_type": batch_type, "priority": priority, "cases": retained, "purpose": purpose})
            if not batches and chosen:
                batches.append({"batch_type": "general_error", "priority": 0, "cases": chosen, "purpose": "repair general target-agent errors"})
            return batches
        if bool(getattr(self.cfg, "boundary_selector_enabled", False)):
            limit = max(1, int(self.cfg.max_homogeneous_cases_per_agent))
            by_type = lambda *names: [
                case for case in target_error_cases if str(case.get("case_type", "")) in set(names)
            ]
            pivotal = by_type("target_wrong_pivotal_vote_fix")
            near_shared = by_type(
                "target_wrong_near_vote_boundary",
                "target_wrong_shared_error",
                "target_wrong_dominant_wrong_cluster",
            )
            protection = by_type("pivotal_correct_protection")
            for state in self.agents[agent_id].accepted_behavior_archive[-5:]:
                for question_hash, entry in state.behavior_fingerprint.items():
                    target_correct = entry.target_correct if isinstance(entry, BehaviorFingerprintEntry) else bool(entry.get("target_correct", False))
                    team_correct = entry.team_vote_correct if isinstance(entry, BehaviorFingerprintEntry) else bool(entry.get("team_vote_correct", False))
                    if target_correct and not team_correct:
                        protection.append({
                            "case_id": self._hash(f"{question_hash}|historical_unique_correct|{agent_id}"),
                            "case_type": "pivotal_correct_protection",
                            "sample_hash": str(question_hash),
                            "target_agent_id": agent_id,
                            "target_correct": True,
                            "team_correct": False,
                            "historical_unique_correct": True,
                            "repair_hint": "preserve the mechanism that supplied a historically unique correct path",
                            "capability_residual_family": CapabilityResidualFamily.UNKNOWN.value,
                        })
            general = by_type(
                "target_wrong_peer_correct_nonboundary",
                "target_wrong_vote_already_correct",
                "target_invalid",
            )
            gaps = diagnosis.get("capability_coverage_gap", {})
            residual = [
                case for case in target_error_cases
                if str(case.get("case_type", "")) != "pivotal_correct_protection"
            ]
            residual.sort(
                key=lambda case: (
                    -float(gaps.get(str(case.get("capability_residual_family", "unknown")), 0.0)),
                    -float(self.agents[agent_id].capability_profile.get(str(case.get("capability_residual_family", "unknown")), 0.0)),
                    int(case.get("window_index", 0) or 0),
                )
            )
            random_cases = self._window_random_case_summaries(
                agent_id, max(0, int(self.cfg.random_window_cases_per_agent))
            )
            invalid_rate = float(diagnosis.get("per_agent_invalid_rate", [0.0] * len(self.agents))[agent_id])
            batches = [
                {
                    "batch_type": "pivotal_error_repair",
                    "priority": 0,
                    "cases": pivotal[:limit],
                    "purpose": "repair errors whose correction can directly recover a team vote",
                },
                {
                    "batch_type": "near_boundary_shared_error_repair",
                    "priority": 1,
                    "cases": near_shared[:limit],
                    "purpose": "reduce harmful shared-error mechanisms near the vote boundary",
                },
                {
                    "batch_type": "residual_capability_repair",
                    "priority": 2,
                    "cases": residual[:limit],
                    "purpose": "make one local executable repair for the highest-pressure residual capability family",
                },
                {
                    "batch_type": "pivotal_correct_protection",
                    "priority": 3,
                    "cases": protection[:limit],
                    "purpose": "preserve mechanisms that already supply pivotal correct votes",
                },
                {
                    "batch_type": "general_accuracy_repair",
                    "priority": 4,
                    "cases": general[:limit],
                    "purpose": "repair remaining target-agent errors without broad prompt replacement",
                },
                {
                    "batch_type": "random_robustness",
                    "priority": 5,
                    "cases": random_cases,
                    "purpose": "check that the local repair remains robust on nearby cases",
                },
            ]
            if invalid_rate >= float(self.cfg.invalid_repair_rate_threshold):
                invalid_cases = by_type("target_invalid") or self._validity_cases_for_agent(diagnosis, agent_id)
                batches.insert(0, {
                    "batch_type": "hard_validity_repair",
                    "priority": -1,
                    "cases": invalid_cases[:limit],
                    "purpose": "repair elevated invalid output failures before other mechanisms",
                })
            return [batch for batch in batches if batch.get("cases")]
        top_cases = self._cases_for_agent(diagnosis, agent_id)[: max(0, int(self.cfg.max_homogeneous_cases_per_agent))]
        random_cases = self._window_random_case_summaries(agent_id, max(0, int(self.cfg.random_window_cases_per_agent)))
        validity_cases = self._validity_cases_for_agent(diagnosis, agent_id)[: max(0, int(self.cfg.hard_validity_cases_per_agent))]
        invalid_rate = 0.0
        rates = diagnosis.get("per_agent_invalid_rate", [])
        if agent_id < len(rates):
            invalid_rate = float(rates[agent_id])
        batches = [
            {
                "batch_type": "target_error_repair",
                "priority": -3,
                "cases": target_error_cases[:target_error_limit],
                "purpose": "repair target-agent wrong, invalid, or unhelpful behavior using abstract error-pattern evidence",
            },
            {
                "batch_type": "target_error_repair",
                "priority": -2,
                "cases": target_error_cases[target_error_limit : target_error_limit * 2] or target_error_cases[:target_error_limit],
                "purpose": "produce an alternative accuracy-repair procedure for the same target-agent blind spots",
            },
            {
                "batch_type": "useful_diversity_repair",
                "priority": 1,
                "cases": top_cases,
                "purpose": "turn redundant or correlated target-agent behavior into useful complementary reasoning",
            },
            {
                "batch_type": "random_window",
                "priority": 2,
                "cases": random_cases,
                "purpose": "avoid overfitting to only the highest-overlap cases",
            },
        ]
        if validity_cases or invalid_rate >= float(self.cfg.invalid_repair_rate_threshold):
            batches.append(
                {
                    "batch_type": "hard_validity_repair",
                    "priority": 0 if invalid_rate >= float(self.cfg.invalid_repair_rate_threshold) else 3,
                    "cases": validity_cases,
                    "purpose": "repair invalid or fragile target-agent outputs before pushing diversity",
                }
            )
        batches.sort(key=lambda x: int(x.get("priority", 0)))
        return [b for b in batches if b.get("cases") or str(b.get("batch_type")) != "target_error_repair"]

    def _optimizer_case_payload(self, case: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        allowed = [
            "case_id",
            "target_agent_id",
            "peer_agent_id",
            "pair_overlap",
            "target_prompt_preview",
            "peer_prompt_preview",
            "target_valid",
            "peer_valid",
            "case_type",
            "target_overlap_pressure",
            "invalid_reasons",
            "answer_present",
            "purpose",
            "team_correct",
            "window_index",
            "target_answer_preview",
            "peer_behavior_summary",
            "target_invalid",
            "target_correct",
            "peer_correct_available",
            "error_pattern",
            "repair_hint",
        ]
        if self._v7_residual_protocol_enabled():
            allowed.extend(["capability_residual_family", "confidence", "peer_wrong_count", "vote_context"])
        for key in allowed:
            if key in case:
                payload[key] = case.get(key)
        if "target_trace_preview" in case:
            payload["target_trace_preview"] = self._redact_optimizer_text(str(case.get("target_trace_preview", "")))
        if "trace_preview" in case and "target_trace_preview" not in payload:
            payload["target_trace_preview"] = self._redact_optimizer_text(str(case.get("trace_preview", "")))
        if "peer_trace_preview" in case:
            value = case.get("peer_trace_preview")
            if isinstance(value, list):
                payload["peer_trace_preview"] = [self._redact_optimizer_text(str(x), max_chars=240) for x in value[:2]]
            else:
                payload["peer_trace_preview"] = self._redact_optimizer_text(str(value), max_chars=240)
        if "target_answer_preview" not in payload and "target_answer" in case:
            payload["target_answer_preview"] = self._answer_behavior_preview(str(case.get("target_answer", "")))
        return payload

    def _target_case_keys(self, cases: List[Dict[str, Any]]) -> set:
        return {str(c.get("case_key", "")) for c in cases if str(c.get("case_key", ""))}

    def _homogeneity_impact_metrics(
        self,
        agent_id: int,
        rollout: Dict[str, Any],
        baseline_case_keys: set,
        sample_hash: str,
    ) -> Dict[str, Any]:
        high_pairs = [
            p for p in rollout.get("high_overlap_pairs", [])
            if isinstance(p, dict) and not bool(p.get("invalid_pair", False))
        ]
        target_pairs = []
        current_keys = set()
        target_pressure = 0.0
        pressures = list(rollout.get("per_agent_overlap", []))
        if agent_id < len(pressures):
            target_pressure = float(pressures[agent_id])
        for pair in high_pairs:
            ids = pair.get("pair", [])
            if not isinstance(ids, list) or len(ids) != 2:
                continue
            a, b = int(ids[0]), int(ids[1])
            if agent_id not in (a, b):
                continue
            target_pairs.append(pair)
            suffix = f"{a}-{b}" if a < b else f"{b}-{a}"
            current_keys.add(f"{sample_hash}:{suffix}")
        relevant_baselines = {x for x in baseline_case_keys if str(x).startswith(f"{sample_hash}:")}
        resolved = relevant_baselines - current_keys
        new_cases = current_keys - relevant_baselines
        return {
            "target_overlap_pressure": target_pressure,
            "homogeneous_case_count": int(len(target_pairs)),
            "resolved_case_count": int(len(resolved)),
            "new_homogeneous_case_count": int(len(new_cases)),
        }

    async def _propose_accuracy_candidates(
        self,
        agent_id: int,
        parent_prompt: str,
        accuracy_diagnosis: Dict[str, Any],
        num_candidates: int,
        generation_batches: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prompt_roles = [r for r in accuracy_diagnosis.get("prompt_roles", []) if isinstance(r, dict)]
        target_role_spec = next((r for r in prompt_roles if int(r.get("agent_id", -1)) == int(agent_id)), {})
        peer_role_specs = [r for r in prompt_roles if int(r.get("agent_id", -1)) != int(agent_id)]
        error_counts = accuracy_diagnosis.get("per_agent_error_count", [])
        agent_acc = accuracy_diagnosis.get("per_agent_accuracy", [])
        window_stats = {
            "window_size": accuracy_diagnosis.get("window_size", 0),
            "team_accuracy": accuracy_diagnosis.get("team_accuracy", 0.0),
            "target_error_count": error_counts[agent_id] if agent_id < len(error_counts) else 0,
            "target_accuracy": agent_acc[agent_id] if agent_id < len(agent_acc) else 0.0,
        }
        safe_generation_batches = []
        for batch in generation_batches:
            if not isinstance(batch, dict):
                continue
            safe_generation_batches.append(
                {
                    **batch,
                    "cases": [
                        self._optimizer_case_payload(c)
                        for c in batch.get("cases", [])
                        if isinstance(c, dict)
                    ],
                }
            )
        if not safe_generation_batches:
            safe_generation_batches = [{"batch_type": "accuracy_error_cases", "cases": [], "purpose": "general accuracy repair"}]
        system_prompt = (
            "You are a prompt optimizer for a multi-agent reasoning team.\n"
            "Your objective is to improve the target agent's answer accuracy on observed error patterns.\n"
            "Use the parent prompt, prompt-role previews, window accuracy statistics, and target-agent error cases.\n"
            "Useful reasoning diversity is allowed only when it helps the target agent repair mistakes.\n"
            "Do not optimize for semantic overlap, invalid-rate metrics, trace difference alone, or stylistic novelty.\n"
            "Do not use gold answers, concrete task text, options, labels, or answer-specific content.\n"
            "Treat trace previews as behavioral evidence of mistakes; do not copy their wording into the new prompt.\n"
            "Return strict JSON only."
        )
        user_prompt = (
            "Revise the target agent prompt to reduce the observed answer mistakes.\n"
            "Priority order:\n"
            "1. Repair the target agent's observed error patterns.\n"
            "2. Preserve or improve target-agent answer accuracy.\n"
            "3. Add useful reasoning diversity only when it helps correctness or error rescue.\n"
            "4. Avoid invalid, verbose, generic, or merely paraphrased prompts.\n"
            "5. Do not optimize for trace difference alone.\n"
            "Each candidate must describe an executable reasoning procedure that can improve correctness on similar examples. "
            "Prefer concrete checks such as concept disambiguation, option comparison, contradiction testing, qualifier inspection, "
            "or final verification when they fit the observed mistake pattern.\n"
            "A candidate is invalid if it only paraphrases the parent prompt, appends generic caution, asks the solver to be more accurate, "
            "or changes style without adding a concrete error-repair procedure.\n"
            "Each candidate_prompt must contain a concrete reasoning procedure, a specific error-repair behavior, final answer discipline, "
            "and a short verification step.\n"
            "Write a complete short role prompt, not a suffix to append to the parent prompt. "
            "Do not repeat generic instructions already present in the parent prompt. "
            "Do not use the phrase 'Use a distinct decision procedure'. "
            "The prompt should remain short and usable by a solver agent. It must still end with exactly one final answer in normal solving, "
            "but do not include concrete answer labels or sample content inside candidate_prompt.\n"
            "Do not mention reward, beam search, candidates, evaluation metrics, or this optimizer instruction inside candidate_prompt.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "candidates": [\n'
            '    {"candidate_prompt": str, "role_name": str, "decision_procedure": [str, ...], "when_to_use": str, "fallback_strategy": str, "accuracy_checks": [str, ...], "target_error_pattern": str, "accuracy_repair_rule": str, "expected_accuracy_effect": str, "rationale": str, "source_batch_type": str},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Return exactly requested_candidates distinct candidates. "
            "If multiple candidates use the same source_batch_type, they must repair the mistakes with meaningfully different executable procedures.\n\n"
            f"target_agent_id: {agent_id}\n"
            f"requested_candidates: {num_candidates}\n\n"
            f"current_parent_prompt:\n{parent_prompt}\n\n"
            f"target_role_spec:\n{json.dumps(target_role_spec, ensure_ascii=False, indent=2)}\n\n"
            f"peer_role_specs:\n{json.dumps(peer_role_specs, ensure_ascii=False, indent=2)}\n\n"
            f"window_accuracy_statistics:\n{json.dumps(window_stats, ensure_ascii=False, indent=2)}\n\n"
            f"generation_batches:\n{json.dumps(safe_generation_batches, ensure_ascii=False, indent=2)}"
        )
        if self._v7_residual_protocol_enabled():
            system_prompt = (
                system_prompt.replace("role prompts", "solver instructions")
                .replace("prompt-role previews", "prompt summaries")
                .replace("role previews", "prompt summaries")
            )
            user_prompt = (
                user_prompt.replace("role_name", "mechanism_name")
                .replace("target_role_spec", "target_prompt_state")
                .replace("peer_role_specs", "peer_prompt_summaries")
                .replace("complete short role prompt", "complete short solver instruction")
                .replace("role prompt", "solver instruction")
                .replace("peer roles", "peer prompts")
                .replace("prompt-role previews", "prompt summaries")
            )
        text = await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(self.cfg.optimizer_temperature),
            max_tokens=int(self.cfg.optimizer_max_tokens),
            stage=f"accuracy_optimizer_agent_{agent_id}",
        )
        diagnostics = self._empty_optimizer_generation_diagnostics()
        diagnostics["optimizer_raw_response_empty"] = int(not str(text or "").strip())
        obj = extract_json_obj(text)
        diagnostics["optimizer_json_parse_failed"] = int(bool(str(text or "").strip()) and obj is None)
        if obj is None:
            obj = {}
        candidates = obj.get("candidates", []) if isinstance(obj, dict) else []
        if isinstance(candidates, list):
            diagnostics["optimizer_raw_candidate_count"] = len(candidates)
        else:
            diagnostics["optimizer_schema_filtered_count"] += 1
            candidates = []
        parsed: List[Dict[str, Any]] = []
        seen_signatures: set = set()
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    diagnostics["optimizer_schema_filtered_count"] += 1
                    continue
                if not self._candidate_has_required_optimizer_fields(item, architecture="one_shot"):
                    diagnostics["optimizer_empty_prompt_count"] += 1
                    diagnostics["optimizer_schema_filtered_count"] += 1
                    continue
                prompt = str(item.get("candidate_prompt", "")).strip()
                prompt, sanitized = self._sanitize_prompt(prompt, agent_id)
                diagnostics["optimizer_sanitized_count"] += int(bool(sanitized))
                if self._is_redundant_candidate_prompt(parent_prompt, prompt, seen_signatures):
                    diagnostics["optimizer_redundant_filtered_count"] += 1
                    continue
                if not prompt:
                    diagnostics["optimizer_empty_prompt_count"] += 1
                    continue
                seen_signatures.add(self._prompt_signature(prompt))
                batch_idx = min(len(parsed), len(generation_batches) - 1)
                parsed.append(
                    {
                        "candidate_prompt": prompt,
                        "role_name": str(item.get("role_name", "")),
                        "mechanism_name": str(item.get("mechanism_name", item.get("role_name", ""))),
                        "decision_procedure": item.get("decision_procedure", []),
                        "when_to_use": str(item.get("when_to_use", "")),
                        "fallback_strategy": str(item.get("fallback_strategy", "")),
                        "accuracy_checks": item.get("accuracy_checks", []),
                        "target_error_pattern": str(item.get("target_error_pattern", "")),
                        "accuracy_repair_rule": str(item.get("accuracy_repair_rule", "")),
                        "expected_accuracy_effect": str(item.get("expected_accuracy_effect", "")),
                        "rationale": str(item.get("rationale", "")),
                        "candidate_source": "optimizer",
                        "optimizer_generation_diagnostics": dict(diagnostics),
                        "generation_batch_type": str(item.get("source_batch_type", "")) or str(generation_batches[batch_idx].get("batch_type", "")),
                        "generation_case_ids": [
                            str(c.get("case_id", ""))
                            for c in generation_batches[batch_idx].get("cases", [])
                            if isinstance(c, dict)
                        ],
                    }
                )
                if len(parsed) >= num_candidates:
                    break
        fallback_mode_cfg = str(getattr(self.cfg, "optimizer_fallback_mode", "none") or "none").lower()
        if fallback_mode_cfg == "template":
            while len(parsed) < num_candidates:
                batch_idx = min(len(parsed), len(generation_batches) - 1)
                fallback = self._structured_fallback_role(agent_id, len(parsed), mode="accuracy")
                prompt = str(fallback["candidate_prompt"])
                seen_signatures.add(self._prompt_signature(prompt))
                parsed.append(
                    {
                        "candidate_prompt": prompt,
                        "role_name": str(fallback.get("role_name", fallback.get("mechanism_name", ""))),
                        "mechanism_name": str(fallback.get("mechanism_name", fallback.get("role_name", ""))),
                        "decision_procedure": list(fallback["decision_procedure"]),
                        "when_to_use": str(fallback["when_to_use"]),
                        "fallback_strategy": str(fallback["fallback_strategy"]),
                        "accuracy_checks": list(fallback["accuracy_checks"]),
                        "target_error_pattern": str(fallback.get("target_error_pattern", "")),
                        "accuracy_repair_rule": str(fallback.get("accuracy_repair_rule", "")),
                        "expected_accuracy_effect": str(fallback.get("expected_accuracy_effect", "")),
                        "rationale": "Fallback candidate when optimizer returns too few usable prompts.",
                        "candidate_source": "accuracy_repair_fallback",
                        "optimizer_generation_diagnostics": dict(diagnostics),
                        "generation_batch_type": str(generation_batches[batch_idx].get("batch_type", "")),
                        "generation_case_ids": [
                            str(c.get("case_id", ""))
                            for c in generation_batches[batch_idx].get("cases", [])
                            if isinstance(c, dict)
                        ],
                    }
                )
        diagnostics["optimizer_final_candidate_count"] = sum(1 for item in parsed[:num_candidates] if str(item.get("candidate_source", "")) == "optimizer")
        diagnostics["optimizer_underfilled"] = bool(diagnostics["optimizer_final_candidate_count"] < int(num_candidates))
        diagnostics = self._record_optimizer_generation_diagnostics(agent_id, parent_prompt, diagnostics)
        for item in parsed:
            item["optimizer_generation_diagnostics"] = dict(diagnostics)
        return parsed[:num_candidates]

    def _build_teacher_context(
        self,
        agent_id: int,
        parent_prompt: str,
        target_role_spec: Dict[str, Any],
        peer_role_specs: List[Dict[str, Any]],
        window_stats: Dict[str, Any],
        validity_constraints: Dict[str, Any],
        generation_batches: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        safe_generation_batches: List[Dict[str, Any]] = []
        target_error_patterns: List[str] = []
        invalid_output_patterns: List[str] = []
        peer_behavior_summary: List[str] = []
        batch_types: List[str] = []
        for batch in generation_batches:
            if not isinstance(batch, dict):
                continue
            safe_cases = []
            batch_type = str(batch.get("batch_type", ""))
            if batch_type:
                batch_types.append(batch_type)
            for case in batch.get("cases", []):
                if not isinstance(case, dict):
                    continue
                case_type = str(case.get("case_type", "") or case.get("purpose", "") or batch_type)
                if case_type:
                    target_error_patterns.append(case_type)
                invalids = case.get("invalid_reasons", [])
                if isinstance(invalids, list):
                    invalid_output_patterns.extend(str(x) for x in invalids if str(x))
                elif invalids:
                    invalid_output_patterns.append(str(invalids))
                peer_summary = str(case.get("peer_behavior_summary", "") or case.get("purpose", "") or "").strip()
                if peer_summary:
                    peer_behavior_summary.append(normalize_spaces(peer_summary)[:180])
                safe_case = {
                        "case_type": case_type,
                        "target_agent_id": int(case.get("target_agent_id", agent_id) or agent_id),
                        "target_correct": case.get("target_correct", ""),
                        "target_invalid": case.get("target_invalid", ""),
                        "peer_correct_available": case.get("peer_correct_available", ""),
                        "purpose": normalize_spaces(str(case.get("purpose", "")))[:160],
                        "repair_hint": normalize_spaces(str(case.get("repair_hint", "")))[:180],
                        "target_overlap_pressure": case.get("target_overlap_pressure", ""),
                }
                if self._v7_residual_protocol_enabled():
                    safe_case.update({
                        "capability_residual_family": str(case.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value)),
                        "vote_context": str(case.get("vote_context", "")),
                    })
                safe_cases.append(safe_case)
            safe_generation_batches.append(
                {
                    "batch_type": batch_type,
                    "purpose": normalize_spaces(str(batch.get("purpose", "")))[:200],
                    "case_count": len(safe_cases),
                    "cases": safe_cases,
                }
            )
        answer_format = str(getattr(self.cfg, "answer_format", "") or "").strip() or str(getattr(self.cfg, "task_type", "auto"))
        problem_type = str(getattr(self.cfg, "comparison_task_id", "") or getattr(self.cfg, "benchmark", "") or getattr(self.cfg, "task_type", "auto"))
        target_pressure = float(window_stats.get("target_overlap_pressure", 0.0) or 0.0)
        mean_overlap = float(window_stats.get("mean_window_overlap", 0.0) or 0.0)
        target_invalid_rate = float(window_stats.get("target_invalid_rate", 0.0) or 0.0)
        target_error_count = int(window_stats.get("target_error_count", 0) or 0)
        target_team_wrong_error_count = int(window_stats.get("target_team_wrong_error_count", 0) or 0)
        target_pivotal_fix_count = int(window_stats.get("target_pivotal_fix_count", 0) or 0)
        target_dominant_wrong_count = int(window_stats.get("target_dominant_wrong_redundancy_count", 0) or 0)
        window_vote_acc = float(window_stats.get("window_vote_acc", window_stats.get("team_accuracy", 0.0)) or 0.0)
        window_vote_margin = float(window_stats.get("window_mean_vote_margin", -1.0) if window_stats.get("window_mean_vote_margin") is not None else -1.0)
        window_boundary_diversity = float(window_stats.get("window_mean_boundary_useful_diversity", 0.0) or 0.0)
        residual_families = sorted({
            str(case.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value))
            for batch in generation_batches if isinstance(batch, dict)
            for case in batch.get("cases", []) if isinstance(case, dict)
        })
        context = {
            "target_agent_id": agent_id,
            "parent_prompt_preview": normalize_spaces(parent_prompt)[:600],
            "target_role_spec": target_role_spec,
            "peer_role_specs": peer_role_specs,
            "window_stats": window_stats,
            "validity_constraints": validity_constraints,
            "generation_batches": safe_generation_batches,
            "diagnostic_focus": {
                "problem_type": problem_type,
                "answer_format": answer_format,
                "target_error_patterns": sorted(set(target_error_patterns))[:12],
                "invalid_output_patterns": sorted(set(invalid_output_patterns))[:12],
                "diversity_gap_summary": (
                    f"window_vote_acc={window_vote_acc:.3f}; window_mean_vote_margin={window_vote_margin:.3f}; "
                    f"window_boundary_useful_diversity={window_boundary_diversity:.3f}; "
                    f"target_pivotal_fix_count={target_pivotal_fix_count}; target_dominant_wrong_redundancy_count={target_dominant_wrong_count}; "
                    f"diagnostic_embedding_overlap={mean_overlap:.3f}; target_embedding_overlap_pressure={target_pressure:.3f}; "
                    f"batch_types={sorted(set(batch_types))[:8]}"
                ),
                "prompt_redundancy_summary": (
                    f"target prompt preview is compared against {len(peer_role_specs)} peer role previews; "
                    f"avoid duplicating peer procedures and parent wording."
                ),
                "error_correlation_summary": (
                    "Use target error cases and vote-boundary diagnostics as abstract repair signals. "
                    "Voting failures are included by default."
                    if not bool(getattr(self.cfg, "teacher_critic_use_voting_failure", False))
                    else "Voting failures, pivotal fixes, and dominant wrong-answer redundancy are primary diagnostics."
                ),
                "peer_behavior_summary": peer_behavior_summary[:8],
                "target_error_summary": (
                    f"target_error_count={target_error_count}; "
                    f"target_team_wrong_error_count={target_team_wrong_error_count}; "
                    f"target_pivotal_fix_count={target_pivotal_fix_count}; "
                    f"target_dominant_wrong_redundancy_count={target_dominant_wrong_count}"
                ),
                "invalid_output_summary": f"target_invalid_rate={target_invalid_rate:.3f}; invalid patterns are abstracted above.",
            },
        }
        if self._v7_residual_protocol_enabled():
            context["diagnostic_focus"]["capability_residual_families"] = residual_families[:12]
            context["target_prompt_state"] = context.pop("target_role_spec")
            context["peer_prompt_summaries"] = context.pop("peer_role_specs")
        if self._residual_specialization_enabled():
            agent = self.agents[agent_id]
            ordered_profile = sorted(agent.capability_profile.items(), key=lambda item: (-item[1], item[0]))
            context["observed_long_term_capability_profile"] = {
                "strongest_supported_residual_families": [key for key, value in ordered_profile[:3] if value > 0.0],
                "capability_coverage_gap": dict(window_stats.get("capability_coverage_gap", {})),
                "residual_guidance_strength": float(self.specialization_strength) if bool(getattr(self.cfg, "competence_progressive_residual_enabled", False)) else 1.0,
                "guidance": (
                    "Treat residual-family evidence as observation only; do not steer the prompt from it yet."
                    if bool(getattr(self.cfg, "competence_progressive_residual_enabled", False)) and float(self.specialization_strength) <= 0.0
                    else "Use residual-family evidence as a strength-scaled historical affinity, never as an assigned role."
                ),
            }
        if self._is_stable_qd_lineage():
            target_state = self.agents[agent_id].lineage_state
            target_profile = getattr(self, "behavior_profile_by_prompt_hash", {}).get(
                self._normalized_prompt_hash(parent_prompt), {}
            )
            context["stable_lineage_context"] = {
                "lineage_status": str(target_state.get("lineage_status", "uncommitted")),
                "anchor_mechanism": list(target_state.get("lineage_anchor_mechanism_signature", [])),
                "stay_near_anchor_required": bool(target_state.get("lineage_status") == "committed"),
                "committed_peer_mechanisms": [
                    {
                        "agent_id": peer_id,
                        "mechanism": list(peer.lineage_state.get("lineage_anchor_mechanism_signature", [])),
                    }
                    for peer_id, peer in enumerate(self.agents)
                    if peer_id != agent_id and peer.lineage_state.get("lineage_status") == "committed"
                ],
                "target_behavior_residual": {
                    "rescue_support": int(sum(target_profile.get("rescue_vector", []))),
                    "unique_correct_support": int(sum(target_profile.get("unique_correct_vector", []))),
                    "shared_error_support": int(sum(target_profile.get("shared_error_vector", []))),
                    "window_target_error_count": target_error_count,
                    "window_target_team_wrong_error_count": target_team_wrong_error_count,
                },
                "guidance": (
                    "The target has no committed lineage: permit substantial mechanism changes while preserving competence."
                    if target_state.get("lineage_status") != "committed"
                    else "Prefer structural variants near the committed anchor, but allow a justified alternative for joint selection."
                ),
            }
        if isinstance(window_stats.get("refill_feedback"), dict):
            context["candidate_refill_feedback"] = dict(window_stats["refill_feedback"])
        return context

    async def propose_teacher_question(
        self,
        agent_id: int,
        parent_prompt: str,
        teacher_context: Dict[str, Any],
        requested_candidates: int,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are the Teacher in a Teacher-Critic-Student prompt optimization system.\n\n"
            "Your job is not to write a prompt.\n"
            "Your job is to formulate a high-quality Socratic guiding question that will help the Student rewrite the target agent prompt.\n\n"
            "The guiding question must be grounded in:\n"
            "- problem type\n- answer format\n- target-agent error patterns\n- diversity gap\n"
            "- prompt redundancy\n- error correlation with peer agents\n- peer behavior summaries\n"
            "- invalid-output patterns if present\n\n"
            "Do not use gold answers.\nDo not use concrete question text.\nDo not use concrete answer labels.\n"
            "Do not create task-specific hard-coded roles.\nDo not optimize for voting failure in this step.\n"
            "Do not ask a generic question such as 'How can the prompt be improved?'\n\n"
            "A good guiding question should force the Student to create a candidate prompt that:\n"
            "- aligns with the task/problem type\n- repairs a specific observed error pattern\n"
            "- improves target-agent accuracy\n- contributes useful reasoning diversity\n"
            "- avoids duplicating peer prompts\n- avoids invalid or overlong outputs\n\n"
            "Return strict JSON only."
        )
        if self._v7_residual_protocol_enabled():
            system_prompt = system_prompt.replace(
                "Do not optimize for voting failure in this step.\n",
                "Use voting failures only as abstract evidence of harmful shared-error mechanisms. "
                "Do not game the vote directly, memorize sample answers, or optimize for disagreement by itself.\n",
            )
            system_prompt += (
                "\nFocus the guiding question on one residual error mechanism, a possible pivotal correction, "
                "and how to preserve pivotal-correct behavior. Ask which local executable reasoning mechanism "
                "could make the target correct when several peers also fail."
            )
        user_prompt = (
            "Create one Socratic guiding question for the Student.\n"
            "Return JSON with keys: problem_type_analysis, answer_format_analysis, target_error_analysis, "
            "diversity_gap_analysis, error_correlation_analysis, peer_difference_analysis, socratic_guiding_question, "
            "question_objective, expected_prompt_change, expected_accuracy_effect, expected_diversity_effect, risk_to_avoid.\n\n"
            f"target_agent_id: {agent_id}\nrequested_candidates: {requested_candidates}\n"
            f"parent_prompt_preview:\n{normalize_spaces(parent_prompt)[:600]}\n\n"
            f"teacher_context:\n{json.dumps(teacher_context, ensure_ascii=False, indent=2)}"
        )
        text = await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(getattr(self.cfg, "teacher_temperature", self.cfg.optimizer_temperature)),
            max_tokens=int(getattr(self.cfg, "teacher_max_tokens", self.cfg.optimizer_max_tokens)),
            stage=f"teacher_agent_{agent_id}",
            client_role="optimizer",
        )
        obj = extract_json_obj(text) or {}
        return obj if isinstance(obj, dict) else {}

    async def critique_teacher_question(
        self,
        agent_id: int,
        teacher_question: Dict[str, Any],
        teacher_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are the Critic in a Teacher-Critic-Student prompt optimization system.\n\n"
            "Your job is to audit the Teacher's Socratic guiding question before the Student sees it.\n\n"
            "Reject the question if it is:\n"
            "- generic\n- not grounded in observed diagnostics\n- not aligned with the problem type\n"
            "- not aligned with the target error pattern\n- only about surface-level diversity\n"
            "- likely to duplicate peer prompts\n- likely to reduce answer accuracy\n"
            "- using gold answers, concrete sample text, or answer labels\n"
            "- using hard-coded task-specific roles\n"
            "- focused on voting failure rather than prompt quality/diversity/accuracy\n\n"
            "Also reject persona-only changes, wholesale rewrites for a narrow error, deletion of repeatedly effective mechanisms, "
            "repetition of recent failed edits, or vague non-executable reasoning steps. Prefer a concrete local mechanism edit.\n\n"
            "Return strict JSON only."
        )
        if self._v7_residual_protocol_enabled():
            system_prompt = system_prompt.replace(
                "- focused on voting failure rather than prompt quality/diversity/accuracy\n\n",
                "- gaming vote outcomes, memorizing answers, or pursuing disagreement by itself\n\n",
            )
            system_prompt += (
                "\nAudit whether the question targets a concrete residual error mechanism, can reduce a shared-error "
                "mechanism through a pivotal correction, preserves pivotal-correct behavior, avoids persona-only or "
                "wholesale rewrites, avoids repeated failed mechanisms, and specifies one executable local decision step."
            )
        user_prompt = (
            "Audit the Teacher question. Pass only if score >= threshold, the question is specific, grounded in diagnostics, "
            "contains no leakage or hard-coded task role, and is useful for both accuracy and diversity.\n"
            "Return JSON with keys: passed, score, quality_critique, specificity_critique, task_alignment_critique, "
            "error_alignment_critique, diversity_critique, redundancy_critique, safety_critique, rewrite_instruction.\n\n"
            f"target_agent_id: {agent_id}\n"
            f"pass_threshold: {float(getattr(self.cfg, 'teacher_question_pass_threshold', 0.75))}\n"
            f"teacher_question:\n{json.dumps(teacher_question, ensure_ascii=False, indent=2)}\n\n"
            f"teacher_context:\n{json.dumps(teacher_context, ensure_ascii=False, indent=2)}"
        )
        text = await self._chat(
            model=self.cfg.evaluator_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(getattr(self.cfg, "critic_temperature", self.cfg.evaluator_temperature)),
            max_tokens=int(getattr(self.cfg, "critic_max_tokens", self.cfg.evaluator_max_tokens)),
            stage=f"teacher_critic_agent_{agent_id}",
            client_role="evaluator",
        )
        obj = extract_json_obj(text) or {}
        return obj if isinstance(obj, dict) else {}

    async def rewrite_teacher_question(
        self,
        agent_id: int,
        previous_question: Dict[str, Any],
        critic_review: Dict[str, Any],
        teacher_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are the Teacher revising a Socratic guiding question after Critic feedback.\n"
            "Revise only the guiding question and its rationale. Do not write candidate prompts.\n"
            "Do not use gold answers, concrete sample text, answer labels, or hard-coded task-specific roles.\n"
            "Return strict JSON only."
        )
        user_prompt = (
            "Rewrite the Teacher JSON so it can pass Critic review while staying grounded in the abstract diagnostics.\n"
            "Keep the same JSON schema as the Teacher output.\n\n"
            f"target_agent_id: {agent_id}\n"
            f"previous_question:\n{json.dumps(previous_question, ensure_ascii=False, indent=2)}\n\n"
            f"critic_review:\n{json.dumps(critic_review, ensure_ascii=False, indent=2)}\n\n"
            f"teacher_context:\n{json.dumps(teacher_context, ensure_ascii=False, indent=2)}"
        )
        text = await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(getattr(self.cfg, "teacher_temperature", self.cfg.optimizer_temperature)),
            max_tokens=int(getattr(self.cfg, "teacher_max_tokens", self.cfg.optimizer_max_tokens)),
            stage=f"teacher_rewrite_agent_{agent_id}",
            client_role="optimizer",
        )
        obj = extract_json_obj(text) or {}
        return obj if isinstance(obj, dict) else {}

    async def generate_approved_teacher_question(
        self,
        agent_id: int,
        parent_prompt: str,
        teacher_context: Dict[str, Any],
        requested_candidates: int,
    ) -> Dict[str, Any]:
        threshold = float(getattr(self.cfg, "teacher_question_pass_threshold", 0.75) or 0.75)
        max_rounds = max(1, int(getattr(self.cfg, "teacher_critic_max_rounds", 3) or 3))
        teacher_question = await self.propose_teacher_question(agent_id, parent_prompt, teacher_context, requested_candidates)
        reviews: List[Dict[str, Any]] = []
        question_versions: List[Dict[str, Any]] = []
        rewrite_count = 0

        def has_guiding_question(question: Any) -> bool:
            return bool(
                isinstance(question, dict)
                and str(question.get("socratic_guiding_question", "")).strip()
            )

        for round_id in range(max_rounds):
            round_context = dict(TCS_AUDIT_CONTEXT.get() or {})
            round_context["teacher_critic_round"] = round_id + 1
            round_token = TCS_AUDIT_CONTEXT.set(round_context)
            try:
                review = await self.critique_teacher_question(agent_id, teacher_question, teacher_context)
            finally:
                TCS_AUDIT_CONTEXT.reset(round_token)
            reviews.append(review)
            question_versions.append(teacher_question)
            score = self._safe_float(review.get("score", 0.0), 0.0)
            if (
                has_guiding_question(teacher_question)
                and bool(review.get("passed"))
                and self._safe_float(review.get("score", 0.0), 0.0) >= threshold
            ):
                return {
                    "approved": True,
                    "teacher_question": teacher_question,
                    "critic_reviews": reviews,
                    "teacher_critic_rounds": round_id + 1,
                    "teacher_rewrite_count": rewrite_count,
                    "teacher_question_forced_best_score": False,
                    "teacher_question_forced_best_round": 0,
                    "teacher_question_forced_below_threshold": False,
                }
            if round_id < max_rounds - 1:
                rewrite_context = dict(TCS_AUDIT_CONTEXT.get() or {})
                rewrite_context["teacher_critic_round"] = round_id + 1
                rewrite_token = TCS_AUDIT_CONTEXT.set(rewrite_context)
                try:
                    teacher_question = await self.rewrite_teacher_question(agent_id, teacher_question, review, teacher_context)
                finally:
                    TCS_AUDIT_CONTEXT.reset(rewrite_token)
                rewrite_count += 1
        usable_indices = [
            idx for idx, question in enumerate(question_versions)
            if has_guiding_question(question)
        ]
        if not usable_indices:
            return {
                "approved": False,
                "teacher_question": {},
                "critic_reviews": reviews,
                "teacher_critic_rounds": len(reviews),
                "teacher_rewrite_count": rewrite_count,
                "teacher_question_forced_best_score": False,
                "teacher_question_forced_best_round": 0,
                "teacher_question_forced_below_threshold": True,
                "teacher_question_forced_best_review": reviews[-1] if reviews else {},
                "teacher_question_rejection_reason": "empty_teacher_question",
            }

        best_idx = usable_indices[0]
        best_score = -1.0
        for idx in usable_indices:
            review = reviews[idx]
            score = self._safe_float(review.get("score", 0.0), 0.0)
            if score > best_score:
                best_idx = idx
                best_score = score
        best_review = reviews[best_idx] if reviews else {}
        best_question = question_versions[best_idx] if question_versions else teacher_question
        return {
            # The question did not pass Critic; it is nevertheless the legal
            # best-score fallback that may be handed to Student.
            "approved": False,
            "teacher_question": best_question,
            "critic_reviews": reviews,
            "teacher_critic_rounds": len(reviews),
            "teacher_rewrite_count": rewrite_count,
            "teacher_question_forced_best_score": True,
            "teacher_question_forced_best_round": best_idx + 1,
            "teacher_question_forced_below_threshold": best_score < threshold,
            "teacher_question_forced_best_review": best_review,
        }

    async def retry_student_candidates_json_only(
        self,
        previous_raw_text: str,
        approved_teacher_question: Dict[str, Any],
        num_candidates: int,
        agent_id: int = 0,
    ) -> str:
        schema = self._student_candidate_schema_json()
        prompt_max = int(
            getattr(self.cfg, "student_candidate_prompt_hard_max_chars", 1400)
            if bool(getattr(self.cfg, "competence_depth_enabled", False))
            else getattr(self.cfg, "student_candidate_prompt_max_chars", 900)
        )
        field_max = int(getattr(self.cfg, "student_candidate_max_chars_per_field", 320) or 320)
        system_prompt = (
            "Your previous response was not valid JSON.\n\n"
            "Return only valid minified JSON matching this exact schema:\n"
            f"{schema}\n\n"
            "Do not add markdown.\n"
            "Do not add explanations.\n"
            "Do not use multiline strings.\n"
            "Use double quotes for every key and string value.\n"
            "Do not include trailing commas or comments.\n"
            f"Use at most {int(num_candidates)} candidates.\n"
            f"candidate_prompt must be <= {prompt_max} characters.\n"
            f"Every other field must be <= {field_max} characters.\n"
            "Each candidate_prompt must be concise.\n"
            'If you cannot comply, return {"candidates":[]}.'
        )
        user_prompt = (
            "Approved Teacher question and context for the retry:\n"
            f"{json.dumps(approved_teacher_question, ensure_ascii=False, indent=2)}\n\n"
            "Previous invalid JSON-like response, for reference only:\n"
            f"{str(previous_raw_text or '')[:4000]}"
        )
        return await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(getattr(self.cfg, "student_temperature", self.cfg.optimizer_temperature)),
            max_tokens=int(getattr(self.cfg, "student_max_tokens", self.cfg.optimizer_max_tokens)),
            stage=f"student_json_retry_agent_{agent_id}",
            client_role="optimizer",
        )

    async def repair_student_json_response(
        self,
        raw_text: str,
        expected_num_candidates: int,
    ) -> Dict[str, Any]:
        schema = self._student_candidate_schema_json()
        if not str(raw_text or "").strip():
            return {
                "repaired": False,
                "repair_raw_response_preview": "",
                "repair_json_parse_failed": True,
                "repair_failure_reason": "empty_raw_text",
                "obj": None,
            }
        system_prompt = (
            "You are a JSON repair utility.\n\n"
            "You will receive malformed JSON-like text that was intended to match this schema:\n"
            f"{schema}\n\n"
            "Your job is only to repair JSON syntax:\n"
            "- close braces and brackets if needed\n"
            "- escape unescaped quotes inside strings\n"
            "- remove trailing commas\n"
            "- keep only the candidates that are already present in the input\n"
            "- do not invent new candidates\n"
            "- do not change semantic content\n"
            "- do not add explanations\n"
            "- return minified JSON only\n\n"
            'If the input cannot be repaired without inventing content, return {"candidates":[]}.'
        )
        user_prompt = (
            f"expected_num_candidates: {int(expected_num_candidates)}\n\n"
            "Malformed JSON-like input:\n"
            f"{str(raw_text or '')[:6000]}"
        )
        try:
            repair_text = await self._chat(
                model=self.cfg.optimizer_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=float(getattr(self.cfg, "student_json_repair_temperature", 0.0) or 0.0),
                max_tokens=int(getattr(self.cfg, "student_json_repair_max_tokens", 1200) or 1200),
                stage="student_json_repair",
                client_role="optimizer",
            )
        except Exception as exc:
            return {
                "repaired": False,
                "repair_raw_response_preview": "",
                "repair_json_parse_failed": True,
                "repair_failure_reason": type(exc).__name__,
                "obj": None,
            }
        obj = extract_json_obj(repair_text or "")
        parse_failed = obj is None or not isinstance(obj, dict)
        return {
            "repaired": not parse_failed,
            "repair_raw_response_preview": normalize_spaces(repair_text or "")[:1000],
            "repair_json_parse_failed": bool(parse_failed),
            "repair_failure_reason": "" if not parse_failed else "repair_json_parse_failed",
            "obj": obj if isinstance(obj, dict) else None,
        }

    async def generate_student_candidates(
        self,
        agent_id: int,
        parent_prompt: str,
        approved_teacher_question: Dict[str, Any],
        teacher_context: Dict[str, Any],
        num_candidates: int,
    ) -> Dict[str, Any]:
        schema = self._student_candidate_schema_json()
        schema_mode = str(getattr(self.cfg, "student_candidate_schema_mode", "compact") or "compact").lower()
        prompt_max = int(
            getattr(self.cfg, "student_candidate_prompt_hard_max_chars", 1400)
            if bool(getattr(self.cfg, "competence_depth_enabled", False))
            else getattr(self.cfg, "student_candidate_prompt_max_chars", 900)
        )
        field_max = int(getattr(self.cfg, "student_candidate_max_chars_per_field", 320) or 320)
        compact_output_rules = (
            "Output format requirements:\n"
            "- Return exactly one JSON object.\n"
            "- The first character must be `{`.\n"
            "- The last character must be `}`.\n"
            "- Return minified JSON only.\n"
            "- Use double quotes for all JSON keys and string values.\n"
            "- Do not use Markdown.\n"
            "- Do not wrap the JSON in code fences.\n"
            "- Do not add explanations before or after the JSON.\n"
            "- Do not use multiline strings.\n"
            "- Do not include newline characters inside string values.\n"
            "- Do not include bullet lists inside string values.\n"
            "- Escape all quotes inside strings.\n"
            "- Do not include trailing commas.\n"
            "- Do not include comments.\n"
            f"- candidate_prompt must be <= {prompt_max} characters.\n"
            f"- Every other field must be <= {field_max} characters.\n"
            "- Each non-prompt field must be one short sentence.\n"
            "- Each candidate_prompt should be a concise solver instruction, not a long essay.\n"
            "- Return a complete standalone prompt. Do not end mid-sentence.\n"
            "- Preserve useful mechanisms but merge repeated instructions; avoid repeatedly saying explicitly, systematically, before final selection, or check every constraint.\n"
            "- Prefer semicolon-separated steps over numbered multiline lists.\n"
            '- If you cannot safely generate a candidate, return {"candidates":[]}.\n'
            f"Exact schema:\n{schema}"
        )
        if schema_mode == "compact":
            return_mode = (
                "Return minified JSON only. Do not use Markdown, code fences, explanations, or multiline strings. "
                "The JSON must match the exact compact schema."
            )
            item_instruction = (
                "Each item must match the compact schema. Keep candidate_prompt concise and standalone; "
                "all other fields must be one short sentence."
            )
            format_rules = compact_output_rules
        else:
            return_mode = "Return strict JSON only."
            item_instruction = (
                "Each item must include candidate_prompt, student_interpretation_of_question, target_error_pattern, "
                "accuracy_repair_rule, diversity_contribution, error_correlation_reduction, task_alignment_rule, "
                "peer_redundancy_avoidance, expected_accuracy_effect, expected_diversity_effect, risk_control, rationale."
            )
            format_rules = (
                "Return JSON with a candidates list. Do not use Markdown or code fences. "
                f"Exact schema:\n{schema}"
            )
        system_prompt = (
            "You are the Student in a Teacher-Critic-Student prompt optimization system.\n\n"
            "You will receive:\n- the current parent prompt\n- an approved Socratic guiding question from the Teacher\n"
            "- Critic reviews of that question\n- abstract diagnostics about problem type, error type, diversity gap, "
            "error correlation, and peer behavior\n\n"
            "Your job is to generate candidate prompts for the target agent.\n\n"
            "Each candidate prompt must:\n- directly answer the approved guiding question\n- be a complete standalone prompt\n"
            "- align with the problem type and answer format\n- repair the target error pattern\n"
            "- improve target-agent accuracy\n- contribute useful reasoning diversity\n"
            "- reduce redundant behavior with peer prompts\n- avoid invalid, overlong, or generic outputs\n\n"
            "Preserve effective mechanisms from the parent and make one local, executable reasoning change. "
            "Do not create superficial diversity by changing persona or expert-role labels.\n"
            "Do not use gold answers.\nDo not include concrete sample text.\nDo not include answer labels from examples.\n"
            "Do not write hard-coded task-specific roles.\nDo not simply ask the solver to 'think more carefully'.\n"
            f"Do not only paraphrase the parent prompt.\n\n{return_mode}"
        )
        if self._v7_residual_protocol_enabled():
            item_instruction += (
                " Include non-empty preserved_mechanisms, exactly one modified_mechanism, change_summary, "
                "target_residual_family, expected_shared_error_effect, and risk_control."
            )
            system_prompt += (
                "\nMake exactly one local mechanism change. Preserve the listed effective and pivotal-correct mechanisms. "
                "State how the change should reduce shared error without manufacturing disagreement."
            )
        if self._is_v82_hybrid():
            item_instruction += (
                " Return exactly two candidates in this order: task_specific_repair, then mechanism_alternative. "
                "Each must include candidate_type, ordered mechanism_steps, target_failure_buckets, and expected_effect."
            )
            system_prompt += (
                "\nThe first candidate must be a task-specific repair grounded in the supplied failure buckets. "
                "The second must change at least one substantive decision operation relative to the first and parent; "
                "persona changes, synonyms, renumbering, and extra generic verification do not count."
            )
        if self._is_stable_qd_lineage():
            lineage_context = teacher_context.get("stable_lineage_context", {})
            committed = str(lineage_context.get("lineage_status", "uncommitted")) == "committed"
            system_prompt += (
                "\nThe mechanism_alternative must differ from the parent or committed peer mechanisms in at least one core operation. "
                "It must target the same observed task errors; do not create novelty unrelated to competence. "
                + (
                    "For this committed agent, prefer a structural variant near its anchor, but still emit a genuine alternative when justified."
                    if committed
                    else "This agent is not committed: a substantial mechanism departure is allowed and must not be reduced to a local paraphrase."
                )
            )
        user_prompt = (
            "Generate up to requested_candidates candidate prompts. Return JSON with a candidates list. "
            f"{item_instruction}\n\n"
            f"{format_rules}\n\n"
            f"target_agent_id: {agent_id}\nrequested_candidates: {num_candidates}\n\n"
            f"parent_prompt:\n{parent_prompt}\n\n"
            f"approved_teacher_question:\n{json.dumps(approved_teacher_question, ensure_ascii=False, indent=2)}\n\n"
            f"teacher_context:\n{json.dumps(teacher_context, ensure_ascii=False, indent=2)}"
        )
        text = await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(getattr(self.cfg, "student_temperature", self.cfg.optimizer_temperature)),
            max_tokens=int(getattr(self.cfg, "student_max_tokens", self.cfg.optimizer_max_tokens)),
            stage=f"student_optimizer_agent_{agent_id}",
            client_role="optimizer",
        )
        raw_text = text or ""
        raw_preview = normalize_spaces(raw_text)[:1000]
        diagnostics = self._empty_optimizer_generation_diagnostics()
        diagnostics.update(
            {
                "student_raw_response_empty": not bool(raw_text.strip()),
                "student_raw_response_preview": raw_preview,
                "student_json_parse_failed": False,
                "student_json_parse_error": "",
                "student_json_has_candidates_key": False,
                "student_candidates_is_list": False,
                "student_candidates_empty_list": False,
                "student_refusal_or_explanation": False,
                "student_failure_stage": "none",
            }
        )
        obj = extract_json_obj(raw_text)
        if diagnostics["student_raw_response_empty"] or obj is None or not isinstance(obj, dict):
            diagnostics["student_failure_stage"] = "raw_empty" if diagnostics["student_raw_response_empty"] else "json_parse_failed"
            diagnostics["student_json_parse_failed"] = not diagnostics["student_raw_response_empty"]
            diagnostics["student_refusal_or_explanation"] = self._student_refusal_or_explanation(raw_preview)
            if diagnostics["student_refusal_or_explanation"]:
                diagnostics["student_failure_stage"] = "refusal_or_explanation"

            max_retries = max(0, int(getattr(self.cfg, "student_json_max_retries", 1) or 0))
            retry_enabled = bool(int(getattr(self.cfg, "student_json_retry_on_parse_fail", True)))
            retry_text = ""
            if retry_enabled and max_retries > 0:
                diagnostics["student_json_retry_attempted"] = True
                for _ in range(max_retries):
                    retry_text = await self.retry_student_candidates_json_only(
                        previous_raw_text=raw_text,
                        approved_teacher_question=approved_teacher_question,
                        num_candidates=num_candidates,
                        agent_id=agent_id,
                    )
                    diagnostics["student_json_retry_raw_response_preview"] = normalize_spaces(retry_text or "")[:1000]
                    if retry_text and retry_text.strip():
                        diagnostics["student_raw_response_empty"] = False
                    retry_obj = extract_json_obj(retry_text or "")
                    if isinstance(retry_obj, dict):
                        obj = retry_obj
                        diagnostics["student_json_retry_succeeded"] = True
                        diagnostics["student_json_parse_failed"] = False
                        diagnostics["student_json_parse_error"] = ""
                        diagnostics["student_failure_stage"] = "none"
                        break
                if not diagnostics["student_json_retry_succeeded"]:
                    diagnostics["student_json_parse_error"] = (
                        "retry_raw_empty"
                        if diagnostics["student_failure_stage"] == "raw_empty"
                        else "retry_json_parse_failed"
                    )

            if obj is None or not isinstance(obj, dict):
                repair_enabled = bool(int(getattr(self.cfg, "student_json_repair_enabled", True)))
                repair_source = retry_text or raw_text
                if repair_enabled and bool(str(repair_source or "").strip()):
                    repair_source = retry_text or raw_text
                    repair = await self.repair_student_json_response(
                        raw_text=repair_source,
                        expected_num_candidates=num_candidates,
                    )
                    diagnostics["student_json_repair_attempted"] = True
                    diagnostics["student_json_repair_succeeded"] = bool(repair.get("repaired", False))
                    diagnostics["student_json_repair_raw_response_preview"] = str(repair.get("repair_raw_response_preview", ""))[:1000]
                    diagnostics["student_json_repair_failure_reason"] = str(repair.get("repair_failure_reason", ""))[:500]
                    repair_obj = repair.get("obj")
                    if isinstance(repair_obj, dict):
                        obj = repair_obj
                        diagnostics["student_json_parse_failed"] = False
                        diagnostics["student_json_parse_error"] = ""
                        diagnostics["student_failure_stage"] = "none"

            if obj is None or not isinstance(obj, dict):
                if diagnostics.get("student_failure_stage") != "raw_empty":
                    diagnostics["student_failure_stage"] = "json_parse_failed"
                    diagnostics["student_json_parse_failed"] = True
                if not diagnostics.get("student_json_parse_error"):
                    diagnostics["student_json_parse_error"] = (
                        str(diagnostics.get("student_json_repair_failure_reason", ""))
                        or ("raw_empty" if diagnostics["student_failure_stage"] == "raw_empty" else "json_parse_failed")
                    )
                return {"candidates": [], "diagnostics": diagnostics}

        diagnostics["student_json_has_candidates_key"] = "candidates" in obj
        if "candidates" not in obj:
            diagnostics["student_failure_stage"] = "missing_candidates_key"
            return {"candidates": [], "diagnostics": diagnostics}

        candidates = obj.get("candidates", None)
        if not isinstance(candidates, list):
            diagnostics["student_candidates_is_list"] = False
            diagnostics["student_failure_stage"] = "candidates_not_list"
            return {"candidates": [], "diagnostics": diagnostics}

        diagnostics["student_candidates_is_list"] = True
        diagnostics["student_candidates_empty_list"] = len(candidates) == 0
        if len(candidates) == 0:
            diagnostics["student_failure_stage"] = "empty_candidates_list"
        return {"candidates": candidates, "diagnostics": diagnostics}

    async def propose_candidates_teacher_critic_student(
        self,
        agent_id: int,
        parent_prompt: str,
        overlap_diagnosis: Dict[str, Any],
        num_candidates: int,
        generation_batch: Optional[Dict[str, Any]] = None,
        generation_batches: Optional[List[Dict[str, Any]]] = None,
        refill_feedback: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        update_diagnosis = overlap_diagnosis
        prompt_roles = [
            r for r in update_diagnosis.get("prompt_roles", [])
            if isinstance(r, dict)
        ]
        target_role_spec = next((r for r in prompt_roles if int(r.get("agent_id", -1)) == int(agent_id)), {})
        peer_role_specs = [r for r in prompt_roles if int(r.get("agent_id", -1)) != int(agent_id)]
        if generation_batches is None:
            generation_batches = [dict(generation_batch or {"batch_type": "window_update_diagnosis", "cases": [], "purpose": "general reward-relevant window repair"})]
        generation_batches = [dict(x) for x in generation_batches if isinstance(x, dict)]
        if not generation_batches:
            generation_batches = [{"batch_type": "window_update_diagnosis", "cases": [], "purpose": "general reward-relevant window repair"}]

        agent_pressures = update_diagnosis.get("per_agent_overlap_pressure", [])
        agent_invalid_rates = update_diagnosis.get("per_agent_invalid_rate", [])
        agent_error_counts = update_diagnosis.get("per_agent_error_count", [])
        agent_team_wrong_counts = update_diagnosis.get("per_agent_team_wrong_error_count", [])
        agent_pivotal_fix_counts = update_diagnosis.get("per_agent_pivotal_fix_count", [])
        agent_dominant_wrong_counts = update_diagnosis.get("per_agent_dominant_wrong_redundancy_count", [])
        window_stats = {
            "diagnosis_type": update_diagnosis.get("diagnosis_type", "vote_update"),
            "window_vote_acc": update_diagnosis.get("window_vote_acc", update_diagnosis.get("team_accuracy", 0.0)),
            "window_mean_vote_margin": update_diagnosis.get("window_mean_vote_margin", -1.0),
            "window_mean_boundary_useful_diversity": update_diagnosis.get("window_mean_boundary_useful_diversity", 0.0),
            "mean_reward_pressure": update_diagnosis.get("mean_reward_pressure", 0.0),
            "mean_window_overlap": update_diagnosis.get("mean_window_overlap", 0.0),
            "homogeneity_overlap_threshold": update_diagnosis.get("homogeneity_overlap_threshold", self.cfg.homogeneity_overlap_threshold),
            "target_overlap_pressure": agent_pressures[agent_id] if agent_id < len(agent_pressures) else 0.0,
            "target_homogeneous_case_count": (update_diagnosis.get("homogeneous_case_counts", [0] * len(self.agents))[agent_id] if agent_id < len(update_diagnosis.get("homogeneous_case_counts", [])) else 0),
            "target_invalid_rate": agent_invalid_rates[agent_id] if agent_id < len(agent_invalid_rates) else 0.0,
            "target_error_count": agent_error_counts[agent_id] if agent_id < len(agent_error_counts) else 0,
            "target_team_wrong_error_count": agent_team_wrong_counts[agent_id] if agent_id < len(agent_team_wrong_counts) else 0,
            "target_pivotal_fix_count": agent_pivotal_fix_counts[agent_id] if agent_id < len(agent_pivotal_fix_counts) else 0,
            "target_dominant_wrong_redundancy_count": agent_dominant_wrong_counts[agent_id] if agent_id < len(agent_dominant_wrong_counts) else 0,
            "target_pivotal_fix_rate": (update_diagnosis.get("per_agent_pivotal_fix_rate", [0.0] * len(self.agents))[agent_id] if agent_id < len(update_diagnosis.get("per_agent_pivotal_fix_rate", [])) else 0.0),
            "target_near_boundary_error_rate": (update_diagnosis.get("per_agent_near_boundary_error_rate", [0.0] * len(self.agents))[agent_id] if agent_id < len(update_diagnosis.get("per_agent_near_boundary_error_rate", [])) else 0.0),
            "target_shared_error_rate": (update_diagnosis.get("per_agent_shared_error_rate", [0.0] * len(self.agents))[agent_id] if agent_id < len(update_diagnosis.get("per_agent_shared_error_rate", [])) else 0.0),
            "target_pivotal_hold_rate": (update_diagnosis.get("per_agent_pivotal_hold_rate", [0.0] * len(self.agents))[agent_id] if agent_id < len(update_diagnosis.get("per_agent_pivotal_hold_rate", [])) else 0.0),
            "capability_coverage_gap": dict(update_diagnosis.get("capability_coverage_gap", {})),
            "refill_feedback": dict(refill_feedback or {}),
        }
        validity_constraints = {
            "invalid_repair_priority": bool(window_stats["target_invalid_rate"] >= float(self.cfg.invalid_repair_rate_threshold)),
            "required_final_answer_line": True,
            "avoid_empty_or_repetitive_trace": True,
            "do_not_copy_case_content": True,
        }
        teacher_context = self._build_teacher_context(
            agent_id=agent_id,
            parent_prompt=parent_prompt,
            target_role_spec=target_role_spec,
            peer_role_specs=peer_role_specs,
            window_stats=window_stats,
            validity_constraints=validity_constraints,
            generation_batches=generation_batches,
        )
        # Preserve the beam parent's stable ID for call-level provenance.
        parent_id = str((TCS_AUDIT_CONTEXT.get() or {}).get("parent_id") or self._hash(parent_prompt))
        tcs_call_group_id = str((TCS_AUDIT_CONTEXT.get() or {}).get("tcs_call_group_id") or "")
        execution_session_id = str((TCS_AUDIT_CONTEXT.get() or {}).get("execution_session_id") or getattr(self, "execution_session_id", ""))
        update_attempt_id = str((TCS_AUDIT_CONTEXT.get() or {}).get("update_attempt_id") or "")
        call_context = dict(TCS_AUDIT_CONTEXT.get() or {})
        call_context.update(
            {
                "optimizer_architecture": "teacher_critic_student",
                "agent_id": int(agent_id),
                "parent_id": parent_id,
                "teacher_critic_round": 1,
            }
        )
        context_token = TCS_AUDIT_CONTEXT.set(call_context)
        try:
            approved = await self.generate_approved_teacher_question(
                agent_id=agent_id,
                parent_prompt=parent_prompt,
                teacher_context=teacher_context,
                requested_candidates=num_candidates,
            )
        finally:
            TCS_AUDIT_CONTEXT.reset(context_token)
        diagnostics = self._empty_optimizer_generation_diagnostics()
        diagnostics["optimizer_architecture"] = "teacher_critic_student"
        diagnostics["tcs_call_group_id"] = tcs_call_group_id
        diagnostics["execution_session_id"] = execution_session_id
        diagnostics["update_attempt_id"] = update_attempt_id
        teacher_question = approved.get("teacher_question", {}) if isinstance(approved, dict) else {}
        critic_reviews = approved.get("critic_reviews", []) if isinstance(approved, dict) else []
        forced_best = bool(approved.get("teacher_question_forced_best_score", False)) if isinstance(approved, dict) else False
        approved_for_student = bool(approved.get("approved", False)) or forced_best
        forced_best_review = approved.get("teacher_question_forced_best_review", {}) if isinstance(approved, dict) else {}
        last_review = (
            forced_best_review
            if forced_best and isinstance(forced_best_review, dict)
            else (critic_reviews[-1] if critic_reviews and isinstance(critic_reviews[-1], dict) else {})
        )
        guiding_question = (
            str(teacher_question.get("socratic_guiding_question", "")).strip()
            if isinstance(teacher_question, dict)
            else ""
        )
        teacher_question_usable = bool(guiding_question)
        approved_for_student = approved_for_student and teacher_question_usable
        diagnostics.update(
            {
                "teacher_question": guiding_question,
                "teacher_question_approved": bool(approved.get("approved", False)) and teacher_question_usable,
                "teacher_question_rejected": not approved_for_student,
                "teacher_question_forced_best_score": forced_best,
                "teacher_question_forced_best_round": int(approved.get("teacher_question_forced_best_round", 0) or 0),
                "teacher_question_forced_below_threshold": bool(approved.get("teacher_question_forced_below_threshold", False)),
                "teacher_question_score": self._safe_float(last_review.get("score", 0.0), 0.0),
                "teacher_critic_rounds": int(approved.get("teacher_critic_rounds", len(critic_reviews)) or 0),
                "teacher_quality_critique": str(last_review.get("quality_critique", "")),
                "teacher_specificity_critique": str(last_review.get("specificity_critique", "")),
                "teacher_task_alignment_critique": str(last_review.get("task_alignment_critique", "")),
                "teacher_error_alignment_critique": str(last_review.get("error_alignment_critique", "")),
                "teacher_diversity_critique": str(last_review.get("diversity_critique", "")),
                "teacher_rewrite_count": int(approved.get("teacher_rewrite_count", 0) or 0),
                "num_teacher_calls": 1,
                "num_critic_calls": int(approved.get("teacher_critic_rounds", len(critic_reviews)) or 0),
                "num_teacher_rewrite_calls": int(approved.get("teacher_rewrite_count", 0) or 0),
            }
        )
        if not approved_for_student:
            diagnostics["teacher_question_rejection_reason"] = (
                "empty_teacher_question"
                if not teacher_question_usable
                else str(
                    approved.get("teacher_question_rejection_reason", "")
                    or last_review.get("rewrite_instruction", "")
                    or last_review.get("quality_critique", "")
                    or "teacher question failed critic review"
                )
            )
            diagnostics["optimizer_underfilled"] = True
            self._record_optimizer_generation_diagnostics(agent_id, parent_prompt, diagnostics)
            return []

        student_context = dict(TCS_AUDIT_CONTEXT.get() or {})
        student_context.update(
            {
                "optimizer_architecture": "teacher_critic_student",
                "agent_id": int(agent_id),
                "parent_id": parent_id,
                "teacher_critic_round": int(diagnostics["teacher_critic_rounds"]),
            }
        )
        student_context_token = TCS_AUDIT_CONTEXT.set(student_context)
        try:
            student_result = await self.generate_student_candidates(
                agent_id=agent_id,
                parent_prompt=parent_prompt,
                approved_teacher_question=approved,
                teacher_context=teacher_context,
                num_candidates=num_candidates,
            )
        finally:
            TCS_AUDIT_CONTEXT.reset(student_context_token)
        if isinstance(student_result, dict):
            student_candidates = student_result.get("candidates", [])
            student_diag = student_result.get("diagnostics", {})
            if isinstance(student_diag, dict):
                self._merge_student_diagnostics(diagnostics, student_diag)
        else:
            student_candidates = student_result
        diagnostics["student_candidate_count_raw"] = len(student_candidates) if isinstance(student_candidates, list) else 0
        diagnostics["num_student_calls"] = 1
        diagnostics["num_student_retry_calls"] = int(bool(diagnostics.get("student_json_retry_attempted", False)))
        diagnostics["num_student_repair_calls"] = int(bool(diagnostics.get("student_json_repair_attempted", False)))
        diagnostics["optimizer_raw_candidate_count"] = int(diagnostics["student_candidate_count_raw"])
        parsed: List[Dict[str, Any]] = []
        seen_signatures: set = set()
        seen_candidate_types: set = set()
        filter_reasons: List[str] = []
        if isinstance(student_candidates, list):
            for item in student_candidates:
                if not isinstance(item, dict):
                    diagnostics["optimizer_schema_filtered_count"] += 1
                    filter_reasons.append("schema")
                    continue
                length_audit: Dict[str, Any] = {}
                if bool(getattr(self.cfg, "competence_depth_enabled", False)):
                    prepared, length_audit = self._prepare_v8_candidate_text_fields(item)
                    if prepared is None:
                        diagnostics["optimizer_schema_filtered_count"] += 1
                        filter_reasons.append("candidate_prompt_overlength")
                        continue
                    item = prepared
                else:
                    original_prompt = str(item.get("candidate_prompt", ""))
                    item = self._truncate_candidate_text_fields(item)
                    self.truncated_prompt_count = int(getattr(self, "truncated_prompt_count", 0)) + int(
                        str(item.get("candidate_prompt", "")) != normalize_spaces(original_prompt)
                    )
                if self._v7_residual_protocol_enabled():
                    if not str(item.get("modified_mechanism", "")).strip():
                        item["modified_mechanism"] = str(item.get("new_or_modified_mechanism", "")).strip()
                    if not str(item.get("target_residual_family", "")).strip():
                        item["target_residual_family"] = CapabilityResidualFamily.UNKNOWN.value
                    if not str(item.get("expected_shared_error_effect", "")).strip():
                        item["expected_shared_error_effect"] = str(item.get("error_correlation_reduction", "")).strip()
                    if not str(item.get("change_summary", "")).strip():
                        item["change_summary"] = str(item.get("modified_mechanism", "")).strip()
                if self._is_v82_hybrid():
                    candidate_type = str(item.get("candidate_type", "")).strip().lower()
                    type_rejection = self._hybrid_candidate_type_rejection_reason(candidate_type, seen_candidate_types)
                    if type_rejection:
                        diagnostics["optimizer_schema_filtered_count"] += 1
                        filter_reasons.append(type_rejection)
                        continue
                missing_fields = self._missing_optimizer_fields(item, architecture="teacher_critic_student")
                if missing_fields:
                    diagnostics["optimizer_schema_filtered_count"] += 1
                    if "candidate_prompt" in missing_fields:
                        diagnostics["optimizer_empty_prompt_count"] += 1
                    diagnostics["student_missing_required_field_count"] += len(missing_fields)
                    existing = list(diagnostics.get("student_missing_required_fields", []))
                    existing.extend(missing_fields)
                    diagnostics["student_missing_required_fields"] = sorted(set(str(x) for x in existing))
                    reason = "missing_required_student_fields:" + ",".join(missing_fields)
                    filter_reasons.append(reason)
                    continue
                prompt = str(item.get("candidate_prompt", "")).strip()
                prompt, sanitized = self._sanitize_prompt(prompt, agent_id)
                diagnostics["optimizer_sanitized_count"] += int(bool(sanitized))
                mechanism_signature = normalize_mechanism_signature(item.get("mechanism_steps", []))
                allow_substantive_parent_extension = bool(
                    self._is_v82_hybrid()
                    and mechanism_signature
                    and self._prompt_signature(prompt) != self._prompt_signature(parent_prompt)
                )
                if self._is_redundant_candidate_prompt(
                    parent_prompt,
                    prompt,
                    seen_signatures,
                    allow_substantive_parent_extension=allow_substantive_parent_extension,
                ):
                    diagnostics["optimizer_redundant_filtered_count"] += 1
                    filter_reasons.append("redundant")
                    continue
                if not prompt:
                    diagnostics["optimizer_empty_prompt_count"] += 1
                    filter_reasons.append("empty_prompt")
                    continue
                mechanism_alternative_invalid = bool(
                    self._is_v82_hybrid()
                    and str(item.get("candidate_type", "")).strip().lower() == "mechanism_alternative"
                    and parsed
                    and mechanism_signature == list(parsed[0].get("mechanism_signature", []))
                )
                if mechanism_alternative_invalid:
                    diagnostics["mechanism_alternative_invalid"] = True
                    diagnostics["mechanism_alternative_invalid_count"] = int(
                        diagnostics.get("mechanism_alternative_invalid_count", 0) or 0
                    ) + 1
                    filter_reasons.append("mechanism_alternative_same_signature")
                    continue
                seen_signatures.add(self._prompt_signature(prompt))
                if self._is_v82_hybrid():
                    seen_candidate_types.add(str(item.get("candidate_type", "")).strip().lower())
                batch_idx = min(len(parsed), len(generation_batches) - 1)
                parsed.append(
                    {
                        "candidate_prompt": prompt,
                        "student_interpretation_of_question": str(item.get("student_interpretation_of_question", "")),
                        "target_error_pattern": str(item.get("target_error_pattern", "")),
                        "accuracy_repair_rule": str(item.get("accuracy_repair_rule", "")),
                        "diversity_contribution": str(item.get("diversity_contribution", "")),
                        "error_correlation_reduction": str(item.get("error_correlation_reduction", "")),
                        "task_alignment_rule": str(item.get("task_alignment_rule", "")),
                        "peer_redundancy_avoidance": str(item.get("peer_redundancy_avoidance", "")),
                        "expected_accuracy_effect": str(item.get("expected_accuracy_effect", "")),
                        "expected_diversity_effect": str(item.get("expected_diversity_effect", "")),
                        "risk_control": str(item.get("risk_control", "")),
                        "rationale": str(item.get("rationale", "")),
                        "change_summary": str(item.get("change_summary", "")),
                        "preserved_mechanisms": list(item.get("preserved_mechanisms", [])) if isinstance(item.get("preserved_mechanisms", []), list) else [],
                        "new_or_modified_mechanism": str(item.get("new_or_modified_mechanism", "")),
                        "modified_mechanism": str(item.get("modified_mechanism", item.get("new_or_modified_mechanism", ""))),
                        "target_residual_family": str(item.get("target_residual_family", CapabilityResidualFamily.UNKNOWN.value)),
                        "expected_shared_error_effect": str(item.get("expected_shared_error_effect", "")),
                        "candidate_type": str(item.get("candidate_type", "")),
                        "mechanism_steps": [str(value) for value in item.get("mechanism_steps", [])] if isinstance(item.get("mechanism_steps", []), list) else [],
                        "target_failure_buckets": [str(value) for value in item.get("target_failure_buckets", [])] if isinstance(item.get("target_failure_buckets", []), list) else [],
                        "expected_effect": str(item.get("expected_effect", "")),
                        "mechanism_signature": mechanism_signature,
                        "mechanism_alternative_invalid": mechanism_alternative_invalid,
                        **length_audit,
                        "candidate_source": "teacher_critic_student",
                        "tcs_call_group_id": tcs_call_group_id,
                        "execution_session_id": execution_session_id,
                        "update_attempt_id": update_attempt_id,
                        "teacher_question": teacher_question,
                        "teacher_question_score": self._safe_float(diagnostics.get("teacher_question_score", 0.0), 0.0),
                        "teacher_question_approved": bool(diagnostics.get("teacher_question_approved", False)),
                        "teacher_critic_rounds": int(diagnostics["teacher_critic_rounds"]),
                        "critic_reviews": critic_reviews,
                        "generation_batch_type": str(generation_batches[batch_idx].get("batch_type", "")),
                        "generation_case_ids": [
                            str(c.get("case_id", ""))
                            for c in generation_batches[batch_idx].get("cases", [])
                            if isinstance(c, dict)
                        ],
                    }
                )
                if len(parsed) >= num_candidates:
                    break
        if self._is_v82_hybrid():
            type_order = {"task_specific_repair": 0, "mechanism_alternative": 1}
            parsed.sort(key=lambda item: type_order.get(str(item.get("candidate_type", "")), 99))
        diagnostics["student_candidate_count_final"] = len(parsed)
        diagnostics["student_candidate_filtered_count"] = max(0, int(diagnostics["student_candidate_count_raw"]) - len(parsed))
        diagnostics["student_candidate_filter_reasons"] = filter_reasons
        diagnostics["student_all_candidates_filtered"] = bool(int(diagnostics["student_candidate_count_raw"]) > 0 and not parsed)
        if diagnostics["student_all_candidates_filtered"]:
            has_schema = any(
                "missing_required" in str(reason)
                or "schema" in str(reason)
                or "empty_prompt" in str(reason)
                for reason in filter_reasons
            )
            has_redundant = any("redundant" in str(reason) for reason in filter_reasons)
            if has_schema and has_redundant:
                diagnostics["student_failure_stage"] = "all_candidates_filtered_mixed"
            elif has_schema:
                diagnostics["student_failure_stage"] = "all_candidates_filtered_schema"
            elif has_redundant:
                diagnostics["student_failure_stage"] = "all_candidates_filtered_redundant"
            else:
                diagnostics["student_failure_stage"] = "unknown"
        diagnostics["optimizer_final_candidate_count"] = len(parsed)
        diagnostics["optimizer_underfilled"] = bool(len(parsed) < int(num_candidates))
        diagnostics = self._record_optimizer_generation_diagnostics(agent_id, parent_prompt, diagnostics)
        metadata = self._teacher_metadata_from_diagnostics(diagnostics)
        for item in parsed:
            item["optimizer_generation_diagnostics"] = dict(diagnostics)
            item.update(metadata)
        return parsed[:num_candidates]

    async def propose_candidates(
        self,
        agent_id: int,
        parent_prompt: str,
        overlap_diagnosis: Dict[str, Any],
        num_candidates: int,
        generation_batch: Optional[Dict[str, Any]] = None,
        generation_batches: Optional[List[Dict[str, Any]]] = None,
        refill_feedback: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        architecture = str(getattr(self.cfg, "optimizer_architecture", "teacher_critic_student") or "teacher_critic_student").lower()
        if architecture == "teacher_critic_student":
            return await self.propose_candidates_teacher_critic_student(
                agent_id=agent_id,
                parent_prompt=parent_prompt,
                overlap_diagnosis=overlap_diagnosis,
                num_candidates=num_candidates,
                generation_batch=generation_batch,
                generation_batches=generation_batches,
                refill_feedback=refill_feedback,
            )
        return await self.propose_candidates_one_shot(
            agent_id=agent_id,
            parent_prompt=parent_prompt,
            overlap_diagnosis=overlap_diagnosis,
            num_candidates=num_candidates,
            generation_batch=generation_batch,
            generation_batches=generation_batches,
        )

    async def propose_candidates_one_shot(
        self,
        agent_id: int,
        parent_prompt: str,
        overlap_diagnosis: Dict[str, Any],
        num_candidates: int,
        generation_batch: Optional[Dict[str, Any]] = None,
        generation_batches: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        update_diagnosis = overlap_diagnosis
        prompt_roles = [
            r for r in update_diagnosis.get("prompt_roles", [])
            if isinstance(r, dict)
        ]
        target_role_spec = next((r for r in prompt_roles if int(r.get("agent_id", -1)) == int(agent_id)), {})
        peer_role_specs = [r for r in prompt_roles if int(r.get("agent_id", -1)) != int(agent_id)]
        if generation_batches is None:
            generation_batches = [dict(generation_batch or {"batch_type": "window_update_diagnosis", "cases": [], "purpose": "general reward-relevant window repair"})]
        generation_batches = [dict(x) for x in generation_batches if isinstance(x, dict)]
        if not generation_batches:
            generation_batches = [{"batch_type": "window_update_diagnosis", "cases": [], "purpose": "general reward-relevant window repair"}]
        if self._is_accuracy_only_mode():
            return await self._propose_accuracy_candidates(
                agent_id=agent_id,
                parent_prompt=parent_prompt,
                accuracy_diagnosis=update_diagnosis,
                num_candidates=num_candidates,
                generation_batches=generation_batches,
            )
        agent_pressures = update_diagnosis.get("per_agent_overlap_pressure", [])
        agent_invalid_rates = update_diagnosis.get("per_agent_invalid_rate", [])
        agent_error_counts = update_diagnosis.get("per_agent_error_count", [])
        agent_team_wrong_counts = update_diagnosis.get("per_agent_team_wrong_error_count", [])
        agent_pivotal_fix_counts = update_diagnosis.get("per_agent_pivotal_fix_count", [])
        agent_dominant_wrong_counts = update_diagnosis.get("per_agent_dominant_wrong_redundancy_count", [])
        window_stats = {
            "diagnosis_type": update_diagnosis.get("diagnosis_type", "vote_update"),
            "window_vote_acc": update_diagnosis.get("window_vote_acc", update_diagnosis.get("team_accuracy", 0.0)),
            "window_mean_vote_margin": update_diagnosis.get("window_mean_vote_margin", -1.0),
            "window_mean_boundary_useful_diversity": update_diagnosis.get("window_mean_boundary_useful_diversity", 0.0),
            "mean_reward_pressure": update_diagnosis.get("mean_reward_pressure", 0.0),
            "mean_window_overlap": update_diagnosis.get("mean_window_overlap", 0.0),
            "homogeneity_overlap_threshold": update_diagnosis.get("homogeneity_overlap_threshold", self.cfg.homogeneity_overlap_threshold),
            "target_overlap_pressure": agent_pressures[agent_id] if agent_id < len(agent_pressures) else 0.0,
            "target_homogeneous_case_count": (update_diagnosis.get("homogeneous_case_counts", [0] * len(self.agents))[agent_id] if agent_id < len(update_diagnosis.get("homogeneous_case_counts", [])) else 0),
            "target_invalid_rate": agent_invalid_rates[agent_id] if agent_id < len(agent_invalid_rates) else 0.0,
            "target_error_count": agent_error_counts[agent_id] if agent_id < len(agent_error_counts) else 0,
            "target_team_wrong_error_count": agent_team_wrong_counts[agent_id] if agent_id < len(agent_team_wrong_counts) else 0,
            "target_pivotal_fix_count": agent_pivotal_fix_counts[agent_id] if agent_id < len(agent_pivotal_fix_counts) else 0,
            "target_dominant_wrong_redundancy_count": agent_dominant_wrong_counts[agent_id] if agent_id < len(agent_dominant_wrong_counts) else 0,
        }
        validity_constraints = {
            "invalid_repair_priority": bool(window_stats["target_invalid_rate"] >= float(self.cfg.invalid_repair_rate_threshold)),
            "required_final_answer_line": True,
            "avoid_empty_or_repetitive_trace": True,
            "do_not_copy_case_content": True,
        }
        safe_generation_batches = []
        for batch in generation_batches:
            safe_generation_batches.append(
                {
                    **batch,
                    "cases": [
                        self._optimizer_case_payload(c)
                        for c in batch.get("cases", [])
                        if isinstance(c, dict)
                    ],
                }
            )
        has_target_error_batches = any(str(b.get("batch_type", "")) == "target_error_repair" and b.get("cases") for b in generation_batches)
        if has_target_error_batches:
            system_prompt = (
                "You are a prompt optimizer for a multi-agent reasoning team.\n"
                "Generate executable role prompts that improve the target agent's answer accuracy on its observed error patterns while preserving useful reasoning diversity.\n"
                "Diversity is valuable only when it creates valid, answer-improving reasoning behavior, not superficial wording differences.\n"
                "Use the supplied target-error cases, peer behavior summaries, parent prompt, role previews, and batch diagnoses.\n"
                "Do not use gold answers, concrete task text, answer labels, or sample-specific content.\n"
                "Treat trace previews as abstract behavioral evidence; do not copy their wording into the new prompt.\n"
                "Optimize for behavior that will be visible in the solver trace and easy to evaluate for role execution.\n"
                "Return strict JSON only."
            )
        else:
            system_prompt = (
                "You are a prompt optimizer for a multi-agent reasoning team.\n"
                "Generate executable role prompts that preserve answer reliability while adding useful reasoning diversity.\n"
                "Use only the supplied parent prompt, prompt-role previews, window statistics, and generation-batch diagnoses.\n"
                "The homogeneous cases were selected by the system, not by you. You are only a candidate prompt proposer.\n"
                "Do not use gold answers, concrete task text, options, labels, or answer-specific content.\n"
                "Treat trace previews as abstract behavioral evidence; do not copy their wording into the new prompt.\n"
                "Optimize for behavior that will be visible in the solver trace and easy to evaluate for role execution.\n"
                "Return strict JSON only."
            )
        user_prompt = (
            "Revise the target agent prompt using the case-aware generation batches below.\n"
            "Priority order:\n"
            "1. Repair the target agent's observed error patterns.\n"
            "2. Preserve or improve target-agent answer accuracy.\n"
            "3. Add useful reasoning diversity only when it helps correctness or error rescue.\n"
            "4. Avoid invalid, verbose, generic, or merely paraphrased prompts.\n"
            "5. Do not optimize for trace difference alone.\n"
            "Each candidate must primarily address one supplied generation batch; do not merge all batches into one generic prompt.\n"
            "The new prompt must address the provided cases as reasoning-pattern evidence, not as sample content to memorize.\n"
            "Write concrete reasoning behavior, not slogans such as 'be diverse' or 'avoid redundancy'.\n"
            "A candidate is invalid if it only paraphrases the parent prompt, appends generic caution, asks the solver to be more accurate, "
            "or changes style without adding a concrete error-repair procedure.\n"
            "Each candidate_prompt must contain a concrete reasoning procedure, a specific error-repair behavior, final answer discipline, "
            "and a short verification step.\n"
            "Write a complete short role prompt, not a suffix to append to the parent prompt. "
            "Do not repeat generic instructions already present in the parent prompt. "
            "Do not use the phrase 'Use a distinct decision procedure'. "
            "Prefer a short role prompt with 2-4 explicit procedure steps, a fallback strategy, and validity checks.\n"
            "The prompt should create a different reasoning route from peer roles only when that helps correctness or error rescue.\n"
            "Never include concrete question text, answer text, options, labels, sample hashes, or FINAL_ANSWER templates.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "candidates": [\n'
            '    {"candidate_prompt": str, "role_name": str, "decision_procedure": [str, ...], "when_to_use": str, "fallback_strategy": str, "anti_overlap_rule": str, "validity_checks": [str, ...], "target_error_pattern": str, "accuracy_repair_rule": str, "expected_accuracy_effect": str, "rationale": str, "source_batch_type": str},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Return exactly requested_candidates distinct candidates. "
            "Set source_batch_type to the exact batch_type that the candidate primarily addresses. "
            "target_error_pattern names the main observed pattern repaired by the candidate. "
            "accuracy_repair_rule is the concrete behavior enforced to improve target-agent correctness. "
            "expected_accuracy_effect explains why this improves the target agent rather than merely changing wording. "
            "If source_batch_type repeats, the repeated candidates must use meaningfully different executable procedures. "
            "Do not include batch names, sample identifiers, or meta-evaluation language inside candidate_prompt.\n\n"
            f"target_agent_id: {agent_id}\n"
            f"requested_candidates: {num_candidates}\n\n"
            f"current_parent_prompt:\n{parent_prompt}\n\n"
            f"target_role_spec:\n{json.dumps(target_role_spec, ensure_ascii=False, indent=2)}\n\n"
            f"peer_role_specs:\n{json.dumps(peer_role_specs, ensure_ascii=False, indent=2)}\n\n"
            f"window_overlap_statistics:\n{json.dumps(window_stats, ensure_ascii=False, indent=2)}\n\n"
            f"validity_constraints:\n{json.dumps(validity_constraints, ensure_ascii=False, indent=2)}\n\n"
            f"generation_batches:\n{json.dumps(safe_generation_batches, ensure_ascii=False, indent=2)}"
        )
        if self._v7_residual_protocol_enabled():
            system_prompt = (
                system_prompt.replace("role prompts", "solver instructions")
                .replace("prompt-role previews", "prompt summaries")
                .replace("role previews", "prompt summaries")
                .replace("role execution", "mechanism execution")
            )
            user_prompt = (
                user_prompt.replace("role_name", "mechanism_name")
                .replace("target_role_spec", "target_prompt_state")
                .replace("peer_role_specs", "peer_prompt_summaries")
                .replace("complete short role prompt", "complete short solver instruction")
                .replace("short role prompt", "short solver instruction")
                .replace("peer roles", "peer prompts")
                .replace("role prompts", "solver instructions")
                .replace("prompt-role previews", "prompt summaries")
            )
        text = await self._chat(
            model=self.cfg.optimizer_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(self.cfg.optimizer_temperature),
            max_tokens=int(self.cfg.optimizer_max_tokens),
            stage=f"optimizer_agent_{agent_id}",
        )
        diagnostics = self._empty_optimizer_generation_diagnostics()
        diagnostics["optimizer_raw_response_empty"] = int(not str(text or "").strip())
        obj = extract_json_obj(text)
        diagnostics["optimizer_json_parse_failed"] = int(bool(str(text or "").strip()) and obj is None)
        if obj is None:
            obj = {}
        candidates = obj.get("candidates", []) if isinstance(obj, dict) else []
        if isinstance(candidates, list):
            diagnostics["optimizer_raw_candidate_count"] = len(candidates)
        else:
            diagnostics["optimizer_schema_filtered_count"] += 1
            candidates = []
        parsed: List[Dict[str, Any]] = []
        seen_signatures: set = set()
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    diagnostics["optimizer_schema_filtered_count"] += 1
                    continue
                if not self._candidate_has_required_optimizer_fields(item, architecture="one_shot"):
                    diagnostics["optimizer_empty_prompt_count"] += 1
                    diagnostics["optimizer_schema_filtered_count"] += 1
                    continue
                prompt = str(item.get("candidate_prompt", "")).strip()
                prompt, sanitized = self._sanitize_prompt(prompt, agent_id)
                diagnostics["optimizer_sanitized_count"] += int(bool(sanitized))
                if self._is_redundant_candidate_prompt(parent_prompt, prompt, seen_signatures):
                    diagnostics["optimizer_redundant_filtered_count"] += 1
                    continue
                if not prompt:
                    diagnostics["optimizer_empty_prompt_count"] += 1
                    continue
                seen_signatures.add(self._prompt_signature(prompt))
                parsed.append(
                    {
                        "candidate_prompt": prompt,
                        "role_name": str(item.get("role_name", "")),
                        "mechanism_name": str(item.get("mechanism_name", item.get("role_name", ""))),
                        "decision_procedure": item.get("decision_procedure", []),
                        "when_to_use": str(item.get("when_to_use", "")),
                        "fallback_strategy": str(item.get("fallback_strategy", "")),
                        "anti_overlap_rule": str(item.get("anti_overlap_rule", "")),
                        "validity_checks": item.get("validity_checks", []),
                        "target_error_pattern": str(item.get("target_error_pattern", "")),
                        "accuracy_repair_rule": str(item.get("accuracy_repair_rule", "")),
                        "expected_accuracy_effect": str(item.get("expected_accuracy_effect", "")),
                        "rationale": str(item.get("rationale", "")),
                        "candidate_source": "optimizer",
                        "optimizer_generation_diagnostics": dict(diagnostics),
                        "generation_batch_type": str(item.get("source_batch_type", "")) or str(safe_generation_batches[min(len(parsed), len(safe_generation_batches) - 1)].get("batch_type", "")),
                        "generation_case_ids": [
                            str(c.get("case_id", ""))
                            for c in generation_batches[min(len(parsed), len(generation_batches) - 1)].get("cases", [])
                            if isinstance(c, dict)
                        ],
                    }
                )
                if len(parsed) >= num_candidates:
                    break
        fallback_mode_cfg = str(getattr(self.cfg, "optimizer_fallback_mode", "none") or "none").lower()
        if fallback_mode_cfg == "template":
            while len(parsed) < num_candidates:
                batch_idx = min(len(parsed), len(generation_batches) - 1)
                fallback_mode = "accuracy_repair" if has_target_error_batches else "diversity"
                fallback = self._structured_fallback_role(agent_id, len(parsed), mode=fallback_mode)
                prompt = str(fallback["candidate_prompt"])
                seen_signatures.add(self._prompt_signature(prompt))
                parsed.append(
                    {
                        "candidate_prompt": prompt,
                        "role_name": str(fallback.get("role_name", fallback.get("mechanism_name", ""))),
                        "mechanism_name": str(fallback.get("mechanism_name", fallback.get("role_name", ""))),
                        "decision_procedure": list(fallback["decision_procedure"]),
                        "when_to_use": str(fallback["when_to_use"]),
                        "fallback_strategy": str(fallback["fallback_strategy"]),
                        "anti_overlap_rule": str(fallback["anti_overlap_rule"]),
                        "validity_checks": list(fallback["validity_checks"]),
                        "target_error_pattern": str(fallback.get("target_error_pattern", "")),
                        "accuracy_repair_rule": str(fallback.get("accuracy_repair_rule", "")),
                        "expected_accuracy_effect": str(fallback.get("expected_accuracy_effect", "")),
                        "rationale": "Fallback candidate when optimizer returns too few usable prompts.",
                        "candidate_source": f"{fallback_mode}_fallback",
                        "optimizer_generation_diagnostics": dict(diagnostics),
                        "generation_batch_type": str(generation_batches[batch_idx].get("batch_type", "")),
                        "generation_case_ids": [
                            str(c.get("case_id", ""))
                            for c in generation_batches[batch_idx].get("cases", [])
                            if isinstance(c, dict)
                        ],
                    }
                )
        diagnostics["optimizer_final_candidate_count"] = sum(1 for item in parsed[:num_candidates] if str(item.get("candidate_source", "")) == "optimizer")
        diagnostics["optimizer_underfilled"] = bool(diagnostics["optimizer_final_candidate_count"] < int(num_candidates))
        diagnostics = self._record_optimizer_generation_diagnostics(agent_id, parent_prompt, diagnostics)
        for item in parsed:
            item["optimizer_generation_diagnostics"] = dict(diagnostics)
        return parsed[:num_candidates]

    async def evaluate_joint_trace_diversity(self, traces: List[str], candidate_agent_id: int) -> Dict[str, Any]:
        cache_key = self._hash("|".join([str(candidate_agent_id), *[self._hash(t) for t in traces]]))
        if cache_key in self.joint_diversity_cache:
            return dict(self.joint_diversity_cache[cache_key])
        system_prompt = (
            "You evaluate semantic diversity among a team's solver traces for diagnosis only. Return strict JSON only."
        )
        trace_payload = [
            {"agent_id": i, "trace": normalize_spaces(t)[:1800]}
            for i, t in enumerate(traces)
        ]
        user_prompt = (
            "Judge whether the candidate agent contributes a distinct reasoning behavior relative to the team.\n"
            "Do not use gold answers. Do not reward nonsense; invalid, vacuous, or copied traces are not diverse. "
            "This judgment is diagnostic and must not assume it controls prompt adoption.\n"
            "Return JSON:\n"
            "{\n"
            '  "joint_trace_diversity": 0.0,\n'
            '  "semantic_overlap_score": 0.0,\n'
            '  "candidate_agent_contribution": "distinct / redundant / harmful",\n'
            '  "redundant_agent_pairs": [[0, 1]],\n'
            '  "reason": "..."\n'
            "}\n\n"
            f"candidate_agent_id: {candidate_agent_id}\n"
            f"traces:\n{json.dumps(trace_payload, ensure_ascii=False, indent=2)}"
        )
        try:
            text = await self._chat(
                model=self.cfg.evaluator_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=float(self.cfg.evaluator_temperature),
                max_tokens=int(self.cfg.evaluator_max_tokens),
                stage=f"joint_trace_diversity_agent_{candidate_agent_id}",
            )
            obj = extract_json_obj(text) or {}
            if not isinstance(obj, dict):
                obj = {}
        except Exception as e:
            obj = {"joint_trace_diversity": 0.0, "semantic_overlap_score": 1.0, "candidate_agent_contribution": "harmful", "redundant_agent_pairs": [], "reason": normalize_spaces(str(e))[:240]}
        result = {
            "joint_trace_diversity": self._clip01(obj.get("joint_trace_diversity", 0.0)),
            "semantic_overlap_score": self._clip01(obj.get("semantic_overlap_score", 1.0)),
            "candidate_agent_contribution": str(obj.get("candidate_agent_contribution", "")),
            "redundant_agent_pairs": obj.get("redundant_agent_pairs", []) if isinstance(obj.get("redundant_agent_pairs", []), list) else [],
            "reason": str(obj.get("reason", "")),
        }
        self.joint_diversity_cache[cache_key] = dict(result)
        return result

    def _clip01(self, value: Any) -> float:
        try:
            x = float(value)
        except Exception:
            x = 0.0
        if np.isnan(x):
            return 0.0
        return float(max(0.0, min(1.0, x)))

    def _nonnegative(self, value: Any) -> float:
        try:
            x = float(value)
        except Exception:
            x = 0.0
        if np.isnan(x):
            return 0.0
        return float(max(0.0, x))

    def _reward_phase_state(self) -> Dict[str, float]:
        agents = list(getattr(self, "agents", []) or [])
        accepted_updates = sum(int(getattr(agent, "accept_count", 0) or 0) for agent in agents)
        attempted_updates = sum(
            int(getattr(agent, "accept_count", 0) or 0) + int(getattr(agent, "reject_count", 0) or 0)
            for agent in agents
        )
        prompt_hashes = [self._hash(getattr(agent, "current_prompt", "")) for agent in agents]
        unique_prompt_ratio = float(len(set(prompt_hashes)) / max(1, len(prompt_hashes)))
        update_progress = min(
            1.0,
            float(accepted_updates) / max(1, int(getattr(self.cfg, "reward_diversity_warmup_updates", 10) or 10)),
        )
        phase_progress = update_progress if self._v7_residual_protocol_enabled() else max(unique_prompt_ratio, update_progress)
        diversity_need = 1.0 - phase_progress
        return {
            "accepted_updates": float(accepted_updates),
            "attempted_updates": float(attempted_updates),
            "unique_prompt_ratio": float(unique_prompt_ratio),
            "update_progress": float(update_progress),
            "phase_progress": float(phase_progress),
            "diversity_need": float(diversity_need),
        }

    def _effective_reward_weights(self) -> Dict[str, float]:
        if str(getattr(self.cfg, "reward_schedule_mode", "static") or "static").lower() == "static":
            state = self._reward_phase_state()
            return {
                "target_accuracy": 1.0,
                "div_delta": self._nonnegative(getattr(self.cfg, "reward_weight_div_delta", 0.0)),
                "vote_delta": self._nonnegative(getattr(self.cfg, "reward_weight_vote_delta", 0.0)),
                "vote_margin": self._nonnegative(getattr(self.cfg, "reward_weight_vote_margin", 0.0)),
                "boundary_diversity": self._nonnegative(getattr(self.cfg, "reward_weight_boundary_diversity", 0.0)),
                "coverage": self._nonnegative(getattr(self.cfg, "reward_weight_coverage", 0.3)),
                "useful_diversity": self._nonnegative(getattr(self.cfg, "reward_weight_useful_diversity", 0.2)),
                "invalid_delta": self._nonnegative(getattr(self.cfg, "reward_weight_invalid_delta", 0.0)),
                "accuracy_guard_epsilon": self._nonnegative(getattr(self.cfg, "accuracy_guard_epsilon", 0.0)),
                **state,
            }

        state = self._reward_phase_state()
        need = float(state["diversity_need"])
        progress = float(state["phase_progress"])
        target_weight = (
            float(getattr(self.cfg, "reward_weight_target_accuracy_late", 1.0)) * progress
            + float(getattr(self.cfg, "reward_weight_target_accuracy_early", 0.9)) * need
        )
        div_weight = (
            float(getattr(self.cfg, "reward_weight_div_delta_late", 0.2)) * progress
            + float(getattr(self.cfg, "reward_weight_div_delta_early", 0.8)) * need
        )
        vote_delta_weight = (
            float(getattr(self.cfg, "reward_weight_vote_delta_late", 0.3)) * progress
            + float(getattr(self.cfg, "reward_weight_vote_delta_early", 0.4)) * need
        )
        vote_margin_weight = (
            float(getattr(self.cfg, "reward_weight_vote_margin_late", 0.25)) * progress
            + float(getattr(self.cfg, "reward_weight_vote_margin_early", 0.5)) * need
        )
        boundary_diversity_weight = (
            float(getattr(self.cfg, "reward_weight_boundary_diversity_late", 0.2)) * progress
            + float(getattr(self.cfg, "reward_weight_boundary_diversity_early", 0.3)) * need
        )
        guard_epsilon = (
            float(getattr(self.cfg, "accuracy_guard_epsilon_late", 0.01)) * progress
            + float(getattr(self.cfg, "accuracy_guard_epsilon_early", 0.03)) * need
        )
        coverage_weight = float(getattr(self.cfg, "reward_weight_coverage_late", 0.3)) * progress + float(getattr(self.cfg, "reward_weight_coverage_early", 0.4)) * need
        useful_weight = float(getattr(self.cfg, "reward_weight_useful_diversity_late", 0.25)) * progress + float(getattr(self.cfg, "reward_weight_useful_diversity_early", 0.5)) * need
        return {
            "target_accuracy": self._nonnegative(target_weight),
            "div_delta": self._nonnegative(div_weight),
            "vote_delta": self._nonnegative(vote_delta_weight),
            "vote_margin": self._nonnegative(vote_margin_weight),
            "boundary_diversity": self._nonnegative(boundary_diversity_weight),
            "coverage": self._nonnegative(coverage_weight),
            "useful_diversity": self._nonnegative(useful_weight),
            "invalid_delta": self._nonnegative(getattr(self.cfg, "reward_weight_invalid_delta", 0.0)),
            "accuracy_guard_epsilon": self._nonnegative(guard_epsilon),
            **state,
        }

    def _effective_reward_log_fields(self, weights: Dict[str, float]) -> Dict[str, float]:
        return {
            "effective_weight_target_accuracy": float(weights.get("target_accuracy", 0.0)),
            "effective_weight_div_delta": float(weights.get("div_delta", 0.0)),
            "effective_weight_vote_delta": float(weights.get("vote_delta", 0.0)),
            "effective_weight_vote_margin": float(weights.get("vote_margin", 0.0)),
            "effective_weight_boundary_diversity": float(weights.get("boundary_diversity", 0.0)),
            "effective_weight_coverage": float(weights.get("coverage", 0.0)),
            "effective_weight_useful_diversity": float(weights.get("useful_diversity", 0.0)),
            "effective_accuracy_guard_epsilon": float(weights.get("accuracy_guard_epsilon", 0.0)),
            "reward_phase_progress": float(weights.get("phase_progress", 0.0)),
            "reward_diversity_need": float(weights.get("diversity_need", 0.0)),
            "reward_unique_prompt_ratio": float(weights.get("unique_prompt_ratio", 0.0)),
            "reward_accepted_updates": float(weights.get("accepted_updates", 0.0)),
        }

    def _candidate_eval_audit_fields(self, eval_batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        question_hashes = {
            hashlib.sha256(normalize_spaces(str(example.get("question", ""))).lower().encode("utf-8")).hexdigest()
            for example in eval_batch
            if isinstance(example, Mapping)
        }
        return {
            "candidate_eval_data_source": str(getattr(self.cfg, "candidate_eval_data_source", "optimization_train")),
            "candidate_eval_total_count": len(eval_batch),
            "candidate_eval_unique_question_count": len(question_hashes),
            "candidate_eval_repeat_count": int(getattr(self.cfg, "candidate_eval_repeats", 1) or 1),
        }

    def _candidate_reward_guarded(
        self,
        baseline_team_accuracy: float,
        candidate_team_accuracy: float,
        baseline_target_accuracy: float,
        candidate_target_accuracy: float,
        baseline_embedding_diversity: float,
        candidate_embedding_diversity: float,
        baseline_invalid_rate: float,
        candidate_invalid_rate: float,
    ) -> Dict[str, Any]:
        baseline_team_accuracy = self._clip01(baseline_team_accuracy)
        candidate_team_accuracy = self._clip01(candidate_team_accuracy)
        baseline_target_accuracy = self._clip01(baseline_target_accuracy)
        candidate_target_accuracy = self._clip01(candidate_target_accuracy)
        baseline_embedding_diversity = self._clip01(baseline_embedding_diversity)
        candidate_embedding_diversity = self._clip01(candidate_embedding_diversity)
        baseline_invalid_rate = self._clip01(baseline_invalid_rate)
        candidate_invalid_rate = self._clip01(candidate_invalid_rate)

        deltas = compute_candidate_metric_deltas(
            baseline_target_accuracy=baseline_target_accuracy,
            candidate_target_accuracy=candidate_target_accuracy,
            baseline_team_accuracy=baseline_team_accuracy,
            candidate_team_accuracy=candidate_team_accuracy,
            baseline_oracle_accuracy=0.0,
            candidate_oracle_accuracy=0.0,
            baseline_embedding_diversity=baseline_embedding_diversity,
            candidate_embedding_diversity=candidate_embedding_diversity,
            baseline_invalid_rate=baseline_invalid_rate,
            candidate_invalid_rate=candidate_invalid_rate,
        )
        acc_delta = deltas["accuracy_delta"]
        vote_delta = deltas["vote_delta"]
        div_delta = deltas["diversity_delta"]
        invalid_delta = deltas["invalid_delta"]
        weights = self._effective_reward_weights()
        guard_passed = candidate_target_accuracy >= baseline_target_accuracy - float(weights["accuracy_guard_epsilon"])
        if not guard_passed:
            reward = -1.0 + acc_delta - float(weights["invalid_delta"]) * max(0.0, invalid_delta)
        else:
            reward = (
                float(weights["target_accuracy"]) * candidate_target_accuracy
                + float(weights["div_delta"]) * div_delta
                - float(weights["invalid_delta"]) * max(0.0, invalid_delta)
            )
        result = {
            "reward": float(reward),
            "reward_total": float(reward),
            **deltas,
            "accuracy_guard_passed": bool(guard_passed),
            "baseline_target_accuracy": float(baseline_target_accuracy),
            "candidate_target_accuracy": float(candidate_target_accuracy),
            "target_agent_accuracy": float(candidate_target_accuracy),
        }
        result.update(self._effective_reward_log_fields(weights))
        result.update(
            {
                "baseline_team_accuracy": float(baseline_team_accuracy),
                "candidate_team_accuracy": float(candidate_team_accuracy),
                "baseline_invalid_rate": float(baseline_invalid_rate),
                "candidate_invalid_rate": float(candidate_invalid_rate),
                "baseline_embedding_diversity": float(baseline_embedding_diversity),
                "candidate_embedding_diversity": float(candidate_embedding_diversity),
            }
        )
        return result

    def _candidate_reward_vote_useful_diversity(
        self,
        *,
        baseline_team_accuracy: float,
        candidate_team_accuracy: float,
        baseline_target_accuracy: float,
        candidate_target_accuracy: float,
        baseline_invalid_rate: float,
        candidate_invalid_rate: float,
        baseline_mean_vote_margin: float,
        candidate_mean_vote_margin: float,
        baseline_boundary_useful_diversity: float,
        candidate_boundary_useful_diversity: float,
        baseline_oracle_accuracy: Optional[float] = None,
        candidate_oracle_accuracy: Optional[float] = None,
        baseline_embedding_diversity: float = 0.0,
        candidate_embedding_diversity: float = 0.0,
    ) -> Dict[str, Any]:
        baseline_team_accuracy = self._clip01(baseline_team_accuracy)
        candidate_team_accuracy = self._clip01(candidate_team_accuracy)
        baseline_target_accuracy = self._clip01(baseline_target_accuracy)
        candidate_target_accuracy = self._clip01(candidate_target_accuracy)
        baseline_invalid_rate = self._clip01(baseline_invalid_rate)
        candidate_invalid_rate = self._clip01(candidate_invalid_rate)
        baseline_mean_vote_margin = float(np.clip(baseline_mean_vote_margin, -1.0, 1.0))
        candidate_mean_vote_margin = float(np.clip(candidate_mean_vote_margin, -1.0, 1.0))
        baseline_boundary_useful_diversity = self._clip01(baseline_boundary_useful_diversity)
        candidate_boundary_useful_diversity = self._clip01(candidate_boundary_useful_diversity)

        deltas = compute_candidate_metric_deltas(
            baseline_target_accuracy=baseline_target_accuracy,
            candidate_target_accuracy=candidate_target_accuracy,
            baseline_team_accuracy=baseline_team_accuracy,
            candidate_team_accuracy=candidate_team_accuracy,
            baseline_oracle_accuracy=float(baseline_oracle_accuracy or 0.0),
            candidate_oracle_accuracy=float(candidate_oracle_accuracy or 0.0),
            baseline_embedding_diversity=baseline_embedding_diversity,
            candidate_embedding_diversity=candidate_embedding_diversity,
            baseline_invalid_rate=baseline_invalid_rate,
            candidate_invalid_rate=candidate_invalid_rate,
        )
        vote_delta = deltas["vote_delta"]
        invalid_delta = deltas["invalid_delta"]
        vote_margin_delta = candidate_mean_vote_margin - baseline_mean_vote_margin
        boundary_diversity_delta = candidate_boundary_useful_diversity - baseline_boundary_useful_diversity
        # Boundary diversity is an auxiliary signal only while the team remains
        # near a gold-vs-wrong vote boundary. Leaving that boundary through a
        # stronger correct vote must not turn into a diversity penalty.
        boundary_diversity_gain = max(0.0, boundary_diversity_delta)
        weights = self._effective_reward_weights()
        target_guard_passed = candidate_target_accuracy >= baseline_target_accuracy - float(weights["accuracy_guard_epsilon"])
        invalid_guard_passed = candidate_invalid_rate <= baseline_invalid_rate + float(self.cfg.invalid_guard_epsilon)
        reward_components = {
            "reward_component_target_accuracy": 0.0,
            "reward_component_vote_delta": 0.0,
            "reward_component_vote_margin": 0.0,
            "reward_component_boundary_diversity": 0.0,
            "reward_component_invalid_penalty": 0.0,
            "reward_component_guard_penalty": 0.0,
        }
        if not target_guard_passed or not invalid_guard_passed:
            reward = -1.0
            reward_components["reward_component_guard_penalty"] = -1.0
        else:
            reward_components.update(
                {
                    "reward_component_target_accuracy": float(weights["target_accuracy"]) * candidate_target_accuracy,
                    "reward_component_vote_delta": float(weights["vote_delta"]) * vote_delta,
                    "reward_component_vote_margin": float(weights["vote_margin"]) * vote_margin_delta,
                    "reward_component_boundary_diversity": float(weights["boundary_diversity"]) * boundary_diversity_gain,
                    "reward_component_invalid_penalty": -float(weights["invalid_delta"]) * max(0.0, invalid_delta),
                }
            )
            reward = sum(reward_components.values())
        result = {
            "reward": float(reward),
            "reward_total": float(reward),
            "coverage_delta": float(deltas["coverage_delta"]),
            **deltas,
            "baseline_mean_vote_margin": baseline_mean_vote_margin,
            "candidate_mean_vote_margin": candidate_mean_vote_margin,
            "vote_margin_delta": vote_margin_delta,
            "baseline_boundary_useful_diversity": baseline_boundary_useful_diversity,
            "candidate_boundary_useful_diversity": candidate_boundary_useful_diversity,
            "boundary_useful_diversity_delta": boundary_diversity_delta,
            "boundary_diversity_gain": boundary_diversity_gain,
            "baseline_team_accuracy": float(baseline_team_accuracy),
            "candidate_team_accuracy": float(candidate_team_accuracy),
            "baseline_target_accuracy": float(baseline_target_accuracy),
            "candidate_target_accuracy": float(candidate_target_accuracy),
            "target_agent_accuracy": float(candidate_target_accuracy),
            "baseline_invalid_rate": float(baseline_invalid_rate),
            "candidate_invalid_rate": float(candidate_invalid_rate),
            "accuracy_guard_passed": bool(target_guard_passed),
            "invalid_guard_passed": bool(invalid_guard_passed),
            **reward_components,
        }
        result.update(self._effective_reward_log_fields(weights))
        return result

    def _candidate_reward_competence_depth(self, metrics: Dict[str, Any], v7_reward: float) -> Dict[str, Any]:
        accuracy_delta = float(metrics.get("accuracy_delta", 0.0) or 0.0)
        competence_component = (
            float(getattr(self.cfg, "competence_weight_accuracy_gain", 1.0)) * max(0.0, accuracy_delta)
            - float(getattr(self.cfg, "competence_weight_accuracy_loss", 1.5)) * max(0.0, -accuracy_delta)
            + float(getattr(self.cfg, "competence_weight_depth2_gain", 0.8)) * float(metrics.get("depth2_gain_rate", 0.0) or 0.0)
            - float(getattr(self.cfg, "competence_weight_depth2_loss", 1.0)) * float(metrics.get("depth2_loss_rate", 0.0) or 0.0)
            + float(getattr(self.cfg, "competence_weight_vote_gain_early", 0.4)) * float(metrics.get("vote_gain_rate", 0.0) or 0.0)
            - float(getattr(self.cfg, "competence_weight_vote_loss_early", 1.0)) * float(metrics.get("vote_loss_rate", 0.0) or 0.0)
        )
        if self._is_v82_hybrid():
            competence_component += (
                float(getattr(self.cfg, "competence_weight_depth1_gain", 0.8)) * float(metrics.get("depth1_gain_rate", 0.0) or 0.0)
                - float(getattr(self.cfg, "competence_weight_depth1_loss", 1.2)) * float(metrics.get("depth1_loss_rate", 0.0) or 0.0)
            )
        strength = float(self.specialization_strength)
        competence_mix = max(float(getattr(self.cfg, "competence_residual_floor", 0.30)), 1.0 - strength) if self._is_v82_hybrid() else 1.0 - strength
        specialization_mix = 1.0 - competence_mix if self._is_v82_hybrid() else strength
        reward = competence_mix * competence_component + specialization_mix * float(v7_reward)
        depth2_component = float(metrics.get("depth2_net_delta", 0.0) or 0.0) if bool(
            getattr(self.cfg, "competence_depth2_aux_enabled", False)
        ) else 0.0
        boundary_component = float(metrics.get(
            "plurality_boundary_shared_error_net_gain",
            metrics.get("boundary_shared_error_net_gain", 0.0),
        ) or 0.0)
        return {
            "reward": float(reward),
            "reward_total": float(reward),
            "final_reward": float(reward),
            "competence_reward_component": float(competence_component),
            "v7_reward_component": float(v7_reward),
            "effective_reward_specialization_strength": strength,
            "competence_mix": competence_mix,
            "specialization_mix": specialization_mix,
            "stage_aux_depth2_component": competence_mix * depth2_component,
            "stage_aux_boundary_component": specialization_mix * boundary_component,
            "stage_aux_objective": competence_mix * (
                0.5 * float(metrics.get("depth1_net_delta", 0.0) or 0.0) + 0.5 * depth2_component
            ) + specialization_mix * boundary_component if self._is_v82_hybrid() else (1.0 - strength) * depth2_component + strength * boundary_component,
        }

    def _candidate_reward_coverage_useful_diversity(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        weights = self._effective_reward_weights()
        invalid_passed = float(metrics.get("candidate_invalid_rate", 1.0)) <= float(metrics.get("baseline_invalid_rate", 1.0)) + float(self.cfg.invalid_guard_epsilon)
        reward = -1.0 if not invalid_passed else (
            float(weights["target_accuracy"]) * float(metrics.get("candidate_target_accuracy", 0.0))
            + float(weights["coverage"]) * float(metrics.get("coverage_delta", 0.0))
            + float(weights["useful_diversity"]) * float(metrics.get("useful_diversity", 0.0))
        )
        return {"reward": reward, "reward_total": reward, "invalid_guard_passed": invalid_passed, **self._effective_reward_log_fields(weights)}

    async def _evaluate_candidate_prompt_accuracy_only(
        self,
        agent_id: int,
        candidate_prompt: str,
        peer_prompts: Optional[List[str]],
        eval_batch: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        peer_prompts = list(peer_prompts or self._active_prompt_list())

        async def run_one(ex: Dict[str, str]) -> Dict[str, Any]:
            q = ex["question"]
            gold = self.task_spec.parse_gold(ex["answer"], q)
            baseline_prompts = list(peer_prompts)
            while len(baseline_prompts) < len(self.agents):
                baseline_prompts.append(self.agents[len(baseline_prompts)].current_prompt)
            eval_prompts = list(baseline_prompts)
            eval_prompts[agent_id] = candidate_prompt
            question_hash = self._hash(q)
            baseline_traces, baseline_answers, baseline_reuse_stats = await self.solve_with_prompts_reusing_records(
                q,
                baseline_prompts,
                source=f"candidate_accuracy_baseline_agent_{agent_id}",
            )
            baseline_rollout = self.compute_rollout_metrics(
                baseline_traces,
                baseline_answers,
                gold,
                prompts=baseline_prompts,
                question_hash=question_hash,
            )
            traces, answers, reuse_stats = await self.solve_with_prompts_reusing_records(
                q,
                eval_prompts,
                source=f"candidate_accuracy_agent_{agent_id}",
            )
            rollout = self.compute_rollout_metrics(
                traces,
                answers,
                gold,
                prompts=eval_prompts,
                question_hash=question_hash,
            )
            baseline_target_answer = baseline_answers[agent_id] if agent_id < len(baseline_answers) else ""
            target_answer = answers[agent_id] if agent_id < len(answers) else ""
            return {
                "baseline_team_accuracy": int(baseline_rollout.get("vote_correct", 0)),
                "team_accuracy": int(rollout.get("vote_correct", 0)),
                "baseline_target_accuracy": int(self.task_spec.match_answer(baseline_target_answer, gold)),
                "target_agent_accuracy": int(self.task_spec.match_answer(target_answer, gold)),
                "baseline_any_correct": int(baseline_rollout.get("any_correct", 0)),
                "candidate_any_correct": int(rollout.get("any_correct", 0)),
                "baseline_individual_correct": list(baseline_rollout.get("individual_correct", [])),
                "candidate_individual_correct": list(rollout.get("individual_correct", [])),
                "baseline_mean_vote_margin": float(baseline_rollout.get("normalized_vote_margin", -1.0)),
                "candidate_mean_vote_margin": float(rollout.get("normalized_vote_margin", -1.0)),
                "baseline_boundary_useful_diversity": float(baseline_rollout.get("boundary_useful_diversity", 0.0)),
                "candidate_boundary_useful_diversity": float(rollout.get("boundary_useful_diversity", 0.0)),
                "vote_answer": str(rollout.get("vote_answer", "")),
                "vote_tie": bool(rollout.get("vote_tie", False)),
                "tie_candidates": list(rollout.get("tie_candidates", [])),
                "vote_counts": dict(rollout.get("vote_counts", {})),
                "tie_break_method": str(rollout.get("tie_break_method", "")),
                "majority_vote_answer": str(rollout.get("majority_vote_answer", "")),
                "weighted_vote_answer": str(rollout.get("weighted_vote_answer", "")),
                "majority_vote_correct": int(rollout.get("majority_vote_correct", 0)),
                "weighted_vote_correct": int(rollout.get("weighted_vote_correct", 0)),
                "aggregation_mode": str(rollout.get("aggregation_mode", "majority")),
                "target_answer": target_answer,
                "target_trace_hash": self._hash(traces[agent_id]) if agent_id < len(traces) else "",
                "baseline_solver_reuse_hits": int(baseline_reuse_stats.get("solver_reuse_hits", 0) or 0),
                "baseline_solver_reuse_misses": int(baseline_reuse_stats.get("solver_reuse_misses", 0) or 0),
                "baseline_solver_calls": int(baseline_reuse_stats.get("solver_calls", 0) or 0),
                "baseline_solver_reuse_total": int(baseline_reuse_stats.get("solver_reuse_total", 0) or 0),
                **reuse_stats,
            }

        raw = await asyncio.gather(*[run_one(ex) for ex in eval_batch], return_exceptions=True)
        rows = [r for r in raw if isinstance(r, dict)]
        errors = [normalize_spaces(str(r))[:240] for r in raw if isinstance(r, Exception)]
        baseline_team_accuracy = self._clip01(float(np.mean([float(r.get("baseline_team_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        team_accuracy = self._clip01(float(np.mean([float(r.get("team_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        baseline_target_accuracy = self._clip01(float(np.mean([float(r.get("baseline_target_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        target_agent_accuracy = self._clip01(float(np.mean([float(r.get("target_agent_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        baseline_oracle_acc = self._clip01(float(np.mean([float(r.get("baseline_any_correct", 0.0)) for r in rows])) if rows else 0.0)
        candidate_oracle_acc = self._clip01(float(np.mean([float(r.get("candidate_any_correct", 0.0)) for r in rows])) if rows else 0.0)
        baseline_mean_vote_margin = float(np.mean([float(r.get("baseline_mean_vote_margin", -1.0)) for r in rows])) if rows else -1.0
        candidate_mean_vote_margin = float(np.mean([float(r.get("candidate_mean_vote_margin", -1.0)) for r in rows])) if rows else -1.0
        baseline_boundary = self._clip01(float(np.mean([float(r.get("baseline_boundary_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
        candidate_boundary = self._clip01(float(np.mean([float(r.get("candidate_boundary_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
        vote_transitions = compute_vote_transitions(
            [bool(row.get("baseline_team_accuracy", 0)) for row in rows],
            [bool(row.get("team_accuracy", 0)) for row in rows],
        )
        coverage_transitions = compute_oracle_coverage_transitions(
            [list(row.get("baseline_individual_correct", [])) for row in rows],
            [list(row.get("candidate_individual_correct", [])) for row in rows],
        )
        deltas = compute_candidate_metric_deltas(
            baseline_target_accuracy=baseline_target_accuracy,
            candidate_target_accuracy=target_agent_accuracy,
            baseline_team_accuracy=baseline_team_accuracy,
            candidate_team_accuracy=team_accuracy,
            baseline_oracle_accuracy=baseline_oracle_acc,
            candidate_oracle_accuracy=candidate_oracle_acc,
            baseline_embedding_diversity=0.0,
            candidate_embedding_diversity=0.0,
            baseline_invalid_rate=0.0,
            candidate_invalid_rate=0.0,
        )
        if abs(float(vote_transitions["net_vote_delta"]) - float(deltas["vote_delta"])) > PARETO_EPSILON:
            raise RuntimeError("Accuracy-only vote transition delta is inconsistent")
        if abs(float(coverage_transitions["net_coverage_delta"]) - float(deltas["coverage_delta"])) > PARETO_EPSILON:
            raise RuntimeError("Accuracy-only coverage transition delta is inconsistent")
        boundary_delta = candidate_boundary - baseline_boundary
        solver_reuse_hits = int(sum(int(r.get("solver_reuse_hits", 0) or 0) for r in rows))
        solver_reuse_misses = int(sum(int(r.get("solver_reuse_misses", 0) or 0) for r in rows))
        solver_calls = int(sum(int(r.get("solver_calls", 0) or 0) for r in rows))
        solver_reuse_total = int(sum(int(r.get("solver_reuse_total", 0) or 0) for r in rows))
        majority_team_accuracy = self._clip01(float(np.mean([float(r.get("majority_vote_correct", 0.0)) for r in rows])) if rows else 0.0)
        weighted_team_accuracy = self._clip01(float(np.mean([float(r.get("weighted_vote_correct", 0.0)) for r in rows])) if rows else 0.0)
        return {
            "reward": target_agent_accuracy,
            "reward_total": target_agent_accuracy,
            "reward_component_target_accuracy": target_agent_accuracy,
            "reward_component_vote_delta": 0.0,
            "reward_component_vote_margin": 0.0,
            "reward_component_boundary_diversity": 0.0,
            "reward_component_invalid_penalty": 0.0,
            "reward_component_guard_penalty": 0.0,
            "embedding_diversity": 0.0,
            "mean_embedding_overlap": 0.0,
            "target_overlap_pressure": 0.0,
            "homogeneous_case_count": 0.0,
            "resolved_case_count": 0.0,
            "new_homogeneous_case_count": 0.0,
            "team_accuracy": team_accuracy,
            "baseline_team_accuracy": baseline_team_accuracy,
            "candidate_team_accuracy": team_accuracy,
            "majority_team_accuracy": majority_team_accuracy,
            "weighted_team_accuracy": weighted_team_accuracy,
            "aggregation_mode": str(getattr(self.cfg, "aggregation_mode", "majority") or "majority"),
            "target_agent_accuracy": target_agent_accuracy,
            "baseline_target_accuracy": baseline_target_accuracy,
            "candidate_target_accuracy": target_agent_accuracy,
            "baseline_oracle_acc": baseline_oracle_acc,
            "candidate_oracle_acc": candidate_oracle_acc,
            "baseline_mean_vote_margin": baseline_mean_vote_margin,
            "candidate_mean_vote_margin": candidate_mean_vote_margin,
            "vote_margin_delta": candidate_mean_vote_margin - baseline_mean_vote_margin,
            "baseline_boundary_useful_diversity": baseline_boundary,
            "candidate_boundary_useful_diversity": candidate_boundary,
            "boundary_useful_diversity_delta": boundary_delta,
            "boundary_diversity_gain": max(0.0, boundary_delta),
            "baseline_embedding_diversity": 0.0,
            "candidate_embedding_diversity": 0.0,
            "baseline_invalid_rate": 0.0,
            "candidate_invalid_rate": 0.0,
            "accuracy_guard_passed": True,
            "invalid_guard_passed": True,
            **deltas,
            **vote_transitions,
            **coverage_transitions,
            "invalid_rate": 0.0,
            "invalid_score": 1.0,
            "num_eval_samples": len(rows),
            "candidate_prompt": candidate_prompt,
            "errors": errors,
            "accuracy_only": True,
            "accuracy_only_reward_basis": "target_agent_accuracy",
            "solver_reuse_enabled": bool(self.cfg.candidate_reuse_recorded_rollouts),
            "solver_reuse_hits": solver_reuse_hits,
            "solver_reuse_misses": solver_reuse_misses,
            "solver_calls": solver_calls,
            "solver_reuse_total": solver_reuse_total,
            "solver_reuse_hit_rate": float(solver_reuse_hits / solver_reuse_total) if solver_reuse_total else 0.0,
            "baseline_solver_calls": int(sum(int(r.get("baseline_solver_calls", 0) or 0) for r in rows)),
            "baseline_solver_reuse_hits": int(sum(int(r.get("baseline_solver_reuse_hits", 0) or 0) for r in rows)),
            "baseline_solver_reuse_misses": int(sum(int(r.get("baseline_solver_reuse_misses", 0) or 0) for r in rows)),
            "baseline_solver_reuse_total": int(sum(int(r.get("baseline_solver_reuse_total", 0) or 0) for r in rows)),
            "candidate_eval_strategy": str(getattr(self.cfg, "candidate_eval_strategy", "random")),
            "candidate_eval_pool_size": int(getattr(self.cfg, "candidate_eval_pool_size", 0) or 0),
            "candidate_eval_pool_actual_size": int(getattr(self.cfg, "candidate_eval_pool_actual_size", 0) or 0),
            "candidate_eval_batch_size": int(getattr(self.cfg, "candidate_eval_batch_size", 0) or 0),
            "actual_eval_batch_size": len(eval_batch),
            "num_eval_repeats": int(getattr(self.cfg, "candidate_eval_repeats", 1) or 1),
            **self._candidate_eval_audit_fields(eval_batch),
        }

    async def evaluate_candidate_prompt(
        self,
        agent_id: int,
        candidate_prompt: str,
        peer_prompts: Optional[List[str]],
        eval_batch: List[Dict[str, str]],
        role_spec: Optional[Dict[str, Any]] = None,
        baseline_homogeneous_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if self._is_accuracy_only_mode():
            return await self._evaluate_candidate_prompt_accuracy_only(
                agent_id=agent_id,
                candidate_prompt=candidate_prompt,
                peer_prompts=peer_prompts,
                eval_batch=eval_batch,
            )
        role_spec = dict(role_spec or {})
        peer_prompts = list(peer_prompts or self._active_prompt_list())
        baseline_case_keys = self._target_case_keys(list(baseline_homogeneous_cases or []))

        async def run_one(ex: Dict[str, str]) -> Dict[str, Any]:
            q = ex["question"]
            sample_hash = self._hash(q)
            gold = self.task_spec.parse_gold(ex["answer"], q)
            baseline_prompts = list(peer_prompts)
            while len(baseline_prompts) < len(self.agents):
                baseline_prompts.append(self.agents[len(baseline_prompts)].current_prompt)
            eval_prompts = list(baseline_prompts)
            eval_prompts[agent_id] = candidate_prompt
            baseline_rollout = {}
            baseline_reuse_stats: Dict[str, Any] = {}
            if self._uses_baseline_candidate_metrics():
                baseline_traces, baseline_answers, baseline_reuse_stats = await self.solve_with_prompts_reusing_records(
                    q,
                    baseline_prompts,
                    source=f"candidate_baseline_agent_{agent_id}",
                )
                baseline_rollout = self.compute_rollout_metrics(
                    baseline_traces,
                    baseline_answers,
                    gold,
                    prompts=baseline_prompts,
                    question_hash=sample_hash,
                )
            traces, answers, reuse_stats = await self.solve_with_prompts_reusing_records(
                q,
                eval_prompts,
                source=f"candidate_eval_agent_{agent_id}",
            )
            rollout = self.compute_rollout_metrics(traces, answers, gold, prompts=eval_prompts, question_hash=sample_hash)
            agent_invalid = self.rule_invalid_check(traces[agent_id], answers[agent_id] if agent_id < len(answers) else "")
            diversity = float(rollout.get("embedding_diversity", 0.0))
            joint = {}
            if self.cfg.use_joint_trace_diversity_evaluator:
                joint = await self.evaluate_joint_trace_diversity(traces, agent_id)
            impact = self._homogeneity_impact_metrics(agent_id, rollout, baseline_case_keys, sample_hash)
            row = {
                "trace": traces[agent_id] if agent_id < len(traces) else "",
                "answer": answers[agent_id] if agent_id < len(answers) else "",
                "embedding_diversity": self._clip01(diversity),
                "team_accuracy": int(rollout.get("vote_correct", 0)),
                "invalid": float(agent_invalid.get("invalid", 1)),
                "invalid_reasons": agent_invalid.get("reasons", []),
                "mean_embedding_overlap": float(rollout.get("mean_embedding_overlap", 0.0)),
                **impact,
                "joint_trace_evaluation": joint,
                "trace_hash": self._hash(traces[agent_id]),
                **reuse_stats,
            }
            if self._uses_baseline_candidate_metrics():
                baseline_vote_correct = int(baseline_rollout.get("vote_correct", 0))
                candidate_vote_correct = int(rollout.get("vote_correct", 0))
                baseline_any_correct = self._rollout_any_correct(baseline_rollout)
                candidate_any_correct = self._rollout_any_correct(rollout)
                baseline_target_correct = self._safe_agent_correct(baseline_rollout, agent_id)
                target_agent_correct = self._safe_agent_correct(rollout, agent_id)
                target_trace_novelty = self._target_trace_novelty(traces, agent_id)
                target_useful_diversity = (
                    target_trace_novelty
                    * float(target_agent_correct)
                    * (1.0 - float(agent_invalid.get("invalid", 1)))
                )
                rescue = int((baseline_vote_correct == 0) and (target_agent_correct == 1))
                peer_wrong_count = sum(
                    int(not self.task_spec.match_answer(answer, gold))
                    for idx, answer in enumerate(baseline_answers)
                    if idx != agent_id
                )
                counterfactual_answers = list(baseline_answers)
                if agent_id < len(counterfactual_answers):
                    counterfactual_answers[agent_id] = gold
                counterfactual_vote = self._vote_with_diagnostics(
                    counterfactual_answers, question_hash=sample_hash
                )
                counterfactual_gold_vote_correct = bool(
                    self.task_spec.match_answer(str(counterfactual_vote.get("vote_answer", "")), gold)
                )
                counterfactual_gold_diagnostics = compute_gold_vote_diagnostics(
                    counterfactual_answers,
                    gold,
                    self.task_spec.match_answer,
                    len(self.agents),
                )

                def in_dominant_wrong_cluster(candidate_answers: Sequence[str], target_id: int) -> bool:
                    target_value = str(candidate_answers[target_id] if target_id < len(candidate_answers) else "").strip()
                    wrong_counts = Counter(
                        str(answer or "").strip()
                        for answer in candidate_answers
                        if str(answer or "").strip() and not self.task_spec.match_answer(str(answer), gold)
                    )
                    return bool(
                        target_value
                        and not self.task_spec.match_answer(target_value, gold)
                        and wrong_counts.get(target_value, 0) == max(wrong_counts.values(), default=0)
                        and wrong_counts.get(target_value, 0) > 1
                    )

                residual_info = self._infer_target_error_pattern(
                    target_trace=baseline_traces[agent_id] if agent_id < len(baseline_traces) else "",
                    target_answer=baseline_answers[agent_id] if agent_id < len(baseline_answers) else "",
                    peer_traces=[trace for idx, trace in enumerate(baseline_traces) if idx != agent_id],
                    rollout=baseline_rollout,
                    agent_id=agent_id,
                )
                row.update(
                    {
                        "question_hash": sample_hash,
                        "baseline_vote_correct": baseline_vote_correct,
                        "candidate_vote_correct": candidate_vote_correct,
                        "baseline_plurality_vote_correct": baseline_vote_correct,
                        "candidate_plurality_vote_correct": candidate_vote_correct,
                        "baseline_vote_tie": bool(baseline_rollout.get("vote_tie", False)),
                        "candidate_vote_tie": bool(rollout.get("vote_tie", False)),
                        "baseline_plurality_vote_tie": bool(baseline_rollout.get("plurality_vote_tie", False)),
                        "candidate_plurality_vote_tie": bool(rollout.get("plurality_vote_tie", False)),
                        "baseline_plurality_tie_candidates": list(baseline_rollout.get("plurality_tie_candidates", [])),
                        "candidate_plurality_tie_candidates": list(rollout.get("plurality_tie_candidates", [])),
                        "plurality_tie_break_method": str(baseline_rollout.get("plurality_tie_break_method", "")),
                        "plurality_tie_break_question_hash": sample_hash,
                        "baseline_any_correct": baseline_any_correct,
                        "candidate_any_correct": candidate_any_correct,
                        "baseline_individual_correct": [bool(value) for value in baseline_rollout.get("individual_correct", [])],
                        "candidate_individual_correct": [bool(value) for value in rollout.get("individual_correct", [])],
                        "baseline_target_correct": baseline_target_correct,
                        "candidate_target_correct": target_agent_correct,
                        "target_agent_correct": target_agent_correct,
                        "peer_wrong_count": int(peer_wrong_count),
                        "baseline_vote_margin": float(baseline_rollout.get("normalized_vote_margin", -1.0)),
                        "candidate_vote_margin": float(rollout.get("normalized_vote_margin", -1.0)),
                        "baseline_gold_vote_count": int(baseline_rollout.get("gold_vote_count", 0) or 0),
                        "candidate_gold_vote_count": int(rollout.get("gold_vote_count", 0) or 0),
                        "baseline_largest_wrong_vote_count": int(baseline_rollout.get("largest_wrong_vote_count", 0) or 0),
                        "candidate_largest_wrong_vote_count": int(rollout.get("largest_wrong_vote_count", 0) or 0),
                        "baseline_plurality_margin_votes": int(baseline_rollout.get("plurality_margin_votes", 0) or 0),
                        "candidate_plurality_margin_votes": int(rollout.get("plurality_margin_votes", 0) or 0),
                        "baseline_normalized_plurality_margin": float(baseline_rollout.get("normalized_plurality_margin", -1.0)),
                        "candidate_normalized_plurality_margin": float(rollout.get("normalized_plurality_margin", -1.0)),
                        "counterfactual_gold_vote_correct": counterfactual_gold_vote_correct,
                        "plurality_pivotal_fix_opportunity": bool(
                            not baseline_vote_correct and counterfactual_gold_vote_correct
                        ),
                        "plurality_pivotal_fix": bool(
                            not baseline_vote_correct and counterfactual_gold_vote_correct and candidate_vote_correct
                        ),
                        "plurality_pivotal_loss": bool(baseline_vote_correct and not candidate_vote_correct),
                        "counterfactual_gold_margin": float(counterfactual_gold_diagnostics.get("normalized_vote_margin", -1.0)),
                        "baseline_target_in_dominant_wrong_cluster": in_dominant_wrong_cluster(baseline_answers, agent_id),
                        "candidate_target_in_dominant_wrong_cluster": in_dominant_wrong_cluster(answers, agent_id),
                        "capability_residual_family": str(residual_info.get("capability_residual_family", CapabilityResidualFamily.UNKNOWN.value)),
                        "capability_residual_confidence": float(residual_info.get("confidence", 0.0) or 0.0),
                        "target_answer": answers[agent_id] if agent_id < len(answers) else "",
                        "target_trace_novelty": target_trace_novelty,
                        "target_useful_diversity": target_useful_diversity,
                        "rescue": rescue,
                        "rescue_useful_diversity": target_useful_diversity * float(rescue),
                        "baseline_team_accuracy": float(baseline_vote_correct),
                        "baseline_mean_vote_margin": float(baseline_rollout.get("normalized_vote_margin", -1.0)),
                        "baseline_boundary_useful_diversity": float(baseline_rollout.get("boundary_useful_diversity", 0.0)),
                        "baseline_embedding_diversity": float(baseline_rollout.get("embedding_diversity", 0.0)),
                        "baseline_invalid_rate": float(baseline_rollout.get("invalid_rate", 1.0)),
                        "baseline_mean_embedding_overlap": float(baseline_rollout.get("mean_embedding_overlap", 0.0)),
                        "candidate_team_accuracy": float(candidate_vote_correct),
                        "candidate_mean_vote_margin": float(rollout.get("normalized_vote_margin", -1.0)),
                        "candidate_boundary_useful_diversity": float(rollout.get("boundary_useful_diversity", 0.0)),
                        "candidate_embedding_diversity": float(rollout.get("embedding_diversity", 0.0)),
                        "candidate_invalid_rate": float(rollout.get("invalid_rate", 1.0)),
                        "candidate_mean_embedding_overlap": float(rollout.get("mean_embedding_overlap", 0.0)),
                        "baseline_solver_reuse_hits": int(baseline_reuse_stats.get("solver_reuse_hits", 0) or 0),
                        "baseline_solver_reuse_misses": int(baseline_reuse_stats.get("solver_reuse_misses", 0) or 0),
                        "baseline_solver_calls": int(baseline_reuse_stats.get("solver_calls", 0) or 0),
                        "baseline_solver_reuse_total": int(baseline_reuse_stats.get("solver_reuse_total", 0) or 0),
                    }
                )
                if self._v7_residual_protocol_enabled():
                    row["behavior_context"] = self._behavior_context_for_baseline(
                        agent_id=agent_id,
                        answers=baseline_answers,
                        gold=gold,
                        rollout=baseline_rollout,
                        question_hash=sample_hash,
                    )
            return row

        raw = await asyncio.gather(*[run_one(ex) for ex in eval_batch], return_exceptions=True)
        rows = [r for r in raw if isinstance(r, dict)]
        errors = [normalize_spaces(str(r))[:240] for r in raw if isinstance(r, Exception)]
        diversity = self._clip01(float(np.mean([float(r.get("embedding_diversity", 0.0)) for r in rows])) if rows else 0.0)
        team_accuracy = self._clip01(float(np.mean([float(r.get("team_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        invalid_rate = self._clip01(float(np.mean([float(r.get("invalid", 1.0)) for r in rows])) if rows else 1.0)
        invalid_score = self._clip01(1.0 - invalid_rate)
        baseline_candidate_metrics: Dict[str, Any] = {}
        if self._uses_baseline_candidate_metrics():
            baseline_team_accuracy = self._clip01(float(np.mean([float(r.get("baseline_team_accuracy", 0.0)) for r in rows])) if rows else 0.0)
            candidate_team_accuracy = self._clip01(float(np.mean([float(r.get("candidate_team_accuracy", 0.0)) for r in rows])) if rows else team_accuracy)
            baseline_embedding_diversity = self._clip01(float(np.mean([float(r.get("baseline_embedding_diversity", 0.0)) for r in rows])) if rows else 0.0)
            candidate_embedding_diversity = self._clip01(float(np.mean([float(r.get("candidate_embedding_diversity", 0.0)) for r in rows])) if rows else diversity)
            baseline_invalid_rate = self._clip01(float(np.mean([float(r.get("baseline_invalid_rate", 1.0)) for r in rows])) if rows else 1.0)
            candidate_invalid_rate = self._clip01(float(np.mean([float(r.get("candidate_invalid_rate", 1.0)) for r in rows])) if rows else invalid_rate)
            baseline_target_accuracy = self._clip01(float(np.mean([float(r.get("baseline_target_correct", 0.0)) for r in rows])) if rows else 0.0)
            candidate_target_accuracy = self._clip01(float(np.mean([float(r.get("target_agent_correct", 0.0)) for r in rows])) if rows else 0.0)
            baseline_oracle_acc = self._clip01(float(np.mean([float(r.get("baseline_any_correct", 0.0)) for r in rows])) if rows else 0.0)
            candidate_oracle_acc = self._clip01(float(np.mean([float(r.get("candidate_any_correct", 0.0)) for r in rows])) if rows else 0.0)
            baseline_mean_vote_margin = float(np.mean([float(r.get("baseline_mean_vote_margin", -1.0)) for r in rows])) if rows else -1.0
            candidate_mean_vote_margin = float(np.mean([float(r.get("candidate_mean_vote_margin", -1.0)) for r in rows])) if rows else -1.0
            baseline_boundary_useful_diversity = self._clip01(float(np.mean([float(r.get("baseline_boundary_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
            candidate_boundary_useful_diversity = self._clip01(float(np.mean([float(r.get("candidate_boundary_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
            vote_transitions = compute_vote_transitions(
                [bool(row.get("baseline_vote_correct", 0)) for row in rows],
                [bool(row.get("candidate_vote_correct", 0)) for row in rows],
            )
            vote_delta = candidate_team_accuracy - baseline_team_accuracy
            if abs(float(vote_transitions["net_vote_delta"]) - float(vote_delta)) > PARETO_EPSILON:
                raise RuntimeError("Vote transition delta does not match candidate evaluation vote delta")
            coverage_delta = candidate_oracle_acc - baseline_oracle_acc
            coverage_transitions = compute_oracle_coverage_transitions(
                [list(row.get("baseline_individual_correct", [bool(row.get("baseline_any_correct", 0))])) for row in rows],
                [list(row.get("candidate_individual_correct", [bool(row.get("candidate_any_correct", 0))])) for row in rows],
            )
            if abs(float(coverage_transitions["net_coverage_delta"]) - float(coverage_delta)) > PARETO_EPSILON:
                raise RuntimeError("Oracle coverage transition delta does not match candidate evaluation coverage delta")
            coverage_depth_transitions = compute_coverage_depth_transitions(
                [list(row.get("baseline_individual_correct", [])) for row in rows],
                [list(row.get("candidate_individual_correct", [])) for row in rows],
                max_depth=len(self.agents),
            )
            if abs(float(coverage_depth_transitions.get("depth1_net_delta", 0.0)) - float(coverage_delta)) > PARETO_EPSILON:
                raise RuntimeError("Coverage depth-1 delta does not match oracle delta")
            rescue_rate = self._clip01(float(np.mean([float(r.get("rescue", 0.0)) for r in rows])) if rows else 0.0)
            useful_diversity = self._clip01(float(np.mean([float(r.get("target_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
            rescue_useful_diversity = self._clip01(float(np.mean([float(r.get("rescue_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
            if self._is_vote_useful_diversity_mode() or self._is_competence_depth_reward_mode():
                baseline_candidate_metrics = self._candidate_reward_vote_useful_diversity(
                    baseline_team_accuracy=baseline_team_accuracy,
                    candidate_team_accuracy=candidate_team_accuracy,
                    baseline_target_accuracy=baseline_target_accuracy,
                    candidate_target_accuracy=candidate_target_accuracy,
                    baseline_invalid_rate=baseline_invalid_rate,
                    candidate_invalid_rate=candidate_invalid_rate,
                    baseline_mean_vote_margin=baseline_mean_vote_margin,
                    candidate_mean_vote_margin=candidate_mean_vote_margin,
                    baseline_boundary_useful_diversity=baseline_boundary_useful_diversity,
                    candidate_boundary_useful_diversity=candidate_boundary_useful_diversity,
                    baseline_oracle_accuracy=baseline_oracle_acc,
                    candidate_oracle_accuracy=candidate_oracle_acc,
                    baseline_embedding_diversity=baseline_embedding_diversity,
                    candidate_embedding_diversity=candidate_embedding_diversity,
                )
            else:
                baseline_candidate_metrics = self._candidate_reward_guarded(
                    baseline_team_accuracy=baseline_team_accuracy,
                    candidate_team_accuracy=candidate_team_accuracy,
                    baseline_target_accuracy=baseline_target_accuracy,
                    candidate_target_accuracy=candidate_target_accuracy,
                    baseline_embedding_diversity=baseline_embedding_diversity,
                    candidate_embedding_diversity=candidate_embedding_diversity,
                    baseline_invalid_rate=baseline_invalid_rate,
                    candidate_invalid_rate=candidate_invalid_rate,
                )
            baseline_candidate_metrics.update(
                {
                    "baseline_team_accuracy": baseline_team_accuracy,
                    "candidate_team_accuracy": candidate_team_accuracy,
                    "baseline_oracle_acc": baseline_oracle_acc,
                    "candidate_oracle_acc": candidate_oracle_acc,
                    "coverage_delta": float(coverage_delta),
                    **vote_transitions,
                    "plurality_vote_gain_count": int(vote_transitions["vote_gain_count"]),
                    "plurality_vote_gain_rate": float(vote_transitions["vote_gain_rate"]),
                    "plurality_vote_loss_count": int(vote_transitions["vote_loss_count"]),
                    "plurality_vote_loss_rate": float(vote_transitions["vote_loss_rate"]),
                    "plurality_vote_net_count": int(vote_transitions["net_vote_count"]),
                    "plurality_vote_net_delta": float(vote_transitions["net_vote_delta"]),
                    **coverage_transitions,
                    **coverage_depth_transitions,
                    "baseline_gold_vote_count": float(np.mean([float(row.get("baseline_gold_vote_count", 0.0)) for row in rows])) if rows else 0.0,
                    "candidate_gold_vote_count": float(np.mean([float(row.get("candidate_gold_vote_count", 0.0)) for row in rows])) if rows else 0.0,
                    "baseline_largest_wrong_vote_count": float(np.mean([float(row.get("baseline_largest_wrong_vote_count", 0.0)) for row in rows])) if rows else 0.0,
                    "candidate_largest_wrong_vote_count": float(np.mean([float(row.get("candidate_largest_wrong_vote_count", 0.0)) for row in rows])) if rows else 0.0,
                    "baseline_plurality_margin_votes": float(np.mean([float(row.get("baseline_plurality_margin_votes", 0.0)) for row in rows])) if rows else 0.0,
                    "candidate_plurality_margin_votes": float(np.mean([float(row.get("candidate_plurality_margin_votes", 0.0)) for row in rows])) if rows else 0.0,
                    "plurality_margin_vote_delta": float(np.mean([
                        float(row.get("candidate_plurality_margin_votes", 0.0))
                        - float(row.get("baseline_plurality_margin_votes", 0.0)) for row in rows
                    ])) if rows else 0.0,
                    "baseline_normalized_plurality_margin": float(np.mean([float(row.get("baseline_normalized_plurality_margin", -1.0)) for row in rows])) if rows else -1.0,
                    "candidate_normalized_plurality_margin": float(np.mean([float(row.get("candidate_normalized_plurality_margin", -1.0)) for row in rows])) if rows else -1.0,
                    "normalized_plurality_margin_delta": float(np.mean([
                        float(row.get("candidate_normalized_plurality_margin", -1.0))
                        - float(row.get("baseline_normalized_plurality_margin", -1.0)) for row in rows
                    ])) if rows else 0.0,
                    "baseline_plurality_vote_tie": float(np.mean([int(bool(row.get("baseline_plurality_vote_tie", False))) for row in rows])) if rows else 0.0,
                    "candidate_plurality_vote_tie": float(np.mean([int(bool(row.get("candidate_plurality_vote_tie", False))) for row in rows])) if rows else 0.0,
                    "baseline_mean_vote_margin": baseline_mean_vote_margin,
                    "candidate_mean_vote_margin": candidate_mean_vote_margin,
                    "vote_margin_delta": candidate_mean_vote_margin - baseline_mean_vote_margin,
                    "baseline_boundary_useful_diversity": baseline_boundary_useful_diversity,
                    "candidate_boundary_useful_diversity": candidate_boundary_useful_diversity,
                    "boundary_useful_diversity_delta": candidate_boundary_useful_diversity - baseline_boundary_useful_diversity,
                    "baseline_target_accuracy": baseline_target_accuracy,
                    "candidate_target_accuracy": candidate_target_accuracy,
                    "target_agent_accuracy": candidate_target_accuracy,
                    "rescue_rate": rescue_rate,
                    "useful_diversity": useful_diversity,
                    "rescue_useful_diversity": rescue_useful_diversity,
                    "baseline_embedding_diversity": baseline_embedding_diversity,
                    "candidate_embedding_diversity": candidate_embedding_diversity,
                    "baseline_invalid_rate": baseline_invalid_rate,
                    "candidate_invalid_rate": candidate_invalid_rate,
                    "baseline_mean_embedding_overlap": self._clip01(float(np.mean([float(r.get("baseline_mean_embedding_overlap", 0.0)) for r in rows])) if rows else 0.0),
                    "candidate_mean_embedding_overlap": self._clip01(float(np.mean([float(r.get("candidate_mean_embedding_overlap", 0.0)) for r in rows])) if rows else 0.0),
                }
            )
            if bool(getattr(self.cfg, "shared_error_metrics_enabled", False)) or self._uses_vote_error_pareto_selection() or self._uses_competence_depth_pareto_selection() or self._residual_specialization_enabled():
                baseline_candidate_metrics.update(self._candidate_boundary_error_metrics(rows))
                paired_keys = (
                    "question_hash", "baseline_target_correct", "candidate_target_correct",
                    "peer_wrong_count", "baseline_vote_correct", "candidate_vote_correct",
                    "baseline_vote_margin", "candidate_vote_margin", "counterfactual_gold_vote_correct",
                    "counterfactual_gold_margin", "baseline_target_in_dominant_wrong_cluster",
                    "candidate_target_in_dominant_wrong_cluster", "capability_residual_family",
                )
                baseline_candidate_metrics["paired_boundary_transition_rows"] = [
                    {key: row.get(key) for key in paired_keys} for row in rows
                ]
            if self._residual_specialization_enabled():
                baseline_candidate_metrics.update(self._candidate_residual_metrics(rows))
                baseline_candidate_metrics["capability_alignment"] = self.capability_alignment(
                    self.agents[agent_id], baseline_candidate_metrics
                )
            if self._v7_residual_protocol_enabled():
                baseline_candidate_metrics.update(self._candidate_behavior_metrics(rows))
            baseline_candidate_metrics.update(
                compute_candidate_metric_deltas(
                    baseline_target_accuracy=baseline_target_accuracy,
                    candidate_target_accuracy=candidate_target_accuracy,
                    baseline_team_accuracy=baseline_team_accuracy,
                    candidate_team_accuracy=candidate_team_accuracy,
                    baseline_oracle_accuracy=baseline_oracle_acc,
                    candidate_oracle_accuracy=candidate_oracle_acc,
                    baseline_embedding_diversity=baseline_embedding_diversity,
                    candidate_embedding_diversity=candidate_embedding_diversity,
                    baseline_invalid_rate=baseline_invalid_rate,
                    candidate_invalid_rate=candidate_invalid_rate,
                )
            )
            if self._is_coverage_useful_diversity_mode():
                baseline_candidate_metrics.update(self._candidate_reward_coverage_useful_diversity(baseline_candidate_metrics))
            if self._is_competence_depth_reward_mode():
                v7_reward = float(baseline_candidate_metrics.get("reward", 0.0) or 0.0)
                baseline_candidate_metrics.update(
                    self._candidate_reward_competence_depth(baseline_candidate_metrics, v7_reward)
                )
            reward = float(baseline_candidate_metrics.get("reward", 0.0))
        else:
            reward = team_accuracy
        solver_reuse_hits = int(sum(int(r.get("solver_reuse_hits", 0) or 0) for r in rows))
        solver_reuse_misses = int(sum(int(r.get("solver_reuse_misses", 0) or 0) for r in rows))
        solver_calls = int(sum(int(r.get("solver_calls", 0) or 0) for r in rows))
        solver_reuse_total = int(sum(int(r.get("solver_reuse_total", 0) or 0) for r in rows))
        result = {
            "reward": reward,
            "embedding_diversity": diversity,
            "mean_embedding_overlap": self._clip01(float(np.mean([float(r.get("mean_embedding_overlap", 0.0)) for r in rows])) if rows else 0.0),
            "target_overlap_pressure": self._clip01(float(np.mean([float(r.get("target_overlap_pressure", 0.0)) for r in rows])) if rows else 0.0),
            "homogeneous_case_count": float(np.mean([float(r.get("homogeneous_case_count", 0.0)) for r in rows])) if rows else 0.0,
            "resolved_case_count": float(np.mean([float(r.get("resolved_case_count", 0.0)) for r in rows])) if rows else 0.0,
            "new_homogeneous_case_count": float(np.mean([float(r.get("new_homogeneous_case_count", 0.0)) for r in rows])) if rows else 0.0,
            "team_accuracy": team_accuracy,
            "invalid_rate": invalid_rate,
            "invalid_score": invalid_score,
            "num_eval_samples": len(rows),
            "candidate_prompt": candidate_prompt,
            "errors": errors,
            "solver_reuse_enabled": bool(self.cfg.candidate_reuse_recorded_rollouts),
            "solver_reuse_hits": solver_reuse_hits,
            "solver_reuse_misses": solver_reuse_misses,
            "solver_calls": solver_calls,
            "solver_reuse_total": solver_reuse_total,
            "solver_reuse_hit_rate": float(solver_reuse_hits / solver_reuse_total) if solver_reuse_total else 0.0,
            "baseline_solver_calls": int(sum(int(r.get("baseline_solver_calls", 0) or 0) for r in rows)),
            "baseline_solver_reuse_hits": int(sum(int(r.get("baseline_solver_reuse_hits", 0) or 0) for r in rows)),
            "baseline_solver_reuse_misses": int(sum(int(r.get("baseline_solver_reuse_misses", 0) or 0) for r in rows)),
            "baseline_solver_reuse_total": int(sum(int(r.get("baseline_solver_reuse_total", 0) or 0) for r in rows)),
            "candidate_eval_strategy": str(getattr(self.cfg, "candidate_eval_strategy", "random")),
            "candidate_eval_pool_size": int(getattr(self.cfg, "candidate_eval_pool_size", 0) or 0),
            "candidate_eval_pool_actual_size": int(getattr(self.cfg, "candidate_eval_pool_actual_size", 0) or 0),
            "candidate_eval_batch_size": int(getattr(self.cfg, "candidate_eval_batch_size", 0) or 0),
            "actual_eval_batch_size": len(eval_batch),
            "num_eval_repeats": int(getattr(self.cfg, "candidate_eval_repeats", 1) or 1),
            **self._candidate_eval_audit_fields(eval_batch),
        }
        result.update(baseline_candidate_metrics)
        return result

    async def update_prompt_with_beam(
        self,
        agent_id: int,
        overlap_diagnosis: Dict[str, Any],
        eval_batch: List[Dict[str, str]],
        step_id: int,
        epoch_id: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        agent = self.agents[agent_id]
        reference_values = list(getattr(self, "previous_epoch_per_agent_acc", []) or []) or list(overlap_diagnosis.get("per_agent_accuracy", []))
        target_reference = float(reference_values[agent_id]) if agent_id < len(reference_values) else 0.0
        ordered_reference = sorted(float(value) for value in reference_values)
        team_bottom2_reference = float(np.mean(ordered_reference[: min(2, len(ordered_reference))])) if ordered_reference else 0.0
        team_best = max(ordered_reference, default=0.0)
        competence_log_fields = {
            "specialization_strength": float(getattr(self, "specialization_strength", 0.0)),
            "competence_floor_low": float(getattr(self.cfg, "competence_floor_low", 0.55)),
            "competence_floor_high": float(getattr(self.cfg, "competence_floor_high", 0.65)),
            "target_agent_reference_accuracy": target_reference,
            "target_agent_competence_deficit": max(0.0, float(getattr(self.cfg, "competence_floor_high", 0.65)) - target_reference),
            "team_bottom2_reference_accuracy": team_bottom2_reference,
            "team_best_minus_bottom2_gap": team_best - team_bottom2_reference,
        }
        update_attempt_id = self._update_attempt_id(epoch_id, step_id, agent_id)
        agent_update_turn = sum(int(value or 0) for value in agent.optimizer_update_count_by_epoch.values()) + 1
        beam = getattr(agent, "prompt_beam", []) or [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]
        parent_sources = ["active"] * len(beam)
        if self._is_stable_qd_lineage():
            beam, parent_sources = self._select_stable_qd_parents(agent, epoch_id)
        generation = max([int(x.get("generation", 0) or 0) for x in beam] + [0]) + 1
        candidate_pool: List[Dict[str, Any]] = []
        seen = set()
        generation_batches = self._build_case_generation_batches(agent_id, overlap_diagnosis)
        if not generation_batches:
            generation_batches = [{"batch_type": "window_update_diagnosis", "cases": [], "purpose": "general reward-relevant window repair"}]
        requested = max(1, int(self.cfg.num_candidates_per_parent))
        optimizer_generation_records: List[Dict[str, Any]] = []
        parent_jobs = []
        for parent_idx, parent in enumerate(beam):
            parent_prompt = str(parent.get("prompt", agent.current_prompt))
            parent_id = str(parent.get("id", self._hash(parent_prompt)))
            parent_batches = (
                list(generation_batches)
                if self._is_v82_hybrid()
                else [generation_batches[i % len(generation_batches)] for i in range(requested)]
            )
            parent_jobs.append(
                {
                    "parent_idx": parent_idx,
                    "parent": parent,
                    "parent_prompt": parent_prompt,
                    "parent_id": parent_id,
                    "parent_batches": parent_batches,
                    "parent_source": parent_sources[parent_idx] if parent_idx < len(parent_sources) else "active",
                }
            )

        configured_parent_concurrency = int(getattr(self.cfg, "optimizer_parent_concurrency", 1) or 1)
        parent_concurrency = max(1, min(configured_parent_concurrency, len(parent_jobs) or 1))
        parent_sem = asyncio.Semaphore(parent_concurrency)

        async def propose_for_parent(job: Dict[str, Any]) -> Dict[str, Any]:
            async with parent_sem:
                context_token = TCS_AUDIT_CONTEXT.set(
                    {
                        "optimizer_architecture": str(getattr(self.cfg, "optimizer_architecture", "") or ""),
                        "epoch": int(epoch_id),
                        "step": int(step_id),
                        "agent_id": int(agent_id),
                        "parent_id": str(job["parent_id"]),
                        "execution_session_id": self._current_execution_session_id(),
                        "update_attempt_id": update_attempt_id,
                        "tcs_call_group_id": self._tcs_call_group_id(
                            update_attempt_id,
                            str(job["parent_id"]),
                            str(job["parent_prompt"]),
                        ),
                        "teacher_critic_round": 0,
                    }
                )
                try:
                    feedback = job.get("refill_feedback")
                    proposals = await self.propose_candidates(
                        agent_id=agent_id,
                        parent_prompt=str(job["parent_prompt"]),
                        overlap_diagnosis=overlap_diagnosis,
                        num_candidates=requested,
                        generation_batches=job["parent_batches"],
                        refill_feedback=feedback if isinstance(feedback, dict) else None,
                    )
                finally:
                    TCS_AUDIT_CONTEXT.reset(context_token)
                return {**job, "proposals": proposals}

        parent_results = await asyncio.gather(*[propose_for_parent(job) for job in parent_jobs])
        parent_results.sort(key=lambda x: int(x.get("parent_idx", 0)))

        for result in parent_results:
            parent_prompt = str(result.get("parent_prompt", agent.current_prompt))
            parent_id = str(result.get("parent_id", self._hash(parent_prompt)))
            parent_batches = result.get("parent_batches", [])
            if not isinstance(parent_batches, list) or not parent_batches:
                parent_batches = [generation_batches[0]]
            proposals = result.get("proposals", [])
            if not isinstance(proposals, list):
                proposals = []
            parent_diagnostics = self._empty_optimizer_generation_diagnostics()
            if proposals:
                proposal_diag = proposals[0].get("optimizer_generation_diagnostics", {}) if isinstance(proposals[0], dict) else {}
                if isinstance(proposal_diag, dict):
                    parent_diagnostics.update(proposal_diag)
            else:
                parent_diagnostics.update(self._optimizer_generation_diagnostics_for_parent(agent_id, parent_prompt))
            optimizer_generation_records.append(parent_diagnostics)
            for idx, proposal in enumerate(proposals):
                prompt = str(proposal.get("candidate_prompt", "")).strip()
                prompt, _ = self._sanitize_prompt(prompt, agent_id)
                key = normalize_spaces(prompt).lower()
                preserve_duplicate_objects = str(getattr(self.cfg, "candidate_eval_execution_mode", "legacy")) == "factorized_cached"
                if not prompt or (key in seen and not preserve_duplicate_objects):
                    continue
                seen.add(key)
                batch = parent_batches[idx % len(parent_batches)]
                candidate_pool.append(
                    {
                        "candidate_id": f"g{generation}_a{agent_id}_p{self._hash(parent_id)}_{idx}_{self._hash(prompt)}",
                        "prompt": prompt,
                        "parent_id": parent_id,
                        "parent_source": str(result.get("parent_source", "active")),
                        "parent_prompt": parent_prompt,
                        "generation": generation,
                        "source": "optimizer",
                        "candidate_pool_source": "optimizer",
                        "candidate_source": str(proposal.get("candidate_source", "optimizer") or "optimizer"),
                        "generation_batch_type": str(proposal.get("generation_batch_type", "")) or str(batch.get("batch_type", "")),
                        "generation_case_ids": proposal.get("generation_case_ids", []),
                        "target_error_pattern": str(proposal.get("target_error_pattern", "")),
                        "accuracy_repair_rule": str(proposal.get("accuracy_repair_rule", "")),
                        "expected_accuracy_effect": str(proposal.get("expected_accuracy_effect", "")),
                        "diversity_contribution": str(proposal.get("diversity_contribution", "")),
                        "error_correlation_reduction": str(proposal.get("error_correlation_reduction", "")),
                        "task_alignment_rule": str(proposal.get("task_alignment_rule", "")),
                        "peer_redundancy_avoidance": str(proposal.get("peer_redundancy_avoidance", "")),
                        "candidate_prompt_char_count": int(proposal.get("candidate_prompt_char_count", len(prompt)) or len(prompt)),
                        "candidate_prompt_over_soft_limit": bool(proposal.get("candidate_prompt_over_soft_limit", False)),
                        "candidate_prompt_over_hard_limit": bool(proposal.get("candidate_prompt_over_hard_limit", False)),
                        "candidate_prompt_overlength_rejected": bool(proposal.get("candidate_prompt_overlength_rejected", False)),
                        "candidate_prompt_ends_with_sentence_boundary": bool(proposal.get("candidate_prompt_ends_with_sentence_boundary", self._prompt_ends_with_sentence_boundary(prompt))),
                        "optimizer_generation_diagnostics": proposal.get("optimizer_generation_diagnostics", {}),
                        "tcs_call_group_id": str(proposal.get("tcs_call_group_id", "") or ""),
                        "execution_session_id": str(proposal.get("execution_session_id", self._current_execution_session_id()) or self._current_execution_session_id()),
                        "update_attempt_id": str(proposal.get("update_attempt_id", update_attempt_id) or update_attempt_id),
                        "proposal": proposal,
                        "prompt_hash": self._normalized_prompt_hash(prompt),
                    }
                )
                candidate_metadata = {
                    "optimizer_architecture": str(proposal.get("optimizer_architecture", getattr(self.cfg, "optimizer_architecture", ""))),
                    "candidate_source": str(proposal.get("candidate_source", "")),
                    "candidate_pool_source": "optimizer",
                    "tcs_call_group_id": str(proposal.get("tcs_call_group_id", "") or ""),
                    "execution_session_id": str(proposal.get("execution_session_id", self._current_execution_session_id()) or self._current_execution_session_id()),
                    "update_attempt_id": str(proposal.get("update_attempt_id", update_attempt_id) or update_attempt_id),
                    **dict(proposal.get("optimizer_generation_diagnostics", {}) or {}),
                }
                metadata_errors = validate_tcs_candidate_metadata(candidate_metadata)
                if metadata_errors:
                    candidate_id = str(candidate_pool[-1].get("candidate_id", ""))
                    raise RuntimeError(
                        "Invalid Teacher-Critic-Student candidate metadata: "
                        f"agent_id={agent_id} epoch={epoch_id} step={step_id} parent_id={parent_id} "
                        f"candidate_id={candidate_id} tcs_call_group_id={candidate_metadata.get('tcs_call_group_id', '')} "
                        f"metadata_errors={','.join(metadata_errors)}"
                    )
        for parent in beam:
            prompt = str(parent.get("prompt", agent.current_prompt))
            key = normalize_spaces(prompt).lower()
            if key in seen:
                continue
            seen.add(key)
            candidate_pool.append(
                {
                    "candidate_id": str(parent.get("id", "")) or f"beam_{self._hash(prompt)}",
                    "prompt": prompt,
                    "parent_id": parent.get("parent_id"),
                    "generation": int(parent.get("generation", 0) or 0),
                    "source": "existing_beam",
                    "candidate_pool_source": "existing_beam",
                    "candidate_source": "existing_beam",
                    "execution_session_id": self._current_execution_session_id(),
                    "update_attempt_id": update_attempt_id,
                    "generation_batch_type": "",
                    "generation_case_ids": [],
                    "target_error_pattern": "",
                    "accuracy_repair_rule": "",
                    "expected_accuracy_effect": "",
                    "diversity_contribution": "",
                    "error_correlation_reduction": "",
                    "task_alignment_rule": "",
                    "peer_redundancy_avoidance": "",
                    "optimizer_generation_diagnostics": self._empty_optimizer_generation_diagnostics(),
                    "proposal": {},
                    "prompt_hash": self._normalized_prompt_hash(prompt),
                }
            )
        current_key = normalize_spaces(str(agent.current_prompt)).lower()
        if current_key not in seen:
            current_prompt = str(agent.current_prompt)
            candidate_pool.append(
                {
                    "candidate_id": f"active_{self._hash(current_prompt)}",
                    "prompt": current_prompt,
                    "parent_id": None,
                    "generation": generation,
                    "source": "current_active_fallback",
                    "candidate_pool_source": "current_active_fallback",
                    "candidate_source": "current_active_fallback",
                    "execution_session_id": self._current_execution_session_id(),
                    "update_attempt_id": update_attempt_id,
                    "generation_batch_type": "",
                    "generation_case_ids": [],
                    "target_error_pattern": "",
                    "accuracy_repair_rule": "",
                    "expected_accuracy_effect": "",
                    "diversity_contribution": "",
                    "error_correlation_reduction": "",
                    "task_alignment_rule": "",
                    "peer_redundancy_avoidance": "",
                    "optimizer_generation_diagnostics": self._empty_optimizer_generation_diagnostics(),
                    "proposal": {},
                    "prompt_hash": self._normalized_prompt_hash(current_prompt),
                }
            )

        initial_prescreen_failures = []
        if self._is_stable_qd_lineage():
            accepted_pool, prescreen_seen = [], set()
            for candidate in candidate_pool:
                if self._candidate_pool_source(candidate) != "optimizer":
                    accepted_pool.append(candidate)
                    continue
                reasons = cheap_prescreen(
                    candidate,
                    self._normalized_prompt_hash(str(candidate.get("parent_prompt", agent.current_prompt))),
                    prescreen_seen,
                    parent=next(
                        (parent for parent in beam if str(parent.get("id", "")) == str(candidate.get("parent_id", ""))),
                        None,
                    ),
                )
                if reasons:
                    candidate["cheap_prescreen_reasons"] = reasons
                    initial_prescreen_failures.append({
                        "candidate_type": str(candidate.get("proposal", {}).get("candidate_type", "")),
                        "failure_stage": "cheap_prescreen", "reasons": reasons,
                    })
                    continue
                prescreen_seen.update({str(candidate.get("prompt_hash", "")), normalize_spaces(str(candidate.get("prompt", ""))).lower()})
                accepted_pool.append(candidate)
            candidate_pool = accepted_pool

        target_case_ids = {
            str(c.get("case_id", ""))
            for b in generation_batches
            if str(b.get("batch_type", "")) == "target_error_repair"
            for c in b.get("cases", [])
            if isinstance(c, dict) and str(c.get("case_id", ""))
        }
        num_target_error_cases = len(target_case_ids)
        num_accuracy_repair_candidates = sum(
            1
            for c in candidate_pool
            if str(c.get("generation_batch_type", "")) == "target_error_repair"
            or bool(str(c.get("target_error_pattern", "")).strip())
            or "accuracy_repair" in str(c.get("candidate_source", ""))
        )
        num_diversity_candidates = sum(
            1
            for c in candidate_pool
            if str(c.get("generation_batch_type", "")) in {"useful_diversity_repair", "random_window", "window_update_diagnosis"}
            and not bool(str(c.get("target_error_pattern", "")).strip())
        )
        requested_optimizer_candidates = len(beam) * requested
        num_optimizer_candidates = sum(1 for c in candidate_pool if self._is_optimizer_generated_candidate_source(self._candidate_generation_source(c)))
        num_fallback_candidates = sum(1 for c in candidate_pool if "fallback" in self._candidate_generation_source(c))
        num_existing_beam_candidates = sum(1 for c in candidate_pool if self._candidate_pool_source(c) == "existing_beam")
        num_tcs_optimizer_candidates = sum(
            1
            for c in candidate_pool
            if self._candidate_generation_source(c) == "teacher_critic_student" and self._candidate_pool_source(c) == "optimizer"
        )
        num_tcs_metadata_invalid_candidates = 0
        num_tcs_metadata_valid_candidates = num_tcs_optimizer_candidates
        tcs_execution_complete = (
            bool(num_tcs_optimizer_candidates)
            and num_tcs_optimizer_candidates == num_tcs_metadata_valid_candidates
            and num_tcs_metadata_invalid_candidates == 0
        )
        fallback_enabled = str(getattr(self.cfg, "optimizer_fallback_mode", "none") or "none").lower() == "template"
        optimizer_underfilled = num_optimizer_candidates < requested_optimizer_candidates
        optimizer_generation_summary = self._empty_optimizer_generation_diagnostics()
        for record in optimizer_generation_records:
            if not isinstance(record, dict):
                continue
            for key in [
                "optimizer_raw_response_empty",
                "optimizer_json_parse_failed",
                "optimizer_raw_candidate_count",
                "optimizer_empty_prompt_count",
                "optimizer_sanitized_count",
                "optimizer_redundant_filtered_count",
                "optimizer_schema_filtered_count",
                "optimizer_final_candidate_count",
                "teacher_critic_rounds",
                "teacher_rewrite_count",
                "student_candidate_count_raw",
                "student_candidate_count_final",
                "student_candidate_filtered_count",
                "student_missing_required_field_count",
                "num_teacher_calls",
                "num_critic_calls",
                "num_teacher_rewrite_calls",
                "num_student_calls",
                "num_student_retry_calls",
                "num_student_repair_calls",
            ]:
                optimizer_generation_summary[key] += int(record.get(key, 0) or 0)
            for key in [
                "student_raw_response_empty",
                "student_json_parse_failed",
                "student_json_retry_attempted",
                "student_json_retry_succeeded",
                "student_json_repair_attempted",
                "student_json_repair_succeeded",
                "student_json_has_candidates_key",
                "student_candidates_is_list",
                "student_candidates_empty_list",
                "student_refusal_or_explanation",
            ]:
                optimizer_generation_summary[key] = bool(optimizer_generation_summary.get(key, False) or record.get(key, False))
        optimizer_generation_summary["optimizer_underfilled"] = bool(optimizer_underfilled)
        for key in [
            "optimizer_architecture",
            "teacher_question",
            "teacher_question_approved",
            "teacher_question_rejected",
            "teacher_question_rejection_reason",
            "teacher_question_forced_best_score",
            "teacher_question_forced_best_round",
            "teacher_question_forced_below_threshold",
            "teacher_question_score",
            "teacher_quality_critique",
            "teacher_specificity_critique",
            "teacher_task_alignment_critique",
            "teacher_error_alignment_critique",
            "teacher_diversity_critique",
            "student_candidate_filter_reasons",
            "student_all_candidates_filtered",
            "student_missing_required_fields",
            "student_raw_response_preview",
            "student_json_parse_error",
            "student_json_retry_raw_response_preview",
            "student_json_repair_raw_response_preview",
            "student_json_repair_failure_reason",
            "student_failure_stage",
        ]:
            values = [record.get(key) for record in optimizer_generation_records if isinstance(record, dict) and record.get(key) not in (None, "", [])]
            if values:
                optimizer_generation_summary[key] = values[-1]

        evaluated = []
        peer_prompts = self._active_prompt_list()
        if str(getattr(self.cfg, "candidate_eval_execution_mode", "legacy")) == "factorized_cached":
            candidate_eval_cache_stats = await self._prewarm_factorized_candidate_rollouts(
                agent_id=agent_id,
                eval_batch=eval_batch,
                peer_prompts=peer_prompts,
                candidate_pool=candidate_pool,
            )
        else:
            prewarm = await self.ensure_recorded_rollouts_for_prompts(
                eval_batch=eval_batch,
                prompts=peer_prompts,
                source=f"candidate_peer_prewarm_agent_{agent_id}",
            )
            candidate_eval_cache_stats = {
                "candidate_eval_execution_mode": "legacy",
                "candidate_eval_candidate_object_count": len(candidate_pool),
                "candidate_eval_unique_target_prompt_count": len({self._hash(normalize_spaces(str(c.get("prompt", "")))) for c in candidate_pool}),
                "candidate_eval_duplicate_target_prompt_count": 0,
                "candidate_eval_example_count": len(eval_batch),
                "candidate_eval_repeat_count": 1,
                "candidate_eval_naive_rollout_request_count": len(candidate_pool) * len(self.agents) * len(eval_batch),
                "candidate_eval_factorized_rollout_request_count": 0,
                "candidate_eval_unique_rollout_key_count": 0,
                "candidate_eval_memory_cache_hit_count": int(prewarm.get("solver_reuse_hits", 0) or 0),
                "candidate_eval_persisted_cache_hit_count": 0,
                "candidate_eval_inflight_reuse_count": 0,
                "candidate_eval_solver_api_call_count": int(prewarm.get("solver_calls", 0) or 0),
                "candidate_eval_rollout_failure_count": 0,
                "candidate_eval_calls_saved_vs_naive": 0,
                "candidate_eval_cache_hit_rate": float(prewarm.get("solver_reuse_hit_rate", 0.0) or 0.0),
                "candidate_eval_peer_rollout_key_count": len(self.agents) * len(eval_batch),
                "candidate_eval_target_rollout_key_count": 0,
                "candidate_eval_prompt_dedup_savings": 0,
            }
        baseline_cases = self._cases_for_agent(overlap_diagnosis, agent_id)
        configured_concurrency = int(getattr(self.cfg, "candidate_eval_concurrency", 0) or 0)
        eval_concurrency = len(candidate_pool) if configured_concurrency <= 0 else min(configured_concurrency, len(candidate_pool))
        sem = asyncio.Semaphore(max(1, eval_concurrency))

        async def evaluate_one_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                metrics = await self.evaluate_candidate_prompt(
                    agent_id=agent_id,
                    candidate_prompt=str(candidate["prompt"]),
                    peer_prompts=peer_prompts,
                    eval_batch=eval_batch,
                    role_spec=candidate.get("proposal", {}),
                    baseline_homogeneous_cases=baseline_cases,
                )
                return {**candidate, "metrics": metrics, "reward": float(metrics.get("reward", 0.0))}

        raw_evaluated = await asyncio.gather(*[evaluate_one_candidate(c) for c in candidate_pool], return_exceptions=True)
        for idx, item in enumerate(raw_evaluated):
            if isinstance(item, dict):
                evaluated.append(item)
                continue
            candidate = candidate_pool[idx]
            metrics = await self.evaluate_candidate_prompt(
                agent_id=agent_id,
                candidate_prompt=str(candidate["prompt"]),
                peer_prompts=peer_prompts,
                eval_batch=eval_batch,
                role_spec=candidate.get("proposal", {}),
                baseline_homogeneous_cases=baseline_cases,
            )
            evaluated.append({**candidate, "metrics": metrics, "reward": float(metrics.get("reward", 0.0))})
        old_hash = self._hash(agent.current_prompt)
        trajectory_guard_enabled = self._v7_residual_protocol_enabled()
        candidate_guard_enabled = bool(getattr(self.cfg, "competence_depth1_candidate_guard_enabled", False))
        pareto_summary = {
            "num_pareto_feasible": None,
            "num_pareto_infeasible": None,
            "num_pareto_fronts": None,
            "pareto_front0_size": None,
            "pareto_forced_current_fallback": None,
        }
        for item in evaluated:
            metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
            if self._is_v82_hybrid():
                proposal = item.get("proposal", {}) if isinstance(item.get("proposal", {}), dict) else {}
                signature = list(proposal.get("mechanism_signature", [])) or normalize_mechanism_signature(
                    proposal.get("mechanism_steps", [])
                )
                parent_item = next(
                    (row for row in beam if str(row.get("id", "")) == str(item.get("parent_id", ""))),
                    None,
                )
                parent_metrics = parent_item.get("metrics", {}) if isinstance(parent_item, dict) else {}
                parent_signature = list(parent_metrics.get("mechanism_signature", []))
                distance = mechanism_signature_distance(signature, parent_signature)
                metrics.update({
                    "candidate_type": str(proposal.get("candidate_type", "")),
                    "mechanism_signature": signature,
                    "parent_mechanism_signature": parent_signature,
                    "peer_dominant_mechanism_signature": [],
                    "mechanism_signature_distance": distance,
                    "mechanism_novelty_bonus": (
                        0.0
                        if self._is_stable_qd_lineage()
                        else float(getattr(self.cfg, "mechanism_novelty_bonus_weight", 0.2)) * distance
                    ),
                })
                if signature:
                    self.mechanism_signature_by_prompt_hash[self._normalized_prompt_hash(str(item.get("prompt", "")))] = list(signature)
                if self._is_stable_qd_lineage():
                    self._attach_stable_mechanism_representation(item)
            if trajectory_guard_enabled:
                metrics.update(self._candidate_trajectory_feasibility(agent, item))
            if self._is_v82_hybrid():
                metrics = self._apply_hybrid_soft_guards(metrics)
            depth1_guard_passed = True if self._is_stable_qd_lineage() else self._apply_competence_depth1_candidate_guard(metrics)
            if self._is_v82_hybrid():
                _, _, hard_feasible = self._vote_pareto_feasibility(metrics)
                metrics["hard_guard_passed"] = bool(depth1_guard_passed and hard_feasible and not metrics.get("rejection_reason"))
                item["reward"] = float(metrics.get("penalized_reward", item.get("reward", 0.0)) or 0.0)
            item["metrics"] = metrics
            item["trajectory_feasible"] = bool(depth1_guard_passed) and not bool(metrics.get("rejection_reason", ""))
            if not item["trajectory_feasible"]:
                item["pareto_feasible"] = False
                item["pareto_rank"] = None
                item["pareto_crowding_distance"] = None
                item["pareto_selected"] = False
                item["pareto_forced_fallback"] = False
        if self._is_stable_qd_lineage():
            existing_niches = list(getattr(agent, "safe_qd_archive", [])) + list(getattr(agent, "probation_archive", []))
            for item in evaluated:
                item["is_incumbent"] = str(item.get("prompt_hash", "")) == self._normalized_prompt_hash(agent.current_prompt)
                parent = next((row for row in beam if str(row.get("id", "")) == str(item.get("parent_id", ""))), None)
                self._mark_mechanism_novelty(item, parent=parent, existing=existing_niches)
                item["archive_bucket"] = "safe" if item["is_incumbent"] else candidate_quality_bucket(item, self.cfg)
                existing_niches.append(item)
            safe_archive = select_safe_archive(
                [*getattr(agent, "safe_qd_archive", []), *evaluated],
                self._normalized_prompt_hash(agent.current_prompt), int(self.cfg.qd_archive_size_per_agent),
            )
            for item in safe_archive:
                item["archive_bucket"] = "safe"
            probation = [item for item in evaluated if item.get("archive_bucket") == "probation"]
            for item in probation:
                item["probation_created_update"] = int(agent_update_turn)
            prior_probation = list(getattr(agent, "probation_archive", []))
            agent.probation_archive = (probation + prior_probation)[: int(self.cfg.probation_archive_size_per_agent)]
            agent.safe_qd_archive = safe_archive
            self._refresh_joint_representatives(agent)
            requirements = refill_requirements(evaluated, self.cfg)
            selected = list(agent.prompt_beam)
            pareto_summary.update({
                "safe_archive_size": len(agent.safe_qd_archive),
                "probation_archive_count": len(agent.probation_archive),
                **requirements,
            })
            self.per_agent_optimizer_update_count[f"{epoch_id}:{agent_id}"] = int(
                self.per_agent_optimizer_update_count.get(f"{epoch_id}:{agent_id}", 0)
            ) + 1
            refill_round_count = 0
            refill_requested_candidate_count = 0
            refill_actual_candidate_count = 0
            refill_trigger_reasons = list(requirements.get("missing", []))
            refill_stop_reason = "requirements_met" if requirements.get("met") else "max_rounds_reached"
            refill_solver_calls = 0
            refill_solver_call_limit_reached = False
            prior_probation_ids = {
                str(item.get("id", self._hash(str(item.get("prompt", "")))))
                for item in getattr(agent, "probation_archive", [])
            }
            prior_failures = initial_prescreen_failures + [
                {
                    "candidate_type": str(item.get("metrics", {}).get("candidate_type", "")),
                    "failure_stage": "candidate_evaluation",
                    "reasons": [str(item.get("metrics", {}).get("rejection_reason", item.get("archive_bucket", "")))],
                    "accuracy_delta": float(item.get("metrics", {}).get("accuracy_delta", 0.0) or 0.0),
                    "depth1_gain_count": max(0, int(item.get("metrics", {}).get("depth1_net_delta", 0) or 0)),
                    "depth1_loss_count": max(0, -int(item.get("metrics", {}).get("depth1_net_delta", 0) or 0)),
                }
                for item in evaluated if item.get("archive_bucket") != "safe"
            ]
            prior_failures.extend(
                {
                    "candidate_type": str(item.get("metrics", {}).get("candidate_type", "")),
                    "failure_stage": "archive_assignment",
                    "reasons": ["near_duplicate_existing_niche"],
                    "nearest_niche": repr(mechanism_niche_key(item.get("metrics", {}).get("mechanism_representation", {}))),
                }
                for item in evaluated
                if str(item.get("metrics", {}).get("candidate_type", "")) == "mechanism_alternative"
                and not bool(item.get("metrics", {}).get("mechanism_novel", False))
            )
            while (
                bool(self.cfg.candidate_refill_enabled)
                and not requirements.get("met")
                and refill_round_count < int(self.cfg.candidate_refill_max_rounds)
            ):
                active_parent = beam[0]
                active_parent_id = str(active_parent.get("id", self._hash(agent.current_prompt)))
                parent_unique_count = sum(
                    1 for item in evaluated
                    if item.get("candidate_pool_source") == "optimizer"
                    and str(item.get("parent_id", "")) == active_parent_id
                )
                remaining_unique_slots = int(self.cfg.candidate_refill_max_unique_candidates_per_parent) - parent_unique_count
                if remaining_unique_slots <= 0:
                    refill_stop_reason = "max_unique_candidates_reached"
                    break
                refill_round_count += 1
                round_candidate_limit = min(int(self.cfg.candidate_refill_candidates_per_round), remaining_unique_slots)
                refill_requested_candidate_count += round_candidate_limit
                refill_feedback = {
                    "refill_round": refill_round_count,
                    "required_candidate_types_missing": list(requirements.get("missing", [])),
                    "previous_candidate_failures": (
                        prior_failures[-6:] if bool(self.cfg.candidate_refill_feed_rejection_reasons) else []
                    ),
                    "preserve_successes": ["Preserve competence and any valid mechanism steps from safe candidates."],
                }
                refill_job = {
                    "parent_idx": 0, "parent": active_parent,
                    "parent_prompt": str(active_parent.get("prompt", agent.current_prompt)),
                    "parent_id": str(active_parent.get("id", self._hash(agent.current_prompt))),
                    "parent_batches": list(generation_batches), "refill_feedback": refill_feedback,
                }
                refill_result = await propose_for_parent(refill_job)
                proposals = refill_result.get("proposals", []) if isinstance(refill_result.get("proposals", []), list) else []
                if not proposals:
                    refill_stop_reason = "optimizer_failure"
                    break
                new_candidates = []
                for index, proposal in enumerate(proposals[:round_candidate_limit]):
                    prompt = str(proposal.get("candidate_prompt", "")).strip()
                    prompt, _ = self._sanitize_prompt(prompt, agent_id)
                    candidate = {
                        "candidate_id": f"refill{refill_round}_a{agent_id}_{index}_{self._hash(prompt)}",
                        "prompt": prompt, "prompt_hash": self._normalized_prompt_hash(prompt),
                        "parent_id": refill_job["parent_id"], "parent_prompt": refill_job["parent_prompt"],
                        "generation": generation + refill_round_count, "source": "optimizer",
                        "candidate_pool_source": "optimizer",
                        # Preserve TCS provenance; refill is an event, not a new generator.
                        "candidate_source": str(proposal.get("candidate_source", "teacher_critic_student") or "teacher_critic_student"),
                        "refill_candidate": True,
                        "proposal": proposal,
                    }
                    prescreen = cheap_prescreen(
                        candidate,
                        self._normalized_prompt_hash(refill_job["parent_prompt"]),
                        seen,
                        parent=active_parent,
                    )
                    if prescreen:
                        candidate["cheap_prescreen_reasons"] = prescreen
                        prior_failures.append({"candidate_type": str(proposal.get("candidate_type", "")), "failure_stage": "cheap_prescreen", "reasons": prescreen})
                        continue
                    seen.add(normalize_spaces(prompt).lower())
                    new_candidates.append(candidate)
                if not new_candidates:
                    refill_stop_reason = "no_new_unique_candidate"
                    break
                refill_actual_candidate_count += len(new_candidates)
                solver_cap = int(self.cfg.candidate_refill_max_solver_calls_per_agent_update)
                if str(self.cfg.candidate_eval_execution_mode) == "factorized_cached" and solver_cap <= 0:
                    await self._prewarm_factorized_candidate_rollouts(agent_id=agent_id, eval_batch=eval_batch, peer_prompts=peer_prompts, candidate_pool=new_candidates)
                for candidate in new_candidates:
                    # Peer rows are already warm; one target prompt can require at most one call per eval question.
                    candidate_call_upper_bound = len(eval_batch)
                    if solver_cap > 0 and refill_solver_calls + candidate_call_upper_bound > solver_cap:
                        refill_stop_reason = "max_unique_candidates_reached"
                        refill_solver_call_limit_reached = True
                        break
                    metrics = await self.evaluate_candidate_prompt(agent_id, candidate["prompt"], peer_prompts, eval_batch, role_spec=candidate["proposal"], baseline_homogeneous_cases=baseline_cases)
                    candidate["metrics"] = metrics
                    candidate["reward"] = float(metrics.get("reward", 0.0) or 0.0)
                    proposal = candidate["proposal"]
                    candidate["metrics"].update({
                        "candidate_type": str(proposal.get("candidate_type", "")),
                        "mechanism_steps": list(proposal.get("mechanism_steps", [])),
                    })
                    self._attach_stable_mechanism_representation(candidate)
                    self._mark_mechanism_novelty(
                        candidate,
                        parent=active_parent,
                        existing=[*getattr(agent, "safe_qd_archive", []), *getattr(agent, "probation_archive", []), *evaluated],
                    )
                    candidate["archive_bucket"] = candidate_quality_bucket(candidate, self.cfg)
                    evaluated.append(candidate)
                    refill_solver_calls += int(metrics.get("solver_calls", 0) or 0)
                if solver_cap > 0 and refill_solver_calls >= solver_cap:
                    refill_solver_call_limit_reached = True
                    break
                requirements = refill_requirements(evaluated, self.cfg)
                if requirements.get("met") and bool(self.cfg.candidate_refill_stop_when_requirements_met):
                    refill_stop_reason = "requirements_met"
                    break
                parent_unique_count = sum(
                    1 for item in evaluated
                    if item.get("candidate_pool_source") == "optimizer"
                    and str(item.get("parent_id", "")) == str(refill_job["parent_id"])
                )
                if parent_unique_count >= int(self.cfg.candidate_refill_max_unique_candidates_per_parent):
                    refill_stop_reason = "max_unique_candidates_reached"
                    break
            pareto_summary.update({
                "initial_candidate_count": num_optimizer_candidates,
                "cheap_prescreen_rejection_count": sum(1 for failure in prior_failures if failure.get("failure_stage") == "cheap_prescreen"),
                "evaluated_candidate_count": len(evaluated),
                "refill_round_count": refill_round_count,
                "refill_requested_candidate_count": refill_requested_candidate_count,
                "refill_actual_candidate_count": refill_actual_candidate_count,
                "refill_trigger_reasons": refill_trigger_reasons,
                "refill_stop_reason": refill_stop_reason,
                "refill_solver_call_budget_used": refill_solver_calls,
                "refill_solver_call_limit_reached": bool(refill_solver_call_limit_reached),
                **requirements,
            })
            for item in evaluated:
                item["is_incumbent"] = str(item.get("prompt_hash", "")) == self._normalized_prompt_hash(agent.current_prompt)
                item["archive_bucket"] = "safe" if item["is_incumbent"] else candidate_quality_bucket(item, self.cfg)
                if item.get("archive_bucket") == "safe" and str(item.get("parent_id", "")) in prior_probation_ids:
                    self.probation_to_safe_conversion_count += 1
            converted_parent_ids = {
                str(item.get("parent_id", ""))
                for item in evaluated
                if item.get("archive_bucket") == "safe" and str(item.get("parent_id", "")) in prior_probation_ids
            }
            agent.safe_qd_archive = select_safe_archive(
                [*getattr(agent, "safe_qd_archive", []), *evaluated],
                self._normalized_prompt_hash(agent.current_prompt), int(self.cfg.qd_archive_size_per_agent),
            )
            new_probation = [item for item in evaluated if item.get("archive_bucket") == "probation"]
            for item in new_probation:
                item.setdefault("probation_created_update", int(agent_update_turn))
            retained_probation = [
                item for item in getattr(agent, "probation_archive", [])
                if str(item.get("id", self._hash(str(item.get("prompt", ""))))) not in converted_parent_ids
            ]
            agent.probation_archive = (new_probation + retained_probation)[: int(self.cfg.probation_archive_size_per_agent)]
            self._refresh_joint_representatives(agent)
            selected = list(agent.prompt_beam)
            starvation = requirements["safe_non_incumbent_count"] == 0
            mechanism_starvation = requirements["safe_distinct_mechanism_count"] == 0
            self.candidate_starvation_count += int(starvation)
            self.mechanism_starvation_count += int(mechanism_starvation)
            self.search_branch_starvation_count += int(starvation and not agent.probation_archive)
            self.refill_requirements_unmet_count += int(not requirements["met"])
            agent.optimizer_update_count_by_epoch[str(epoch_id)] = int(agent.optimizer_update_count_by_epoch.get(str(epoch_id), 0) or 0) + 1
        else:
            requirements = {}
        selectable = [item for item in evaluated if bool(item.get("trajectory_feasible", True))]
        if not selectable:
            raise RuntimeError("Candidate guards removed the current active prompt fallback")
        beam_size = max(1, int(self.cfg.beam_size))
        if self._is_stable_qd_lineage():
            selected = list(agent.prompt_beam)
        elif self._is_v82_hybrid():
            selected, pareto_summary = self._select_hybrid_beam(
                selectable, beam_size, agent.current_prompt,
                agent_id=agent_id, epoch_id=epoch_id, step_id=step_id,
            )
        elif self._uses_vote_pareto_selection():
            selected, pareto_summary = self._select_vote_pareto_beam(selectable, beam_size, agent.current_prompt)
        else:
            selectable.sort(key=lambda x: float(x.get("reward", 0.0)), reverse=True)
            selected = selectable[:beam_size]
            for item in evaluated:
                item["pareto_feasible"] = None
                item["pareto_rank"] = None
                item["pareto_crowding_distance"] = None
                item["pareto_selected"] = None
                item["pareto_forced_fallback"] = None
        top1_candidate_source = self._candidate_generation_source(selected[0]) if selected else ""
        top1_candidate_pool_source = self._candidate_pool_source(selected[0]) if selected else ""
        selected_by_id = {str(item.get("candidate_id", "")): rank for rank, item in enumerate(selected, start=1)}
        active_candidate_id = str(selected[0].get("candidate_id", "")) if selected else ""
        for item in selected:
            item.setdefault("metrics", {})["beam_slot"] = str(item.get("beam_slot", ""))
        if not self._is_stable_qd_lineage():
            agent.prompt_beam = [
                self._make_beam_item(
                    prompt=str(x["prompt"]),
                    score=float(x.get("reward", 0.0)),
                    metrics=x.get("metrics", {}),
                    parent_id=x.get("parent_id"),
                    generation=int(x.get("generation", generation) or generation),
                    candidate_id=str(x.get("candidate_id", "")) or None,
                )
                for x in selected
            ] or [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]
            agent.current_prompt = str(agent.prompt_beam[0]["prompt"])
        changed = old_hash != self._hash(agent.current_prompt)
        profile_before = dict(agent.capability_profile)
        if changed:
            agent.history.append(agent.current_prompt)
            agent.accept_count += 1
            if self._v7_residual_protocol_enabled():
                active_metrics = selected[0].get("metrics", {}) if selected else {}
                if self._residual_specialization_enabled():
                    self._update_vote_context_profile(agent, active_metrics)
                    self._accumulate_capability_evidence(agent, active_metrics, epoch_id)
                    self._flush_capability_profile(agent, epoch_id, force=False)
                agent.last_accepted_prompt_hash = self._normalized_prompt_hash(agent.current_prompt)
                fingerprint = {
                    str(key): BehaviorFingerprintEntry.from_dict(value)
                    for key, value in dict(active_metrics.get("behavior_fingerprint", {})).items()
                    if isinstance(value, dict)
                }
                state = BehaviorStateSummary(
                    state_id=f"e{int(epoch_id)}_s{int(step_id)}_a{int(agent_id)}_{str(selected[0].get('candidate_id', ''))}",
                    epoch=int(epoch_id),
                    prompt_hash=agent.last_accepted_prompt_hash,
                    behavior_fingerprint=fingerprint,
                    transition_vector={str(key): float(value) for key, value in dict(active_metrics.get("candidate_transition_vector", {})).items()},
                    target_accuracy=float(active_metrics.get("candidate_target_accuracy", 0.0) or 0.0),
                    team_vote_accuracy=float(active_metrics.get("candidate_team_accuracy", 0.0) or 0.0),
                    mean_vote_margin=float(active_metrics.get("candidate_mean_vote_margin", 0.0) or 0.0),
                    preserved_mechanisms=[str(value) for value in selected[0].get("proposal", {}).get("preserved_mechanisms", [])] if isinstance(selected[0].get("proposal", {}).get("preserved_mechanisms", []), list) else [],
                    capability_profile=dict(agent.capability_profile),
                    paired_behavior_utility=self.behavior_fingerprint_utility(fingerprint),
                )
                self._append_bounded_archive(agent.accepted_behavior_archive, state)
        else:
            agent.reject_count += 1

        for item in evaluated:
            metrics = item.get("metrics", {})
            candidate_id = str(item.get("candidate_id", ""))
            rank = selected_by_id.get(candidate_id)
            accepted = rank is not None
            in_top_beam = bool(accepted)
            is_top1 = bool(candidate_id == active_candidate_id)
            active_evolution = bool(is_top1 and changed)
            if self._v7_residual_protocol_enabled() and self._candidate_pool_source(item) == "optimizer":
                rejection_reason = str(metrics.get("rejection_reason", ""))
                retained_inactive = bool(in_top_beam and not active_evolution)
                if not active_evolution and not retained_inactive:
                    if not rejection_reason:
                        rejection_reason = "not_selected"
                        metrics["rejection_reason"] = rejection_reason
                    rejected_state = RejectedBehaviorSummary(
                        state_id=f"e{int(epoch_id)}_s{int(step_id)}_a{int(agent_id)}_{candidate_id}",
                        epoch=int(epoch_id),
                        prompt_hash=str(metrics.get("prompt_hash", self._normalized_prompt_hash(str(item.get("prompt", ""))))),
                        parent_prompt_hash=str(metrics.get("parent_prompt_hash", "")),
                        rejection_reason=rejection_reason,
                        prompt_change_ratio=float(metrics.get("prompt_change_ratio", 0.0) or 0.0),
                        max_behavior_cycle_similarity=float(metrics.get("max_behavior_cycle_similarity", 0.0) or 0.0),
                        behavior_cycle_overlap=int(metrics.get("behavior_cycle_overlap", 0) or 0),
                        transition_vector={str(key): float(value) for key, value in dict(metrics.get("candidate_transition_vector", {})).items()},
                        behavior_fingerprint={
                            str(key): BehaviorFingerprintEntry.from_dict(value)
                            for key, value in dict(metrics.get("behavior_fingerprint", {})).items()
                            if isinstance(value, dict)
                        },
                        paired_behavior_utility=self.behavior_fingerprint_utility(metrics.get("behavior_fingerprint", {})),
                        failure_signature=(
                            f"{rejection_reason}|pivotal_loss={float(metrics.get('pivotal_loss_rate', 0.0) or 0.0):.4f}"
                            f"|shared_creation={float(metrics.get('shared_error_creation_score', 0.0) or 0.0):.4f}"
                        ),
                    )
                    self._append_bounded_archive(agent.rejected_behavior_archive, rejected_state)
                    if rejection_reason == "exact_prompt_cycle":
                        agent.duplicate_prompt_reject_count += 1
                    elif rejection_reason in {"behavior_cycle", "accepted_state_cycle", "rejected_failure_cycle"}:
                        agent.cycle_reject_count += 1
                    elif rejection_reason == "unsupported_large_prompt_shift":
                        agent.large_shift_reject_count += 1
                self.trajectory_events.append(self._trajectory_event(
                    agent_id=agent_id,
                    epoch_id=epoch_id,
                    step_id=step_id,
                    item=item,
                    accepted=active_evolution,
                    profile_before=profile_before,
                    profile_after=dict(agent.capability_profile),
                ))
                if retained_inactive:
                    self.trajectory_events[-1]["decision"] = "retained_beam_inactive"
            active_selection_key = list(
                self._competence_depth_sort_key(item)
                if self._uses_competence_depth_pareto_selection()
                else self._vote_pareto_active_sort_key(item)
            ) if self._uses_vote_pareto_selection() and accepted else None
            item_diagnostics = self._empty_optimizer_generation_diagnostics()
            if isinstance(item.get("optimizer_generation_diagnostics", {}), dict):
                item_diagnostics.update(item.get("optimizer_generation_diagnostics", {}))
            item_diagnostics["optimizer_underfilled"] = bool(optimizer_underfilled)
            tcs_candidate_metadata = {
                "optimizer_architecture": item_diagnostics.get("optimizer_architecture", ""),
                "candidate_source": self._candidate_generation_source(item),
                "candidate_pool_source": self._candidate_pool_source(item),
                "tcs_call_group_id": item.get("tcs_call_group_id", item_diagnostics.get("tcs_call_group_id", "")),
                "execution_session_id": item.get("execution_session_id", item_diagnostics.get("execution_session_id", self._current_execution_session_id())),
                "update_attempt_id": item.get("update_attempt_id", item_diagnostics.get("update_attempt_id", update_attempt_id)),
                **item_diagnostics,
            }
            is_tcs_metadata_applicable = tcs_metadata_applicable(tcs_candidate_metadata)
            tcs_metadata_errors = validate_tcs_candidate_metadata(tcs_candidate_metadata)
            self.update_logs.append(
                {
                    **self._base_log_fields(),
                    "event": "candidate_evaluated",
                    "epoch": epoch_id,
                    "step": step_id,
                    "agent_id": agent_id,
                    "search_mode": "evolutionary_beam",
                    "beam_size": beam_size,
                    "candidate_id": item.get("candidate_id", ""),
                    "candidate_selection_mode": str(getattr(self.cfg, "candidate_selection_mode", "scalar_reward")),
                    "parent_id": item.get("parent_id"),
                    "tcs_call_group_id": str(item.get("tcs_call_group_id", item_diagnostics.get("tcs_call_group_id", "")) or ""),
                    "execution_session_id": str(item.get("execution_session_id", item_diagnostics.get("execution_session_id", self._current_execution_session_id())) or self._current_execution_session_id()),
                    "update_attempt_id": str(item.get("update_attempt_id", item_diagnostics.get("update_attempt_id", update_attempt_id)) or update_attempt_id),
                    "reward": float(metrics.get("reward", 0.0)),
                    "reward_total": float(metrics.get("reward_total", metrics.get("reward", 0.0))),
                    "embedding_diversity": float(metrics.get("embedding_diversity", 0.0)),
                    "mean_embedding_overlap": float(metrics.get("mean_embedding_overlap", 0.0)),
                    "target_overlap_pressure": float(metrics.get("target_overlap_pressure", 0.0)),
                    "homogeneous_case_count": float(metrics.get("homogeneous_case_count", 0.0)),
                    "resolved_case_count": float(metrics.get("resolved_case_count", 0.0)),
                    "new_homogeneous_case_count": float(metrics.get("new_homogeneous_case_count", 0.0)),
                    "team_accuracy": float(metrics.get("team_accuracy", 0.0)),
                    "target_agent_accuracy": float(metrics.get("target_agent_accuracy", 0.0)),
                    "invalid_rate": float(metrics.get("invalid_rate", 0.0)),
                    "invalid_score": float(metrics.get("invalid_score", 0.0)),
                    "baseline_team_accuracy": float(metrics.get("baseline_team_accuracy", 0.0)),
                    "candidate_team_accuracy": float(metrics.get("candidate_team_accuracy", metrics.get("team_accuracy", 0.0))),
                    "accuracy_delta": float(metrics.get("accuracy_delta", 0.0)),
                    "vote_delta": float(metrics.get("vote_delta", metrics.get("accuracy_delta", 0.0))),
                    "vote_gain_count": int(metrics.get("vote_gain_count", 0)),
                    "vote_gain_rate": float(metrics.get("vote_gain_rate", 0.0)),
                    "vote_loss_count": int(metrics.get("vote_loss_count", 0)),
                    "vote_loss_rate": float(metrics.get("vote_loss_rate", 0.0)),
                    "net_vote_count": int(metrics.get("net_vote_count", 0)),
                    "net_vote_delta": float(metrics.get("net_vote_delta", metrics.get("vote_delta", 0.0))),
                    "plurality_vote_gain_count": int(metrics.get("plurality_vote_gain_count", metrics.get("vote_gain_count", 0))),
                    "plurality_vote_gain_rate": float(metrics.get("plurality_vote_gain_rate", metrics.get("vote_gain_rate", 0.0))),
                    "plurality_vote_loss_count": int(metrics.get("plurality_vote_loss_count", metrics.get("vote_loss_count", 0))),
                    "plurality_vote_loss_rate": float(metrics.get("plurality_vote_loss_rate", metrics.get("vote_loss_rate", 0.0))),
                    "plurality_vote_net_count": int(metrics.get("plurality_vote_net_count", metrics.get("net_vote_count", 0))),
                    "plurality_vote_net_delta": float(metrics.get("plurality_vote_net_delta", metrics.get("net_vote_delta", 0.0))),
                    "plurality_pivotal_fix_opportunity_count": int(metrics.get("plurality_pivotal_fix_opportunity_count", 0)),
                    "plurality_pivotal_fix_opportunity_rate": float(metrics.get("plurality_pivotal_fix_opportunity_rate", 0.0)),
                    "plurality_pivotal_fix_count": int(metrics.get("plurality_pivotal_fix_count", 0)),
                    "plurality_pivotal_fix_rate": float(metrics.get("plurality_pivotal_fix_rate", 0.0)),
                    "plurality_pivotal_loss_count": int(metrics.get("plurality_pivotal_loss_count", 0)),
                    "plurality_pivotal_loss_rate": float(metrics.get("plurality_pivotal_loss_rate", 0.0)),
                    "plurality_boundary_shared_error_net_gain": float(metrics.get("plurality_boundary_shared_error_net_gain", 0.0)),
                    "pivotal_definition": str(metrics.get("pivotal_definition", "")),
                    "baseline_gold_vote_count": float(metrics.get("baseline_gold_vote_count", 0.0)),
                    "candidate_gold_vote_count": float(metrics.get("candidate_gold_vote_count", 0.0)),
                    "baseline_largest_wrong_vote_count": float(metrics.get("baseline_largest_wrong_vote_count", 0.0)),
                    "candidate_largest_wrong_vote_count": float(metrics.get("candidate_largest_wrong_vote_count", 0.0)),
                    "baseline_plurality_margin_votes": float(metrics.get("baseline_plurality_margin_votes", 0.0)),
                    "candidate_plurality_margin_votes": float(metrics.get("candidate_plurality_margin_votes", 0.0)),
                    "plurality_margin_vote_delta": float(metrics.get("plurality_margin_vote_delta", 0.0)),
                    "baseline_normalized_plurality_margin": float(metrics.get("baseline_normalized_plurality_margin", -1.0)),
                    "candidate_normalized_plurality_margin": float(metrics.get("candidate_normalized_plurality_margin", -1.0)),
                    "normalized_plurality_margin_delta": float(metrics.get("normalized_plurality_margin_delta", 0.0)),
                    "baseline_plurality_vote_tie": float(metrics.get("baseline_plurality_vote_tie", 0.0)),
                    "candidate_plurality_vote_tie": float(metrics.get("candidate_plurality_vote_tie", 0.0)),
                    "baseline_mean_vote_margin": float(metrics.get("baseline_mean_vote_margin", -1.0)),
                    "candidate_mean_vote_margin": float(metrics.get("candidate_mean_vote_margin", -1.0)),
                    "vote_margin_delta": float(metrics.get("vote_margin_delta", 0.0)),
                    "baseline_boundary_useful_diversity": float(metrics.get("baseline_boundary_useful_diversity", 0.0)),
                    "candidate_boundary_useful_diversity": float(metrics.get("candidate_boundary_useful_diversity", 0.0)),
                    "boundary_useful_diversity_delta": float(metrics.get("boundary_useful_diversity_delta", 0.0)),
                    "boundary_diversity_gain": float(metrics.get("boundary_diversity_gain", 0.0)),
                    "reward_component_target_accuracy": float(metrics.get("reward_component_target_accuracy", 0.0)),
                    "reward_component_vote_delta": float(metrics.get("reward_component_vote_delta", 0.0)),
                    "reward_component_vote_margin": float(metrics.get("reward_component_vote_margin", 0.0)),
                    "reward_component_boundary_diversity": float(metrics.get("reward_component_boundary_diversity", 0.0)),
                    "reward_component_invalid_penalty": float(metrics.get("reward_component_invalid_penalty", 0.0)),
                    "reward_component_guard_penalty": float(metrics.get("reward_component_guard_penalty", 0.0)),
                    "baseline_oracle_acc": float(metrics.get("baseline_oracle_acc", 0.0)),
                    "candidate_oracle_acc": float(metrics.get("candidate_oracle_acc", 0.0)),
                    "coverage_delta": float(metrics.get("coverage_delta", 0.0)),
                    "coverage_gain_count": int(metrics.get("coverage_gain_count", 0)),
                    "coverage_gain_rate": float(metrics.get("coverage_gain_rate", 0.0)),
                    "coverage_loss_count": int(metrics.get("coverage_loss_count", 0)),
                    "coverage_loss_rate": float(metrics.get("coverage_loss_rate", 0.0)),
                    "net_coverage_count": int(metrics.get("net_coverage_count", 0)),
                    "net_coverage_delta": float(metrics.get("net_coverage_delta", 0.0)),
                    **{
                        key: metrics.get(key, 0)
                        for depth in range(1, 4)
                        for key in (
                            f"baseline_coverage_depth_c{depth}", f"candidate_coverage_depth_c{depth}",
                            f"depth{depth}_gain_count", f"depth{depth}_gain_rate",
                            f"depth{depth}_loss_count", f"depth{depth}_loss_rate",
                            f"depth{depth}_net_count", f"depth{depth}_net_delta",
                        )
                    },
                    "competence_reward_component": float(metrics.get("competence_reward_component", 0.0)),
                    "v7_reward_component": float(metrics.get("v7_reward_component", 0.0)),
                    "effective_reward_specialization_strength": float(metrics.get("effective_reward_specialization_strength", 0.0)),
                    "final_reward": float(metrics.get("final_reward", metrics.get("reward", 0.0))),
                    "stage_aux_objective": float(metrics.get("stage_aux_objective", 0.0)),
                    "stage_aux_depth2_component": float(metrics.get("stage_aux_depth2_component", 0.0)),
                    "stage_aux_boundary_component": float(metrics.get("stage_aux_boundary_component", 0.0)),
                    "baseline_target_accuracy": float(metrics.get("baseline_target_accuracy", 0.0)),
                    "candidate_target_accuracy": float(metrics.get("candidate_target_accuracy", metrics.get("target_agent_accuracy", 0.0))),
                    "rescue_rate": float(metrics.get("rescue_rate", 0.0)),
                    "useful_diversity": float(metrics.get("useful_diversity", 0.0)),
                    "rescue_useful_diversity": float(metrics.get("rescue_useful_diversity", 0.0)),
                    "baseline_embedding_diversity": float(metrics.get("baseline_embedding_diversity", 0.0)),
                    "candidate_embedding_diversity": float(metrics.get("candidate_embedding_diversity", metrics.get("embedding_diversity", 0.0))),
                    "diversity_delta": float(metrics.get("diversity_delta", 0.0)),
                    "baseline_invalid_rate": float(metrics.get("baseline_invalid_rate", 0.0)),
                    "candidate_invalid_rate": float(metrics.get("candidate_invalid_rate", metrics.get("invalid_rate", 0.0))),
                    "invalid_delta": float(metrics.get("invalid_delta", 0.0)),
                    "behavior_context_counts": metrics.get("behavior_context_counts", {}),
                    "candidate_transition_vector": metrics.get("candidate_transition_vector", {}),
                    "candidate_transition_support": metrics.get("candidate_transition_support", {}),
                    **self._candidate_v7_log_fields(metrics),
                    "capability_profile_before": profile_before,
                    "capability_profile_after": dict(agent.capability_profile),
                    "prompt_hash": str(metrics.get("prompt_hash", "")),
                    "parent_prompt_hash": str(metrics.get("parent_prompt_hash", "")),
                    "prompt_change_ratio": float(metrics.get("prompt_change_ratio", 0.0) or 0.0),
                    "max_behavior_cycle_similarity": float(metrics.get("max_behavior_cycle_similarity", 0.0) or 0.0),
                    "behavior_cycle_overlap": int(metrics.get("behavior_cycle_overlap", 0) or 0),
                    "matched_behavior_state_id": str(metrics.get("matched_behavior_state_id", "")),
                    "exact_prompt_cycle": bool(metrics.get("exact_prompt_cycle", False)),
                    "behavior_cycle_guard_passed": bool(metrics.get("behavior_cycle_guard_passed", True)),
                    "prompt_trust_region_passed": bool(metrics.get("prompt_trust_region_passed", True)),
                    "rejection_reason": str(metrics.get("rejection_reason", "")),
                    "accuracy_guard_passed": bool(metrics.get("accuracy_guard_passed", True)),
                    "invalid_guard_passed": bool(metrics.get("invalid_guard_passed", True)),
                    "competence_depth1_guard_enabled": bool(metrics.get("competence_depth1_guard_enabled", candidate_guard_enabled)),
                    "competence_depth1_guard_epsilon": float(metrics.get("competence_depth1_guard_epsilon", 0.0) or 0.0),
                    "competence_depth1_guard_passed": bool(metrics.get("competence_depth1_guard_passed", True)),
                    "hard_guard_passed": bool(metrics.get("hard_guard_passed", True)),
                    "hard_rejection_reason": str(metrics.get("rejection_reason", "")),
                    "candidate_type": str(metrics.get("candidate_type", "")),
                    "archive_bucket": str(item.get("archive_bucket", "")),
                    "cheap_prescreen_reasons": list(item.get("cheap_prescreen_reasons", [])),
                    "refill_candidate": bool(item.get("refill_candidate", False)),
                    "mechanism_signature": metrics.get("mechanism_signature", []),
                    "parent_mechanism_signature": metrics.get("parent_mechanism_signature", []),
                    "peer_dominant_mechanism_signature": metrics.get("peer_dominant_mechanism_signature", []),
                    "mechanism_signature_distance": float(metrics.get("mechanism_signature_distance", 0.0) or 0.0),
                    "raw_reward": float(metrics.get("raw_reward", metrics.get("reward", 0.0)) or 0.0),
                    "penalized_reward": float(metrics.get("penalized_reward", metrics.get("reward", 0.0)) or 0.0),
                    "soft_guard_penalty": float(metrics.get("soft_guard_penalty", 0.0) or 0.0),
                    "soft_error_dependence_penalty": float(metrics.get("soft_error_dependence_penalty", 0.0) or 0.0),
                    "soft_cycle_penalty": float(metrics.get("soft_cycle_penalty", 0.0) or 0.0),
                    "soft_mechanism_shift_penalty": float(metrics.get("soft_mechanism_shift_penalty", 0.0) or 0.0),
                    "soft_accuracy_regression_penalty": float(metrics.get("soft_accuracy_regression_penalty", 0.0) or 0.0),
                    "soft_guard_reasons": metrics.get("soft_guard_reasons", []),
                    "beam_slot": str(item.get("beam_slot", "not_retained")),
                    "pareto_feasible": item.get("pareto_feasible"),
                    "pareto_rank": item.get("pareto_rank"),
                    "pareto_crowding_distance": item.get("pareto_crowding_distance"),
                    "pareto_selected": item.get("pareto_selected"),
                    "active_selection_key": active_selection_key,
                    "effective_weight_target_accuracy": float(metrics.get("effective_weight_target_accuracy", 0.0)),
                    "effective_weight_div_delta": float(metrics.get("effective_weight_div_delta", 0.0)),
                    "effective_weight_vote_delta": float(metrics.get("effective_weight_vote_delta", 0.0)),
                    "effective_weight_vote_margin": float(metrics.get("effective_weight_vote_margin", 0.0)),
                    "effective_weight_boundary_diversity": float(metrics.get("effective_weight_boundary_diversity", 0.0)),
                    "effective_accuracy_guard_epsilon": float(metrics.get("effective_accuracy_guard_epsilon", 0.0)),
                    "reward_phase_progress": float(metrics.get("reward_phase_progress", 0.0)),
                    "reward_diversity_need": float(metrics.get("reward_diversity_need", 0.0)),
                    "reward_unique_prompt_ratio": float(metrics.get("reward_unique_prompt_ratio", 0.0)),
                    "reward_accepted_updates": float(metrics.get("reward_accepted_updates", 0.0)),
                    "solver_reuse_enabled": bool(metrics.get("solver_reuse_enabled", False)),
                    "solver_reuse_hits": int(metrics.get("solver_reuse_hits", 0)),
                    "solver_reuse_misses": int(metrics.get("solver_reuse_misses", 0)),
                    "solver_calls": int(metrics.get("solver_calls", 0)),
                    "solver_reuse_total": int(metrics.get("solver_reuse_total", 0)),
                    "solver_reuse_hit_rate": float(metrics.get("solver_reuse_hit_rate", 0.0)),
                    "accepted": bool(accepted),
                    "in_top_beam": bool(in_top_beam),
                    "is_top1": bool(is_top1),
                    "active_prompt_changed": bool(changed),
                    "top1_candidate_source": top1_candidate_source,
                    "top1_candidate_pool_source": top1_candidate_pool_source,
                    "rank_in_beam": rank,
                    "beam_rank": rank,
                    "prompt_preview": normalize_spaces(str(item.get("prompt", "")))[:220],
                    "optimizer_model": self.cfg.optimizer_model,
                    "evaluator_model": self.cfg.evaluator_model,
                    "candidate_source": self._candidate_generation_source(item),
                    "candidate_pool_source": self._candidate_pool_source(item),
                    "generation_batch_type": item.get("generation_batch_type", ""),
                    "generation_case_ids": item.get("generation_case_ids", []),
                    "target_error_pattern": item.get("target_error_pattern", ""),
                    "accuracy_repair_rule": item.get("accuracy_repair_rule", ""),
                    "expected_accuracy_effect": item.get("expected_accuracy_effect", ""),
                    "num_target_error_cases": int(num_target_error_cases),
                    "num_accuracy_repair_candidates": int(num_accuracy_repair_candidates),
                    "num_diversity_candidates": int(num_diversity_candidates),
                    "optimizer_fallback_mode": str(getattr(self.cfg, "optimizer_fallback_mode", "none")),
                    "optimizer_parent_concurrency": int(parent_concurrency),
                    "fallback_enabled": bool(fallback_enabled),
                    "optimizer_underfilled": bool(optimizer_underfilled),
                    "requested_optimizer_candidates": int(requested_optimizer_candidates),
                    "num_optimizer_candidates": int(num_optimizer_candidates),
                    "num_fallback_candidates": int(num_fallback_candidates),
                    "num_existing_beam_candidates": int(num_existing_beam_candidates),
                    "optimizer_architecture": str(item_diagnostics.get("optimizer_architecture", getattr(self.cfg, "optimizer_architecture", "one_shot"))),
                    "teacher_question": item_diagnostics.get("teacher_question", ""),
                    "teacher_question_approved": bool(item_diagnostics.get("teacher_question_approved", False)),
                    "teacher_question_forced_best_score": bool(item_diagnostics.get("teacher_question_forced_best_score", False)),
                    "teacher_question_forced_best_round": int(item_diagnostics.get("teacher_question_forced_best_round", 0) or 0),
                    "teacher_question_forced_below_threshold": bool(item_diagnostics.get("teacher_question_forced_below_threshold", False)),
                    "teacher_question_score": self._safe_float(item_diagnostics.get("teacher_question_score", 0.0), 0.0),
                    "teacher_critic_rounds": int(item_diagnostics.get("teacher_critic_rounds", 0) or 0),
                    "teacher_quality_critique": str(item_diagnostics.get("teacher_quality_critique", "")),
                    "teacher_specificity_critique": str(item_diagnostics.get("teacher_specificity_critique", "")),
                    "teacher_task_alignment_critique": str(item_diagnostics.get("teacher_task_alignment_critique", "")),
                    "teacher_error_alignment_critique": str(item_diagnostics.get("teacher_error_alignment_critique", "")),
                    "teacher_diversity_critique": str(item_diagnostics.get("teacher_diversity_critique", "")),
                    "teacher_rewrite_count": int(item_diagnostics.get("teacher_rewrite_count", 0) or 0),
                    "student_candidate_count_raw": int(item_diagnostics.get("student_candidate_count_raw", 0) or 0),
                    "student_candidate_count_final": int(item_diagnostics.get("student_candidate_count_final", 0) or 0),
                    "student_candidate_filtered_count": int(item_diagnostics.get("student_candidate_filtered_count", 0) or 0),
                    "student_candidate_filter_reasons": item_diagnostics.get("student_candidate_filter_reasons", []),
                    "student_all_candidates_filtered": bool(item_diagnostics.get("student_all_candidates_filtered", False)),
                    "student_missing_required_field_count": int(item_diagnostics.get("student_missing_required_field_count", 0) or 0),
                    "student_missing_required_fields": item_diagnostics.get("student_missing_required_fields", []),
                    **self._student_failure_log_fields(item_diagnostics),
                    "tcs_metadata_applicable": is_tcs_metadata_applicable,
                    "tcs_metadata_valid": (not tcs_metadata_errors) if is_tcs_metadata_applicable else None,
                    "tcs_metadata_errors": tcs_metadata_errors,
                    "diversity_contribution": str(item.get("diversity_contribution", "")),
                    "error_correlation_reduction": str(item.get("error_correlation_reduction", "")),
                    "task_alignment_rule": str(item.get("task_alignment_rule", "")),
                    "peer_redundancy_avoidance": str(item.get("peer_redundancy_avoidance", "")),
                    "declared_mechanism": str(item.get("proposal", {}).get("modified_mechanism", item.get("proposal", {}).get("new_or_modified_mechanism", item.get("proposal", {}).get("mechanism_name", "")))) if isinstance(item.get("proposal", {}), dict) else "",
                    "candidate_prompt_char_count": int(item.get("candidate_prompt_char_count", len(str(item.get("prompt", "")))) or 0),
                    "candidate_prompt_over_soft_limit": bool(item.get("candidate_prompt_over_soft_limit", False)),
                    "candidate_prompt_over_hard_limit": bool(item.get("candidate_prompt_over_hard_limit", False)),
                    "candidate_prompt_overlength_rejected": bool(item.get("candidate_prompt_overlength_rejected", False)),
                    "candidate_prompt_ends_with_sentence_boundary": bool(item.get("candidate_prompt_ends_with_sentence_boundary", self._prompt_ends_with_sentence_boundary(str(item.get("prompt", ""))))),
                    "optimizer_raw_response_empty": int(item_diagnostics.get("optimizer_raw_response_empty", 0) or 0),
                    "optimizer_json_parse_failed": int(item_diagnostics.get("optimizer_json_parse_failed", 0) or 0),
                    "optimizer_raw_candidate_count": int(item_diagnostics.get("optimizer_raw_candidate_count", 0) or 0),
                    "optimizer_empty_prompt_count": int(item_diagnostics.get("optimizer_empty_prompt_count", 0) or 0),
                    "optimizer_sanitized_count": int(item_diagnostics.get("optimizer_sanitized_count", 0) or 0),
                    "optimizer_redundant_filtered_count": int(item_diagnostics.get("optimizer_redundant_filtered_count", 0) or 0),
                    "optimizer_schema_filtered_count": int(item_diagnostics.get("optimizer_schema_filtered_count", 0) or 0),
                    "optimizer_final_candidate_count": int(item_diagnostics.get("optimizer_final_candidate_count", 0) or 0),
                    "num_eval_samples": int(metrics.get("num_eval_samples", 0)),
                    "candidate_eval_strategy": str(metrics.get("candidate_eval_strategy", getattr(self.cfg, "candidate_eval_strategy", "random"))),
                    "candidate_eval_pool_size": int(metrics.get("candidate_eval_pool_size", getattr(self.cfg, "candidate_eval_pool_size", 0))),
                    "candidate_eval_pool_actual_size": int(metrics.get("candidate_eval_pool_actual_size", getattr(self.cfg, "candidate_eval_pool_actual_size", 0))),
                    "candidate_eval_batch_size": int(metrics.get("candidate_eval_batch_size", getattr(self.cfg, "candidate_eval_batch_size", 0))),
                    "actual_eval_batch_size": int(metrics.get("actual_eval_batch_size", metrics.get("num_eval_samples", 0))),
                    "num_eval_repeats": int(metrics.get("num_eval_repeats", getattr(self.cfg, "candidate_eval_repeats", 1))),
                    "candidate_eval_data_source": str(metrics.get("candidate_eval_data_source", getattr(self.cfg, "candidate_eval_data_source", "optimization_train"))),
                    "candidate_eval_total_count": int(metrics.get("candidate_eval_total_count", metrics.get("actual_eval_batch_size", 0))),
                    "candidate_eval_unique_question_count": int(metrics.get("candidate_eval_unique_question_count", metrics.get("actual_eval_batch_size", 0))),
                    "candidate_eval_repeat_count": int(metrics.get("candidate_eval_repeat_count", getattr(self.cfg, "candidate_eval_repeats", 1))),
                    **competence_log_fields,
                }
            )
        self._append_prompt_history_event(agent_id, epoch_id, step_id, "beam_accept" if changed else "beam_keep", changed)
        if bool(getattr(self.cfg, "candidate_eval_cache_logging", True)):
            if not hasattr(self, "cost_summary"):
                self.cost_summary = self._empty_cost_summary()
            self.cost_summary["candidate_eval_solver_api_calls"] = int(self.cost_summary.get("candidate_eval_solver_api_calls", 0) or 0) + int(candidate_eval_cache_stats.get("candidate_eval_solver_api_call_count", 0) or 0)
            self.cost_summary["candidate_eval_cache_hits"] = int(self.cost_summary.get("candidate_eval_cache_hits", 0) or 0) + int(candidate_eval_cache_stats.get("candidate_eval_memory_cache_hit_count", 0) or 0) + int(candidate_eval_cache_stats.get("candidate_eval_persisted_cache_hit_count", 0) or 0)
            self.cost_summary["candidate_eval_inflight_reuses"] = int(self.cost_summary.get("candidate_eval_inflight_reuses", 0) or 0) + int(candidate_eval_cache_stats.get("candidate_eval_inflight_reuse_count", 0) or 0)
            self.cost_summary["candidate_eval_calls_saved_vs_naive"] = int(self.cost_summary.get("candidate_eval_calls_saved_vs_naive", 0) or 0) + int(candidate_eval_cache_stats.get("candidate_eval_calls_saved_vs_naive", 0) or 0)
            self.cost_summary["candidate_eval_prompt_dedup_savings"] = int(self.cost_summary.get("candidate_eval_prompt_dedup_savings", 0) or 0) + int(candidate_eval_cache_stats.get("candidate_eval_prompt_dedup_savings", 0) or 0)
        summary = {
            "agent_id": agent_id,
            "execution_session_id": self._current_execution_session_id(),
            "update_attempt_id": update_attempt_id,
            **competence_log_fields,
            "updated": bool(changed),
            "candidate_count": len(candidate_pool),
            "depth1_guard_rejection_count": sum(str(item.get("metrics", {}).get("rejection_reason", "")) == "competence_depth1_guard" for item in evaluated),
            "accuracy_guard_rejection_count": sum(not bool(item.get("metrics", {}).get("accuracy_guard_passed", True)) for item in evaluated),
            "invalid_guard_rejection_count": sum(not bool(item.get("metrics", {}).get("invalid_guard_passed", True)) for item in evaluated),
            "dependence_guard_rejection_count": sum(str(item.get("metrics", {}).get("rejection_reason", "")) in {"pivotal_loss_guard", "shared_error_creation_guard"} for item in evaluated),
            "pareto_not_retained_count": sum(not bool(item.get("pareto_selected", False)) for item in evaluated),
            "retained_candidate_count": len(selected),
            "active_prompt_changed_count": int(changed),
            "catastrophic_accuracy_guard_rejection_count": sum(not bool(item.get("metrics", {}).get("accuracy_guard_passed", True)) for item in evaluated),
            "soft_error_dependence_penalty_count": sum(float(item.get("metrics", {}).get("soft_error_dependence_penalty", 0.0) or 0.0) > 0.0 for item in evaluated),
            "soft_cycle_penalty_count": sum(float(item.get("metrics", {}).get("soft_cycle_penalty", 0.0) or 0.0) > 0.0 for item in evaluated),
            "soft_mechanism_shift_penalty_count": sum(float(item.get("metrics", {}).get("soft_mechanism_shift_penalty", 0.0) or 0.0) > 0.0 for item in evaluated),
            "exploration_candidate_count": sum(self._candidate_pool_source(item) == "optimizer" and float(item.get("metrics", {}).get("mechanism_signature_distance", 0.0) or 0.0) > 0.0 for item in evaluated),
            "exploration_slot_occupancy_count": sum(
                str(item.get("beam_slot", "")) == ("mechanism_niche" if self._is_stable_qd_lineage() else "explore")
                for item in selected
            ),
            "exploration_to_active_conversion_count": int(bool(selected and selected[0].get("beam_slot") == "explore" and changed)),
            "generation_batches": generation_batches,
            "baseline_homogeneous_case_count": len(baseline_cases),
            "num_target_error_cases": int(num_target_error_cases),
            "num_accuracy_repair_candidates": int(num_accuracy_repair_candidates),
            "num_diversity_candidates": int(num_diversity_candidates),
            "optimizer_fallback_mode": str(getattr(self.cfg, "optimizer_fallback_mode", "none")),
            "optimizer_parent_concurrency": int(parent_concurrency),
            "parent_sources": list(parent_sources),
            "per_niche_parent_count": dict(agent.per_niche_parent_count),
            "probation_parent_count": int(agent.probation_parent_count),
            "probation_to_safe_conversion_count": int(getattr(self, "probation_to_safe_conversion_count", 0)),
            "candidate_starvation": bool(requirements.get("safe_non_incumbent_count", 1) == 0) if self._is_stable_qd_lineage() else False,
            "mechanism_starvation": bool(requirements.get("safe_distinct_mechanism_count", 1) == 0) if self._is_stable_qd_lineage() else False,
            "search_branch_starvation": bool(
                requirements.get("safe_non_incumbent_count", 1) == 0 and not getattr(agent, "probation_archive", [])
            ) if self._is_stable_qd_lineage() else False,
            "candidate_starvation_count": int(getattr(self, "candidate_starvation_count", 0)),
            "mechanism_starvation_count": int(getattr(self, "mechanism_starvation_count", 0)),
            "search_branch_starvation_count": int(getattr(self, "search_branch_starvation_count", 0)),
            "refill_requirements_unmet_count": int(getattr(self, "refill_requirements_unmet_count", 0)),
            "fallback_enabled": bool(fallback_enabled),
            "optimizer_underfilled": bool(optimizer_underfilled),
            "requested_optimizer_candidates": int(requested_optimizer_candidates),
            "num_optimizer_candidates": int(num_optimizer_candidates),
            "num_fallback_candidates": int(num_fallback_candidates),
            "num_existing_beam_candidates": int(num_existing_beam_candidates),
            "num_tcs_optimizer_candidates": int(num_tcs_optimizer_candidates),
            "num_tcs_metadata_valid_candidates": int(num_tcs_metadata_valid_candidates),
            "num_tcs_metadata_invalid_candidates": int(num_tcs_metadata_invalid_candidates),
            "tcs_execution_complete": tcs_execution_complete,
            "tcs_call_group_ids": sorted({str(c.get("tcs_call_group_id", "")) for c in candidate_pool if str(c.get("tcs_call_group_id", ""))}),
            "top1_candidate_source": top1_candidate_source,
            "top1_candidate_pool_source": top1_candidate_pool_source,
            "active_prompt_changed": bool(changed),
            **pareto_summary,
            "top1_pareto_rank": selected[0].get("pareto_rank") if self._uses_vote_pareto_selection() and selected else None,
            "top1_vote_gain_rate": float(selected[0].get("metrics", {}).get("vote_gain_rate", 0.0)) if self._uses_vote_pareto_selection() and selected else None,
            "top1_vote_loss_rate": float(selected[0].get("metrics", {}).get("vote_loss_rate", 0.0)) if self._uses_vote_pareto_selection() and selected else None,
            "top1_vote_delta": float(selected[0].get("metrics", {}).get("vote_delta", 0.0)) if self._uses_vote_pareto_selection() and selected else None,
            **optimizer_generation_summary,
            **self._student_failure_log_fields(optimizer_generation_summary),
            "top_reward": float(agent.prompt_beam[0].get("score", 0.0) or 0.0),
            "top_metrics": agent.prompt_beam[0].get("metrics", {}),
            **candidate_eval_cache_stats,
            "execution_session_id": self._current_execution_session_id(),
            "update_attempt_id": update_attempt_id,
        }
        self.depth1_guard_rejection_count = int(getattr(self, "depth1_guard_rejection_count", 0)) + int(
            summary["depth1_guard_rejection_count"]
        )
        if self._is_stable_qd_lineage():
            self.total_agent_update_count += 1
            self.task_repair_niche_occupancy_count += int(pareto_summary.get("task_repair_niche_occupancy", 0) or 0)
            self.mechanism_niche_occupancy_count += int(pareto_summary.get("mechanism_niche_occupancy", 0) or 0)
        for field in (
            "catastrophic_accuracy_guard_rejection_count",
            "soft_error_dependence_penalty_count",
            "soft_cycle_penalty_count",
            "soft_mechanism_shift_penalty_count",
            "exploration_candidate_count",
            "exploration_slot_occupancy_count",
            "exploration_to_active_conversion_count",
        ):
            setattr(self, field, int(getattr(self, field, 0)) + int(summary.get(field, 0) or 0))
        if self._is_v82_hybrid():
            self.mechanism_signature_history.append({
                "epoch": int(epoch_id),
                "step": int(step_id),
                "agent_id": int(agent_id),
                "retained": [list(item.get("metrics", {}).get("mechanism_signature", [])) for item in selected],
            })
            self.beam_slot_state[str(agent_id)] = [str(item.get("beam_slot", "")) for item in selected]
            self.exploration_slot_candidates = [
                {"agent_id": int(agent_id), "candidate_id": str(item.get("candidate_id", "")), "prompt": str(item.get("prompt", ""))}
                for item in selected if str(item.get("beam_slot", "")) == "explore"
            ]
        self.update_logs.append(
            {
                **self._base_log_fields(),
                "event": "beam_update_summary",
                "epoch": epoch_id,
                "step": step_id,
                "agent_id": agent_id,
                "execution_session_id": self._current_execution_session_id(),
                "update_attempt_id": update_attempt_id,
                **competence_log_fields,
                "search_mode": "evolutionary_beam",
                "beam_size": beam_size,
                "active_prompt_changed": bool(changed),
                "top1_candidate_source": top1_candidate_source,
                "top1_candidate_pool_source": top1_candidate_pool_source,
                "candidate_count": len(candidate_pool),
                "depth1_guard_rejection_count": summary["depth1_guard_rejection_count"],
                "accuracy_guard_rejection_count": summary["accuracy_guard_rejection_count"],
                "invalid_guard_rejection_count": summary["invalid_guard_rejection_count"],
                "dependence_guard_rejection_count": summary["dependence_guard_rejection_count"],
                "pareto_not_retained_count": summary["pareto_not_retained_count"],
                "retained_candidate_count": summary["retained_candidate_count"],
                "active_prompt_changed_count": summary["active_prompt_changed_count"],
                "generation_batches": generation_batches,
                "general_error_case_count": sum(len(batch.get("cases", [])) for batch in generation_batches if str(batch.get("batch_type", "")) == "general_error"),
                "c1_creation_case_count": sum(sum(int(case.get("baseline_correct_count", -1)) == 0 for case in batch.get("cases", [])) for batch in generation_batches if str(batch.get("batch_type", "")) == "c1_c2_creation"),
                "c2_creation_case_count": sum(sum(int(case.get("baseline_correct_count", -1)) == 1 for case in batch.get("cases", [])) for batch in generation_batches if str(batch.get("batch_type", "")) == "c1_c2_creation"),
                "boundary_case_count": sum(len(batch.get("cases", [])) for batch in generation_batches if str(batch.get("batch_type", "")) == "actual_plurality_boundary"),
                "residual_case_count": sum(len(batch.get("cases", [])) for batch in generation_batches if str(batch.get("batch_type", "")) == "residual_shared_error"),
                "catastrophic_accuracy_guard_rejection_count": summary["catastrophic_accuracy_guard_rejection_count"],
                "soft_error_dependence_penalty_count": summary["soft_error_dependence_penalty_count"],
                "soft_cycle_penalty_count": summary["soft_cycle_penalty_count"],
                "soft_mechanism_shift_penalty_count": summary["soft_mechanism_shift_penalty_count"],
                "exploration_candidate_count": summary["exploration_candidate_count"],
                "exploration_slot_occupancy_count": summary["exploration_slot_occupancy_count"],
                "exploration_to_active_conversion_count": summary["exploration_to_active_conversion_count"],
                "optimizer_fallback_mode": str(getattr(self.cfg, "optimizer_fallback_mode", "none")),
                "optimizer_parent_concurrency": int(parent_concurrency),
                "parent_sources": list(parent_sources),
                "per_niche_parent_count": dict(agent.per_niche_parent_count),
                "probation_parent_count": int(agent.probation_parent_count),
                "probation_to_safe_conversion_count": int(getattr(self, "probation_to_safe_conversion_count", 0)),
                "fallback_enabled": bool(fallback_enabled),
                "optimizer_underfilled": bool(optimizer_underfilled),
                "requested_optimizer_candidates": int(requested_optimizer_candidates),
                "num_optimizer_candidates": int(num_optimizer_candidates),
                "num_fallback_candidates": int(num_fallback_candidates),
                "num_existing_beam_candidates": int(num_existing_beam_candidates),
                "num_teacher_calls": int(optimizer_generation_summary.get("num_teacher_calls", 0) or 0),
                "num_critic_calls": int(optimizer_generation_summary.get("num_critic_calls", 0) or 0),
                "num_teacher_rewrite_calls": int(optimizer_generation_summary.get("num_teacher_rewrite_calls", 0) or 0),
                "num_student_calls": int(optimizer_generation_summary.get("num_student_calls", 0) or 0),
                "num_student_retry_calls": int(optimizer_generation_summary.get("num_student_retry_calls", 0) or 0),
                "num_student_repair_calls": int(optimizer_generation_summary.get("num_student_repair_calls", 0) or 0),
                "num_tcs_optimizer_candidates": int(num_tcs_optimizer_candidates),
                "num_tcs_metadata_valid_candidates": int(num_tcs_metadata_valid_candidates),
                "num_tcs_metadata_invalid_candidates": int(num_tcs_metadata_invalid_candidates),
                "tcs_execution_complete": tcs_execution_complete,
                "tcs_call_group_ids": sorted({str(c.get("tcs_call_group_id", "")) for c in candidate_pool if str(c.get("tcs_call_group_id", ""))}),
                "candidate_selection_mode": str(getattr(self.cfg, "candidate_selection_mode", "scalar_reward")),
                **candidate_eval_cache_stats,
                **pareto_summary,
                "candidate_starvation": bool(
                    requirements.get("safe_non_incumbent_count", 1) == 0
                ) if self._is_stable_qd_lineage() else False,
                "mechanism_starvation": bool(
                    requirements.get("safe_distinct_mechanism_count", 1) == 0
                ) if self._is_stable_qd_lineage() else False,
                "search_branch_starvation": bool(
                    requirements.get("safe_non_incumbent_count", 1) == 0
                    and not getattr(agent, "probation_archive", [])
                ) if self._is_stable_qd_lineage() else False,
                "candidate_starvation_count": int(getattr(self, "candidate_starvation_count", 0)),
                "mechanism_starvation_count": int(getattr(self, "mechanism_starvation_count", 0)),
                "search_branch_starvation_count": int(getattr(self, "search_branch_starvation_count", 0)),
                "refill_requirements_unmet_count": int(getattr(self, "refill_requirements_unmet_count", 0)),
                "top1_pareto_rank": selected[0].get("pareto_rank") if self._uses_vote_pareto_selection() and selected else None,
                "top1_vote_gain_rate": float(selected[0].get("metrics", {}).get("vote_gain_rate", 0.0)) if self._uses_vote_pareto_selection() and selected else None,
                "top1_vote_loss_rate": float(selected[0].get("metrics", {}).get("vote_loss_rate", 0.0)) if self._uses_vote_pareto_selection() and selected else None,
                "top1_vote_delta": float(selected[0].get("metrics", {}).get("vote_delta", 0.0)) if self._uses_vote_pareto_selection() and selected else None,
                **optimizer_generation_summary,
                **self._student_failure_log_fields(optimizer_generation_summary),
                "execution_session_id": self._current_execution_session_id(),
                "update_attempt_id": update_attempt_id,
            }
        )
        agent.last_update_record = summary
        return bool(changed), summary

    async def refresh_all_prompt_beams(self, eval_batch: List[Dict[str, str]], epoch_id: int) -> Dict[str, Any]:
        if self._is_stable_qd_lineage():
            for agent in self.agents:
                self._refresh_joint_representatives(agent)
            return {
                "event": "beam_refresh", "enabled": True, "mode": "safe_archive_representatives",
                "agent_count": len(self.agents), "active_prompt_changed_count": 0,
            }
        if not self.cfg.beam_refresh_each_epoch or not eval_batch:
            if self._residual_specialization_enabled():
                for agent in self.agents:
                    self._flush_capability_profile(agent, epoch_id, force=True)
            return {"event": "beam_refresh", "enabled": False, "agent_count": 0}
        records = []
        for agent_id, agent in enumerate(self.agents):
            old_scores = [x.get("score") for x in getattr(agent, "prompt_beam", []) if isinstance(x, dict)]
            old_hash = self._hash(agent.current_prompt)
            refreshed = []
            peer_prompts = self._active_prompt_list()
            for item in getattr(agent, "prompt_beam", []) or [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]:
                prompt = str(item.get("prompt", agent.current_prompt))
                prior_metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
                metrics = await self.evaluate_candidate_prompt(agent_id, prompt, peer_prompts, eval_batch, role_spec=prior_metrics)
                if self._is_v82_hybrid():
                    for key in (
                        "candidate_type", "mechanism_signature", "parent_mechanism_signature",
                        "peer_dominant_mechanism_signature", "mechanism_signature_distance", "beam_slot",
                    ):
                        if key in prior_metrics:
                            metrics[key] = prior_metrics[key]
                refreshed_item = {
                        "candidate_id": str(item.get("id", "")) or self._hash(prompt),
                        "prompt": prompt,
                        "parent_id": item.get("parent_id"),
                        "parent_prompt": agent.current_prompt,
                        "generation": int(item.get("generation", 0) or 0),
                        "source": "existing_beam",
                        "candidate_pool_source": "existing_beam",
                        "candidate_source": "existing_beam",
                        "metrics": metrics,
                        "reward": float(metrics.get("reward", 0.0)),
                    }
                if self._v7_residual_protocol_enabled():
                    metrics.update(self._candidate_trajectory_feasibility(agent, refreshed_item))
                    refreshed_item["metrics"] = metrics
                if self._is_v82_hybrid():
                    metrics = self._apply_hybrid_soft_guards(metrics)
                    refreshed_item["metrics"] = metrics
                    refreshed_item["reward"] = float(metrics.get("penalized_reward", metrics.get("reward", 0.0)) or 0.0)
                self._apply_competence_depth1_candidate_guard(metrics)
                refreshed.append(refreshed_item)
            if self._v7_residual_protocol_enabled() or bool(getattr(self.cfg, "competence_depth1_candidate_guard_enabled", False)):
                refreshed = [item for item in refreshed if not str(item.get("metrics", {}).get("rejection_reason", ""))]
                if not refreshed:
                    raise RuntimeError("Beam refresh trajectory guard removed the current active prompt")
            if self._is_v82_hybrid():
                retained, _ = self._select_hybrid_beam(
                    refreshed, max(1, int(self.cfg.beam_size)), agent.current_prompt,
                    agent_id=agent_id, epoch_id=epoch_id, step_id=0,
                )
            elif self._uses_vote_pareto_selection():
                retained, _ = self._select_vote_pareto_beam(refreshed, max(1, int(self.cfg.beam_size)), agent.current_prompt)
            else:
                retained = sorted(refreshed, key=lambda item: float(item.get("reward", 0.0)), reverse=True)[: max(1, int(self.cfg.beam_size))]
            agent.prompt_beam = [
                self._make_beam_item(
                    prompt=str(item["prompt"]),
                    score=float(item.get("reward", 0.0)),
                    metrics=item.get("metrics", {}),
                    parent_id=item.get("parent_id"),
                    generation=int(item.get("generation", 0) or 0),
                    candidate_id=str(item.get("candidate_id", "")) or None,
                )
                for item in retained
            ]
            agent.current_prompt = str(agent.prompt_beam[0]["prompt"])
            changed = old_hash != self._hash(agent.current_prompt)
            if changed:
                agent.history.append(agent.current_prompt)
                agent.accept_count += 1
                if self._v7_residual_protocol_enabled():
                    profile_before = dict(agent.capability_profile)
                    active_metrics = retained[0].get("metrics", {})
                    if self._residual_specialization_enabled():
                        self._update_vote_context_profile(agent, active_metrics)
                        self._accumulate_capability_evidence(agent, active_metrics, epoch_id)
                        self._flush_capability_profile(agent, epoch_id, force=False)
                    agent.last_accepted_prompt_hash = self._normalized_prompt_hash(agent.current_prompt)
                    fingerprint = {
                        str(key): BehaviorFingerprintEntry.from_dict(value)
                        for key, value in dict(active_metrics.get("behavior_fingerprint", {})).items()
                        if isinstance(value, dict)
                    }
                    self._append_bounded_archive(
                        agent.accepted_behavior_archive,
                        BehaviorStateSummary(
                            state_id=f"e{int(epoch_id)}_refresh_a{int(agent_id)}_{str(retained[0].get('candidate_id', ''))}",
                            epoch=int(epoch_id),
                            prompt_hash=agent.last_accepted_prompt_hash,
                            behavior_fingerprint=fingerprint,
                            transition_vector={str(key): float(value) for key, value in dict(active_metrics.get("candidate_transition_vector", {})).items()},
                            target_accuracy=float(active_metrics.get("candidate_target_accuracy", 0.0) or 0.0),
                            team_vote_accuracy=float(active_metrics.get("candidate_team_accuracy", 0.0) or 0.0),
                            mean_vote_margin=float(active_metrics.get("candidate_mean_vote_margin", 0.0) or 0.0),
                            preserved_mechanisms=[],
                            capability_profile=dict(agent.capability_profile),
                            paired_behavior_utility=self.behavior_fingerprint_utility(fingerprint),
                        ),
                    )
                    self.trajectory_events.append(self._trajectory_event(
                        agent_id=agent_id,
                        epoch_id=epoch_id,
                        step_id=0,
                        item=retained[0],
                        accepted=True,
                        profile_before=profile_before,
                        profile_after=dict(agent.capability_profile),
                    ))
                    self.trajectory_events[-1]["decision"] = "beam_refresh_activated"
            record = {
                **self._base_log_fields(),
                "event": "beam_refresh",
                "epoch": epoch_id,
                "step": 0,
                "agent_id": agent_id,
                "old_beam_scores": old_scores,
                "new_beam_scores": [x.get("score") for x in agent.prompt_beam],
                "active_prompt_changed": bool(changed),
                "beam_size": int(self.cfg.beam_size),
            }
            self.update_logs.append(record)
            self._append_prompt_history_event(agent_id, epoch_id, 0, "beam_refresh_changed" if changed else "beam_refresh_keep", changed)
            records.append(record)
        if self._residual_specialization_enabled():
            for agent in self.agents:
                self._flush_capability_profile(agent, epoch_id, force=True)
        self.flush_update_logs()
        self.flush_prompt_history()
        return {
            "event": "beam_refresh",
            "enabled": True,
            "agent_count": len(records),
            "active_prompt_changed_count": int(sum(1 for r in records if r.get("active_prompt_changed"))),
        }

    async def maybe_update_prompts(self, metrics: Dict[str, Any], eval_batch: List[Dict[str, str]], step_id: int, epoch_id: int) -> Dict[str, Any]:
        if not self.is_update_window_ready():
            return {"update_requested": True, "update_ready": False, "selected_agent_ids": [], "updated_agent_ids": [], "skipped_reason": "window_not_ready"}
        if self._is_accuracy_only_mode():
            diagnosis = self._window_accuracy_diagnosis(self.recent_window_records)
            selected = self.select_reward_agents_for_update(diagnosis, metrics)
            no_selection_reason = "no_reward_relevant_agent"
        else:
            diagnosis = self._window_update_diagnosis(self.recent_window_records)
            selected = self.select_reward_agents_for_update(diagnosis, metrics)
            no_selection_reason = "no_reward_relevant_agent"
        if self._is_v82_hybrid():
            for row in diagnosis.get("hybrid_selector_diagnostics", []):
                row["selected"] = int(row.get("agent_id", -1)) in selected
            missed_niche_agents = []
            if self._is_stable_qd_lineage():
                for agent_id, agent in enumerate(self.agents):
                    if agent_id in selected:
                        continue
                    active_hash = self._normalized_prompt_hash(agent.current_prompt)
                    has_branch = any(
                        str(item.get("prompt_hash", "")) != active_hash
                        for item in getattr(agent, "safe_qd_archive", [])
                    ) or bool(getattr(agent, "probation_archive", []))
                    if has_branch:
                        missed_niche_agents.append(agent_id)
                diagnosis["niche_parent_opportunity_missed_due_to_agent_not_selected"] = missed_niche_agents
            selector_event = {
                "epoch": int(epoch_id),
                "step": int(step_id),
                "applied_specialization_strength": float(getattr(self, "specialization_strength", 0.0)),
                "weights": dict(diagnosis.get("hybrid_selector_weights", {})),
                "agents": list(diagnosis.get("hybrid_selector_diagnostics", [])),
                "fairness_slot_selected": diagnosis.get("fairness_slot_selected"),
                "fairness_slot_skipped_no_evidence": bool(diagnosis.get("fairness_slot_skipped_no_evidence", False)),
                "per_agent_optimizer_update_count": {
                    str(agent_id): int(self.per_agent_optimizer_update_count.get(f"{epoch_id}:{agent_id}", 0))
                    for agent_id in range(len(self.agents))
                },
                "niche_parent_opportunity_missed_due_to_agent_not_selected": list(
                    diagnosis.get("niche_parent_opportunity_missed_due_to_agent_not_selected", [])
                ),
            }
            self.hybrid_selector_history.append(selector_event)
            self.update_logs.append({**self._base_log_fields(), "event": "hybrid_target_selection", **selector_event})
        if not selected:
            self.clear_homogeneity_windows()
            return {
                "update_requested": True,
                "update_ready": True,
                "selected_agent_ids": [],
                "updated_agent_ids": [],
                "skipped_reason": no_selection_reason,
                "requested_optimizer_candidates": 0,
                "num_optimizer_candidates": 0,
                "num_fallback_candidates": 0,
                "num_existing_beam_candidates": 0,
                "active_prompt_changed_count": 0,
                "optimizer_underfilled": False,
            }
        updated = []
        top_metrics = []
        update_summaries = []
        for agent_id in selected:
            changed, summary = await self.update_prompt_with_beam(agent_id, diagnosis, eval_batch, step_id, epoch_id)
            update_summaries.append(summary)
            if changed:
                updated.append(agent_id)
            if isinstance(summary.get("top_metrics", {}), dict):
                top_metrics.append(summary["top_metrics"])
        self.clear_homogeneity_windows()
        self.flush_update_logs()
        self.flush_prompt_history()
        requested_optimizer_candidates = int(sum(int(s.get("requested_optimizer_candidates", 0) or 0) for s in update_summaries))
        num_optimizer_candidates = int(sum(int(s.get("num_optimizer_candidates", 0) or 0) for s in update_summaries))
        num_fallback_candidates = int(sum(int(s.get("num_fallback_candidates", 0) or 0) for s in update_summaries))
        num_existing_beam_candidates = int(sum(int(s.get("num_existing_beam_candidates", 0) or 0) for s in update_summaries))
        diagnostic_keys = [
            "optimizer_raw_response_empty",
            "optimizer_json_parse_failed",
            "optimizer_raw_candidate_count",
            "optimizer_empty_prompt_count",
            "optimizer_sanitized_count",
            "optimizer_redundant_filtered_count",
            "optimizer_schema_filtered_count",
            "optimizer_final_candidate_count",
            "student_missing_required_field_count",
        ]
        optimizer_generation_diagnostics = {
            key: int(sum(int(s.get(key, 0) or 0) for s in update_summaries))
            for key in diagnostic_keys
        }
        metadata_keys = [
            "optimizer_architecture",
            "teacher_question",
            "teacher_question_approved",
            "teacher_question_rejected",
            "teacher_question_rejection_reason",
            "teacher_question_forced_best_score",
            "teacher_question_forced_best_round",
            "teacher_question_forced_below_threshold",
            "teacher_question_score",
            "teacher_critic_rounds",
            "teacher_quality_critique",
            "teacher_specificity_critique",
            "teacher_task_alignment_critique",
            "teacher_error_alignment_critique",
            "teacher_diversity_critique",
            "teacher_rewrite_count",
            "student_candidate_count_raw",
            "student_candidate_count_final",
            "student_candidate_filtered_count",
            "student_candidate_filter_reasons",
            "student_all_candidates_filtered",
            "student_missing_required_fields",
            "student_raw_response_empty",
            "student_raw_response_preview",
            "student_json_parse_failed",
            "student_json_parse_error",
            "student_json_has_candidates_key",
            "student_candidates_is_list",
            "student_candidates_empty_list",
            "student_refusal_or_explanation",
            "student_failure_stage",
        ]
        optimizer_generation_metadata = {}
        for key in metadata_keys:
            values = [s.get(key) for s in update_summaries if isinstance(s, dict) and s.get(key) not in (None, "", [])]
            if values:
                optimizer_generation_metadata[key] = values[-1]
        return {
            "update_requested": True,
            "update_ready": True,
            "selected_agent_ids": selected,
            "updated_agent_ids": updated,
            "skipped_reason": "none",
            "requested_optimizer_candidates": requested_optimizer_candidates,
            "num_optimizer_candidates": num_optimizer_candidates,
            "num_fallback_candidates": num_fallback_candidates,
            "num_existing_beam_candidates": num_existing_beam_candidates,
            "active_prompt_changed_count": int(len(updated)),
            "optimizer_underfilled": bool(num_optimizer_candidates < requested_optimizer_candidates),
            **optimizer_generation_diagnostics,
            **optimizer_generation_metadata,
            "candidate_behavior_diagnostics": self._mean_metric_dict(top_metrics),
            "hybrid_selector_diagnostics": list(diagnosis.get("hybrid_selector_diagnostics", [])),
            "hybrid_selector_weights": dict(diagnosis.get("hybrid_selector_weights", {})),
        }

    def _apply_no_effective_evolution_tracking(
        self,
        update_summary: Dict[str, Any],
        epoch_id: int = 0,
        step_id: int = 0,
    ) -> Dict[str, Any]:
        if not isinstance(update_summary, dict):
            update_summary = {}
        if not bool(update_summary.get("update_requested", False)) or not bool(update_summary.get("update_ready", False)):
            update_summary["no_effective_evolution_counter"] = int(getattr(self, "no_effective_evolution_counter", 0) or 0)
            update_summary["no_effective_evolution_stopped"] = bool(getattr(self, "no_effective_evolution_stopped", False))
            update_summary["no_effective_evolution_reason"] = str(getattr(self, "no_effective_evolution_reason", ""))
            return update_summary

        min_optimizer_candidates = max(
            0,
            int(getattr(self.cfg, "no_effective_evolution_min_optimizer_candidates", 1) or 0),
        )
        num_optimizer_candidates = int(update_summary.get("num_optimizer_candidates", 0) or 0)
        active_prompt_changed_count = int(
            update_summary.get(
                "active_prompt_changed_count",
                len(update_summary.get("updated_agent_ids", []) or []),
            )
            or 0
        )
        ineffective = num_optimizer_candidates < min_optimizer_candidates and active_prompt_changed_count <= 0
        if ineffective:
            self.no_effective_evolution_counter = int(getattr(self, "no_effective_evolution_counter", 0) or 0) + 1
        else:
            self.no_effective_evolution_counter = 0
            self.no_effective_evolution_stopped = False
            self.no_effective_evolution_reason = ""

        enabled = bool(int(getattr(self.cfg, "no_effective_evolution_stop_enabled", True)))
        patience = max(1, int(getattr(self.cfg, "no_effective_evolution_patience", 10) or 10))
        if enabled and self.no_effective_evolution_counter >= patience:
            self.no_effective_evolution_stopped = True
            self.no_effective_evolution_reason = (
                f"num_optimizer_candidates<{min_optimizer_candidates} and no active prompt changed"
            )

        update_summary["no_effective_evolution_counter"] = int(self.no_effective_evolution_counter)
        update_summary["no_effective_evolution_stopped"] = bool(self.no_effective_evolution_stopped)
        update_summary["no_effective_evolution_reason"] = str(self.no_effective_evolution_reason)
        self.update_logs.append(
            {
                **self._base_log_fields(),
                "event": "no_effective_evolution_check",
                "epoch": epoch_id,
                "step": step_id,
                "no_effective_evolution_counter": int(self.no_effective_evolution_counter),
                "no_effective_evolution_stopped": bool(self.no_effective_evolution_stopped),
                "no_effective_evolution_reason": str(self.no_effective_evolution_reason),
                "requested_optimizer_candidates": int(update_summary.get("requested_optimizer_candidates", 0) or 0),
                "num_optimizer_candidates": int(update_summary.get("num_optimizer_candidates", 0) or 0),
                "num_fallback_candidates": int(update_summary.get("num_fallback_candidates", 0) or 0),
                "num_existing_beam_candidates": int(update_summary.get("num_existing_beam_candidates", 0) or 0),
                "active_prompt_changed_count": int(update_summary.get("active_prompt_changed_count", 0) or 0),
                "optimizer_underfilled": bool(update_summary.get("optimizer_underfilled", False)),
            }
        )
        return update_summary

    def _mean_metric_dict(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        keys = [
            "reward",
            "embedding_diversity",
            "mean_embedding_overlap",
            "target_overlap_pressure",
            "homogeneous_case_count",
            "resolved_case_count",
            "new_homogeneous_case_count",
            "team_accuracy",
            "target_agent_accuracy",
            "baseline_team_accuracy",
            "candidate_team_accuracy",
            "baseline_oracle_acc",
            "candidate_oracle_acc",
            "coverage_delta",
            "rescue_rate",
            "useful_diversity",
            "rescue_useful_diversity",
            "vote_delta",
            "invalid_rate",
            "invalid_score",
            "solver_reuse_hits",
            "solver_reuse_misses",
            "solver_calls",
            "solver_reuse_total",
            "solver_reuse_hit_rate",
        ]
        return {k: float(np.mean([float(r.get(k, 0.0)) for r in rows])) if rows else 0.0 for k in keys}

    async def solve_train_example_without_update(
        self,
        question: str,
        gold: str,
    ) -> Dict[str, Any]:
        for i, agent in enumerate(self.agents):
            sanitized, changed = self._sanitize_prompt(agent.current_prompt, i, question)
            if changed:
                agent.current_prompt = sanitized
                if agent.prompt_beam:
                    agent.prompt_beam[0]["prompt"] = sanitized
        prompts = self._active_prompt_list()
        traces, answers = await self.solve_with_prompts(question, prompts)
        question_hash = self._hash(question)
        self._record_solver_rollouts(question_hash, prompts, traces, answers, source="train_rollout")
        metrics = self.compute_rollout_metrics(traces, answers, gold, prompts, question_hash=question_hash)
        if self._is_accuracy_only_mode():
            homogeneous_cases = []
            validity_cases = []
        else:
            homogeneous_cases = self._build_homogeneous_cases(question_hash, traces, answers, prompts, metrics)
            validity_cases = self._build_validity_cases(question_hash, traces, answers, prompts)
        return {
            "question_hash": question_hash,
            "gold": gold,
            "traces": traces,
            "answers": answers,
            "prompts": prompts,
            "metrics": metrics,
            "homogeneous_cases": homogeneous_cases,
            "validity_cases": validity_cases,
        }

    async def record_train_rollout(
        self,
        solved: Dict[str, Any],
        do_update: bool = True,
        eval_batch: Optional[List[Dict[str, str]]] = None,
        step_id: int = 0,
        epoch_id: int = 0,
    ) -> Dict[str, Any]:
        question_hash = str(solved.get("question_hash", ""))
        traces = list(solved.get("traces", []))
        answers = list(solved.get("answers", []))
        prompts = list(solved.get("prompts", []))
        metrics = dict(solved.get("metrics", {}))
        homogeneous_cases = list(solved.get("homogeneous_cases", []))
        validity_cases = list(solved.get("validity_cases", []))
        self.recent_window_records.append(
            {
                "question_hash": question_hash,
                "gold": str(solved.get("gold", "")),
                "traces": traces,
                "answers": answers,
                "prompts": prompts,
                "metrics": metrics,
                "homogeneous_cases": homogeneous_cases,
                "validity_cases": validity_cases,
            }
        )
        self.recent_window_records = self.recent_window_records[-self.homogeneity_window :]
        if not self._is_accuracy_only_mode():
            for i, pressure in enumerate(metrics.get("per_agent_overlap", [])):
                self.agents[i].observe_homogeneity_result(1 if float(pressure) >= float(self.cfg.homogeneity_overlap_threshold) else 0)

        update_summary = {"update_requested": bool(do_update), "update_ready": self.is_update_window_ready(), "selected_agent_ids": [], "updated_agent_ids": []}
        if do_update and eval_batch is not None:
            update_summary = await self.maybe_update_prompts(metrics, eval_batch, step_id, epoch_id)
        update_summary = self._apply_no_effective_evolution_tracking(update_summary, epoch_id=epoch_id, step_id=step_id)
        record = {
            **self._base_log_fields(),
            "epoch": epoch_id,
            "step": step_id,
            "vote_correct": int(metrics.get("vote_correct", 0)),
            "vote_answer": metrics.get("vote_answer", ""),
            "plurality_vote_correct": int(metrics.get("plurality_vote_correct", metrics.get("vote_correct", 0))),
            "plurality_vote_answer": metrics.get("plurality_vote_answer", metrics.get("vote_answer", "")),
            "majority_vote_correct": int(metrics.get("majority_vote_correct", metrics.get("vote_correct", 0))),
            "majority_vote_answer": metrics.get("majority_vote_answer", metrics.get("vote_answer", "")),
            "weighted_vote_correct": int(metrics.get("weighted_vote_correct", 0)),
            "weighted_vote_answer": metrics.get("weighted_vote_answer", ""),
            "aggregation_mode": metrics.get("aggregation_mode", "majority"),
            "requested_aggregation_mode": metrics.get("requested_aggregation_mode", metrics.get("aggregation_mode", "majority")),
            "effective_aggregation_mode": metrics.get("effective_aggregation_mode", "plurality"),
            "any_correct": int(metrics.get("any_correct", 0)),
            "aggregation_gap_available": int(metrics.get("any_correct", 0)) - int(metrics.get("vote_correct", 0)),
            "useful_diversity": float(metrics.get("useful_diversity", 0.0)),
            "vote_tie": bool(metrics.get("vote_tie", False)),
            "tie_candidates": metrics.get("tie_candidates", []),
            "vote_counts": metrics.get("vote_counts", {}),
            "tie_break_method": metrics.get("tie_break_method", ""),
            "embedding_diversity": float(metrics.get("embedding_diversity", 0.0)),
            "mean_embedding_overlap": float(metrics.get("mean_embedding_overlap", 0.0)),
            "homogeneous_case_count": len(homogeneous_cases),
            "validity_case_count": len(validity_cases),
            "invalid_rate": float(metrics.get("invalid_rate", 0.0)),
            "update_summary": update_summary,
            "no_effective_evolution_counter": int(update_summary.get("no_effective_evolution_counter", 0) or 0),
            "no_effective_evolution_stopped": bool(update_summary.get("no_effective_evolution_stopped", False)),
            "no_effective_evolution_reason": str(update_summary.get("no_effective_evolution_reason", "")),
        }
        self.train_step_logs.append(record)
        self.train_trace_history_logs.append(
            {
                **record,
                "question_hash": question_hash,
                "homogeneous_cases": homogeneous_cases,
                "validity_cases": validity_cases,
                "agents": [
                    {
                        "agent_id": i,
                        "prompt_hash": self._hash(prompts[i]),
                        "trace": traces[i],
                        "answer": answers[i],
                        "invalid": {"invalid": 0, "reasons": ["skipped_accuracy_only"]} if self._is_accuracy_only_mode() else self.rule_invalid_check(traces[i], answers[i]),
                    }
                    for i in range(len(self.agents))
                ],
            }
        )
        if len(self.train_step_logs) >= 20:
            self.flush_train_step_logs()
        if len(self.train_trace_history_logs) >= 20:
            self.flush_train_trace_history_logs()
        return {
            **metrics,
            "homogeneous_case_count": len(homogeneous_cases),
            "validity_case_count": len(validity_cases),
            "update_summary": update_summary,
        }

    async def rollout_train_example(
        self,
        question: str,
        gold: str,
        do_update: bool = True,
        eval_batch: Optional[List[Dict[str, str]]] = None,
        step_id: int = 0,
        epoch_id: int = 0,
    ) -> Dict[str, Any]:
        solved = await self.solve_train_example_without_update(question, gold)
        return await self.record_train_rollout(
            solved,
            do_update=do_update,
            eval_batch=eval_batch,
            step_id=step_id,
            epoch_id=epoch_id,
        )

    def _stable_probe_cache_key(self, agent_id: int, prompt: str, question_hash: str) -> str:
        payload = {
            "agent_id": int(agent_id), "prompt_hash": self._normalized_prompt_hash(prompt),
            "agent_model": self.cfg.agent_model, "question_hash": question_hash,
            "task_type": self.cfg.task_type, "answer_format": getattr(self.cfg, "answer_format", ""),
            "aggregation_mode": self.cfg.aggregation_mode, "vote_tie_break": self.cfg.vote_tie_break,
            "seed": int(self.cfg.seed),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    async def _evaluate_prompt_on_stable_probe(
        self, agent_id: int, prompt: str, probe_data: List[Dict[str, str]], mechanism_steps: Sequence[Any] = (),
    ) -> Dict[str, Any]:
        async def evaluate_example(example: Dict[str, str]) -> Dict[str, Any]:
            question = example["question"]
            question_hash = self._hash(question)
            gold = self.task_spec.parse_gold(example["answer"], question)
            cache_key = self._stable_probe_cache_key(agent_id, prompt, question_hash)
            cached = self.prompt_probe_cache.get(cache_key)
            if cached is None:
                self.full_probe_missing_pair_evaluation_count = int(getattr(self, "full_probe_missing_pair_evaluation_count", 0)) + 1
                async with self.solver_call_semaphore:
                    trace, answer = await self.solve_once(question, agent_id, prompt)
                cached = {"trace": trace, "answer": answer, "gold": gold, "question_hash": question_hash}
                self.prompt_probe_cache[cache_key] = dict(cached)
                self._record_solver_rollout(
                    question_hash=question_hash,
                    prompt=prompt,
                    trace=trace,
                    answer=answer,
                    agent_id=agent_id,
                    source="stable_qd_probe",
                )
            else:
                self.full_probe_cache_hit_count = int(getattr(self, "full_probe_cache_hit_count", 0)) + 1
            answer = str(cached.get("answer", ""))
            return {
                "answer": answer,
                "correct": int(self.task_spec.match_answer(answer, gold)),
                "question_hash": question_hash,
                "gold": gold,
            }

        rows = await asyncio.gather(*[evaluate_example(example) for example in probe_data])
        answers = [row["answer"] for row in rows]
        correctness = [row["correct"] for row in rows]
        question_hashes = [row["question_hash"] for row in rows]
        gold_answers = [row["gold"] for row in rows]
        item = {"prompt": prompt, "metrics": {"mechanism_steps": list(mechanism_steps)}}
        representation = self._attach_stable_mechanism_representation(item)
        return {
            "prompt": prompt,
            "prompt_hash": self._normalized_prompt_hash(prompt),
            "answer_vector": answers,
            "correctness_vector": correctness,
            "accuracy": float(np.mean(correctness)) if correctness else 0.0,
            "question_hashes": question_hashes,
            "gold_answers": gold_answers,
            "mechanism_representation": representation,
            "prompt_static_profile": build_prompt_static_profile(answers, correctness),
        }

    async def select_joint_active_team(self, probe_data: List[Dict[str, str]], *, epoch: int) -> Dict[str, Any]:
        if not self._is_stable_qd_lineage():
            return {"enabled": False, "combination_count": 0}
        beams: List[List[Dict[str, Any]]] = []
        for agent_id, agent in enumerate(self.agents):
            if getattr(agent, "safe_qd_archive", []):
                self._refresh_joint_representatives(agent)
            agent_profiles = []
            for beam_index, item in enumerate(agent.prompt_beam):
                metrics = item.get("metrics", {})
                profile = await self._evaluate_prompt_on_stable_probe(
                    agent_id, str(item.get("prompt", agent.current_prompt)), probe_data,
                    metrics.get("mechanism_steps", metrics.get("mechanism_signature", [])),
                )
                profile.update({
                    "beam_index": beam_index,
                    "beam_source": str(item.get("beam_slot", metrics.get("beam_slot", "incumbent"))),
                })
                self.behavior_profile_by_prompt_hash[profile["prompt_hash"]] = dict(profile)
                agent_profiles.append(profile)
            beams.append(agent_profiles)
        question_hashes = beams[0][0]["question_hashes"]
        gold_answers = beams[0][0]["gold_answers"]
        teams = enumerate_joint_teams(
            beams, gold_answers, question_hashes,
            vote_fn=plurality_vote_with_diagnostics, match_fn=self.task_spec.match_answer,
            tie_break_method=self.cfg.vote_tie_break, seed=self.cfg.seed,
        )
        incumbent_indices = []
        for agent_id, agent in enumerate(self.agents):
            current_hash = self._normalized_prompt_hash(agent.current_prompt)
            incumbent_indices.append(next((index for index, profile in enumerate(beams[agent_id]) if profile["prompt_hash"] == current_hash), 0))
        incumbent = next(team for team in teams if team["beam_indices"] == incumbent_indices)
        initial_per_agent = list(self.initial_competence_probe_metrics.get("per_agent_acc", incumbent["per_agent_acc"]))
        joint_selection = select_stable_joint_team(
            teams, incumbent, initial_per_agent,
            [agent.lineage_state for agent in self.agents], len(probe_data), self.cfg,
            gold_answers=gold_answers, question_hashes=question_hashes,
            vote_fn=plurality_vote_with_diagnostics, match_fn=self.task_spec.match_answer,
            tie_break_method=self.cfg.vote_tie_break, seed=self.cfg.seed,
            change_limit=self._current_joint_change_limit(epoch),
        )
        selected = joint_selection["selected"]
        self.peer_collapse_soft_count += int(joint_selection["selected_has_soft_peer_collapse"])
        self.peer_collapse_hard_rejection_count += int(joint_selection["hard_rejection_count"])
        selected_sources, changed_count = [], 0
        for agent_id, beam_index in enumerate(selected["beam_indices"]):
            agent = self.agents[agent_id]
            chosen = agent.prompt_beam[beam_index]
            old_hash = self._normalized_prompt_hash(agent.current_prompt)
            agent.prompt_beam = [chosen] + [item for index, item in enumerate(agent.prompt_beam) if index != beam_index]
            agent.current_prompt = str(chosen["prompt"])
            changed_count += int(old_hash != self._normalized_prompt_hash(agent.current_prompt))
            selected_sources.append(str(chosen.get("beam_slot", chosen.get("metrics", {}).get("beam_slot", "incumbent"))))
            selected_profile = selected["prompt_profiles"][agent_id]
            selected_profile["behavior_profile"] = selected["behavior_profiles"][agent_id]
            selected_profile["cross_fold_diversity_gap"] = float(selected.get("cross_fold_diversity_gap", 0.0))
            selected_profile["fold_quality_gate_passed"] = bool(selected.get("fold_quality_gate_passed", True))
            per_agent_fold_gaps = list(selected.get("per_agent_cross_fold_behavior_gap", []))
            selected_profile["cross_fold_behavior_gap"] = (
                float(per_agent_fold_gaps[agent_id]) if agent_id < len(per_agent_fold_gaps) else 0.0
            )
            selected_profile["fold_behavior_stable"] = bool(
                selected_profile["cross_fold_behavior_gap"] <= float(self.cfg.qd_readiness_max_fold_gap)
            )
            selected_drift = selected_profile.get("lineage_drift", {})
            agent.lineage_state["last_lineage_drift"] = float(selected_drift.get("lineage_drift", 0.0) or 0.0)
            lineage_record = update_lineage_state(
                agent.lineage_state,
                selected_profile,
                epoch=epoch,
                quality_gate_passed=bool(selected.get("fold_quality_gate_passed", True)),
                config=self.cfg,
            )
            agent.lineage_state = {key: value for key, value in lineage_record.items() if key not in {"old_status", "new_status", "reason"}}
            self.lineage_history.append({"epoch": epoch, "agent_id": agent_id, **lineage_record})
        record = {
            "epoch": epoch, "combination_count": len(teams),
            "representative_count_per_agent": [len(beam) for beam in beams],
            "theoretical_combination_count": int(np.prod([len(beam) for beam in beams])) if beams else 0,
            "post_change_limit_combination_count": int(
                len(teams) - int(joint_selection.get("combination_rejected_by_change_limit_count", 0))
            ),
            "feasible_count": int(joint_selection["feasible_count"]),
            "quality_floor_feasible_count": int(joint_selection.get("quality_floor_feasible_count", joint_selection["feasible_count"])),
            "quality_frontier_count": int(joint_selection["quality_frontier_count"]),
            "final_candidate_team_count": int(joint_selection.get("final_candidate_team_count", joint_selection["quality_frontier_count"])),
            "hierarchical_band_counts": list(joint_selection.get("hierarchical_band_counts", [])),
            "hierarchical_band_count_by_name": dict(joint_selection.get("hierarchical_band_count_by_name", {})),
            "combination_rejected_by_change_limit_count": int(joint_selection.get("combination_rejected_by_change_limit_count", 0)),
            "fold_quality_rejection_count": int(joint_selection.get("fold_quality_rejection_count", 0)),
            "incumbent_metrics": {
                key: incumbent[key] for key in (*QUALITY_KEYS, "vote_correct_count", "total_agent_correct_count", "bottom2_correct_count", "per_agent_correct_count", "coverage_depth_c1_correct_count", "coverage_depth_c2_correct_count")
            },
            "selected_metrics": {
                key: selected[key] for key in (*QUALITY_KEYS, "vote_correct_count", "total_agent_correct_count", "bottom2_correct_count", "per_agent_correct_count", "coverage_depth_c1_correct_count", "coverage_depth_c2_correct_count")
            },
            "allowed_quality_losses": {
                "vote_correct_count": int(self.cfg.joint_allowed_vote_loss_questions),
                "total_agent_correct_count": int(self.cfg.joint_allowed_total_agent_correct_loss),
                "bottom2_correct_count": int(self.cfg.joint_allowed_bottom2_correct_loss),
                "c1_correct_count": int(self.cfg.joint_allowed_c1_loss_questions),
                "c2_correct_count": int(self.cfg.joint_allowed_c2_loss_questions),
                "per_agent_correct_count": int(self.cfg.joint_allowed_per_agent_correct_loss),
            },
            "safe_archive_size_per_agent": [len(getattr(agent, "safe_qd_archive", [])) for agent in self.agents],
            "probation_archive_size_per_agent": [len(getattr(agent, "probation_archive", [])) for agent in self.agents],
            "selected_prompt_hashes": selected["prompt_hashes"], "selected_beam_sources": selected_sources,
            "team_diversity_score": selected["team_diversity_score"],
            "stable_team_score": selected["stable_team_score"],
            "mean_behavior_distance": selected["mean_behavior_distance"],
            "min_behavior_distance": selected["min_behavior_distance"],
            "mean_mechanism_distance": selected["mean_mechanism_distance"],
            "fold_diversities": list(selected.get("fold_diversities", [])),
            "fold_quality_gate_passed": bool(selected.get("fold_quality_gate_passed", True)),
            "per_agent_cross_fold_behavior_gap": list(selected.get("per_agent_cross_fold_behavior_gap", [])),
            "cross_fold_diversity_mean": float(selected.get("cross_fold_diversity_mean", 0.0)),
            "cross_fold_diversity_gap": float(selected.get("cross_fold_diversity_gap", 0.0)),
            "stable_diversity_score": float(selected.get("stable_diversity_score", 0.0)),
            "lineage_drift_penalty_mean": float(selected.get("lineage_drift_penalty_mean", 0.0) or 0.0),
            "peer_collapse_penalty_mean": float(selected.get("peer_collapse_penalty_mean", 0.0) or 0.0),
            "active_prompt_changed_count": changed_count,
            "specialization_strength": float(self.specialization_strength),
            "full_probe_cache_hits": int(getattr(self, "full_probe_cache_hit_count", 0)),
            "full_probe_missing_pair_evaluations": int(getattr(self, "full_probe_missing_pair_evaluation_count", 0)),
            "embedding_cache_hits": int(getattr(self, "mechanism_embedding_cache_hit_count", 0)),
            "embedding_cache_misses": int(getattr(self, "mechanism_embedding_cache_miss_count", 0)),
        }
        self.joint_team_selection_history.append(record)
        self.latest_joint_team_metrics = dict(record)
        safe_niches = {
            mechanism_niche_key(item.get("metrics", {}).get("mechanism_representation", {}))
            for agent in self.agents
            for item in getattr(agent, "safe_qd_archive", [])
        }
        initial = dict(self.initial_competence_probe_metrics or {})
        initial_mean = float(initial.get("mean_individual_acc", 0.0) or 0.0)
        initial_c1 = float(initial.get("coverage_depth_c1", 0.0) or 0.0)
        initial_c2 = float(initial.get("coverage_depth_c2", 0.0) or 0.0)
        selected_mean = float(selected.get("mean_individual_acc", 0.0) or 0.0)
        selected_c1 = float(selected.get("coverage_depth_c1", 0.0) or 0.0)
        selected_c2 = float(selected.get("coverage_depth_c2", 0.0) or 0.0)
        competence_mean_gate_passed = selected_mean >= initial_mean - float(self.cfg.competence_mean_guard_epsilon)
        competence_c1_gate_passed = selected_c1 >= initial_c1 - float(self.cfg.competence_c1_guard_epsilon)
        competence_c2_gate_passed = selected_c2 >= initial_c2 - float(self.cfg.competence_c2_guard_epsilon)
        qd_ready = (
            competence_mean_gate_passed
            and competence_c1_gate_passed
            and competence_c2_gate_passed
            and len(safe_niches) >= int(self.cfg.qd_readiness_min_distinct_niches)
            and float(record.get("stable_diversity_score", 0.0)) >= float(self.cfg.qd_readiness_min_diversity)
            and float(record.get("cross_fold_diversity_gap", 0.0)) <= float(self.cfg.qd_readiness_max_fold_gap)
        )
        self._recompute_effective_residual_strength(qd_ready)
        record.update({
            "qd_readiness_passed": bool(qd_ready),
            "safe_distinct_mechanism_niche_count": len(safe_niches),
            "competence_mean_gate_passed": bool(competence_mean_gate_passed),
            "competence_c1_gate_passed": bool(competence_c1_gate_passed),
            "competence_c2_gate_passed": bool(competence_c2_gate_passed),
            "competence_schedule_strength": float(self.specialization_strength),
            "qd_residual_floor_applied": bool(qd_ready and self.effective_residual_strength > self.specialization_strength),
            "effective_residual_strength": float(self.effective_residual_strength),
        })
        self.latest_joint_team_metrics = dict(record)
        active_niches = len({
            tuple(profile["mechanism_representation"].get("normalized_operation_sequence", [])[:4])
            for profile in selected["prompt_profiles"]
        })
        no_new_niche = active_niches <= int(getattr(self, "qd_previous_active_niche_count", 0) or 0)
        incumbent_retained = changed_count == 0
        self.qd_no_diversification_epochs = int(self.qd_no_diversification_epochs) + 1 if (incumbent_retained and no_new_niche) else 0
        self.qd_previous_active_niche_count = int(active_niches)
        if self.qd_no_diversification_epochs >= int(self.cfg.joint_team_no_diversification_patience):
            self.qd_change_limit_relaxed_epoch = int(epoch) + 1
        self._flush_jsonl("joint_team_selection_history.jsonl", [record])
        self._flush_jsonl("lineage_history.jsonl", self.lineage_history[-len(self.agents):])
        self._flush_jsonl("quality_diversity_archive.jsonl", self.quality_diversity_archive_history)
        self.quality_diversity_archive_history = []
        return record

    async def evaluate_competence_probe(
        self,
        probe_data: List[Dict[str, str]],
        *,
        probe_name: str,
        epoch: int,
    ) -> Dict[str, Any]:
        """Evaluate current prompts on a fixed optimization-only probe without training side effects."""
        prompts = list(self._active_prompt_list())
        prompt_hashes = [self._hash(prompt) for prompt in prompts]

        if self._is_stable_qd_lineage():
            profiles = await asyncio.gather(*[
                self._evaluate_prompt_on_stable_probe(
                    agent_id, prompt, probe_data,
                    self.agents[agent_id].prompt_beam[0].get("metrics", {}).get("mechanism_signature", []),
                )
                for agent_id, prompt in enumerate(prompts)
            ])
            question_hashes = profiles[0]["question_hashes"] if profiles else []
            gold_answers = profiles[0]["gold_answers"] if profiles else []
            summary = team_quality_metrics(
                profiles, gold_answers, question_hashes,
                vote_fn=plurality_vote_with_diagnostics, match_fn=self.task_spec.match_answer,
                tie_break_method=self.cfg.vote_tie_break, seed=self.cfg.seed,
            )
            behavior_profiles = build_team_behavior_profiles(summary["answer_vectors"], summary["correctness_vectors"])
            for profile, behavior in zip(profiles, behavior_profiles):
                self.behavior_profile_by_prompt_hash[profile["prompt_hash"]] = dict(behavior)
            record = {
                "probe_name": str(probe_name), "probe_source": "optimization_train", "epoch": int(epoch),
                "probe_size": len(probe_data), "question_hashes": question_hashes,
                "active_prompt_hashes": prompt_hashes, "per_agent_acc": summary["per_agent_acc"],
                "mean_individual_acc": summary["mean_individual_acc"],
                "min_individual_acc": min(summary["per_agent_acc"], default=0.0),
                "bottom2_mean_acc": summary["bottom2_mean_acc"],
                "bottom3_mean_acc": float(np.mean(sorted(summary["per_agent_acc"])[:3])) if summary["per_agent_acc"] else 0.0,
                "max_individual_acc": max(summary["per_agent_acc"], default=0.0),
                "individual_acc_std": float(np.std(summary["per_agent_acc"])) if summary["per_agent_acc"] else 0.0,
                "best_minus_worst_gap": max(summary["per_agent_acc"], default=0.0) - min(summary["per_agent_acc"], default=0.0),
                "best_minus_bottom2_gap": max(summary["per_agent_acc"], default=0.0) - summary["bottom2_mean_acc"],
                "coverage_depth_c1": summary["coverage_depth_c1"], "coverage_depth_c2": summary["coverage_depth_c2"],
                **{f"coverage_depth_c{depth}": float(np.mean([sum(row) >= depth for row in zip(*summary["correctness_vectors"])])) if summary["correctness_vectors"] else 0.0 for depth in range(3, 6)},
                "behavior_profiles": behavior_profiles,
            }
            self.behavior_profile_history.append({"epoch": epoch, "probe_name": probe_name, "profiles": behavior_profiles})
            self._flush_jsonl("behavior_profile_history.jsonl", [self.behavior_profile_history[-1]])
            self.competence_probe_history.append(dict(record))
            self._flush_jsonl("competence_probe_history.jsonl", [record])
            return record

        async def evaluate_one(index: int, example: Dict[str, str]) -> Dict[str, Any]:
            question = example["question"]
            gold = self.task_spec.parse_gold(example["answer"], question)
            traces, answers = await self.solve_with_prompts(question, prompts)
            question_hash = self._hash(question)
            self._record_solver_rollouts(
                question_hash, prompts, traces, answers, source=f"competence_probe_{probe_name}"
            )
            return {
                "index": index,
                "question_hash": question_hash,
                **self.compute_rollout_metrics(traces, answers, gold, prompts, question_hash=question_hash),
            }

        rows = await asyncio.gather(*[
            evaluate_one(index, example) for index, example in enumerate(probe_data)
        ])
        summary = self._summarize_rollout_rows(rows)
        record = {
            "probe_name": str(probe_name),
            "probe_source": "optimization_train",
            "epoch": int(epoch),
            "probe_size": len(probe_data),
            "question_hashes": [str(row["question_hash"]) for row in rows],
            "active_prompt_hashes": prompt_hashes,
            **{key: summary.get(key) for key in (
                "per_agent_acc", "mean_individual_acc", "min_individual_acc",
                "bottom2_mean_acc", "bottom3_mean_acc", "max_individual_acc",
                "individual_acc_std", "best_minus_worst_gap", "best_minus_bottom2_gap",
                "coverage_depth_c1", "coverage_depth_c2", "coverage_depth_c3",
                "coverage_depth_c4", "coverage_depth_c5",
            )},
        }
        self.competence_probe_history.append(dict(record))
        with open(os.path.join(self.cfg.out_dir, "competence_probe_history.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def _capability_specialization_diagnostics(self) -> Dict[str, Any]:
        profiles = [dict(getattr(agent, "capability_profile", {}) or {}) for agent in self.agents]
        families = sorted({str(key) for profile in profiles for key in profile})
        top = []
        for profile in profiles:
            top.append(max(profile, key=lambda key: float(profile[key])) if profile else "")
        nonempty_top = [value for value in top if value]
        counts = {value: nonempty_top.count(value) for value in sorted(set(nonempty_top))}
        total = len(nonempty_top)
        shares = [count / total for count in counts.values()] if total else []
        cosines = []
        for left in range(len(profiles)):
            for right in range(left + 1, len(profiles)):
                a = np.array([float(profiles[left].get(key, 0.0) or 0.0) for key in families])
                b = np.array([float(profiles[right].get(key, 0.0) or 0.0) for key in families])
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                if denom > 0.0:
                    cosines.append(float(np.dot(a, b) / denom))
        return {
            "top_capability_family_per_agent": top,
            "distinct_top_capability_family_count": len(set(nonempty_top)),
            "dominant_capability_family_share": max(shares, default=0.0),
            "capability_family_hhi": sum(value * value for value in shares),
            "mean_pairwise_capability_profile_cosine": float(np.mean(cosines)) if cosines else 0.0,
        }

    def _summarize_rollout_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        individual_matrix = [list(r.get("individual_correct", [])) for r in rows]
        flat_individual = [int(x) for row in individual_matrix for x in row]
        per_agent_acc = []
        agent_count = max((len(row) for row in individual_matrix), default=0)
        for agent_id in range(agent_count):
            vals = [int(row[agent_id]) for row in individual_matrix if agent_id < len(row)]
            per_agent_acc.append(float(np.mean(vals)) if vals else 0.0)
        vote_acc = float(np.mean([r.get("vote_correct", 0) for r in rows])) if rows else 0.0
        plurality_vote_acc = float(np.mean([
            r.get("plurality_vote_correct", r.get("majority_vote_correct", r.get("vote_correct", 0)))
            for r in rows
        ])) if rows else 0.0
        oracle_acc = float(np.mean([1 if any(int(x) for x in r.get("individual_correct", [])) else 0 for r in rows])) if rows else 0.0
        rescue_available_rate = float(
            np.mean([
                1 if int(r.get("vote_correct", 0)) == 0 and any(int(x) for x in r.get("individual_correct", [])) else 0
                for r in rows
            ])
        ) if rows else 0.0
        correct_disagreement_rate = float(
            np.mean([
                1
                if len({str(a).strip() for a in r.get("vote_counts", {}).keys() if str(a).strip()}) > 1
                and any(int(x) for x in r.get("individual_correct", []))
                else 0
                for r in rows
            ])
        ) if rows else 0.0
        pair_double_fault: List[float] = []
        pair_covariance: List[float] = []
        if agent_count >= 2 and rows:
            for left in range(agent_count):
                for right in range(left + 1, agent_count):
                    pairs = [row for row in individual_matrix if left < len(row) and right < len(row)]
                    if not pairs:
                        continue
                    left_errors = np.array([1.0 - float(row[left]) for row in pairs], dtype=float)
                    right_errors = np.array([1.0 - float(row[right]) for row in pairs], dtype=float)
                    pair_double_fault.append(float(np.mean(left_errors * right_errors)))
                    pair_covariance.append(float(np.mean(left_errors * right_errors) - np.mean(left_errors) * np.mean(right_errors)))
        same_wrong_pair_values = []
        dominant_wrong_sizes = []
        boundary_conditional_errors = []
        pivotal_fix_values = []
        pivotal_hold_values = []
        shared_rescue_values = []
        shared_creation_values = []
        correct_depths = []
        plurality_opportunity_values: List[int] = []
        plurality_hold_values: List[int] = []
        for row in rows:
            flags = [int(value) for value in row.get("individual_correct", [])]
            n = len(flags)
            correct_count = sum(flags)
            wrong_count = n - correct_count
            correct_depths.append(correct_count)
            plurality_opportunity_values.extend(
                int(value) for value in row.get("plurality_pivotal_fix_opportunity_per_agent", [])
            )
            plurality_hold_values.extend(
                int(value) for value in row.get("plurality_pivotal_hold_per_agent", [])
            )
            largest_wrong = int(row.get("largest_wrong_vote_count", 0) or 0)
            dominant_wrong_sizes.append(largest_wrong)
            vote_counts = row.get("vote_counts", {}) if isinstance(row.get("vote_counts", {}), dict) else {}
            all_same_pairs = sum(int(count) * (int(count) - 1) / 2 for count in vote_counts.values())
            gold_same_pairs = correct_count * (correct_count - 1) / 2
            same_wrong_pair_values.append(
                max(0.0, all_same_pairs - gold_same_pairs) / max(1.0, n * (n - 1) / 2)
            )
            per_row_pivotal_fix = []
            per_row_pivotal_hold = []
            per_row_boundary_error = []
            per_row_shared_rescue = []
            per_row_shared_creation = []
            for agent_id, correct in enumerate(flags):
                peer_wrong = wrong_count - int(not correct)
                near_shared_boundary = peer_wrong >= max(1, (n - 1) // 2)
                if near_shared_boundary:
                    per_row_boundary_error.append(float(not correct))
                    per_row_shared_rescue.append(float(correct))
                    per_row_shared_creation.append(float(not correct))
                if not correct:
                    per_row_pivotal_fix.append(float(
                        (not bool(row.get("vote_correct", 0)))
                        and correct_count + 1 > largest_wrong
                    ))
                else:
                    per_row_pivotal_hold.append(float(correct_count - 1 <= largest_wrong))
            if per_row_pivotal_fix:
                pivotal_fix_values.append(float(np.mean(per_row_pivotal_fix)))
            if per_row_pivotal_hold:
                pivotal_hold_values.append(float(np.mean(per_row_pivotal_hold)))
            if per_row_boundary_error:
                boundary_conditional_errors.append(float(np.mean(per_row_boundary_error)))
                shared_rescue_values.append(float(np.mean(per_row_shared_rescue)))
                shared_creation_values.append(float(np.mean(per_row_shared_creation)))
        triple_joint_error_rate = float(np.mean([int((agent_count - depth) >= 3) for depth in correct_depths])) if correct_depths else 0.0
        shared_rescue_rate = float(np.mean(shared_rescue_values)) if shared_rescue_values else 0.0
        shared_creation_rate = float(np.mean(shared_creation_values)) if shared_creation_values else 0.0
        ordered_acc = sorted(per_agent_acc)
        min_acc = ordered_acc[0] if ordered_acc else 0.0
        bottom2 = float(np.mean(ordered_acc[: min(2, len(ordered_acc))])) if ordered_acc else 0.0
        bottom3 = float(np.mean(ordered_acc[: min(3, len(ordered_acc))])) if ordered_acc else 0.0
        max_acc = ordered_acc[-1] if ordered_acc else 0.0
        minority_rescue_counts = [0 for _ in range(agent_count)]
        unique_correct_counts = [0 for _ in range(agent_count)]
        for row in rows:
            flags = [int(value) for value in row.get("individual_correct", [])]
            for agent_id, correct in enumerate(flags):
                if correct and not int(row.get("vote_correct", 0)):
                    minority_rescue_counts[agent_id] += 1
                if correct and sum(flags) == 1:
                    unique_correct_counts[agent_id] += 1
        rescue_total = sum(minority_rescue_counts)
        rescue_shares = [count / rescue_total if rescue_total else 0.0 for count in minority_rescue_counts]
        result = {
            "size": len(rows),
            "num_test_samples": len(rows),
            "vote_acc": vote_acc,
            "plurality_vote_acc": plurality_vote_acc,
            "majority_vote_acc": float(np.mean([r.get("majority_vote_correct", r.get("vote_correct", 0)) for r in rows])) if rows else 0.0,
            "weighted_vote_acc": float(np.mean([r.get("weighted_vote_correct", 0) for r in rows])) if rows else 0.0,
            "mean_individual_acc": float(np.mean(flat_individual)) if flat_individual else 0.0,
            "best_individual_acc": float(max(per_agent_acc)) if per_agent_acc else 0.0,
            "per_agent_acc": per_agent_acc,
            "min_individual_acc": min_acc,
            "bottom2_mean_acc": bottom2,
            "bottom3_mean_acc": bottom3,
            "max_individual_acc": max_acc,
            "individual_acc_std": float(np.std(per_agent_acc)) if per_agent_acc else 0.0,
            "best_minus_worst_gap": max_acc - min_acc,
            "best_minus_bottom2_gap": max_acc - bottom2,
            "minority_rescue_count_per_agent": minority_rescue_counts,
            "unique_correct_count_per_agent": unique_correct_counts,
            "minority_rescue_share_per_agent": rescue_shares,
            "max_minority_rescue_share": max(rescue_shares, default=0.0),
            "minority_rescue_hhi": sum(value * value for value in rescue_shares),
            "oracle_acc": oracle_acc,
            "all_wrong_rate": 1.0 - oracle_acc,
            "aggregation_gap": float(oracle_acc - plurality_vote_acc),
            "oracle_minus_plurality_vote": float(oracle_acc - plurality_vote_acc),
            "rescue_available_rate": rescue_available_rate,
            "correct_disagreement_rate": correct_disagreement_rate,
            "mean_useful_diversity": float(np.mean([r.get("useful_diversity", 0.0) for r in rows])) if rows else 0.0,
            "mean_vote_margin": float(np.mean([r.get("normalized_vote_margin", -1.0) for r in rows])) if rows else -1.0,
            "mean_plurality_margin_votes": float(np.mean([r.get("plurality_margin_votes", 0.0) for r in rows])) if rows else 0.0,
            "mean_normalized_plurality_margin": float(np.mean([r.get("normalized_plurality_margin", -1.0) for r in rows])) if rows else -1.0,
            "strict_plurality_win_rate": float(np.mean([int(bool(r.get("strict_plurality_win", False))) for r in rows])) if rows else 0.0,
            "plurality_top_tie_rate": float(np.mean([int(bool(r.get("plurality_gold_top_tied", False))) for r in rows])) if rows else 0.0,
            "plurality_pivotal_fix_opportunity_rate": float(np.mean(plurality_opportunity_values)) if plurality_opportunity_values else 0.0,
            "plurality_pivotal_fix_rate": float(np.mean(plurality_hold_values)) if plurality_hold_values else 0.0,
            "plurality_pivotal_hold_rate": float(np.mean(plurality_hold_values)) if plurality_hold_values else 0.0,
            "mean_boundary_useful_diversity": float(np.mean([r.get("boundary_useful_diversity", 0.0) for r in rows])) if rows else 0.0,
            "aggregation_mode": str(getattr(self.cfg, "aggregation_mode", "majority") or "majority"),
            "requested_aggregation_mode": str(getattr(self.cfg, "aggregation_mode", "majority") or "majority"),
            "effective_aggregation_mode": canonical_aggregation_mode(str(getattr(self.cfg, "aggregation_mode", "majority") or "majority")),
            "plurality_boundary_version": PLURALITY_BOUNDARY_VERSION,
            "vote_tie_rate": float(np.mean([1 if r.get("vote_tie", False) else 0 for r in rows])) if rows else 0.0,
            "mean_embedding_diversity": float(np.mean([r.get("embedding_diversity", 0.0) for r in rows])) if rows else 0.0,
            "mean_embedding_overlap": float(np.mean([r.get("mean_embedding_overlap", 0.0) for r in rows])) if rows else 0.0,
            "mean_invalid_rate": float(np.mean([r.get("invalid_rate", 0.0) for r in rows])) if rows else 0.0,
            "mean_pairwise_double_fault": float(np.mean(pair_double_fault)) if pair_double_fault else 0.0,
            "mean_pairwise_error_covariance": float(np.mean(pair_covariance)) if pair_covariance else 0.0,
            "same_wrong_pair_rate": float(np.mean(same_wrong_pair_values)) if same_wrong_pair_values else 0.0,
            "triple_joint_error_rate": triple_joint_error_rate,
            "majority_failure_tail_rate": float(np.mean([int((agent_count - depth) >= ((agent_count // 2) + 1)) for depth in correct_depths])) if correct_depths else 0.0,
            **{
                f"coverage_depth_c{depth}": float(np.mean([int(value >= depth) for value in correct_depths])) if correct_depths else 0.0
                for depth in range(1, 6)
            },
            **{f"correct_agent_count_{depth}": int(sum(value == depth for value in correct_depths)) for depth in range(6)},
            "c1_minus_c2": float(np.mean([int(value >= 1) - int(value >= 2) for value in correct_depths])) if correct_depths else 0.0,
            "c2_minus_c3": float(np.mean([int(value >= 2) - int(value >= 3) for value in correct_depths])) if correct_depths else 0.0,
            "c2_minus_plurality_vote": float(np.mean([int(value >= 2) for value in correct_depths])) - plurality_vote_acc if correct_depths else 0.0,
            "c3_minus_plurality_vote": float(np.mean([int(value >= 3) for value in correct_depths])) - plurality_vote_acc if correct_depths else 0.0,
            "specialization_strength_final": float(getattr(self, "specialization_strength", 0.0)),
            "mean_specialization_strength": float(np.mean(getattr(self, "specialization_strength_history", [0.0]))) if getattr(self, "specialization_strength_history", None) else 0.0,
            "first_nonzero_specialization_epoch": getattr(self, "first_nonzero_specialization_epoch", None),
            "effective_specialization_epoch_count": int(getattr(self, "effective_specialization_epoch_count", 0)),
            "max_specialization_strength": max(getattr(self, "specialization_strength_history", [0.0]) or [0.0]),
            "progressive_stage_exercised": int(getattr(self, "effective_specialization_epoch_count", 0)) >= int(getattr(self.cfg, "competence_min_effective_specialization_epochs", 1)),
            "progressive_stage_not_exercised_reason": (
                "" if int(getattr(self, "effective_specialization_epoch_count", 0)) >= int(getattr(self.cfg, "competence_min_effective_specialization_epochs", 1))
                else "activation_after_final_epoch" if float(getattr(self, "specialization_strength", 0.0)) > 0.0
                else "never_activated"
            ),
            "depth1_guard_rejection_count": int(getattr(self, "depth1_guard_rejection_count", 0)),
            "catastrophic_accuracy_guard_rejection_count": int(getattr(self, "catastrophic_accuracy_guard_rejection_count", 0)),
            "soft_error_dependence_penalty_count": int(getattr(self, "soft_error_dependence_penalty_count", 0)),
            "soft_cycle_penalty_count": int(getattr(self, "soft_cycle_penalty_count", 0)),
            "soft_mechanism_shift_penalty_count": int(getattr(self, "soft_mechanism_shift_penalty_count", 0)),
            "exploration_candidate_count": int(getattr(self, "exploration_candidate_count", 0)),
            "exploration_slot_occupancy_rate": float(np.clip(
                float(getattr(self, "exploration_slot_occupancy_count", 0))
                / max(1, int(getattr(self, "total_agent_update_count", 0) or len(getattr(self, "mechanism_signature_history", [])))),
                0.0, 1.0,
            )),
            "exploration_to_active_conversion_count": int(getattr(self, "exploration_to_active_conversion_count", 0)),
            "prompt_overlength_rejection_count": int(getattr(self, "prompt_overlength_rejection_count", 0)),
            "truncated_prompt_count": int(getattr(self, "truncated_prompt_count", 0)),
            "mean_boundary_conditional_error": float(np.mean(boundary_conditional_errors)) if boundary_conditional_errors else 0.0,
            "mean_pivotal_fix_rate": float(np.mean(pivotal_fix_values)) if pivotal_fix_values else 0.0,
            "mean_pivotal_hold_rate": float(np.mean(pivotal_hold_values)) if pivotal_hold_values else 0.0,
            "shared_error_rescue_rate": shared_rescue_rate,
            "shared_error_creation_rate": shared_creation_rate,
            "boundary_shared_error_net_gain": shared_rescue_rate - 1.5 * shared_creation_rate,
            "dominant_wrong_cluster_size": float(np.mean(dominant_wrong_sizes)) if dominant_wrong_sizes else 0.0,
            "gold_vs_largest_wrong_margin": float(np.mean([r.get("normalized_vote_margin", -1.0) for r in rows])) if rows else -1.0,
        }
        if self._residual_specialization_enabled():
            result.update({
                "capability_profile_per_agent": [dict(agent.capability_profile) for agent in self.agents],
                "vote_context_profile_per_agent": [dict(agent.vote_context_profile) for agent in self.agents],
                "capability_profile_update_count_per_agent": [int(agent.capability_profile_update_count) for agent in self.agents],
                **self._capability_specialization_diagnostics(),
            })
        if self._is_v82_hybrid():
            final_signatures = []
            for agent in self.agents:
                metrics = agent.prompt_beam[0].get("metrics", {}) if agent.prompt_beam else {}
                signature = list(metrics.get("mechanism_signature", []))
                if not signature:
                    signature = list(self.mechanism_signature_by_prompt_hash.get(
                        self._normalized_prompt_hash(agent.current_prompt), []
                    ))
                final_signatures.append(signature)
            encoded = [json.dumps(value, ensure_ascii=True, separators=(",", ":")) for value in final_signatures]
            counts = Counter(encoded)
            pair_distances = [
                mechanism_signature_distance(final_signatures[left], final_signatures[right])
                for left in range(len(final_signatures))
                for right in range(left + 1, len(final_signatures))
            ]
            result.update({
                "distinct_final_mechanism_signature_count": len(counts),
                "dominant_final_mechanism_signature_share": max(counts.values(), default=0) / max(1, len(final_signatures)),
                "mean_pairwise_mechanism_signature_distance": float(np.mean(pair_distances)) if pair_distances else 0.0,
                "final_mechanism_signatures": final_signatures,
            })
        if self._is_stable_qd_lineage():
            latest = dict(getattr(self, "latest_joint_team_metrics", {}) or {})
            statuses = [str(agent.lineage_state.get("lineage_status", "uncommitted")) for agent in self.agents]
            lineage_drifts = []
            for agent in self.agents:
                state = agent.lineage_state
                lineage_drifts.append(0.0 if state.get("lineage_status") != "committed" else float(
                    state.get("last_lineage_drift", 0.0) or 0.0
                ))
            mean_behavior = float(latest.get("mean_behavior_distance", 0.0) or 0.0)
            min_behavior = float(latest.get("min_behavior_distance", 0.0) or 0.0)
            mean_mechanism = float(latest.get("mean_mechanism_distance", 0.0) or 0.0)
            mean_drift = float(np.mean(lineage_drifts)) if lineage_drifts else 0.0
            task_rate = float(np.clip(self.task_repair_niche_occupancy_count / max(1, self.total_agent_update_count), 0.0, 1.0))
            mechanism_rate = float(np.clip(self.mechanism_niche_occupancy_count / max(1, self.total_agent_update_count), 0.0, 1.0))
            exploration_rate = float(result["exploration_slot_occupancy_rate"])
            assert 0.0 <= exploration_rate <= 1.0
            result.update({
                "mean_inter_agent_behavior_distance": mean_behavior,
                "min_inter_agent_behavior_distance": min_behavior,
                "mean_inter_agent_mechanism_distance": mean_mechanism,
                "mean_intra_agent_lineage_drift": mean_drift,
                "max_intra_agent_lineage_drift": max(lineage_drifts, default=0.0),
                "stable_specialization_score": mean_behavior + 0.5 * min_behavior + 0.25 * mean_mechanism - 0.5 * mean_drift,
                "uncommitted_agent_count": statuses.count("uncommitted"),
                "provisional_agent_count": statuses.count("provisional"),
                "committed_agent_count": statuses.count("committed"),
                "lineage_commit_count": sum(int(agent.lineage_state.get("lineage_commit_count", 0)) for agent in self.agents),
                "lineage_switch_attempt_count": sum(int(agent.lineage_state.get("lineage_switch_attempt_count", 0)) for agent in self.agents),
                "lineage_switch_commit_count": sum(int(agent.lineage_state.get("lineage_switch_commit_count", 0)) for agent in self.agents),
                "lineage_switch_cancel_count": sum(int(agent.lineage_state.get("lineage_switch_cancel_count", 0)) for agent in self.agents),
                "lineage_committed_but_not_exercised": sum(
                    int(
                        agent.lineage_state.get("lineage_status") == "committed"
                        and int(agent.lineage_state.get("lineage_anchor_epoch", -1)) >= int(self.cfg.epochs)
                    )
                    for agent in self.agents
                ),
                "peer_collapse_soft_count": int(self.peer_collapse_soft_count),
                "peer_collapse_hard_rejection_count": int(self.peer_collapse_hard_rejection_count),
                "joint_team_combination_count": int(latest.get("combination_count", 0) or 0),
                "joint_team_feasible_count": int(latest.get("feasible_count", 0) or 0),
                "joint_team_quality_frontier_count": int(latest.get("quality_frontier_count", 0) or 0),
                "joint_team_quality_floor_feasible_count": int(latest.get("quality_floor_feasible_count", latest.get("feasible_count", 0)) or 0),
                "joint_team_final_candidate_count": int(latest.get("final_candidate_team_count", latest.get("quality_frontier_count", 0)) or 0),
                "joint_team_change_limit_rejection_count": int(latest.get("combination_rejected_by_change_limit_count", 0) or 0),
                "joint_team_fold_quality_rejection_count": int(latest.get("fold_quality_rejection_count", 0) or 0),
                "joint_team_selected_diversity_score": float(latest.get("team_diversity_score", 0.0) or 0.0),
                "joint_team_selected_stability_score": float(latest.get("stable_team_score", 0.0) or 0.0),
                "active_from_incumbent_count": list(latest.get("selected_beam_sources", [])).count("incumbent"),
                "active_from_task_repair_niche_count": list(latest.get("selected_beam_sources", [])).count("task_repair_niche"),
                "active_from_mechanism_niche_count": list(latest.get("selected_beam_sources", [])).count("mechanism_niche"),
                "mechanism_niche_occupancy_rate": mechanism_rate,
                "task_repair_niche_occupancy_rate": task_rate,
                "candidate_starvation_count": int(self.candidate_starvation_count),
                "mechanism_starvation_count": int(self.mechanism_starvation_count),
                "search_branch_starvation_count": int(self.search_branch_starvation_count),
                "probation_to_safe_conversion_count": int(self.probation_to_safe_conversion_count),
                "probation_expired_count": int(self.probation_expired_count),
                "refill_requirements_unmet_count": int(self.refill_requirements_unmet_count),
                "method_version": self.cfg.method_version,
                "active_team_selector_version": self.cfg.active_team_selector_version,
                "lineage_policy_version": self.cfg.lineage_policy_version,
                "mechanism_distance_version": self.cfg.mechanism_distance_version,
                "candidate_refill_version": self.cfg.candidate_refill_version,
                "archive_policy_version": self.cfg.archive_policy_version,
                "joint_quality_filter_version": self.cfg.joint_quality_filter_version,
                "probe_stability_version": self.cfg.probe_stability_version,
                "parent_selection_version": self.cfg.parent_selection_version,
            })
        initial_probe = dict(getattr(self, "initial_competence_probe_metrics", {}) or {})
        final_probe = dict(getattr(self, "latest_competence_probe_metrics", {}) or initial_probe)
        if initial_probe:
            for label, key in (
                ("bottom2", "bottom2_mean_acc"), ("mean_acc", "mean_individual_acc"),
                ("c1", "coverage_depth_c1"), ("c2", "coverage_depth_c2"),
            ):
                initial_value = float(initial_probe.get(key, 0.0) or 0.0)
                final_value = float(final_probe.get(key, initial_value) or 0.0)
                result[f"initial_competence_probe_{label}"] = initial_value
                result[f"final_competence_probe_{label}"] = final_value
                result[f"competence_probe_{label}_gain"] = final_value - initial_value
            baseline_gap = float(initial_probe.get("oracle_acc", 0.0) or 0.0) - float(
                initial_probe.get("plurality_vote_acc", initial_probe.get("vote_acc", 0.0)) or 0.0
            )
            initial_c1 = float(initial_probe.get("coverage_depth_c1", 0.0) or 0.0)
            final_c1 = float(final_probe.get("coverage_depth_c1", initial_c1) or 0.0)
            result.update({
                "baseline_aggregation_gap": baseline_gap,
                "oracle_preserving_gap_reduction": bool(
                    float(result.get("aggregation_gap", 0.0)) < baseline_gap
                    and final_c1 >= initial_c1 - float(getattr(self.cfg, "competence_c1_guard_epsilon", 0.01))
                ),
            })
        return result

    async def evaluate_dataset(self, data: List[Dict[str, str]], split_name: str = "test") -> Dict[str, Any]:
        prompts = self._active_prompt_list()

        async def evaluate_one(idx: int, ex: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            q = ex["question"]
            gold = self.task_spec.parse_gold(ex["answer"], q)
            traces, answers = await self.solve_with_prompts(q, prompts)
            question_hash = self._hash(q)
            self._record_solver_rollouts(question_hash, prompts, traces, answers, source=f"{split_name}_rollout")
            metrics = self.compute_rollout_metrics(traces, answers, gold, prompts, question_hash=question_hash)
            row = {"index": idx, "question_hash": question_hash, **metrics}
            agent_correct = [int(x) for x in metrics.get("individual_correct", [])]
            prediction = {
                "index": idx,
                "sample_id": idx,
                "question_hash": question_hash,
                "question": q,
                "vote_answer": metrics.get("vote_answer", ""),
                "plurality_vote_answer": metrics.get("plurality_vote_answer", metrics.get("vote_answer", "")),
                "majority_vote_answer": metrics.get("majority_vote_answer", metrics.get("vote_answer", "")),
                "weighted_vote_answer": metrics.get("weighted_vote_answer", ""),
                "gold": gold,
                "agent_answers": list(answers),
                "agent_correct": agent_correct,
                "vote_correct": int(metrics.get("vote_correct", 0)),
                "plurality_vote_correct": int(metrics.get("plurality_vote_correct", metrics.get("vote_correct", 0))),
                "majority_vote_correct": int(metrics.get("majority_vote_correct", metrics.get("vote_correct", 0))),
                "weighted_vote_correct": int(metrics.get("weighted_vote_correct", 0)),
                "aggregation_mode": metrics.get("aggregation_mode", "majority"),
                "requested_aggregation_mode": metrics.get("requested_aggregation_mode", metrics.get("aggregation_mode", "majority")),
                "effective_aggregation_mode": metrics.get("effective_aggregation_mode", "plurality"),
                "aggregation_fallback": metrics.get("aggregation_fallback", ""),
                "vote_tie": bool(metrics.get("vote_tie", False)),
                "tie_candidates": metrics.get("tie_candidates", []),
                "vote_counts": metrics.get("vote_counts", {}),
                "gold_vote_count": int(metrics.get("gold_vote_count", 0)),
                "largest_wrong_vote_count": int(metrics.get("largest_wrong_vote_count", 0)),
                "plurality_margin_votes": int(metrics.get("plurality_margin_votes", 0)),
                "normalized_plurality_margin": float(metrics.get("normalized_plurality_margin", -1.0)),
                "normalized_vote_margin": float(metrics.get("normalized_vote_margin", -1.0)),
                "boundary_useful_diversity": float(metrics.get("boundary_useful_diversity", 0.0)),
                "tie_break_method": metrics.get("tie_break_method", ""),
                "weighted_vote_scores": metrics.get("weighted_vote_scores", {}),
                "weighted_vote_agent_weights": metrics.get("weighted_vote_agent_weights", []),
                "any_correct": int(metrics.get("any_correct", 0)),
                "useful_diversity": float(metrics.get("useful_diversity", 0.0)),
                "embedding_diversity": float(metrics.get("embedding_diversity", 0.0)),
                "mean_embedding_overlap": float(metrics.get("mean_embedding_overlap", 0.0)),
                "invalid_rate": float(metrics.get("invalid_rate", 0.0)),
                "agents": [
                    {
                        "agent_id": i,
                        "prompt_hash": self._hash(prompts[i]),
                        "trace": traces[i],
                        "answer": answers[i],
                        "correct": agent_correct[i] if i < len(agent_correct) else 0,
                        "invalid": {"invalid": 0, "reasons": ["skipped_accuracy_only"]} if self._is_accuracy_only_mode() else self.rule_invalid_check(traces[i], answers[i]),
                    }
                    for i in range(len(self.agents))
                ],
            }
            return row, prediction

        evaluated = await asyncio.gather(*[evaluate_one(idx, ex) for idx, ex in enumerate(data)])
        evaluated.sort(key=lambda x: int(x[0].get("index", 0)))
        rows = [row for row, _ in evaluated]
        predictions = [prediction for _, prediction in evaluated]
        pred_path = os.path.join(self.cfg.out_dir, f"{split_name}_predictions.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if split_name.startswith("test") or split_name.startswith("val"):
            self.test_trace_history_logs.extend(predictions)
            self.flush_test_trace_history_logs()
        return self._summarize_rollout_rows(rows)

    def save_state(self, name: str, extra: Optional[Dict[str, Any]] = None):
        payload = {
            **self._base_log_fields(),
            "agents": [
                {
                    "agent_id": i,
                    "initial_prompt": a.initial_prompt,
                    "current_prompt": a.current_prompt,
                    "prompt_beam": a.prompt_beam,
                    "history": a.history,
                    "accept_count": a.accept_count,
                    "reject_count": a.reject_count,
                    **a.trajectory_state_dict(),
                }
                for i, a in enumerate(self.agents)
            ],
            "extra": extra or {},
        }
        with open(os.path.join(self.cfg.out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _flush_jsonl(self, filename: str, rows: List[Dict[str, Any]]):
        if not rows:
            return
        path = os.path.join(self.cfg.out_dir, filename)
        with open(path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def flush_update_logs(self):
        self._flush_jsonl("update_logs.jsonl", self.update_logs)
        self.update_logs = []
        self._flush_jsonl("trajectory_events.jsonl", self.trajectory_events)
        self.trajectory_events = []

    def flush_train_step_logs(self):
        self._flush_jsonl("train_step_logs.jsonl", self.train_step_logs)
        self.train_step_logs = []

    def flush_train_trace_history_logs(self):
        self._flush_jsonl("train_trace_history.jsonl", self.train_trace_history_logs)
        self.train_trace_history_logs = []

    def flush_test_trace_history_logs(self):
        self._flush_jsonl("test_trace_history.jsonl", self.test_trace_history_logs)
        self.test_trace_history_logs = []

    def _write_json_snapshot(self, filename: str, payload: Any):
        """Write a replaceable snapshot so a transient Windows handle does not truncate it."""
        path = os.path.join(self.cfg.out_dir, filename)
        tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
        for attempt in range(3):
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, path)
                return
            except OSError:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                if attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def flush_prompt_history(self):
        self._write_json_snapshot("prompt_history.json", self.prompt_history)

    def flush_llm_call_logs(self):
        self._flush_jsonl("llm_calls.jsonl", self.llm_call_logs)
        self.llm_call_logs = []

    def write_cost_summary(self):
        self.cost_summary.update({
            "full_probe_cache_hits": int(getattr(self, "full_probe_cache_hit_count", 0)),
            "full_probe_missing_pair_evaluations": int(getattr(self, "full_probe_missing_pair_evaluation_count", 0)),
            "embedding_cache_hits": int(getattr(self, "mechanism_embedding_cache_hit_count", 0)),
            "embedding_cache_misses": int(getattr(self, "mechanism_embedding_cache_miss_count", 0)),
        })
        self._write_json_snapshot("cost_summary.json", self.cost_summary)


TextualGradientRLSystem = TraceBeamSearchSystem
