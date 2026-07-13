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
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from openai import AsyncOpenAI

from .answer_formats import canonical_answer as canonical_answer_format
from .answer_formats import extract_prediction as extract_prediction_format
from .answer_formats import match_answer as match_answer_format
from .config import Config
from .policy import AgentState
from .tasks import TaskSpec, get_task_spec
from .utils import (
    compute_gold_vote_diagnostics,
    ensure_dir,
    extract_json_obj,
    infer_task_type,
    majority_vote_with_diagnostics,
    normalize_spaces,
    set_seed,
)


PARETO_EPSILON = 1e-12
TCS_AUDIT_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("tcs_audit_context", default={})
EXPERIMENT_PROTOCOL_VERSION = "vote_oriented_v1"


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


def non_dominated_sort(candidates: Sequence[Dict[str, Any]], eps: float = PARETO_EPSILON) -> List[List[int]]:
    """Deterministic non-dominated sorting; returned indices reference the input sequence."""
    remaining = set(range(len(candidates)))
    fronts: List[List[int]] = []
    while remaining:
        front = [
            index
            for index in remaining
            if not any(pareto_dominates(candidates[other], candidates[index], eps) for other in remaining if other != index)
        ]
        front.sort(key=lambda index: str(candidates[index].get("candidate_id", "")))
        fronts.append(front)
        remaining.difference_update(front)
    return fronts


def compute_crowding_distances(candidates: Sequence[Dict[str, Any]], front_indices: Sequence[int]) -> Dict[int, float]:
    """Compute normalized NSGA-style crowding distances for one Pareto front."""
    distances = {index: 0.0 for index in front_indices}
    if len(front_indices) <= 2:
        return {index: float("inf") for index in front_indices}
    objectives = (
        lambda item: _pareto_value(item, "vote_gain_rate"),
        lambda item: -_pareto_value(item, "vote_loss_rate"),
        lambda item: _pareto_value(item, "candidate_target_accuracy"),
    )
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

    def _uses_baseline_candidate_metrics(self) -> bool:
        return self._is_guarded_reward_mode() or self._is_vote_useful_diversity_mode()

    def _uses_vote_pareto_selection(self) -> bool:
        return str(getattr(self.cfg, "candidate_selection_mode", "scalar_reward") or "scalar_reward").lower() == "vote_pareto"

    def _vote_pareto_feasibility(self, metrics: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        weights = self._effective_reward_weights()
        baseline_target = float(metrics.get("baseline_target_accuracy", 0.0) or 0.0)
        candidate_target = float(metrics.get("candidate_target_accuracy", metrics.get("target_agent_accuracy", 0.0)) or 0.0)
        baseline_invalid = float(metrics.get("baseline_invalid_rate", 0.0) or 0.0)
        candidate_invalid = float(metrics.get("candidate_invalid_rate", metrics.get("invalid_rate", 0.0)) or 0.0)
        accuracy_guard_passed = candidate_target >= baseline_target - float(weights.get("accuracy_guard_epsilon", 0.0))
        invalid_guard_passed = candidate_invalid <= baseline_invalid + float(getattr(self.cfg, "invalid_guard_epsilon", 0.0) or 0.0)
        return bool(accuracy_guard_passed), bool(invalid_guard_passed), bool(accuracy_guard_passed and invalid_guard_passed)

    @staticmethod
    def _vote_pareto_active_sort_key(item: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float, int, str]:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        rank = item.get("pareto_rank")
        normalized_rank = int(rank) if isinstance(rank, int) and rank >= 0 else 10**9
        return (
            -float(metrics.get("vote_delta", 0.0) or 0.0),
            float(metrics.get("vote_loss_rate", 0.0) or 0.0),
            -float(metrics.get("vote_gain_rate", 0.0) or 0.0),
            -float(metrics.get("vote_margin_delta", 0.0) or 0.0),
            -float(metrics.get("candidate_target_accuracy", metrics.get("target_agent_accuracy", 0.0)) or 0.0),
            -float(metrics.get("boundary_useful_diversity_delta", 0.0) or 0.0),
            float(metrics.get("candidate_invalid_rate", metrics.get("invalid_rate", 0.0)) or 0.0),
            normalized_rank,
            str(item.get("candidate_id", "")),
        )

    @staticmethod
    def _vote_pareto_crowding_sort_key(item: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float, float, str]:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        distance = float(item.get("pareto_crowding_distance", 0.0) or 0.0)
        return (
            -distance,
            -float(metrics.get("vote_delta", 0.0) or 0.0),
            float(metrics.get("vote_loss_rate", 0.0) or 0.0),
            -float(metrics.get("vote_gain_rate", 0.0) or 0.0),
            -float(metrics.get("vote_margin_delta", 0.0) or 0.0),
            -float(metrics.get("candidate_target_accuracy", metrics.get("target_agent_accuracy", 0.0)) or 0.0),
            -float(metrics.get("boundary_useful_diversity_delta", 0.0) or 0.0),
            float(metrics.get("candidate_invalid_rate", metrics.get("invalid_rate", 0.0)) or 0.0),
            str(item.get("candidate_id", "")),
        )

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

        fronts_by_item = non_dominated_sort(feasible)
        retained: List[Dict[str, Any]] = []
        for rank, front_indices in enumerate(fronts_by_item):
            distances = compute_crowding_distances(feasible, front_indices)
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
                retained.extend(sorted(front, key=self._vote_pareto_crowding_sort_key)[:slots])
                break

        if not retained:
            raise RuntimeError("Vote Pareto selection produced an empty beam")
        retained.sort(key=self._vote_pareto_active_sort_key)
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
            agent.prompt_beam = [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]

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
        }

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
        return {
            "execution_session_id": self._current_execution_session_id(),
            "comparison_task_id": getattr(self.cfg, "comparison_task_id", ""),
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
        }

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
            "optimizer_architecture": getattr(self.cfg, "optimizer_architecture", ""),
            "optimizer_fallback_mode": getattr(self.cfg, "optimizer_fallback_mode", ""),
            "teacher_critic_max_rounds": getattr(self.cfg, "teacher_critic_max_rounds", 0),
            "teacher_question_pass_threshold": getattr(self.cfg, "teacher_question_pass_threshold", 0.0),
            "teacher_critic_use_voting_failure": bool(getattr(self.cfg, "teacher_critic_use_voting_failure", False)),
            "execution_session_id": self.execution_session_id,
            "previous_execution_session_id": self.previous_execution_session_id,
            "experiment_protocol_version": EXPERIMENT_PROTOCOL_VERSION,
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
        if parent_sig and candidate_sig.startswith(parent_sig) and len(candidate_sig) > len(parent_sig) + 40:
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
            return [
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
        schema = {
            "candidates": [
                {
                    "candidate_prompt": "standalone prompt, <= 900 chars",
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
                }
            ]
        }
        return json.dumps(schema, ensure_ascii=False, separators=(",", ":"))

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

    def _structured_fallback_role(self, agent_id: int, index: int, mode: str = "diversity") -> Dict[str, Any]:
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
            return {
                **role,
                "anti_overlap_rule": "Use the named repair procedure because it fixes a target-agent error pattern, not because it sounds different.",
                "validity_checks": ["trace shows the repair procedure", "final answer is explicit", "no sample text is copied"],
                "accuracy_checks": ["repair rule is executed", "final answer is verified before output"],
            }

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
        return {
            **role,
            "anti_overlap_rule": "Use the named procedure explicitly instead of repeating the default decomposition order.",
            "validity_checks": ["trace shows the named procedure", "final answer is explicit", "no sample text is copied"],
            "accuracy_checks": ["compare plausible alternatives", "verify the final choice against the question"],
        }

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
        return majority_vote_with_diagnostics(
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
        majority_vote = self._vote_with_diagnostics(answers, question_hash=question_hash)
        majority_vote_answer = str(majority_vote.get("vote_answer", ""))
        individual_correct = [int(self.task_spec.match_answer(a, gold)) for a in answers]
        majority_vote_correct = int(self.task_spec.match_answer(majority_vote_answer, gold))
        gold_vote_diagnostics = compute_gold_vote_diagnostics(
            answers,
            gold,
            self.task_spec.match_answer,
            len(self.agents),
        )
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
        aggregation_mode = str(getattr(self.cfg, "aggregation_mode", "majority") or "majority").lower()
        aggregation_fallback = ""
        if aggregation_mode == "weighted_vote":
            vote_answer = weighted_vote_answer
            vote_correct = weighted_vote_correct
            vote_tie = bool(weighted_vote.get("weighted_vote_tie", False))
            tie_candidates = list(weighted_vote.get("weighted_tie_candidates", []))
            tie_break_method = str(weighted_vote.get("weighted_tie_break_method", ""))
        else:
            if aggregation_mode == "verifier_select":
                aggregation_fallback = "verifier_select_not_implemented_fallback_majority"
            aggregation_mode = "majority" if aggregation_mode not in {"weighted_vote", "verifier_select"} else aggregation_mode
            vote_answer = majority_vote_answer
            vote_correct = majority_vote_correct
            vote_tie = bool(majority_vote.get("vote_tie", False))
            tie_candidates = list(majority_vote.get("tie_candidates", []))
            tie_break_method = str(majority_vote.get("tie_break_method", ""))
        any_correct = int(any(individual_correct))
        useful_diversity = 0.0 if self._is_accuracy_only_mode() else self._useful_trace_diversity(traces, individual_correct, [int(x) for x in invalids])
        return {
            "vote_answer": vote_answer,
            "vote_correct": vote_correct,
            "individual_correct": individual_correct,
            "vote_tie": vote_tie,
            "tie_candidates": tie_candidates,
            "vote_counts": dict(majority_vote.get("vote_counts", {})),
            "tie_break_method": tie_break_method,
            "aggregation_mode": aggregation_mode,
            "aggregation_fallback": aggregation_fallback,
            "majority_vote_answer": majority_vote_answer,
            "majority_vote_correct": majority_vote_correct,
            "majority_vote_tie": bool(majority_vote.get("vote_tie", False)),
            "majority_tie_candidates": list(majority_vote.get("tie_candidates", [])),
            "majority_vote_counts": dict(majority_vote.get("vote_counts", {})),
            "majority_tie_break_method": str(majority_vote.get("tie_break_method", "")),
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
    ) -> Dict[str, str]:
        invalid_flags = list(rollout.get("invalid_flags", [])) if isinstance(rollout, dict) else []
        invalid = int(invalid_flags[agent_id]) if agent_id < len(invalid_flags) else int(self.rule_invalid_check(target_trace, target_answer).get("invalid", 0))
        text = normalize_spaces(str(target_trace or ""))
        lower = text.lower()
        answer_preview = self._answer_behavior_preview(target_answer)
        target_words = len(re.findall(r"\w+", text))
        peer_words = [len(re.findall(r"\w+", str(t or ""))) for t in peer_traces]
        peer_mean_words = float(np.mean(peer_words)) if peer_words else 0.0

        if invalid or not answer_preview["present"]:
            if not answer_preview["present"] or "final_answer:" not in str(target_trace or ""):
                return {
                    "error_pattern": "invalid_or_missing_final_answer",
                    "repair_hint": "add a final answer audit that emits exactly one answer in the required format",
                }
            return {
                "error_pattern": "format_violation",
                "repair_hint": "check answer format and remove extra alternatives before finalizing",
            }
        if target_words < 35:
            return {
                "error_pattern": "premature_answer",
                "repair_hint": "delay the final answer until after evidence comparison and a short verification step",
            }
        if re.search(r"\b(calculate|compute|equation|number|sum|difference|multiply|divide|symbol|formula)\b", lower):
            if not re.search(r"\b(check|verify|substitut|unit|sanity)\b", lower):
                return {
                    "error_pattern": "calculation_or_symbolic_slip",
                    "repair_hint": "add a numeric or symbolic sanity check before the final answer",
                }
        if re.search(r"\b(option|choice|alternative|candidate)\b", lower) and not re.search(r"\b(eliminate|reject|compare|fail|against)\b", lower):
            return {
                "error_pattern": "insufficient_option_elimination",
                "repair_hint": "force option-by-option elimination before selecting the final answer",
            }
        if re.search(r"\b(constraint|except|unless|only|must|not|qualifier|condition)\b", lower) and not re.search(r"\b(list|check|satisfy|violate)\b", lower):
            return {
                "error_pattern": "missed_constraint",
                "repair_hint": "force the agent to list explicit constraints before selecting an answer",
            }
        if not re.search(r"\b(check|verify|therefore|because|contradiction|assumption|consistent)\b", lower):
            return {
                "error_pattern": "weak_verification",
                "repair_hint": "add a final consistency check against the question before output",
            }
        if peer_mean_words >= max(45.0, float(target_words) * 1.35):
            return {
                "error_pattern": "peer_has_more_specific_reasoning",
                "repair_hint": "require grounding the answer in specific clues rather than generic reasoning",
            }
        generic_terms = len(re.findall(r"\b(careful|think|analyze|reason|solve|answer)\b", lower))
        evidence_terms = len(re.findall(r"\b(because|therefore|constraint|eliminate|verify|assumption|example|case)\b", lower))
        if generic_terms >= 4 and evidence_terms <= 1:
            return {
                "error_pattern": "overly_generic_reasoning",
                "repair_hint": "replace generic reasoning with a concrete evidence-comparison procedure",
            }
        return {
            "error_pattern": "unknown_error_pattern",
            "repair_hint": "use a concrete compare-then-verify procedure before the final answer",
        }

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
            score = (
                3.0 * value(error_counts, agent_id)
                + 2.0 * value(team_wrong_counts, agent_id)
                + 2.0 * value(invalid_rates, agent_id)
                + 2.0 * value(pivotal_fix_counts, agent_id)
                + 1.0 * value(dominant_wrong_counts, agent_id)
            )
            if score > 0.0:
                scored.append((float(score), agent_id))
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
            row_cases = []
            for agent_id in range(len(self.agents)):
                if agent_id >= len(individual):
                    continue
                per_agent_seen[agent_id] += 1
                per_agent_correct[agent_id] += int(individual[agent_id])
                target_invalid = int(invalids[agent_id]) if agent_id < len(invalids) else 0
                if int(individual[agent_id]) and not target_invalid:
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
                if not int(individual[agent_id]) and any(peer_correct_flags):
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
                if agent_id >= len(individual) or individual[agent_id]:
                    continue
                answer = answers[agent_id] if agent_id < len(answers) else ""
                remaining_wrong = dict(wrong_counts)
                if answer and answer in remaining_wrong:
                    remaining_wrong[answer] -= 1
                    if remaining_wrong[answer] <= 0:
                        remaining_wrong.pop(answer, None)
                counterfactual_largest_wrong = max(remaining_wrong.values(), default=0)
                if (not vote_correct or vote_tie) and gold_count + 1 > counterfactual_largest_wrong:
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
                        "role_name": str(fallback["role_name"]),
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
                safe_cases.append(
                    {
                        "case_type": case_type,
                        "target_agent_id": int(case.get("target_agent_id", agent_id) or agent_id),
                        "target_correct": case.get("target_correct", ""),
                        "target_invalid": case.get("target_invalid", ""),
                        "peer_correct_available": case.get("peer_correct_available", ""),
                        "purpose": normalize_spaces(str(case.get("purpose", "")))[:160],
                        "repair_hint": normalize_spaces(str(case.get("repair_hint", "")))[:180],
                        "target_overlap_pressure": case.get("target_overlap_pressure", ""),
                    }
                )
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
            "Return strict JSON only."
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
            if bool(review.get("passed")) and self._safe_float(review.get("score", 0.0), 0.0) >= threshold:
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
        best_idx = 0
        best_score = -1.0
        for idx, review in enumerate(reviews):
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
        prompt_max = int(getattr(self.cfg, "student_candidate_prompt_max_chars", 900) or 900)
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
        prompt_max = int(getattr(self.cfg, "student_candidate_prompt_max_chars", 900) or 900)
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
            "Do not use gold answers.\nDo not include concrete sample text.\nDo not include answer labels from examples.\n"
            "Do not write hard-coded task-specific roles.\nDo not simply ask the solver to 'think more carefully'.\n"
            f"Do not only paraphrase the parent prompt.\n\n{return_mode}"
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
        guiding_question = str(teacher_question.get("socratic_guiding_question", "")) if isinstance(teacher_question, dict) else ""
        diagnostics.update(
            {
                "teacher_question": guiding_question,
                "teacher_question_approved": bool(approved.get("approved", False)),
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
            diagnostics["teacher_question_rejection_reason"] = str(
                last_review.get("rewrite_instruction", "")
                or last_review.get("quality_critique", "")
                or "teacher question failed critic review"
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
        filter_reasons: List[str] = []
        if isinstance(student_candidates, list):
            for item in student_candidates:
                if not isinstance(item, dict):
                    diagnostics["optimizer_schema_filtered_count"] += 1
                    filter_reasons.append("schema")
                    continue
                item = self._truncate_candidate_text_fields(item)
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
                if self._is_redundant_candidate_prompt(parent_prompt, prompt, seen_signatures):
                    diagnostics["optimizer_redundant_filtered_count"] += 1
                    filter_reasons.append("redundant")
                    continue
                if not prompt:
                    diagnostics["optimizer_empty_prompt_count"] += 1
                    filter_reasons.append("empty_prompt")
                    continue
                seen_signatures.add(self._prompt_signature(prompt))
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
                        "role_name": str(fallback["role_name"]),
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
        phase_progress = max(unique_prompt_ratio, update_progress)
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
        return {
            "target_accuracy": self._nonnegative(target_weight),
            "div_delta": self._nonnegative(div_weight),
            "vote_delta": self._nonnegative(vote_delta_weight),
            "vote_margin": self._nonnegative(vote_margin_weight),
            "boundary_diversity": self._nonnegative(boundary_diversity_weight),
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
            eval_prompts = list(peer_prompts)
            while len(eval_prompts) < len(self.agents):
                eval_prompts.append(self.agents[len(eval_prompts)].current_prompt)
            eval_prompts[agent_id] = candidate_prompt
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
                question_hash=self._hash(q),
            )
            target_answer = answers[agent_id] if agent_id < len(answers) else ""
            return {
                "team_accuracy": int(rollout.get("vote_correct", 0)),
                "target_agent_accuracy": int(self.task_spec.match_answer(target_answer, gold)),
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
                **reuse_stats,
            }

        raw = await asyncio.gather(*[run_one(ex) for ex in eval_batch], return_exceptions=True)
        rows = [r for r in raw if isinstance(r, dict)]
        errors = [normalize_spaces(str(r))[:240] for r in raw if isinstance(r, Exception)]
        team_accuracy = self._clip01(float(np.mean([float(r.get("team_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        target_agent_accuracy = self._clip01(float(np.mean([float(r.get("target_agent_accuracy", 0.0)) for r in rows])) if rows else 0.0)
        solver_reuse_hits = int(sum(int(r.get("solver_reuse_hits", 0) or 0) for r in rows))
        solver_reuse_misses = int(sum(int(r.get("solver_reuse_misses", 0) or 0) for r in rows))
        solver_calls = int(sum(int(r.get("solver_calls", 0) or 0) for r in rows))
        solver_reuse_total = int(sum(int(r.get("solver_reuse_total", 0) or 0) for r in rows))
        majority_team_accuracy = self._clip01(float(np.mean([float(r.get("majority_vote_correct", 0.0)) for r in rows])) if rows else 0.0)
        weighted_team_accuracy = self._clip01(float(np.mean([float(r.get("weighted_vote_correct", 0.0)) for r in rows])) if rows else 0.0)
        return {
            "reward": target_agent_accuracy,
            "embedding_diversity": 0.0,
            "mean_embedding_overlap": 0.0,
            "target_overlap_pressure": 0.0,
            "homogeneous_case_count": 0.0,
            "resolved_case_count": 0.0,
            "new_homogeneous_case_count": 0.0,
            "team_accuracy": team_accuracy,
            "majority_team_accuracy": majority_team_accuracy,
            "weighted_team_accuracy": weighted_team_accuracy,
            "aggregation_mode": str(getattr(self.cfg, "aggregation_mode", "majority") or "majority"),
            "target_agent_accuracy": target_agent_accuracy,
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
                row.update(
                    {
                        "baseline_vote_correct": baseline_vote_correct,
                        "candidate_vote_correct": candidate_vote_correct,
                        "baseline_any_correct": baseline_any_correct,
                        "candidate_any_correct": candidate_any_correct,
                        "baseline_individual_correct": [bool(value) for value in baseline_rollout.get("individual_correct", [])],
                        "candidate_individual_correct": [bool(value) for value in rollout.get("individual_correct", [])],
                        "baseline_target_correct": baseline_target_correct,
                        "target_agent_correct": target_agent_correct,
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
            rescue_rate = self._clip01(float(np.mean([float(r.get("rescue", 0.0)) for r in rows])) if rows else 0.0)
            useful_diversity = self._clip01(float(np.mean([float(r.get("target_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
            rescue_useful_diversity = self._clip01(float(np.mean([float(r.get("rescue_useful_diversity", 0.0)) for r in rows])) if rows else 0.0)
            if self._is_vote_useful_diversity_mode():
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
                    **coverage_transitions,
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
        update_attempt_id = self._update_attempt_id(epoch_id, step_id, agent_id)
        beam = getattr(agent, "prompt_beam", []) or [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]
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
            parent_batches = [generation_batches[i % len(generation_batches)] for i in range(requested)]
            parent_jobs.append(
                {
                    "parent_idx": parent_idx,
                    "parent": parent,
                    "parent_prompt": parent_prompt,
                    "parent_id": parent_id,
                    "parent_batches": parent_batches,
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
                    proposals = await self.propose_candidates(
                        agent_id=agent_id,
                        parent_prompt=str(job["parent_prompt"]),
                        overlap_diagnosis=overlap_diagnosis,
                        num_candidates=requested,
                        generation_batches=job["parent_batches"],
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
                        "generation": generation,
                        "source": "optimizer",
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
                        "optimizer_generation_diagnostics": proposal.get("optimizer_generation_diagnostics", {}),
                        "tcs_call_group_id": str(proposal.get("tcs_call_group_id", "") or ""),
                        "execution_session_id": str(proposal.get("execution_session_id", self._current_execution_session_id()) or self._current_execution_session_id()),
                        "update_attempt_id": str(proposal.get("update_attempt_id", update_attempt_id) or update_attempt_id),
                        "proposal": proposal,
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
                }
            )

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
        num_optimizer_candidates = sum(1 for c in candidate_pool if self._is_optimizer_generated_candidate_source(c.get("candidate_source", "")))
        num_fallback_candidates = sum(1 for c in candidate_pool if "fallback" in str(c.get("candidate_source", "")))
        num_existing_beam_candidates = sum(1 for c in candidate_pool if str(c.get("candidate_source", "")) == "existing_beam")
        num_tcs_optimizer_candidates = sum(
            1
            for c in candidate_pool
            if str(c.get("candidate_source", "")) == "teacher_critic_student" and str(c.get("source", "")) == "optimizer"
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
        beam_size = max(1, int(self.cfg.beam_size))
        pareto_summary = {
            "num_pareto_feasible": None,
            "num_pareto_infeasible": None,
            "num_pareto_fronts": None,
            "pareto_front0_size": None,
            "pareto_forced_current_fallback": None,
        }
        if self._uses_vote_pareto_selection():
            selected, pareto_summary = self._select_vote_pareto_beam(evaluated, beam_size, agent.current_prompt)
        else:
            evaluated.sort(key=lambda x: float(x.get("reward", 0.0)), reverse=True)
            selected = evaluated[:beam_size]
            for item in evaluated:
                item["pareto_feasible"] = None
                item["pareto_rank"] = None
                item["pareto_crowding_distance"] = None
                item["pareto_selected"] = None
                item["pareto_forced_fallback"] = None
        top1_candidate_source = str(selected[0].get("candidate_source", "")) if selected else ""
        top1_candidate_pool_source = str(selected[0].get("source", "")) if selected else ""
        selected_by_id = {str(item.get("candidate_id", "")): rank for rank, item in enumerate(selected, start=1)}
        active_candidate_id = str(selected[0].get("candidate_id", "")) if selected else ""
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
        if changed:
            agent.history.append(agent.current_prompt)
            agent.accept_count += 1
        else:
            agent.reject_count += 1

        for item in evaluated:
            metrics = item.get("metrics", {})
            candidate_id = str(item.get("candidate_id", ""))
            rank = selected_by_id.get(candidate_id)
            accepted = rank is not None
            in_top_beam = bool(accepted)
            is_top1 = bool(candidate_id == active_candidate_id)
            active_selection_key = list(self._vote_pareto_active_sort_key(item)) if self._uses_vote_pareto_selection() and accepted else None
            item_diagnostics = self._empty_optimizer_generation_diagnostics()
            if isinstance(item.get("optimizer_generation_diagnostics", {}), dict):
                item_diagnostics.update(item.get("optimizer_generation_diagnostics", {}))
            item_diagnostics["optimizer_underfilled"] = bool(optimizer_underfilled)
            tcs_candidate_metadata = {
                "optimizer_architecture": item_diagnostics.get("optimizer_architecture", ""),
                "candidate_source": item.get("candidate_source", item.get("source", "")),
                "candidate_pool_source": item.get("source", ""),
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
                    "accuracy_guard_passed": bool(metrics.get("accuracy_guard_passed", True)),
                    "invalid_guard_passed": bool(metrics.get("invalid_guard_passed", True)),
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
                    "candidate_source": item.get("candidate_source", item.get("source", "")),
                    "candidate_pool_source": item.get("source", ""),
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
            "updated": bool(changed),
            "candidate_count": len(candidate_pool),
            "generation_batches": generation_batches,
            "baseline_homogeneous_case_count": len(baseline_cases),
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
        self.update_logs.append(
            {
                **self._base_log_fields(),
                "event": "beam_update_summary",
                "epoch": epoch_id,
                "step": step_id,
                "agent_id": agent_id,
                "execution_session_id": self._current_execution_session_id(),
                "update_attempt_id": update_attempt_id,
                "search_mode": "evolutionary_beam",
                "beam_size": beam_size,
                "active_prompt_changed": bool(changed),
                "top1_candidate_source": top1_candidate_source,
                "top1_candidate_pool_source": top1_candidate_pool_source,
                "candidate_count": len(candidate_pool),
                "optimizer_fallback_mode": str(getattr(self.cfg, "optimizer_fallback_mode", "none")),
                "optimizer_parent_concurrency": int(parent_concurrency),
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
        if not self.cfg.beam_refresh_each_epoch or not eval_batch:
            return {"event": "beam_refresh", "enabled": False, "agent_count": 0}
        records = []
        for agent_id, agent in enumerate(self.agents):
            old_scores = [x.get("score") for x in getattr(agent, "prompt_beam", []) if isinstance(x, dict)]
            old_hash = self._hash(agent.current_prompt)
            refreshed = []
            peer_prompts = self._active_prompt_list()
            for item in getattr(agent, "prompt_beam", []) or [self._make_beam_item(agent.current_prompt, None, {}, None, 0)]:
                prompt = str(item.get("prompt", agent.current_prompt))
                metrics = await self.evaluate_candidate_prompt(agent_id, prompt, peer_prompts, eval_batch, role_spec=item.get("metrics", {}))
                refreshed.append(
                    {
                        "candidate_id": str(item.get("id", "")) or self._hash(prompt),
                        "prompt": prompt,
                        "parent_id": item.get("parent_id"),
                        "generation": int(item.get("generation", 0) or 0),
                        "metrics": metrics,
                        "reward": float(metrics.get("reward", 0.0)),
                    }
                )
            if self._uses_vote_pareto_selection():
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
            "majority_vote_correct": int(metrics.get("majority_vote_correct", metrics.get("vote_correct", 0))),
            "majority_vote_answer": metrics.get("majority_vote_answer", metrics.get("vote_answer", "")),
            "weighted_vote_correct": int(metrics.get("weighted_vote_correct", 0)),
            "weighted_vote_answer": metrics.get("weighted_vote_answer", ""),
            "aggregation_mode": metrics.get("aggregation_mode", "majority"),
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

    def _summarize_rollout_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        individual_matrix = [list(r.get("individual_correct", [])) for r in rows]
        flat_individual = [int(x) for row in individual_matrix for x in row]
        per_agent_acc = []
        agent_count = max((len(row) for row in individual_matrix), default=0)
        for agent_id in range(agent_count):
            vals = [int(row[agent_id]) for row in individual_matrix if agent_id < len(row)]
            per_agent_acc.append(float(np.mean(vals)) if vals else 0.0)
        vote_acc = float(np.mean([r.get("vote_correct", 0) for r in rows])) if rows else 0.0
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
        return {
            "size": len(rows),
            "num_test_samples": len(rows),
            "vote_acc": vote_acc,
            "majority_vote_acc": float(np.mean([r.get("majority_vote_correct", r.get("vote_correct", 0)) for r in rows])) if rows else 0.0,
            "weighted_vote_acc": float(np.mean([r.get("weighted_vote_correct", 0) for r in rows])) if rows else 0.0,
            "mean_individual_acc": float(np.mean(flat_individual)) if flat_individual else 0.0,
            "best_individual_acc": float(max(per_agent_acc)) if per_agent_acc else 0.0,
            "per_agent_acc": per_agent_acc,
            "oracle_acc": oracle_acc,
            "aggregation_gap": float(oracle_acc - vote_acc),
            "rescue_available_rate": rescue_available_rate,
            "correct_disagreement_rate": correct_disagreement_rate,
            "mean_useful_diversity": float(np.mean([r.get("useful_diversity", 0.0) for r in rows])) if rows else 0.0,
            "mean_vote_margin": float(np.mean([r.get("normalized_vote_margin", -1.0) for r in rows])) if rows else -1.0,
            "mean_boundary_useful_diversity": float(np.mean([r.get("boundary_useful_diversity", 0.0) for r in rows])) if rows else 0.0,
            "aggregation_mode": str(getattr(self.cfg, "aggregation_mode", "majority") or "majority"),
            "vote_tie_rate": float(np.mean([1 if r.get("vote_tie", False) else 0 for r in rows])) if rows else 0.0,
            "mean_embedding_diversity": float(np.mean([r.get("embedding_diversity", 0.0) for r in rows])) if rows else 0.0,
            "mean_embedding_overlap": float(np.mean([r.get("mean_embedding_overlap", 0.0) for r in rows])) if rows else 0.0,
            "mean_invalid_rate": float(np.mean([r.get("invalid_rate", 0.0) for r in rows])) if rows else 0.0,
        }

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
                "majority_vote_answer": metrics.get("majority_vote_answer", metrics.get("vote_answer", "")),
                "weighted_vote_answer": metrics.get("weighted_vote_answer", ""),
                "gold": gold,
                "agent_answers": list(answers),
                "agent_correct": agent_correct,
                "vote_correct": int(metrics.get("vote_correct", 0)),
                "majority_vote_correct": int(metrics.get("majority_vote_correct", metrics.get("vote_correct", 0))),
                "weighted_vote_correct": int(metrics.get("weighted_vote_correct", 0)),
                "aggregation_mode": metrics.get("aggregation_mode", "majority"),
                "aggregation_fallback": metrics.get("aggregation_fallback", ""),
                "vote_tie": bool(metrics.get("vote_tie", False)),
                "tie_candidates": metrics.get("tie_candidates", []),
                "vote_counts": metrics.get("vote_counts", {}),
                "gold_vote_count": int(metrics.get("gold_vote_count", 0)),
                "largest_wrong_vote_count": int(metrics.get("largest_wrong_vote_count", 0)),
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
        self._write_json_snapshot("cost_summary.json", self.cost_summary)


TextualGradientRLSystem = TraceBeamSearchSystem
