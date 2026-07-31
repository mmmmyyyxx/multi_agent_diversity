from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .candidate_selection import (
    CandidateEvaluation,
    ConstraintDecision,
    StageASelectionDecision,
    candidate_is_acceptable,
    evaluate_constraints,
    individual_accuracy_key,
    member_aware_pareto_front,
    member_first_key,
    stage_a_multichannel_shortlist,
    vote_first_key,
)
from .config import Config
from .diagnosis_aggregation import (
    ANSWER_ROLE_ENCODING_VERSION,
    DIAGNOSIS_AGGREGATION_VERSION,
    PATTERN_SELECTION_VERSION,
    aggregate_probe_diagnosis,
)
from .evaluation.fixed_probe import (
    FixedProbeEvaluator,
    ProbeExample,
    PromptAnswer,
    evaluate_candidate_profile,
    subset_profiles,
)
from .evaluation.validation import (
    DatasetEvaluationRow,
    DatasetMetrics,
    ValidationProbeEvaluator,
    dataset_metrics_from_dict,
)
from .evaluation.prompt_question import PromptQuestionEvaluator
from .evaluation.output_contract import (
    SOLVER_OUTPUT_CONTRACT_VERSION,
    SOLVER_REQUEST_TEMPLATE_VERSION,
    solver_output_contract,
    solver_system_prompt,
)
from .evaluation.persistent_solver_cache import PersistentSolverCache
from .evaluation.solver_output import parse_solver_output
from .llm_client import LLMCallResult, RoleAwareLLMClient
from .member_objectives import member_gain_metrics, team_member_gain_state
from .peer_state import (
    PeerVoteContext,
    TeamVoteState,
    build_peer_vote_context,
    build_team_vote_state,
    soft_vote_utility,
)
from .proposal_memory import (
    ProposalMemoryEntry,
    ProposalMemoryKey,
    SanitizedCandidateSummary,
    assigned_residual_set_hash,
    entry_to_dict,
    feedback_for,
)
from .persistence.artifacts import ArtifactWriter
from .persistence.identity import RunIdentity, solver_request_components, solver_request_identity
from .protocol import CandidateBudgetContract, ExperimentProtocol, experiment_protocol
from .responsibility import (
    MemberAwareRepairOpportunity,
    ResponsibilityState,
    compute_repair_eligibility_sets,
    compute_member_aware_repair_opportunity,
    build_target_selection_decision,
    target_priorities,
)
from .tasks import get_task_spec
from .team_differentiation import (
    g_transition_audit_rows,
    team_behavior_metrics,
    vote_transition_decomposition,
)
from .tcs import (
    CRITIC_SCHEMA_VERSION,
    ROLE_RETRY_POLICY_VERSION,
    STUDENT_SCHEMA_VERSION,
    TEACHER_SCHEMA_VERSION,
    AccuracyDiagnosisContext,
    AnyDiagnosisContext,
    CriticDecision,
    AssignedResidualDiagnosisContext,
    PeerStateDiagnosisContext,
    ProposalFailureFeedback,
    PreviousUpdateOutcome,
    StudentPromptCandidate,
    TCSContextDiagnostics,
    TCS_PROTOCOL_VERSION,
    TEACHER_REVISION_PROTOCOL_VERSION,
    SAMPLE_MEMORIZATION_FILTER_VERSION,
    TeacherRepairPlan,
    build_teacher_revision_request,
    build_critic_request,
    build_student_request,
    build_student_recovery_request,
    build_teacher_regeneration_request,
    build_teacher_request,
    changed_teacher_plan_fields,
    critic_decision_hash,
    contains_supplied_example_text,
    context_payload,
    limit_diagnosis_context,
    parse_critic_decision,
    parse_student_candidates,
    parse_teacher_repair_plan,
    response_truncated,
    serialize_context,
    teacher_repair_plan_hash,
)
from .utils import extract_json_obj, normalize_prompt_text, normalize_spaces
from .versions import (
    CANDIDATE_ACCEPTANCE_VERSION,
    CHECKPOINT_SELECTION_VERSION,
    CHECKPOINT_VERSION,
    EVALUATION_PROTOCOL_VERSION,
    METHOD_VERSION,
    PRESERVATION_POLICY_VERSION,
    PROPOSAL_MEMORY_VERSION,
    RESPONSIBILITY_VERSION,
    STUDENT_INVALID_RECOVERY_VERSION,
    TARGET_SELECTION_VERSION,
    TCS_CONTEXT_VERSION,
    TEST_ISOLATION_VERSION,
)


SOLVER_INVALID_RETRY_POLICY_VERSION = "retry_until_first_valid_v1"
PROMPT_QUESTION_EVALUATOR_VERSION = "prompt_question_recovered_invalid_v2"


@dataclass
class AgentRuntime:
    initial_prompt: str
    current_prompt: str
    previous_active_prompt: str | None = None


@dataclass
class CandidateRuntime:
    student_candidate: StudentPromptCandidate
    prompt: str
    prompt_hash: str
    generation: int
    parent_prompt_hash: str
    repair_plan_hash: str = ""
    stage_a_evaluation: CandidateEvaluation | None = None
    final_evaluation: CandidateEvaluation | None = None
    profile: tuple[PromptAnswer, ...] | None = None
    stage_a_decision: StageASelectionDecision | None = None
    constraint: ConstraintDecision | None = None


@dataclass
class CandidateFunnel:
    parents_considered: int = 0
    teacher_calls: int = 0
    teacher_invalid_responses: int = 0
    teacher_truncated_responses: int = 0
    critic_calls: int = 0
    critic_invalid_responses: int = 0
    critic_truncated_responses: int = 0
    critic_semantic_rejections: int = 0
    critic_approved: int = 0
    student_calls: int = 0
    student_invalid_responses: int = 0
    student_truncated_responses: int = 0
    student_partially_valid_responses: int = 0
    infrastructure_failed_updates: int = 0
    requested_candidate_count: int = 0
    raw_candidate_count: int = 0
    valid_candidate_count: int = 0
    schema_valid_count: int = 0
    sample_memorization_rejected: int = 0
    non_parent_count: int = 0
    deduplicated_count: int = 0
    student_retry_triggered: bool = False
    student_retry_count: int = 0
    student_recovered: bool = False
    student_cycle_exhausted: bool = False
    upstream_regeneration_triggered: bool = False
    upstream_regeneration_count: int = 0
    terminal_student_failure_class: str = ""
    stage_a_requested_size_per_pool: dict[str, int] = field(default_factory=dict)
    stage_a_available_size_per_pool: dict[str, int] = field(default_factory=dict)
    stage_a_selected_size_per_pool: dict[str, int] = field(default_factory=dict)
    stage_a_overlap_removed: int = 0
    actual_stage_a_size: int = 0
    stage_a_evaluated: int = 0
    selected_by_team_vote_channel: int = 0
    selected_by_worst_member_channel: int = 0
    selected_by_mean_member_channel: int = 0
    stage_b_evaluated: int = 0
    constraint_feasible: int = 0
    rejected_target_regression: int = 0
    rejected_team_vote_regression: int = 0
    rejected_no_target_or_vote_progress: int = 0
    rejected_member_objective_regression: int = 0
    rejected_terminal_invalid_regression: int = 0
    acceptable_candidates: int = 0
    accepted_candidate: bool = False
    terminal_failure_class: str = ""
    terminal_failure_role: str = ""


@dataclass(frozen=True)
class StageAPools:
    representative: tuple[int, ...]
    coverage: tuple[int, ...]
    conversion: tuple[int, ...]
    preservation: tuple[int, ...]
    requested_size_per_pool: dict[str, int]
    available_size_per_pool: dict[str, int]
    selected_size_per_pool: dict[str, int]
    overlap_removed: int
    final_unique_size: int

    def indices(self) -> list[int]:
        return [*self.coverage, *self.conversion, *self.preservation, *self.representative]


def _recursive_field_paths(value: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths.update(_recursive_field_paths(child, path))
    elif isinstance(value, (list, tuple)):
        for child in value:
            paths.update(_recursive_field_paths(child, f"{prefix}[]"))
    return paths


def _response_excerpt(value: str, limit: int = 600) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    half = max(1, (limit - 24) // 2)
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def _request_hash(*parts: str) -> str:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class PromptEnsembleOptimizationSystem:
    def __init__(
        self,
        cfg: Config,
        *,
        solver: Callable[[str, int, str], Awaitable[PromptAnswer]] | None = None,
        optimizer_chat: Callable[[str, str, float, int], Awaitable[str]] | None = None,
    ):
        if cfg.training.method_version != METHOD_VERSION:
            raise ValueError(f"Unsupported method_version: {cfg.training.method_version}")
        if cfg.training.agents != 5:
            raise ValueError("member_aware_peer_state_v8 requires exactly five agents")
        if cfg.peer_state.aggregation_mode != "plurality":
            raise ValueError("member_aware_peer_state_v8 requires plurality aggregation")
        if cfg.peer_state.vote_tie_break != "abstain":
            raise ValueError("member_aware_peer_state_v8 requires tie-as-abstain")
        if cfg.peer_state.solver_output_contract_version != SOLVER_OUTPUT_CONTRACT_VERSION:
            raise ValueError(
                "solver_output_contract_version does not match the implemented task contract"
            )
        if cfg.tcs.teacher_critic_max_rounds <= 0:
            raise ValueError("teacher_critic_max_rounds must be positive")
        if cfg.tcs.teacher_json_max_retries < 0:
            raise ValueError("teacher_json_max_retries cannot be negative")
        if cfg.tcs.critic_json_max_retries < 0:
            raise ValueError("critic_json_max_retries cannot be negative")
        if cfg.tcs.student_invalid_max_retries < 0:
            raise ValueError("student_invalid_max_retries cannot be negative")
        if cfg.tcs.student_upstream_regeneration_max_count not in {0, 1}:
            raise ValueError(
                "student_upstream_regeneration_max_count must be zero or one"
            )
        if cfg.tcs.proposal_memory_mode not in {"off", "state_local_v1"}:
            raise ValueError("proposal_memory_mode must be 'off' or 'state_local_v1'")
        if cfg.responsibility.member_catchup_mode not in {"off", "fallback_v1"}:
            raise ValueError("member_catchup_mode must be 'off' or 'fallback_v1'")
        if cfg.responsibility.responsibility_mode != "compact_member_aware_v8":
            raise ValueError(
                "responsibility_mode must be 'compact_member_aware_v8'"
            )
        if not 0 < cfg.tcs.tcs_max_pattern_summaries <= 3:
            raise ValueError("tcs_max_pattern_summaries must be between one and three")
        if not 0 < cfg.tcs.tcs_max_evidence_cases <= 3:
            raise ValueError("tcs_max_evidence_cases must be between one and three")
        if min(
            cfg.tcs.tcs_context_max_chars,
            cfg.tcs.teacher_field_max_chars,
            cfg.tcs.critic_feedback_max_chars,
            cfg.tcs.candidate_prompt_max_chars,
        ) <= 0:
            raise ValueError("TCS character limits must be positive")
        self.cfg = cfg
        self.protocol = self._build_protocol()
        if (
            cfg.tcs.proposal_memory_mode == "state_local_v1"
            and self.protocol.tcs_context_policy
            != "member_aware_responsibility_conditioned"
        ):
            raise ValueError(
                "state_local_v1 proposal memory requires responsibility-conditioned TCS"
            )
        self.task_spec = get_task_spec(cfg.data.task_type)
        prompts = self._initial_prompts()
        self.agents = [AgentRuntime(prompt, prompt) for prompt in prompts]
        self.responsibility_state = ResponsibilityState(
            updates_since_selected_by_agent={agent_id: 0 for agent_id in range(cfg.training.agents)},
            accepted_updates_by_agent={agent_id: 0 for agent_id in range(cfg.training.agents)},
            target_attempt_count_by_agent={agent_id: 0 for agent_id in range(cfg.training.agents)},
        )
        self.history: list[dict[str, Any]] = []
        self.peer_state_history: list[dict[str, Any]] = []
        self.responsibility_assignments: list[dict[str, Any]] = []
        self.member_opportunities: list[dict[str, Any]] = []
        self.g_transition_audit: list[dict[str, Any]] = []
        self.specialization_trajectory: list[dict[str, Any]] = []
        self.candidate_decisions: list[dict[str, Any]] = []
        self.tcs_context_history: list[dict[str, Any]] = []
        self.tcs_rounds: list[dict[str, Any]] = []
        self.student_recovery_observations: list[dict[str, Any]] = []
        self.student_recovery_state: dict[str, Any] = {
            "in_progress": False,
            "update_index": -1,
            "target_agent_id": -1,
            "student_generation_cycle_index": 0,
            "student_attempt_index": 0,
            "upstream_regeneration_count": 0,
        }
        self.solver_invalid_outputs: list[dict[str, Any]] = []
        self.solver_recovery_observations: list[dict[str, Any]] = []
        self._audited_invalid_keys: set[tuple[str, str]] = set()
        self._observed_solver_keys: set[tuple[str, str]] = set()
        self.cached_responsibility_eligibility: dict[str, tuple[int, ...]] = {}
        self.cached_responsibility_assignments: dict[int, list[MemberAwareRepairOpportunity]] = {}
        self.cached_member_opportunities: dict[
            str, tuple[MemberAwareRepairOpportunity, ...]
        ] = {}
        self.team_state_version = 0
        self.responsibility_state_version = -1
        self.responsibility_refresh_count = 0
        self.target_priority_audit: list[dict[str, Any]] = []
        self.responsibility_portfolio_trajectory: list[dict[str, Any]] = []
        self.target_responsibility_context_alignment: list[
            dict[str, Any]
        ] = []
        self.proposal_memory_entries: dict[str, ProposalMemoryEntry] = {}
        self.proposal_memory_events: list[dict[str, Any]] = []
        self.proposal_rotation_trajectory: list[dict[str, Any]] = []
        self._proposal_memory_attempts: dict[int, dict[str, Any]] = {}
        self.proposal_memory_run_id = ""
        self.previous_update_outcomes = {
            agent_id: PreviousUpdateOutcome() for agent_id in range(5)
        }
        self.agent_selection_counts = {agent_id: 0 for agent_id in range(5)}
        self.fixed_probe: FixedProbeEvaluator | None = None
        # Compatibility-only reader state. The active final-state lifecycle never
        # builds or evaluates this probe.
        self.validation_probe: ValidationProbeEvaluator | None = None
        self.validation_state_cache: dict[str, dict[str, Any]] = {}
        self.validation_evaluation_count = 0
        self.validation_reuse_count = 0
        self._compat_validation_selection_completed = False
        self.training_dynamics: list[dict[str, Any]] = []
        self.team_differentiation_trajectory: list[dict[str, Any]] = []
        self.update_transition_decomposition: list[dict[str, Any]] = []
        self.final_test_differentiation: dict[str, Any] = {}
        self.planned_update_count = 0
        self.completed_update_count = 0
        self.training_completed = False
        self.final_state_selection: dict[str, Any] = {}
        self.test_evaluation_count = 0
        self.test_used_for_selection = False
        self.test_used_for_training = False
        self.test_called_before_training_complete = False
        self.selected_test_metrics: dict[str, Any] = {}
        self._last_evaluated_examples: tuple[ProbeExample, ...] = ()
        self._last_evaluated_profiles: list[tuple[PromptAnswer, ...]] = []
        request_identity = solver_request_identity(cfg)
        request_components = solver_request_components(cfg)
        cache_path = str(cfg.persistence.shared_solver_cache_path or "").strip()
        self.shared_solver_cache = (
            PersistentSolverCache(
                cache_path,
                stale_after_seconds=max(
                    1800.0,
                    cfg.persistence.llm_call_timeout
                    * max(1, cfg.persistence.max_retries + cfg.persistence.max_transient_retries),
                ),
            )
            if cache_path
            else None
        )
        self.prompt_question_evaluator = PromptQuestionEvaluator(
            model_request_identity=request_identity,
            parser_version=cfg.peer_state.parser_version,
            temperature=cfg.models.temperature,
            decoding_seed=cfg.training.seed,
            cache_metadata=request_components,
            shared_cache=self.shared_solver_cache,
            observation_callback=self._record_solver_observation,
            version=PROMPT_QUESTION_EVALUATOR_VERSION,
        )
        self.active_profiles: list[tuple[PromptAnswer, ...]] = []
        self.initial_profiles: list[tuple[PromptAnswer, ...]] = []
        self.run_identity: RunIdentity | None = None
        self.artifacts = ArtifactWriter(cfg.persistence.out_dir)
        self._solver_override = solver
        self.llm = RoleAwareLLMClient(cfg, optimizer_chat)
        self.solver_semaphore = asyncio.Semaphore(max(1, cfg.evaluation.eval_solver_call_concurrency))

    def _build_protocol(self) -> ExperimentProtocol:
        budget = CandidateBudgetContract(
            generated_per_update=self.cfg.tcs.num_candidates_per_parent,
            stage_a_channel_top_k=self.cfg.evaluation.stage_a_channel_top_k,
            stage_b_candidate_budget=self.cfg.evaluation.stage_b_candidate_budget,
            representative_size=self.cfg.evaluation.stage_a_representative_size,
            coverage_size=self.cfg.evaluation.stage_a_coverage_size,
            conversion_size=self.cfg.evaluation.stage_a_conversion_size,
            preservation_size=self.cfg.evaluation.stage_a_preservation_size,
        )
        return experiment_protocol(
            self.cfg.training.experiment_setting,
            initialization_mode=self.cfg.training.initialization_mode,
            tie_policy=self.cfg.peer_state.vote_tie_break,
            candidate_budget_contract=budget,
        )

    def _initial_prompts(self) -> list[str]:
        if self.cfg.training.initialization_mode == "shared_identical":
            prompt = normalize_prompt_text(self.cfg.training.shared_prompt)
            if not prompt:
                raise ValueError("shared_prompt must be non-empty")
            return [prompt] * self.cfg.training.agents
        if self.cfg.training.initialization_mode != "provided_prompt_set":
            raise ValueError(f"Unknown initialization mode: {self.cfg.training.initialization_mode}")
        try:
            values = json.loads(self.cfg.training.provided_prompts_json)
        except json.JSONDecodeError as exc:
            raise ValueError("provided_prompts_json is not valid JSON") from exc
        if not isinstance(values, list) or len(values) != self.cfg.training.agents:
            raise ValueError("provided_prompt_set must contain exactly five prompts")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("provided_prompt_set prompts must be non-empty strings")
        prompts = [normalize_prompt_text(value) for value in values]
        if any(not value for value in prompts):
            raise ValueError("provided_prompt_set prompts must be non-empty strings")
        return prompts

    def set_run_identity(self, identity: RunIdentity) -> None:
        if identity.method_version != METHOD_VERSION:
            raise ValueError("run identity method version does not match the system")
        if identity.experiment_setting != self.protocol.name:
            raise ValueError("run identity experiment setting does not match the protocol")
        self.run_identity = identity
        payload = {
            "run_identity": identity.to_dict(),
            "out_dir": str(self.cfg.persistence.out_dir),
        }
        self.proposal_memory_run_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _proposal_memory_key(
        self,
        *,
        target_agent_id: int,
        parent_prompt: str,
        assigned_hashes: set[str],
    ) -> ProposalMemoryKey:
        if self.run_identity is None or not self.proposal_memory_run_id:
            raise RuntimeError("proposal memory requires an exact run identity")
        if any(
            target_agent_id not in self.cached_responsibility_eligibility.get(question_hash, ())
            for question_hash in assigned_hashes
        ):
            raise RuntimeError(
                "proposal memory key contains an ineligible residual"
            )
        return ProposalMemoryKey(
            run_id=self.proposal_memory_run_id,
            team_state_version=self.team_state_version,
            target_agent_id=target_agent_id,
            target_prompt_hash=self.prompt_hash(parent_prompt),
            assigned_residual_set_hash=assigned_residual_set_hash(tuple(assigned_hashes)),
        )

    def _proposal_memory_entry(
        self,
        key: ProposalMemoryKey,
        assigned_hashes: set[str],
    ) -> ProposalMemoryEntry | None:
        entry = self.proposal_memory_entries.get(key.key_hash())
        if entry is None:
            return None
        if entry.key != key or entry.assigned_question_hashes != tuple(sorted(assigned_hashes)):
            raise RuntimeError("proposal memory lifecycle/schema mismatch")
        if any(
            key.target_agent_id not in self.cached_responsibility_eligibility.get(question_hash, ())
            for question_hash in entry.assigned_question_hashes
        ):
            raise RuntimeError(
                "proposal memory contains an ineligible residual"
            )
        return entry

    @staticmethod
    def _evidence_bundle_hash(context: AnyDiagnosisContext) -> str:
        payload = {
            "patterns": [row.pattern_id for row in context.patterns],
            "questions": [row.question_hash for row in context.evidence_cases],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def prompt_hash(prompt: str) -> str:
        return hashlib.sha256(normalize_prompt_text(prompt).encode("utf-8")).hexdigest()

    def normalize_answer(self, answer: str) -> str:
        return self.task_spec.extract_pred(f"FINAL_ANSWER: {answer}", None)

    def match_answer(self, prediction: str, gold: str) -> bool:
        return self.task_spec.match_answer(prediction, gold)

    def _record_solver_observation(
        self,
        prompt_hash: str,
        question_hash: str,
        answer: PromptAnswer,
    ) -> None:
        key = (prompt_hash, question_hash)
        if key in self._observed_solver_keys:
            return
        self._observed_solver_keys.add(key)
        self.solver_recovery_observations.append({
            "prompt_hash": prompt_hash,
            "question_hash": question_hash,
            "request_identity": answer.request_identity
            or self.prompt_question_evaluator.model_request_identity,
            "solver_attempt_count": answer.solver_attempt_count,
            "first_attempt_valid": answer.first_attempt_valid,
            "recovered_from_invalid": answer.recovered_from_invalid,
            "terminal_invalid": answer.terminal_invalid,
            "raw_invalid_attempt_count": answer.raw_invalid_attempt_count,
            "attempt_validity_statuses": list(answer.attempt_validity_statuses),
            "attempt_finish_reasons": list(answer.attempt_finish_reasons),
            "attempt_prompt_tokens": list(answer.attempt_prompt_tokens),
            "attempt_completion_tokens": list(answer.attempt_completion_tokens),
            "attempt_total_tokens": list(answer.attempt_total_tokens),
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "total_tokens": answer.total_tokens,
            "recovery_prompt_tokens": answer.recovery_prompt_tokens,
            "recovery_completion_tokens": answer.recovery_completion_tokens,
            "recovery_total_tokens": answer.recovery_total_tokens,
            "validity_status": answer.validity_status,
            "response_hash": answer.response_hash,
        })
        if answer.valid or key in self._audited_invalid_keys:
            return
        self._audited_invalid_keys.add(key)
        self.solver_invalid_outputs.append({
            "question_hash": question_hash,
            "prompt_hash": prompt_hash,
            "answer_format": self.cfg.data.answer_format,
            "validity_status": answer.validity_status,
            "raw_final_answer_payload": answer.raw_final_answer_payload,
            "final_answer_line_count": answer.final_answer_line_count,
            "response_excerpt": _response_excerpt(answer.trace),
            "response_hash": answer.response_hash,
            "request_identity": answer.request_identity
            or self.prompt_question_evaluator.model_request_identity,
            "terminal_invalid": answer.terminal_invalid,
            "solver_attempt_count": answer.solver_attempt_count,
            "raw_invalid_attempt_count": answer.raw_invalid_attempt_count,
            "created_at": answer.created_at,
        })

    def _current_run_recovery_summary(self) -> dict[str, Any]:
        rows = [
            row for row in self.llm.calls
            if row.get("role") == "solver"
            and int(row.get("solver_invalid_attempt_index", 1)) > 1
        ]
        return {
            "current_run_recovery_api_calls": len(rows),
            "current_run_recovery_prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in rows),
            "current_run_recovery_completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in rows),
            "current_run_recovery_total_tokens": sum(int(row.get("total_tokens", 0)) for row in rows),
        }

    async def _chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
        client_role: str,
        logical_role: str | None = None,
    ) -> LLMCallResult:
        if logical_role is None:
            if client_role == "solver":
                logical_role = "solver"
            elif client_role == "evaluator":
                logical_role = "critic"
            elif system_prompt == "Return strict JSON only.":
                logical_role = "student"
            else:
                logical_role = "teacher"
        return await self.llm.chat_result(
            model, system_prompt, user_prompt, temperature, max_tokens, client_role,
            logical_role,
        )

    async def solve(self, question: str, agent_id: int, prompt: str) -> PromptAnswer:
        request_identity = self.prompt_question_evaluator.model_request_identity
        attempts = []
        system_prompt = solver_system_prompt(prompt, self.cfg.data.answer_format)
        async with self.solver_semaphore:
            for _ in range(1 + max(0, self.cfg.models.solver_invalid_max_retries)):
                if self._solver_override is not None:
                    started = time.time()
                    parsed = await self._solver_override(question, agent_id, prompt)
                    self.llm.record_override_solver(started=started)
                    result = LLMCallResult(
                        text=parsed.trace,
                        prompt_tokens=parsed.prompt_tokens,
                        completion_tokens=parsed.completion_tokens,
                        total_tokens=parsed.total_tokens,
                        latency_seconds=0.0,
                        finish_reason="stop",
                    )
                else:
                    result = await self.llm.chat_result(
                        self.cfg.models.agent_model,
                        system_prompt,
                        question,
                        self.cfg.models.temperature,
                        self.cfg.models.solver_max_tokens,
                        "solver",
                    )
                    parsed = parse_solver_output(
                        result.text,
                        question=question,
                        task_spec=self.task_spec,
                        answer_format=self.cfg.data.answer_format,
                    )
                if self.llm.calls:
                    self.llm.calls[-1]["solver_invalid_attempt_index"] = len(attempts) + 1
                attempts.append((result, parsed))
                if parsed.valid:
                    break
        result, answer = attempts[-1]
        first_result, first_answer = attempts[0]
        invalid_attempts = [item for _, item in attempts if not item.valid]
        return replace(
            answer,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            request_identity=request_identity,
            solver_attempt_count=len(attempts),
            first_attempt_valid=first_answer.valid,
            recovered_from_invalid=(not first_answer.valid and answer.valid),
            terminal_invalid=not answer.valid,
            raw_invalid_attempt_count=len(invalid_attempts),
            attempt_validity_statuses=tuple(item.validity_status for _, item in attempts),
            attempt_finish_reasons=tuple(item.finish_reason for item, _ in attempts),
            attempt_response_hashes=tuple(
                hashlib.sha256(item.text.encode("utf-8")).hexdigest()
                for item, _ in attempts
            ),
            attempt_prompt_tokens=tuple(item.prompt_tokens for item, _ in attempts),
            attempt_completion_tokens=tuple(item.completion_tokens for item, _ in attempts),
            attempt_total_tokens=tuple(item.total_tokens for item, _ in attempts),
            recovery_prompt_tokens=sum(item.prompt_tokens for item, _ in attempts[1:]),
            recovery_completion_tokens=sum(item.completion_tokens for item, _ in attempts[1:]),
            recovery_total_tokens=sum(item.total_tokens for item, _ in attempts[1:]),
        )

    def build_probe(self, data: Sequence[Mapping[str, Any]]) -> FixedProbeEvaluator:
        return FixedProbeEvaluator(
            self._probe_examples(data),
            self.cfg.peer_state.probe_version,
            self.prompt_question_evaluator,
        )

    def build_validation_probe(
        self,
        data: Sequence[Mapping[str, Any]],
    ) -> ValidationProbeEvaluator:
        """Historical-report compatibility; never used by the active run loop."""
        return ValidationProbeEvaluator(
            self._probe_examples(data),
            model_identity=self.cfg.models.agent_model,
            parser_version=self.cfg.peer_state.parser_version,
            temperature=self.cfg.models.temperature,
            seed=self.cfg.training.seed,
            prompt_question_evaluator=self.prompt_question_evaluator,
        )

    def _probe_examples(self, data: Sequence[Mapping[str, Any]]) -> tuple[ProbeExample, ...]:
        return tuple(
            ProbeExample(
                question=str(row["question"]),
                question_hash=hashlib.sha256(normalize_spaces(str(row["question"])).encode("utf-8")).hexdigest(),
                gold_answer=self.task_spec.parse_gold(row["answer"], str(row["question"])),
            )
            for row in data
        )

    async def initialize_fixed_probe(self, data: Sequence[Mapping[str, Any]]) -> None:
        self.fixed_probe = self.build_probe(data)
        self.active_profiles = list(await asyncio.gather(*(
            self.fixed_probe.evaluate_prompt(
                agent_id,
                agent.current_prompt,
                self.prompt_hash(agent.current_prompt),
                self.solve,
            )
            for agent_id, agent in enumerate(self.agents)
        )))
        self.initial_profiles = list(self.active_profiles)

    def _member_correct_counts(
        self,
        profiles: Sequence[Sequence[PromptAnswer]],
    ) -> tuple[int, ...]:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        return tuple(
            sum(
                int(
                    answer.valid
                    and self.match_answer(answer.answer, example.gold_answer)
                )
                for answer, example in zip(
                    profile,
                    self.fixed_probe.examples,
                    strict=True,
                )
            )
            for profile in profiles
        )

    def current_team_member_gain_state(self) -> dict[str, Any]:
        initial_counts = self._member_correct_counts(self.initial_profiles)
        current_counts = self._member_correct_counts(self.active_profiles)
        return asdict(team_member_gain_state(initial_counts, current_counts))

    def frozen_initialization_snapshot(self) -> dict[str, Any]:
        """Hash-only evidence used to prove matched treatment initialization.

        This is intentionally derived after the shared fixed-probe rollout and
        before the first update.  It contains no prompt, question, gold answer,
        or raw model response, so an experiment runner can compare treatments
        without exporting private solver material.
        """
        if self.fixed_probe is None or self.run_identity is None:
            raise RuntimeError("fixed probe and run identity are required for initialization audit")
        states, _, _ = self.current_states_and_opportunities()
        state_rows = [
            {
                "question_hash": state.question_hash,
                "vote_correct": bool(state.vote_correct),
                "oracle_covered": bool(state.gold_vote_count > 0),
                "G": state.gold_vote_count,
                "H": state.largest_wrong_vote_count,
                "M": state.plurality_margin,
                "member_correctness": list(state.team_correctness),
                "member_validity": list(state.team_validity),
            }
            for state in sorted(states, key=lambda value: value.question_hash)
        ]
        behavior = team_behavior_metrics(
            examples=self.fixed_probe.examples,
            profiles=self.active_profiles,
            normalize_answer=self.normalize_answer,
            match_answer=self.match_answer,
            tie_break=self.protocol.tie_policy,
            seed=self.cfg.training.seed,
        )
        initial_prompt_hashes = [self.prompt_hash(agent.initial_prompt) for agent in self.agents]
        member_counts = list(self._member_correct_counts(self.initial_profiles))
        immutable_identity = {
            key: getattr(self.run_identity, key)
            for key in (
                "git_commit", "git_dirty", "manifest_sha256",
                "train_file_sha256", "val_file_sha256", "test_file_sha256",
                "train_question_set_hash", "val_question_set_hash", "test_question_set_hash",
            )
        }
        solver_identity = self.prompt_question_evaluator.identity()
        state_payload = {
            "initial_prompt_hashes": initial_prompt_hashes,
            "initial_member_correct_counts": member_counts,
            "state_rows": state_rows,
            "probe_hash": self.fixed_probe.probe_hash,
            "solver_request_identity": self.prompt_question_evaluator.model_request_identity,
            "solver_identity": solver_identity,
            "immutable_run_identity": immutable_identity,
        }
        initial_train_state_hash = hashlib.sha256(json.dumps(
            state_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        return {
            "snapshot_version": "frozen_initialization_snapshot_v1",
            "initial_prompt_hashes": initial_prompt_hashes,
            "initial_member_correct_counts": member_counts,
            "initial_team_outcome": {
                key: behavior[key]
                for key in (
                    "team_vote_correct_count", "oracle_correct_count", "terminal_invalid_count",
                    "mean_G", "mean_H", "mean_M", "oracle_covered_but_vote_wrong_rate",
                    "per_agent_correct_counts",
                )
            },
            "initial_vote_oracle_ghm_hash": hashlib.sha256(json.dumps(
                state_rows, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            "initial_train_state_hash": initial_train_state_hash,
            "probe_hash": self.fixed_probe.probe_hash,
            "solver_request_identity": self.prompt_question_evaluator.model_request_identity,
            "solver_identity": solver_identity,
            "immutable_run_identity": immutable_identity,
        }

    def current_states_and_opportunities(
        self,
    ) -> tuple[
        tuple[TeamVoteState, ...],
        dict[str, dict[int, PeerVoteContext]],
        dict[str, tuple[MemberAwareRepairOpportunity, ...]],
    ]:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        states: list[TeamVoteState] = []
        contexts: dict[str, dict[int, PeerVoteContext]] = {}
        opportunities: dict[str, tuple[MemberAwareRepairOpportunity, ...]] = {}
        for index, example in enumerate(self.fixed_probe.examples):
            state = build_team_vote_state(
                question_hash=example.question_hash,
                gold_answer=example.gold_answer,
                answers=[profile[index].answer for profile in self.active_profiles],
                valid_vector=[profile[index].valid for profile in self.active_profiles],
                normalize_answer=self.normalize_answer,
                match_answer=self.match_answer,
                tie_break=self.protocol.tie_policy,
                seed=self.cfg.training.seed,
            )
            peer_by_agent = {agent_id: build_peer_vote_context(state, agent_id) for agent_id in range(5)}
            states.append(state)
            contexts[state.question_hash] = peer_by_agent
        for state in states:
            peer_by_agent = contexts[state.question_hash]
            opportunities[state.question_hash] = tuple(
                compute_member_aware_repair_opportunity(
                    team_state=state,
                    peer_context=peer_by_agent[agent_id],
                    tau=self.cfg.peer_state.soft_vote_tau,
                )
                for agent_id in range(5)
            )
        return tuple(states), contexts, opportunities

    def assign_responsibilities(
        self,
    ) -> tuple[dict[str, tuple[int, ...]], dict[int, list[MemberAwareRepairOpportunity]]]:
        if self.responsibility_state_version == self.team_state_version:
            return (
                dict(self.cached_responsibility_eligibility),
                {
                    agent_id: list(rows)
                    for agent_id, rows in self.cached_responsibility_assignments.items()
                },
            )
        states, _, opportunities = self.current_states_and_opportunities()
        self.cached_member_opportunities = dict(opportunities)
        state_by_hash = {state.question_hash: state for state in states}
        eligibility, assigned, eligibility_audits = (
            compute_repair_eligibility_sets(
                team_states=state_by_hash,
                opportunities=opportunities,
                state=self.responsibility_state,
            )
        )
        for question_hash, audit in eligibility_audits.items():
            team_state = state_by_hash[question_hash]
            audit.update({
                "G": team_state.gold_vote_count,
                "H": team_state.largest_wrong_vote_count,
                "M": team_state.plurality_margin,
            })
        rows = [row for values in assigned.values() for row in values]
        self.peer_state_history.extend(asdict(state) for state in states)
        self.responsibility_assignments.append({
            "artifact_schema_version": "compact_vote_margin_responsibility_v1",
            "team_state_version": self.team_state_version,
            "eligible_agents_by_question": {key: list(value) for key, value in eligibility.items()},
            "direct_fix_responsibility_count": sum(row.vote_flip_gain > 0 for row in rows),
            "margin_gain_responsibility_sum": sum(row.margin_gain for row in rows),
            "coverage_responsibility_count": sum(row.coverage_opportunity for row in rows),
            "conversion_responsibility_count": sum(row.conversion_opportunity for row in rows),
            "dominant_wrong_responsibility_count": sum(row.dominant_wrong_member for row in rows),
            "candidate_counterfactual_values_by_question": {
                question_hash: audit["candidate_counterfactual_values"]
                for question_hash, audit in eligibility_audits.items()
            },
            "eligibility_audit_by_question": eligibility_audits,
            "assigned_opportunities": {
                str(agent_id): [asdict(row) for row in values] for agent_id, values in assigned.items()
            },
        })
        self.responsibility_portfolio_trajectory.append({
            "team_state_version": self.team_state_version,
            "portfolio_size_by_agent": {str(agent): len(rows) for agent, rows in assigned.items()},
        })
        for state in states:
            question_hash = state.question_hash
            eligibility_audit = eligibility_audits.get(question_hash, {})
            for opportunity in opportunities[question_hash]:
                self.member_opportunities.append({
                    "artifact_schema_version": "counterfactual_vote_margin_opportunities_v1",
                    "team_state_version": self.team_state_version,
                    "question_hash": question_hash,
                    "G": state.gold_vote_count,
                    "H": state.largest_wrong_vote_count,
                    "M": state.plurality_margin,
                    "eligible": opportunity.agent_id in eligibility.get(question_hash, ()),
                    "eligibility_audit": eligibility_audit,
                    **asdict(opportunity),
                })
        self.cached_responsibility_eligibility = dict(eligibility)
        self.cached_responsibility_assignments = {
            agent_id: list(values) for agent_id, values in assigned.items()
        }
        self.responsibility_state_version = self.team_state_version
        self.responsibility_refresh_count += 1
        return eligibility, assigned

    def ensure_responsibility_current(
        self,
    ) -> tuple[dict[str, tuple[int, ...]], dict[int, list[MemberAwareRepairOpportunity]]]:
        return self.assign_responsibilities()

    def refresh_responsibility_after_commit(
        self,
    ) -> tuple[dict[str, tuple[int, ...]], dict[int, list[MemberAwareRepairOpportunity]]]:
        self.team_state_version += 1
        if self.responsibility_state_version == self.team_state_version:
            raise AssertionError("committed team state must invalidate responsibility state")
        return self.assign_responsibilities()

    def select_target(
        self,
        assigned: Mapping[int, Sequence[MemberAwareRepairOpportunity]],
        update_index: int,
    ) -> tuple[int | None, bool, list[dict[str, Any]]]:
        max_wait = self.cfg.responsibility.responsibility_max_wait_updates
        current_counts = self._member_correct_counts(self.active_profiles)
        initial_counts = self._member_correct_counts(self.initial_profiles)
        priorities = target_priorities(
            assignments=assigned,
            state=self.responsibility_state,
            seed=self.cfg.training.seed,
            max_wait_updates=max_wait,
            current_member_correct_counts=current_counts,
            initial_member_correct_counts=initial_counts,
            member_uplift_tolerance=self.cfg.responsibility.member_uplift_tolerance,
        )
        fairness = any(row.overdue for row in priorities)
        if self.protocol.target_selection_policy == "round_robin":
            target = update_index % 5
            fairness = False
        elif self.protocol.target_selection_policy == "member_aware_responsibility":
            selection = build_target_selection_decision(
                priorities,
                all_member_gains=[current - initial for current, initial in zip(current_counts, initial_counts, strict=True)],
                state=self.responsibility_state,
                max_wait_updates=max_wait,
                member_uplift_tolerance=self.cfg.responsibility.member_uplift_tolerance,
                member_catchup_mode=self.cfg.responsibility.member_catchup_mode,
            )
            target = selection.selected_agent_id
        else:
            raise ValueError(
                f"Protocol has no optimization target selector: {self.protocol.name}"
            )
        priority_payload = [
            {
                **asdict(row),
                "D_i": row.direct_fix_count,
                "S_i": row.margin_gain_sum,
                "g_i": row.member_gain,
                "d_i": row.uplift_deficit,
                "selected": row.agent_id == target,
            }
            for row in priorities
        ]
        if self.protocol.target_selection_policy == "member_aware_responsibility":
            pool_stage = selection.selection_pool_stage
            eligible_ids = list(selection.eligible_agent_ids)
            overdue_ids = list(selection.overdue_agent_ids)
            actual_ids = list(selection.actual_candidate_agent_ids)
            target_fronts = selection.target_pareto_fronts
            target_frontier_ids = list(
                selection.target_frontier_agent_ids
            )
        else:
            eligible_ids = overdue_ids = []
            actual_ids = []
            target_fronts = {}
            target_frontier_ids = []
            pool_stage = "round_robin"
        self.target_priority_audit.append({
            "update_index": int(update_index),
            "priorities": priority_payload,
            "overdue_first": fairness,
            "update_lane": selection.update_lane if self.protocol.target_selection_policy == "member_aware_responsibility" else "protocol_control",
            "selection_pool_stage": pool_stage,
            "eligible_agent_ids": eligible_ids,
            "overdue_agent_ids": overdue_ids,
            "actual_candidate_agent_ids": actual_ids,
            "target_pareto_fronts": {
                str(agent_id): target_fronts[agent_id]
                for agent_id in actual_ids
                if agent_id in target_fronts
            },
            "target_frontier_agent_ids": [
                agent_id for agent_id in target_frontier_ids
            ],
            "no_actionable_reason": selection.no_actionable_reason if self.protocol.target_selection_policy == "member_aware_responsibility" else "",
            "selected_agent_id": target,
        })
        return target, fairness, priority_payload

    def _representative_indices(self, count: int) -> list[int]:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        return sorted(
            range(len(self.fixed_probe.examples)),
            key=lambda index: hashlib.sha256(
                f"{self.cfg.training.seed}:{self.fixed_probe.examples[index].question_hash}".encode("utf-8")
            ).hexdigest(),
        )[:count]

    def _pool_indices(
        self,
        target_agent_id: int,
        assigned_hashes: set[str],
    ) -> StageAPools:
        states, _, opportunities = self.current_states_and_opportunities()
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        deterministic = self._representative_indices(len(states))
        requested = {
            "representative": self.cfg.evaluation.stage_a_representative_size,
            "coverage": self.cfg.evaluation.stage_a_coverage_size,
            "conversion": self.cfg.evaluation.stage_a_conversion_size,
            "preservation": self.cfg.evaluation.stage_a_preservation_size,
        }
        if self.protocol.sample_pool_policy == "individual_errors":
            errors = [
                index for index, state in enumerate(states)
                if opportunities[state.question_hash][target_agent_id].member_error
            ]
            error_set = set(errors)
            representative_candidates = [index for index in deterministic if index in error_set] + [
                index for index in deterministic if index not in error_set
            ]
            coverage: list[int] = []
            conversion: list[int] = []
            preservation = [index for index in deterministic if index not in error_set]
        else:
            coverage = []
            conversion = []
            preservation = []
            assigned_mode = self.protocol.sample_pool_policy == "member_aware_residuals"
            if self.protocol.sample_pool_policy not in {
                "global_peer_state", "member_aware_residuals"
            }:
                raise ValueError(f"Unsupported sample pool policy: {self.protocol.sample_pool_policy}")
            for index, state in enumerate(states):
                opportunity = opportunities[state.question_hash][target_agent_id]
                included = state.question_hash in assigned_hashes if assigned_mode else True
                if included and state.gold_vote_count == 0:
                    coverage.append(index)
                if included and not state.vote_correct and state.gold_vote_count > 0:
                    conversion.append(index)
                if opportunity.unique_correct or opportunity.pivotal_correct:
                    preservation.append(index)
            coverage.sort(key=lambda index: (
                -int(
                    opportunities[states[index].question_hash][
                        target_agent_id
                    ].vote_flip_gain
                    > 0
                ),
                -opportunities[states[index].question_hash][
                    target_agent_id
                ].margin_gain,
                -opportunities[states[index].question_hash][target_agent_id].oracle_soft_utility_gain,
                states[index].question_hash,
            ))
            conversion.sort(key=lambda index: (
                -int(
                    opportunities[states[index].question_hash][
                        target_agent_id
                    ].vote_flip_gain
                    > 0
                ),
                -opportunities[states[index].question_hash][
                    target_agent_id
                ].margin_gain,
                -opportunities[states[index].question_hash][target_agent_id].oracle_soft_utility_gain,
                abs(states[index].plurality_margin),
                states[index].question_hash,
            ))
            preservation.sort(key=lambda index: (
                -int(opportunities[states[index].question_hash][target_agent_id].pivotal_correct),
                -int(opportunities[states[index].question_hash][target_agent_id].unique_correct),
                states[index].plurality_margin,
                states[index].question_hash,
            ))
            representative_candidates = deterministic

        raw = {
            "coverage": coverage,
            "conversion": conversion,
            "preservation": preservation,
        }
        selected: dict[str, tuple[int, ...]] = {}
        used: set[int] = set()
        overlap_removed = 0
        for name in ("coverage", "conversion", "preservation"):
            available = raw[name]
            non_overlapping = [index for index in available if index not in used]
            overlap_removed += len(available) - len(non_overlapping)
            chosen = tuple(non_overlapping[: requested[name]])
            selected[name] = chosen
            used.update(chosen)
        target_size = min(len(states), sum(requested.values()))
        representative = tuple(
            index for index in representative_candidates if index not in used
        )[: max(0, target_size - len(used))]
        used.update(representative)
        specialized_indices = set().union(*selected.values())
        available_sizes = {
            "representative": sum(index not in specialized_indices for index in representative_candidates),
            **{name: len(values) for name, values in raw.items()},
        }
        selected_sizes = {
            "representative": len(representative),
            **{name: len(values) for name, values in selected.items()},
        }
        return StageAPools(
            representative=representative,
            coverage=selected["coverage"],
            conversion=selected["conversion"],
            preservation=selected["preservation"],
            requested_size_per_pool=requested,
            available_size_per_pool=available_sizes,
            selected_size_per_pool=selected_sizes,
            overlap_removed=overlap_removed,
            final_unique_size=len(used),
        )

    def stage_a_indices(self, target_agent_id: int, assigned_hashes: set[str]) -> list[int]:
        return self._pool_indices(target_agent_id, assigned_hashes).indices()

    def _proposal_context(
        self,
        target_agent_id: int,
        parent_prompt: str,
        assigned_hashes: set[str],
        *,
        rotation_cursor: int = 0,
        proposal_failure_feedback: ProposalFailureFeedback | None = None,
    ) -> tuple[AnyDiagnosisContext, TCSContextDiagnostics]:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        states, contexts, opportunities = self.current_states_and_opportunities()
        target_rows = [
            opportunities[state.question_hash][target_agent_id] for state in states
        ]
        context_policy = self.protocol.tcs_context_policy
        if (context_policy == "member_aware_responsibility_conditioned"
                and not assigned_hashes):
            context_policy = "generic_member_catchup_context_v1"
        aggregation = aggregate_probe_diagnosis(
            target_agent_id=target_agent_id,
            examples=self.fixed_probe.examples,
            states=states,
            peer_contexts=contexts,
            opportunities=opportunities,
            assigned_hashes=assigned_hashes,
            context_policy=context_policy,
            max_patterns=self.cfg.tcs.tcs_max_pattern_summaries,
            max_cases=self.cfg.tcs.tcs_max_evidence_cases,
            rotation_cursor=rotation_cursor,
        )
        common = {
            "target_agent_id": target_agent_id,
            "parent_prompt": parent_prompt,
            "parent_prompt_hash": self.prompt_hash(parent_prompt),
            "patterns": aggregation.selected_patterns,
            "evidence_cases": aggregation.evidence_cases,
            "previous_outcome": self.previous_update_outcomes[target_agent_id],
        }
        if self.protocol.tcs_context_policy == "generic_accuracy":
            target_profile = self.active_profiles[target_agent_id]
            target_correct_count = sum(not row.member_error for row in target_rows)
            context: AnyDiagnosisContext = AccuracyDiagnosisContext(
                **common,
                target_correct_count=target_correct_count,
                target_error_count=len(target_rows) - target_correct_count,
                target_invalid_count=sum(not row.valid for row in target_profile),
            )
        elif self.protocol.tcs_context_policy == "generic_peer_state":
            context = PeerStateDiagnosisContext(
                **common,
                vote_wrong_count=sum(not state.vote_correct for state in states),
                coverage_failure_count=sum(state.gold_vote_count == 0 for state in states),
                conversion_failure_count=sum(
                    not state.vote_correct and state.gold_vote_count > 0
                    for state in states
                ),
                preservation_count=sum(
                    row.unique_correct or row.pivotal_correct for row in target_rows
                ),
            )
        elif context_policy == "member_aware_responsibility_conditioned":
            current_counts = self._member_correct_counts(self.active_profiles)
            initial_counts = self._member_correct_counts(self.initial_profiles)
            member_gains = [
                current - initial
                for current, initial in zip(
                    current_counts,
                    initial_counts,
                    strict=True,
                )
            ]
            maximum_gain = max(member_gains, default=0)
            assigned_rows = tuple(
                row
                for row in target_rows
                if row.question_hash in assigned_hashes
            )
            context = AssignedResidualDiagnosisContext(
                **common,
                assigned_residual_count=len(assigned_hashes),
                target_member_gain=member_gains[target_agent_id],
                uplift_deficit=max(
                    0,
                    maximum_gain
                    - member_gains[target_agent_id]
                    - self.cfg.responsibility.member_uplift_tolerance,
                ),
                direct_fix_responsibility_count=sum(
                    row.vote_flip_gain > 0 for row in assigned_rows
                ),
                margin_gain_responsibility_sum=sum(
                    row.margin_gain for row in assigned_rows
                ),
                coverage_residual_count=sum(
                    row.coverage_opportunity for row in assigned_rows
                ),
                conversion_residual_count=sum(
                    row.conversion_opportunity for row in assigned_rows
                ),
                preservation_count=sum(
                    row.unique_correct or row.pivotal_correct
                    for row in target_rows
                ),
                proposal_failure_feedback=proposal_failure_feedback,
            )
        elif context_policy == "generic_member_catchup_context_v1":
            context = PeerStateDiagnosisContext(
                **common,
                vote_wrong_count=sum(not state.vote_correct for state in states),
                coverage_failure_count=sum(state.gold_vote_count == 0 for state in states),
                conversion_failure_count=sum(not state.vote_correct and state.gold_vote_count > 0 for state in states),
                preservation_count=sum(row.unique_correct or row.pivotal_correct for row in target_rows),
            )
        else:
            raise ValueError(f"Unsupported TCS context policy: {self.protocol.tcs_context_policy}")
        return limit_diagnosis_context(
            context,
            max_chars=self.cfg.tcs.tcs_context_max_chars,
            full_probe_case_count=aggregation.full_probe_case_count,
            available_pattern_count=len(aggregation.available_patterns),
        )

    async def _run_student_cycle(
        self,
        *,
        context: AnyDiagnosisContext,
        parent_prompt: str,
        repair_plan: TeacherRepairPlan,
        funnel: CandidateFunnel,
        update_index: int,
        target_agent_id: int,
        semantic_round: int,
        context_hash: str,
        student_generation_cycle_index: int,
        previous_teacher_plan_hash: str = "",
    ) -> tuple[tuple[StudentPromptCandidate, ...], tuple[str, ...], bool]:
        base_request = build_student_request(
            parent_prompt=parent_prompt,
            approved_plan=repair_plan,
            answer_format=self.cfg.data.answer_format,
            candidate_count=self.cfg.tcs.num_candidates_per_parent,
            candidate_prompt_max_chars=self.cfg.tcs.candidate_prompt_max_chars,
            total_candidate_prompt_max_chars=(
                self.cfg.tcs.total_candidate_prompt_max_chars
            ),
        )
        parent_prompt_hash = self.prompt_hash(parent_prompt)
        repair_plan_hash = teacher_repair_plan_hash(repair_plan)
        previous_rejection_classes: tuple[str, ...] = ()
        parsed_candidates: tuple[StudentPromptCandidate, ...] = ()
        for attempt_index in range(self.cfg.tcs.student_invalid_max_retries + 1):
            student_request = (
                base_request
                if attempt_index == 0
                else build_student_recovery_request(
                    base_request=base_request,
                    previous_rejection_classes=previous_rejection_classes,
                    required_candidate_count=self.cfg.tcs.num_candidates_per_parent,
                    parent_prompt_hash=parent_prompt_hash,
                    approved_repair_plan_hash=repair_plan_hash,
                )
            )
            self.student_recovery_state = {
                "in_progress": True,
                "update_index": int(update_index),
                "target_agent_id": int(target_agent_id),
                "student_generation_cycle_index": int(
                    student_generation_cycle_index
                ),
                "student_attempt_index": int(attempt_index),
                "upstream_regeneration_count": int(
                    funnel.upstream_regeneration_count
                ),
            }
            funnel.student_calls += 1
            try:
                student_result = await self._chat(
                    self.cfg.models.optimizer_model,
                    "Return strict JSON only.",
                    student_request,
                    self.cfg.tcs.student_temperature,
                    None,
                    "optimizer",
                )
            except Exception as exc:
                funnel.infrastructure_failed_updates += 1
                funnel.terminal_failure_class = "transport_failure"
                funnel.terminal_failure_role = "student"
                funnel.terminal_student_failure_class = "transport_failure"
                self.tcs_rounds.append({
                    "update_index": update_index,
                    "target_agent_id": target_agent_id,
                    "role": "student",
                    "semantic_round": semantic_round,
                    "format_attempt": attempt_index,
                    "student_generation_cycle_index": (
                        student_generation_cycle_index
                    ),
                    "student_attempt_index": attempt_index,
                    "schema_valid": False,
                    "finish_reason": "",
                    "response_truncated": False,
                    "failure_class": "transport_failure",
                    "candidate_rejection_classes": ["transport_failure"],
                    "student_retry_triggered": False,
                    "student_retry_reason": type(exc).__name__,
                    "input_characters": len(student_request),
                    "output_characters": 0,
                    "raw_response_characters": 0,
                    "parsed_payload_characters": 0,
                })
                self.student_recovery_state["in_progress"] = False
                return (), ("transport_failure",), False

            student_raw = student_result.text
            parsed = extract_json_obj(student_raw)
            truncated = response_truncated(student_result)
            parse_result = None
            parse_error = ""
            failure_class = ""
            raw_count = 0
            total_candidate_characters = 0
            rejection_reasons: tuple[tuple[str, ...], ...] = ()
            rejection_classes: tuple[str, ...] = ()
            if truncated:
                failure_class = "provider_completion_truncation"
                rejection_classes = ("provider_completion_truncation",)
            elif parsed is None:
                failure_class = "invalid_json"
                rejection_classes = ("invalid_json",)
            else:
                raw_values = (
                    parsed.get("candidate_prompts")
                    if isinstance(parsed, Mapping) else None
                )
                raw_count = len(raw_values) if isinstance(raw_values, list) else 0
                try:
                    parse_result = parse_student_candidates(
                        parsed,
                        parent_prompt=parent_prompt,
                        context=context,
                        expected_count=self.cfg.tcs.num_candidates_per_parent,
                        candidate_prompt_max_chars=(
                            self.cfg.tcs.candidate_prompt_max_chars
                        ),
                        total_candidate_prompt_max_chars=(
                            self.cfg.tcs.total_candidate_prompt_max_chars
                        ),
                    )
                    parsed_candidates = parse_result.candidates
                    raw_count = parse_result.raw_count
                    rejection_reasons = parse_result.rejection_reasons
                    total_candidate_characters = (
                        parse_result.total_candidate_characters
                    )
                    rejection_classes = tuple(
                        reason
                        for reasons in rejection_reasons
                        for reason in reasons
                    )
                    funnel.sample_memorization_rejected += sum(
                        "sample_memorization" in reasons
                        for reasons in rejection_reasons
                    )
                    if raw_count and 0 < len(parsed_candidates) < raw_count:
                        funnel.student_partially_valid_responses += 1
                    if not parsed_candidates:
                        failure_class = "zero_valid_student_candidates"
                        if not rejection_classes:
                            rejection_classes = (
                                "candidate_list_missing",
                            )
                except (TypeError, ValueError) as exc:
                    parse_error = str(exc)
                    failure_class = (
                        parse_error
                        if parse_error in {
                            "candidate_list_missing",
                            "schema_invalid",
                            "too_long",
                        }
                        else "schema_invalid"
                    )
                    rejection_classes = (failure_class,)

            funnel.raw_candidate_count += raw_count
            funnel.valid_candidate_count = len(parsed_candidates)
            funnel.schema_valid_count = len(parsed_candidates)
            if failure_class:
                funnel.student_invalid_responses += 1
                if truncated:
                    funnel.student_truncated_responses += 1
            retry_triggered = bool(
                not parsed_candidates
                and attempt_index < self.cfg.tcs.student_invalid_max_retries
                and failure_class != "transport_failure"
            )
            if retry_triggered:
                funnel.student_retry_triggered = True
                funnel.student_retry_count += 1
            if parsed_candidates and (
                attempt_index > 0 or student_generation_cycle_index > 0
            ):
                funnel.student_recovered = True
            student_round = {
                "update_index": update_index,
                "target_agent_id": target_agent_id,
                "role": "student",
                "context_type": type(context).__name__,
                "context_hash": context_hash,
                "request_hash": _request_hash(
                    "Return strict JSON only.", student_request
                ),
                "response_hash": hashlib.sha256(
                    student_raw.encode("utf-8")
                ).hexdigest(),
                "json_extracted": parsed is not None,
                "schema_valid": parse_result is not None,
                "requested_count": self.cfg.tcs.num_candidates_per_parent,
                "raw_count": raw_count,
                "valid_count": len(parsed_candidates),
                "raw_candidate_count": raw_count,
                "valid_candidate_count": len(parsed_candidates),
                "per_candidate_rejection_reasons": [
                    list(row) for row in rejection_reasons
                ],
                "candidate_rejection_classes": list(rejection_classes),
                "semantic_round": semantic_round,
                "format_attempt": attempt_index,
                "student_generation_cycle_index": (
                    student_generation_cycle_index
                ),
                "student_attempt_index": attempt_index,
                "finish_reason": student_result.finish_reason,
                "response_truncated": truncated,
                "failure_class": failure_class,
                "retry_reason": (
                    failure_class if retry_triggered else ""
                ),
                "student_retry_triggered": retry_triggered,
                "student_retry_reason": (
                    ",".join(rejection_classes) if retry_triggered else ""
                ),
                "student_recovered": bool(
                    parsed_candidates
                    and (
                        attempt_index > 0
                        or student_generation_cycle_index > 0
                    )
                ),
                "student_cycle_exhausted": bool(
                    not parsed_candidates
                    and attempt_index == self.cfg.tcs.student_invalid_max_retries
                ),
                "upstream_regeneration_triggered": bool(
                    funnel.upstream_regeneration_triggered
                ),
                "upstream_regeneration_count": int(
                    funnel.upstream_regeneration_count
                ),
                "parse_error": parse_error,
                "response_excerpt": _response_excerpt(student_raw),
                "input_characters": len(student_request),
                "output_characters": len(student_raw),
                "raw_response_characters": len(student_raw),
                "parsed_payload_characters": (
                    len(json.dumps(parsed, ensure_ascii=False, sort_keys=True))
                    if parsed is not None else 0
                ),
                "total_candidate_characters": total_candidate_characters,
                "previous_teacher_plan_hash": (
                    previous_teacher_plan_hash
                ),
                "upstream_teacher_plan_hash": (
                    repair_plan_hash if student_generation_cycle_index > 0 else ""
                ),
            }
            self.tcs_rounds.append(student_round)
            self.student_recovery_observations.append({
                key: student_round[key]
                for key in (
                    "update_index",
                    "target_agent_id",
                    "student_generation_cycle_index",
                    "student_attempt_index",
                    "raw_candidate_count",
                    "valid_candidate_count",
                    "candidate_rejection_classes",
                    "student_retry_triggered",
                    "student_retry_reason",
                    "student_recovered",
                    "student_cycle_exhausted",
                    "upstream_regeneration_triggered",
                    "upstream_regeneration_count",
                )
            })
            if parsed_candidates:
                self.student_recovery_state["in_progress"] = False
                return parsed_candidates, rejection_classes, True
            previous_rejection_classes = rejection_classes

        funnel.student_cycle_exhausted = True
        self.student_recovery_state["in_progress"] = False
        return (), previous_rejection_classes, True

    async def _regenerate_teacher_critic_plan(
        self,
        *,
        context: AnyDiagnosisContext,
        teacher_request: str,
        previous_approved_plan: TeacherRepairPlan,
        student_rejection_classes: tuple[str, ...],
        funnel: CandidateFunnel,
        update_index: int,
        target_agent_id: int,
        context_hash: str,
    ) -> tuple[TeacherRepairPlan | None, CriticDecision | None, int]:
        funnel.upstream_regeneration_triggered = True
        funnel.upstream_regeneration_count += 1
        previous_approved_hash = teacher_repair_plan_hash(
            previous_approved_plan
        )
        repair_plan: TeacherRepairPlan | None = None
        critic_decision: CriticDecision | None = None
        semantic_round_used = 1
        for semantic_round in range(
            1, self.cfg.tcs.teacher_critic_max_rounds + 1
        ):
            semantic_round_used = semantic_round
            prior_plan = repair_plan
            prior_critic = critic_decision
            if semantic_round == 1:
                user_request = build_teacher_regeneration_request(
                    previous_plan_hash=previous_approved_hash,
                    student_rejection_classes=student_rejection_classes,
                )
            else:
                if prior_plan is None or prior_critic is None:
                    raise RuntimeError(
                        "Upstream Teacher revision requires prior plan and Critic"
                    )
                user_request = build_teacher_revision_request(
                    context=context,
                    previous_plan=prior_plan,
                    critic_decision=prior_critic,
                    field_max_chars=self.cfg.tcs.teacher_field_max_chars,
                    total_max_chars=self.cfg.tcs.teacher_total_max_chars,
                    feedback_max_chars=self.cfg.tcs.critic_feedback_max_chars,
                )
            repair_plan = None
            teacher_request_hash = _request_hash(
                teacher_request, user_request
            )
            for format_attempt in range(
                self.cfg.tcs.teacher_json_max_retries + 1
            ):
                funnel.teacher_calls += 1
                try:
                    teacher_result = await self._chat(
                        self.cfg.models.optimizer_model,
                        teacher_request,
                        user_request,
                        self.cfg.tcs.teacher_temperature,
                        None,
                        "optimizer",
                    )
                except Exception as exc:
                    funnel.infrastructure_failed_updates += 1
                    funnel.terminal_failure_class = "transport_failure"
                    funnel.terminal_failure_role = "teacher"
                    self.tcs_rounds.append({
                        "update_index": update_index,
                        "target_agent_id": target_agent_id,
                        "role": "teacher",
                        "semantic_round": semantic_round,
                        "format_attempt": format_attempt,
                        "student_generation_cycle_index": 1,
                        "upstream_regeneration_triggered": True,
                        "request_hash": teacher_request_hash,
                        "context_hash": context_hash,
                        "schema_valid": False,
                        "failure_class": "transport_failure",
                        "retry_reason": type(exc).__name__,
                        "previous_teacher_plan_hash": previous_approved_hash,
                        "upstream_teacher_plan_hash": "",
                        "upstream_plan_changed": False,
                        "input_characters": (
                            len(teacher_request) + len(user_request)
                        ),
                        "output_characters": 0,
                    })
                    return None, None, semantic_round_used
                teacher_raw = teacher_result.text
                parsed_teacher = extract_json_obj(teacher_raw)
                truncated = response_truncated(teacher_result)
                failure_class = (
                    "provider_completion_truncation"
                    if truncated else "invalid_json"
                    if parsed_teacher is None else ""
                )
                parse_error = ""
                if not failure_class:
                    try:
                        repair_plan = parse_teacher_repair_plan(
                            parsed_teacher,
                            field_max_chars=(
                                self.cfg.tcs.teacher_field_max_chars
                            ),
                            total_max_chars=(
                                self.cfg.tcs.teacher_total_max_chars
                            ),
                        )
                        if contains_supplied_example_text(
                            json.dumps(
                                asdict(repair_plan), ensure_ascii=False
                            ),
                            context,
                        ):
                            raise ValueError(
                                "teacher repair plan copies supplied sample text"
                            )
                    except (TypeError, ValueError) as exc:
                        repair_plan = None
                        failure_class = "schema_error"
                        parse_error = str(exc)
                if failure_class:
                    funnel.teacher_invalid_responses += 1
                    if truncated:
                        funnel.teacher_truncated_responses += 1
                new_hash = (
                    teacher_repair_plan_hash(repair_plan)
                    if repair_plan is not None else ""
                )
                self.tcs_rounds.append({
                    "update_index": update_index,
                    "target_agent_id": target_agent_id,
                    "role": "teacher",
                    "context_type": type(context).__name__,
                    "context_hash": context_hash,
                    "request_hash": teacher_request_hash,
                    "response_hash": hashlib.sha256(
                        teacher_raw.encode("utf-8")
                    ).hexdigest(),
                    "response_excerpt": _response_excerpt(teacher_raw),
                    "repair_plan": (
                        asdict(repair_plan) if repair_plan else None
                    ),
                    "schema_valid": repair_plan is not None,
                    "semantic_round": semantic_round,
                    "format_attempt": format_attempt,
                    "student_generation_cycle_index": 1,
                    "upstream_regeneration_triggered": True,
                    "finish_reason": teacher_result.finish_reason,
                    "response_truncated": truncated,
                    "failure_class": failure_class,
                    "retry_reason": (
                        failure_class
                        if failure_class and format_attempt == 0 else ""
                    ),
                    "parse_error": parse_error,
                    "previous_teacher_plan_hash": previous_approved_hash,
                    "upstream_teacher_plan_hash": new_hash,
                    "upstream_plan_changed": bool(
                        new_hash and new_hash != previous_approved_hash
                    ),
                    "teacher_plan_hash": new_hash,
                    "input_characters": (
                        len(teacher_request) + len(user_request)
                    ),
                    "output_characters": len(teacher_raw),
                    "raw_response_characters": len(teacher_raw),
                    "parsed_payload_characters": (
                        len(json.dumps(
                            parsed_teacher,
                            ensure_ascii=False,
                            sort_keys=True,
                        ))
                        if parsed_teacher is not None else 0
                    ),
                })
                if repair_plan is not None:
                    break
            if repair_plan is None:
                funnel.terminal_failure_class = (
                    "upstream_teacher_invalid_exhausted"
                )
                funnel.terminal_failure_role = "teacher"
                return None, None, semantic_round_used

            critic_request = build_critic_request(
                context,
                repair_plan,
                feedback_max_chars=self.cfg.tcs.critic_feedback_max_chars,
            )
            critic_decision = None
            critic_request_hash = _request_hash(
                critic_request, "Audit the repair plan."
            )
            for format_attempt in range(
                self.cfg.tcs.critic_json_max_retries + 1
            ):
                funnel.critic_calls += 1
                try:
                    critic_result = await self._chat(
                        self.cfg.models.evaluator_model,
                        critic_request,
                        "Audit the repair plan.",
                        self.cfg.tcs.critic_temperature,
                        None,
                        "evaluator",
                    )
                except Exception as exc:
                    funnel.infrastructure_failed_updates += 1
                    funnel.terminal_failure_class = "transport_failure"
                    funnel.terminal_failure_role = "critic"
                    self.tcs_rounds.append({
                        "update_index": update_index,
                        "target_agent_id": target_agent_id,
                        "role": "critic",
                        "semantic_round": semantic_round,
                        "format_attempt": format_attempt,
                        "student_generation_cycle_index": 1,
                        "upstream_regeneration_triggered": True,
                        "schema_valid": False,
                        "failure_class": "transport_failure",
                        "retry_reason": type(exc).__name__,
                        "upstream_critic_decision_hash": "",
                    })
                    return None, None, semantic_round_used
                critic_raw = critic_result.text
                parsed_critic = extract_json_obj(critic_raw)
                truncated = response_truncated(critic_result)
                failure_class = (
                    "provider_completion_truncation"
                    if truncated else "invalid_json"
                    if parsed_critic is None else ""
                )
                parse_error = ""
                if not failure_class:
                    try:
                        critic_decision = parse_critic_decision(
                            parsed_critic,
                            allowed_case_ids={
                                row.case_id for row in context.evidence_cases
                            },
                            feedback_max_chars=(
                                self.cfg.tcs.critic_feedback_max_chars
                            ),
                        )
                        if contains_supplied_example_text(
                            json.dumps(
                                asdict(critic_decision), ensure_ascii=False
                            ),
                            context,
                        ):
                            raise ValueError(
                                "critic response copies supplied sample text"
                            )
                    except (TypeError, ValueError) as exc:
                        critic_decision = None
                        failure_class = "schema_error"
                        parse_error = str(exc)
                if failure_class:
                    funnel.critic_invalid_responses += 1
                    if truncated:
                        funnel.critic_truncated_responses += 1
                elif critic_decision is not None and not critic_decision.approved:
                    failure_class = "semantic_rejection"
                    funnel.critic_semantic_rejections += 1
                decision_hash = (
                    critic_decision_hash(critic_decision)
                    if critic_decision is not None else ""
                )
                self.tcs_rounds.append({
                    "update_index": update_index,
                    "target_agent_id": target_agent_id,
                    "role": "critic",
                    "context_type": type(context).__name__,
                    "context_hash": context_hash,
                    "request_hash": critic_request_hash,
                    "response_hash": hashlib.sha256(
                        critic_raw.encode("utf-8")
                    ).hexdigest(),
                    "response_excerpt": _response_excerpt(critic_raw),
                    "schema_valid": critic_decision is not None,
                    "failed_checks": (
                        list(critic_decision.failed_checks)
                        if critic_decision else []
                    ),
                    "risk_case_ids": (
                        list(critic_decision.risk_case_ids)
                        if critic_decision else []
                    ),
                    "feedback": (
                        critic_decision.feedback if critic_decision else ""
                    ),
                    "effective_approved": bool(
                        critic_decision and critic_decision.approved
                    ),
                    "semantic_round": semantic_round,
                    "format_attempt": format_attempt,
                    "student_generation_cycle_index": 1,
                    "upstream_regeneration_triggered": True,
                    "finish_reason": critic_result.finish_reason,
                    "response_truncated": truncated,
                    "failure_class": failure_class,
                    "retry_reason": (
                        failure_class
                        if failure_class and format_attempt == 0 else ""
                    ),
                    "parse_error": parse_error,
                    "teacher_plan_hash": teacher_repair_plan_hash(
                        repair_plan
                    ),
                    "critic_decision_hash": decision_hash,
                    "upstream_critic_decision_hash": decision_hash,
                    "input_characters": len(critic_request),
                    "output_characters": len(critic_raw),
                    "raw_response_characters": len(critic_raw),
                    "parsed_payload_characters": (
                        len(json.dumps(
                            parsed_critic,
                            ensure_ascii=False,
                            sort_keys=True,
                        ))
                        if parsed_critic is not None else 0
                    ),
                })
                if critic_decision is not None:
                    break
            if critic_decision is None:
                funnel.terminal_failure_class = (
                    "upstream_critic_schema_exhausted"
                )
                funnel.terminal_failure_role = "critic"
                return None, None, semantic_round_used
            if critic_decision.approved:
                funnel.critic_approved += 1
                return repair_plan, critic_decision, semantic_round_used

        funnel.terminal_failure_class = (
            "upstream_critic_semantic_rejection_exhausted"
        )
        funnel.terminal_failure_role = "critic"
        return None, critic_decision, semantic_round_used

    async def propose_candidates(
        self,
        target_agent_id: int,
        assigned_hashes: set[str],
        funnel: CandidateFunnel,
        update_index: int = -1,
    ) -> list[CandidateRuntime]:
        parent_prompt = self.agents[target_agent_id].current_prompt
        memory_key: ProposalMemoryKey | None = None
        memory_entry: ProposalMemoryEntry | None = None
        memory_feedback: ProposalFailureFeedback | None = None
        rotation_cursor = 0
        if self.cfg.tcs.proposal_memory_mode == "state_local_v1":
            memory_key = self._proposal_memory_key(
                target_agent_id=target_agent_id,
                parent_prompt=parent_prompt,
                assigned_hashes=assigned_hashes,
            )
            memory_entry = self._proposal_memory_entry(memory_key, assigned_hashes)
            if memory_entry is not None:
                memory_feedback = feedback_for(memory_entry)
                rotation_cursor = memory_entry.rotation_cursor
        context, diagnostics = self._proposal_context(
            target_agent_id,
            parent_prompt,
            assigned_hashes,
            rotation_cursor=rotation_cursor,
            proposal_failure_feedback=memory_feedback,
        )
        evidence_bundle_hash = self._evidence_bundle_hash(context)
        if (
            memory_entry is not None
            and memory_entry.immediate_tabu_bundle_hash == evidence_bundle_hash
            and not memory_entry.rotation_exhausted
        ):
            context, diagnostics = self._proposal_context(
                target_agent_id,
                parent_prompt,
                assigned_hashes,
                rotation_cursor=rotation_cursor + 1,
                proposal_failure_feedback=memory_feedback,
            )
            evidence_bundle_hash = self._evidence_bundle_hash(context)
        self._proposal_memory_attempts[update_index] = {
            "key": memory_key,
            "memory_hit": memory_entry is not None,
            "feedback": memory_feedback,
            "evidence_bundle_hash": evidence_bundle_hash,
            "rotation_cursor": rotation_cursor,
            "rotation_exhausted": bool(
                memory_entry and memory_entry.rotation_exhausted
            ),
        }
        context_mode = (
            "generic_member_catchup_context_v1"
            if self.protocol.tcs_context_policy == "member_aware_responsibility_conditioned" and not assigned_hashes
            else self.protocol.tcs_context_policy
        )
        context_serialized = serialize_context(context)
        context_object = context_payload(context)
        field_paths = sorted(_recursive_field_paths(context_object))
        if isinstance(context, AccuracyDiagnosisContext):
            forbidden_tokens = (
                "gold_vote_count", "largest_wrong_vote_count", "plurality_margin",
                "peer_", "responsibility", "owner", "assigned", "member_gain",
                "improvement_need", "answer_role",
            )
        elif isinstance(context, PeerStateDiagnosisContext):
            forbidden_tokens = (
                "assigned", "responsibility", "member_gain",
                "uplift_deficit",
            )
        else:
            forbidden_tokens = ()
        lowered_paths = tuple(path.lower() for path in field_paths)
        forbidden_check = {
            token: any(token in path for path in lowered_paths)
            for token in forbidden_tokens
        }
        responsibility_tokens = (
            "assigned",
            "responsibility",
            "member_gain",
            "uplift_deficit",
        )
        self.tcs_context_history.append({
            "update_index": update_index,
            "target_agent_id": target_agent_id,
            "context_type": type(context).__name__,
            "context_class": type(context).__name__,
            "context_mode": context_mode,
            "parent_prompt_hash": self.prompt_hash(parent_prompt),
            "proposal_context_hash": hashlib.sha256(context_serialized.encode("utf-8")).hexdigest(),
            "proposal_memory_mode": self.cfg.tcs.proposal_memory_mode,
            "proposal_memory_hit": memory_entry is not None,
            "proposal_memory_key_hash": memory_key.key_hash() if memory_key else "",
            "evidence_bundle_hash": evidence_bundle_hash,
            "proposal_rotation_cursor": rotation_cursor,
            "serialized_top_level_fields": sorted(context_object),
            "serialized_recursive_field_paths": field_paths,
            "forbidden_field_check": forbidden_check,
            "forbidden_field_violations": sorted(
                token for token, present in forbidden_check.items() if present
            ),
            "responsibility_specific_field_count": sum(
                any(token in path for token in responsibility_tokens)
                for path in lowered_paths
            ),
            "diagnosis_aggregation_version": DIAGNOSIS_AGGREGATION_VERSION,
            "selected_context_pattern_question_hashes": {
                pattern.pattern_id: list(pattern.represented_question_hashes)
                for pattern in context.patterns
            },
            **asdict(diagnostics),
        })
        funnel.parents_considered = 1
        teacher_request = build_teacher_request(
            context,
            field_max_chars=self.cfg.tcs.teacher_field_max_chars,
            total_max_chars=self.cfg.tcs.teacher_total_max_chars,
        )
        repair_plan: TeacherRepairPlan | None = None
        critic_decision: CriticDecision | None = None
        context_hash = hashlib.sha256(context_serialized.encode("utf-8")).hexdigest()
        last_teacher_failure = ""
        last_critic_failure = ""
        for semantic_round in range(1, self.cfg.tcs.teacher_critic_max_rounds + 1):
            previous_plan = repair_plan
            previous_critic_decision = critic_decision
            if semantic_round == 1:
                user_request = "Produce the repair proposal."
            else:
                if previous_plan is None or previous_critic_decision is None:
                    raise RuntimeError(
                        "Teacher revision requires the previous plan and Critic decision"
                    )
                user_request = build_teacher_revision_request(
                    context=context,
                    previous_plan=previous_plan,
                    critic_decision=previous_critic_decision,
                    field_max_chars=self.cfg.tcs.teacher_field_max_chars,
                    total_max_chars=self.cfg.tcs.teacher_total_max_chars,
                    feedback_max_chars=self.cfg.tcs.critic_feedback_max_chars,
                )
            repair_plan = None
            teacher_request_hash = _request_hash(teacher_request, user_request)
            for format_attempt in range(self.cfg.tcs.teacher_json_max_retries + 1):
                funnel.teacher_calls += 1
                try:
                    teacher_result = await self._chat(
                        self.cfg.models.optimizer_model,
                        teacher_request,
                        user_request,
                        self.cfg.tcs.teacher_temperature,
                        None,
                        "optimizer",
                    )
                except Exception as exc:
                    funnel.infrastructure_failed_updates += 1
                    funnel.terminal_failure_class = "transport_failure"
                    funnel.terminal_failure_role = "teacher"
                    self.tcs_rounds.append({
                        "update_index": update_index,
                        "target_agent_id": target_agent_id,
                        "role": "teacher",
                        "semantic_round": semantic_round,
                        "format_attempt": format_attempt,
                        "request_hash": teacher_request_hash,
                        "context_hash": context_hash,
                        "schema_valid": False,
                        "finish_reason": "",
                        "response_truncated": False,
                        "failure_class": "transport_failure",
                        "retry_reason": type(exc).__name__,
                        "previous_plan_hash": (
                            teacher_repair_plan_hash(previous_plan)
                            if previous_plan is not None else ""
                        ),
                        "revision_critic_hash": (
                            critic_decision_hash(previous_critic_decision)
                            if previous_critic_decision is not None else ""
                        ),
                        "teacher_plan_hash": "",
                        "revision_changed_fields": [],
                        "input_characters": len(teacher_request) + len(user_request),
                        "output_characters": 0,
                        "raw_response_characters": 0,
                        "parsed_payload_characters": 0,
                    })
                    return []
                teacher_raw = teacher_result.text
                parsed_teacher = extract_json_obj(teacher_raw)
                truncated = response_truncated(teacher_result)
                failure_class = (
                    "provider_completion_truncation"
                    if truncated else "invalid_json"
                    if parsed_teacher is None else ""
                )
                parse_error = ""
                if not failure_class:
                    try:
                        repair_plan = parse_teacher_repair_plan(
                            parsed_teacher,
                            field_max_chars=self.cfg.tcs.teacher_field_max_chars,
                            total_max_chars=self.cfg.tcs.teacher_total_max_chars,
                        )
                        if contains_supplied_example_text(
                            json.dumps(asdict(repair_plan), ensure_ascii=False), context,
                        ):
                            raise ValueError("teacher repair plan copies supplied sample text")
                    except (TypeError, ValueError) as exc:
                        failure_class = "schema_error"
                        parse_error = str(exc)
                if failure_class:
                    last_teacher_failure = failure_class
                    funnel.teacher_invalid_responses += 1
                    if truncated:
                        funnel.teacher_truncated_responses += 1
                self.tcs_rounds.append({
                    "update_index": update_index,
                    "target_agent_id": target_agent_id,
                    "role": "teacher",
                    "context_type": type(context).__name__,
                    "context_hash": context_hash,
                    "request_hash": teacher_request_hash,
                    "response_hash": hashlib.sha256(teacher_raw.encode("utf-8")).hexdigest(),
                    "response_excerpt": _response_excerpt(teacher_raw),
                    "repair_plan": asdict(repair_plan) if repair_plan else None,
                    "schema_valid": repair_plan is not None,
                    "semantic_round": semantic_round,
                    "format_attempt": format_attempt,
                    "finish_reason": teacher_result.finish_reason,
                    "response_truncated": truncated,
                    "failure_class": failure_class,
                    "retry_reason": failure_class if failure_class and format_attempt == 0 else "",
                    "parse_error": parse_error,
                    "previous_plan_hash": (
                        teacher_repair_plan_hash(previous_plan)
                        if previous_plan is not None else ""
                    ),
                    "revision_critic_hash": (
                        critic_decision_hash(previous_critic_decision)
                        if previous_critic_decision is not None else ""
                    ),
                    "teacher_plan_hash": (
                        teacher_repair_plan_hash(repair_plan)
                        if repair_plan is not None else ""
                    ),
                    "revision_changed_fields": (
                        list(changed_teacher_plan_fields(previous_plan, repair_plan))
                        if previous_plan is not None and repair_plan is not None
                        else []
                    ),
                    "input_characters": len(teacher_request) + len(user_request),
                    "output_characters": len(teacher_raw),
                    "raw_response_characters": len(teacher_raw),
                    "parsed_payload_characters": (
                        len(json.dumps(parsed_teacher, ensure_ascii=False, sort_keys=True))
                        if parsed_teacher is not None else 0
                    ),
                })
                if repair_plan is not None:
                    break
            if repair_plan is None:
                funnel.terminal_failure_role = "teacher"
                if last_teacher_failure == "provider_completion_truncation":
                    funnel.terminal_failure_class = "teacher_provider_truncation"
                    funnel.infrastructure_failed_updates += 1
                else:
                    funnel.terminal_failure_class = "teacher_schema_exhausted"
                return []

            critic_request = build_critic_request(
                context,
                repair_plan,
                feedback_max_chars=self.cfg.tcs.critic_feedback_max_chars,
            )
            critic_decision = None
            critic_request_hash = _request_hash(critic_request, "Audit the repair plan.")
            for format_attempt in range(self.cfg.tcs.critic_json_max_retries + 1):
                funnel.critic_calls += 1
                try:
                    critic_result = await self._chat(
                        self.cfg.models.evaluator_model,
                        critic_request,
                        "Audit the repair plan.",
                        self.cfg.tcs.critic_temperature,
                        None,
                        "evaluator",
                    )
                except Exception as exc:
                    funnel.infrastructure_failed_updates += 1
                    funnel.terminal_failure_class = "transport_failure"
                    funnel.terminal_failure_role = "critic"
                    self.tcs_rounds.append({
                        "update_index": update_index,
                        "target_agent_id": target_agent_id,
                        "role": "critic",
                        "semantic_round": semantic_round,
                        "format_attempt": format_attempt,
                        "request_hash": critic_request_hash,
                        "context_hash": context_hash,
                        "schema_valid": False,
                        "finish_reason": "",
                        "response_truncated": False,
                        "failure_class": "transport_failure",
                        "retry_reason": type(exc).__name__,
                        "teacher_plan_hash": teacher_repair_plan_hash(repair_plan),
                        "critic_decision_hash": "",
                        "input_characters": len(critic_request),
                        "output_characters": 0,
                        "raw_response_characters": 0,
                        "parsed_payload_characters": 0,
                    })
                    return []
                critic_raw = critic_result.text
                parsed_critic = extract_json_obj(critic_raw)
                truncated = response_truncated(critic_result)
                failure_class = (
                    "provider_completion_truncation"
                    if truncated else "invalid_json"
                    if parsed_critic is None else ""
                )
                parse_error = ""
                if not failure_class:
                    try:
                        critic_decision = parse_critic_decision(
                            parsed_critic,
                            allowed_case_ids={row.case_id for row in context.evidence_cases},
                            feedback_max_chars=self.cfg.tcs.critic_feedback_max_chars,
                        )
                        if contains_supplied_example_text(
                            json.dumps(asdict(critic_decision), ensure_ascii=False),
                            context,
                        ):
                            raise ValueError("critic response copies supplied sample text")
                    except (TypeError, ValueError) as exc:
                        failure_class = "schema_error"
                        parse_error = str(exc)
                if failure_class:
                    last_critic_failure = failure_class
                    funnel.critic_invalid_responses += 1
                    if truncated:
                        funnel.critic_truncated_responses += 1
                elif critic_decision is not None and not critic_decision.approved:
                    failure_class = "semantic_rejection"
                    funnel.critic_semantic_rejections += 1
                self.tcs_rounds.append({
                    "update_index": update_index,
                    "target_agent_id": target_agent_id,
                    "role": "critic",
                    "context_type": type(context).__name__,
                    "context_hash": context_hash,
                    "request_hash": critic_request_hash,
                    "response_hash": hashlib.sha256(critic_raw.encode("utf-8")).hexdigest(),
                    "response_excerpt": _response_excerpt(critic_raw),
                    "json_extracted": parsed_critic is not None,
                    "schema_valid": critic_decision is not None,
                    "failed_checks": (
                        list(critic_decision.failed_checks) if critic_decision else []
                    ),
                    "risk_case_ids": (
                        list(critic_decision.risk_case_ids) if critic_decision else []
                    ),
                    "feedback": critic_decision.feedback if critic_decision else "",
                    "effective_approved": bool(
                        critic_decision and critic_decision.approved
                    ),
                    "semantic_round": semantic_round,
                    "format_attempt": format_attempt,
                    "finish_reason": critic_result.finish_reason,
                    "response_truncated": truncated,
                    "failure_class": failure_class,
                    "retry_reason": failure_class if failure_class and format_attempt == 0 else "",
                    "parse_error": parse_error,
                    "teacher_plan_hash": teacher_repair_plan_hash(repair_plan),
                    "critic_decision_hash": (
                        critic_decision_hash(critic_decision)
                        if critic_decision is not None else ""
                    ),
                    "input_characters": len(critic_request),
                    "output_characters": len(critic_raw),
                    "raw_response_characters": len(critic_raw),
                    "parsed_payload_characters": (
                        len(json.dumps(parsed_critic, ensure_ascii=False, sort_keys=True))
                        if parsed_critic is not None else 0
                    ),
                })
                if critic_decision is not None:
                    break
            if critic_decision is None:
                funnel.terminal_failure_role = "critic"
                if last_critic_failure == "provider_completion_truncation":
                    funnel.terminal_failure_class = "critic_provider_truncation"
                    funnel.infrastructure_failed_updates += 1
                else:
                    funnel.terminal_failure_class = "critic_schema_exhausted"
                return []
            if critic_decision.approved:
                funnel.critic_approved += 1
                break
        if repair_plan is None or critic_decision is None or not critic_decision.approved:
            funnel.terminal_failure_class = "critic_semantic_rejection_exhausted"
            funnel.terminal_failure_role = "critic"
            return []

        parsed_candidates: tuple[StudentPromptCandidate, ...] = ()
        funnel.requested_candidate_count = self.cfg.tcs.num_candidates_per_parent
        parsed_candidates, rejection_classes, recoverable = (
            await self._run_student_cycle(
                context=context,
                parent_prompt=parent_prompt,
                repair_plan=repair_plan,
                funnel=funnel,
                update_index=update_index,
                target_agent_id=target_agent_id,
                semantic_round=semantic_round,
                context_hash=context_hash,
                student_generation_cycle_index=0,
            )
        )
        original_plan_hash = teacher_repair_plan_hash(repair_plan)
        if (
            not parsed_candidates
            and recoverable
            and self.cfg.tcs.student_upstream_regeneration_max_count > 0
        ):
            regenerated_plan, regenerated_critic, upstream_semantic_round = (
                await self._regenerate_teacher_critic_plan(
                    context=context,
                    teacher_request=teacher_request,
                    previous_approved_plan=repair_plan,
                    student_rejection_classes=rejection_classes,
                    funnel=funnel,
                    update_index=update_index,
                    target_agent_id=target_agent_id,
                    context_hash=context_hash,
                )
            )
            if regenerated_plan is None or regenerated_critic is None:
                return []
            repair_plan = regenerated_plan
            critic_decision = regenerated_critic
            parsed_candidates, rejection_classes, recoverable = (
                await self._run_student_cycle(
                    context=context,
                    parent_prompt=parent_prompt,
                    repair_plan=repair_plan,
                    funnel=funnel,
                    update_index=update_index,
                    target_agent_id=target_agent_id,
                    semantic_round=upstream_semantic_round,
                    context_hash=context_hash,
                    student_generation_cycle_index=1,
                    previous_teacher_plan_hash=original_plan_hash,
                )
            )
            if not parsed_candidates and recoverable:
                funnel.terminal_failure_class = (
                    "student_invalid_exhausted_after_upstream_regeneration"
                )
                funnel.terminal_failure_role = "student"
                funnel.terminal_student_failure_class = (
                    funnel.terminal_failure_class
                )
        elif not parsed_candidates and recoverable:
            funnel.terminal_failure_class = "student_invalid_exhausted"
            funnel.terminal_failure_role = "student"
            funnel.terminal_student_failure_class = (
                funnel.terminal_failure_class
            )
        funnel.valid_candidate_count = len(parsed_candidates)
        funnel.schema_valid_count = len(parsed_candidates)
        if not parsed_candidates:
            return []
        unique: dict[str, CandidateRuntime] = {}
        non_parent = 0
        repair_plan_hash = teacher_repair_plan_hash(repair_plan)
        for candidate in parsed_candidates:
            prompt = normalize_prompt_text(candidate.candidate_prompt)
            prompt_hash = self.prompt_hash(prompt)
            non_parent += 1
            unique.setdefault(prompt_hash, CandidateRuntime(
                student_candidate=candidate,
                prompt=prompt,
                prompt_hash=prompt_hash,
                generation=1,
                parent_prompt_hash=self.prompt_hash(parent_prompt),
                repair_plan_hash=repair_plan_hash,
            ))
        funnel.non_parent_count = non_parent
        funnel.deduplicated_count = len(unique)
        return list(unique.values())

    async def evaluate_candidates(
        self,
        target_agent_id: int,
        candidates: Sequence[CandidateRuntime],
        assigned_hashes: set[str],
        funnel: CandidateFunnel,
    ) -> tuple[CandidateRuntime | None, CandidateEvaluation, list[CandidateRuntime]]:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        active_prompt = self.agents[target_agent_id].current_prompt
        incumbent = evaluate_candidate_profile(
            prompt=active_prompt,
            prompt_hash=self.prompt_hash(active_prompt),
            examples=self.fixed_probe.examples,
            active_profiles=self.active_profiles,
            initial_profiles=self.initial_profiles,
            candidate_profile=self.active_profiles[target_agent_id],
            target_agent_id=target_agent_id,
            assigned_question_hashes=assigned_hashes,
            normalize_answer=self.normalize_answer,
            match_answer=self.match_answer,
            tie_break=self.protocol.tie_policy,
            seed=self.cfg.training.seed,
            tau=self.cfg.peer_state.soft_vote_tau,
        )
        pools = self._pool_indices(target_agent_id, assigned_hashes)
        indices = pools.indices()
        funnel.stage_a_requested_size_per_pool = dict(pools.requested_size_per_pool)
        funnel.stage_a_available_size_per_pool = dict(pools.available_size_per_pool)
        funnel.stage_a_selected_size_per_pool = dict(pools.selected_size_per_pool)
        funnel.stage_a_overlap_removed = pools.overlap_removed
        funnel.actual_stage_a_size = pools.final_unique_size
        stage_a_examples, stage_a_active = subset_profiles(
            self.fixed_probe.examples, self.active_profiles, indices
        )
        _, stage_a_initial = subset_profiles(
            self.fixed_probe.examples, self.initial_profiles, indices
        )
        for candidate in candidates:
            partial = await self.fixed_probe.evaluate_prompt_indices(
                target_agent_id, candidate.prompt, candidate.prompt_hash, indices, self.solve,
            )
            stage_a_profile = tuple(partial[index] for index in indices)
            candidate.stage_a_evaluation = evaluate_candidate_profile(
                prompt=candidate.prompt,
                prompt_hash=candidate.prompt_hash,
                examples=stage_a_examples,
                active_profiles=stage_a_active,
                initial_profiles=stage_a_initial,
                candidate_profile=stage_a_profile,
                target_agent_id=target_agent_id,
                assigned_question_hashes=assigned_hashes,
                normalize_answer=self.normalize_answer,
                match_answer=self.match_answer,
                tie_break=self.protocol.tie_policy,
                seed=self.cfg.training.seed,
                tau=self.cfg.peer_state.soft_vote_tau,
            )
        funnel.stage_a_evaluated = len(candidates)
        if self.protocol.candidate_selection_policy == "individual_accuracy":
            shortlist = sorted(
                candidates,
                key=lambda row: individual_accuracy_key(row.stage_a_evaluation, row.generation),
                reverse=True,
            )[: self.cfg.evaluation.stage_b_candidate_budget]
            for candidate in candidates:
                candidate.stage_a_decision = StageASelectionDecision(
                    selected=candidate in shortlist,
                    selected_by_channels=("individual_accuracy",) if candidate in shortlist else (),
                    pareto_front=1,
                    aggregate_rank=0,
                )
        elif self.protocol.candidate_selection_policy == "vote_first":
            shortlist = sorted(
                candidates,
                key=lambda row: vote_first_key(row.stage_a_evaluation, row.generation),
                reverse=True,
            )[: self.cfg.evaluation.stage_b_candidate_budget]
            for candidate in candidates:
                candidate.stage_a_decision = StageASelectionDecision(
                    selected=candidate in shortlist,
                    selected_by_channels=("team_vote",) if candidate in shortlist else (),
                    pareto_front=1,
                    aggregate_rank=0,
                )
        else:
            evaluation_to_runtime = {row.stage_a_evaluation.prompt_hash: row for row in candidates}
            selected, decisions = stage_a_multichannel_shortlist(
                [row.stage_a_evaluation for row in candidates],
                channel_top_k=self.cfg.evaluation.stage_a_channel_top_k,
                total_budget=self.cfg.evaluation.stage_b_candidate_budget,
            )
            shortlist = [evaluation_to_runtime[row.prompt_hash] for row in selected]
            for candidate in candidates:
                candidate.stage_a_decision = decisions[candidate.prompt_hash]
        funnel.selected_by_team_vote_channel = sum(
            candidate.stage_a_decision.selected
            and "team_vote" in candidate.stage_a_decision.selected_by_channels
            for candidate in candidates
        )
        funnel.selected_by_worst_member_channel = sum(
            candidate.stage_a_decision.selected
            and "worst_member" in candidate.stage_a_decision.selected_by_channels
            for candidate in candidates
        )
        funnel.selected_by_mean_member_channel = sum(
            candidate.stage_a_decision.selected
            and "mean_member" in candidate.stage_a_decision.selected_by_channels
            for candidate in candidates
        )

        feasible: list[CandidateRuntime] = []
        acceptable: list[CandidateRuntime] = []
        for candidate in shortlist:
            candidate.profile = await self.fixed_probe.evaluate_prompt(
                target_agent_id, candidate.prompt, candidate.prompt_hash, self.solve,
            )
            candidate.final_evaluation = evaluate_candidate_profile(
                prompt=candidate.prompt,
                prompt_hash=candidate.prompt_hash,
                examples=self.fixed_probe.examples,
                active_profiles=self.active_profiles,
                initial_profiles=self.initial_profiles,
                candidate_profile=candidate.profile,
                target_agent_id=target_agent_id,
                assigned_question_hashes=assigned_hashes,
                normalize_answer=self.normalize_answer,
                match_answer=self.match_answer,
                tie_break=self.protocol.tie_policy,
                seed=self.cfg.training.seed,
                tau=self.cfg.peer_state.soft_vote_tau,
            )
            candidate.constraint = evaluate_constraints(
                candidate.final_evaluation,
                incumbent,
            )
            if candidate.constraint.passed:
                feasible.append(candidate)
            for reason in candidate.constraint.rejection_reasons:
                field = {
                    "target_regression": "rejected_target_regression",
                    "team_vote_regression": "rejected_team_vote_regression",
                    "no_target_or_vote_progress": (
                        "rejected_no_target_or_vote_progress"
                    ),
                    "member_objective_regression": (
                        "rejected_member_objective_regression"
                    ),
                    "terminal_invalid_regression": (
                        "rejected_terminal_invalid_regression"
                    ),
                }[reason]
                setattr(funnel, field, getattr(funnel, field) + 1)
        funnel.stage_b_evaluated = len(shortlist)
        funnel.constraint_feasible = len(feasible)

        if self.protocol.candidate_selection_policy == "individual_accuracy":
            acceptable = list(feasible)
            accepted = max(
                acceptable,
                key=lambda row: individual_accuracy_key(row.final_evaluation, row.generation),
                default=None,
            )
        elif self.protocol.candidate_selection_policy == "vote_first":
            acceptable = list(feasible)
            accepted = max(
                acceptable,
                key=lambda row: vote_first_key(row.final_evaluation, row.generation),
                default=None,
            )
        else:
            acceptable = [
                row for row in feasible
                if candidate_is_acceptable(row.final_evaluation, incumbent)
            ]
            front_hashes = set(
                member_aware_pareto_front(
                    [row.final_evaluation for row in acceptable]
                )
            )
            nondominated = [
                row for row in acceptable
                if row.prompt_hash in front_hashes
            ]
            accepted = max(
                nondominated,
                key=lambda row: member_first_key(row.final_evaluation, row.generation),
                default=None,
            )
        funnel.acceptable_candidates = len(acceptable)
        funnel.accepted_candidate = accepted is not None
        return accepted, incumbent, list(candidates)

    def _update_candidate_search_outcome(
        self,
        *,
        target: int,
        update_index: int,
        stage_b_evaluations: Sequence[CandidateEvaluation],
    ) -> tuple[int | None, int]:
        """Expose observed proposal quality as audit-only data, never scheduling state."""
        if not stage_b_evaluations:
            return None, 0
        best_attempt_target_gain = max(
            row.member_gain.target_gain_vs_incumbent
            for row in stage_b_evaluations
        )
        return best_attempt_target_gain, 0

    def _record_proposal_memory_outcome(
        self,
        *,
        update_index: int,
        target_agent_id: int,
        assigned_hashes: set[str],
        evaluated: Sequence[CandidateRuntime],
        funnel: CandidateFunnel,
        accepted: CandidateRuntime | None,
    ) -> None:
        attempt = self._proposal_memory_attempts.pop(update_index, None)
        if self.cfg.tcs.proposal_memory_mode == "off":
            return
        if attempt is None or not isinstance(attempt.get("key"), ProposalMemoryKey):
            raise RuntimeError("proposal memory attempt is missing its exact key")
        key = attempt["key"]
        if key.target_agent_id != target_agent_id:
            raise RuntimeError("proposal memory cross-agent lifecycle mismatch")
        summaries: list[SanitizedCandidateSummary] = []
        reasons: list[str] = []
        for row in evaluated:
            if row.final_evaluation is None:
                continue
            evaluation = row.final_evaluation
            constraint = row.constraint
            rejection_reasons = tuple(constraint.rejection_reasons) if constraint else ()
            reasons.extend(rejection_reasons)
            summaries.append(SanitizedCandidateSummary(
                prompt_hash=row.prompt_hash,
                target_gain=int(evaluation.member_gain.target_gain_vs_incumbent),
                vote_gain_count=int(evaluation.marginal.vote_gain_count),
                vote_loss_count=int(evaluation.marginal.vote_loss_count),
                vote_net_gain=int(evaluation.marginal.net_vote_delta),
                assigned_residual_repair_count=int(evaluation.marginal.assigned_residual_repair_count),
                assigned_residual_utility_delta=float(evaluation.marginal.assigned_residual_utility_delta),
                coverage_gain_count=int(evaluation.marginal.coverage_gain_count),
                coverage_loss_count=int(evaluation.marginal.coverage_loss_count),
                unique_correct_gain_count=int(evaluation.protection.unique_correct_gain_count),
                unique_correct_loss_count=int(evaluation.protection.unique_correct_loss_count),
                pivotal_correct_gain_count=int(evaluation.protection.pivotal_correct_gain_count),
                pivotal_correct_loss_count=int(evaluation.protection.pivotal_correct_loss_count),
                rejection_reasons=rejection_reasons,
            ))
        max_target_gain = max((row.target_gain for row in summaries), default=None)
        max_vote_net_gain = max((row.vote_net_gain for row in summaries), default=None)
        max_assigned_repair = max(
            (row.assigned_residual_repair_count for row in summaries), default=None
        )
        if accepted is not None:
            failure_stage = "accepted"
        elif funnel.terminal_failure_role == "critic" or funnel.critic_semantic_rejections:
            failure_stage = "critic"
        elif not summaries:
            failure_stage = "pipeline"
        elif max_assigned_repair == 0:
            failure_stage = "zero_repair_behavior"
        elif any(
            row.target_gain > 0 or row.vote_net_gain > 0
            or row.assigned_residual_repair_count > 0
            for row in summaries
        ):
            failure_stage = "regressive_progress"
        else:
            failure_stage = "stage_b_rejection"
        histogram = {reason: reasons.count(reason) for reason in sorted(set(reasons))}
        teacher_hashes = tuple(
            row["teacher_plan_hash"]
            for row in self.tcs_rounds
            if row.get("update_index") == update_index
            and row.get("target_agent_id") == target_agent_id
            and row.get("role") == "teacher"
            and row.get("teacher_plan_hash")
        )
        event = {
            "update_index": update_index,
            "memory_mode": self.cfg.tcs.proposal_memory_mode,
            "memory_hit": bool(attempt["memory_hit"]),
            "memory_key_hash": key.key_hash(),
            "target_agent_id": target_agent_id,
            "team_state_version": key.team_state_version,
            "assigned_residual_set_hash": key.assigned_residual_set_hash,
            "previous_evidence_bundle_hash": (
                attempt["feedback"].previous_evidence_bundle_hash
                if attempt["feedback"] else None
            ),
            "current_evidence_bundle_hash": attempt["evidence_bundle_hash"],
            "failure_stage": failure_stage,
            "revision_mode": (
                attempt["feedback"].required_revision_mode if attempt["feedback"] else "none"
            ),
            "rotation_triggered": bool(attempt["memory_hit"]),
            "rotation_level": (
                attempt["feedback"].rotation_level if attempt["feedback"] else "none"
            ),
            "rotation_exhausted": bool(attempt["rotation_exhausted"]),
            "cross_agent_guard_passed": True,
            "acceptable_after_memory_hit": bool(accepted and attempt["memory_hit"]),
        }
        self.proposal_memory_events.append(event)
        self.proposal_rotation_trajectory.append(dict(event))
        if accepted is not None:
            return
        existing = self._proposal_memory_entry(key, assigned_hashes)
        entry = existing or ProposalMemoryEntry(
            key=key,
            assigned_question_hashes=tuple(sorted(assigned_hashes)),
        )
        prior_bundles = entry.previous_evidence_bundle_hashes
        current_bundle = str(attempt["evidence_bundle_hash"])
        entry.attempt_count += 1
        entry.previous_evidence_bundle_hashes = prior_bundles + (current_bundle,)
        entry.previous_repair_plan_hashes = (
            entry.previous_repair_plan_hashes + teacher_hashes
        )
        entry.last_failure_stage = failure_stage
        entry.last_rejection_reason_histogram = histogram
        entry.candidate_summaries = tuple(summaries)
        entry.max_target_gain = max_target_gain
        entry.max_vote_net_gain = max_vote_net_gain
        entry.max_assigned_residual_repair_count = max_assigned_repair
        entry.rotation_cursor += 1
        entry.rotation_exhausted = bool(
            prior_bundles and prior_bundles[-1] == current_bundle
        )
        entry.immediate_tabu_bundle_hash = current_bundle
        self.proposal_memory_entries[key.key_hash()] = entry

    async def update_once(self, update_index: int) -> bool:
        if not self.protocol.optimization_enabled:
            return False
        if self.protocol.target_selection_policy == "member_aware_responsibility":
            if self.protocol.responsibility_refresh_policy != "online":
                raise AssertionError("dynamic responsibility protocol requires online refresh")
            eligibility, assigned = self.ensure_responsibility_current()
        else:
            eligibility = {}
            assigned = {agent_id: [] for agent_id in range(5)}
            states, _, _ = self.current_states_and_opportunities()
            self.peer_state_history.extend(asdict(state) for state in states)
        target, fairness_triggered, target_priorities_payload = self.select_target(
            assigned,
            update_index,
        )
        if target is None:
            self.candidate_decisions.append({
                "update_index": update_index,
                "update_lane": "no_actionable_responsibility",
                "target_agent_id": None,
                "target_assigned_residual_count": 0,
                "assigned_question_hashes": [],
                "stop_reason": "no_actionable_responsibility",
                "agent_target_priorities": target_priorities_payload,
                "funnel": asdict(CandidateFunnel()),
                "candidates": [],
            })
            self.responsibility_portfolio_trajectory.append({
                "update_index": update_index,
                "event": "no_actionable_responsibility",
                "selected_agent_id": None,
            })
            return False
        self.agent_selection_counts[target] += 1
        self.responsibility_state.target_attempt_count_by_agent[target] = (
            self.responsibility_state.target_attempt_count_by_agent.get(target, 0) + 1
        )
        update_lane = (self.target_priority_audit[-1].get("update_lane")
                       if self.protocol.target_selection_policy == "member_aware_responsibility" else "protocol_control")
        assigned_hashes = ({row.question_hash for row in assigned.get(target, ())}
                           if update_lane == "responsibility_conditioned" else set())
        if update_lane == "responsibility_conditioned" and not assigned_hashes:
            raise AssertionError(
                "responsibility target must have a nonempty portfolio"
            )
        funnel = CandidateFunnel()
        candidates = await self.propose_candidates(
            target, assigned_hashes, funnel, update_index=update_index,
        )
        accepted, incumbent, evaluated = await self.evaluate_candidates(
            target, candidates, assigned_hashes, funnel,
        )
        self._record_proposal_memory_outcome(
            update_index=update_index,
            target_agent_id=target,
            assigned_hashes=assigned_hashes,
            evaluated=evaluated,
            funnel=funnel,
            accepted=accepted,
        )
        stage_b_evaluations = [
            row.final_evaluation for row in evaluated
            if row.final_evaluation is not None
        ]
        best_attempt_target_gain, cooldown_length = self._update_candidate_search_outcome(
            target=target,
            update_index=update_index,
            stage_b_evaluations=stage_b_evaluations,
        )
        for agent_id in self.responsibility_state.updates_since_selected_by_agent:
            self.responsibility_state.updates_since_selected_by_agent[agent_id] += 1
        self.responsibility_state.updates_since_selected_by_agent[target] = 0
        decision = {
            "update_index": update_index,
            "update_lane": update_lane,
            "target_agent_id": target,
            "agent_selection_distribution": dict(self.agent_selection_counts),
            "assigned_question_hashes": sorted(assigned_hashes),
            "generic_member_catchup": update_lane == "generic_member_catchup",
            "borrowed_residual_count": 0,
            "target_assigned_residual_count": len(assigned_hashes),
            "max_wait_fairness_trigger_count": int(fairness_triggered),
            "agent_target_priorities": target_priorities_payload,
            "best_attempt_target_gain": best_attempt_target_gain,
            "positive_target_gain_candidate_found": bool(
                best_attempt_target_gain is not None and best_attempt_target_gain > 0
            ),
            "candidate_search_outcome_updated": bool(stage_b_evaluations),
            "cooldown_length_assigned": cooldown_length,
            "funnel": asdict(funnel),
            "accepted_prompt_hash": accepted.prompt_hash if accepted else "",
            "incumbent": asdict(incumbent),
            "candidates": [
                {
                    "prompt_hash": row.prompt_hash,
                    "generation": row.generation,
                    "repair_plan_hash": row.repair_plan_hash,
                    "stage_a_decision": asdict(row.stage_a_decision) if row.stage_a_decision else None,
                    "evaluation": asdict(row.final_evaluation) if row.final_evaluation else None,
                    "constraint": asdict(row.constraint) if row.constraint else None,
                    **(
                        asdict(row.constraint)
                        if row.constraint is not None else {}
                    ),
                }
                for row in evaluated
            ],
        }
        self.candidate_decisions.append(decision)
        self.target_responsibility_context_alignment.append({
            "update_index": update_index,
            "target_agent_id": target,
            "update_lane": decision["update_lane"],
            "target_assigned_residual_count": len(assigned_hashes),
            "nonassigned_residual_in_context": False,
            "assertion_passed": bool(assigned_hashes) or self.protocol.target_selection_policy != "member_aware_responsibility",
        })
        if accepted is None:
            empirical_evaluation_completed = funnel.stage_a_evaluated > 0
            rejection_reasons = sorted({
                reason
                for row in evaluated
                if row.constraint is not None
                for reason in row.constraint.rejection_reasons
            })
            if empirical_evaluation_completed and not rejection_reasons:
                rejection_reasons = ["no_acceptable_candidate"]
            self.previous_update_outcomes[target] = PreviousUpdateOutcome(
                attempted=True,
                empirical_evaluation_completed=empirical_evaluation_completed,
                accepted=False,
                rejection_reasons=(
                    tuple(rejection_reasons)
                    if empirical_evaluation_completed else ()
                ),
            )
            return False

        agent = self.agents[target]
        old_prompt = agent.current_prompt
        old_previous_prompt = agent.previous_active_prompt
        old_profile = self.active_profiles[target]
        old_responsibility_state = deepcopy(self.responsibility_state)
        old_cached_eligibility = deepcopy(self.cached_responsibility_eligibility)
        old_cached_assignments = deepcopy(self.cached_responsibility_assignments)
        old_cached_opportunities = deepcopy(self.cached_member_opportunities)
        old_team_state_version = self.team_state_version
        old_responsibility_state_version = self.responsibility_state_version
        old_responsibility_refresh_count = self.responsibility_refresh_count
        old_peer_history_length = len(self.peer_state_history)
        old_responsibility_history_length = len(self.responsibility_assignments)
        old_target_audit_length = len(self.target_priority_audit)
        agent.previous_active_prompt = old_prompt
        try:
            agent.current_prompt = accepted.prompt
            if accepted.profile is None:
                raise AssertionError("accepted candidate has no full fixed-probe profile")
            self.active_profiles[target] = accepted.profile
            self.responsibility_state.accepted_updates_by_agent[target] = (
                self.responsibility_state.accepted_updates_by_agent.get(target, 0) + 1
            )
            if self.protocol.responsibility_refresh_policy == "online":
                self.refresh_responsibility_after_commit()
            else:
                self.team_state_version += 1
        except Exception:
            agent.current_prompt = old_prompt
            agent.previous_active_prompt = old_previous_prompt
            self.active_profiles[target] = old_profile
            self.responsibility_state = old_responsibility_state
            self.cached_responsibility_eligibility = old_cached_eligibility
            self.cached_responsibility_assignments = old_cached_assignments
            self.cached_member_opportunities = old_cached_opportunities
            self.team_state_version = old_team_state_version
            self.responsibility_state_version = old_responsibility_state_version
            self.responsibility_refresh_count = old_responsibility_refresh_count
            del self.peer_state_history[old_peer_history_length:]
            del self.responsibility_assignments[old_responsibility_history_length:]
            del self.target_priority_audit[old_target_audit_length:]
            raise
        evaluation = accepted.final_evaluation
        competence_delta = evaluation.competence.correct_count - incumbent.competence.correct_count
        self.previous_update_outcomes[target] = PreviousUpdateOutcome(
            attempted=True,
            empirical_evaluation_completed=True,
            accepted=True,
            target_correct_delta=competence_delta,
            vote_correct_delta=(
                evaluation.team_outcome.vote_correct_count
                - incumbent.team_outcome.vote_correct_count
            ),
            minimum_member_gain_delta=(
                evaluation.member_gain.minimum_gain_count
                - incumbent.member_gain.minimum_gain_count
            ),
            total_member_gain_delta=(
                evaluation.member_gain.total_gain_count
                - incumbent.member_gain.total_gain_count
            ),
            assigned_repair_count=(
                evaluation.marginal.assigned_residual_repair_count
            ),
        )
        return True

    def _dataset_metrics_from_profiles(
        self,
        examples: Sequence[ProbeExample],
        profiles: Sequence[Sequence[PromptAnswer]],
    ) -> DatasetMetrics:
        if len(profiles) != 5:
            raise ValueError("dataset evaluation requires five profiles")
        correct_per_agent = [0] * 5
        vote_correct = invalid = terminal_invalid = c0 = tie_count = 0
        validity_status_counts: dict[str, int] = {}
        utility = 0.0
        rows: list[DatasetEvaluationRow] = []
        for index, example in enumerate(examples):
            state = build_team_vote_state(
                question_hash=example.question_hash,
                gold_answer=example.gold_answer,
                answers=[profile[index].answer for profile in profiles],
                valid_vector=[profile[index].valid for profile in profiles],
                normalize_answer=self.normalize_answer,
                match_answer=self.match_answer,
                tie_break=self.protocol.tie_policy,
                seed=self.cfg.training.seed,
            )
            for agent_id, correct in enumerate(state.team_correctness):
                correct_per_agent[agent_id] += int(correct)
                status = profiles[agent_id][index].validity_status
                validity_status_counts[status] = validity_status_counts.get(status, 0) + 1
            vote_correct += int(state.vote_correct)
            invalid += sum(not value for value in state.team_validity)
            terminal_invalid += sum(
                int(profile[index].terminal_invalid) for profile in profiles
            )
            c0 += int(state.gold_vote_count == 0)
            tie_count += int(state.top_tie)
            utility += soft_vote_utility(
                state.gold_vote_count, state.plurality_margin, self.cfg.peer_state.soft_vote_tau,
            )
            rows.append(DatasetEvaluationRow(
                question_hash=state.question_hash,
                vote_correct=state.vote_correct,
                top_tie=state.top_tie,
                gold_vote_count=state.gold_vote_count,
                largest_wrong_vote_count=state.largest_wrong_vote_count,
                plurality_margin=state.plurality_margin,
            ))
        size = max(1, len(examples))
        return DatasetMetrics(
            vote_correct_count=vote_correct,
            per_agent_correct_counts=tuple(correct_per_agent),
            plurality_vote_acc=vote_correct / size,
            vote_acc=vote_correct / size,
            mean_individual_acc=sum(correct_per_agent) / (size * 5),
            min_individual_acc=min(correct_per_agent) / size,
            per_agent_acc=tuple(value / size for value in correct_per_agent),
            mean_soft_vote_utility=utility / size,
            c0_count=c0,
            mean_invalid_rate=invalid / (size * 5),
            tie_count=tie_count,
            tie_rate=tie_count / size,
            rows=tuple(rows),
            validity_status_counts=validity_status_counts,
            terminal_invalid_count=terminal_invalid,
        )

    def active_probe_metrics(self) -> DatasetMetrics:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        return self._dataset_metrics_from_profiles(
            self.fixed_probe.examples,
            self.active_profiles,
        )

    def team_prompt_state_hash(self) -> str:
        prompt_hashes = [
            self.prompt_hash(agent.current_prompt) for agent in self.agents
        ]
        return hashlib.sha256(json.dumps(
            prompt_hashes,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    async def _evaluate_profiles(
        self,
        examples: Sequence[ProbeExample],
    ) -> list[tuple[PromptAnswer, ...]]:
        return list(await asyncio.gather(*(
            asyncio.gather(*(
                self.prompt_question_evaluator.evaluate(
                    question=example.question,
                    question_hash=example.question_hash,
                    prompt=agent.current_prompt,
                    prompt_hash=self.prompt_hash(agent.current_prompt),
                    agent_id=agent_id,
                    solve=self.solve,
                )
                for example in examples
            ))
            for agent_id, agent in enumerate(self.agents)
        )))

    async def evaluate_dataset(
        self,
        data: Sequence[Mapping[str, Any]],
        *,
        validation: bool = False,
    ) -> DatasetMetrics:
        if validation:
            if self.validation_probe is None:
                raise RuntimeError(
                    "validation evaluation is disabled by the active final-state lifecycle unless a compatibility probe is supplied"
                )
            examples = self._probe_examples(data)
            profiles = list(await asyncio.gather(*(
                self.validation_probe.evaluate_prompt(
                    agent_id,
                    agent.current_prompt,
                    self.prompt_hash(agent.current_prompt),
                    self.solve,
                )
                for agent_id, agent in enumerate(self.agents)
            )))
            return self._dataset_metrics_from_profiles(examples, profiles)
        examples = self._probe_examples(data)
        profiles = await self._evaluate_profiles(examples)
        self._last_evaluated_examples = examples
        self._last_evaluated_profiles = profiles
        return self._dataset_metrics_from_profiles(examples, profiles)

    async def evaluate_validation_state(
        self,
        data: Sequence[Mapping[str, Any]],
    ) -> tuple[DatasetMetrics, dict[str, Any]]:
        """Historical-report compatibility; excluded from active optimization runs."""
        state_hash = self.team_prompt_state_hash()
        cached = self.validation_state_cache.get(state_hash)
        if cached is not None:
            self.validation_reuse_count += 1
            return dataset_metrics_from_dict(cached["metrics"]), {
                "team_prompt_state_hash": state_hash,
                "validation_cache_hit": True,
                "validation_result_source": "compatibility_cache",
            }
        metrics = await self.evaluate_dataset(data, validation=True)
        self.validation_evaluation_count += 1
        self.validation_state_cache[state_hash] = {"metrics": metrics.to_dict()}
        return metrics, {
            "team_prompt_state_hash": state_hash,
            "validation_cache_hit": False,
            "validation_result_source": "compatibility_solver_evaluation",
        }

    def validation_key(
        self,
        metrics: DatasetMetrics,
        initial: DatasetMetrics,
        epoch: int,
    ) -> tuple | None:
        """Legacy read-only ordering retained outside the active optimization loop."""
        if any(
            current < baseline
            for current, baseline in zip(
                metrics.per_agent_correct_counts,
                initial.per_agent_correct_counts,
                strict=True,
            )
        ) or metrics.terminal_invalid_count > initial.terminal_invalid_count + 1:
            return None
        if metrics.vote_correct_count < initial.vote_correct_count:
            return None
        gains = tuple(
            current - baseline
            for current, baseline in zip(
                metrics.per_agent_correct_counts,
                initial.per_agent_correct_counts,
                strict=True,
            )
        )
        return (
            min(gains), metrics.vote_correct_count, sum(gains),
            sum(value > 0 for value in gains), metrics.mean_soft_vote_utility,
            -metrics.c0_count, -metrics.mean_invalid_rate, -int(epoch),
        )

    def complete_validation_selection(self, _checkpoint: Mapping[str, Any]) -> None:
        self._compat_validation_selection_completed = True

    async def evaluate_selected_test(
        self,
        data: Sequence[Mapping[str, Any]],
    ) -> DatasetMetrics:
        if not self._compat_validation_selection_completed:
            raise RuntimeError("test evaluation is forbidden before validation selection")
        self.training_completed = True
        return await self.evaluate_final_test(data)

    def mark_training_complete(self, planned_update_count: int) -> None:
        if self.completed_update_count != int(planned_update_count):
            raise RuntimeError(
                "cannot complete training before every planned update finishes"
            )
        self.planned_update_count = int(planned_update_count)
        self.training_completed = True

    async def evaluate_final_test(
        self,
        data: Sequence[Mapping[str, Any]],
    ) -> DatasetMetrics:
        if not self.training_completed:
            self.test_called_before_training_complete = True
            raise RuntimeError("test evaluation is forbidden before training completes")
        if self.test_evaluation_count:
            if not self.selected_test_metrics:
                raise RuntimeError(
                    "test count is non-zero without persisted final metrics"
                )
            return dataset_metrics_from_dict(self.selected_test_metrics)
        metrics = await self.evaluate_dataset(data)
        examples = self._last_evaluated_examples
        profiles = self._last_evaluated_profiles
        differentiation = team_behavior_metrics(
            examples=examples,
            profiles=profiles,
            normalize_answer=self.normalize_answer,
            match_answer=self.match_answer,
            tie_break=self.protocol.tie_policy,
            seed=self.cfg.training.seed,
        )
        self.selected_test_metrics = metrics.to_dict()
        self.final_test_differentiation = differentiation
        self.test_evaluation_count = 1
        return metrics

    def record_training_dynamics(
        self,
        *,
        update_index: int,
        incumbent_profiles: Sequence[Sequence[PromptAnswer]] | None = None,
    ) -> dict[str, Any]:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        behavior = team_behavior_metrics(
            examples=self.fixed_probe.examples,
            profiles=self.active_profiles,
            normalize_answer=self.normalize_answer,
            match_answer=self.match_answer,
            tie_break=self.protocol.tie_policy,
            seed=self.cfg.training.seed,
        )
        decision = (
            self.candidate_decisions[-1]
            if update_index >= 0 and self.candidate_decisions else {}
        )
        accepted_hash = str(decision.get("accepted_prompt_hash", ""))
        accepted = bool(accepted_hash)
        state_hash = self.team_prompt_state_hash()
        previous_hash = (
            str(self.training_dynamics[-1]["active_team_state_hash"])
            if self.training_dynamics else state_hash
        )
        candidate = next(
            (
                row for row in decision.get("candidates", [])
                if row.get("prompt_hash") == accepted_hash
            ),
            None,
        )
        if candidate is None and decision.get("candidates"):
            candidate = max(
                decision["candidates"],
                key=lambda row: (
                    int((row.get("constraint") or {}).get("target_gain", -10**9)),
                    str(row.get("prompt_hash", "")),
                ),
            )
        constraint = (candidate or {}).get("constraint") or {}
        rejection_reasons = sorted({
            reason
            for row in decision.get("candidates", [])
            for reason in ((row.get("constraint") or {}).get("rejection_reasons", []))
        })
        priorities = list(decision.get("agent_target_priorities", []))
        target_id = decision.get("target_agent_id")
        target_priority = next(
            (row for row in priorities if row.get("agent_id") == target_id), {}
        )
        row = {
            "update_index": int(update_index),
            "target_agent_id": target_id,
            "accepted": accepted,
            "active_team_state_hash": state_hash,
            "state_changed": state_hash != previous_hash,
            "target_sequence": [
                item.get("target_agent_id") for item in self.candidate_decisions
            ],
            "accepted_target_sequence": [
                item.get("target_agent_id") for item in self.candidate_decisions
                if item.get("accepted_prompt_hash")
            ],
            "target_attempt_count": target_priority.get("target_attempt_count"),
            "updates_since_selected": target_priority.get("updates_since_selected"),
            "max_wait_trigger": int(
                decision.get("max_wait_fairness_trigger_count", 0)
            ),
            "responsibility_portfolio": {
                key: target_priority.get(key)
                for key in (
                    "direct_fix_count",
                    "margin_gain_sum",
                    "target_pareto_front",
                )
            },
            "target_gain": constraint.get("target_gain"),
            "vote_gain_count": constraint.get("vote_gain_count"),
            "vote_loss_count": constraint.get("vote_loss_count"),
            "vote_net_gain": constraint.get("vote_net_gain"),
            "candidate_objective": constraint.get("candidate_objective"),
            "incumbent_objective": constraint.get("incumbent_objective"),
            "rejection_reasons": rejection_reasons,
            **behavior,
            "accepted_update_count_so_far": sum(
                bool(item.get("accepted_prompt_hash"))
                for item in self.candidate_decisions
            ),
            "distinct_improved_member_count": sum(
                current > initial
                for current, initial in zip(
                    behavior["per_agent_correct_counts"],
                    [
                        sum(self.match_answer(answer.answer, example.gold_answer)
                            for answer, example in zip(profile, self.fixed_probe.examples, strict=True))
                        for profile in self.initial_profiles
                    ],
                    strict=True,
                )
            ),
            "distinct_prompt_hash_count": len({
                self.prompt_hash(agent.current_prompt) for agent in self.agents
            }),
        }
        self.training_dynamics.append(row)
        if update_index < 0 or row["state_changed"]:
            self.team_differentiation_trajectory.append(dict(row))
        if accepted and incumbent_profiles is not None:
            transition = vote_transition_decomposition(
                examples=self.fixed_probe.examples,
                incumbent_profiles=incumbent_profiles,
                candidate_profiles=self.active_profiles,
                normalize_answer=self.normalize_answer,
                match_answer=self.match_answer,
                tie_break=self.protocol.tie_policy,
                seed=self.cfg.training.seed,
            )
            transition.update({
                "update_index": int(update_index),
                "target_agent_id": target_id,
                "active_team_state_hash": state_hash,
            })
            self.update_transition_decomposition.append(transition)
            transition_rows = g_transition_audit_rows(
                examples=self.fixed_probe.examples,
                incumbent_profiles=incumbent_profiles,
                candidate_profiles=self.active_profiles,
                target_agent_id=int(target_id),
                normalize_answer=self.normalize_answer,
                match_answer=self.match_answer,
                tie_break=self.protocol.tie_policy,
                seed=self.cfg.training.seed,
            )
            for audit_row in transition_rows:
                audit_row.update({
                    "artifact_schema_version": "g_transition_audit_v1",
                    "update_index": int(update_index),
                    "team_state_version": self.team_state_version,
                })
                self.g_transition_audit.append(audit_row)
            target = int(target_id)
            old_correct = [
                bool(answer.valid and self.match_answer(answer.answer, example.gold_answer))
                for answer, example in zip(
                    incumbent_profiles[target], self.fixed_probe.examples, strict=True
                )
            ]
            new_correct = [
                bool(answer.valid and self.match_answer(answer.answer, example.gold_answer))
                for answer, example in zip(
                    self.active_profiles[target], self.fixed_probe.examples, strict=True
                )
            ]
            context = next(
                (
                    item for item in reversed(self.tcs_context_history)
                    if item.get("update_index") == update_index
                    and item.get("target_agent_id") == target
                ),
                {},
            )
            selected_context_pattern_question_hashes = {
                str(pattern_id): tuple(str(question_hash) for question_hash in question_hashes)
                for pattern_id, question_hashes in dict(
                    context.get("selected_context_pattern_question_hashes", {})
                ).items()
            }
            repaired_question_hashes = {
                example.question_hash
                for example, old, new in zip(
                    self.fixed_probe.examples, old_correct, new_correct, strict=True
                )
                if not old and new
            }
            assigned_hashes = set(decision.get("assigned_question_hashes", ()))
            self.specialization_trajectory.append({
                "artifact_schema_version": "specialization_trajectory_v2",
                "update_index": int(update_index),
                "agent_id": target,
                "prompt_hash_before": str((decision.get("incumbent") or {}).get("prompt_hash", "")),
                "prompt_hash_after": accepted_hash,
                "prompt_character_count_before": len(str((decision.get("incumbent") or {}).get("prompt", ""))),
                "prompt_character_count_after": len(self.agents[target].current_prompt),
                "prompt_estimated_token_count_before": None,
                "prompt_estimated_token_count_after": None,
                "selected_context_pattern_ids": list(context.get("selected_pattern_ids", [])),
                "selected_context_pattern_question_hashes": {
                    pattern_id: list(question_hashes)
                    for pattern_id, question_hashes in selected_context_pattern_question_hashes.items()
                },
                "repaired_selected_pattern_ids": sorted(
                    pattern_id
                    for pattern_id, question_hashes in selected_context_pattern_question_hashes.items()
                    if repaired_question_hashes.intersection(question_hashes)
                ),
                "assigned_repaired_pattern_ids": sorted(
                    pattern_id
                    for pattern_id, question_hashes in selected_context_pattern_question_hashes.items()
                    if repaired_question_hashes.intersection(question_hashes).intersection(assigned_hashes)
                ),
                "correct_set_gain_count": sum(not old and new for old, new in zip(old_correct, new_correct, strict=True)),
                "correct_set_loss_count": sum(old and not new for old, new in zip(old_correct, new_correct, strict=True)),
                "correct_set_churn_count": sum(old != new for old, new in zip(old_correct, new_correct, strict=True)),
                "unique_coverage_gain_count": sum(
                    not item["unique_correct_before"] and item["unique_correct_after"]
                    for item in transition_rows
                ),
                "unique_coverage_loss_count": sum(
                    item["unique_correct_before"] and not item["unique_correct_after"]
                    for item in transition_rows
                ),
            })
        return row

    def proposal_memory_summary(self) -> dict[str, Any]:
        events = list(self.proposal_memory_events)
        hits = [row for row in events if row["memory_hit"]]
        failures = [row for row in events if row["failure_stage"] != "accepted"]
        accepted = [row for row in events if row["failure_stage"] == "accepted"]
        summaries = [
            summary for entry in self.proposal_memory_entries.values()
            for summary in entry.candidate_summaries
        ]
        return {
            "proposal_memory_version": PROPOSAL_MEMORY_VERSION,
            "memory_mode": self.cfg.tcs.proposal_memory_mode,
            "memory_hit_count": len(hits),
            "entry_count_by_agent": {
                str(agent_id): sum(
                    entry.key.target_agent_id == agent_id
                    for entry in self.proposal_memory_entries.values()
                )
                for agent_id in range(5)
            },
            "cross_agent_collision_count": 0,
            "exact_bundle_immediate_repeat_rate": (
                sum(
                    row["previous_evidence_bundle_hash"]
                    == row["current_evidence_bundle_hash"]
                    for row in events if row["previous_evidence_bundle_hash"]
                )
                / max(1, len(hits))
            ),
            "critic_level_failure_count": sum(
                row["failure_stage"] == "critic" for row in failures
            ),
            "behavior_level_failure_count": sum(
                row["failure_stage"] in {"zero_repair_behavior", "regressive_progress"}
                for row in failures
            ),
            "assigned_residual_repair_candidate_rate": (
                sum(summary.assigned_residual_repair_count > 0 for summary in summaries)
                / len(summaries) if summaries else None
            ),
            "target_or_vote_positive_candidate_rate": (
                sum(
                    summary.target_gain > 0 or summary.vote_net_gain > 0
                    for summary in summaries
                ) / len(summaries) if summaries else None
            ),
            "acceptable_candidate_rate": len(accepted) / max(1, len(events)),
            "longest_consecutive_rejection_length": self._longest_rejection_streak(events),
            "memory_hit_acceptance_rate": (
                sum(row["acceptable_after_memory_hit"] for row in hits) / max(1, len(hits))
            ),
            "tokens_per_accepted_update": self.cost_summary().get("tokens_per_accepted_update"),
        }

    @staticmethod
    def _longest_rejection_streak(events: Sequence[Mapping[str, Any]]) -> int:
        longest = current = 0
        for row in events:
            if row.get("failure_stage") == "accepted":
                current = 0
            else:
                current += 1
                longest = max(longest, current)
        return longest

    def proposal_memory_isolation_audit(self) -> dict[str, Any]:
        entries = list(self.proposal_memory_entries.values())
        keys_unique = len({entry.key.key_hash() for entry in entries}) == len(entries)
        eligibility_valid = all(
            all(
                entry.key.target_agent_id in self.cached_responsibility_eligibility.get(question_hash, ())
                for question_hash in entry.assigned_question_hashes
            )
            for entry in entries
            if entry.key.team_state_version == self.team_state_version
        )
        return {
            "proposal_memory_version": PROPOSAL_MEMORY_VERSION,
            "memory_mode": self.cfg.tcs.proposal_memory_mode,
            "entry_count": len(entries),
            "key_hashes_unique": keys_unique,
            "cross_agent_collision_count": 0,
            "all_entry_eligible_residuals_match_target": eligibility_valid,
            "partial_key_fallback_used": False,
        }

    def run_meta(self) -> dict[str, Any]:
        if self.run_identity is None:
            raise RuntimeError("run identity must be set before writing run metadata")
        initial_hashes = [self.prompt_hash(agent.initial_prompt) for agent in self.agents]
        return {
            "method_version": METHOD_VERSION,
            "experiment_protocol": asdict(self.protocol),
            "run_identity": self.run_identity.to_dict(),
            "initialization_mode": self.protocol.initialization_mode.value,
            "initial_prompt_hashes": initial_hashes,
            "initial_prompts_identical": len(set(initial_hashes)) == 1,
            "tie_policy": self.protocol.tie_policy,
            "update_mode": "single_agent_paired_counterfactual",
            "candidate_selector": self.protocol.candidate_selection_policy,
            "candidate_generator": self.protocol.tcs_context_policy,
            "member_objective_version": "integer_vote_min_sum_v2",
            "responsibility_version": RESPONSIBILITY_VERSION,
            "responsibility_lifecycle_version": "one_refresh_per_team_state_v1",
            "target_selection_version": TARGET_SELECTION_VERSION,
            "pareto_preference_version": "member_first_candidate_preference_v1",
            "stage_a_version": "team_vote_worst_mean_v2",
            "stage_b_version": CANDIDATE_ACCEPTANCE_VERSION,
            "candidate_acceptance_version": CANDIDATE_ACCEPTANCE_VERSION,
            "preservation_policy_version": PRESERVATION_POLICY_VERSION,
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "checkpoint_selection_version": CHECKPOINT_SELECTION_VERSION,
            "test_isolation_version": TEST_ISOLATION_VERSION,
            "tcs_context_version": TCS_CONTEXT_VERSION,
            "proposal_memory_version": PROPOSAL_MEMORY_VERSION,
            "proposal_memory_mode": self.cfg.tcs.proposal_memory_mode,
            "proposal_memory_run_id": self.proposal_memory_run_id,
            "diagnosis_aggregation_version": DIAGNOSIS_AGGREGATION_VERSION,
            "answer_role_encoding_version": ANSWER_ROLE_ENCODING_VERSION,
            "pattern_selection_version": PATTERN_SELECTION_VERSION,
            "teacher_schema_version": TEACHER_SCHEMA_VERSION,
            "teacher_revision_protocol_version": TEACHER_REVISION_PROTOCOL_VERSION,
            "critic_schema_version": CRITIC_SCHEMA_VERSION,
            "student_schema_version": STUDENT_SCHEMA_VERSION,
            "role_retry_policy_version": ROLE_RETRY_POLICY_VERSION,
            "completion_policy": "provider_default",
            "solver_invalid_retry_policy_version": SOLVER_INVALID_RETRY_POLICY_VERSION,
            "prompt_question_evaluator_version": PROMPT_QUESTION_EVALUATOR_VERSION,
            "solver_invalid_max_retries": self.cfg.models.solver_invalid_max_retries,
            "max_pattern_count": self.cfg.tcs.tcs_max_pattern_summaries,
            "max_evidence_case_count": self.cfg.tcs.tcs_max_evidence_cases,
            "teacher_total_character_limit": self.cfg.tcs.teacher_total_max_chars,
            "candidate_prompt_length_limit": self.cfg.tcs.candidate_prompt_max_chars,
            "total_candidate_prompt_length_limit": self.cfg.tcs.total_candidate_prompt_max_chars,
            "student_count_policy": "reject_excess_keep_individually_valid_v1",
            "student_invalid_recovery_version": STUDENT_INVALID_RECOVERY_VERSION,
            "model_facing_payload_version": "audit_hash_isolated_v2",
            "terminal_failure_version": "role_specific_terminal_failure_v1",
            "checkpoint_version": CHECKPOINT_VERSION,
            "tcs_protocol_version": TCS_PROTOCOL_VERSION,
            "critic_approval_basis": "failed_checks_empty",
            "task_general_scope": "unseen_examples_within_current_task",
            "student_sample_memorization_filter": SAMPLE_MEMORIZATION_FILTER_VERSION,
            "solver_sampling_semantics": "shared_prompt_question_output",
            "solver_output_contract_version": self.cfg.peer_state.solver_output_contract_version,
            "solver_request_template_version": SOLVER_REQUEST_TEMPLATE_VERSION,
            "prompt_question_evaluator_identity": self.prompt_question_evaluator.identity(),
            "prompt_question_cache_hits": self.prompt_question_evaluator.cache_hits,
            "prompt_question_cache_misses": self.prompt_question_evaluator.cache_misses,
            "shared_solver_cache_path": str(self.cfg.persistence.shared_solver_cache_path or ""),
            "shared_solver_cache_hits": (
                self.shared_solver_cache.hits if self.shared_solver_cache is not None else 0
            ),
            "shared_solver_cache_misses": (
                self.shared_solver_cache.misses if self.shared_solver_cache is not None else 0
            ),
            "shared_solver_cache_waits": (
                self.shared_solver_cache.waits if self.shared_solver_cache is not None else 0
            ),
            "shared_solver_cache_ready_entries": (
                self.shared_solver_cache.ready_entry_count()
                if self.shared_solver_cache is not None
                else len(self.prompt_question_evaluator.cache)
            ),
            "shared_solver_cache_content_hash": (
                self.shared_solver_cache.ready_content_hash()
                if self.shared_solver_cache is not None
                else ""
            ),
            "true_plurality_vote_used": True,
            "generic_diversity_reward_used": False,
            "trace_diversity_used_for_selection": False,
            "legacy_compatibility_enabled": False,
            "probe_version": self.cfg.peer_state.probe_version,
            "probe_hash": self.fixed_probe.probe_hash if self.fixed_probe else "",
            "validation_used": False,
            "validation_probe_hash": "",
            "validation_unique_state_count": 0,
            "validation_evaluation_count": 0,
            "validation_reuse_count": 0,
            "planned_update_count": self.planned_update_count,
            "completed_update_count": self.completed_update_count,
            "training_completed": self.training_completed,
            "final_state_selection": dict(self.final_state_selection),
            "test_evaluation_count": self.test_evaluation_count,
            "test_used_for_selection": self.test_used_for_selection,
            "test_used_for_training": self.test_used_for_training,
            "test_called_before_training_complete": (
                self.test_called_before_training_complete
            ),
            "config": self.cfg.to_flat_dict(),
        }

    def candidate_funnel_summary(
        self,
        decisions: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        rows = list(self.candidate_decisions if decisions is None else decisions)
        funnels = [dict(row.get("funnel", {})) for row in rows]
        terminal_counts: dict[str, int] = {}
        for funnel in funnels:
            failure = str(funnel.get("terminal_failure_class", ""))
            if failure:
                terminal_counts[failure] = terminal_counts.get(failure, 0) + 1
        return {
            "update_count": len(funnels),
            "terminal_failure_counts": terminal_counts,
            "terminal_failures": [
                {
                    "update_index": row.get("update_index"),
                    "target_agent_id": row.get("target_agent_id"),
                    "terminal_failure_class": row.get("funnel", {}).get(
                        "terminal_failure_class", ""
                    ),
                    "terminal_failure_role": row.get("funnel", {}).get(
                        "terminal_failure_role", ""
                    ),
                }
                for row in rows
                if row.get("funnel", {}).get("terminal_failure_class")
            ],
            "updates": funnels,
        }

    def cost_summary(self) -> dict[str, Any]:
        summary = self.llm.cost_summary()
        successful_candidates = sum(
            int(row.get("funnel", {}).get("deduplicated_count", 0))
            for row in self.candidate_decisions
        )
        stage_a_candidates = sum(
            int(row.get("funnel", {}).get("stage_a_evaluated", 0))
            for row in self.candidate_decisions
        )
        accepted_updates = sum(
            bool(row.get("funnel", {}).get("accepted_candidate", False))
            for row in self.candidate_decisions
        )
        total_tokens = int(summary["total_tokens"])
        summary.update({
            "successful_candidate_count": successful_candidates,
            "stage_a_candidate_count": stage_a_candidates,
            "accepted_update_count": accepted_updates,
            "tokens_per_successful_candidate": (
                total_tokens / successful_candidates if successful_candidates else None
            ),
            "tokens_per_stage_a_candidate": (
                total_tokens / stage_a_candidates if stage_a_candidates else None
            ),
            "tokens_per_accepted_update": (
                total_tokens / accepted_updates if accepted_updates else None
            ),
        })
        return summary

    def flush_artifacts(self) -> None:
        self.artifacts.write_json("run_meta.json", self.run_meta())
        self.artifacts.write_json("history.json", self.history)
        self.artifacts.write_json("best_prompts.json", [agent.current_prompt for agent in self.agents])
        self.artifacts.write_jsonl("peer_state_history.jsonl", self.peer_state_history)
        self.artifacts.write_jsonl("responsibility_assignments.jsonl", self.responsibility_assignments)
        self.artifacts.write_jsonl("member_opportunities.jsonl", self.member_opportunities)
        self.artifacts.write_jsonl("g_transition_audit.jsonl", self.g_transition_audit)
        self.artifacts.write_jsonl("specialization_trajectory.jsonl", self.specialization_trajectory)
        self.artifacts.write_jsonl("target_priority_audit.jsonl", self.target_priority_audit)
        self.artifacts.write_jsonl(
            "responsibility_portfolio_trajectory.jsonl",
            self.responsibility_portfolio_trajectory,
        )
        self.artifacts.write_jsonl(
            "target_responsibility_context_alignment.jsonl",
            self.target_responsibility_context_alignment,
        )
        self.artifacts.write_jsonl("candidate_decisions.jsonl", self.candidate_decisions)
        self.artifacts.write_jsonl(
            "proposal_memory_events_sanitized.jsonl", self.proposal_memory_events,
        )
        self.artifacts.write_json(
            "proposal_memory_summary.json",
            self.proposal_memory_summary(),
        )
        self.artifacts.write_json(
            "proposal_memory_key_isolation_audit.json",
            self.proposal_memory_isolation_audit(),
        )
        self.artifacts.write_jsonl(
            "proposal_rotation_trajectory.jsonl", self.proposal_rotation_trajectory,
        )
        self.artifacts.write_json(
            "candidate_funnel.json", self.candidate_funnel_summary()
        )
        self.artifacts.write_jsonl("tcs_context_history.jsonl", self.tcs_context_history)
        self.artifacts.write_jsonl("tcs_rounds.jsonl", self.tcs_rounds)
        self.artifacts.write_jsonl(
            "student_recovery_observations.jsonl",
            self.student_recovery_observations,
        )
        self.artifacts.write_jsonl("solver_invalid_outputs.jsonl", self.solver_invalid_outputs)
        self.artifacts.write_json("solver_recovery_summary.json", self.solver_recovery_summary())
        self.artifacts.write_jsonl("llm_calls.jsonl", self.llm.calls)
        self.artifacts.write_json("cost_summary.json", self.cost_summary())
        self.artifacts.write_jsonl("training_dynamics.jsonl", self.training_dynamics)
        self.artifacts.write_jsonl(
            "team_differentiation_trajectory.jsonl",
            self.team_differentiation_trajectory,
        )
        self.artifacts.write_jsonl(
            "update_transition_decomposition.jsonl",
            self.update_transition_decomposition,
        )
        self.artifacts.write_json(
            "final_test_differentiation.json",
            self.final_test_differentiation,
        )

    def solver_recovery_summary(self) -> dict[str, Any]:
        rows = list(self.solver_recovery_observations)
        attempts = [int(row["solver_attempt_count"]) for row in rows]
        first_valid = sum(bool(row["first_attempt_valid"]) for row in rows)
        recovered = sum(bool(row["recovered_from_invalid"]) for row in rows)
        terminal = sum(bool(row["terminal_invalid"]) for row in rows)
        first_status: dict[str, int] = {}
        terminal_status: dict[str, int] = {}
        finish_by_attempt: dict[str, int] = {}
        for row in rows:
            statuses = list(row["attempt_validity_statuses"])
            if statuses:
                first_status[statuses[0]] = first_status.get(statuses[0], 0) + 1
            if row["terminal_invalid"]:
                status = str(row["validity_status"])
                terminal_status[status] = terminal_status.get(status, 0) + 1
            for attempt_index, finish_reason in enumerate(row["attempt_finish_reasons"], 1):
                key = f"{attempt_index}:{finish_reason}"
                finish_by_attempt[key] = finish_by_attempt.get(key, 0) + 1
        extra_calls = sum(max(0, count - 1) for count in attempts)
        extra_prompt = sum(int(row["recovery_prompt_tokens"]) for row in rows)
        extra_completion = sum(int(row["recovery_completion_tokens"]) for row in rows)
        extra_total = sum(int(row["recovery_total_tokens"]) for row in rows)
        total_tokens = sum(
            sum(int(value) for value in row["attempt_total_tokens"])
            for row in rows
        )
        count = len(rows)
        current = self._current_run_recovery_summary()
        current_count = int(current["current_run_recovery_api_calls"])
        current_total = int(current["current_run_recovery_total_tokens"])
        current["current_run_recovery_call_overhead_rate"] = (
            current_count / count if count else 0.0
        )
        current["current_run_recovery_token_overhead_rate"] = (
            current_total / total_tokens if total_tokens else 0.0
        )
        return {
            "unique_resolved_request_count": count,
            "first_attempt_valid_count": first_valid,
            "first_attempt_invalid_count": count - first_valid,
            "recovered_invalid_count": recovered,
            "terminal_invalid_count": terminal,
            "attempt_count_histogram": {
                str(value): attempts.count(value) for value in sorted(set(attempts))
            },
            "first_pass_validity_status_counts": first_status,
            "terminal_validity_status_counts": terminal_status,
            "finish_reason_counts_by_attempt": finish_by_attempt,
            "invalid_recovery_extra_calls": extra_calls,
            "invalid_recovery_prompt_tokens": extra_prompt,
            "invalid_recovery_completion_tokens": extra_completion,
            "invalid_recovery_total_tokens": extra_total,
            "first_attempt_valid_rate": first_valid / count if count else 1.0,
            "eventual_valid_rate": (count - terminal) / count if count else 1.0,
            "recovery_call_overhead_rate": extra_calls / count if count else 0.0,
            "recovery_token_overhead_rate": extra_total / total_tokens if total_tokens else 0.0,
            **current,
        }
