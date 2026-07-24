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
    ConstraintLimits,
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
)
from .evaluation.prompt_question import PromptQuestionEvaluator
from .evaluation.output_contract import SOLVER_OUTPUT_CONTRACT_VERSION, solver_output_contract
from .evaluation.persistent_solver_cache import PersistentSolverCache
from .evaluation.solver_output import parse_solver_output
from .llm_client import LLMBudgetExceeded, LLMCallResult, RoleAwareLLMClient
from .member_objectives import member_gain_metrics, team_member_gain_state
from .peer_state import (
    PeerVoteContext,
    TeamVoteState,
    build_peer_vote_context,
    build_team_vote_state,
    soft_vote_utility,
)
from .persistence.artifacts import ArtifactWriter
from .persistence.identity import RunIdentity, solver_request_components, solver_request_identity
from .protocol import CandidateBudgetContract, ExperimentProtocol, experiment_protocol
from .responsibility import (
    MemberAwareRepairOpportunity,
    ResponsibilityState,
    assign_primary_responsibilities,
    compute_member_aware_repair_opportunity,
    select_target_agent,
    target_priorities,
)
from .tasks import get_task_spec
from .tcs import (
    CRITIC_SCHEMA_VERSION,
    ROLE_RETRY_POLICY_VERSION,
    STUDENT_SCHEMA_VERSION,
    TEACHER_SCHEMA_VERSION,
    AccuracyDiagnosisContext,
    AnyDiagnosisContext,
    CriticDecision,
    MemberAwareDiagnosisContext,
    PeerStateDiagnosisContext,
    PreviousUpdateOutcome,
    StudentPromptCandidate,
    TCSContextDiagnostics,
    TCS_PROTOCOL_VERSION,
    SAMPLE_MEMORIZATION_FILTER_VERSION,
    TeacherRepairPlan,
    build_critic_request,
    build_student_request,
    build_teacher_request,
    contains_supplied_example_text,
    context_payload,
    limit_diagnosis_context,
    parse_critic_decision,
    parse_student_candidates,
    parse_teacher_repair_plan,
    response_truncated,
    serialize_context,
)
from .utils import extract_json_obj, normalize_prompt_text, normalize_spaces


METHOD_VERSION = "member_aware_peer_state_v2"


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
    schema_valid_count: int = 0
    sample_memorization_rejected: int = 0
    non_parent_count: int = 0
    deduplicated_count: int = 0
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
    rejected_local_accuracy: int = 0
    rejected_initial_accuracy: int = 0
    rejected_invalid: int = 0
    rejected_vote_loss: int = 0
    rejected_unique_loss: int = 0
    rejected_pivotal_loss: int = 0
    acceptable_candidates: int = 0
    accepted_candidate: bool = False


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
            raise ValueError("member_aware_peer_state_v2 requires exactly five agents")
        if cfg.peer_state.aggregation_mode != "plurality":
            raise ValueError("member_aware_peer_state_v2 requires plurality aggregation")
        if cfg.peer_state.vote_tie_break != "abstain":
            raise ValueError("member_aware_peer_state_v2 requires tie-as-abstain")
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
        if cfg.tcs.student_json_max_retries < 0:
            raise ValueError("student_json_max_retries cannot be negative")
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
        self.task_spec = get_task_spec(cfg.data.task_type)
        prompts = self._initial_prompts()
        self.agents = [AgentRuntime(prompt, prompt) for prompt in prompts]
        self.responsibility_state = ResponsibilityState(
            assigned_load_by_agent={agent_id: 0 for agent_id in range(cfg.training.agents)},
            updates_since_selected_by_agent={agent_id: 0 for agent_id in range(cfg.training.agents)},
            accepted_updates_by_agent={agent_id: 0 for agent_id in range(cfg.training.agents)},
        )
        self.history: list[dict[str, Any]] = []
        self.peer_state_history: list[dict[str, Any]] = []
        self.responsibility_assignments: list[dict[str, Any]] = []
        self.candidate_decisions: list[dict[str, Any]] = []
        self.tcs_context_history: list[dict[str, Any]] = []
        self.tcs_rounds: list[dict[str, Any]] = []
        self.solver_invalid_outputs: list[dict[str, Any]] = []
        self._audited_invalid_keys: set[tuple[str, str]] = set()
        self.cached_responsibility_owners: dict[str, int] = {}
        self.cached_responsibility_assignments: dict[int, list[MemberAwareRepairOpportunity]] = {}
        self.cached_member_opportunities: dict[
            str, tuple[MemberAwareRepairOpportunity, ...]
        ] = {}
        self.team_state_version = 0
        self.responsibility_state_version = -1
        self.responsibility_refresh_count = 0
        self.target_priority_audit: list[dict[str, Any]] = []
        self.previous_update_outcomes = {
            agent_id: PreviousUpdateOutcome() for agent_id in range(5)
        }
        self.agent_selection_counts = {agent_id: 0 for agent_id in range(5)}
        self.fixed_probe: FixedProbeEvaluator | None = None
        self.validation_probe: ValidationProbeEvaluator | None = None
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
            "created_at": answer.created_at,
        })

    async def _chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        client_role: str,
    ) -> LLMCallResult:
        return await self.llm.chat_result(
            model, system_prompt, user_prompt, temperature, max_tokens, client_role,
        )

    async def solve(self, question: str, agent_id: int, prompt: str) -> PromptAnswer:
        request_identity = self.prompt_question_evaluator.model_request_identity
        if self._solver_override is not None:
            self.llm.check_budget()
            started = time.time()
            answer = await self._solver_override(question, agent_id, prompt)
            self.llm.record_override_solver(started=started)
            return answer if answer.request_identity else replace(
                answer,
                request_identity=request_identity,
            )
        async with self.solver_semaphore:
            result = await self.llm.chat_result(
                self.cfg.models.agent_model,
                (
                    "Follow the supplied decision procedure.\n\n"
                    + solver_output_contract(self.cfg.data.answer_format)
                    + "\n\nDecision procedure:\n"
                    + prompt
                ),
                question,
                self.cfg.models.temperature,
                self.cfg.models.solver_max_tokens,
                "solver",
            )
        answer = parse_solver_output(
            result.text,
            question=question,
            task_spec=self.task_spec,
            answer_format=self.cfg.data.answer_format,
        )
        return replace(
            answer,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            request_identity=request_identity,
        )

    def build_probe(self, data: Sequence[Mapping[str, Any]]) -> FixedProbeEvaluator:
        return FixedProbeEvaluator(
            self._probe_examples(data),
            self.cfg.peer_state.probe_version,
            self.prompt_question_evaluator,
        )

    def build_validation_probe(self, data: Sequence[Mapping[str, Any]]) -> ValidationProbeEvaluator:
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
        member_correct_counts = self._member_correct_counts(self.active_profiles)
        initial_correct_counts = self._member_correct_counts(self.initial_profiles)
        member_gains = tuple(
            current - initial
            for current, initial in zip(
                member_correct_counts, initial_correct_counts, strict=True
            )
        )
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
        unique_correct_counts = tuple(
            sum(
                int(
                    state.team_correctness[agent_id]
                    and contexts[state.question_hash][agent_id].peer_gold_vote_count == 0
                )
                for state in states
            )
            for agent_id in range(5)
        )
        pivotal_correct_counts = tuple(
            sum(
                int(
                    state.team_correctness[agent_id]
                    and state.vote_correct
                    and contexts[state.question_hash][agent_id].peer_margin <= 0
                )
                for state in states
            )
            for agent_id in range(5)
        )
        for state in states:
            peer_by_agent = contexts[state.question_hash]
            opportunities[state.question_hash] = tuple(
                compute_member_aware_repair_opportunity(
                    team_state=state,
                    peer_context=peer_by_agent[agent_id],
                    initial_correct_counts=initial_correct_counts,
                    member_correct_counts=member_correct_counts,
                    member_gains_from_initial=member_gains,
                    unique_correct_counts=unique_correct_counts,
                    pivotal_correct_counts=pivotal_correct_counts,
                    tau=self.cfg.peer_state.soft_vote_tau,
                )
                for agent_id in range(5)
            )
        return tuple(states), contexts, opportunities

    def assign_responsibilities(
        self,
    ) -> tuple[dict[str, int], dict[int, list[MemberAwareRepairOpportunity]]]:
        if self.responsibility_state_version == self.team_state_version:
            return (
                dict(self.cached_responsibility_owners),
                {
                    agent_id: list(rows)
                    for agent_id, rows in self.cached_responsibility_assignments.items()
                },
            )
        states, _, opportunities = self.current_states_and_opportunities()
        self.cached_member_opportunities = dict(opportunities)
        state_by_hash = {state.question_hash: state for state in states}
        old_owners = dict(self.responsibility_state.primary_owner_by_question)
        owners, assigned, owner_audits = assign_primary_responsibilities(
            team_states=state_by_hash,
            opportunities=opportunities,
            state=self.responsibility_state,
            seed=self.cfg.training.seed,
            responsibility_switch_margin=(
                self.cfg.responsibility.responsibility_switch_margin
            ),
        )
        owner_switch_count = sum(
            question_hash in old_owners and old_owners[question_hash] != owner
            for question_hash, owner in owners.items()
        )
        rows = [row for values in assigned.values() for row in values]
        member_gain_state = self.current_team_member_gain_state()
        first_question_rows = next(iter(opportunities.values()), ())
        opportunity_by_agent = {
            row.agent_id: row for row in first_question_rows
        }
        self.peer_state_history.extend(asdict(state) for state in states)
        self.responsibility_assignments.append({
            "team_state_version": self.team_state_version,
            "member_gain_counts": member_gain_state["gain_counts"],
            "minimum_member_gain_count": member_gain_state["minimum_gain_count"],
            "total_member_gain_count": member_gain_state["total_gain_count"],
            "improvement_need_by_agent": {
                str(agent_id): opportunity_by_agent[agent_id].improvement_need
                for agent_id in sorted(opportunity_by_agent)
            },
            "protection_counts_by_agent": {
                str(agent_id): {
                    "unique_correct_count": opportunity_by_agent[agent_id].unique_correct_count,
                    "pivotal_correct_count": opportunity_by_agent[agent_id].pivotal_correct_count,
                }
                for agent_id in sorted(opportunity_by_agent)
            },
            "owner_distribution": {
                str(agent_id): sum(owner == agent_id for owner in owners.values()) for agent_id in range(5)
            },
            "owners": owners,
            "owner_switch_count": owner_switch_count,
            "owner_age": dict(self.responsibility_state.owner_age_by_question),
            "assigned_load_by_agent": dict(self.responsibility_state.assigned_load_by_agent),
            "direct_fix_responsibility_count": sum(row.direct_vote_fix for row in rows),
            "coverage_responsibility_count": sum(row.coverage_opportunity for row in rows),
            "dominant_wrong_responsibility_count": sum(row.dominant_wrong_member for row in rows),
            "owner_candidate_pareto_fronts": {
                question_hash: audit["candidate_pareto_fronts"]
                for question_hash, audit in owner_audits.items()
            },
            "owner_chosen_reasons": {
                question_hash: audit["chosen_reason"]
                for question_hash, audit in owner_audits.items()
            },
            "owner_assignment_audit": owner_audits,
            "assigned_opportunities": {
                str(agent_id): [asdict(row) for row in values] for agent_id, values in assigned.items()
            },
        })
        self.cached_responsibility_owners = dict(owners)
        self.cached_responsibility_assignments = {
            agent_id: list(values) for agent_id, values in assigned.items()
        }
        self.responsibility_state_version = self.team_state_version
        self.responsibility_refresh_count += 1
        return owners, assigned

    def ensure_responsibility_current(
        self,
    ) -> tuple[dict[str, int], dict[int, list[MemberAwareRepairOpportunity]]]:
        return self.assign_responsibilities()

    def refresh_responsibility_after_commit(
        self,
    ) -> tuple[dict[str, int], dict[int, list[MemberAwareRepairOpportunity]]]:
        self.team_state_version += 1
        if self.responsibility_state_version == self.team_state_version:
            raise AssertionError("committed team state must invalidate responsibility state")
        return self.assign_responsibilities()

    def select_target(
        self,
        assigned: Mapping[int, Sequence[MemberAwareRepairOpportunity]],
        update_index: int,
    ) -> tuple[int, bool, list[dict[str, Any]]]:
        max_wait = self.cfg.responsibility.responsibility_max_wait_updates
        _, _, opportunities = self.current_states_and_opportunities()
        priorities = target_priorities(
            opportunities=opportunities,
            assignments=assigned,
            state=self.responsibility_state,
            seed=self.cfg.training.seed,
            max_wait_updates=max_wait,
        )
        fairness = any(row.overdue for row in priorities)
        if self.protocol.target_selection_policy == "round_robin":
            target = update_index % 5
            fairness = False
        elif self.protocol.target_selection_policy == "member_aware_responsibility":
            target = select_target_agent(priorities)
        else:
            raise ValueError(
                f"Protocol has no optimization target selector: {self.protocol.name}"
            )
        priority_payload = [
            {
                **asdict(row),
                "protection_risk": row.protection_risk,
                "selected": row.agent_id == target,
            }
            for row in priorities
        ]
        self.target_priority_audit.append({
            "update_index": int(update_index),
            "priorities": priority_payload,
            "overdue_first": fairness,
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
                if not opportunities[state.question_hash][target_agent_id].current_correct
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
                -self.responsibility_state.owner_age_by_question.get(states[index].question_hash, 0),
                -opportunities[states[index].question_hash][target_agent_id].oracle_soft_utility_gain,
                states[index].question_hash,
            ))
            conversion.sort(key=lambda index: (
                -int(opportunities[states[index].question_hash][target_agent_id].direct_vote_fix),
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
    ) -> tuple[AnyDiagnosisContext, TCSContextDiagnostics]:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        states, contexts, opportunities = self.current_states_and_opportunities()
        target_rows = [
            opportunities[state.question_hash][target_agent_id] for state in states
        ]
        target_improvement_need = max(
            (row.improvement_need for row in target_rows if row.member_error),
            default=0,
        )
        aggregation = aggregate_probe_diagnosis(
            target_agent_id=target_agent_id,
            examples=self.fixed_probe.examples,
            states=states,
            peer_contexts=contexts,
            opportunities=opportunities,
            assigned_hashes=assigned_hashes,
            owner_age_by_question=self.responsibility_state.owner_age_by_question,
            context_policy=self.protocol.tcs_context_policy,
            target_improvement_need=target_improvement_need,
            max_patterns=self.cfg.tcs.tcs_max_pattern_summaries,
            max_cases=self.cfg.tcs.tcs_max_evidence_cases,
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
            target_correct_count = sum(row.current_correct for row in target_rows)
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
        elif self.protocol.tcs_context_policy == "member_aware_responsibility_conditioned":
            current_counts = tuple(
                sum(
                    int(answer.valid and self.match_answer(answer.answer, example.gold_answer))
                    for answer, example in zip(profile, self.fixed_probe.examples, strict=True)
                )
                for profile in self.active_profiles
            )
            initial_counts = tuple(
                sum(
                    int(answer.valid and self.match_answer(answer.answer, example.gold_answer))
                    for answer, example in zip(profile, self.fixed_probe.examples, strict=True)
                )
                for profile in self.initial_profiles
            )
            context = MemberAwareDiagnosisContext(
                **common,
                member_correct_counts=current_counts,
                member_gains_from_initial=tuple(
                    current - initial
                    for current, initial in zip(current_counts, initial_counts, strict=True)
                ),
                target_improvement_need=target_improvement_need,
                assigned_residual_count=len(assigned_hashes),
            )
        else:
            raise ValueError(f"Unsupported TCS context policy: {self.protocol.tcs_context_policy}")
        return limit_diagnosis_context(
            context,
            max_chars=self.cfg.tcs.tcs_context_max_chars,
            full_probe_case_count=aggregation.full_probe_case_count,
            available_pattern_count=len(aggregation.available_patterns),
        )

    async def propose_candidates(
        self,
        target_agent_id: int,
        assigned_hashes: set[str],
        funnel: CandidateFunnel,
        update_index: int = -1,
    ) -> list[CandidateRuntime]:
        parent_prompt = self.agents[target_agent_id].current_prompt
        context, diagnostics = self._proposal_context(target_agent_id, parent_prompt, assigned_hashes)
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
                "assigned", "owner", "owner_age", "responsibility",
                "member_gain", "improvement_need",
            )
        else:
            forbidden_tokens = ()
        lowered_paths = tuple(path.lower() for path in field_paths)
        forbidden_check = {
            token: any(token in path for path in lowered_paths)
            for token in forbidden_tokens
        }
        responsibility_tokens = ("assigned", "owner_age", "responsibility")
        self.tcs_context_history.append({
            "update_index": update_index,
            "target_agent_id": target_agent_id,
            "context_type": type(context).__name__,
            "context_class": type(context).__name__,
            "parent_prompt_hash": self.prompt_hash(parent_prompt),
            "proposal_context_hash": hashlib.sha256(context_serialized.encode("utf-8")).hexdigest(),
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
            **asdict(diagnostics),
        })
        funnel.parents_considered = 1
        teacher_request = build_teacher_request(context)
        repair_plan: TeacherRepairPlan | None = None
        critic_decision: CriticDecision | None = None
        critic_feedback = ""
        context_hash = hashlib.sha256(context_serialized.encode("utf-8")).hexdigest()
        for semantic_round in range(1, self.cfg.tcs.teacher_critic_max_rounds + 1):
            user_request = (
                "Produce the repair proposal."
                if semantic_round == 1
                else f"Revise the proposal using this critic feedback: {critic_feedback}"
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
                        self.cfg.tcs.teacher_max_tokens,
                        "optimizer",
                    )
                except LLMBudgetExceeded:
                    raise
                except Exception as exc:
                    funnel.infrastructure_failed_updates += 1
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
                        "hit_completion_limit": False,
                        "response_truncated": False,
                        "failure_class": "transport_failure",
                        "retry_reason": type(exc).__name__,
                        "input_characters": len(teacher_request) + len(user_request),
                        "output_characters": 0,
                    })
                    return []
                teacher_raw = teacher_result.text
                parsed_teacher = extract_json_obj(teacher_raw)
                truncated = response_truncated(teacher_result)
                failure_class = (
                    "completion_truncation"
                    if truncated else "invalid_json"
                    if parsed_teacher is None else ""
                )
                parse_error = ""
                if not failure_class:
                    try:
                        repair_plan = parse_teacher_repair_plan(
                            parsed_teacher,
                            field_max_chars=self.cfg.tcs.teacher_field_max_chars,
                        )
                        if contains_supplied_example_text(
                            json.dumps(asdict(repair_plan), ensure_ascii=False), context,
                        ):
                            raise ValueError("teacher repair plan copies supplied sample text")
                    except (TypeError, ValueError) as exc:
                        failure_class = "schema_error"
                        parse_error = str(exc)
                if failure_class:
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
                    "hit_completion_limit": teacher_result.hit_completion_limit,
                    "response_truncated": truncated,
                    "failure_class": failure_class,
                    "retry_reason": failure_class if failure_class and format_attempt == 0 else "",
                    "parse_error": parse_error,
                    "input_characters": len(teacher_request) + len(user_request),
                    "output_characters": len(teacher_raw),
                })
                if repair_plan is not None:
                    break
            if repair_plan is None:
                return []

            critic_request = build_critic_request(context, repair_plan)
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
                        self.cfg.tcs.critic_max_tokens,
                        "evaluator",
                    )
                except LLMBudgetExceeded:
                    raise
                except Exception as exc:
                    funnel.infrastructure_failed_updates += 1
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
                        "hit_completion_limit": False,
                        "response_truncated": False,
                        "failure_class": "transport_failure",
                        "retry_reason": type(exc).__name__,
                        "input_characters": len(critic_request),
                        "output_characters": 0,
                    })
                    return []
                critic_raw = critic_result.text
                parsed_critic = extract_json_obj(critic_raw)
                truncated = response_truncated(critic_result)
                failure_class = (
                    "completion_truncation"
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
                    "hit_completion_limit": critic_result.hit_completion_limit,
                    "response_truncated": truncated,
                    "failure_class": failure_class,
                    "retry_reason": failure_class if failure_class and format_attempt == 0 else "",
                    "parse_error": parse_error,
                    "input_characters": len(critic_request),
                    "output_characters": len(critic_raw),
                })
                if critic_decision is not None:
                    break
            if critic_decision is None:
                return []
            if critic_decision.approved:
                funnel.critic_approved += 1
                break
            critic_feedback = critic_decision.feedback
        if repair_plan is None or critic_decision is None or not critic_decision.approved:
            return []

        parsed_candidates: tuple[StudentPromptCandidate, ...] = ()
        funnel.requested_candidate_count = self.cfg.tcs.num_candidates_per_parent
        student_request = build_student_request(
            parent_prompt=parent_prompt,
            approved_plan=repair_plan,
            answer_format=self.cfg.data.answer_format,
            candidate_count=self.cfg.tcs.num_candidates_per_parent,
            candidate_prompt_max_chars=self.cfg.tcs.candidate_prompt_max_chars,
        )
        for format_attempt in range(self.cfg.tcs.student_json_max_retries + 1):
            funnel.student_calls += 1
            try:
                student_result = await self._chat(
                    self.cfg.models.optimizer_model,
                    "Return strict JSON only.",
                    student_request,
                    self.cfg.tcs.student_temperature,
                    self.cfg.tcs.student_max_tokens,
                    "optimizer",
                )
            except LLMBudgetExceeded:
                raise
            except Exception as exc:
                funnel.infrastructure_failed_updates += 1
                self.tcs_rounds.append({
                    "update_index": update_index,
                    "target_agent_id": target_agent_id,
                    "role": "student",
                    "semantic_round": semantic_round,
                    "format_attempt": format_attempt,
                    "schema_valid": False,
                    "finish_reason": "",
                    "hit_completion_limit": False,
                    "response_truncated": False,
                    "failure_class": "transport_failure",
                    "retry_reason": type(exc).__name__,
                    "input_characters": len(student_request),
                    "output_characters": 0,
                })
                return []
            student_raw = student_result.text
            parsed = extract_json_obj(student_raw)
            truncated = response_truncated(student_result)
            failure_class = (
                "completion_truncation"
                if truncated else "invalid_json"
                if parsed is None else ""
            )
            raw_count = 0
            rejection_reasons: tuple[tuple[str, ...], ...] = ()
            parse_error = ""
            if not failure_class:
                try:
                    parsed_result = parse_student_candidates(
                        parsed,
                        parent_prompt=parent_prompt,
                        context=context,
                        candidate_prompt_max_chars=self.cfg.tcs.candidate_prompt_max_chars,
                    )
                    parsed_candidates = parsed_result.candidates
                    raw_count = parsed_result.raw_count
                    rejection_reasons = parsed_result.rejection_reasons
                    funnel.sample_memorization_rejected += sum(
                        "sample_text_copy" in reasons
                        for reasons in rejection_reasons
                    )
                    if raw_count and 0 < len(parsed_candidates) < raw_count:
                        funnel.student_partially_valid_responses += 1
                    if not parsed_candidates:
                        failure_class = "zero_valid_student_candidates"
                except (TypeError, ValueError) as exc:
                    failure_class = "schema_error"
                    parse_error = str(exc)
            if failure_class:
                funnel.student_invalid_responses += 1
                if truncated:
                    funnel.student_truncated_responses += 1
            student_round = {
                "update_index": update_index,
                "target_agent_id": target_agent_id,
                "role": "student",
                "context_type": type(context).__name__,
                "context_hash": context_hash,
                "request_hash": _request_hash("Return strict JSON only.", student_request),
                "response_hash": hashlib.sha256(student_raw.encode("utf-8")).hexdigest(),
                "json_extracted": parsed is not None,
                "schema_valid": bool(parsed is not None and failure_class in {"", "zero_valid_student_candidates"}),
                "requested_count": self.cfg.tcs.num_candidates_per_parent,
                "raw_count": raw_count,
                "valid_count": len(parsed_candidates),
                "per_candidate_rejection_reasons": [
                    list(row) for row in rejection_reasons
                ],
                "semantic_round": semantic_round,
                "format_attempt": format_attempt,
                "finish_reason": student_result.finish_reason,
                "hit_completion_limit": student_result.hit_completion_limit,
                "response_truncated": truncated,
                "failure_class": failure_class,
                "retry_reason": failure_class if failure_class and format_attempt == 0 else "",
                "parse_error": parse_error,
                "response_excerpt": _response_excerpt(student_raw),
                "input_characters": len(student_request),
                "output_characters": len(student_raw),
            }
            self.tcs_rounds.append(student_round)
            funnel.raw_candidate_count = raw_count
            if parsed_candidates:
                break
        funnel.schema_valid_count = len(parsed_candidates)
        unique: dict[str, CandidateRuntime] = {}
        non_parent = 0
        repair_plan_hash = hashlib.sha256(
            json.dumps(
                asdict(repair_plan),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
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

    def _limits(self, size: int) -> ConstraintLimits:
        return ConstraintLimits(
            local_accuracy_allowance=int(self.cfg.constraints.local_accuracy_loss_epsilon * size),
            global_accuracy_allowance=int(self.cfg.constraints.global_accuracy_loss_epsilon * size),
            invalid_allowance=int(self.cfg.constraints.invalid_guard_epsilon * size),
            vote_loss_limit=self.cfg.constraints.vote_loss_limit,
            unique_correct_loss_limit=self.cfg.constraints.unique_correct_loss_limit,
            pivotal_loss_limit=self.cfg.constraints.pivotal_loss_limit,
        )

    @staticmethod
    def _competence_only_constraint(
        candidate: CandidateEvaluation,
        active: CandidateEvaluation,
        initial: CandidateEvaluation,
        limits: ConstraintLimits,
    ) -> ConstraintDecision:
        local = candidate.competence.correct_count >= active.competence.correct_count - limits.local_accuracy_allowance
        global_ = candidate.competence.correct_count >= initial.competence.correct_count - limits.global_accuracy_allowance
        invalid = candidate.competence.invalid_count <= active.competence.invalid_count + limits.invalid_allowance
        reasons = tuple(
            name for name, passed in (
                ("local_accuracy", local), ("initial_accuracy", global_), ("invalid", invalid),
            ) if not passed
        )
        return ConstraintDecision(
            passed=not reasons,
            local_accuracy_passed=local,
            initial_accuracy_passed=global_,
            invalid_passed=invalid,
            vote_loss_passed=True,
            unique_correct_passed=True,
            pivotal_correct_passed=True,
            rejection_reasons=reasons,
        )

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
        initial = evaluate_candidate_profile(
            prompt=self.agents[target_agent_id].initial_prompt,
            prompt_hash=self.prompt_hash(self.agents[target_agent_id].initial_prompt),
            examples=self.fixed_probe.examples,
            active_profiles=self.initial_profiles,
            initial_profiles=self.initial_profiles,
            candidate_profile=self.initial_profiles[target_agent_id],
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

        limits = self._limits(len(self.fixed_probe.examples))
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
            candidate.constraint = (
                self._competence_only_constraint(candidate.final_evaluation, incumbent, initial, limits)
                if self.protocol.candidate_selection_policy == "individual_accuracy"
                else evaluate_constraints(candidate.final_evaluation, incumbent, initial, limits)
            )
            if candidate.constraint.passed:
                feasible.append(candidate)
            for reason in candidate.constraint.rejection_reasons:
                field = {
                    "local_accuracy": "rejected_local_accuracy",
                    "initial_accuracy": "rejected_initial_accuracy",
                    "invalid": "rejected_invalid",
                    "vote_loss": "rejected_vote_loss",
                    "unique_correct": "rejected_unique_loss",
                    "pivotal_correct": "rejected_pivotal_loss",
                }[reason]
                setattr(funnel, field, getattr(funnel, field) + 1)
        funnel.stage_b_evaluated = len(shortlist)
        funnel.constraint_feasible = len(feasible)

        if self.protocol.candidate_selection_policy == "individual_accuracy":
            acceptable = [
                row for row in feasible
                if individual_accuracy_key(row.final_evaluation, row.generation)
                > individual_accuracy_key(incumbent, 0)
                and row.final_evaluation.competence.correct_count > incumbent.competence.correct_count
            ]
            accepted = max(
                acceptable,
                key=lambda row: individual_accuracy_key(row.final_evaluation, row.generation),
                default=None,
            )
        elif self.protocol.candidate_selection_policy == "vote_first":
            acceptable = [
                row for row in feasible
                if vote_first_key(row.final_evaluation, row.generation)
                > vote_first_key(incumbent, 0)
            ]
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

    async def update_once(self, update_index: int) -> bool:
        if not self.protocol.optimization_enabled:
            return False
        if self.protocol.target_selection_policy == "member_aware_responsibility":
            if self.protocol.responsibility_refresh_policy != "online":
                raise AssertionError("dynamic responsibility protocol requires online refresh")
            owners, assigned = self.ensure_responsibility_current()
        else:
            owners = {}
            assigned = {agent_id: [] for agent_id in range(5)}
            states, _, _ = self.current_states_and_opportunities()
            self.peer_state_history.extend(asdict(state) for state in states)
        target, fairness_triggered, target_priorities_payload = self.select_target(
            assigned,
            update_index,
        )
        self.agent_selection_counts[target] += 1
        assigned_hashes = {question_hash for question_hash, owner in owners.items() if owner == target}
        funnel = CandidateFunnel()
        candidates = await self.propose_candidates(
            target, assigned_hashes, funnel, update_index=update_index,
        )
        accepted, incumbent, evaluated = await self.evaluate_candidates(
            target, candidates, assigned_hashes, funnel,
        )
        for agent_id in self.responsibility_state.updates_since_selected_by_agent:
            self.responsibility_state.updates_since_selected_by_agent[agent_id] += 1
        self.responsibility_state.updates_since_selected_by_agent[target] = 0
        decision = {
            "update_index": update_index,
            "target_agent_id": target,
            "agent_selection_distribution": dict(self.agent_selection_counts),
            "assigned_question_hashes": sorted(assigned_hashes),
            "max_wait_fairness_trigger_count": int(fairness_triggered),
            "agent_target_priorities": target_priorities_payload,
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
                }
                for row in evaluated
            ],
        }
        self.candidate_decisions.append(decision)
        if accepted is None:
            rejection_reasons = sorted({
                reason
                for row in evaluated
                if row.constraint is not None
                for reason in row.constraint.rejection_reasons
            })
            if not rejection_reasons:
                rejection_reasons = ["no_acceptable_candidate"]
            self.previous_update_outcomes[target] = PreviousUpdateOutcome(
                attempted=True,
                accepted=False,
                rejection_reasons=tuple(rejection_reasons),
            )
            return False

        agent = self.agents[target]
        old_prompt = agent.current_prompt
        old_previous_prompt = agent.previous_active_prompt
        old_profile = self.active_profiles[target]
        old_responsibility_state = deepcopy(self.responsibility_state)
        old_cached_owners = deepcopy(self.cached_responsibility_owners)
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
            self.cached_responsibility_owners = old_cached_owners
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
        vote_correct = invalid = c0 = tie_count = 0
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
        )

    def active_probe_metrics(self) -> DatasetMetrics:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        return self._dataset_metrics_from_profiles(
            self.fixed_probe.examples,
            self.active_profiles,
        )

    async def evaluate_dataset(
        self,
        data: Sequence[Mapping[str, Any]],
        *,
        validation: bool = False,
    ) -> DatasetMetrics:
        examples = self._probe_examples(data)
        if validation:
            if self.validation_probe is None:
                self.validation_probe = self.build_validation_probe(data)
            if tuple(row.question_hash for row in self.validation_probe.examples) != tuple(row.question_hash for row in examples):
                raise ValueError("validation dataset changed after validation cache initialization")
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

        profiles = list(await asyncio.gather(*(
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
        return self._dataset_metrics_from_profiles(examples, profiles)

    def validation_key(
        self,
        metrics: DatasetMetrics,
        initial: DatasetMetrics,
        epoch: int,
    ) -> tuple | None:
        if (
            len(initial.per_agent_correct_counts) != 5
            or len(metrics.per_agent_correct_counts) != 5
        ):
            raise ValueError("validation metrics must contain five agent accuracies")
        size = max(1, len(initial.rows))
        allowance = int(self.cfg.constraints.validation_accuracy_epsilon * size)
        if any(
            current < baseline - allowance
            for current, baseline in zip(
                metrics.per_agent_correct_counts,
                initial.per_agent_correct_counts,
                strict=True,
            )
        ):
            return None
        if metrics.mean_invalid_rate > initial.mean_invalid_rate + self.cfg.constraints.invalid_guard_epsilon:
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
            min(gains),
            metrics.vote_correct_count,
            sum(gains),
            sum(value > 0 for value in gains),
            metrics.mean_soft_vote_utility,
            -metrics.c0_count,
            -metrics.mean_invalid_rate,
            -int(epoch),
        )

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
            "responsibility_version": "five_axis_member_need_pareto_v2",
            "responsibility_lifecycle_version": "one_refresh_per_team_state_v1",
            "target_selection_version": "five_axis_overdue_member_pareto_v2",
            "pareto_preference_version": "member_first_candidate_preference_v1",
            "stage_a_version": "team_vote_worst_mean_v2",
            "stage_b_version": "competence_guard_member_pareto_v2",
            "validation_selection_version": "initial_member_feasible_v1",
            "tcs_context_version": "aggregated_diagnosis_context_v1",
            "diagnosis_aggregation_version": DIAGNOSIS_AGGREGATION_VERSION,
            "answer_role_encoding_version": ANSWER_ROLE_ENCODING_VERSION,
            "pattern_selection_version": PATTERN_SELECTION_VERSION,
            "teacher_schema_version": TEACHER_SCHEMA_VERSION,
            "critic_schema_version": CRITIC_SCHEMA_VERSION,
            "student_schema_version": STUDENT_SCHEMA_VERSION,
            "role_retry_policy_version": ROLE_RETRY_POLICY_VERSION,
            "role_token_budgets": {
                "teacher": self.cfg.tcs.teacher_max_tokens,
                "critic": self.cfg.tcs.critic_max_tokens,
                "student": self.cfg.tcs.student_max_tokens,
            },
            "max_pattern_count": self.cfg.tcs.tcs_max_pattern_summaries,
            "max_evidence_case_count": self.cfg.tcs.tcs_max_evidence_cases,
            "candidate_prompt_length_limit": self.cfg.tcs.candidate_prompt_max_chars,
            "checkpoint_version": 6,
            "tcs_protocol_version": TCS_PROTOCOL_VERSION,
            "critic_approval_basis": "failed_checks_empty",
            "task_general_scope": "unseen_examples_within_current_task",
            "student_sample_memorization_filter": SAMPLE_MEMORIZATION_FILTER_VERSION,
            "solver_sampling_semantics": "shared_prompt_question_output",
            "solver_output_contract_version": self.cfg.peer_state.solver_output_contract_version,
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
            "validation_probe_hash": self.validation_probe.probe_hash if self.validation_probe else "",
            "config": self.cfg.to_flat_dict(),
        }

    def cost_summary(self) -> dict[str, Any]:
        return self.llm.cost_summary()

    def flush_artifacts(self) -> None:
        self.artifacts.write_json("run_meta.json", self.run_meta())
        self.artifacts.write_json("history.json", self.history)
        self.artifacts.write_json("best_prompts.json", [agent.current_prompt for agent in self.agents])
        self.artifacts.write_jsonl("peer_state_history.jsonl", self.peer_state_history)
        self.artifacts.write_jsonl("responsibility_assignments.jsonl", self.responsibility_assignments)
        self.artifacts.write_jsonl("target_priority_audit.jsonl", self.target_priority_audit)
        self.artifacts.write_jsonl("candidate_decisions.jsonl", self.candidate_decisions)
        self.artifacts.write_jsonl("tcs_context_history.jsonl", self.tcs_context_history)
        self.artifacts.write_jsonl("tcs_rounds.jsonl", self.tcs_rounds)
        self.artifacts.write_jsonl("solver_invalid_outputs.jsonl", self.solver_invalid_outputs)
        self.artifacts.write_jsonl("llm_calls.jsonl", self.llm.calls)
        self.artifacts.write_json("cost_summary.json", self.cost_summary())
