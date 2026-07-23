from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .candidate_selection import (
    CandidateEvaluation,
    ConstraintDecision,
    ConstraintLimits,
    StageASelectionDecision,
    candidate_is_acceptable,
    evaluate_constraints,
    individual_accuracy_key,
    stage_a_multichannel_shortlist,
    vote_first_key,
)
from .config import Config
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
from .llm_client import RoleAwareLLMClient
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
    OracleRepairOpportunity,
    ResponsibilityState,
    assign_primary_responsibilities,
    compute_oracle_repair_opportunity,
    select_target_agent,
)
from .tasks import get_task_spec
from .tcs import (
    AccuracyCase,
    AccuracyProposalContext,
    AnyProposalContext,
    CriticDecision,
    PeerStateCase,
    PeerStateProposalContext,
    PreservationCase,
    RepresentativeCase,
    ResponsibilityCase,
    ResponsibilityProposalContext,
    StudentCandidate,
    TCSContextDiagnostics,
    TCSContextLimits,
    TeacherProposal,
    build_critic_request,
    build_student_request,
    build_teacher_request,
    limit_proposal_context,
    parse_critic_decision,
    parse_student_candidates,
    parse_teacher_proposal,
)
from .utils import extract_json_obj, normalize_prompt_text, normalize_spaces


METHOD_VERSION = "peer_state_counterfactual_v1"


@dataclass
class AgentRuntime:
    initial_prompt: str
    current_prompt: str
    previous_active_prompt: str | None = None


@dataclass
class CandidateRuntime:
    student_candidate: StudentCandidate
    prompt: str
    prompt_hash: str
    generation: int
    parent_prompt_hash: str
    stage_a_evaluation: CandidateEvaluation | None = None
    final_evaluation: CandidateEvaluation | None = None
    profile: tuple[PromptAnswer, ...] | None = None
    stage_a_decision: StageASelectionDecision | None = None
    constraint: ConstraintDecision | None = None


@dataclass
class CandidateFunnel:
    parents_considered: int = 0
    teacher_calls: int = 0
    critic_calls: int = 0
    critic_approved: int = 0
    student_calls: int = 0
    requested_candidate_count: int = 0
    raw_candidate_count: int = 0
    schema_valid_count: int = 0
    non_parent_count: int = 0
    deduplicated_count: int = 0
    stage_a_requested_size_per_pool: dict[str, int] = field(default_factory=dict)
    stage_a_available_size_per_pool: dict[str, int] = field(default_factory=dict)
    stage_a_selected_size_per_pool: dict[str, int] = field(default_factory=dict)
    stage_a_overlap_removed: int = 0
    actual_stage_a_size: int = 0
    stage_a_evaluated: int = 0
    selected_by_accuracy_channel: int = 0
    selected_by_vote_channel: int = 0
    selected_by_responsibility_channel: int = 0
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
            raise ValueError("peer_state_counterfactual_v1 requires exactly five agents")
        if cfg.peer_state.aggregation_mode != "plurality":
            raise ValueError("peer_state_counterfactual_v1 requires plurality aggregation")
        if cfg.peer_state.vote_tie_break != "abstain":
            raise ValueError("canonical peer_state_counterfactual_v1 requires tie-as-abstain")
        if cfg.peer_state.solver_output_contract_version != SOLVER_OUTPUT_CONTRACT_VERSION:
            raise ValueError(
                "solver_output_contract_version does not match the implemented task contract"
            )
        self.cfg = cfg
        self.protocol = self._build_protocol()
        self.task_spec = get_task_spec(cfg.data.task_type)
        prompts = self._initial_prompts()
        self.agents = [AgentRuntime(prompt, prompt) for prompt in prompts]
        self.responsibility_state = ResponsibilityState(
            assigned_load_by_agent={agent_id: 0 for agent_id in range(cfg.training.agents)},
            updates_since_selected_by_agent={agent_id: 0 for agent_id in range(cfg.training.agents)},
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
        self.cached_responsibility_assignments: dict[int, list[OracleRepairOpportunity]] = {}
        self.previous_accuracy_summaries = {
            agent_id: "No prior accepted update; individual accuracy and invalid rate are unchanged."
            for agent_id in range(5)
        }
        self.previous_peer_summaries = {
            agent_id: "No prior accepted update; vote and competence metrics are unchanged."
            for agent_id in range(5)
        }
        self.previous_responsibility_summaries = {
            agent_id: "No prior accepted update on assigned residual cases."
            for agent_id in range(5)
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
    ) -> str:
        return await self.llm.chat(
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
                self.cfg.models.max_tokens,
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

    def current_states_and_opportunities(
        self,
    ) -> tuple[
        tuple[TeamVoteState, ...],
        dict[str, dict[int, PeerVoteContext]],
        dict[str, tuple[OracleRepairOpportunity, ...]],
    ]:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        states: list[TeamVoteState] = []
        contexts: dict[str, dict[int, PeerVoteContext]] = {}
        opportunities: dict[str, tuple[OracleRepairOpportunity, ...]] = {}
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
            opportunities[state.question_hash] = tuple(
                compute_oracle_repair_opportunity(
                    team_state=state,
                    peer_context=peer_by_agent[agent_id],
                    tau=self.cfg.peer_state.soft_vote_tau,
                )
                for agent_id in range(5)
            )
        return tuple(states), contexts, opportunities

    def assign_responsibilities(
        self,
    ) -> tuple[dict[str, int], dict[int, list[OracleRepairOpportunity]]]:
        states, contexts, opportunities = self.current_states_and_opportunities()
        state_by_hash = {state.question_hash: state for state in states}
        old_owners = dict(self.responsibility_state.primary_owner_by_question)
        owners, assigned = assign_primary_responsibilities(
            team_states=state_by_hash,
            peer_contexts=contexts,
            opportunities=opportunities,
            state=self.responsibility_state,
            switch_margin=self.cfg.responsibility.responsibility_switch_margin,
        )
        owner_switch_count = sum(
            question_hash in old_owners and old_owners[question_hash] != owner
            for question_hash, owner in owners.items()
        )
        rows = [row for values in assigned.values() for row in values]
        self.peer_state_history.extend(asdict(state) for state in states)
        self.responsibility_assignments.append({
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
            "assigned_opportunities": {
                str(agent_id): [asdict(row) for row in values] for agent_id, values in assigned.items()
            },
        })
        return owners, assigned

    def select_target(
        self,
        assigned: Mapping[int, Sequence[OracleRepairOpportunity]],
        update_index: int,
    ) -> tuple[int, bool]:
        if self.protocol.target_selection_policy == "round_robin":
            return update_index % 5, False
        if self.protocol.target_selection_policy != "dynamic_residual_responsibility":
            raise ValueError(f"Protocol has no optimization target selector: {self.protocol.name}")
        max_wait = self.cfg.responsibility.responsibility_max_wait_updates
        fairness = any(value >= max_wait for value in self.responsibility_state.updates_since_selected_by_agent.values())
        return select_target_agent(assigned, self.responsibility_state, max_wait), fairness

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
            assigned_mode = self.protocol.sample_pool_policy == "assigned_residuals"
            if self.protocol.sample_pool_policy not in {"global_peer_state", "assigned_residuals"}:
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
    ) -> tuple[AnyProposalContext, TCSContextDiagnostics]:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        states, contexts, opportunities = self.current_states_and_opportunities()
        pools = self._pool_indices(target_agent_id, assigned_hashes)
        coverage_set = {self.fixed_probe.examples[index].question_hash for index in pools.coverage}
        conversion_set = {self.fixed_probe.examples[index].question_hash for index in pools.conversion}
        preservation_set = {self.fixed_probe.examples[index].question_hash for index in pools.preservation}
        representative_set = {self.fixed_probe.examples[index].question_hash for index in pools.representative}
        examples = {row.question_hash: row for row in self.fixed_probe.examples}
        accuracy_errors: list[AccuracyCase] = []
        accuracy_protection: list[AccuracyCase] = []
        peer_coverage_cases: list[PeerStateCase] = []
        peer_conversion_cases: list[PeerStateCase] = []
        assigned_coverage_cases: list[ResponsibilityCase] = []
        assigned_conversion_cases: list[ResponsibilityCase] = []
        preservation_cases: list[PreservationCase] = []
        representative_cases: list[RepresentativeCase] = []
        for state in states:
            opportunity = opportunities[state.question_hash][target_agent_id]
            peer = contexts[state.question_hash][target_agent_id]
            example = examples[state.question_hash]
            accuracy_case = AccuracyCase(
                question_hash=state.question_hash,
                question=example.question,
                gold_answer=example.gold_answer,
                target_current_answer=state.team_answers[target_agent_id],
                target_current_correct=opportunity.current_correct,
                target_current_invalid=opportunity.current_invalid,
            )
            peer_case = PeerStateCase(
                question_hash=state.question_hash,
                question=example.question,
                gold_answer=example.gold_answer,
                target_current_answer=state.team_answers[target_agent_id],
                team_G=state.gold_vote_count,
                team_H=state.largest_wrong_vote_count,
                team_M=state.plurality_margin,
                team_wrong_histogram=state.wrong_vote_histogram,
                peer_G=peer.peer_gold_vote_count,
                peer_H=peer.peer_largest_wrong_vote_count,
                peer_M=peer.peer_margin,
                peer_wrong_histogram=peer.peer_wrong_vote_histogram,
                direct_vote_fix=opportunity.direct_vote_fix,
                oracle_soft_utility_gain=opportunity.oracle_soft_utility_gain,
                dominant_wrong_member=opportunity.dominant_wrong_member,
            )
            if state.question_hash in coverage_set:
                peer_coverage_cases.append(peer_case)
                assigned_coverage_cases.append(ResponsibilityCase(
                    **asdict(peer_case),
                    responsibility_reason="assigned residual owner",
                    owner_age=self.responsibility_state.owner_age_by_question.get(state.question_hash, 0),
                ))
            if state.question_hash in conversion_set:
                peer_conversion_cases.append(peer_case)
                assigned_conversion_cases.append(ResponsibilityCase(
                    **asdict(peer_case),
                    responsibility_reason="assigned residual owner",
                    owner_age=self.responsibility_state.owner_age_by_question.get(state.question_hash, 0),
                ))
            if state.question_hash in preservation_set:
                accuracy_protection.append(accuracy_case)
                preservation_cases.append(PreservationCase(
                    question_hash=state.question_hash,
                    question=example.question,
                    gold_answer=example.gold_answer,
                    target_current_answer=state.team_answers[target_agent_id],
                    unique_correct=opportunity.unique_correct,
                    pivotal_correct=opportunity.pivotal_correct,
                    team_margin=state.plurality_margin,
                    peer_G=peer.peer_gold_vote_count,
                    peer_H=peer.peer_largest_wrong_vote_count,
                    peer_M=peer.peer_margin,
                    peer_wrong_histogram=peer.peer_wrong_vote_histogram,
                ))
            if state.question_hash in representative_set:
                if not opportunity.current_correct:
                    accuracy_errors.append(accuracy_case)
                representative_cases.append(RepresentativeCase(
                    question_hash=state.question_hash,
                    question=example.question,
                    gold_answer=example.gold_answer,
                    target_current_answer=state.team_answers[target_agent_id],
                    target_current_correct=opportunity.current_correct,
                    target_current_invalid=opportunity.current_invalid,
                ))
        common = {
            "target_agent_id": target_agent_id,
            "parent_prompt": parent_prompt,
            "parent_prompt_hash": self.prompt_hash(parent_prompt),
        }
        if self.protocol.tcs_context_policy == "generic_accuracy":
            context: AnyProposalContext = AccuracyProposalContext(
                **common,
                error_cases=tuple(accuracy_errors),
                protection_cases=tuple(accuracy_protection),
                previous_accuracy_summary=self.previous_accuracy_summaries[target_agent_id],
            )
        elif self.protocol.tcs_context_policy == "generic_peer_state":
            context = PeerStateProposalContext(
                **common,
                coverage_cases=tuple(peer_coverage_cases),
                conversion_cases=tuple(peer_conversion_cases),
                preservation_cases=tuple(preservation_cases),
                representative_cases=tuple(representative_cases),
                previous_vote_competence_summary=self.previous_peer_summaries[target_agent_id],
            )
        elif self.protocol.tcs_context_policy == "responsibility_conditioned":
            context = ResponsibilityProposalContext(
                **common,
                assigned_coverage_cases=tuple(assigned_coverage_cases),
                assigned_conversion_cases=tuple(assigned_conversion_cases),
                preservation_cases=tuple(preservation_cases),
                representative_cases=tuple(representative_cases),
                responsibility_summary=(
                    f"Agent {target_agent_id} owns {len(assigned_hashes)} current residual cases."
                ),
                assigned_repair_history=self.previous_responsibility_summaries[target_agent_id],
            )
        else:
            raise ValueError(f"Unsupported TCS context policy: {self.protocol.tcs_context_policy}")
        return limit_proposal_context(context, TCSContextLimits(
            assigned_coverage=self.cfg.tcs.tcs_assigned_coverage_limit,
            assigned_conversion=self.cfg.tcs.tcs_assigned_conversion_limit,
            preservation=self.cfg.tcs.tcs_preservation_limit,
            representative=self.cfg.tcs.tcs_representative_limit,
            max_chars=self.cfg.tcs.tcs_context_max_chars,
        ))

    async def propose_candidates(
        self,
        target_agent_id: int,
        assigned_hashes: set[str],
        funnel: CandidateFunnel,
        update_index: int = -1,
    ) -> list[CandidateRuntime]:
        parent_prompt = self.agents[target_agent_id].current_prompt
        context, diagnostics = self._proposal_context(target_agent_id, parent_prompt, assigned_hashes)
        context_payload = json.dumps(asdict(context), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        context_object = asdict(context)
        field_paths = sorted(_recursive_field_paths(context_object))
        if isinstance(context, AccuracyProposalContext):
            forbidden_tokens = (
                "team_g", "team_h", "team_m", "peer_g", "peer_h", "peer_m",
                "wrong_histogram", "responsibility", "owner", "assigned", "vote_delta",
            )
        elif isinstance(context, PeerStateProposalContext):
            forbidden_tokens = (
                "assigned", "owner", "owner_age", "responsibility", "responsibility_reason",
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
            "proposal_context_hash": hashlib.sha256(context_payload.encode("utf-8")).hexdigest(),
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
            **asdict(diagnostics),
        })
        funnel.parents_considered = 1
        teacher_request = build_teacher_request(context)
        teacher_proposal: TeacherProposal | None = None
        critic_decision: CriticDecision | None = None
        critic_feedback = ""
        for round_index in range(self.cfg.tcs.teacher_critic_max_rounds):
            funnel.teacher_calls += 1
            user_request = (
                "Produce the repair proposal."
                if round_index == 0
                else f"Revise the proposal using this critic feedback: {critic_feedback}"
            )
            teacher_request_hash = _request_hash(teacher_request, user_request)
            teacher_raw = await self._chat(
                self.cfg.models.optimizer_model,
                teacher_request,
                user_request,
                self.cfg.tcs.teacher_temperature,
                self.cfg.tcs.teacher_max_tokens,
                "optimizer",
            )
            parsed_teacher = extract_json_obj(teacher_raw)
            teacher_round = {
                "update_index": update_index,
                "target_agent_id": target_agent_id,
                "round_index": round_index,
                "role": "teacher",
                "context_type": type(context).__name__,
                "context_hash": hashlib.sha256(context_payload.encode("utf-8")).hexdigest(),
                "request_hash": teacher_request_hash,
                "response_hash": hashlib.sha256(teacher_raw.encode("utf-8")).hexdigest(),
                "json_extracted": parsed_teacher is not None,
                "schema_valid": False,
                "parse_error": "",
                "response_excerpt": _response_excerpt(teacher_raw),
                "revision_round": round_index,
            }
            try:
                if parsed_teacher is None:
                    raise ValueError("teacher response is not JSON")
                teacher_proposal = parse_teacher_proposal(parsed_teacher)
            except (KeyError, TypeError, ValueError) as exc:
                teacher_round["parse_error"] = str(exc)
                self.tcs_rounds.append(teacher_round)
                critic_feedback = f"Teacher schema failure: {exc}"
                continue
            teacher_round.update({
                "schema_valid": True,
                "target_failure_mechanism": teacher_proposal.target_failure_mechanism,
                "repair_procedure": teacher_proposal.repair_procedure,
                "preservation_rule": teacher_proposal.preservation_rule,
                "expected_effect": teacher_proposal.expected_effect,
            })
            self.tcs_rounds.append(teacher_round)
            funnel.critic_calls += 1
            critic_request = build_critic_request(context, teacher_proposal)
            critic_request_hash = _request_hash(critic_request, "Audit the proposal.")
            critic_raw = await self._chat(
                self.cfg.models.evaluator_model,
                critic_request,
                "Audit the proposal.",
                self.cfg.tcs.critic_temperature,
                self.cfg.tcs.critic_max_tokens,
                "evaluator",
            )
            parsed_critic = extract_json_obj(critic_raw)
            critic_round = {
                "update_index": update_index,
                "target_agent_id": target_agent_id,
                "round_index": round_index,
                "role": "critic",
                "context_type": type(context).__name__,
                "context_hash": hashlib.sha256(context_payload.encode("utf-8")).hexdigest(),
                "request_hash": critic_request_hash,
                "response_hash": hashlib.sha256(critic_raw.encode("utf-8")).hexdigest(),
                "json_extracted": parsed_critic is not None,
                "schema_valid": False,
                "approved_raw": (
                    parsed_critic.get("approved")
                    if isinstance(parsed_critic, Mapping)
                    and isinstance(parsed_critic.get("approved"), bool)
                    else None
                ),
                "score": (
                    parsed_critic.get("score")
                    if isinstance(parsed_critic, Mapping)
                    and isinstance(parsed_critic.get("score"), (int, float))
                    else None
                ),
                "effective_approved": False,
                "rejection_reasons": [],
                "parse_error": "",
                "response_excerpt": _response_excerpt(critic_raw),
            }
            try:
                if parsed_critic is None:
                    raise ValueError("critic response is not JSON")
                critic_decision = parse_critic_decision(
                    parsed_critic,
                    self.cfg.tcs.critic_approval_threshold,
                )
            except (KeyError, TypeError, ValueError) as exc:
                critic_round["parse_error"] = str(exc)
                self.tcs_rounds.append(critic_round)
                critic_feedback = f"Critic schema failure: {exc}"
                continue
            critic_round.update({
                "schema_valid": True,
                "effective_approved": critic_decision.approved,
                "score": critic_decision.score,
                "rejection_reasons": list(critic_decision.rejection_reasons),
                "feedback": critic_decision.feedback,
            })
            self.tcs_rounds.append(critic_round)
            if critic_decision.approved:
                funnel.critic_approved += 1
                break
            critic_feedback = critic_decision.feedback
        if teacher_proposal is None or critic_decision is None or not critic_decision.approved:
            return []

        parsed_candidates: tuple[StudentCandidate, ...] = ()
        funnel.requested_candidate_count = self.cfg.tcs.num_candidates_per_parent
        for student_attempt in range(self.cfg.tcs.student_json_max_retries + 1):
            funnel.student_calls += 1
            student_request = build_student_request(
                context, teacher_proposal, self.cfg.tcs.num_candidates_per_parent,
            )
            student_raw = await self._chat(
                self.cfg.models.optimizer_model,
                "Return strict JSON only.",
                student_request,
                self.cfg.tcs.student_temperature,
                self.cfg.tcs.student_max_tokens,
                "optimizer",
            )
            parsed = extract_json_obj(student_raw)
            candidates_value = parsed.get("candidates") if isinstance(parsed, Mapping) else None
            raw_count = len(candidates_value) if isinstance(candidates_value, list) else 0
            student_round = {
                "update_index": update_index,
                "target_agent_id": target_agent_id,
                "round_index": student_attempt,
                "role": "student",
                "context_type": type(context).__name__,
                "context_hash": hashlib.sha256(context_payload.encode("utf-8")).hexdigest(),
                "request_hash": _request_hash("Return strict JSON only.", student_request),
                "response_hash": hashlib.sha256(student_raw.encode("utf-8")).hexdigest(),
                "json_extracted": parsed is not None,
                "schema_valid": False,
                "requested_count": self.cfg.tcs.num_candidates_per_parent,
                "raw_count": raw_count,
                "schema_valid_count": 0,
                "parse_error": "",
                "response_excerpt": _response_excerpt(student_raw),
            }
            if parsed is None:
                funnel.raw_candidate_count = 0
                student_round["parse_error"] = "student response is not JSON"
                self.tcs_rounds.append(student_round)
                continue
            funnel.raw_candidate_count = raw_count
            try:
                parsed_candidates = parse_student_candidates(
                    parsed,
                    expected_count=self.cfg.tcs.num_candidates_per_parent,
                )
                student_round["schema_valid"] = True
                student_round["schema_valid_count"] = len(parsed_candidates)
                self.tcs_rounds.append(student_round)
                break
            except (KeyError, TypeError, ValueError) as exc:
                student_round["parse_error"] = str(exc)
                self.tcs_rounds.append(student_round)
                continue
        funnel.schema_valid_count = len(parsed_candidates)
        unique: dict[str, CandidateRuntime] = {}
        non_parent = 0
        for candidate in parsed_candidates:
            prompt = normalize_prompt_text(candidate.candidate_prompt)
            prompt_hash = self.prompt_hash(prompt)
            if prompt_hash == self.prompt_hash(parent_prompt):
                continue
            non_parent += 1
            unique.setdefault(prompt_hash, CandidateRuntime(
                student_candidate=candidate,
                prompt=prompt,
                prompt_hash=prompt_hash,
                generation=1,
                parent_prompt_hash=self.prompt_hash(parent_prompt),
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
            min_soft_utility_gain=self.cfg.constraints.min_soft_utility_gain,
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
        stage_a_examples, stage_a_active = subset_profiles(self.fixed_probe.examples, self.active_profiles, indices)
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
                    selected_by_channels=("accuracy",) if candidate in shortlist else (),
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
        funnel.selected_by_accuracy_channel = sum(
            candidate.stage_a_decision.selected
            and "accuracy" in candidate.stage_a_decision.selected_by_channels
            for candidate in candidates
        )
        funnel.selected_by_vote_channel = sum(
            candidate.stage_a_decision.selected
            and "vote" in candidate.stage_a_decision.selected_by_channels
            for candidate in candidates
        )
        funnel.selected_by_responsibility_channel = sum(
            candidate.stage_a_decision.selected
            and "responsibility" in candidate.stage_a_decision.selected_by_channels
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
        else:
            acceptable = [
                row for row in feasible
                if candidate_is_acceptable(row.final_evaluation, incumbent, limits)
            ]
            accepted = max(
                acceptable,
                key=lambda row: vote_first_key(row.final_evaluation, row.generation),
                default=None,
            )
        funnel.acceptable_candidates = len(acceptable)
        funnel.accepted_candidate = accepted is not None
        return accepted, incumbent, list(candidates)

    async def update_once(self, update_index: int) -> bool:
        if not self.protocol.optimization_enabled:
            return False
        if self.protocol.target_selection_policy == "dynamic_residual_responsibility":
            if self.protocol.responsibility_refresh_policy == "online" or not self.cached_responsibility_owners:
                owners, assigned = self.assign_responsibilities()
                self.cached_responsibility_owners = dict(owners)
                self.cached_responsibility_assignments = {agent_id: list(rows) for agent_id, rows in assigned.items()}
            else:
                raise AssertionError("dynamic responsibility protocol requires online refresh")
        else:
            owners = {}
            assigned = {agent_id: [] for agent_id in range(5)}
            states, _, _ = self.current_states_and_opportunities()
            self.peer_state_history.extend(asdict(state) for state in states)
        target, fairness_triggered = self.select_target(assigned, update_index)
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
            "funnel": asdict(funnel),
            "accepted_prompt_hash": accepted.prompt_hash if accepted else "",
            "incumbent": asdict(incumbent),
            "candidates": [
                {
                    "prompt_hash": row.prompt_hash,
                    "generation": row.generation,
                    "stage_a_decision": asdict(row.stage_a_decision) if row.stage_a_decision else None,
                    "evaluation": asdict(row.final_evaluation) if row.final_evaluation else None,
                    "constraint": asdict(row.constraint) if row.constraint else None,
                }
                for row in evaluated
            ],
        }
        self.candidate_decisions.append(decision)
        if accepted is None:
            self.previous_accuracy_summaries[target] = (
                "No candidate accepted; individual accuracy and invalid rate are unchanged."
            )
            self.previous_peer_summaries[target] = (
                "No candidate accepted; vote and competence metrics are unchanged."
            )
            self.previous_responsibility_summaries[target] = (
                "No candidate accepted on assigned residual cases."
            )
            return False

        agent = self.agents[target]
        old_prompt = agent.current_prompt
        old_profile = self.active_profiles[target]
        agent.previous_active_prompt = old_prompt
        try:
            agent.current_prompt = accepted.prompt
            if accepted.profile is None:
                raise AssertionError("accepted candidate has no full fixed-probe profile")
            self.active_profiles[target] = accepted.profile
            if self.protocol.responsibility_refresh_policy == "online":
                owners, assigned = self.assign_responsibilities()
                self.cached_responsibility_owners = dict(owners)
                self.cached_responsibility_assignments = {agent_id: list(rows) for agent_id, rows in assigned.items()}
        except Exception:
            agent.current_prompt = old_prompt
            self.active_profiles[target] = old_profile
            raise
        evaluation = accepted.final_evaluation
        competence_delta = evaluation.competence.correct_count - incumbent.competence.correct_count
        invalid_delta = evaluation.competence.invalid_count - incumbent.competence.invalid_count
        self.previous_accuracy_summaries[target] = (
            f"Individual correct-count change={competence_delta}; invalid-count change={invalid_delta}."
        )
        self.previous_peer_summaries[target] = (
            f"Net vote change={evaluation.marginal.net_vote_delta}; vote losses={evaluation.marginal.vote_loss_count}; "
            f"individual correct-count change={competence_delta}; invalid-count change={invalid_delta}."
        )
        self.previous_responsibility_summaries[target] = (
            f"Assigned repairs={evaluation.marginal.assigned_residual_repair_count}; assigned utility change="
            f"{evaluation.marginal.assigned_residual_utility_delta:.6f}; net vote change="
            f"{evaluation.marginal.net_vote_delta}."
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
        if len(initial.per_agent_acc) != 5 or len(metrics.per_agent_acc) != 5:
            raise ValueError("validation metrics must contain five agent accuracies")
        if any(
            current < baseline - self.cfg.constraints.validation_accuracy_epsilon
            for current, baseline in zip(metrics.per_agent_acc, initial.per_agent_acc, strict=True)
        ):
            return None
        if metrics.mean_individual_acc < initial.mean_individual_acc - self.cfg.constraints.validation_mean_epsilon:
            return None
        if metrics.mean_invalid_rate > initial.mean_invalid_rate + self.cfg.constraints.invalid_guard_epsilon:
            return None
        initial_rows = {row.question_hash: row.vote_correct for row in initial.rows}
        current_rows = {row.question_hash: row.vote_correct for row in metrics.rows}
        if set(initial_rows) != set(current_rows):
            raise ValueError("validation question sets differ")
        vote_gain = sum(not initial_rows[key] and current_rows[key] for key in initial_rows)
        vote_loss = sum(initial_rows[key] and not current_rows[key] for key in initial_rows)
        return (
            metrics.plurality_vote_acc,
            vote_gain - vote_loss,
            -vote_loss,
            metrics.mean_soft_vote_utility,
            -metrics.c0_count,
            metrics.mean_individual_acc,
            metrics.min_individual_acc,
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
        self.artifacts.write_jsonl("candidate_decisions.jsonl", self.candidate_decisions)
        self.artifacts.write_jsonl("tcs_context_history.jsonl", self.tcs_context_history)
        self.artifacts.write_jsonl("tcs_rounds.jsonl", self.tcs_rounds)
        self.artifacts.write_jsonl("solver_invalid_outputs.jsonl", self.solver_invalid_outputs)
        self.artifacts.write_jsonl("llm_calls.jsonl", self.llm.calls)
        self.artifacts.write_json("cost_summary.json", self.cost_summary())
