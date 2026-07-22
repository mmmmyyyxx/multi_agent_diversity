from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from openai import AsyncOpenAI

from .candidate_selection import (
    ConstraintLimits, candidate_is_acceptable, candidate_is_feasible,
    stage_a_multichannel_shortlist, vote_first_key,
)
from .config import Config
from .evaluation.fixed_probe import (
    FixedProbeEvaluator, ProbeExample, PromptAnswer, candidate_probe_metrics, subset_profiles,
)
from .peer_state import build_peer_vote_state, soft_vote_utility
from .persistence.artifacts import ArtifactWriter
from .prompt_memory import contribution_signature, rebuild_prompt_memory, select_generation_parents
from .responsibility import (
    AgentExampleCredit, ResponsibilityState, assign_primary_responsibilities,
    counterfactual_credit, select_target_agent,
)
from .tasks import get_task_spec
from .tcs import (
    build_responsibility_context, critic_instruction, critic_rejects_surface_rewrite,
    student_instruction, teacher_instruction, validate_student_candidate,
)
from .utils import extract_json_obj, normalize_spaces


METHOD_VERSION = "peer_state_counterfactual_v1"


@dataclass
class AgentRuntime:
    initial_prompt: str
    current_prompt: str
    prompt_memory: list[dict[str, Any]] = field(default_factory=list)


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
        self.cfg = cfg
        self.task_spec = get_task_spec(cfg.data.task_type)
        prompts = [cfg.training.shared_prompt for _ in range(cfg.training.agents)]
        self.agents = [AgentRuntime(prompt, prompt) for prompt in prompts]
        self.responsibility_state = ResponsibilityState(
            agent_updates_since_last_selected={agent_id: 0 for agent_id in range(cfg.training.agents)},
            assigned_load_per_agent={agent_id: 0 for agent_id in range(cfg.training.agents)},
        )
        self.history: list[dict[str, Any]] = []
        self.peer_state_history: list[dict[str, Any]] = []
        self.responsibility_assignments: list[dict[str, Any]] = []
        self.candidate_decisions: list[dict[str, Any]] = []
        self.prompt_memory_history: list[dict[str, Any]] = []
        self.cached_responsibility_owners: dict[str, int] = {}
        self.cached_responsibility_assignments: dict[int, list[AgentExampleCredit]] = {}
        self.fixed_probe: FixedProbeEvaluator | None = None
        self.active_profiles: list[tuple[PromptAnswer, ...]] = []
        self.initial_profiles: list[tuple[PromptAnswer, ...]] = []
        self.artifacts = ArtifactWriter(cfg.persistence.out_dir)
        self._solver_override = solver
        self._optimizer_override = optimizer_chat
        self.solver_semaphore = asyncio.Semaphore(max(1, cfg.evaluation.eval_solver_call_concurrency))
        self._clients: dict[str, AsyncOpenAI] = {}
        self.llm_calls: list[dict[str, Any]] = []

    @staticmethod
    def prompt_hash(prompt: str) -> str:
        return hashlib.sha256(normalize_spaces(prompt).encode("utf-8")).hexdigest()

    def normalize_answer(self, answer: str) -> str:
        return self.task_spec.extract_pred(f"FINAL_ANSWER: {answer}", None)

    def match_answer(self, prediction: str, gold: str) -> bool:
        return self.task_spec.match_answer(prediction, gold)

    def _client_or_raise(self, role: str) -> AsyncOpenAI:
        if role not in {"solver", "optimizer", "evaluator"}:
            raise ValueError(f"Unknown client role: {role}")
        if role not in self._clients:
            if role == "evaluator":
                key_env = self.cfg.models.evaluator_api_key_env
                base_env = self.cfg.models.evaluator_base_url_env
            elif role == "solver":
                key_env = self.cfg.models.solver_api_key_env
                base_env = self.cfg.models.solver_base_url_env
            else:
                key_env = base_env = ""
            key = os.getenv(key_env) if key_env else os.getenv("OPENAI_API_KEY")
            base = os.getenv(base_env) if base_env else (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
            if not key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._clients[role] = AsyncOpenAI(api_key=key, base_url=base)
        return self._clients[role]

    async def _chat(
        self, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int,
        client_role: str,
    ) -> str:
        if self._optimizer_override is not None:
            return await self._optimizer_override(system_prompt, user_prompt, temperature, max_tokens)
        last_error = None
        for attempt in range(max(1, self.cfg.persistence.max_retries) + self.cfg.persistence.max_transient_retries):
            started = time.time()
            try:
                response = await self._client_or_raise(client_role).chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.cfg.persistence.llm_call_timeout,
                )
                text = response.choices[0].message.content or ""
                self.llm_calls.append({"model": model, "success": True, "latency_seconds": time.time() - started})
                return text
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(min(self.cfg.persistence.max_retry_backoff, self.cfg.persistence.retry_sleep * (2 ** min(attempt, 8))))
        raise RuntimeError(f"LLM call failed: {last_error}")

    async def solve(self, question: str, agent_id: int, prompt: str) -> PromptAnswer:
        if self._solver_override is not None:
            return await self._solver_override(question, agent_id, prompt)
        async with self.solver_semaphore:
            text = await self._chat(
                self.cfg.models.agent_model,
                "Follow the supplied decision procedure. End with exactly one FINAL_ANSWER line.\n\n" + prompt,
                question,
                self.cfg.models.temperature,
                self.cfg.models.max_tokens,
                "solver",
            )
        answer = self.task_spec.extract_pred(text, question)
        valid = bool(answer and "FINAL_ANSWER" in text and len(text.split()) >= 3)
        return PromptAnswer(answer=answer, trace=text, valid=valid)

    def build_probe(self, data: Sequence[Mapping[str, Any]]) -> FixedProbeEvaluator:
        examples = [
            ProbeExample(
                question=str(row["question"]),
                question_hash=hashlib.sha256(str(row["question"]).encode("utf-8")).hexdigest(),
                gold_answer=self.task_spec.parse_gold(row["answer"], str(row["question"])),
            )
            for row in data
        ]
        return FixedProbeEvaluator(examples, self.cfg.peer_state.probe_version)

    async def initialize_fixed_probe(self, data: Sequence[Mapping[str, Any]]) -> None:
        self.fixed_probe = self.build_probe(data)
        self.active_profiles = []
        for agent_id, agent in enumerate(self.agents):
            profile = await self.fixed_probe.evaluate_prompt(
                agent_id, agent.current_prompt, self.prompt_hash(agent.current_prompt), self.solve,
            )
            self.active_profiles.append(profile)
        self.initial_profiles = list(self.active_profiles)
        await self.refresh_memories()

    def current_states_and_credits(self) -> tuple[list[Any], dict[str, list[AgentExampleCredit]]]:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        states = []
        credits: dict[str, list[AgentExampleCredit]] = {}
        for index, example in enumerate(self.fixed_probe.examples):
            answers = [profile[index].answer for profile in self.active_profiles]
            valid = [profile[index].valid for profile in self.active_profiles]
            state = build_peer_vote_state(
                question_hash=example.question_hash, gold_answer=example.gold_answer,
                answers=answers, valid_vector=valid, normalize_answer=self.normalize_answer,
                match_answer=self.match_answer, tie_break=self.cfg.peer_state.vote_tie_break, seed=self.cfg.training.seed,
            )
            states.append(state)
            credits[example.question_hash] = [
                counterfactual_credit(
                    agent_id=agent_id, current_state=state, gold_answer=example.gold_answer,
                    normalize_answer=self.normalize_answer, match_answer=self.match_answer,
                    tie_break=self.cfg.peer_state.vote_tie_break, seed=self.cfg.training.seed, tau=self.cfg.peer_state.soft_vote_tau,
                )
                for agent_id in range(self.cfg.training.agents)
            ]
        return states, credits

    def assign_responsibilities(self) -> tuple[dict[str, int], dict[int, list[AgentExampleCredit]]]:
        states, credits = self.current_states_and_credits()
        self.peer_state_history.extend(asdict(state) for state in states)
        failed = {state.question_hash: credits[state.question_hash] for state in states if not state.vote_correct}
        owners, assigned = assign_primary_responsibilities(
            failed, self.responsibility_state,
            self.cfg.responsibility.responsibility_switch_margin if self.cfg.responsibility.responsibility_inertia_enabled else -1.0,
        )
        self.responsibility_assignments.append({
            "owners": owners,
            "assigned_load_per_agent": dict(self.responsibility_state.assigned_load_per_agent),
            "assigned_credits": {
                str(agent_id): [asdict(credit) for credit in rows]
                for agent_id, rows in assigned.items()
            },
        })
        return owners, assigned

    def stage_a_indices(self, target_agent_id: int, assigned_hashes: set[str]) -> list[int]:
        if self.cfg.training.independent_accuracy_only:
            budget = (
                self.cfg.evaluation.stage_a_representative_size
                + self.cfg.evaluation.stage_a_coverage_size
                + self.cfg.evaluation.stage_a_conversion_size
                + self.cfg.evaluation.stage_a_preservation_size
            )
            return list(range(min(len(self.fixed_probe.examples), budget)))
        states, credits = self.current_states_and_credits()
        assigned_coverage = []
        assigned_conversion = []
        preservation = []
        assigned = []
        for index, state in enumerate(states):
            credit = credits[state.question_hash][target_agent_id]
            if state.question_hash in assigned_hashes:
                assigned.append(index)
                if state.gold_vote_count == 0:
                    assigned_coverage.append(index)
                else:
                    assigned_conversion.append(index)
            if credit.unique_correct or credit.pivotal_vote_correct:
                preservation.append(index)
        reserved = set(assigned_coverage + assigned_conversion + preservation)
        representative = [index for index in range(len(states)) if index not in reserved]
        selected = (
            representative[:self.cfg.evaluation.stage_a_representative_size]
            + assigned_coverage[:self.cfg.evaluation.stage_a_coverage_size]
            + assigned_conversion[:self.cfg.evaluation.stage_a_conversion_size]
            + preservation[:self.cfg.evaluation.stage_a_preservation_size]
        )
        if not selected:
            selected = assigned or list(range(len(states)))
        return list(dict.fromkeys(selected))

    def select_target(self, assigned: Mapping[int, Sequence[AgentExampleCredit]], update_index: int) -> int:
        if self.cfg.training.target_selector == "round_robin":
            return int(update_index) % self.cfg.training.agents
        return select_target_agent(
            assigned, self.responsibility_state.agent_updates_since_last_selected,
            self.cfg.responsibility.responsibility_max_wait_updates,
        )

    def _case_payloads(self, target_agent_id: int, assigned_hashes: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        states, credits = self.current_states_and_credits()
        assigned_cases = []
        preservation = []
        for state in states:
            credit = credits[state.question_hash][target_agent_id]
            example = next(row for row in self.fixed_probe.examples if row.question_hash == state.question_hash)
            peers = [answer for index, answer in enumerate(state.normalized_answers) if index != target_agent_id]
            payload = {
                "question": example.question,
                "gold_answer": example.gold_answer,
                "target_current_answer": state.normalized_answers[target_agent_id],
                "peer_answer_histogram": dict(state.wrong_vote_counts),
                "G": state.gold_vote_count, "H": state.largest_wrong_vote_count, "M": state.plurality_margin,
                "dominant_wrong_answers": list(state.dominant_wrong_answers),
                "direct_vote_fix": credit.direct_vote_fix,
                "fix_soft_utility_gain": credit.fix_soft_utility_gain,
                "peer_gold_vote_count": sum(self.match_answer(answer, example.gold_answer) for answer in peers),
                "unique_correct": credit.unique_correct,
                "pivotal_vote_correct": credit.pivotal_vote_correct,
                "responsibility_reason": "assigned residual" if state.question_hash in assigned_hashes else "preservation",
            }
            if state.question_hash in assigned_hashes:
                assigned_cases.append(payload)
            if credit.unique_correct or credit.pivotal_vote_correct:
                preservation.append(payload)
        return assigned_cases, preservation

    def _ordinary_error_cases(self, target_agent_id: int) -> list[dict[str, Any]]:
        states, credits = self.current_states_and_credits()
        rows = []
        for state in states:
            credit = credits[state.question_hash][target_agent_id]
            if credit.current_correct:
                continue
            example = next(row for row in self.fixed_probe.examples if row.question_hash == state.question_hash)
            payload = {
                "question": example.question,
                "gold_answer": example.gold_answer,
                "target_current_answer": state.normalized_answers[target_agent_id],
                "responsibility_reason": "ordinary target error",
            }
            if not self.cfg.training.independent_accuracy_only:
                payload.update({
                    "peer_answer_histogram": dict(state.wrong_vote_counts),
                    "G": state.gold_vote_count,
                    "H": state.largest_wrong_vote_count,
                    "M": state.plurality_margin,
                    "direct_vote_fix": credit.direct_vote_fix,
                    "fix_soft_utility_gain": credit.fix_soft_utility_gain,
                })
            rows.append(payload)
        return rows

    async def propose_candidates(self, target_agent_id: int, assigned_hashes: set[str]) -> list[dict[str, Any]]:
        parent_rows = select_generation_parents(self.agents[target_agent_id].prompt_memory) or [{
            "prompt": self.agents[target_agent_id].current_prompt,
            "prompt_hash": self.prompt_hash(self.agents[target_agent_id].current_prompt),
            "prompt_memory_slot": "active",
        }]
        parent_rows = parent_rows[:max(1, int(self.cfg.tcs.generation_parent_limit))]
        assigned_cases, preservation = self._case_payloads(target_agent_id, assigned_hashes)
        ordinary_cases = self._ordinary_error_cases(target_agent_id)
        context = build_responsibility_context(
            target_agent_id=target_agent_id,
            assigned_cases=assigned_cases if self.cfg.tcs.responsibility_conditioned_tcs else [],
            preservation_cases=preservation,
            representative_cases=(
                assigned_cases if self.cfg.tcs.responsibility_conditioned_tcs else ordinary_cases
            )[:self.cfg.evaluation.stage_a_representative_size],
        )
        candidates = []
        for parent in parent_rows:
            parent_prompt = str(parent["prompt"])
            teacher_prompt = teacher_instruction(
                context,
                responsibility_conditioned=self.cfg.tcs.responsibility_conditioned_tcs,
                accuracy_only=self.cfg.training.independent_accuracy_only,
            )
            teacher = await self._chat(
                self.cfg.models.optimizer_model, teacher_prompt, "Produce a concise repair proposal.",
                self.cfg.tcs.teacher_temperature, self.cfg.tcs.teacher_max_tokens, "optimizer",
            )
            feedback = teacher
            approved = False
            for critic_round in range(self.cfg.tcs.teacher_critic_max_rounds):
                critic_raw = await self._chat(
                    self.cfg.models.evaluator_model, critic_instruction(), feedback,
                    self.cfg.tcs.critic_temperature, self.cfg.tcs.critic_max_tokens, "evaluator",
                )
                review = extract_json_obj(critic_raw) or {}
                if review.get("approved") and not critic_rejects_surface_rewrite(feedback):
                    approved = True
                    break
                if critic_round + 1 < self.cfg.tcs.teacher_critic_max_rounds:
                    feedback = await self._chat(
                        self.cfg.models.optimizer_model, teacher_prompt,
                        f"Revise using critic feedback: {review.get('feedback', critic_raw)}",
                        self.cfg.tcs.teacher_temperature, self.cfg.tcs.teacher_max_tokens, "optimizer",
                    )
            if not approved:
                continue
            student_raw = ""
            parsed = None
            for _ in range(self.cfg.tcs.student_json_max_retries + 1):
                student_raw = await self._chat(
                    self.cfg.models.optimizer_model,
                    "Return strict JSON only.",
                    student_instruction(parent_prompt, feedback, self.cfg.tcs.num_candidates_per_parent),
                    self.cfg.tcs.student_temperature, self.cfg.tcs.student_max_tokens, "optimizer",
                )
                parsed = extract_json_obj(student_raw)
                if isinstance(parsed, dict) and isinstance(parsed.get("candidates"), list):
                    break
            for proposal in (parsed or {}).get("candidates", []):
                if not isinstance(proposal, dict) or validate_student_candidate(proposal):
                    continue
                prompt = normalize_spaces(str(proposal["candidate_prompt"]))
                candidates.append({
                    "prompt": prompt, "prompt_hash": self.prompt_hash(prompt), "proposal": dict(proposal),
                    "generation": 1, "parent_prompt_hash": str(parent.get("prompt_hash", "")),
                })
        return list({row["prompt_hash"]: row for row in candidates}.values())

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

    def _candidate_feasible(
        self, metrics: Mapping[str, Any], active_metrics: Mapping[str, Any],
        initial_metrics: Mapping[str, Any], limits: ConstraintLimits,
    ) -> bool:
        if self.cfg.training.independent_accuracy_only:
            correct = int(metrics.get("candidate_target_correct_count", 0))
            invalid = int(metrics.get("candidate_invalid_count", 0))
            return bool(
                correct >= int(active_metrics.get("candidate_target_correct_count", 0)) - limits.local_accuracy_allowance
                and correct >= int(initial_metrics.get("candidate_target_correct_count", 0)) - limits.global_accuracy_allowance
                and invalid <= int(active_metrics.get("candidate_invalid_count", 0)) + limits.invalid_allowance
            )
        return candidate_is_feasible(metrics, active_metrics, initial_metrics, limits)

    async def evaluate_candidates(
        self, target_agent_id: int, candidates: Sequence[Mapping[str, Any]], assigned_hashes: set[str],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        if self.fixed_probe is None:
            raise RuntimeError("fixed probe is not initialized")
        active_prompt = self.agents[target_agent_id].current_prompt
        incumbent = {"prompt": active_prompt, "prompt_hash": self.prompt_hash(active_prompt), "generation": 0}
        incumbent_profile = self.active_profiles[target_agent_id]
        incumbent_metrics = candidate_probe_metrics(
            examples=self.fixed_probe.examples, active_profiles=self.active_profiles,
            candidate_profile=incumbent_profile, target_agent_id=target_agent_id,
            assigned_question_hashes=assigned_hashes, normalize_answer=self.normalize_answer,
            match_answer=self.match_answer, tie_break=self.cfg.peer_state.vote_tie_break,
            seed=self.cfg.training.seed, tau=self.cfg.peer_state.soft_vote_tau,
        )
        incumbent = {**incumbent, "profile": incumbent_profile, "metrics": incumbent_metrics}
        initial_metrics = candidate_probe_metrics(
            examples=self.fixed_probe.examples, active_profiles=self.initial_profiles,
            candidate_profile=self.initial_profiles[target_agent_id], target_agent_id=target_agent_id,
            assigned_question_hashes=assigned_hashes, normalize_answer=self.normalize_answer,
            match_answer=self.match_answer, tie_break=self.cfg.peer_state.vote_tie_break,
            seed=self.cfg.training.seed, tau=self.cfg.peer_state.soft_vote_tau,
        )
        limits = self._limits(len(self.fixed_probe.examples))
        stage_a_indices = self.stage_a_indices(target_agent_id, assigned_hashes)
        stage_a_examples, stage_a_active = subset_profiles(
            self.fixed_probe.examples, self.active_profiles, stage_a_indices,
        )
        stage_a_rows = []
        candidate_by_hash = {str(row["prompt_hash"]): dict(row) for row in candidates}
        for row in candidate_by_hash.values():
            partial = await self.fixed_probe.evaluate_prompt_indices(
                target_agent_id, str(row["prompt"]), str(row["prompt_hash"]),
                stage_a_indices, self.solve,
            )
            stage_a_profile = tuple(partial[index] for index in stage_a_indices)
            stage_a_metrics = candidate_probe_metrics(
                examples=stage_a_examples, active_profiles=stage_a_active,
                candidate_profile=stage_a_profile, target_agent_id=target_agent_id,
                assigned_question_hashes=assigned_hashes, normalize_answer=self.normalize_answer,
                match_answer=self.match_answer, tie_break=self.cfg.peer_state.vote_tie_break,
                seed=self.cfg.training.seed, tau=self.cfg.peer_state.soft_vote_tau,
            )
            stage_a_rows.append({**row, "stage_a_metrics": stage_a_metrics, "metrics": stage_a_metrics})
        if self.cfg.training.independent_accuracy_only:
            shortlist = sorted(stage_a_rows, key=lambda row: (
                row["metrics"]["candidate_target_correct_count"],
                -row["metrics"]["candidate_invalid_count"], row["prompt_hash"],
            ), reverse=True)[:self.cfg.evaluation.stage_b_candidate_budget]
        else:
            shortlist = stage_a_multichannel_shortlist(
                stage_a_rows, channel_top_k=self.cfg.evaluation.stage_a_channel_top_k,
                total_budget=self.cfg.evaluation.stage_b_candidate_budget,
            )
        memory_candidates = [
            dict(row) for row in self.agents[target_agent_id].prompt_memory
            if str(row.get("prompt_hash", "")) != incumbent["prompt_hash"]
        ]
        stage_b_by_hash = {str(row["prompt_hash"]): dict(row) for row in [*shortlist, *memory_candidates]}
        stage_b_rows = list(stage_b_by_hash.values())[:self.cfg.evaluation.stage_b_candidate_budget]
        evaluated = [incumbent]
        for row in stage_b_rows:
            profile = await self.fixed_probe.evaluate_prompt(
                target_agent_id, str(row["prompt"]), str(row["prompt_hash"]), self.solve,
            )
            metrics = candidate_probe_metrics(
                examples=self.fixed_probe.examples, active_profiles=self.active_profiles,
                candidate_profile=profile, target_agent_id=target_agent_id,
                assigned_question_hashes=assigned_hashes, normalize_answer=self.normalize_answer,
                match_answer=self.match_answer, tie_break=self.cfg.peer_state.vote_tie_break,
                seed=self.cfg.training.seed, tau=self.cfg.peer_state.soft_vote_tau,
            )
            metrics["constraints_passed"] = self._candidate_feasible(
                metrics, incumbent["metrics"], initial_metrics, limits,
            )
            evaluated.append({**row, "profile": profile, "metrics": metrics})
        incumbent["metrics"]["constraints_passed"] = True
        feasible = [row for row in evaluated[1:] if row["metrics"]["constraints_passed"]]
        if self.cfg.training.independent_accuracy_only:
            accepted = max(feasible, key=lambda row: (
                row["metrics"]["candidate_target_correct_count"],
                -row["metrics"]["candidate_invalid_count"], row["prompt_hash"],
            ), default=None)
            if accepted and accepted["metrics"]["candidate_target_correct_count"] <= incumbent["metrics"]["candidate_target_correct_count"]:
                accepted = None
        else:
            accepted = max(feasible, key=vote_first_key, default=None)
            if accepted and not candidate_is_acceptable(accepted, incumbent, limits):
                accepted = None
        return accepted, evaluated

    async def refresh_memories(self, previous_active: Mapping[int, Mapping[str, Any]] | None = None) -> None:
        previous_active = previous_active or {}
        for agent_id, agent in enumerate(self.agents):
            assigned_hashes = {
                question_hash for question_hash, owner in self.cached_responsibility_owners.items()
                if owner == agent_id
            }
            pool = [
                {"prompt": agent.current_prompt, "prompt_hash": self.prompt_hash(agent.current_prompt)},
                {"prompt": agent.initial_prompt, "prompt_hash": self.prompt_hash(agent.initial_prompt)},
                *agent.prompt_memory,
            ]
            if agent_id in previous_active:
                pool.append(dict(previous_active[agent_id]))
            unique = {str(row.get("prompt_hash", "")): dict(row) for row in pool if row.get("prompt_hash")}
            rows = []
            initial_metrics = None
            active_metrics = None
            for row in unique.values():
                profile = await self.fixed_probe.evaluate_prompt(
                    agent_id, str(row["prompt"]), str(row["prompt_hash"]), self.solve,
                )
                metrics = candidate_probe_metrics(
                    examples=self.fixed_probe.examples, active_profiles=self.active_profiles,
                    candidate_profile=profile, target_agent_id=agent_id,
                    assigned_question_hashes=assigned_hashes, normalize_answer=self.normalize_answer,
                    match_answer=self.match_answer, tie_break=self.cfg.peer_state.vote_tie_break,
                    seed=self.cfg.training.seed, tau=self.cfg.peer_state.soft_vote_tau,
                )
                if str(row["prompt_hash"]) == self.prompt_hash(agent.current_prompt):
                    active_metrics = metrics
                if str(row["prompt_hash"]) == self.prompt_hash(self.agents[agent_id].initial_prompt):
                    initial_metrics = metrics
                rows.append({**row, "profile": profile, "metrics": metrics})
            active_metrics = active_metrics or next(row["metrics"] for row in rows if row["prompt_hash"] == self.prompt_hash(agent.current_prompt))
            initial_metrics = initial_metrics or active_metrics
            limits = self._limits(len(self.fixed_probe.examples))
            for row in rows:
                row["metrics"]["constraints_passed"] = self._candidate_feasible(
                    row["metrics"], active_metrics, initial_metrics, limits,
                )
                if self.cfg.training.independent_accuracy_only:
                    row["metrics"].update({
                        "net_vote_delta": 0,
                        "vote_loss_count": 0,
                        "soft_vote_utility_delta": 0.0,
                        "coverage_gain_count": 0,
                        "assigned_residual_utility_delta": 0.0,
                    })
                row["contribution_signature"] = contribution_signature(
                    fixed_probe_hash=self.fixed_probe.probe_hash,
                    question_hashes=[example.question_hash for example in self.fixed_probe.examples],
                    answer_hashes=row["metrics"]["answer_hashes"],
                    correctness_vector=[profile.valid and self.match_answer(profile.answer, example.gold_answer) for profile, example in zip(row["profile"], self.fixed_probe.examples)],
                    invalid_vector=[not profile.valid for profile in row["profile"]],
                    vote_contribution_vector=row["metrics"]["vote_contribution_vector"],
                    coverage_contribution_vector=row["metrics"]["coverage_contribution_vector"],
                    unique_correct_vector=row["metrics"]["unique_correct_vector"],
                    pivotal_correct_vector=row["metrics"]["pivotal_correct_vector"],
                    dominant_wrong_membership_vector=row["metrics"]["dominant_wrong_membership_vector"],
                )
            memory_rows = [{key: value for key, value in row.items() if key != "profile"} for row in rows]
            previous_row = previous_active.get(agent_id)
            if previous_row is not None:
                previous_hash = str(previous_row.get("prompt_hash", ""))
                previous_row = next(
                    (row for row in memory_rows if str(row.get("prompt_hash", "")) == previous_hash),
                    None,
                )
            agent.prompt_memory = rebuild_prompt_memory(
                memory_rows, active_prompt_hash=self.prompt_hash(agent.current_prompt), previous_active_item=previous_row,
            )
            self.prompt_memory_history.append({
                "agent_id": agent_id,
                "slots": [{"slot": row["prompt_memory_slot"], "prompt_hash": row["prompt_hash"]} for row in agent.prompt_memory],
            })

    async def update_once(self, update_index: int) -> bool:
        if self.cfg.responsibility.responsibility_assignment_enabled:
            should_refresh = self.cfg.training.online_responsibility_refresh or not self.cached_responsibility_owners
            if should_refresh:
                owners, assigned = self.assign_responsibilities()
                self.cached_responsibility_owners = dict(owners)
                self.cached_responsibility_assignments = {
                    int(agent_id): list(rows) for agent_id, rows in assigned.items()
                }
            else:
                owners = dict(self.cached_responsibility_owners)
                assigned = {agent_id: list(rows) for agent_id, rows in self.cached_responsibility_assignments.items()}
        else:
            owners, assigned = {}, {}
            if not self.cfg.training.independent_accuracy_only:
                states, _ = self.current_states_and_credits()
                self.peer_state_history.extend(asdict(state) for state in states)
        target = self.select_target(assigned, update_index)
        assigned_hashes = {question_hash for question_hash, owner in owners.items() if owner == target}
        candidates = await self.propose_candidates(target, assigned_hashes)
        accepted, evaluated = await self.evaluate_candidates(target, candidates, assigned_hashes)
        decision = {
            "update_index": update_index, "target_agent_id": target,
            "candidate_count": len(candidates), "accepted_prompt_hash": accepted["prompt_hash"] if accepted else "",
            "evaluated": [{"prompt_hash": row["prompt_hash"], "metrics": row["metrics"]} for row in evaluated],
        }
        self.candidate_decisions.append(decision)
        for agent_id in self.responsibility_state.agent_updates_since_last_selected:
            self.responsibility_state.agent_updates_since_last_selected[agent_id] += 1
        self.responsibility_state.agent_updates_since_last_selected[target] = 0
        if accepted is None:
            return False
        previous = {
            "prompt": self.agents[target].current_prompt,
            "prompt_hash": self.prompt_hash(self.agents[target].current_prompt),
            "metrics": {"constraints_passed": True},
        }
        self.agents[target].current_prompt = str(accepted["prompt"])
        self.active_profiles[target] = tuple(accepted["profile"])
        self.agents[target].prompt_memory.append(dict(accepted))
        if (
            self.cfg.responsibility.responsibility_assignment_enabled
            and self.cfg.training.online_responsibility_refresh
        ):
            owners, assigned = self.assign_responsibilities()
            self.cached_responsibility_owners = dict(owners)
            self.cached_responsibility_assignments = {
                int(agent_id): list(rows) for agent_id, rows in assigned.items()
            }
        await self.refresh_memories({target: previous})
        return True

    async def evaluate_dataset(self, data: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        correct_per_agent = [0] * self.cfg.training.agents
        vote_correct = invalid = c0 = 0
        utility = 0.0
        rows = []
        for row in data:
            question = str(row["question"])
            gold = self.task_spec.parse_gold(row["answer"], question)
            outputs = await asyncio.gather(*[
                self.solve(question, agent_id, agent.current_prompt) for agent_id, agent in enumerate(self.agents)
            ])
            state = build_peer_vote_state(
                question_hash=hashlib.sha256(question.encode("utf-8")).hexdigest(), gold_answer=gold,
                answers=[output.answer for output in outputs], valid_vector=[output.valid for output in outputs],
                normalize_answer=self.normalize_answer, match_answer=self.match_answer,
                tie_break=self.cfg.peer_state.vote_tie_break, seed=self.cfg.training.seed,
            )
            for agent_id, value in enumerate(state.correctness_vector):
                correct_per_agent[agent_id] += int(value)
            vote_correct += int(state.vote_correct)
            invalid += sum(not value for value in state.valid_vector)
            c0 += int(state.gold_vote_count == 0)
            utility += soft_vote_utility(state.gold_vote_count, state.plurality_margin, self.cfg.peer_state.soft_vote_tau)
            rows.append({"question_hash": state.question_hash, "vote_correct": state.vote_correct, "state": asdict(state)})
        size = max(1, len(data))
        return {
            "plurality_vote_acc": vote_correct / size,
            "vote_acc": vote_correct / size,
            "mean_individual_acc": sum(correct_per_agent) / (size * self.cfg.training.agents),
            "min_individual_acc": min(correct_per_agent, default=0) / size,
            "per_agent_acc": [value / size for value in correct_per_agent],
            "mean_soft_vote_utility": utility / size,
            "c0_count": c0,
            "mean_invalid_rate": invalid / (size * self.cfg.training.agents),
            "rows": rows,
        }

    def validation_key(self, metrics: Mapping[str, Any], initial: Mapping[str, Any], epoch: int) -> tuple | None:
        initial_agents = list(initial.get("per_agent_acc", []))
        current_agents = list(metrics.get("per_agent_acc", []))
        if len(initial_agents) != len(current_agents):
            return None
        if any(
            current < baseline - self.cfg.constraints.validation_accuracy_epsilon
            for current, baseline in zip(current_agents, initial_agents)
        ):
            return None
        if (
            metrics.get("mean_individual_acc", 0.0)
            < initial.get("mean_individual_acc", 0.0) - self.cfg.constraints.validation_mean_epsilon
        ):
            return None
        if (
            metrics.get("mean_invalid_rate", 0.0)
            > initial.get("mean_invalid_rate", 0.0) + self.cfg.constraints.invalid_guard_epsilon
        ):
            return None
        initial_vote = {
            str(row.get("question_hash", "")): bool(row.get("vote_correct", False))
            for row in initial.get("rows", []) if isinstance(row, Mapping)
        }
        current_vote = {
            str(row.get("question_hash", "")): bool(row.get("vote_correct", False))
            for row in metrics.get("rows", []) if isinstance(row, Mapping)
        }
        common = sorted(set(initial_vote) & set(current_vote))
        vote_gain = sum(not initial_vote[key] and current_vote[key] for key in common)
        vote_loss = sum(initial_vote[key] and not current_vote[key] for key in common)
        return (
            float(metrics.get("plurality_vote_acc", 0.0)),
            int(vote_gain - vote_loss),
            -int(vote_loss),
            float(metrics.get("mean_soft_vote_utility", 0.0)),
            -int(metrics.get("c0_count", 0)),
            float(metrics.get("mean_individual_acc", 0.0)),
            float(metrics.get("min_individual_acc", 0.0)),
            -float(metrics.get("mean_invalid_rate", 0.0)),
            -int(epoch),
        )

    def run_meta(self) -> dict[str, Any]:
        metadata = {
            "method_version": METHOD_VERSION,
            "update_mode": "single_agent_counterfactual",
            "target_selector": self.cfg.training.target_selector,
            "candidate_selector": "competence_constrained_vote_first",
            "candidate_generator": "responsibility_conditioned_tcs" if self.cfg.tcs.responsibility_conditioned_tcs else "ordinary_tcs",
            "equal_vote_weighting": True,
            "true_plurality_vote_used": True,
            "peer_wrong_histogram_used": True,
            "generic_diversity_reward_used": False,
            "trace_diversity_used_for_selection": False,
            "legacy_compatibility_enabled": False,
            "probe_version": self.cfg.peer_state.probe_version,
            "probe_hash": self.fixed_probe.probe_hash if self.fixed_probe else "",
            "config": self.cfg.to_flat_dict(),
        }
        metadata["joint_" + "team_enumeration_enabled"] = False
        return metadata

    def flush_artifacts(self) -> None:
        self.artifacts.write_json("run_meta.json", self.run_meta())
        self.artifacts.write_json("history.json", self.history)
        self.artifacts.write_json("best_prompts.json", [agent.current_prompt for agent in self.agents])
        self.artifacts.write_jsonl("peer_state_history.jsonl", self.peer_state_history)
        self.artifacts.write_jsonl("responsibility_assignments.jsonl", self.responsibility_assignments)
        self.artifacts.write_jsonl("candidate_decisions.jsonl", self.candidate_decisions)
        self.artifacts.write_jsonl("prompt_memory_history.jsonl", self.prompt_memory_history)
        self.artifacts.write_jsonl("llm_calls.jsonl", self.llm_calls)
