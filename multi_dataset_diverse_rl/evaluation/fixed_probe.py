from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from ..candidate_selection import CandidateEvaluation, PromptCompetenceMetrics, TeamOutcomeMetrics
from ..member_objectives import member_gain_metrics
from ..peer_state import build_peer_vote_context, build_team_vote_state, soft_vote_utility
from ..responsibility import (
    CandidateMarginalContribution,
    ProtectionContribution,
    compute_member_aware_repair_opportunity,
)
from .prompt_question import PromptAnswer, PromptQuestionEvaluator


@dataclass(frozen=True)
class ProbeExample:
    question: str
    question_hash: str
    gold_answer: str


def fixed_probe_hash(examples: Sequence[ProbeExample], version: str) -> str:
    payload = [(row.question_hash, row.gold_answer) for row in examples]
    return hashlib.sha256(json.dumps([version, payload], sort_keys=True).encode("utf-8")).hexdigest()


class FixedProbeEvaluator:
    def __init__(
        self,
        examples: Sequence[ProbeExample],
        version: str,
        prompt_question_evaluator: PromptQuestionEvaluator,
    ):
        self.examples = tuple(examples)
        self.version = str(version)
        self.probe_hash = fixed_probe_hash(self.examples, self.version)
        self.prompt_question_evaluator = prompt_question_evaluator

    @property
    def cache_hits(self) -> int:
        return self.prompt_question_evaluator.cache_hits

    @property
    def cache_misses(self) -> int:
        return self.prompt_question_evaluator.cache_misses

    async def evaluate_prompt(
        self,
        agent_id: int,
        prompt: str,
        prompt_hash: str,
        solve: Callable[[str, int, str], Awaitable[PromptAnswer]],
    ) -> tuple[PromptAnswer, ...]:
        rows = await self.evaluate_prompt_indices(agent_id, prompt, prompt_hash, range(len(self.examples)), solve)
        return tuple(rows[index] for index in range(len(self.examples)))

    async def evaluate_prompt_indices(
        self,
        agent_id: int,
        prompt: str,
        prompt_hash: str,
        indices: Sequence[int],
        solve: Callable[[str, int, str], Awaitable[PromptAnswer]],
    ) -> dict[int, PromptAnswer]:
        selected = tuple(dict.fromkeys(int(index) for index in indices))
        if any(index < 0 or index >= len(self.examples) for index in selected):
            raise IndexError("fixed-probe index is outside the probe")

        async def one(index: int) -> tuple[int, PromptAnswer]:
            example = self.examples[index]
            answer = await self.prompt_question_evaluator.evaluate(
                question=example.question,
                question_hash=example.question_hash,
                prompt=prompt,
                prompt_hash=prompt_hash,
                agent_id=agent_id,
                solve=solve,
            )
            return index, answer

        return dict(await asyncio.gather(*(one(index) for index in selected)))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "probe_hash": self.probe_hash,
            "prompt_question_evaluator": self.prompt_question_evaluator.to_dict(),
        }

    def restore(self, payload: dict[str, object]) -> None:
        if str(payload["version"]) != self.version or str(payload["probe_hash"]) != self.probe_hash:
            raise ValueError("Fixed probe cache version or hash mismatch. Start a new run.")
        evaluator_payload = payload["prompt_question_evaluator"]
        if not isinstance(evaluator_payload, dict):
            raise ValueError("fixed probe prompt-question evaluator must be an object")
        self.prompt_question_evaluator.restore(evaluator_payload)


def evaluate_candidate_profile(
    *,
    prompt: str,
    prompt_hash: str,
    examples: Sequence[ProbeExample],
    active_profiles: Sequence[Sequence[PromptAnswer]],
    initial_profiles: Sequence[Sequence[PromptAnswer]],
    candidate_profile: Sequence[PromptAnswer],
    target_agent_id: int,
    assigned_question_hashes: set[str],
    normalize_answer: Callable[[str], str],
    match_answer: Callable[[str, str], bool],
    tie_break: str,
    seed: int,
    tau: float,
) -> CandidateEvaluation:
    if len(active_profiles) != 5:
        raise ValueError("candidate evaluation requires five active agent profiles")
    if any(len(profile) != len(examples) for profile in active_profiles):
        raise ValueError("active profile length differs from fixed probe")
    if len(candidate_profile) != len(examples):
        raise ValueError("candidate profile length differs from fixed probe")
    if len(initial_profiles) != 5 or any(
        len(profile) != len(examples) for profile in initial_profiles
    ):
        raise ValueError("initial profiles must contain five complete fixed-probe profiles")

    target_correct = invalid_count = terminal_invalid_count = 0
    vote_gain = vote_loss = coverage_gain = coverage_loss = 0
    unique_loss = pivotal_loss = 0
    dominant_exit = dominant_join = assigned_repair = 0
    utility_delta = assigned_utility_delta = utility_total = 0.0
    vote_vector: list[bool] = []
    gold_counts: list[int] = []
    wrong_counts: list[int] = []
    margins: list[int] = []
    initial_member_counts = [
        sum(
            int(
                profile[row_index].valid
                and match_answer(profile[row_index].answer, row.gold_answer)
            )
            for row_index, row in enumerate(examples)
        )
        for profile in initial_profiles
    ]
    incumbent_member_counts = [
        sum(
            int(
                profile[row_index].valid
                and match_answer(profile[row_index].answer, row.gold_answer)
            )
            for row_index, row in enumerate(examples)
        )
        for profile in active_profiles
    ]
    candidate_member_counts = list(incumbent_member_counts)
    candidate_member_counts[target_agent_id] = sum(
        int(
            answer.valid
            and match_answer(answer.answer, example.gold_answer)
        )
        for answer, example in zip(candidate_profile, examples, strict=True)
    )
    active_gains = [
        current_count - initial_count
        for current_count, initial_count in zip(
            incumbent_member_counts, initial_member_counts, strict=True
        )
    ]
    candidate_gains = [
        current_count - initial_count
        for current_count, initial_count in zip(
            candidate_member_counts, initial_member_counts, strict=True
        )
    ]
    zero_protection_counts = (0,) * 5

    for index, example in enumerate(examples):
        active_answers = [profile[index].answer for profile in active_profiles]
        active_validity = [profile[index].valid for profile in active_profiles]
        candidate_answers = list(active_answers)
        candidate_validity = list(active_validity)
        candidate_answers[target_agent_id] = candidate_profile[index].answer
        candidate_validity[target_agent_id] = candidate_profile[index].valid
        current = build_team_vote_state(
            question_hash=example.question_hash,
            gold_answer=example.gold_answer,
            answers=active_answers,
            valid_vector=active_validity,
            normalize_answer=normalize_answer,
            match_answer=match_answer,
            tie_break=tie_break,
            seed=seed,
        )
        candidate = build_team_vote_state(
            question_hash=example.question_hash,
            gold_answer=example.gold_answer,
            answers=candidate_answers,
            valid_vector=candidate_validity,
            normalize_answer=normalize_answer,
            match_answer=match_answer,
            tie_break=tie_break,
            seed=seed,
        )
        current_opportunity = compute_member_aware_repair_opportunity(
            team_state=current,
            peer_context=build_peer_vote_context(current, target_agent_id),
            initial_correct_counts=initial_member_counts,
            member_correct_counts=incumbent_member_counts,
            member_gains_from_initial=active_gains,
            unique_correct_counts=zero_protection_counts,
            pivotal_correct_counts=zero_protection_counts,
            tau=tau,
        )
        candidate_opportunity = compute_member_aware_repair_opportunity(
            team_state=candidate,
            peer_context=build_peer_vote_context(candidate, target_agent_id),
            initial_correct_counts=initial_member_counts,
            member_correct_counts=candidate_member_counts,
            member_gains_from_initial=candidate_gains,
            unique_correct_counts=zero_protection_counts,
            pivotal_correct_counts=zero_protection_counts,
            tau=tau,
        )
        candidate_correct = candidate.team_correctness[target_agent_id]
        target_correct += int(candidate_correct)
        invalid_count += int(not candidate.team_validity[target_agent_id])
        terminal_invalid_count += int(candidate_profile[index].terminal_invalid)
        vote_gain += int(not current.vote_correct and candidate.vote_correct)
        vote_loss += int(current.vote_correct and not candidate.vote_correct)
        coverage_gain += int(current.gold_vote_count == 0 and candidate.gold_vote_count > 0)
        coverage_loss += int(current.gold_vote_count > 0 and candidate.gold_vote_count == 0)
        unique_loss += int(current_opportunity.unique_correct and not candidate_correct)
        pivotal_loss += int(current_opportunity.pivotal_correct and not candidate_correct)
        dominant_exit += int(
            current_opportunity.dominant_wrong_member and not candidate_opportunity.dominant_wrong_member
        )
        dominant_join += int(
            not current_opportunity.dominant_wrong_member and candidate_opportunity.dominant_wrong_member
        )
        current_utility = soft_vote_utility(current.gold_vote_count, current.plurality_margin, tau)
        candidate_utility = soft_vote_utility(candidate.gold_vote_count, candidate.plurality_margin, tau)
        delta = candidate_utility - current_utility
        if current.gold_vote_count == 0 and candidate.gold_vote_count == 0:
            delta = 0.0
        utility_delta += delta
        utility_total += candidate_utility
        if example.question_hash in assigned_question_hashes:
            assigned_utility_delta += delta
            assigned_repair += int(not current_opportunity.current_correct and candidate_correct)
        vote_vector.append(candidate.vote_correct)
        gold_counts.append(candidate.gold_vote_count)
        wrong_counts.append(candidate.largest_wrong_vote_count)
        margins.append(candidate.plurality_margin)
    size = len(examples)
    denominator = max(1, size)
    gains = member_gain_metrics(
        initial_member_counts,
        incumbent_member_counts,
        candidate_member_counts,
        target_agent_id,
    )
    return CandidateEvaluation(
        prompt=str(prompt),
        prompt_hash=str(prompt_hash),
        competence=PromptCompetenceMetrics(
            correct_count=target_correct,
            accuracy=target_correct / denominator,
            invalid_count=invalid_count,
            invalid_rate=invalid_count / denominator,
            terminal_invalid_count=terminal_invalid_count,
        ),
        team_outcome=TeamOutcomeMetrics(
            vote_correct_vector=tuple(vote_vector),
            vote_correct_count=sum(vote_vector),
            plurality_vote_accuracy=sum(vote_vector) / denominator,
            gold_vote_counts=tuple(gold_counts),
            largest_wrong_vote_counts=tuple(wrong_counts),
            plurality_margins=tuple(margins),
            mean_soft_vote_utility=utility_total / denominator,
        ),
        marginal=CandidateMarginalContribution(
            vote_gain_count=vote_gain,
            vote_loss_count=vote_loss,
            net_vote_delta=vote_gain - vote_loss,
            soft_utility_delta=utility_delta / denominator,
            coverage_gain_count=coverage_gain,
            coverage_loss_count=coverage_loss,
            dominant_wrong_exit_count=dominant_exit,
            dominant_wrong_join_count=dominant_join,
            assigned_residual_repair_count=assigned_repair,
            assigned_residual_utility_delta=assigned_utility_delta / denominator,
        ),
        protection=ProtectionContribution(
            unique_correct_loss_count=unique_loss,
            pivotal_correct_loss_count=pivotal_loss,
        ),
        member_gain=gains,
    )


def subset_profiles(
    examples: Sequence[ProbeExample],
    profiles: Sequence[Sequence[PromptAnswer]],
    indices: Sequence[int],
) -> tuple[tuple[ProbeExample, ...], list[tuple[PromptAnswer, ...]]]:
    selected_examples = tuple(examples[index] for index in indices)
    selected_profiles = [tuple(profile[index] for index in indices) for profile in profiles]
    return selected_examples, selected_profiles
