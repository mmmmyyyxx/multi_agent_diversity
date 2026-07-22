from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

from ..peer_state import build_peer_vote_state, soft_vote_utility
from ..responsibility import counterfactual_credit


@dataclass(frozen=True)
class ProbeExample:
    question: str
    question_hash: str
    gold_answer: str


@dataclass(frozen=True)
class PromptAnswer:
    answer: str
    trace: str
    valid: bool


def fixed_probe_hash(examples: Sequence[ProbeExample], version: str) -> str:
    payload = [(row.question_hash, row.gold_answer) for row in examples]
    return hashlib.sha256(json.dumps([version, payload], sort_keys=True).encode("utf-8")).hexdigest()


class FixedProbeEvaluator:
    def __init__(self, examples: Sequence[ProbeExample], version: str):
        self.examples = tuple(examples)
        self.version = str(version)
        self.probe_hash = fixed_probe_hash(self.examples, self.version)
        self.cache: dict[str, PromptAnswer] = {}
        self.inflight: dict[str, asyncio.Future] = {}
        self.lock = asyncio.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def _key(self, agent_id: int, prompt_hash: str, question_hash: str) -> str:
        return hashlib.sha256(
            f"{self.version}|{self.probe_hash}|{agent_id}|{prompt_hash}|{question_hash}".encode("utf-8")
        ).hexdigest()

    async def evaluate_prompt(
        self,
        agent_id: int,
        prompt: str,
        prompt_hash: str,
        solve: Callable[[str, int, str], Awaitable[PromptAnswer]],
    ) -> tuple[PromptAnswer, ...]:
        rows = await self.evaluate_prompt_indices(
            agent_id, prompt, prompt_hash, range(len(self.examples)), solve,
        )
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
            key = self._key(agent_id, prompt_hash, example.question_hash)
            cached = self.cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                return index, cached
            owner = False
            async with self.lock:
                future = self.inflight.get(key)
                if future is None:
                    future = asyncio.get_running_loop().create_future()
                    self.inflight[key] = future
                    owner = True
            if not owner:
                self.cache_hits += 1
                return index, await future
            try:
                self.cache_misses += 1
                answer = await solve(example.question, agent_id, prompt)
                self.cache[key] = answer
                future.set_result(answer)
                return index, answer
            except Exception as exc:
                future.set_exception(exc)
                raise
            finally:
                async with self.lock:
                    self.inflight.pop(key, None)

        return dict(await asyncio.gather(*(one(index) for index in selected)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "probe_hash": self.probe_hash,
            "cache": {key: value.__dict__ for key, value in self.cache.items()},
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        if str(payload.get("version", "")) != self.version or str(payload.get("probe_hash", "")) != self.probe_hash:
            raise ValueError("Fixed probe cache version or hash mismatch. Start a new run.")
        self.cache = {
            str(key): PromptAnswer(**value) for key, value in dict(payload.get("cache", {})).items()
        }
        self.cache_hits = int(payload.get("cache_hits", 0))
        self.cache_misses = int(payload.get("cache_misses", 0))


def candidate_probe_metrics(
    *,
    examples: Sequence[ProbeExample],
    active_profiles: Sequence[Sequence[PromptAnswer]],
    candidate_profile: Sequence[PromptAnswer],
    target_agent_id: int,
    assigned_question_hashes: set[str],
    normalize_answer: Callable[[str], str],
    match_answer: Callable[[str, str], bool],
    tie_break: str,
    seed: int,
    tau: float,
) -> dict[str, Any]:
    target_correct = 0
    invalid_count = 0
    vote_gain = vote_loss = coverage_gain = coverage_loss = 0
    unique_loss = pivotal_loss = 0
    dominant_exit = dominant_join = assigned_repair = 0
    utility_delta = assigned_utility_delta = 0.0
    vote_correct_vector: list[bool] = []
    gold_counts: list[int] = []
    wrong_counts: list[int] = []
    margins: list[int] = []
    utility_vector: list[float] = []
    normalized_answer_hashes: list[str] = []
    vote_contribution: list[int] = []
    coverage_contribution: list[int] = []
    unique_vector: list[bool] = []
    pivotal_vector: list[bool] = []
    dominant_vector: list[bool] = []

    for index, example in enumerate(examples):
        active_answers = [profile[index].answer for profile in active_profiles]
        active_valid = [profile[index].valid for profile in active_profiles]
        candidate_answers = list(active_answers)
        candidate_valid = list(active_valid)
        candidate_answers[target_agent_id] = candidate_profile[index].answer
        candidate_valid[target_agent_id] = candidate_profile[index].valid
        current = build_peer_vote_state(
            question_hash=example.question_hash, gold_answer=example.gold_answer,
            answers=active_answers, valid_vector=active_valid, normalize_answer=normalize_answer,
            match_answer=match_answer, tie_break=tie_break, seed=seed,
        )
        candidate = build_peer_vote_state(
            question_hash=example.question_hash, gold_answer=example.gold_answer,
            answers=candidate_answers, valid_vector=candidate_valid, normalize_answer=normalize_answer,
            match_answer=match_answer, tie_break=tie_break, seed=seed,
        )
        current_credit = counterfactual_credit(
            agent_id=target_agent_id, current_state=current, gold_answer=example.gold_answer,
            normalize_answer=normalize_answer, match_answer=match_answer,
            tie_break=tie_break, seed=seed, tau=tau,
        )
        candidate_credit = counterfactual_credit(
            agent_id=target_agent_id, current_state=candidate, gold_answer=example.gold_answer,
            normalize_answer=normalize_answer, match_answer=match_answer,
            tie_break=tie_break, seed=seed, tau=tau,
        )
        candidate_correct = bool(candidate.correctness_vector[target_agent_id])
        target_correct += int(candidate_correct)
        invalid_count += int(not candidate.valid_vector[target_agent_id])
        vote_gain += int(not current.vote_correct and candidate.vote_correct)
        vote_loss += int(current.vote_correct and not candidate.vote_correct)
        coverage_gain += int(current.gold_vote_count == 0 and candidate.gold_vote_count > 0)
        coverage_loss += int(current.gold_vote_count > 0 and candidate.gold_vote_count == 0)
        unique_loss += int(current_credit.unique_correct and not candidate_correct)
        pivotal_loss += int(current_credit.pivotal_vote_correct and not candidate_correct)
        dominant_exit += int(current_credit.dominant_wrong_member and not candidate_credit.dominant_wrong_member)
        dominant_join += int(not current_credit.dominant_wrong_member and candidate_credit.dominant_wrong_member)
        current_utility = soft_vote_utility(current.gold_vote_count, current.plurality_margin, tau)
        candidate_utility = soft_vote_utility(candidate.gold_vote_count, candidate.plurality_margin, tau)
        delta = candidate_utility - current_utility
        utility_delta += delta
        if example.question_hash in assigned_question_hashes:
            assigned_utility_delta += delta
            assigned_repair += int(not current_credit.current_correct and candidate_correct)
        vote_correct_vector.append(candidate.vote_correct)
        gold_counts.append(candidate.gold_vote_count)
        wrong_counts.append(candidate.largest_wrong_vote_count)
        margins.append(candidate.plurality_margin)
        utility_vector.append(candidate_utility)
        normalized_answer_hashes.append(hashlib.sha256(candidate.normalized_answers[target_agent_id].encode("utf-8")).hexdigest())
        vote_contribution.append(int(candidate.vote_correct) - int(current.vote_correct))
        coverage_contribution.append(int(candidate.gold_vote_count > 0) - int(current.gold_vote_count > 0))
        unique_vector.append(candidate_credit.unique_correct)
        pivotal_vector.append(candidate_credit.pivotal_vote_correct)
        dominant_vector.append(candidate_credit.dominant_wrong_member)

    size = max(1, len(examples))
    return {
        "candidate_target_correct_count": target_correct,
        "candidate_target_accuracy": target_correct / size,
        "candidate_invalid_count": invalid_count,
        "candidate_invalid_rate": invalid_count / size,
        "vote_gain_count": vote_gain,
        "vote_loss_count": vote_loss,
        "net_vote_delta": vote_gain - vote_loss,
        "soft_vote_utility_delta": utility_delta / size,
        "coverage_gain_count": coverage_gain,
        "coverage_loss_count": coverage_loss,
        "unique_correct_loss_count": unique_loss,
        "pivotal_vote_correct_loss_count": pivotal_loss,
        "assigned_residual_repair_count": assigned_repair,
        "assigned_residual_utility_delta": assigned_utility_delta / size,
        "dominant_wrong_exit_count": dominant_exit,
        "dominant_wrong_join_count": dominant_join,
        "vote_correct_vector": vote_correct_vector,
        "plurality_vote_accuracy": sum(vote_correct_vector) / size,
        "gold_vote_count_vector": gold_counts,
        "largest_wrong_vote_count_vector": wrong_counts,
        "plurality_margin_vector": margins,
        "soft_vote_utility_vector": utility_vector,
        "mean_soft_vote_utility": sum(utility_vector) / size,
        "answer_hashes": normalized_answer_hashes,
        "vote_contribution_vector": vote_contribution,
        "coverage_contribution_vector": coverage_contribution,
        "unique_correct_vector": unique_vector,
        "pivotal_correct_vector": pivotal_vector,
        "dominant_wrong_membership_vector": dominant_vector,
    }


def subset_profiles(
    examples: Sequence[ProbeExample], profiles: Sequence[Sequence[PromptAnswer]], indices: Sequence[int],
) -> tuple[tuple[ProbeExample, ...], list[tuple[PromptAnswer, ...]]]:
    selected_examples = tuple(examples[index] for index in indices)
    selected_profiles = [tuple(profile[index] for index in indices) for profile in profiles]
    return selected_examples, selected_profiles
