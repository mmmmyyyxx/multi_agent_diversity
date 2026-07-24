from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .member_objectives import (
    MemberGainMetrics,
    pareto_dominates,
    pareto_front,
    team_objective_vector,
)
from .responsibility import CandidateMarginalContribution, ProtectionContribution


@dataclass(frozen=True)
class PromptCompetenceMetrics:
    correct_count: int
    accuracy: float
    invalid_count: int
    invalid_rate: float


@dataclass(frozen=True)
class TeamOutcomeMetrics:
    vote_correct_vector: tuple[bool, ...]
    vote_correct_count: int
    plurality_vote_accuracy: float
    gold_vote_counts: tuple[int, ...]
    largest_wrong_vote_counts: tuple[int, ...]
    plurality_margins: tuple[int, ...]
    mean_soft_vote_utility: float


@dataclass(frozen=True)
class CandidateEvaluation:
    prompt: str
    prompt_hash: str
    competence: PromptCompetenceMetrics
    team_outcome: TeamOutcomeMetrics
    marginal: CandidateMarginalContribution
    protection: ProtectionContribution
    member_gain: MemberGainMetrics


@dataclass(frozen=True)
class StageAScores:
    team_vote_key: tuple
    worst_member_key: tuple
    mean_member_key: tuple


@dataclass(frozen=True)
class StageASelectionDecision:
    selected: bool
    selected_by_channels: tuple[str, ...]
    pareto_front: int
    aggregate_rank: int


@dataclass(frozen=True)
class ConstraintLimits:
    local_accuracy_allowance: int = 0
    global_accuracy_allowance: int = 0
    invalid_allowance: int = 0
    vote_loss_limit: int = 0
    unique_correct_loss_limit: int = 0
    pivotal_loss_limit: int = 0


@dataclass(frozen=True)
class ConstraintDecision:
    passed: bool
    local_accuracy_passed: bool
    initial_accuracy_passed: bool
    invalid_passed: bool
    vote_loss_passed: bool
    unique_correct_passed: bool
    pivotal_correct_passed: bool
    rejection_reasons: tuple[str, ...]


def stage_a_scores(candidate: CandidateEvaluation) -> StageAScores:
    incumbent_gains = tuple(
        current - initial
        for current, initial in zip(
            candidate.member_gain.incumbent_correct_counts,
            candidate.member_gain.initial_correct_counts,
            strict=True,
        )
    )
    return StageAScores(
        team_vote_key=(
            candidate.team_outcome.vote_correct_count,
            candidate.marginal.net_vote_delta,
            -candidate.marginal.vote_loss_count,
            candidate.team_outcome.mean_soft_vote_utility,
            candidate.marginal.assigned_residual_repair_count,
        ),
        worst_member_key=(
            candidate.member_gain.minimum_gain_count,
            candidate.member_gain.minimum_gain_count - min(incumbent_gains),
            candidate.member_gain.improved_agent_count,
            candidate.member_gain.target_gain_vs_incumbent,
            -candidate.competence.invalid_count,
        ),
        mean_member_key=(
            candidate.member_gain.total_gain_count,
            candidate.member_gain.target_gain_vs_incumbent,
            candidate.member_gain.improved_agent_count,
            candidate.marginal.assigned_residual_repair_count,
            -candidate.competence.invalid_count,
        ),
    )


def _ordinal_ranks(candidates: Sequence[CandidateEvaluation], attribute: str) -> dict[str, int]:
    keys = {candidate.prompt_hash: getattr(stage_a_scores(candidate), attribute) for candidate in candidates}
    ordered_unique = sorted(set(keys.values()), reverse=True)
    rank_by_key = {key: index + 1 for index, key in enumerate(ordered_unique)}
    return {prompt_hash: rank_by_key[key] for prompt_hash, key in keys.items()}


def _pareto_fronts(rank_vectors: dict[str, tuple[int, int, int]]) -> dict[str, int]:
    remaining = set(rank_vectors)
    fronts: dict[str, int] = {}
    front = 1
    while remaining:
        current = []
        for prompt_hash in sorted(remaining):
            vector = rank_vectors[prompt_hash]
            dominated = any(
                all(other_value <= value for other_value, value in zip(rank_vectors[other], vector, strict=True))
                and any(other_value < value for other_value, value in zip(rank_vectors[other], vector, strict=True))
                for other in remaining
                if other != prompt_hash
            )
            if not dominated:
                current.append(prompt_hash)
        if not current:
            raise AssertionError("Pareto front construction made no progress")
        for prompt_hash in current:
            fronts[prompt_hash] = front
            remaining.remove(prompt_hash)
        front += 1
    return fronts


def stage_a_multichannel_shortlist(
    candidates: Sequence[CandidateEvaluation],
    *,
    channel_top_k: int = 2,
    total_budget: int,
) -> tuple[list[CandidateEvaluation], dict[str, StageASelectionDecision]]:
    unique = {candidate.prompt_hash: candidate for candidate in candidates}
    rows = list(unique.values())
    if total_budget < 0 or channel_top_k < 0:
        raise ValueError("Stage A budgets cannot be negative")
    channels = {
        "team_vote": _ordinal_ranks(rows, "team_vote_key"),
        "worst_member": _ordinal_ranks(rows, "worst_member_key"),
        "mean_member": _ordinal_ranks(rows, "mean_member_key"),
    }
    rank_vectors = {
        candidate.prompt_hash: tuple(channels[name][candidate.prompt_hash] for name in channels)
        for candidate in rows
    }
    fronts = _pareto_fronts(rank_vectors) if rows else {}
    selected_by: dict[str, set[str]] = {candidate.prompt_hash: set() for candidate in rows}
    union: set[str] = set()
    for name, ranks in channels.items():
        ordered = sorted(rows, key=lambda row: (ranks[row.prompt_hash], row.prompt_hash))
        for candidate in ordered[:channel_top_k]:
            union.add(candidate.prompt_hash)
            selected_by[candidate.prompt_hash].add(name)

    ordering = sorted(
        rows,
        key=lambda row: (
            fronts[row.prompt_hash],
            sum(rank_vectors[row.prompt_hash]),
            rank_vectors[row.prompt_hash],
            row.prompt_hash,
        ),
    )
    if len(union) > total_budget:
        selected_hashes = {
            row.prompt_hash for row in ordering if row.prompt_hash in union
        }
        selected_hashes = set(sorted(
            selected_hashes,
            key=lambda prompt_hash: (
                fronts[prompt_hash], sum(rank_vectors[prompt_hash]), rank_vectors[prompt_hash], prompt_hash,
            ),
        )[:total_budget])
    else:
        selected_hashes = set(union)
        for row in ordering:
            if len(selected_hashes) >= total_budget:
                break
            selected_hashes.add(row.prompt_hash)

    decisions = {
        row.prompt_hash: StageASelectionDecision(
            selected=row.prompt_hash in selected_hashes,
            selected_by_channels=tuple(sorted(selected_by[row.prompt_hash])),
            pareto_front=fronts[row.prompt_hash],
            aggregate_rank=sum(rank_vectors[row.prompt_hash]),
        )
        for row in rows
    }
    shortlist = [row for row in ordering if row.prompt_hash in selected_hashes]
    return shortlist, decisions


def evaluate_constraints(
    candidate: CandidateEvaluation,
    active: CandidateEvaluation,
    initial: CandidateEvaluation,
    limits: ConstraintLimits,
) -> ConstraintDecision:
    local = candidate.competence.correct_count >= active.competence.correct_count - limits.local_accuracy_allowance
    global_ = candidate.competence.correct_count >= initial.competence.correct_count - limits.global_accuracy_allowance
    invalid = candidate.competence.invalid_count <= active.competence.invalid_count + limits.invalid_allowance
    vote_loss = candidate.marginal.vote_loss_count <= limits.vote_loss_limit
    unique = candidate.protection.unique_correct_loss_count <= limits.unique_correct_loss_limit
    pivotal = candidate.protection.pivotal_correct_loss_count <= limits.pivotal_loss_limit
    checks = (
        ("local_accuracy", local),
        ("initial_accuracy", global_),
        ("invalid", invalid),
        ("vote_loss", vote_loss),
        ("unique_correct", unique),
        ("pivotal_correct", pivotal),
    )
    reasons = tuple(name for name, passed in checks if not passed)
    return ConstraintDecision(
        passed=not reasons,
        local_accuracy_passed=local,
        initial_accuracy_passed=global_,
        invalid_passed=invalid,
        vote_loss_passed=vote_loss,
        unique_correct_passed=unique,
        pivotal_correct_passed=pivotal,
        rejection_reasons=reasons,
    )


def vote_first_key(candidate: CandidateEvaluation, generation: int = 0) -> tuple:
    return (
        candidate.marginal.net_vote_delta,
        -candidate.marginal.vote_loss_count,
        candidate.marginal.soft_utility_delta,
        candidate.marginal.coverage_gain_count,
        candidate.marginal.assigned_residual_utility_delta,
        candidate.competence.correct_count,
        -candidate.competence.invalid_count,
        -int(generation),
        candidate.prompt_hash,
    )


def member_first_key(candidate: CandidateEvaluation, generation: int = 0) -> tuple:
    return (
        candidate.member_gain.minimum_gain_count,
        candidate.team_outcome.vote_correct_count,
        candidate.member_gain.total_gain_count,
        candidate.member_gain.improved_agent_count,
        -candidate.marginal.vote_loss_count,
        candidate.team_outcome.mean_soft_vote_utility,
        candidate.marginal.assigned_residual_repair_count,
        candidate.competence.correct_count,
        -candidate.competence.invalid_count,
        -int(generation),
        candidate.prompt_hash,
    )


def member_aware_pareto_front(
    candidates: Sequence[CandidateEvaluation],
) -> tuple[str, ...]:
    vectors = tuple(
        team_objective_vector(
            candidate.team_outcome.vote_correct_count,
            candidate.member_gain,
        )
        for candidate in candidates
    )
    return tuple(candidates[index].prompt_hash for index in pareto_front(vectors))


def individual_accuracy_key(candidate: CandidateEvaluation, generation: int = 0) -> tuple:
    return (
        candidate.competence.correct_count,
        -candidate.competence.invalid_count,
        -int(generation),
        candidate.prompt_hash,
    )


def candidate_is_acceptable(
    candidate: CandidateEvaluation,
    incumbent: CandidateEvaluation,
) -> bool:
    return pareto_dominates(
        team_objective_vector(
            candidate.team_outcome.vote_correct_count,
            candidate.member_gain,
        ),
        team_objective_vector(
            incumbent.team_outcome.vote_correct_count,
            incumbent.member_gain,
        ),
    )
