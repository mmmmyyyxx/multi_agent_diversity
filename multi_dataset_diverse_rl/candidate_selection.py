from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .member_objectives import (
    MemberGainMetrics,
    pareto_dominates,
    team_objective_vector,
)
from .responsibility import CandidateMarginalContribution, ProtectionContribution
from .responsibility_contribution import ResponsibilityContributionMetrics


@dataclass(frozen=True)
class PromptCompetenceMetrics:
    correct_count: int
    accuracy: float
    invalid_count: int
    invalid_rate: float
    terminal_invalid_count: int = 0


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
    responsibility_contribution: ResponsibilityContributionMetrics | None = None


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
class ConstraintDecision:
    passed: bool
    hard_feasible: bool
    target_correct_incumbent: int
    target_correct_candidate: int
    target_gain: int
    vote_correct_incumbent: int
    vote_correct_candidate: int
    vote_gain_count: int
    vote_loss_count: int
    vote_net_gain: int
    unique_correct_gain_count: int
    unique_correct_loss_count: int
    pivotal_correct_gain_count: int
    pivotal_correct_loss_count: int
    incumbent_objective: tuple[int, int, int]
    candidate_objective: tuple[int, int, int]
    derived_team_pareto_passed: bool
    objective_invariant_checked: bool
    minimum_gain_delta: int
    total_gain_delta: int
    target_is_unique_weakest: bool
    target_is_tied_weakest: bool
    pareto_dominates_incumbent: bool
    target_nonregression_passed: bool
    target_strict_improvement: bool
    team_vote_nonregression_passed: bool
    vote_strict_improvement: bool
    target_or_vote_progress_passed: bool
    member_objective_dominance_passed: bool
    terminal_invalid_nonregression_passed: bool
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


def stage_a_rcru_shortlist(
    candidates: Sequence[CandidateEvaluation],
    *,
    channel_top_k: int = 2,
    total_budget: int,
) -> tuple[list[CandidateEvaluation], dict[str, StageASelectionDecision]]:
    unique = {candidate.prompt_hash: candidate for candidate in candidates}
    rows = list(unique.values())
    if total_budget < 0 or channel_top_k < 0:
        raise ValueError("Stage A budgets cannot be negative")
    for row in rows:
        if row.responsibility_contribution is None:
            raise ValueError("rcru_metrics_missing_for_candidate")

    channel_keys = {
        "team_vote": {
            row.prompt_hash: (
                row.team_outcome.vote_correct_count,
                row.marginal.net_vote_delta,
                -row.marginal.vote_loss_count,
                row.team_outcome.mean_soft_vote_utility,
            )
            for row in rows
        },
        "lane_fulfillment": {
            row.prompt_hash: (
                row.responsibility_contribution.utility.utility_total,
                row.responsibility_contribution.utility.utility_delta,
                row.responsibility_contribution.utility.positive_support_count,
                -row.responsibility_contribution.utility.negative_support_count,
            )
            for row in rows
        },
        "coalition_contribution": {
            row.prompt_hash: (
                row.responsibility_contribution.coalition.net_contribution,
                -row.responsibility_contribution.coalition.negative_pivotal_count,
                row.responsibility_contribution.coalition.positive_pivotal_count,
            )
            for row in rows
        },
    }
    channels: dict[str, dict[str, int]] = {}
    for name, keys in channel_keys.items():
        ordered_unique = sorted(set(keys.values()), reverse=True)
        rank_by_key = {key: index + 1 for index, key in enumerate(ordered_unique)}
        channels[name] = {
            prompt_hash: rank_by_key[key] for prompt_hash, key in keys.items()
        }
    rank_vectors = {
        row.prompt_hash: tuple(
            channels[name][row.prompt_hash] for name in channels
        )
        for row in rows
    }
    fronts = _pareto_fronts(rank_vectors) if rows else {}
    selected_by = {row.prompt_hash: set() for row in rows}
    union: set[str] = set()
    for name, ranks in channels.items():
        ordered = sorted(
            rows, key=lambda row: (ranks[row.prompt_hash], row.prompt_hash)
        )
        for row in ordered[:channel_top_k]:
            union.add(row.prompt_hash)
            selected_by[row.prompt_hash].add(name)
    ordering = sorted(
        rows,
        key=lambda row: (
            fronts[row.prompt_hash],
            sum(rank_vectors[row.prompt_hash]),
            rank_vectors[row.prompt_hash],
            row.prompt_hash,
        ),
    )
    selected_hashes = set(
        sorted(
            union,
            key=lambda prompt_hash: (
                fronts[prompt_hash],
                sum(rank_vectors[prompt_hash]),
                rank_vectors[prompt_hash],
                prompt_hash,
            ),
        )[:total_budget]
    )
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
    return (
        [row for row in ordering if row.prompt_hash in selected_hashes],
        decisions,
    )


def evaluate_constraints(
    candidate: CandidateEvaluation,
    active: CandidateEvaluation,
) -> ConstraintDecision:
    incumbent_objective = team_objective_vector(
        active.team_outcome.vote_correct_count,
        active.member_gain,
    )
    candidate_objective = team_objective_vector(
        candidate.team_outcome.vote_correct_count,
        candidate.member_gain,
    )
    target_strict_improvement = (
        candidate.competence.correct_count > active.competence.correct_count
    )
    target_nonregression = (
        candidate.competence.correct_count >= active.competence.correct_count
    )
    team_vote_nonregression = (
        candidate.team_outcome.vote_correct_count
        >= active.team_outcome.vote_correct_count
    )
    vote_strict_improvement = (
        candidate.team_outcome.vote_correct_count
        > active.team_outcome.vote_correct_count
    )
    target_or_vote_progress = target_strict_improvement or vote_strict_improvement
    derived_team_pareto = pareto_dominates(
        candidate_objective,
        incumbent_objective,
    )
    target_gain = candidate.competence.correct_count - active.competence.correct_count
    minimum_gain_delta = (
        candidate.member_gain.minimum_gain_count
        - active.member_gain.minimum_gain_count
    )
    total_gain_delta = (
        candidate.member_gain.total_gain_count
        - active.member_gain.total_gain_count
    )
    if total_gain_delta != target_gain:
        raise AssertionError("single_target_total_gain_delta_mismatch")

    incumbent_gains = active.member_gain.gain_counts
    target_gain_before = active.member_gain.target_gain_vs_initial
    minimum_gain_before = min(incumbent_gains)
    minimum_count = incumbent_gains.count(minimum_gain_before)
    target_is_unique_weakest = (
        target_gain_before == minimum_gain_before and minimum_count == 1
    )
    target_is_tied_weakest = (
        target_gain_before == minimum_gain_before and minimum_count > 1
    )
    objective_invariant_checked = (
        target_nonregression
        and team_vote_nonregression
        and target_or_vote_progress
    )
    if objective_invariant_checked and not derived_team_pareto:
        raise AssertionError(
            "fixed_peer_single_target_objective_invariant_broken"
        )
    terminal_invalid_nonregression = (
        candidate.competence.terminal_invalid_count
        <= active.competence.terminal_invalid_count
    )
    checks = (
        ("target_regression", target_nonregression),
        ("team_vote_regression", team_vote_nonregression),
        ("no_target_or_vote_progress", target_or_vote_progress),
        ("terminal_invalid_regression", terminal_invalid_nonregression),
    )
    reasons = tuple(name for name, passed in checks if not passed)
    return ConstraintDecision(
        passed=not reasons,
        hard_feasible=not reasons,
        target_correct_incumbent=active.competence.correct_count,
        target_correct_candidate=candidate.competence.correct_count,
        target_gain=target_gain,
        vote_correct_incumbent=active.team_outcome.vote_correct_count,
        vote_correct_candidate=candidate.team_outcome.vote_correct_count,
        vote_gain_count=candidate.marginal.vote_gain_count,
        vote_loss_count=candidate.marginal.vote_loss_count,
        vote_net_gain=candidate.marginal.net_vote_delta,
        unique_correct_gain_count=(
            candidate.protection.unique_correct_gain_count
        ),
        unique_correct_loss_count=(
            candidate.protection.unique_correct_loss_count
        ),
        pivotal_correct_gain_count=(
            candidate.protection.pivotal_correct_gain_count
        ),
        pivotal_correct_loss_count=(
            candidate.protection.pivotal_correct_loss_count
        ),
        incumbent_objective=incumbent_objective.as_tuple(),
        candidate_objective=candidate_objective.as_tuple(),
        derived_team_pareto_passed=derived_team_pareto,
        objective_invariant_checked=objective_invariant_checked,
        minimum_gain_delta=minimum_gain_delta,
        total_gain_delta=total_gain_delta,
        target_is_unique_weakest=target_is_unique_weakest,
        target_is_tied_weakest=target_is_tied_weakest,
        pareto_dominates_incumbent=derived_team_pareto,
        target_nonregression_passed=target_nonregression,
        target_strict_improvement=target_strict_improvement,
        team_vote_nonregression_passed=team_vote_nonregression,
        vote_strict_improvement=vote_strict_improvement,
        target_or_vote_progress_passed=target_or_vote_progress,
        member_objective_dominance_passed=derived_team_pareto,
        terminal_invalid_nonregression_passed=terminal_invalid_nonregression,
        rejection_reasons=reasons,
    )


def vote_first_key(candidate: CandidateEvaluation, generation: int = 0) -> tuple:
    return (
        candidate.marginal.net_vote_delta,
        -candidate.marginal.vote_loss_count,
        candidate.marginal.soft_utility_delta,
        candidate.marginal.coverage_gain_count,
        candidate.competence.correct_count,
        -candidate.competence.invalid_count,
        -int(generation),
        candidate.prompt_hash,
    )


def common_monotone_safe_key(
    candidate: CandidateEvaluation,
    generation: int = 0,
) -> tuple:
    """Shared S1-S4 ranking over the common monotone-safe feasible set."""

    return (
        candidate.team_outcome.vote_correct_count,
        candidate.competence.correct_count,
        candidate.team_outcome.mean_soft_vote_utility,
        -candidate.marginal.vote_loss_count,
        -candidate.competence.invalid_count,
        -int(generation),
        candidate.prompt_hash,
    )


def member_first_safe_key(candidate: CandidateEvaluation, generation: int = 0) -> tuple:
    return (
        candidate.member_gain.minimum_gain_count,
        candidate.team_outcome.vote_correct_count,
        candidate.member_gain.target_gain_vs_incumbent,
        candidate.marginal.net_vote_delta,
        candidate.marginal.assigned_residual_repair_count,
        candidate.team_outcome.mean_soft_vote_utility,
        candidate.marginal.coverage_gain_count,
        -candidate.marginal.vote_loss_count,
        -candidate.protection.pivotal_correct_loss_count,
        -candidate.protection.unique_correct_loss_count,
        -int(generation),
        candidate.prompt_hash,
    )

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
    return evaluate_constraints(candidate, incumbent).passed
