from __future__ import annotations

import difflib
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from .peer_state import TeamVoteState, build_peer_vote_context, build_team_vote_state
from .versions import RCRU_VERSION


BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_LOWER_QUANTILE = 0.10


class ResponsibilityUtilityKind(str, Enum):
    COVERAGE = "coverage"
    DIRECT_FLIP = "direct_flip"
    MARGIN_SUPPORT = "margin_support"


@dataclass(frozen=True)
class ResponsibilityUtilityMetrics:
    repair_lane: str
    active_residual_count: int
    utility_total: int
    incumbent_utility_total: int
    utility_delta: int
    positive_support_count: int
    negative_support_count: int
    unchanged_support_count: int
    per_example_deltas: tuple[int, ...]
    per_example_question_hashes: tuple[str, ...]


@dataclass(frozen=True)
class CoalitionContributionMetrics:
    positive_pivotal_count: int
    negative_pivotal_count: int
    net_contribution: int
    incumbent_positive_pivotal_count: int
    incumbent_negative_pivotal_count: int
    incumbent_net_contribution: int
    positive_pivotal_delta: int
    negative_pivotal_delta: int
    net_contribution_delta: int


@dataclass(frozen=True)
class RobustSupportMetrics:
    bootstrap_replicates: int
    bootstrap_lower_quantile: float
    bootstrap_mean_delta: float
    bootstrap_lcb: float
    deterministic_seed_hash: str


@dataclass(frozen=True)
class PromptEditMetrics:
    parent_character_count: int
    candidate_character_count: int
    character_growth: int
    parent_token_count: int
    candidate_token_count: int
    inserted_token_count: int
    deleted_token_count: int
    replaced_token_count: int
    total_edit_token_count: int
    normalized_edit_ratio: float


@dataclass(frozen=True)
class ResponsibilityContributionMetrics:
    utility: ResponsibilityUtilityMetrics
    coalition: CoalitionContributionMetrics
    robust_support: RobustSupportMetrics
    edit: PromptEditMetrics


@dataclass(frozen=True)
class RCRUIncumbentCache:
    question_hashes: tuple[str, ...]
    incumbent_states: Mapping[str, TeamVoteState]
    incumbent_full_vote_vector: tuple[bool, ...]
    peer_vote_vector: tuple[bool, ...]
    active_question_hashes: tuple[str, ...]
    active_lane: str
    incumbent_utility_by_hash: Mapping[str, int]
    parent_normalized: str
    parent_tokens: tuple[str, ...]


@dataclass(frozen=True)
class RobustContributionDecision:
    passed: bool
    hard_feasible: bool
    target_nonregression_passed: bool
    team_vote_nonregression_passed: bool
    terminal_invalid_nonregression_passed: bool
    active_lane_nonregression_passed: bool
    vote_or_lane_progress_passed: bool
    minimum_support_required: bool
    minimum_support_passed: bool
    no_negative_support_required: bool
    no_negative_support_passed: bool
    bootstrap_guard_required: bool
    bootstrap_guard_passed: bool
    vote_gain: int
    target_gain: int
    lane_utility_delta: int
    net_contribution_delta: int
    positive_support_count: int
    negative_support_count: int
    bootstrap_lcb: float
    rejection_reasons: tuple[str, ...]


def responsibility_utility(
    kind: ResponsibilityUtilityKind | str,
    *,
    target_correct: bool,
    team_vote_correct: bool,
    plurality_margin: int,
) -> int:
    lane = ResponsibilityUtilityKind(kind)
    if lane is ResponsibilityUtilityKind.COVERAGE:
        return int(target_correct)
    if lane is ResponsibilityUtilityKind.DIRECT_FLIP:
        return int(team_vote_correct)
    return min(int(plurality_margin), 0)


def responsibility_utility_metrics(
    *,
    repair_lane: ResponsibilityUtilityKind | str,
    active_question_hashes: Sequence[str],
    incumbent_states: Mapping[str, TeamVoteState],
    candidate_states: Mapping[str, TeamVoteState],
    target_agent_id: int,
    incumbent_utility_by_hash: Mapping[str, int] | None = None,
) -> ResponsibilityUtilityMetrics:
    lane = ResponsibilityUtilityKind(repair_lane)
    hashes = tuple(sorted(map(str, active_question_hashes)))
    if not hashes:
        raise ValueError("rcru_active_slice_empty")
    incumbent_total = candidate_total = 0
    deltas: list[int] = []
    for question_hash in hashes:
        if question_hash not in incumbent_states or question_hash not in candidate_states:
            raise ValueError("rcru_active_residual_missing_from_probe")
        incumbent = incumbent_states[question_hash]
        candidate = candidate_states[question_hash]
        incumbent_utility = (
            int(incumbent_utility_by_hash[question_hash])
            if incumbent_utility_by_hash is not None
            else responsibility_utility(
                lane,
                target_correct=incumbent.team_correctness[target_agent_id],
                team_vote_correct=incumbent.vote_correct,
                plurality_margin=incumbent.plurality_margin,
            )
        )
        candidate_utility = responsibility_utility(
            lane,
            target_correct=candidate.team_correctness[target_agent_id],
            team_vote_correct=candidate.vote_correct,
            plurality_margin=candidate.plurality_margin,
        )
        incumbent_total += incumbent_utility
        candidate_total += candidate_utility
        deltas.append(candidate_utility - incumbent_utility)
    return ResponsibilityUtilityMetrics(
        repair_lane=lane.value,
        active_residual_count=len(hashes),
        utility_total=candidate_total,
        incumbent_utility_total=incumbent_total,
        utility_delta=candidate_total - incumbent_total,
        positive_support_count=sum(delta > 0 for delta in deltas),
        negative_support_count=sum(delta < 0 for delta in deltas),
        unchanged_support_count=sum(delta == 0 for delta in deltas),
        per_example_deltas=tuple(deltas),
        per_example_question_hashes=hashes,
    )


def coalition_contribution_metrics(
    *,
    incumbent_full_vote_vector: Sequence[bool],
    candidate_full_vote_vector: Sequence[bool],
    incumbent_peer_vote_vector: Sequence[bool],
    candidate_peer_vote_vector: Sequence[bool],
) -> CoalitionContributionMetrics:
    incumbent_peer = tuple(map(bool, incumbent_peer_vote_vector))
    candidate_peer = tuple(map(bool, candidate_peer_vote_vector))
    if candidate_peer != incumbent_peer:
        raise AssertionError("fixed_peer_leave_one_out_state_changed")
    incumbent_full = tuple(map(bool, incumbent_full_vote_vector))
    candidate_full = tuple(map(bool, candidate_full_vote_vector))
    if not (
        len(incumbent_full) == len(candidate_full) == len(incumbent_peer)
    ):
        raise ValueError("coalition contribution vector lengths differ")

    incumbent_positive = sum(
        full and not peer
        for full, peer in zip(incumbent_full, incumbent_peer, strict=True)
    )
    incumbent_negative = sum(
        not full and peer
        for full, peer in zip(incumbent_full, incumbent_peer, strict=True)
    )
    candidate_positive = sum(
        full and not peer
        for full, peer in zip(candidate_full, candidate_peer, strict=True)
    )
    candidate_negative = sum(
        not full and peer
        for full, peer in zip(candidate_full, candidate_peer, strict=True)
    )
    incumbent_net = incumbent_positive - incumbent_negative
    candidate_net = candidate_positive - candidate_negative
    return CoalitionContributionMetrics(
        positive_pivotal_count=candidate_positive,
        negative_pivotal_count=candidate_negative,
        net_contribution=candidate_net,
        incumbent_positive_pivotal_count=incumbent_positive,
        incumbent_negative_pivotal_count=incumbent_negative,
        incumbent_net_contribution=incumbent_net,
        positive_pivotal_delta=candidate_positive - incumbent_positive,
        negative_pivotal_delta=candidate_negative - incumbent_negative,
        net_contribution_delta=candidate_net - incumbent_net,
    )


def deterministic_paired_bootstrap(
    per_example_deltas: Sequence[int],
    *,
    run_seed: int,
    team_state_version: int,
    update_index: int,
    target_agent_id: int,
    candidate_prompt_hash: str,
    active_lane: str,
    active_question_hashes: Sequence[str],
    replicates: int = BOOTSTRAP_REPLICATES,
    lower_quantile: float = BOOTSTRAP_LOWER_QUANTILE,
) -> RobustSupportMetrics:
    deltas = tuple(map(int, per_example_deltas))
    if not deltas:
        raise ValueError("rcru_bootstrap_requires_active_residuals")
    if replicates <= 0 or not 0 <= lower_quantile <= 1:
        raise ValueError("invalid RCRU bootstrap parameters")
    seed_payload = {
        "rcru_version": RCRU_VERSION,
        "run_seed": int(run_seed),
        "team_state_version": int(team_state_version),
        "update_index": int(update_index),
        "target_agent_id": int(target_agent_id),
        "candidate_prompt_hash": str(candidate_prompt_hash),
        "active_lane": str(active_lane),
        "active_question_hashes": sorted(map(str, active_question_hashes)),
    }
    seed_hash = hashlib.sha256(
        json.dumps(seed_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    generator = random.Random(int(seed_hash, 16))
    size = len(deltas)
    means = [
        sum(deltas[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(replicates)
    ]
    ordered = sorted(means)
    quantile_index = max(0, min(replicates - 1, math.ceil(lower_quantile * replicates) - 1))
    return RobustSupportMetrics(
        bootstrap_replicates=replicates,
        bootstrap_lower_quantile=lower_quantile,
        bootstrap_mean_delta=sum(means) / replicates,
        bootstrap_lcb=ordered[quantile_index],
        deterministic_seed_hash=seed_hash,
    )


def prompt_edit_metrics(
    parent_prompt: str,
    candidate_prompt: str,
    *,
    parent_normalized: str | None = None,
    parent_tokens: Sequence[str] | None = None,
) -> PromptEditMetrics:
    parent = (
        " ".join(str(parent_prompt).split())
        if parent_normalized is None else str(parent_normalized)
    )
    candidate = " ".join(str(candidate_prompt).split())
    token_pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
    parent_token_rows = (
        token_pattern.findall(parent)
        if parent_tokens is None else list(parent_tokens)
    )
    candidate_tokens = token_pattern.findall(candidate)
    inserted = deleted = replaced = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, parent_token_rows, candidate_tokens, autojunk=False
    ).get_opcodes():
        if tag == "insert":
            inserted += j2 - j1
        elif tag == "delete":
            deleted += i2 - i1
        elif tag == "replace":
            replaced += max(i2 - i1, j2 - j1)
    total = inserted + deleted + replaced
    denominator = max(len(parent_token_rows), len(candidate_tokens), 1)
    return PromptEditMetrics(
        parent_character_count=len(parent),
        candidate_character_count=len(candidate),
        character_growth=len(candidate) - len(parent),
        parent_token_count=len(parent_token_rows),
        candidate_token_count=len(candidate_tokens),
        inserted_token_count=inserted,
        deleted_token_count=deleted,
        replaced_token_count=replaced,
        total_edit_token_count=total,
        normalized_edit_ratio=total / denominator,
    )


def build_rcru_incumbent_cache(
    *,
    examples: Sequence[Any],
    active_profiles: Sequence[Sequence[Any]],
    target_agent_id: int,
    active_question_hashes: Sequence[str],
    active_lane: str,
    parent_prompt: str,
    normalize_answer: Callable[[str], str],
    match_answer: Callable[[str, str], bool],
    tie_break: str,
    run_seed: int,
) -> RCRUIncumbentCache:
    lane = ResponsibilityUtilityKind(active_lane)
    active_hashes = tuple(sorted(map(str, active_question_hashes)))
    states: dict[str, TeamVoteState] = {}
    peer_votes: list[bool] = []
    for index, example in enumerate(examples):
        state = build_team_vote_state(
            question_hash=example.question_hash,
            gold_answer=example.gold_answer,
            answers=[profile[index].answer for profile in active_profiles],
            valid_vector=[profile[index].valid for profile in active_profiles],
            normalize_answer=normalize_answer,
            match_answer=match_answer,
            tie_break=tie_break,
            seed=run_seed,
        )
        states[example.question_hash] = state
        peer_votes.append(
            build_peer_vote_context(state, target_agent_id).peer_margin > 0
        )
    token_pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
    parent_normalized = " ".join(str(parent_prompt).split())
    return RCRUIncumbentCache(
        question_hashes=tuple(states),
        incumbent_states=states,
        incumbent_full_vote_vector=tuple(
            state.vote_correct for state in states.values()
        ),
        peer_vote_vector=tuple(peer_votes),
        active_question_hashes=active_hashes,
        active_lane=lane.value,
        incumbent_utility_by_hash={
            question_hash: responsibility_utility(
                lane,
                target_correct=states[question_hash].team_correctness[
                    target_agent_id
                ],
                team_vote_correct=states[question_hash].vote_correct,
                plurality_margin=states[question_hash].plurality_margin,
            )
            for question_hash in active_hashes
        },
        parent_normalized=parent_normalized,
        parent_tokens=tuple(token_pattern.findall(parent_normalized)),
    )


def build_responsibility_contribution_metrics(
    *,
    examples: Sequence[Any],
    active_profiles: Sequence[Sequence[Any]],
    candidate_profile: Sequence[Any],
    target_agent_id: int,
    active_question_hashes: Sequence[str],
    active_lane: str,
    parent_prompt: str,
    candidate_prompt: str,
    candidate_prompt_hash: str,
    normalize_answer: Callable[[str], str],
    match_answer: Callable[[str, str], bool],
    tie_break: str,
    run_seed: int,
    team_state_version: int,
    update_index: int,
    incumbent_cache: RCRUIncumbentCache | None = None,
) -> ResponsibilityContributionMetrics:
    if len(active_profiles) != 5:
        raise ValueError("RCRU requires five active profiles")
    cache = incumbent_cache or build_rcru_incumbent_cache(
        examples=examples,
        active_profiles=active_profiles,
        target_agent_id=target_agent_id,
        active_question_hashes=active_question_hashes,
        active_lane=active_lane,
        parent_prompt=parent_prompt,
        normalize_answer=normalize_answer,
        match_answer=match_answer,
        tie_break=tie_break,
        run_seed=run_seed,
    )
    expected_hashes = tuple(example.question_hash for example in examples)
    if (
        cache.question_hashes != expected_hashes
        or cache.active_question_hashes
        != tuple(sorted(map(str, active_question_hashes)))
        or cache.active_lane != str(active_lane)
    ):
        raise ValueError("rcru_incumbent_cache_identity_mismatch")
    incumbent_states = cache.incumbent_states
    candidate_states: dict[str, TeamVoteState] = {}
    candidate_peer_votes: list[bool] = []
    for index, example in enumerate(examples):
        incumbent = incumbent_states[example.question_hash]
        incumbent_answers = list(incumbent.team_answers)
        incumbent_validity = list(incumbent.team_validity)
        candidate_answers = list(incumbent_answers)
        candidate_validity = list(incumbent_validity)
        candidate_answers[target_agent_id] = candidate_profile[index].answer
        candidate_validity[target_agent_id] = candidate_profile[index].valid
        candidate = build_team_vote_state(
            question_hash=example.question_hash,
            gold_answer=example.gold_answer,
            answers=candidate_answers,
            valid_vector=candidate_validity,
            normalize_answer=normalize_answer,
            match_answer=match_answer,
            tie_break=tie_break,
            seed=run_seed,
        )
        candidate_states[example.question_hash] = candidate
        candidate_peer_votes.append(
            build_peer_vote_context(candidate, target_agent_id).peer_margin > 0
        )
    utility = responsibility_utility_metrics(
        repair_lane=active_lane,
        active_question_hashes=active_question_hashes,
        incumbent_states=incumbent_states,
        candidate_states=candidate_states,
        target_agent_id=target_agent_id,
        incumbent_utility_by_hash=cache.incumbent_utility_by_hash,
    )
    coalition = coalition_contribution_metrics(
        incumbent_full_vote_vector=cache.incumbent_full_vote_vector,
        candidate_full_vote_vector=tuple(
            candidate_states[key].vote_correct for key in incumbent_states
        ),
        incumbent_peer_vote_vector=cache.peer_vote_vector,
        candidate_peer_vote_vector=candidate_peer_votes,
    )
    robust = deterministic_paired_bootstrap(
        utility.per_example_deltas,
        run_seed=run_seed,
        team_state_version=team_state_version,
        update_index=update_index,
        target_agent_id=target_agent_id,
        candidate_prompt_hash=candidate_prompt_hash,
        active_lane=active_lane,
        active_question_hashes=active_question_hashes,
    )
    return ResponsibilityContributionMetrics(
        utility=utility,
        coalition=coalition,
        robust_support=robust,
        edit=prompt_edit_metrics(
            parent_prompt,
            candidate_prompt,
            parent_normalized=cache.parent_normalized,
            parent_tokens=cache.parent_tokens,
        ),
    )


def evaluate_robust_contribution_constraints(
    candidate: Any,
    incumbent: Any,
) -> RobustContributionDecision:
    metrics = candidate.responsibility_contribution
    if metrics is None:
        raise ValueError("rcru_metrics_missing_for_candidate")
    target_gain = (
        candidate.competence.correct_count - incumbent.competence.correct_count
    )
    vote_gain = (
        candidate.team_outcome.vote_correct_count
        - incumbent.team_outcome.vote_correct_count
    )
    target_nonregression = target_gain >= 0
    vote_nonregression = vote_gain >= 0
    terminal_nonregression = (
        candidate.competence.terminal_invalid_count
        <= incumbent.competence.terminal_invalid_count
    )
    lane_nonregression = metrics.utility.utility_delta >= 0
    vote_or_lane_progress = vote_gain > 0 or metrics.utility.utility_delta > 0
    role_only = vote_gain == 0 and metrics.utility.utility_delta > 0
    minimum_support_passed = (
        not role_only or metrics.utility.positive_support_count >= 1
    )
    no_negative_support_passed = (
        not role_only or metrics.utility.negative_support_count == 0
    )
    bootstrap_passed = (
        not role_only or metrics.robust_support.bootstrap_lcb >= 0
    )
    checks = (
        ("target_regression", target_nonregression),
        ("team_vote_regression", vote_nonregression),
        ("terminal_invalid_regression", terminal_nonregression),
        ("active_lane_regression", lane_nonregression),
        ("no_vote_or_lane_progress", vote_or_lane_progress),
        ("insufficient_lane_support", minimum_support_passed),
        ("negative_lane_support", no_negative_support_passed),
        ("negative_lane_bootstrap_lcb", bootstrap_passed),
    )
    reasons = tuple(name for name, passed in checks if not passed)
    return RobustContributionDecision(
        passed=not reasons,
        hard_feasible=not reasons,
        target_nonregression_passed=target_nonregression,
        team_vote_nonregression_passed=vote_nonregression,
        terminal_invalid_nonregression_passed=terminal_nonregression,
        active_lane_nonregression_passed=lane_nonregression,
        vote_or_lane_progress_passed=vote_or_lane_progress,
        minimum_support_required=role_only,
        minimum_support_passed=minimum_support_passed,
        no_negative_support_required=role_only,
        no_negative_support_passed=no_negative_support_passed,
        bootstrap_guard_required=role_only,
        bootstrap_guard_passed=bootstrap_passed,
        vote_gain=vote_gain,
        target_gain=target_gain,
        lane_utility_delta=metrics.utility.utility_delta,
        net_contribution_delta=metrics.coalition.net_contribution_delta,
        positive_support_count=metrics.utility.positive_support_count,
        negative_support_count=metrics.utility.negative_support_count,
        bootstrap_lcb=metrics.robust_support.bootstrap_lcb,
        rejection_reasons=reasons,
    )


def responsibility_contribution_pareto_front(
    candidates: Sequence[Any],
) -> tuple[Any, ...]:
    rows = tuple(candidates)
    values = {}
    for row in rows:
        metrics = row.responsibility_contribution
        if metrics is None:
            raise ValueError("rcru_metrics_missing_for_candidate")
        values[row.prompt_hash] = (
            row.team_outcome.vote_correct_count,
            metrics.utility.utility_total,
            metrics.coalition.net_contribution,
        )
    frontier = []
    for row in rows:
        current = values[row.prompt_hash]
        dominated = any(
            other.prompt_hash != row.prompt_hash
            and all(
                left >= right
                for left, right in zip(
                    values[other.prompt_hash], current, strict=True
                )
            )
            and any(
                left > right
                for left, right in zip(
                    values[other.prompt_hash], current, strict=True
                )
            )
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return tuple(frontier)


def responsibility_contribution_pareto_front_numbers(
    candidates: Sequence[Any],
) -> dict[str, int]:
    remaining = {row.prompt_hash: row for row in candidates}
    fronts: dict[str, int] = {}
    number = 1
    while remaining:
        frontier = responsibility_contribution_pareto_front(
            tuple(remaining.values())
        )
        if not frontier:
            raise AssertionError("RCRU Pareto front construction made no progress")
        for row in frontier:
            fronts[row.prompt_hash] = number
            del remaining[row.prompt_hash]
        number += 1
    return fronts


def robust_contribution_key(candidate: Any, generation: int = 0) -> tuple:
    metrics = candidate.responsibility_contribution
    if metrics is None:
        raise ValueError("rcru_metrics_missing_for_candidate")
    return (
        candidate.team_outcome.vote_correct_count,
        metrics.utility.utility_total,
        metrics.coalition.net_contribution,
        metrics.robust_support.bootstrap_lcb,
        metrics.utility.positive_support_count,
        -metrics.utility.negative_support_count,
        -metrics.edit.total_edit_token_count,
        -metrics.edit.normalized_edit_ratio,
        -max(metrics.edit.character_growth, 0),
        -int(generation),
        candidate.prompt_hash,
    )
