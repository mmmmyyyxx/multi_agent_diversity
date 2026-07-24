from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Sequence

from .evaluation.fixed_probe import ProbeExample
from .peer_state import PeerVoteContext, TeamVoteState
from .responsibility import MemberAwareRepairOpportunity


DIAGNOSIS_AGGREGATION_VERSION = "peer_state_pattern_aggregation_v1"
ANSWER_ROLE_ENCODING_VERSION = "stable_wrong_cluster_roles_v1"
PATTERN_SELECTION_VERSION = "three_slot_lexicographic_v1"


class FailureFamily(str, Enum):
    INDIVIDUAL_ERROR = "individual_error"
    COVERAGE_FAILURE = "coverage_failure"
    CONVERSION_FAILURE = "conversion_failure"
    DOMINANT_WRONG = "dominant_wrong"
    MEMBER_COMPETENCE = "member_competence"
    PRESERVATION = "preservation"


@dataclass(frozen=True)
class FailurePatternKey:
    case_family: str
    target_status: str
    team_vote_status: str
    target_answer_role: str
    gold_vote_count: int
    largest_wrong_vote_count: int
    plurality_margin: int
    peer_gold_vote_count: int
    peer_largest_wrong_vote_count: int
    peer_margin: int
    direct_vote_fix: bool
    dominant_wrong_member: bool
    unique_correct: bool
    pivotal_correct: bool


@dataclass(frozen=True)
class AggregatedFailurePattern:
    pattern_id: str
    key: FailurePatternKey
    case_count: int
    assigned_case_count: int
    direct_vote_fix_count: int
    dominant_wrong_count: int
    mean_oracle_soft_utility_gain: float
    max_oracle_soft_utility_gain: float
    max_owner_age: int
    repair_goal: str
    represented_question_hashes: tuple[str, ...]


@dataclass(frozen=True)
class CompactEvidenceCase:
    case_id: str
    pattern_id: str
    case_family: str
    question_hash: str
    question: str
    gold_answer: str
    target_current_answer: str
    answer_role_signature: tuple[str, ...]
    target_answer_role: str
    gold_vote_count: int
    largest_wrong_vote_count: int
    plurality_margin: int
    peer_gold_vote_count: int
    peer_largest_wrong_vote_count: int
    peer_margin: int
    direct_vote_fix: bool
    dominant_wrong_member: bool
    unique_correct: bool
    pivotal_correct: bool
    repair_goal: str


@dataclass(frozen=True)
class _PatternCase:
    key: FailurePatternKey
    question_hash: str
    question: str
    gold_answer: str
    target_current_answer: str
    answer_role_signature: tuple[str, ...]
    assigned: bool
    oracle_soft_utility_gain: float
    owner_age: int
    repair_goal: str


@dataclass(frozen=True)
class DiagnosisAggregation:
    full_probe_case_count: int
    available_patterns: tuple[AggregatedFailurePattern, ...]
    selected_patterns: tuple[AggregatedFailurePattern, ...]
    evidence_cases: tuple[CompactEvidenceCase, ...]


def _stable_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def answer_role_signature(state: TeamVoteState) -> tuple[str, ...]:
    """Encode answer clusters without using agent identity as a tie breaker."""
    ranked_wrong = sorted(
        state.wrong_vote_histogram,
        key=lambda row: (-row[1], _stable_hash(row[0])),
    )
    wrong_roles = {
        answer: f"W{min(index + 1, 3)}"
        for index, (answer, _count) in enumerate(ranked_wrong)
    }
    roles = []
    for answer, valid, correct in zip(
        state.team_answers,
        state.team_validity,
        state.team_correctness,
        strict=True,
    ):
        if not valid:
            roles.append("I")
        elif correct:
            roles.append("G")
        else:
            roles.append(wrong_roles[answer])
    return tuple(roles)


def _pattern_id(key: FailurePatternKey) -> str:
    canonical = json.dumps(
        asdict(key), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return _stable_hash(f"{DIAGNOSIS_AGGREGATION_VERSION}:{canonical}")[:20]


def _case_id(pattern_id: str, question_hash: str) -> str:
    return _stable_hash(f"{pattern_id}:{question_hash}")[:20]


def _status(opportunity: MemberAwareRepairOpportunity) -> str:
    if opportunity.current_invalid:
        return "invalid"
    return "correct" if opportunity.current_correct else "wrong"


def _vote_status(state: TeamVoteState) -> str:
    if state.vote_correct:
        return "correct"
    return "tie_abstain" if state.top_tie else "wrong"


def _repair_goal(family: FailureFamily, state: TeamVoteState) -> str:
    return {
        FailureFamily.INDIVIDUAL_ERROR: "repair_target_reasoning",
        FailureFamily.COVERAGE_FAILURE: "introduce_first_gold_vote",
        FailureFamily.CONVERSION_FAILURE: "convert_existing_gold_coverage",
        FailureFamily.DOMINANT_WRONG: "exit_dominant_wrong_cluster",
        FailureFamily.MEMBER_COMPETENCE: "improve_target_member_competence",
        FailureFamily.PRESERVATION: "preserve_unique_or_pivotal_correctness",
    }[family]


def _families(
    state: TeamVoteState,
    opportunity: MemberAwareRepairOpportunity,
) -> tuple[FailureFamily, ...]:
    rows: list[FailureFamily] = []
    if opportunity.member_error:
        rows.append(FailureFamily.INDIVIDUAL_ERROR)
    if state.gold_vote_count == 0:
        rows.append(FailureFamily.COVERAGE_FAILURE)
    if not state.vote_correct and state.gold_vote_count > 0:
        rows.append(FailureFamily.CONVERSION_FAILURE)
    if opportunity.dominant_wrong_member:
        rows.append(FailureFamily.DOMINANT_WRONG)
    if opportunity.member_error and opportunity.improvement_need > 0:
        rows.append(FailureFamily.MEMBER_COMPETENCE)
    if opportunity.unique_correct or opportunity.pivotal_correct:
        rows.append(FailureFamily.PRESERVATION)
    return tuple(rows)


def _build_pattern_cases(
    *,
    target_agent_id: int,
    examples: Sequence[ProbeExample],
    states: Sequence[TeamVoteState],
    peer_contexts: Mapping[str, Mapping[int, PeerVoteContext]],
    opportunities: Mapping[str, Sequence[MemberAwareRepairOpportunity]],
    assigned_hashes: set[str],
    owner_age_by_question: Mapping[str, int],
) -> list[_PatternCase]:
    example_by_hash = {row.question_hash: row for row in examples}
    opportunity_by_hash = {
        question_hash: tuple(rows)[target_agent_id]
        for question_hash, rows in opportunities.items()
    }
    result: list[_PatternCase] = []
    for state in states:
        opportunity = opportunity_by_hash[state.question_hash]
        peer = peer_contexts[state.question_hash][target_agent_id]
        example = example_by_hash[state.question_hash]
        roles = answer_role_signature(state)
        for family in _families(state, opportunity):
            key = FailurePatternKey(
                case_family=family.value,
                target_status=_status(opportunity),
                team_vote_status=_vote_status(state),
                target_answer_role=roles[target_agent_id],
                gold_vote_count=state.gold_vote_count,
                largest_wrong_vote_count=state.largest_wrong_vote_count,
                plurality_margin=state.plurality_margin,
                peer_gold_vote_count=peer.peer_gold_vote_count,
                peer_largest_wrong_vote_count=peer.peer_largest_wrong_vote_count,
                peer_margin=peer.peer_margin,
                direct_vote_fix=opportunity.direct_vote_fix,
                dominant_wrong_member=opportunity.dominant_wrong_member,
                unique_correct=opportunity.unique_correct,
                pivotal_correct=opportunity.pivotal_correct,
            )
            result.append(_PatternCase(
                key=key,
                question_hash=state.question_hash,
                question=example.question,
                gold_answer=example.gold_answer,
                target_current_answer=state.team_answers[target_agent_id],
                answer_role_signature=roles,
                assigned=state.question_hash in assigned_hashes,
                oracle_soft_utility_gain=opportunity.oracle_soft_utility_gain,
                owner_age=int(owner_age_by_question.get(state.question_hash, 0)),
                repair_goal=_repair_goal(family, state),
            ))
    return result


def aggregate_patterns(
    rows: Sequence[_PatternCase],
) -> tuple[tuple[AggregatedFailurePattern, ...], dict[str, tuple[_PatternCase, ...]]]:
    grouped: dict[FailurePatternKey, list[_PatternCase]] = defaultdict(list)
    for row in rows:
        grouped[row.key].append(row)
    patterns: list[AggregatedFailurePattern] = []
    members: dict[str, tuple[_PatternCase, ...]] = {}
    for key, group in grouped.items():
        pattern_id = _pattern_id(key)
        ordered = tuple(sorted(group, key=lambda row: row.question_hash))
        members[pattern_id] = ordered
        patterns.append(AggregatedFailurePattern(
            pattern_id=pattern_id,
            key=key,
            case_count=len(ordered),
            assigned_case_count=sum(row.assigned for row in ordered),
            direct_vote_fix_count=sum(row.key.direct_vote_fix for row in ordered),
            dominant_wrong_count=sum(row.key.dominant_wrong_member for row in ordered),
            mean_oracle_soft_utility_gain=(
                sum(row.oracle_soft_utility_gain for row in ordered) / len(ordered)
            ),
            max_oracle_soft_utility_gain=max(
                row.oracle_soft_utility_gain for row in ordered
            ),
            max_owner_age=max(row.owner_age for row in ordered),
            repair_goal=ordered[0].repair_goal,
            represented_question_hashes=tuple(row.question_hash for row in ordered),
        ))
    return tuple(sorted(patterns, key=lambda row: row.pattern_id)), members


def _descending(pattern: AggregatedFailurePattern) -> tuple:
    return (
        pattern.assigned_case_count,
        pattern.direct_vote_fix_count,
        pattern.max_oracle_soft_utility_gain,
        pattern.max_owner_age,
        pattern.case_count,
        pattern.pattern_id,
    )


def _select_best(
    patterns: Sequence[AggregatedFailurePattern],
    selected_ids: set[str],
    key,
) -> AggregatedFailurePattern | None:
    available = [row for row in patterns if row.pattern_id not in selected_ids]
    return max(available, key=key, default=None)


def select_patterns(
    patterns: Sequence[AggregatedFailurePattern],
    *,
    context_policy: str,
    target_improvement_need: int,
    max_patterns: int,
) -> tuple[AggregatedFailurePattern, ...]:
    if max_patterns < 0:
        raise ValueError("max_patterns cannot be negative")
    selected: list[AggregatedFailurePattern] = []
    selected_ids: set[str] = set()

    def add(families: set[str], key) -> None:
        row = _select_best(
            [pattern for pattern in patterns if pattern.key.case_family in families],
            selected_ids,
            key,
        )
        if row is not None and len(selected) < max_patterns:
            selected.append(row)
            selected_ids.add(row.pattern_id)

    individual_key = lambda row: (
        row.case_count,
        row.direct_vote_fix_count,
        row.max_oracle_soft_utility_gain,
        row.pattern_id,
    )
    preservation_key = lambda row: (
        int(row.key.pivotal_correct),
        int(row.key.unique_correct),
        -row.key.plurality_margin if row.key.plurality_margin > 0 else -10**9,
        row.case_count,
        row.pattern_id,
    )
    if context_policy == "generic_accuracy":
        add({FailureFamily.INDIVIDUAL_ERROR.value}, individual_key)
        add({FailureFamily.INDIVIDUAL_ERROR.value}, individual_key)
        add({FailureFamily.PRESERVATION.value}, preservation_key)
    elif context_policy == "generic_peer_state":
        add({FailureFamily.COVERAGE_FAILURE.value}, _descending)
        add(
            {
                FailureFamily.CONVERSION_FAILURE.value,
                FailureFamily.DOMINANT_WRONG.value,
            },
            _descending,
        )
        add({FailureFamily.PRESERVATION.value}, preservation_key)
    elif context_policy == "member_aware_responsibility_conditioned":
        add(
            {
                FailureFamily.COVERAGE_FAILURE.value,
                FailureFamily.CONVERSION_FAILURE.value,
                FailureFamily.DOMINANT_WRONG.value,
            },
            _descending,
        )
        add(
            {FailureFamily.MEMBER_COMPETENCE.value},
            lambda row: (
                target_improvement_need,
                row.direct_vote_fix_count,
                row.case_count,
                row.max_oracle_soft_utility_gain,
                row.pattern_id,
            ),
        )
        add({FailureFamily.PRESERVATION.value}, preservation_key)
    else:
        raise ValueError(f"Unsupported TCS context policy: {context_policy}")

    while len(selected) < min(max_patterns, len(patterns)):
        row = _select_best(patterns, selected_ids, _descending)
        if row is None:
            break
        selected.append(row)
        selected_ids.add(row.pattern_id)
    return tuple(selected)


def select_representative_cases(
    selected_patterns: Sequence[AggregatedFailurePattern],
    pattern_members: Mapping[str, Sequence[_PatternCase]],
    *,
    max_cases: int,
) -> tuple[CompactEvidenceCase, ...]:
    selected: list[CompactEvidenceCase] = []
    used_questions: set[str] = set()
    for pattern in selected_patterns:
        if len(selected) >= max_cases:
            break
        rows = sorted(
            pattern_members[pattern.pattern_id],
            key=lambda row: (
                -int(row.assigned),
                -int(row.key.direct_vote_fix),
                -row.oracle_soft_utility_gain,
                -row.owner_age,
                abs(row.key.plurality_margin),
                row.question_hash,
            ),
        )
        representative = next(
            (row for row in rows if row.question_hash not in used_questions),
            None,
        )
        if representative is None:
            continue
        used_questions.add(representative.question_hash)
        selected.append(CompactEvidenceCase(
            case_id=_case_id(pattern.pattern_id, representative.question_hash),
            pattern_id=pattern.pattern_id,
            case_family=pattern.key.case_family,
            question_hash=representative.question_hash,
            question=representative.question,
            gold_answer=representative.gold_answer,
            target_current_answer=representative.target_current_answer,
            answer_role_signature=representative.answer_role_signature,
            target_answer_role=pattern.key.target_answer_role,
            gold_vote_count=pattern.key.gold_vote_count,
            largest_wrong_vote_count=pattern.key.largest_wrong_vote_count,
            plurality_margin=pattern.key.plurality_margin,
            peer_gold_vote_count=pattern.key.peer_gold_vote_count,
            peer_largest_wrong_vote_count=pattern.key.peer_largest_wrong_vote_count,
            peer_margin=pattern.key.peer_margin,
            direct_vote_fix=pattern.key.direct_vote_fix,
            dominant_wrong_member=pattern.key.dominant_wrong_member,
            unique_correct=pattern.key.unique_correct,
            pivotal_correct=pattern.key.pivotal_correct,
            repair_goal=pattern.repair_goal,
        ))
    return tuple(selected)


def aggregate_probe_diagnosis(
    *,
    target_agent_id: int,
    examples: Sequence[ProbeExample],
    states: Sequence[TeamVoteState],
    peer_contexts: Mapping[str, Mapping[int, PeerVoteContext]],
    opportunities: Mapping[str, Sequence[MemberAwareRepairOpportunity]],
    assigned_hashes: set[str],
    owner_age_by_question: Mapping[str, int],
    context_policy: str,
    target_improvement_need: int,
    max_patterns: int = 3,
    max_cases: int = 3,
) -> DiagnosisAggregation:
    if len(examples) != len(states):
        raise ValueError("full fixed probe examples and states must have equal length")
    rows = _build_pattern_cases(
        target_agent_id=target_agent_id,
        examples=examples,
        states=states,
        peer_contexts=peer_contexts,
        opportunities=opportunities,
        assigned_hashes=assigned_hashes,
        owner_age_by_question=owner_age_by_question,
    )
    patterns, members = aggregate_patterns(rows)
    selected = select_patterns(
        patterns,
        context_policy=context_policy,
        target_improvement_need=target_improvement_need,
        max_patterns=max_patterns,
    )
    evidence = select_representative_cases(
        selected, members, max_cases=max_cases,
    )
    return DiagnosisAggregation(
        full_probe_case_count=len(states),
        available_patterns=patterns,
        selected_patterns=selected,
        evidence_cases=evidence,
    )
