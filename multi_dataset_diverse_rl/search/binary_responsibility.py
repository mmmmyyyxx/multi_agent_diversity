"""Exhaustive count-based responsibility; actual vote physics remain unchanged."""
from __future__ import annotations
from dataclasses import dataclass

from ..peer_state import build_team_vote_state, build_peer_vote_context
from ..responsibility import compute_member_aware_repair_opportunity
from .. import current_contract as versions
from .weighted_responsibility import LABELS, WEIGHTS, classify_error, POLICY
from .policies import ResponsibilitySignal
from .schemas import BenchmarkCapabilities, Diagnosis, SearchContractError, TeamStateSnapshot


def require_binary_plurality(capabilities: BenchmarkCapabilities) -> None:
    if not capabilities.binary_plurality_responsibility:
        raise SearchContractError("BINARY_PLURALITY_RESPONSIBILITY_CAPABILITY_REQUIRED")


@dataclass(frozen=True)
class BinaryPluralityObservation:
    input_id: str
    vote_classes: tuple[str, ...]
    validity: tuple[bool, ...]
    member_success: tuple[bool, ...]


def binary_plurality_snapshot(*, state_id: str, member_prompts: tuple[str, ...],
        observations: tuple[BinaryPluralityObservation, ...],
        capabilities: BenchmarkCapabilities) -> TeamStateSnapshot:
    """Bridge only valid, Boolean-scored classes into the existing voting algebra."""
    require_binary_plurality(capabilities)
    if len(member_prompts) != 5:
        raise SearchContractError("TEAM_REQUIRES_EXACTLY_FIVE_EQUAL_MEMBERS")
    states = []
    for row in observations:
        if (not row.input_id or len(row.vote_classes) != 5 or len(row.validity) != 5
                or len(row.member_success) != 5
                or any(not isinstance(x, str) for x in row.vote_classes)
                or any(type(x) is not bool for x in row.validity + row.member_success)):
            raise SearchContractError("BINARY_PLURALITY_OBSERVATION_INVALID")
        classes: dict[str, bool] = {}
        correct_classes = set()
        for answer, valid, correct in zip(row.vote_classes, row.validity, row.member_success, strict=True):
            if correct and not valid or valid and not answer:
                raise SearchContractError("BINARY_PLURALITY_OBSERVATION_INVALID")
            if valid:
                if answer in classes and classes[answer] != correct:
                    raise SearchContractError("EQUIVALENCE_RELATION_INCONSISTENT")
                classes[answer] = correct
                if correct:
                    correct_classes.add(answer)
        if len(correct_classes) > 1:
            raise SearchContractError("EQUIVALENCE_RELATION_INCONSISTENT")
        mapped = tuple("gold" if correct else "wrong:" + answer
            for answer, correct in zip(row.vote_classes, row.member_success, strict=True))
        states.append(build_team_vote_state(question_hash=row.input_id, gold_answer="gold",
            answers=mapped, valid_vector=row.validity))
    if len({row.question_hash for row in states}) != len(states):
        raise SearchContractError("BINARY_PLURALITY_DUPLICATE_INPUT")
    opportunities = {row.question_hash: tuple(compute_member_aware_repair_opportunity(
        team_state=row, peer_context=build_peer_vote_context(row, member)) for member in range(5))
        for row in states}
    return TeamStateSnapshot(state_id, member_prompts,
        diagnostics={"team_states": tuple(states), "opportunities": opportunities})


def observation_from_outputs(*, benchmark, item, member_outputs: tuple[str, ...],
                             gold: object) -> BinaryPluralityObservation:
    """Use benchmark Boolean scoring and its actual equivalence classes, without thresholds."""
    require_binary_plurality(benchmark.capabilities)
    from .scientific_aggregation import equivalence_classes, require_five
    require_five(member_outputs)
    parsed = tuple(benchmark.parse_member_output(raw, item) for raw in member_outputs)
    valid_indices = tuple(i for i, p in enumerate(parsed) if p.valid)
    relation = getattr(benchmark, "equivalent", lambda a, b: a == b)
    groups = equivalence_classes(tuple(parsed[i].answer for i in valid_indices), relation)
    classes = [""] * 5
    for group_index, group in enumerate(groups):
        for index in group:
            classes[valid_indices[index]] = f"class:{group_index}"
    scores = tuple(benchmark.score_member_output(p, gold) for p in parsed)
    if any(s not in (0, 1) for s in scores):
        raise SearchContractError("BINARY_PLURALITY_MEMBER_SUCCESS_NOT_BOOLEAN")
    return BinaryPluralityObservation(item.input_id, tuple(classes), tuple(p.valid for p in parsed),
                                      tuple(s == 1 for s in scores))


class BinaryPluralityResponsibilityAnalyzer:
    identity = versions.BINARY_PLURALITY_RESPONSIBILITY_VERSION

    def __init__(self, capabilities: BenchmarkCapabilities) -> None:
        require_binary_plurality(capabilities)
        self.capabilities = capabilities

    def analyze(self, state, history, *, responsibility_state=None):
        del history, responsibility_state
        require_binary_plurality(self.capabilities)
        if len(state.member_prompts) != 5:
            raise SearchContractError("TEAM_REQUIRES_EXACTLY_FIVE_EQUAL_MEMBERS")
        states = tuple(state.diagnostics["team_states"])
        if len({row.question_hash for row in states}) != len(states):
            raise SearchContractError('BINARY_PLURALITY_DUPLICATE_INPUT')
        # Retained observation key, now ALL member errors; never an eligibility filter.
        assigned = {member: tuple(state.diagnostics['opportunities'][row.question_hash][member]
            for row in states if not row.team_correctness[member]) for member in range(5)}
        margins = {row.question_hash: int(row.plurality_margin) for row in states}
        signals: dict[int, ResponsibilitySignal] = {}
        labels_by_member = {}
        uncovered_by_member = {}
        for member in range(len(state.member_prompts)):
            counts = {label: 0 for label in LABELS}
            labels = {}; uncovered = {}
            for row in states:
                lane, reason = classify_error(row, member)
                labels[row.question_hash] = (lane,) if lane else ()
                if lane:
                    counts[lane] += 1
                if reason:
                    uncovered[row.question_hash] = reason
            labels_by_member[member] = labels
            uncovered_by_member[member] = uncovered
            scores = {label: WEIGHTS[label] * count for label, count in counts.items()}
            # This descriptive lane never replaces the summed ranking value.
            lane = min(scores, key=lambda name: (-scores[name],
                                                 LABELS.index(name)))
            if sum(scores.values()) == 0:
                lane = "general"
            signals[member] = ResponsibilitySignal(
                member, counts["direct_flip"], counts["near_margin"],
                counts["coverage"], lane, counts['r3'], counts['r4'], len(uncovered),
            )
        return Diagnosis(
            responsibility=signals,
            benchmark_signals={"assigned": {key: tuple(value) for key, value in assigned.items()},
                               "margins": margins,
                               "responsibility_policy": POLICY,
                               "responsibility_labels": labels_by_member,
                               "uncovered_errors": uncovered_by_member},
        )
