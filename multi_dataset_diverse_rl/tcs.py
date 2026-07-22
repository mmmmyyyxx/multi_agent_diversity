from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AccuracyCase:
    question_hash: str
    question: str
    gold_answer: str
    target_current_answer: str
    target_current_correct: bool
    target_current_invalid: bool


@dataclass(frozen=True)
class PeerStateCase:
    question_hash: str
    question: str
    gold_answer: str
    target_current_answer: str
    team_G: int
    team_H: int
    team_M: int
    team_wrong_histogram: tuple[tuple[str, int], ...]
    peer_G: int
    peer_H: int
    peer_M: int
    peer_wrong_histogram: tuple[tuple[str, int], ...]
    direct_vote_fix: bool
    oracle_soft_utility_gain: float
    dominant_wrong_member: bool


@dataclass(frozen=True)
class ResponsibilityCase(PeerStateCase):
    responsibility_reason: str
    owner_age: int = 0


@dataclass(frozen=True)
class PreservationCase:
    question_hash: str
    question: str
    gold_answer: str
    target_current_answer: str
    unique_correct: bool
    pivotal_correct: bool
    team_margin: int
    peer_G: int
    peer_H: int
    peer_M: int
    peer_wrong_histogram: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class RepresentativeCase:
    question_hash: str
    question: str
    gold_answer: str
    target_current_answer: str
    target_current_correct: bool
    target_current_invalid: bool


@dataclass(frozen=True)
class AccuracyProposalContext:
    target_agent_id: int
    parent_prompt: str
    parent_prompt_hash: str
    error_cases: tuple[AccuracyCase, ...]
    protection_cases: tuple[AccuracyCase, ...]
    previous_accuracy_summary: str


@dataclass(frozen=True)
class PeerStateProposalContext:
    target_agent_id: int
    parent_prompt: str
    parent_prompt_hash: str
    coverage_cases: tuple[PeerStateCase, ...]
    conversion_cases: tuple[PeerStateCase, ...]
    preservation_cases: tuple[PreservationCase, ...]
    representative_cases: tuple[RepresentativeCase, ...]
    previous_vote_competence_summary: str


@dataclass(frozen=True)
class ResponsibilityProposalContext:
    target_agent_id: int
    parent_prompt: str
    parent_prompt_hash: str
    assigned_coverage_cases: tuple[ResponsibilityCase, ...]
    assigned_conversion_cases: tuple[ResponsibilityCase, ...]
    preservation_cases: tuple[PreservationCase, ...]
    representative_cases: tuple[RepresentativeCase, ...]
    responsibility_summary: str
    assigned_repair_history: str


AnyProposalContext = AccuracyProposalContext | PeerStateProposalContext | ResponsibilityProposalContext


@dataclass(frozen=True)
class TeacherProposal:
    target_failure_mechanism: str
    repair_procedure: str
    preservation_rule: str
    expected_effect: str


@dataclass(frozen=True)
class CriticDecision:
    approved: bool
    score: float
    feedback: str
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class StudentCandidate:
    candidate_prompt: str
    target_failure_mechanism: str
    repair_procedure: str
    preservation_rule: str
    expected_effect: str


@dataclass(frozen=True)
class TCSContextLimits:
    assigned_coverage: int = 6
    assigned_conversion: int = 6
    preservation: int = 6
    representative: int = 6
    max_chars: int = 24000


@dataclass(frozen=True)
class TCSContextDiagnostics:
    available_cases: dict[str, int]
    selected_cases: dict[str, int]
    truncated_cases: dict[str, int]
    context_characters: int
    estimated_input_tokens: int


def _payload(context: AnyProposalContext) -> str:
    return json.dumps(asdict(context), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def limit_proposal_context(
    context: AnyProposalContext,
    limits: TCSContextLimits,
) -> tuple[AnyProposalContext, TCSContextDiagnostics]:
    if limits.max_chars <= 0:
        raise ValueError("tcs_context_max_chars must be positive")
    if isinstance(context, AccuracyProposalContext):
        available = {
            "errors": len(context.error_cases),
            "protection": len(context.protection_cases),
        }
        bounded = replace(
            context,
            error_cases=tuple(sorted(context.error_cases, key=lambda row: row.question_hash))[
                : limits.assigned_conversion
            ],
            protection_cases=tuple(sorted(context.protection_cases, key=lambda row: row.question_hash))[
                : limits.preservation
            ],
        )
        fields = ("error_cases", "protection_cases")
        selected_counts = lambda value: {
            "errors": len(value.error_cases),
            "protection": len(value.protection_cases),
        }
    else:
        coverage_field = (
            "assigned_coverage_cases"
            if isinstance(context, ResponsibilityProposalContext)
            else "coverage_cases"
        )
        conversion_field = (
            "assigned_conversion_cases"
            if isinstance(context, ResponsibilityProposalContext)
            else "conversion_cases"
        )
        coverage_rows = getattr(context, coverage_field)
        conversion_rows = getattr(context, conversion_field)
        available = {
            "coverage": len(coverage_rows),
            "conversion": len(conversion_rows),
            "preservation": len(context.preservation_cases),
            "representative": len(context.representative_cases),
        }
        coverage_key = (
            (lambda row: (-row.owner_age, -row.oracle_soft_utility_gain, row.question_hash))
            if isinstance(context, ResponsibilityProposalContext)
            else (lambda row: (-row.oracle_soft_utility_gain, row.question_hash))
        )
        bounded = replace(
            context,
            **{
                coverage_field: tuple(sorted(coverage_rows, key=coverage_key)[: limits.assigned_coverage]),
                conversion_field: tuple(sorted(
                    conversion_rows,
                    key=lambda row: (
                        -int(row.direct_vote_fix), -row.oracle_soft_utility_gain,
                        abs(row.team_M), row.question_hash,
                    ),
                )[: limits.assigned_conversion]),
                "preservation_cases": tuple(sorted(
                    context.preservation_cases,
                    key=lambda row: (
                        -int(row.pivotal_correct), -int(row.unique_correct),
                        row.team_margin, row.question_hash,
                    ),
                )[: limits.preservation]),
                "representative_cases": tuple(sorted(
                    context.representative_cases, key=lambda row: row.question_hash,
                )[: limits.representative]),
            },
        )
        fields = ("representative_cases", coverage_field, conversion_field, "preservation_cases")
        selected_counts = lambda value: {
            "coverage": len(getattr(value, coverage_field)),
            "conversion": len(getattr(value, conversion_field)),
            "preservation": len(value.preservation_cases),
            "representative": len(value.representative_cases),
        }
    while len(_payload(bounded)) > limits.max_chars:
        removable = next((field for field in fields if getattr(bounded, field)), None)
        if removable is None:
            raise ValueError("TCS context metadata and parent prompt exceed tcs_context_max_chars")
        bounded = replace(bounded, **{removable: getattr(bounded, removable)[:-1]})
    selected = selected_counts(bounded)
    characters = len(_payload(bounded))
    return bounded, TCSContextDiagnostics(
        available_cases=available,
        selected_cases=selected,
        truncated_cases={key: available[key] - selected[key] for key in available},
        context_characters=characters,
        estimated_input_tokens=(characters + 3) // 4,
    )


def build_teacher_request(context: AnyProposalContext) -> str:
    if isinstance(context, AccuracyProposalContext):
        objective = "Repair the target Student's generalizable individual reasoning failure."
    elif isinstance(context, PeerStateProposalContext):
        objective = "Repair a peer-state failure using G, H, M and preservation evidence."
    elif isinstance(context, ResponsibilityProposalContext):
        objective = "Repair this Student's assigned residual cases while preserving unique and pivotal correct cases."
    else:
        raise TypeError(f"Unsupported ProposalContext: {type(context).__name__}")
    schema = {
        "target_failure_mechanism": "task-general failure mechanism",
        "repair_procedure": "executable decision procedure",
        "preservation_rule": "rule protecting existing competence",
        "expected_effect": "expected effect on the supplied cases",
    }
    return (
        f"{objective} Do not memorize answers, assign a preset role, copy a peer procedure, "
        "or propose generic chain-of-thought. Diagnose a real defect in the parent prompt. Return strict JSON "
        f"matching this schema: {json.dumps(schema)}\nProposalContext:\n{_payload(context)}"
    )


def build_critic_request(context: AnyProposalContext, teacher_proposal: TeacherProposal) -> str:
    if isinstance(context, AccuracyProposalContext):
        policy_checks = "individual-error diagnosis and individual competence preservation"
    elif isinstance(context, PeerStateProposalContext):
        policy_checks = "team/peer-state interpretation and preservation evidence"
    elif isinstance(context, ResponsibilityProposalContext):
        policy_checks = "assigned residual targeting, team/peer-state interpretation, and preservation evidence"
    else:
        raise TypeError(f"Unsupported ProposalContext: {type(context).__name__}")
    return (
        f"Audit the Teacher proposal against the same ProposalContext. Check {policy_checks}, "
        "generic-CoT behavior, peer copying, answer memorization, the parent "
        "prompt's actual defect, and whether the procedure is executable and task-general. Return strict JSON with "
        "approved (boolean), score (0..1), feedback (string), and rejection_reasons (array of strings).\n"
        f"ProposalContext:\n{_payload(context)}\nTeacherProposal:\n"
        f"{json.dumps(asdict(teacher_proposal), ensure_ascii=False, sort_keys=True)}"
    )


def build_student_request(
    context: AnyProposalContext,
    approved_proposal: TeacherProposal,
    candidate_count: int,
) -> str:
    if isinstance(context, AccuracyProposalContext):
        policy_instruction = "address the supplied individual-error evidence"
    elif isinstance(context, PeerStateProposalContext):
        policy_instruction = "address the supplied peer-state evidence"
    elif isinstance(context, ResponsibilityProposalContext):
        policy_instruction = "address the supplied assigned responsibilities"
    else:
        raise TypeError(f"Unsupported ProposalContext: {type(context).__name__}")
    return (
        f"Generate exactly {candidate_count} task-general prompt candidates. Each candidate must implement the "
        f"approved repair, preserve protected cases, and {policy_instruction}. Return strict JSON "
        "with a candidates array; every item must contain candidate_prompt, target_failure_mechanism, "
        "repair_procedure, preservation_rule, and expected_effect.\n"
        f"ProposalContext:\n{_payload(context)}\nApprovedTeacherProposal:\n"
        f"{json.dumps(asdict(approved_proposal), ensure_ascii=False, sort_keys=True)}"
    )


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def parse_teacher_proposal(payload: Mapping[str, Any]) -> TeacherProposal:
    return TeacherProposal(
        target_failure_mechanism=_required_text(payload, "target_failure_mechanism"),
        repair_procedure=_required_text(payload, "repair_procedure"),
        preservation_rule=_required_text(payload, "preservation_rule"),
        expected_effect=_required_text(payload, "expected_effect"),
    )


def parse_critic_decision(payload: Mapping[str, Any], approval_threshold: float) -> CriticDecision:
    approved = payload["approved"]
    score = payload["score"]
    reasons = payload["rejection_reasons"]
    if type(approved) is not bool:
        raise ValueError("critic approved must be a JSON boolean")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("critic score must be numeric")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise ValueError("critic rejection_reasons must be an array of strings")
    feedback = _required_text(payload, "feedback")
    numeric_score = float(score)
    effective_approval = bool(approved is True and numeric_score >= approval_threshold)
    if approved and not effective_approval and not reasons:
        reasons = ["score_below_threshold"]
    return CriticDecision(
        approved=effective_approval,
        score=numeric_score,
        feedback=feedback,
        rejection_reasons=tuple(reasons),
    )


def parse_student_candidates(
    payload: Mapping[str, Any], *, expected_count: int,
) -> tuple[StudentCandidate, ...]:
    rows = payload["candidates"]
    if not isinstance(rows, list):
        raise ValueError("student candidates must be an array")
    if len(rows) != expected_count:
        raise ValueError(
            f"student candidate count must equal {expected_count}, received {len(rows)}"
        )
    candidates = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each student candidate must be an object")
        candidates.append(StudentCandidate(
            candidate_prompt=_required_text(row, "candidate_prompt"),
            target_failure_mechanism=_required_text(row, "target_failure_mechanism"),
            repair_procedure=_required_text(row, "repair_procedure"),
            preservation_rule=_required_text(row, "preservation_rule"),
            expected_effect=_required_text(row, "expected_effect"),
        ))
    return tuple(candidates)
