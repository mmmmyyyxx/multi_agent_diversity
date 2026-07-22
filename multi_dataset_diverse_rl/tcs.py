from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ResponsibilityCase:
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
class ProposalContext:
    target_agent_id: int
    context_policy: str
    parent_prompt: str
    parent_prompt_hash: str
    assigned_coverage_cases: tuple[ResponsibilityCase, ...]
    assigned_conversion_cases: tuple[ResponsibilityCase, ...]
    preservation_cases: tuple[PreservationCase, ...]
    representative_cases: tuple[RepresentativeCase, ...]
    responsibility_summary: str
    previous_update_summary: str


@dataclass(frozen=True)
class TeacherProposal:
    target_failure_mechanism: str
    repair_procedure: str
    preservation_rule: str
    expected_responsibility_effect: str


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
    expected_responsibility_effect: str


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


def _payload(context: ProposalContext) -> str:
    return json.dumps(asdict(context), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def limit_proposal_context(
    context: ProposalContext,
    limits: TCSContextLimits,
) -> tuple[ProposalContext, TCSContextDiagnostics]:
    if limits.max_chars <= 0:
        raise ValueError("tcs_context_max_chars must be positive")
    available = {
        "assigned_coverage": len(context.assigned_coverage_cases),
        "assigned_conversion": len(context.assigned_conversion_cases),
        "preservation": len(context.preservation_cases),
        "representative": len(context.representative_cases),
    }
    bounded = replace(
        context,
        assigned_coverage_cases=tuple(sorted(
            context.assigned_coverage_cases,
            key=lambda row: (-row.owner_age, -row.oracle_soft_utility_gain, row.question_hash),
        )[: limits.assigned_coverage]),
        assigned_conversion_cases=tuple(sorted(
            context.assigned_conversion_cases,
            key=lambda row: (
                -int(row.direct_vote_fix), -row.oracle_soft_utility_gain, abs(row.team_M), row.question_hash,
            ),
        )[: limits.assigned_conversion]),
        preservation_cases=tuple(sorted(
            context.preservation_cases,
            key=lambda row: (-int(row.pivotal_correct), -int(row.unique_correct), row.team_margin, row.question_hash),
        )[: limits.preservation]),
        representative_cases=tuple(sorted(
            context.representative_cases, key=lambda row: row.question_hash,
        )[: limits.representative]),
    )
    fields = (
        "representative_cases", "assigned_coverage_cases",
        "assigned_conversion_cases", "preservation_cases",
    )
    while len(_payload(bounded)) > limits.max_chars:
        removable = next((field for field in fields if getattr(bounded, field)), None)
        if removable is None:
            raise ValueError("TCS context metadata and parent prompt exceed tcs_context_max_chars")
        bounded = replace(bounded, **{removable: getattr(bounded, removable)[:-1]})
    selected = {
        "assigned_coverage": len(bounded.assigned_coverage_cases),
        "assigned_conversion": len(bounded.assigned_conversion_cases),
        "preservation": len(bounded.preservation_cases),
        "representative": len(bounded.representative_cases),
    }
    characters = len(_payload(bounded))
    return bounded, TCSContextDiagnostics(
        available_cases=available,
        selected_cases=selected,
        truncated_cases={key: available[key] - selected[key] for key in available},
        context_characters=characters,
        estimated_input_tokens=(characters + 3) // 4,
    )


def build_teacher_request(context: ProposalContext) -> str:
    objectives = {
        "generic_accuracy": "Repair the target Student's generalizable individual reasoning failure.",
        "generic_peer_state": (
            "Repair a peer-state failure using G, H, M and preservation evidence without claiming residual ownership."
        ),
        "responsibility_conditioned": (
            "Repair the residual cases assigned to this Student while preserving its unique and pivotal correct cases."
        ),
    }
    if context.context_policy not in objectives:
        raise ValueError(f"Unsupported TCS context policy: {context.context_policy}")
    schema = {
        "target_failure_mechanism": "task-general failure mechanism",
        "repair_procedure": "executable decision procedure",
        "preservation_rule": "rule protecting existing competence",
        "expected_responsibility_effect": "expected effect on the supplied cases",
    }
    return (
        f"{objectives[context.context_policy]} Do not memorize answers, assign a preset role, copy a peer procedure, "
        "or propose generic chain-of-thought. Diagnose a real defect in the parent prompt. Return strict JSON "
        f"matching this schema: {json.dumps(schema)}\nProposalContext:\n{_payload(context)}"
    )


def build_critic_request(context: ProposalContext, teacher_proposal: TeacherProposal) -> str:
    policy_checks = {
        "generic_accuracy": "individual-error diagnosis and preservation evidence",
        "generic_peer_state": "team/peer-state interpretation and preservation evidence",
        "responsibility_conditioned": (
            "assigned residual targeting, team/peer-state interpretation, and preservation evidence"
        ),
    }
    if context.context_policy not in policy_checks:
        raise ValueError(f"Unsupported TCS context policy: {context.context_policy}")
    return (
        f"Audit the Teacher proposal against the same ProposalContext. Check {policy_checks[context.context_policy]}, "
        "generic-CoT behavior, peer copying, answer memorization, the parent "
        "prompt's actual defect, and whether the procedure is executable and task-general. Return strict JSON with "
        "approved (boolean), score (0..1), feedback (string), and rejection_reasons (array of strings).\n"
        f"ProposalContext:\n{_payload(context)}\nTeacherProposal:\n"
        f"{json.dumps(asdict(teacher_proposal), ensure_ascii=False, sort_keys=True)}"
    )


def build_student_request(
    context: ProposalContext,
    approved_proposal: TeacherProposal,
    candidate_count: int,
) -> str:
    policy_instruction = {
        "generic_accuracy": "address the supplied individual-error evidence",
        "generic_peer_state": "address the supplied peer-state evidence without claiming residual ownership",
        "responsibility_conditioned": "address the supplied assigned responsibilities",
    }
    if context.context_policy not in policy_instruction:
        raise ValueError(f"Unsupported TCS context policy: {context.context_policy}")
    return (
        f"Generate exactly {candidate_count} task-general prompt candidates. Each candidate must implement the "
        f"approved repair, preserve protected cases, and {policy_instruction[context.context_policy]}. Return strict JSON "
        "with a candidates array; every item must contain candidate_prompt, target_failure_mechanism, "
        "repair_procedure, preservation_rule, and expected_responsibility_effect.\n"
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
        expected_responsibility_effect=_required_text(payload, "expected_responsibility_effect"),
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


def parse_student_candidates(payload: Mapping[str, Any]) -> tuple[StudentCandidate, ...]:
    rows = payload["candidates"]
    if not isinstance(rows, list):
        raise ValueError("student candidates must be an array")
    candidates = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each student candidate must be an object")
        candidates.append(StudentCandidate(
            candidate_prompt=_required_text(row, "candidate_prompt"),
            target_failure_mechanism=_required_text(row, "target_failure_mechanism"),
            repair_procedure=_required_text(row, "repair_procedure"),
            preservation_rule=_required_text(row, "preservation_rule"),
            expected_responsibility_effect=_required_text(row, "expected_responsibility_effect"),
        ))
    return tuple(candidates)
