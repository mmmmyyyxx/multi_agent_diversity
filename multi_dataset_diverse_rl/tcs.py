from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from .diagnosis_aggregation import AggregatedFailurePattern, CompactEvidenceCase
from .evaluation.output_contract import solver_output_contract
from .llm_client import LLMCallResult
from .utils import extract_json_obj, normalize_prompt_text


TCS_PROTOCOL_VERSION = "assigned_residual_only_context_v1"
TEACHER_REVISION_PROTOCOL_VERSION = "critic_grounded_full_plan_revision_v1"
TEACHER_SCHEMA_VERSION = "three_field_repair_plan_v1"
CRITIC_SCHEMA_VERSION = "four_hard_blocker_v1"
STUDENT_SCHEMA_VERSION = "replacement_prompt_list_v1"
ROLE_RETRY_POLICY_VERSION = "uncapped_completion_semantic_round_v2"
SAMPLE_MEMORIZATION_FILTER_VERSION = "exact_supplied_example_text_v1"

CRITIC_FAILED_CHECKS = (
    "evidence_mismatch",
    "actionable_specificity",
    "shortcut_or_copying",
    "preservation_or_output_risk",
)


@dataclass(frozen=True)
class PreviousUpdateOutcome:
    attempted: bool = False
    empirical_evaluation_completed: bool = False
    accepted: bool = False
    target_correct_delta: int = 0
    vote_correct_delta: int = 0
    minimum_member_gain_delta: int = 0
    total_member_gain_delta: int = 0
    assigned_repair_count: int = 0
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class AccuracyDiagnosisContext:
    target_agent_id: int
    parent_prompt: str
    parent_prompt_hash: str
    target_correct_count: int
    target_error_count: int
    target_invalid_count: int
    patterns: tuple[AggregatedFailurePattern, ...]
    evidence_cases: tuple[CompactEvidenceCase, ...]
    previous_outcome: PreviousUpdateOutcome


@dataclass(frozen=True)
class PeerStateDiagnosisContext:
    target_agent_id: int
    parent_prompt: str
    parent_prompt_hash: str
    vote_wrong_count: int
    coverage_failure_count: int
    conversion_failure_count: int
    preservation_count: int
    patterns: tuple[AggregatedFailurePattern, ...]
    evidence_cases: tuple[CompactEvidenceCase, ...]
    previous_outcome: PreviousUpdateOutcome


@dataclass(frozen=True)
class AssignedResidualDiagnosisContext:
    target_agent_id: int
    parent_prompt: str
    parent_prompt_hash: str
    assigned_residual_count: int
    patterns: tuple[AggregatedFailurePattern, ...]
    evidence_cases: tuple[CompactEvidenceCase, ...]
    previous_outcome: PreviousUpdateOutcome


AnyDiagnosisContext = (
    AccuracyDiagnosisContext
    | PeerStateDiagnosisContext
    | AssignedResidualDiagnosisContext
)


@dataclass(frozen=True)
class TeacherRepairPlan:
    failure_pattern: str
    repair_rule: str
    preservation_rule: str


@dataclass(frozen=True)
class CriticDecision:
    approved: bool
    failed_checks: tuple[str, ...]
    risk_case_ids: tuple[str, ...]
    feedback: str


def teacher_repair_plan_hash(plan: TeacherRepairPlan) -> str:
    payload = json.dumps(
        asdict(plan),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def critic_decision_hash(decision: CriticDecision) -> str:
    payload = json.dumps(
        {
            "failed_checks": list(decision.failed_checks),
            "risk_case_ids": list(decision.risk_case_ids),
            "feedback": decision.feedback,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def changed_teacher_plan_fields(
    previous: TeacherRepairPlan,
    revised: TeacherRepairPlan,
) -> tuple[str, ...]:
    return tuple(
        field
        for field in ("failure_pattern", "repair_rule", "preservation_rule")
        if getattr(previous, field) != getattr(revised, field)
    )


@dataclass(frozen=True)
class StudentPromptCandidate:
    candidate_prompt: str


@dataclass(frozen=True)
class StudentParseResult:
    candidates: tuple[StudentPromptCandidate, ...]
    raw_count: int
    rejection_reasons: tuple[tuple[str, ...], ...]
    total_candidate_characters: int


@dataclass(frozen=True)
class TCSContextDiagnostics:
    full_probe_case_count: int
    available_pattern_count: int
    selected_pattern_count: int
    selected_pattern_ids: tuple[str, ...]
    selected_case_count: int
    selected_case_ids: tuple[str, ...]
    cases_represented_by_selected_patterns: int
    context_characters: int
    estimated_input_tokens: int


def _accuracy_pattern_payload(pattern: AggregatedFailurePattern) -> dict[str, Any]:
    return {
        "pattern_id": pattern.pattern_id,
        "case_family": pattern.key.case_family,
        "target_status": pattern.key.target_status,
        "case_count": pattern.case_count,
        "repair_goal": pattern.repair_goal,
    }


def _accuracy_case_payload(row: CompactEvidenceCase) -> dict[str, Any]:
    return {
        "case_id": row.case_id,
        "pattern_id": row.pattern_id,
        "case_family": row.case_family,
        "question": row.question,
        "gold_answer": row.gold_answer,
        "target_current_answer": row.target_current_answer,
        "repair_goal": row.repair_goal,
    }


def _peer_pattern_payload(pattern: AggregatedFailurePattern) -> dict[str, Any]:
    return {
        "pattern_id": pattern.pattern_id,
        "key": asdict(pattern.key),
        "case_count": pattern.case_count,
        "direct_vote_fix_count": pattern.direct_vote_fix_count,
        "dominant_wrong_count": pattern.dominant_wrong_count,
        "mean_oracle_soft_utility_gain": pattern.mean_oracle_soft_utility_gain,
        "max_oracle_soft_utility_gain": pattern.max_oracle_soft_utility_gain,
        "repair_goal": pattern.repair_goal,
    }


def _member_pattern_payload(pattern: AggregatedFailurePattern) -> dict[str, Any]:
    payload = _peer_pattern_payload(pattern)
    payload["assigned_case_count"] = pattern.assigned_case_count
    payload["max_owner_age"] = pattern.max_owner_age
    return payload


def _peer_case_payload(row: CompactEvidenceCase) -> dict[str, Any]:
    return {
        "case_id": row.case_id,
        "pattern_id": row.pattern_id,
        "case_family": row.case_family,
        "question": row.question,
        "gold_answer": row.gold_answer,
        "target_current_answer": row.target_current_answer,
        "answer_role_signature": list(row.answer_role_signature),
        "target_answer_role": row.target_answer_role,
        "gold_vote_count": row.gold_vote_count,
        "largest_wrong_vote_count": row.largest_wrong_vote_count,
        "plurality_margin": row.plurality_margin,
        "peer_gold_vote_count": row.peer_gold_vote_count,
        "peer_largest_wrong_vote_count": row.peer_largest_wrong_vote_count,
        "peer_margin": row.peer_margin,
        "direct_vote_fix": row.direct_vote_fix,
        "dominant_wrong_member": row.dominant_wrong_member,
        "unique_correct": row.unique_correct,
        "pivotal_correct": row.pivotal_correct,
        "repair_goal": row.repair_goal,
    }


def context_payload(context: AnyDiagnosisContext) -> dict[str, Any]:
    common: dict[str, Any] = {
        "target_agent_id": context.target_agent_id,
        "parent_prompt": context.parent_prompt,
    }
    if isinstance(context, AccuracyDiagnosisContext):
        common.update({
            "target_correct_count": context.target_correct_count,
            "target_error_count": context.target_error_count,
            "target_invalid_count": context.target_invalid_count,
            "patterns": [_accuracy_pattern_payload(row) for row in context.patterns],
            "evidence_cases": [_accuracy_case_payload(row) for row in context.evidence_cases],
        })
    elif isinstance(context, PeerStateDiagnosisContext):
        common.update({
            "vote_wrong_count": context.vote_wrong_count,
            "coverage_failure_count": context.coverage_failure_count,
            "conversion_failure_count": context.conversion_failure_count,
            "preservation_count": context.preservation_count,
            "patterns": [_peer_pattern_payload(row) for row in context.patterns],
            "evidence_cases": [_peer_case_payload(row) for row in context.evidence_cases],
        })
    elif isinstance(context, AssignedResidualDiagnosisContext):
        common.update({
            "assigned_residual_count": context.assigned_residual_count,
            "patterns": [_member_pattern_payload(row) for row in context.patterns],
            "evidence_cases": [_peer_case_payload(row) for row in context.evidence_cases],
        })
    else:
        raise TypeError(f"Unsupported diagnosis context: {type(context).__name__}")
    outcome = asdict(context.previous_outcome)
    empirical_feedback_available = bool(
        outcome.pop("empirical_evaluation_completed")
    )
    if not empirical_feedback_available:
        common["previous_outcome"] = {
            "attempted": bool(outcome["attempted"]),
            "empirical_feedback_available": False,
        }
        return common
    outcome["empirical_feedback_available"] = True
    if isinstance(context, AccuracyDiagnosisContext):
        outcome = {
            key: outcome[key]
            for key in (
                "attempted", "empirical_feedback_available", "accepted",
                "target_correct_delta",
                "rejection_reasons",
            )
        }
    elif isinstance(context, PeerStateDiagnosisContext):
        outcome = {
            key: outcome[key]
            for key in (
                "attempted", "empirical_feedback_available", "accepted",
                "target_correct_delta",
                "vote_correct_delta", "rejection_reasons",
            )
        }
    common["previous_outcome"] = outcome
    return common


def serialize_context(context: AnyDiagnosisContext) -> str:
    return json.dumps(
        context_payload(context),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def limit_diagnosis_context(
    context: AnyDiagnosisContext,
    *,
    max_chars: int,
    full_probe_case_count: int,
    available_pattern_count: int,
) -> tuple[AnyDiagnosisContext, TCSContextDiagnostics]:
    if max_chars <= 0:
        raise ValueError("tcs_context_max_chars must be positive")
    bounded = context
    while len(serialize_context(bounded)) > max_chars and bounded.patterns:
        kept = bounded.patterns[:-1]
        kept_ids = {row.pattern_id for row in kept}
        bounded = replace(
            bounded,
            patterns=kept,
            evidence_cases=tuple(
                row for row in bounded.evidence_cases if row.pattern_id in kept_ids
            ),
        )
    while len(serialize_context(bounded)) > max_chars and bounded.evidence_cases:
        bounded = replace(bounded, evidence_cases=bounded.evidence_cases[:-1])
    characters = len(serialize_context(bounded))
    if characters > max_chars:
        raise ValueError("TCS context metadata and parent prompt exceed tcs_context_max_chars")
    return bounded, TCSContextDiagnostics(
        full_probe_case_count=full_probe_case_count,
        available_pattern_count=available_pattern_count,
        selected_pattern_count=len(bounded.patterns),
        selected_pattern_ids=tuple(row.pattern_id for row in bounded.patterns),
        selected_case_count=len(bounded.evidence_cases),
        selected_case_ids=tuple(row.case_id for row in bounded.evidence_cases),
        cases_represented_by_selected_patterns=sum(
            row.case_count for row in bounded.patterns
        ),
        context_characters=characters,
        estimated_input_tokens=(characters + 3) // 4,
    )


def _case_rows(context: AnyDiagnosisContext) -> tuple[CompactEvidenceCase, ...]:
    return context.evidence_cases


def contains_supplied_example_text(text: str, context: AnyDiagnosisContext) -> bool:
    normalized_text = " ".join(str(text or "").lower().split())
    if not normalized_text:
        return False
    fragments: set[str] = set()
    for row in _case_rows(context):
        normalized_question = " ".join(row.question.lower().split())
        if len(normalized_question) >= 32:
            fragments.add(normalized_question)
        for line in row.question.splitlines():
            normalized_line = " ".join(line.lower().split())
            if len(normalized_line) >= 48:
                fragments.add(normalized_line)
    return any(fragment in normalized_text for fragment in fragments)


def build_teacher_request(
    context: AnyDiagnosisContext,
    *,
    field_max_chars: int = 800,
    total_max_chars: int = 1800,
) -> str:
    schema = {
        "failure_pattern": "concise diagnosis",
        "repair_rule": "concrete executable rule including uncertainty handling",
        "preservation_rule": "concrete rule protecting existing correct behavior",
    }
    return (
        "Propose one task-general, testable prompt repair plan from the typed aggregate "
        "diagnosis. Do not quote cases or answers, describe per-case transitions, copy a "
        "peer procedure, predict performance, or generate candidate prompts. The repair "
        "rule must specify executable behavior and integrate uncertainty handling. The "
        "preservation rule must protect correct behavior and the strict output contract. "
        f"Return strict JSON with exactly these fields: {json.dumps(schema)}\n"
        f"TeacherFieldMaxCharacters: {field_max_chars}\n"
        f"TeacherTotalMaxCharacters: {total_max_chars}\n"
        f"DiagnosisContext:\n{serialize_context(context)}"
    )


def build_teacher_revision_request(
    *,
    context: AnyDiagnosisContext,
    previous_plan: TeacherRepairPlan,
    critic_decision: CriticDecision,
    field_max_chars: int = 800,
    total_max_chars: int = 1800,
    feedback_max_chars: int = 500,
) -> str:
    if critic_decision.approved or not critic_decision.failed_checks:
        raise ValueError("Teacher revision requires a rejected Critic decision")
    critic_payload = {
        "failed_checks": list(critic_decision.failed_checks),
        "risk_case_ids": list(critic_decision.risk_case_ids),
        "feedback": critic_decision.feedback,
    }
    context_hash = hashlib.sha256(
        serialize_context(context).encode("utf-8")
    ).hexdigest()
    return (
        "Revise the previous TeacherRepairPlan into a complete replacement plan. "
        "Return all three original fields, not a patch or commentary. Address every "
        "failed_check while preserving rules that were not challenged. A revision "
        "must satisfy all four hard checks cumulatively; do not evade a prior "
        "evidence_mismatch by replacing an executable rule with vague language. "
        "Base every repair rule on observable checks available in the task input, "
        "and do not claim that supplied evidence establishes facts it does not show. "
        "Do not quote cases or answers, copy a peer procedure, predict performance, "
        "or generate candidate prompts. The preservation rule must protect correct "
        "behavior and the strict output contract. "
        f"RevisionProtocolVersion: {TEACHER_REVISION_PROTOCOL_VERSION}\n"
        f"TeacherFieldMaxCharacters: {field_max_chars}\n"
        f"TeacherTotalMaxCharacters: {total_max_chars}\n"
        f"CriticFeedbackMaxCharacters: {feedback_max_chars}\n"
        f"PreviousTeacherRepairPlan:\n{json.dumps(asdict(previous_plan), ensure_ascii=False, sort_keys=True)}\n"
        f"CriticDecision:\n{json.dumps(critic_payload, ensure_ascii=False, sort_keys=True)}\n"
        f"DiagnosisContextHash: {context_hash}"
    )


def build_critic_request(
    context: AnyDiagnosisContext,
    repair_plan: TeacherRepairPlan,
    *,
    feedback_max_chars: int = 500,
) -> str:
    schema = {"failed_checks": [], "risk_case_ids": [], "feedback": ""}
    return (
        "Check only explicit hard blockers in the repair plan. Allowed failed_checks are "
        f"{json.dumps(CRITIC_FAILED_CHECKS)}. evidence_mismatch means a clear conflict "
        "with the aggregate or representative evidence; actionable_specificity means the "
        "rule is generic, non-executable, or contradictory; shortcut_or_copying means "
        "sample memorization, specific-answer or peer copying, or stereotype shortcuts; "
        "preservation_or_output_risk means preservation is inoperable or the strict output "
        "contract is endangered. Do not score, predict candidate performance, restate "
        "facts, or report soft concerns. risk_case_ids may only name supplied case IDs. "
        "Use empty feedback when approved; when rejecting give one concrete revision, at "
        f"most {feedback_max_chars} characters. "
        f"CriticFeedbackMaxCharacters: {feedback_max_chars}. "
        f"Return exactly: {json.dumps(schema)}\n"
        f"DiagnosisContext:\n{serialize_context(context)}\n"
        f"TeacherRepairPlan:\n{json.dumps(asdict(repair_plan), ensure_ascii=False, sort_keys=True)}"
    )


def build_student_request(
    *,
    parent_prompt: str,
    approved_plan: TeacherRepairPlan,
    answer_format: str,
    candidate_count: int,
    candidate_prompt_max_chars: int,
    total_candidate_prompt_max_chars: int = 5000,
) -> str:
    return (
        "Implement the approved repair plan as complete replacement decision procedures. "
        "Each candidate is only the mutable reasoning procedure: it must stand alone, "
        "contain no training example or answer, and be no longer than the stated limit. "
        "The system appends the immutable output interface after every candidate at Solver "
        "request time. Do not duplicate that full interface or return a patch or diagnosis "
        "metadata. Do not introduce instructions that conflict with the supplied contract. "
        "Return strict JSON with the sole field candidate_prompts.\n"
        f"ParentPrompt:\n{parent_prompt}\n"
        f"ApprovedRepairPlan:\n{json.dumps(asdict(approved_plan), ensure_ascii=False, sort_keys=True)}\n"
        f"OutputContract:\n{solver_output_contract(answer_format)}\n"
        f"RequestedCandidateCount: {candidate_count}\n"
        f"CandidatePromptMaxCharacters: {candidate_prompt_max_chars}\n"
        f"TotalCandidatePromptMaxCharacters: {total_candidate_prompt_max_chars}"
    )


def build_student_recovery_request(
    *,
    base_request: str,
    previous_rejection_classes: tuple[str, ...],
    required_candidate_count: int,
    parent_prompt_hash: str,
    approved_repair_plan_hash: str,
) -> str:
    feedback = {
        "student_recovery": True,
        "previous_rejection_classes": list(previous_rejection_classes),
        "required_candidate_count": int(required_candidate_count),
        "parent_prompt_hash": str(parent_prompt_hash),
        "approved_repair_plan_hash": str(approved_repair_plan_hash),
        "requirements": [
            "Return candidate_prompts as a JSON array.",
            "Every candidate must be a non-empty string.",
            "Do not return null, objects, nested arrays, or empty strings.",
            "Each candidate must differ from the parent prompt and other candidates.",
            "Return only the required schema.",
        ],
    }
    return (
        f"{base_request}\n"
        "StudentRecoveryFeedback:\n"
        f"{json.dumps(feedback, ensure_ascii=False, sort_keys=True)}"
    )


def build_teacher_regeneration_request(
    *,
    previous_plan_hash: str,
    student_rejection_classes: tuple[str, ...],
) -> str:
    payload = {
        "student_upstream_regeneration": True,
        "previous_approved_plan_hash": str(previous_plan_hash),
        "student_rejection_classes": list(student_rejection_classes),
        "requirements": [
            "Produce a materially different complete repair plan.",
            "Return failure_pattern, repair_rule, and preservation_rule.",
            "Use the same bounded diagnosis context.",
            "Do not infer or reproduce invalid candidate text.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_teacher_repair_plan(
    payload: Mapping[str, Any],
    *,
    field_max_chars: int = 800,
    total_max_chars: int = 1800,
) -> TeacherRepairPlan:
    expected = {"failure_pattern", "repair_rule", "preservation_rule"}
    if set(payload) != expected:
        raise ValueError("teacher response must contain exactly three repair-plan fields")
    values = {}
    for field in sorted(expected):
        value = payload[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"teacher field {field} must be a non-empty string")
        value = value.strip()
        if len(value) > field_max_chars:
            raise ValueError(f"teacher field {field} exceeds character limit")
        values[field] = value
    if sum(len(value) for value in values.values()) > total_max_chars:
        raise ValueError("teacher repair plan exceeds total character limit")
    normalized_rule = " ".join(values["repair_rule"].lower().split())
    if normalized_rule in {
        "think carefully",
        "double-check",
        "think carefully and double-check",
        "be careful",
    }:
        raise ValueError("teacher repair_rule is generic rather than executable")
    return TeacherRepairPlan(**values)


def parse_critic_decision(
    payload: Mapping[str, Any],
    *,
    allowed_case_ids: set[str],
    feedback_max_chars: int = 500,
) -> CriticDecision:
    expected = {"failed_checks", "risk_case_ids", "feedback"}
    if set(payload) != expected:
        raise ValueError("critic response must contain exactly three fields")
    failed = payload["failed_checks"]
    risk_ids = payload["risk_case_ids"]
    feedback = payload["feedback"]
    if not isinstance(failed, list) or any(not isinstance(row, str) for row in failed):
        raise ValueError("failed_checks must be a list of strings")
    if len(set(failed)) != len(failed) or any(row not in CRITIC_FAILED_CHECKS for row in failed):
        raise ValueError("failed_checks contains an unknown or duplicate value")
    if not isinstance(risk_ids, list) or any(not isinstance(row, str) for row in risk_ids):
        raise ValueError("risk_case_ids must be a list of strings")
    if len(set(risk_ids)) != len(risk_ids) or any(row not in allowed_case_ids for row in risk_ids):
        raise ValueError("risk_case_ids contains an unknown or duplicate case ID")
    if not isinstance(feedback, str):
        raise ValueError("feedback must be a string")
    feedback = feedback.strip()
    if len(feedback) > feedback_max_chars:
        raise ValueError("critic feedback exceeds character limit")
    if failed and not feedback:
        raise ValueError("critic rejection requires non-empty feedback")
    return CriticDecision(
        approved=not failed,
        failed_checks=tuple(failed),
        risk_case_ids=tuple(risk_ids),
        feedback=feedback,
    )


def parse_student_candidates(
    payload: Mapping[str, Any],
    *,
    parent_prompt: str,
    context: AnyDiagnosisContext,
    expected_count: int,
    candidate_prompt_max_chars: int = 3000,
    total_candidate_prompt_max_chars: int = 5000,
) -> StudentParseResult:
    if set(payload) != {"candidate_prompts"}:
        if "candidate_prompts" not in payload:
            raise ValueError("candidate_list_missing")
        raise ValueError("schema_invalid")
    values = payload["candidate_prompts"]
    if not isinstance(values, list):
        raise ValueError("candidate_list_missing")
    if len(values) > expected_count:
        raise ValueError("schema_invalid")
    total_candidate_characters = sum(
        len(normalize_prompt_text(value))
        for value in values
        if isinstance(value, str)
    )
    if total_candidate_characters > total_candidate_prompt_max_chars:
        raise ValueError("too_long")
    parent_hash = hashlib.sha256(
        normalize_prompt_text(parent_prompt).encode("utf-8")
    ).hexdigest()
    seen: set[str] = set()
    accepted: list[StudentPromptCandidate] = []
    rejections: list[tuple[str, ...]] = []
    for value in values:
        reasons: list[str] = []
        if not isinstance(value, str) or not normalize_prompt_text(value):
            reasons.append("empty_or_non_string")
            prompt = ""
        else:
            prompt = normalize_prompt_text(value)
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if prompt_hash == parent_hash:
                reasons.append("parent_identical")
            if prompt_hash in seen:
                reasons.append("duplicate_candidate")
            if len(prompt) > candidate_prompt_max_chars:
                reasons.append("too_long")
            if contains_supplied_example_text(prompt, context):
                reasons.append("sample_memorization")
            seen.add(prompt_hash)
        rejections.append(tuple(reasons))
        if not reasons:
            accepted.append(StudentPromptCandidate(prompt))
    return StudentParseResult(
        candidates=tuple(accepted),
        raw_count=len(values),
        rejection_reasons=tuple(rejections),
        total_candidate_characters=total_candidate_characters,
    )


def response_truncated(result: LLMCallResult) -> bool:
    return result.finish_reason == "length"
