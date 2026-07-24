from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from .diagnosis_aggregation import AggregatedFailurePattern, CompactEvidenceCase
from .evaluation.output_contract import solver_output_contract
from .llm_client import LLMCallResult
from .utils import extract_json_obj, normalize_prompt_text


TCS_PROTOCOL_VERSION = "aggregated_small_model_tcs_v2"
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
class MemberAwareDiagnosisContext:
    target_agent_id: int
    parent_prompt: str
    parent_prompt_hash: str
    member_correct_counts: tuple[int, ...]
    member_gains_from_initial: tuple[int, ...]
    target_improvement_need: int
    assigned_residual_count: int
    patterns: tuple[AggregatedFailurePattern, ...]
    evidence_cases: tuple[CompactEvidenceCase, ...]
    previous_outcome: PreviousUpdateOutcome


AnyDiagnosisContext = (
    AccuracyDiagnosisContext
    | PeerStateDiagnosisContext
    | MemberAwareDiagnosisContext
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
    elif isinstance(context, MemberAwareDiagnosisContext):
        common.update({
            "member_correct_counts": list(context.member_correct_counts),
            "member_gains_from_initial": list(context.member_gains_from_initial),
            "target_improvement_need": context.target_improvement_need,
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
        "Implement the approved repair plan as complete replacement prompts. Each prompt "
        "must stand alone, preserve the task output contract, contain no training example "
        "or answer, and be no longer than the stated limit. Do not return a patch or repeat "
        "diagnosis metadata. Return strict JSON with the sole field candidate_prompts.\n"
        f"ParentPrompt:\n{parent_prompt}\n"
        f"ApprovedRepairPlan:\n{json.dumps(asdict(approved_plan), ensure_ascii=False, sort_keys=True)}\n"
        f"OutputContract:\n{solver_output_contract(answer_format)}\n"
        f"RequestedCandidateCount: {candidate_count}\n"
        f"CandidatePromptMaxCharacters: {candidate_prompt_max_chars}\n"
        f"TotalCandidatePromptMaxCharacters: {total_candidate_prompt_max_chars}"
    )


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
        raise ValueError("student response must contain only candidate_prompts")
    values = payload["candidate_prompts"]
    if not isinstance(values, list):
        raise ValueError("candidate_prompts must be a list")
    if len(values) > expected_count:
        raise ValueError("candidate_count_exceeds_requested")
    total_candidate_characters = sum(
        len(normalize_prompt_text(value))
        for value in values
        if isinstance(value, str)
    )
    if total_candidate_characters > total_candidate_prompt_max_chars:
        raise ValueError("candidate_total_too_long")
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
                reasons.append("duplicate")
            if len(prompt) > candidate_prompt_max_chars:
                reasons.append("candidate_too_long")
            if contains_supplied_example_text(prompt, context):
                reasons.append("sample_text_copy")
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
