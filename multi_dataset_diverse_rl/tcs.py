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
    target_status: str
    required_transition: str
    case_role: str
    repair_goal: str
    forbidden_transition: str = ""


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
    target_status: str
    required_transition: str
    team_vote_status: str
    case_role: str
    repair_goal: str


@dataclass(frozen=True)
class MemberAwareResponsibilityCase(PeerStateCase):
    responsibility_reason: str
    initial_correct_count: int
    current_correct_count: int
    gain_count: int
    improvement_need: int
    unique_correct_count: int
    pivotal_correct_count: int
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
    target_status: str
    required_transition: str
    team_vote_status: str
    case_role: str
    repair_goal: str
    forbidden_transition: str


@dataclass(frozen=True)
class RepresentativeCase:
    question_hash: str
    question: str
    gold_answer: str
    target_current_answer: str
    target_current_correct: bool
    target_current_invalid: bool
    target_status: str
    required_transition: str
    team_vote_status: str
    case_role: str
    repair_goal: str
    forbidden_transition: str = ""


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
class MemberAwareResponsibilityProposalContext:
    target_agent_id: int
    parent_prompt: str
    parent_prompt_hash: str
    member_correct_counts: tuple[int, ...]
    member_gains_from_initial: tuple[int, ...]
    target_improvement_need: int
    assigned_coverage_cases: tuple[MemberAwareResponsibilityCase, ...]
    assigned_conversion_cases: tuple[MemberAwareResponsibilityCase, ...]
    member_error_cases: tuple[MemberAwareResponsibilityCase, ...]
    preservation_cases: tuple[PreservationCase, ...]
    representative_cases: tuple[RepresentativeCase, ...]
    responsibility_summary: str
    previous_member_summary: str


AnyProposalContext = (
    AccuracyProposalContext
    | PeerStateProposalContext
    | MemberAwareResponsibilityProposalContext
)


@dataclass(frozen=True)
class TeacherProposal:
    observed_failure_pattern: str
    generalizable_mechanism: str
    decision_rule: str
    uncertainty_or_abstention_rule: str
    preservation_conditions: str
    evidence_summary: str


@dataclass(frozen=True)
class CriticCaseFact:
    question_hash: str
    target_status: str
    case_role: str
    required_transition: str = ""
    forbidden_transition: str = ""
    team_vote_status: str = ""
    repair_goal: str = ""


@dataclass(frozen=True)
class CriticDecision:
    approved: bool
    score: float
    feedback: str
    context_consistent: bool
    sample_memorization_free: bool
    executable_change: bool
    internally_consistent: bool
    preservation_rule_present: bool
    output_contract_safe: bool
    peer_copying_free: bool
    stereotype_forcing_free: bool
    non_generic_change: bool
    blocking_reasons: tuple[str, ...]
    soft_concerns: tuple[str, ...]
    case_fact_restatements: tuple[CriticCaseFact, ...]

    @property
    def hard_checks(self) -> dict[str, bool]:
        return {
            field: bool(getattr(self, field))
            for field in CRITIC_HARD_CHECK_FIELDS
        }


@dataclass(frozen=True)
class StudentCandidate:
    candidate_prompt: str
    observed_failure_pattern: str
    generalizable_mechanism: str
    decision_rule: str
    uncertainty_or_abstention_rule: str
    preservation_conditions: str
    evidence_summary: str


CRITIC_HARD_CHECK_FIELDS = (
    "context_consistent",
    "sample_memorization_free",
    "executable_change",
    "internally_consistent",
    "preservation_rule_present",
    "output_contract_safe",
    "peer_copying_free",
    "stereotype_forcing_free",
    "non_generic_change",
)

TASK_GENERAL_DEFINITION = (
    "A task-general procedure means a rule that can apply to unseen examples "
    "within the current task. It does not need to transfer to unrelated "
    "benchmarks or task families. It must not memorize the supplied examples "
    "or their gold answers."
)

TCS_PROTOCOL_VERSION = "hard_blocker_gate_v2"
SAMPLE_MEMORIZATION_FILTER_VERSION = "exact_supplied_example_text_v1"


@dataclass(frozen=True)
class TCSContextLimits:
    assigned_coverage: int = 6
    assigned_conversion: int = 6
    preservation: int = 6
    representative: int = 6
    member_error: int = 6
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


def _context_case_rows(context: AnyProposalContext) -> tuple[Any, ...]:
    if isinstance(context, AccuracyProposalContext):
        return context.error_cases + context.protection_cases
    if isinstance(context, PeerStateProposalContext):
        return (
            context.coverage_cases
            + context.conversion_cases
            + context.preservation_cases
            + context.representative_cases
        )
    if isinstance(context, MemberAwareResponsibilityProposalContext):
        return (
            context.assigned_coverage_cases
            + context.assigned_conversion_cases
            + context.member_error_cases
            + context.preservation_cases
            + context.representative_cases
        )
    raise TypeError(f"Unsupported ProposalContext: {type(context).__name__}")


def proposal_context_case_facts(
    context: AnyProposalContext,
) -> tuple[CriticCaseFact, ...]:
    facts = []
    for row in _context_case_rows(context):
        facts.append(CriticCaseFact(
            question_hash=row.question_hash,
            target_status=row.target_status,
            case_role=row.case_role,
            required_transition=getattr(row, "required_transition", ""),
            forbidden_transition=getattr(row, "forbidden_transition", ""),
            team_vote_status=getattr(row, "team_vote_status", ""),
            repair_goal=getattr(row, "repair_goal", ""),
        ))
    return tuple(sorted(
        facts,
        key=lambda row: (row.question_hash, row.case_role),
    ))


def contains_supplied_example_text(
    text: str,
    context: AnyProposalContext,
) -> bool:
    normalized_text = " ".join(str(text or "").lower().split())
    if not normalized_text:
        return False
    fragments: set[str] = set()
    for row in _context_case_rows(context):
        question = str(row.question or "")
        normalized_question = " ".join(question.lower().split())
        if len(normalized_question) >= 32:
            fragments.add(normalized_question)
        for line in question.splitlines():
            normalized_line = " ".join(line.lower().split())
            if len(normalized_line) >= 48:
                fragments.add(normalized_line)
    return any(fragment in normalized_text for fragment in fragments)


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
            if isinstance(context, MemberAwareResponsibilityProposalContext)
            else "coverage_cases"
        )
        conversion_field = (
            "assigned_conversion_cases"
            if isinstance(context, MemberAwareResponsibilityProposalContext)
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
        if isinstance(context, MemberAwareResponsibilityProposalContext):
            available["member_error"] = len(context.member_error_cases)
        coverage_key = (
            (lambda row: (-row.owner_age, -row.oracle_soft_utility_gain, row.question_hash))
            if isinstance(context, MemberAwareResponsibilityProposalContext)
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
        if isinstance(context, MemberAwareResponsibilityProposalContext):
            bounded = replace(
                bounded,
                member_error_cases=tuple(sorted(
                    context.member_error_cases,
                    key=lambda row: (
                        -row.improvement_need,
                        -int(row.direct_vote_fix),
                        row.question_hash,
                    ),
                )[: limits.member_error]),
            )
        fields = (
            "representative_cases",
            coverage_field,
            conversion_field,
            "member_error_cases",
            "preservation_cases",
        ) if isinstance(context, MemberAwareResponsibilityProposalContext) else (
            "representative_cases", coverage_field, conversion_field, "preservation_cases"
        )
        selected_counts = lambda value: {
            "coverage": len(getattr(value, coverage_field)),
            "conversion": len(getattr(value, conversion_field)),
            "preservation": len(value.preservation_cases),
            "representative": len(value.representative_cases),
            **(
                {"member_error": len(value.member_error_cases)}
                if isinstance(value, MemberAwareResponsibilityProposalContext)
                else {}
            ),
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
    elif isinstance(context, MemberAwareResponsibilityProposalContext):
        objective = (
            "Improve this member's weak competence and assigned residual cases while "
            "preserving every member and the team vote."
        )
    else:
        raise TypeError(f"Unsupported ProposalContext: {type(context).__name__}")
    schema = {
        "observed_failure_pattern": "abstract pattern supported by the supplied evidence",
        "generalizable_mechanism": "mechanism that can transfer to unseen examples in this task",
        "decision_rule": "concrete executable behavior to add or revise",
        "uncertainty_or_abstention_rule": "operational rule for unresolved evidence",
        "preservation_conditions": "operational conditions protecting existing correct behavior",
        "evidence_summary": "abstract evidence summary without sample text, answers, or per-case transitions",
    }
    return (
        f"{objective} {TASK_GENERAL_DEFINITION} Propose a testable repair hypothesis; Stage A/B rollout, "
        "not the Teacher, will determine empirical effectiveness. Do not memorize answers, quote supplied "
        "questions, describe per-case answer transitions, assign a preset role, copy a peer procedure, "
        "or propose generic chain-of-thought. Diagnose a real defect in the parent prompt. Return strict JSON "
        f"matching this schema: {json.dumps(schema)}\nProposalContext:\n{_payload(context)}"
    )


def build_critic_request(context: AnyProposalContext, teacher_proposal: TeacherProposal) -> str:
    if isinstance(context, AccuracyProposalContext):
        policy_checks = "individual-error diagnosis and individual competence preservation"
    elif isinstance(context, PeerStateProposalContext):
        policy_checks = "team/peer-state interpretation and preservation evidence"
    elif isinstance(context, MemberAwareResponsibilityProposalContext):
        policy_checks = (
            "member improvement need, assigned residual targeting, team/peer-state "
            "interpretation, and preservation evidence"
        )
    else:
        raise TypeError(f"Unsupported ProposalContext: {type(context).__name__}")
    case_facts = [asdict(row) for row in proposal_context_case_facts(context)]
    schema = {
        "case_fact_restatements": case_facts,
        "context_consistent": True,
        "sample_memorization_free": True,
        "executable_change": True,
        "internally_consistent": True,
        "preservation_rule_present": True,
        "output_contract_safe": True,
        "peer_copying_free": True,
        "stereotype_forcing_free": True,
        "non_generic_change": True,
        "blocking_reasons": [],
        "soft_concerns": [],
        "score": 0.5,
        "feedback": "concise audit summary",
    }
    return (
        f"Audit the Teacher proposal against the same ProposalContext. Check {policy_checks}. "
        f"{TASK_GENERAL_DEFINITION} Your authority is limited to hard blockers. Reject only if the proposal "
        "contradicts the derived context facts; quotes or memorizes supplied samples or answers; adds no "
        "executable behavior; contradicts its preservation conditions; violates the output contract; explicitly "
        "copies a peer answer or prompt; forces a resolution from occupational or social stereotypes, typicality, "
        "or frequency rather than explicit sentence constraints; or contains only a role description, 'think "
        "carefully', or similarly generic advice. Subjectivity, incomplete coverage, use of a common task method, overlap with the parent "
        "prompt, uncertain benefit, and failure to prove empirical improvement are soft concerns only. Stage A/B "
        "rollout decides effectiveness. First copy DERIVED_CASE_FACTS exactly into case_fact_restatements. A "
        "missing or incorrect restatement makes the response invalid and it will be retried. Approval is computed "
        "from all hard-check booleans being true and blocking_reasons being empty; score is diagnostic only. "
        f"Return strict JSON matching this schema: {json.dumps(schema, ensure_ascii=False)}\n"
        f"DERIVED_CASE_FACTS:\n{json.dumps(case_facts, ensure_ascii=False, sort_keys=True)}\n"
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
    elif isinstance(context, MemberAwareResponsibilityProposalContext):
        policy_instruction = (
            "address the target member's error evidence and assigned responsibilities "
            "without regressing another member"
        )
    else:
        raise TypeError(f"Unsupported ProposalContext: {type(context).__name__}")
    candidate_fields = (
        "candidate_prompt, observed_failure_pattern, generalizable_mechanism, decision_rule, "
        "uncertainty_or_abstention_rule, preservation_conditions, and evidence_summary"
    )
    return (
        f"Generate exactly {candidate_count} task-general prompt candidates. {TASK_GENERAL_DEFINITION} "
        "Each candidate must implement the approved repair, preserve protected cases, and "
        f"{policy_instruction}. Do not quote any supplied question, answer, or per-case transition. Return strict "
        f"JSON with a candidates array; every item must contain {candidate_fields}.\n"
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
        observed_failure_pattern=_required_text(payload, "observed_failure_pattern"),
        generalizable_mechanism=_required_text(payload, "generalizable_mechanism"),
        decision_rule=_required_text(payload, "decision_rule"),
        uncertainty_or_abstention_rule=_required_text(payload, "uncertainty_or_abstention_rule"),
        preservation_conditions=_required_text(payload, "preservation_conditions"),
        evidence_summary=_required_text(payload, "evidence_summary"),
    )


def _required_bool(payload: Mapping[str, Any], field: str) -> bool:
    value = payload[field]
    if type(value) is not bool:
        raise ValueError(f"critic {field} must be a JSON boolean")
    return value


def _required_text_array(payload: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = payload[field]
    if not isinstance(value, list) or any(not isinstance(row, str) for row in value):
        raise ValueError(f"critic {field} must be an array of strings")
    return tuple(row.strip() for row in value if row.strip())


def _parse_case_fact(payload: Mapping[str, Any]) -> CriticCaseFact:
    fields = {
        "question_hash", "target_status", "case_role", "required_transition",
        "forbidden_transition", "team_vote_status", "repair_goal",
    }
    if set(payload) != fields:
        raise ValueError("critic case fact fields do not match the required schema")
    values = {}
    for field in fields:
        value = payload[field]
        if not isinstance(value, str):
            raise ValueError(f"critic case fact {field} must be a string")
        values[field] = value.strip()
    if not values["question_hash"] or not values["target_status"] or not values["case_role"]:
        raise ValueError("critic case facts require question_hash, target_status, and case_role")
    return CriticCaseFact(**values)


def parse_critic_decision(
    payload: Mapping[str, Any],
    context: AnyProposalContext,
) -> CriticDecision:
    score = payload["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("critic score must be numeric")
    numeric_score = float(score)
    if not 0.0 <= numeric_score <= 1.0:
        raise ValueError("critic score must be between 0 and 1")
    hard_checks = {
        field: _required_bool(payload, field)
        for field in CRITIC_HARD_CHECK_FIELDS
    }
    blocking_reasons = _required_text_array(payload, "blocking_reasons")
    soft_concerns = _required_text_array(payload, "soft_concerns")
    feedback = _required_text(payload, "feedback")
    restatements_value = payload["case_fact_restatements"]
    if not isinstance(restatements_value, list) or any(
        not isinstance(row, Mapping) for row in restatements_value
    ):
        raise ValueError("critic case_fact_restatements must be an array of objects")
    restatements = tuple(sorted(
        (_parse_case_fact(row) for row in restatements_value),
        key=lambda row: (row.question_hash, row.case_role),
    ))
    expected = proposal_context_case_facts(context)
    if restatements != expected:
        raise ValueError("critic case fact restatement mismatch")
    all_hard_checks_passed = all(hard_checks.values())
    if all_hard_checks_passed == bool(blocking_reasons):
        raise ValueError("critic hard checks and blocking_reasons disagree")
    return CriticDecision(
        approved=bool(all_hard_checks_passed and not blocking_reasons),
        score=numeric_score,
        feedback=feedback,
        **hard_checks,
        blocking_reasons=blocking_reasons,
        soft_concerns=soft_concerns,
        case_fact_restatements=restatements,
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
            observed_failure_pattern=_required_text(row, "observed_failure_pattern"),
            generalizable_mechanism=_required_text(row, "generalizable_mechanism"),
            decision_rule=_required_text(row, "decision_rule"),
            uncertainty_or_abstention_rule=_required_text(
                row, "uncertainty_or_abstention_rule",
            ),
            preservation_conditions=_required_text(row, "preservation_conditions"),
            evidence_summary=_required_text(row, "evidence_summary"),
        ))
    return tuple(candidates)
