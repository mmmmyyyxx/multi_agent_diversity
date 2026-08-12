from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from .diagnosis_aggregation import (
    AggregatedFailurePattern,
    CompactEvidenceCase,
    CompactLaneEvidenceCase,
    LanePatternSummary,
)
from .evaluation.output_contract import solver_output_contract
from .evaluation.mutable_prompt_contract import mutable_prompt_violation_reasons
from .llm_client import LLMCallResult
from .proposal_memory import ProposalFailureFeedback
from .module2_context import (
    C2_BOUNDARY_PLUS_PRESERVATION,
    C3_COALITION_AWARE_PRESERVATION,
    PreservationContextItem,
    RepairContextItem,
)
from .utils import extract_json_obj, normalize_prompt_text
from .versions import STUDENT_PROMPT_CONTRACT_VERSION


TCS_PROTOCOL_VERSION = "assigned_residual_only_context_v1"
TEACHER_REVISION_PROTOCOL_VERSION = "critic_grounded_full_plan_revision_v1"
TEACHER_SCHEMA_VERSION = "three_field_repair_plan_v1"
CRITIC_SCHEMA_VERSION = "four_hard_blocker_v1"
STUDENT_SCHEMA_VERSION = "replacement_prompt_list_v1"
ROLE_RETRY_POLICY_VERSION = "uncapped_completion_semantic_round_v2"
SAMPLE_MEMORIZATION_FILTER_VERSION = "exact_supplied_example_text_v1"
STUDENT_SYSTEM_PROMPT = (
    "Return strict JSON only. Generate only mutable reasoning and decision "
    "procedures. Do not include, quote, imitate, or describe the solver output "
    "interface. Do not include FINAL_ANSWER or any fixed answer value; the program "
    "appends the immutable solver output interface later. Do not prescribe final "
    "response formatting, confidence labels, or fallback answer tokens."
)

CRITIC_FAILED_CHECKS = (
    "evidence_mismatch",
    "actionable_specificity",
    "shortcut_or_copying",
    "preservation_or_output_risk",
)
RELEVANCE_CRITIC_FAILED_CHECKS = (
    "ungrounded_diagnosis",
    "generic_not_member_specific",
    "edit_not_responsibility_aligned",
    "example_memorization",
    "unnecessary_broad_rewrite",
    "no_actionable_change",
    "schema_or_contract_invalid",
)
M20_CURRENT_V15 = "m20_current_v15"
M2A_RESIDUAL_DIAGNOSIS = "m2a_residual_diagnosis"
M2B_DIAGNOSIS_MINIMAL_EDIT = "m2b_diagnosis_minimal_edit"
M2C_DIAGNOSIS_MINIMAL_EDIT_RELEVANCE_CRITIC = (
    "m2c_diagnosis_minimal_edit_relevance_critic"
)
MODULE2_EVOLUTION_VARIANTS = (
    M20_CURRENT_V15,
    M2A_RESIDUAL_DIAGNOSIS,
    M2B_DIAGNOSIS_MINIMAL_EDIT,
    M2C_DIAGNOSIS_MINIMAL_EDIT_RELEVANCE_CRITIC,
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
    target_member_gain: int
    uplift_deficit: int
    direct_fix_responsibility_count: int
    margin_gain_responsibility_sum: int
    coverage_residual_count: int
    conversion_residual_count: int
    preservation_count: int
    patterns: tuple[AggregatedFailurePattern, ...]
    evidence_cases: tuple[CompactEvidenceCase, ...]
    previous_outcome: PreviousUpdateOutcome
    proposal_failure_feedback: ProposalFailureFeedback | None = None


@dataclass(frozen=True)
class CompactPreviousOutcome:
    status: str
    main_rejection: str | None


@dataclass(frozen=True)
class SingleLaneDiagnosisContext:
    parent_prompt: str
    parent_prompt_hash: str
    repair_lane: str
    repair_goal: str
    active_residual_count: int
    dominant_target_role: str
    dominant_pattern_case_count: int
    dominant_pattern: LanePatternSummary
    repair_cases: tuple[CompactLaneEvidenceCase, ...]
    preservation_case: CompactLaneEvidenceCase | None
    previous_outcome: CompactPreviousOutcome

    @property
    def patterns(self) -> tuple[LanePatternSummary, ...]:
        return (self.dominant_pattern,)

    @property
    def evidence_cases(self) -> tuple[CompactLaneEvidenceCase, ...]:
        return self.repair_cases + (
            (self.preservation_case,) if self.preservation_case is not None else ()
        )


@dataclass(frozen=True)
class ExperimentalModule2DiagnosisContext:
    parent_prompt: str
    parent_prompt_hash: str
    context_variant: str
    repair_cases: tuple[RepairContextItem, ...]
    preservation_cases: tuple[PreservationContextItem, ...]
    previous_outcome: CompactPreviousOutcome

    @property
    def patterns(self) -> tuple[Any, ...]:
        return ()

    @property
    def evidence_cases(
        self,
    ) -> tuple[RepairContextItem | PreservationContextItem, ...]:
        return self.repair_cases + self.preservation_cases


AnyDiagnosisContext = (
    AccuracyDiagnosisContext
    | PeerStateDiagnosisContext
    | AssignedResidualDiagnosisContext
    | SingleLaneDiagnosisContext
    | ExperimentalModule2DiagnosisContext
)


@dataclass(frozen=True)
class TeacherRepairPlan:
    failure_pattern: str
    repair_rule: str
    preservation_rule: str
    diagnosis_primary_failure_mode: str = ""
    diagnosis_evidence_patterns: tuple[str, ...] = ()
    diagnosis_peer_contrast: str = ""
    diagnosis_desired_behavior_changes: tuple[str, ...] = ()
    edit_plan: tuple[str, ...] = ()


@dataclass(frozen=True)
class CriticDecision:
    approved: bool
    failed_checks: tuple[str, ...]
    risk_case_ids: tuple[str, ...]
    feedback: str


def teacher_repair_plan_hash(plan: TeacherRepairPlan) -> str:
    payload = json.dumps(
        teacher_repair_plan_payload(plan),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def teacher_repair_plan_payload(plan: TeacherRepairPlan) -> dict[str, Any]:
    """Serialize a plan without changing the exact legacy M2-0 wire payload."""
    payload = asdict(plan)
    diagnosis_fields = (
        "diagnosis_primary_failure_mode",
        "diagnosis_evidence_patterns",
        "diagnosis_peer_contrast",
        "diagnosis_desired_behavior_changes",
        "edit_plan",
    )
    if not any(payload[field] for field in diagnosis_fields):
        for field in diagnosis_fields:
            payload.pop(field)
    return payload


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
        for field in TeacherRepairPlan.__dataclass_fields__
        if getattr(previous, field) != getattr(revised, field)
    )


def compact_previous_outcome(
    outcome: PreviousUpdateOutcome,
) -> CompactPreviousOutcome:
    if not outcome.attempted:
        return CompactPreviousOutcome(status="none", main_rejection=None)
    if not outcome.empirical_evaluation_completed:
        if any("semantic" in reason for reason in outcome.rejection_reasons):
            return CompactPreviousOutcome(
                status="rejected", main_rejection="semantic_rejection"
            )
        return CompactPreviousOutcome(
            status="operational_failure", main_rejection=None
        )
    if outcome.accepted:
        return CompactPreviousOutcome(status="accepted", main_rejection=None)

    reasons = set(outcome.rejection_reasons)
    priorities = (
        ("target_not_improved", {"target_regression", "no_target_or_vote_progress"}),
        ("team_vote_regression", {"team_vote_regression"}),
        ("terminal_invalid_regression", {"terminal_invalid_regression"}),
    )
    for label, matches in priorities:
        if reasons & matches:
            return CompactPreviousOutcome(
                status="rejected", main_rejection=label
            )
    if any("semantic" in reason for reason in reasons):
        return CompactPreviousOutcome(
            status="rejected", main_rejection="semantic_rejection"
        )
    if "member_objective_regression" in reasons:
        return CompactPreviousOutcome(
            status="rejected", main_rejection="legacy_objective_guard"
        )
    return CompactPreviousOutcome(status="rejected", main_rejection="other")


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
        "margin_gain_sum": pattern.margin_gain_sum,
        "dominant_wrong_count": pattern.dominant_wrong_count,
        "mean_oracle_soft_utility_gain": pattern.mean_oracle_soft_utility_gain,
        "max_oracle_soft_utility_gain": pattern.max_oracle_soft_utility_gain,
        "repair_goal": pattern.repair_goal,
    }


def _member_pattern_payload(pattern: AggregatedFailurePattern) -> dict[str, Any]:
    payload = _peer_pattern_payload(pattern)
    payload["assigned_case_count"] = pattern.assigned_case_count
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
        "vote_flip_gain": row.vote_flip_gain,
        "margin_gain": row.margin_gain,
        "dominant_wrong_member": row.dominant_wrong_member,
        "unique_correct": row.unique_correct,
        "pivotal_correct": row.pivotal_correct,
        "repair_goal": row.repair_goal,
    }


def _lane_case_payload(row: CompactLaneEvidenceCase) -> dict[str, Any]:
    return {
        "case_id": row.case_id,
        "question": row.question,
        "gold_answer": row.gold_answer,
        "target_current_answer": row.target_current_answer,
    }


def _lane_pattern_payload(
    context: SingleLaneDiagnosisContext,
) -> dict[str, Any]:
    return {
        "repair_lane": context.repair_lane,
        "repair_goal": context.repair_goal,
        "target_error_role": context.dominant_target_role,
        "case_count": context.dominant_pattern_case_count,
    }


def context_payload(context: AnyDiagnosisContext) -> dict[str, Any]:
    if isinstance(context, ExperimentalModule2DiagnosisContext):
        expose_metadata = (
            context.context_variant == C3_COALITION_AWARE_PRESERVATION
        )
        repair_rows = []
        for index, row in enumerate(context.repair_cases, start=1):
            payload = {
                "case_id": f"repair_{index}",
                "question": row.question,
                "gold_answer": row.gold_answer,
                "target_current_answer": row.target_current_answer,
            }
            if expose_metadata:
                payload.update({
                    "gold_vote_count": row.gold_vote_count,
                    "repair_distance": row.repair_distance,
                    "boundary_class": row.boundary_class,
                    "target_role": row.target_role,
                })
            repair_rows.append(payload)
        preservation_rows = []
        for index, row in enumerate(context.preservation_cases, start=1):
            payload = {
                "case_id": f"preservation_{index}",
                "question": row.question,
                "gold_answer": row.gold_answer,
                "target_current_answer": row.target_current_answer,
            }
            if expose_metadata:
                payload["preservation_tier"] = row.tier
            preservation_rows.append(payload)
        return {
            "parent_prompt": context.parent_prompt,
            "context_variant": context.context_variant,
            "repair_responsibilities": repair_rows,
            "preservation_responsibilities": preservation_rows,
            "previous_outcome": asdict(context.previous_outcome),
        }
    if isinstance(context, SingleLaneDiagnosisContext):
        return {
            "parent_prompt": context.parent_prompt,
            "repair_lane": context.repair_lane,
            "repair_goal": context.repair_goal,
            "active_residual_count": context.active_residual_count,
            "dominant_target_role": context.dominant_target_role,
            "dominant_pattern_case_count": (
                context.dominant_pattern_case_count
            ),
            "pattern_summary": _lane_pattern_payload(context),
            "repair_cases": [
                _lane_case_payload(row) for row in context.repair_cases
            ],
            "preservation_case": (
                _lane_case_payload(context.preservation_case)
                if context.preservation_case is not None else None
            ),
            "previous_outcome": asdict(context.previous_outcome),
        }
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
            "target_member_gain": context.target_member_gain,
            "uplift_deficit": context.uplift_deficit,
            "direct_fix_responsibility_count": (
                context.direct_fix_responsibility_count
            ),
            "margin_gain_responsibility_sum": (
                context.margin_gain_responsibility_sum
            ),
            "coverage_residual_count": context.coverage_residual_count,
            "conversion_residual_count": context.conversion_residual_count,
            "preservation_count": context.preservation_count,
            "patterns": [_member_pattern_payload(row) for row in context.patterns],
            "evidence_cases": [_peer_case_payload(row) for row in context.evidence_cases],
        })
        if context.proposal_failure_feedback is not None:
            common["proposal_failure_feedback"] = asdict(
                context.proposal_failure_feedback
            )
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
    if isinstance(context, ExperimentalModule2DiagnosisContext):
        characters = len(serialize_context(context))
        if characters > max_chars:
            raise ValueError(
                "experimental Module2 context exceeds tcs_context_max_chars; "
                "membership-preserving truncation is forbidden"
            )
        return context, TCSContextDiagnostics(
            full_probe_case_count=full_probe_case_count,
            available_pattern_count=available_pattern_count,
            selected_pattern_count=0,
            selected_pattern_ids=(),
            selected_case_count=len(context.evidence_cases),
            selected_case_ids=tuple(
                [f"repair_{index}" for index in range(1, len(context.repair_cases) + 1)]
                + [
                    f"preservation_{index}"
                    for index in range(1, len(context.preservation_cases) + 1)
                ]
            ),
            cases_represented_by_selected_patterns=len(context.evidence_cases),
            context_characters=characters,
            estimated_input_tokens=(characters + 3) // 4,
        )
    if isinstance(context, SingleLaneDiagnosisContext):
        effective_cap = min(max_chars, 6000)
        bounded = context
        if (
            len(serialize_context(bounded)) > effective_cap
            and bounded.preservation_case is not None
        ):
            bounded = replace(bounded, preservation_case=None)
        if (
            len(serialize_context(bounded)) > effective_cap
            and len(bounded.repair_cases) > 1
        ):
            bounded = replace(bounded, repair_cases=bounded.repair_cases[:1])
        characters = len(serialize_context(bounded))
        if characters > effective_cap:
            raise ValueError(
                "single-lane context metadata and parent prompt exceed 6000 characters"
            )
        return bounded, TCSContextDiagnostics(
            full_probe_case_count=full_probe_case_count,
            available_pattern_count=available_pattern_count,
            selected_pattern_count=1,
            selected_pattern_ids=(bounded.dominant_pattern.pattern_id,),
            selected_case_count=len(bounded.repair_cases) + int(
                bounded.preservation_case is not None
            ),
            selected_case_ids=tuple(
                row.case_id for row in bounded.evidence_cases
            ),
            cases_represented_by_selected_patterns=(
                bounded.dominant_pattern_case_count
            ),
            context_characters=characters,
            estimated_input_tokens=(characters + 3) // 4,
        )
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


def _case_rows(
    context: AnyDiagnosisContext,
) -> tuple[CompactEvidenceCase | CompactLaneEvidenceCase, ...]:
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
    evolution_variant: str = M20_CURRENT_V15,
) -> str:
    schema = {
        "failure_pattern": "concise diagnosis",
        "repair_rule": "concrete executable rule including uncertainty handling",
        "preservation_rule": "concrete rule protecting existing correct behavior",
    }
    diagnosis_instruction = ""
    if evolution_variant != M20_CURRENT_V15:
        schema.update({
            "diagnosis_primary_failure_mode": "one compact behavioral failure mode",
            "diagnosis_evidence_patterns": ["at most two abstract evidence patterns"],
            "diagnosis_peer_contrast": "compact contrast with successful peer behavior",
            "diagnosis_desired_behavior_changes": ["at most two behavioral changes"],
            "edit_plan": ["at most two responsibility-aligned edit actions"],
        })
        diagnosis_instruction = (
            " In the same Teacher call, infer one member-specific, residual-grounded "
            "behavioral diagnosis. Contrast only abstract successful peer behavior "
            "legally represented in the supplied aggregate; never expose raw peer "
            "answers or prompts. Use at most two evidence patterns, two desired "
            "behavior changes, and two edit-plan actions. Avoid generic advice and "
            "example-specific lookup or answer rules."
        )
    feedback_instruction = ""
    if isinstance(context, AssignedResidualDiagnosisContext) and context.proposal_failure_feedback:
        feedback_instruction = (
            " The state-local proposal failure feedback may only revise this target's "
            "evidence selection and repair plan. Do not reassign responsibility or "
            "discuss other agents, member competence, gains, ranks, cooldowns, or "
            "residuals outside this target's portfolio."
        )
    lane_instruction = ""
    experimental_instruction = ""
    if isinstance(context, ExperimentalModule2DiagnosisContext):
        experimental_instruction = (
            " The program supplies bounded Repair responsibilities and Preservation "
            "responsibilities. Focus only on unresolved repair items and provide the "
            "minimum additional support required by the team. Do not propagate an "
            "item after the team decision is already correct. Preserve listed existing "
            "capabilities and do not trade useful competence for isolated local gains."
        )
        if context.context_variant == C3_COALITION_AWARE_PRESERVATION:
            experimental_instruction += (
                " Treat repair_distance=1 items as high-priority decision boundaries; "
                "use only minimum redundancy for fragmented items. Preservation tier P1 "
                "is strong vote-critical guidance, P2 is soft coalition-support guidance, "
                "and P3 is stable-competence guidance. These labels guide generation only."
            )
    if isinstance(context, SingleLaneDiagnosisContext):
        lane_instruction = (
            " The program has already selected the sole RepairLane. failure_pattern "
            "must summarize only that lane; repair_rule must address only that lane "
            "and must not add coverage, conversion, margin, dominant-wrong, or any "
            "other parallel repair objective. preservation_rule may only protect "
            "existing behavior and must not introduce another repair target."
        )
    return (
        "Propose one task-general, testable prompt repair plan from the typed aggregate "
        "diagnosis. Do not quote cases or answers, describe per-case transitions, copy a "
        "peer procedure, predict performance, or generate candidate prompts. The repair "
        "rule must specify executable behavior and integrate uncertainty handling. The "
        "preservation rule must protect correct behavior and the strict output contract. "
        + feedback_instruction
        + lane_instruction
        + experimental_instruction
        + diagnosis_instruction
        + f"Return strict JSON with exactly these fields: {json.dumps(schema)}\n"
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
    evolution_variant: str = M20_CURRENT_V15,
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
    diagnosis_revision = (
        "Return all eight required fields, including the complete compact residual "
        "diagnosis and edit plan."
        if evolution_variant != M20_CURRENT_V15
        else "Return all three original fields, not a patch or commentary."
    )
    return (
        "Revise the previous TeacherRepairPlan into a complete replacement plan. "
        + diagnosis_revision
        + " Address every "
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
        f"PreviousTeacherRepairPlan:\n{json.dumps(teacher_repair_plan_payload(previous_plan), ensure_ascii=False, sort_keys=True)}\n"
        f"CriticDecision:\n{json.dumps(critic_payload, ensure_ascii=False, sort_keys=True)}\n"
        f"DiagnosisContextHash: {context_hash}"
    )


def build_critic_request(
    context: AnyDiagnosisContext,
    repair_plan: TeacherRepairPlan,
    *,
    feedback_max_chars: int = 500,
    evolution_variant: str = M20_CURRENT_V15,
) -> str:
    schema = {"failed_checks": [], "risk_case_ids": [], "feedback": ""}
    lane_instruction = ""
    if isinstance(context, SingleLaneDiagnosisContext):
        lane_instruction = (
            " A plan that introduces more than the supplied repair lane must fail "
            "actionable_specificity. A failure_pattern inconsistent with the supplied "
            "repair_lane must fail evidence_mismatch."
        )
    if evolution_variant == M2C_DIAGNOSIS_MINIMAL_EDIT_RELEVANCE_CRITIC:
        return (
            "Check only semantic grounding and edit relevance. Allowed failed_checks are "
            f"{json.dumps(RELEVANCE_CRITIC_FAILED_CHECKS)}. Verify that the diagnosis is "
            "grounded in assigned residual evidence, member-specific, behavioral, "
            "actionable, non-memorizing, and that the edit directly addresses it while "
            "preserving the broad parent role. Reject a clearly unrelated broad rewrite "
            "or schema/output-contract violation. Do not predict downstream accuracy, "
            "preservation loss, collateral risk, or candidate performance; actual fixed-"
            "probe common-safe evaluation handles those outcomes. risk_case_ids may only "
            "name supplied case IDs. Use empty feedback when approved; otherwise give one "
            f"concrete revision within {feedback_max_chars} characters. Return exactly: "
            f"{json.dumps(schema)}\nDiagnosisContext:\n{serialize_context(context)}\n"
            f"TeacherRepairPlan:\n{json.dumps(teacher_repair_plan_payload(repair_plan), ensure_ascii=False, sort_keys=True)}"
        )
    return (
        "Check only explicit hard blockers in the repair plan. Allowed failed_checks are "
        f"{json.dumps(CRITIC_FAILED_CHECKS)}. evidence_mismatch means a clear conflict "
        "with the aggregate or representative evidence; actionable_specificity means the "
        "rule is generic, non-executable, or contradictory; shortcut_or_copying means "
        "sample memorization, specific-answer or peer copying, or stereotype shortcuts; "
        "preservation_or_output_risk means preservation is inoperable or the strict output "
        "contract is endangered. Reject a repair plan under preservation_or_output_risk "
        "if it directs the Student to emit, copy, specialize, or hard-code the solver "
        "output interface, a FINAL_ANSWER line, or a fixed answer label. Do not score, "
        "predict candidate performance, restate "
        "facts, or report soft concerns. risk_case_ids may only name supplied case IDs. "
        + lane_instruction
        + " "
        "Use empty feedback when approved; when rejecting give one concrete revision, at "
        f"most {feedback_max_chars} characters. "
        f"CriticFeedbackMaxCharacters: {feedback_max_chars}. "
        f"Return exactly: {json.dumps(schema)}\n"
        f"DiagnosisContext:\n{serialize_context(context)}\n"
        f"TeacherRepairPlan:\n{json.dumps(teacher_repair_plan_payload(repair_plan), ensure_ascii=False, sort_keys=True)}"
    )


def build_student_request(
    *,
    parent_prompt: str,
    approved_plan: TeacherRepairPlan,
    answer_format: str,
    candidate_count: int,
    candidate_prompt_max_chars: int,
    total_candidate_prompt_max_chars: int = 5000,
    single_lane: bool = False,
    evolution_variant: str = M20_CURRENT_V15,
) -> str:
    lane_instruction = (
        " Implement only the current repair lane's one core rule. Prefer replacing or "
        "merging an old rule instead of appending sections; do not add a second unrelated "
        "strategy, and preserve parent rules that remain valid."
        if single_lane else ""
    )
    minimal_instruction = (
        " Make only the smallest role-consistent change required by the residual "
        "diagnosis and edit plan. Preserve the parent member's existing role and "
        "general strategy. Prefer replacing or merging one relevant rule; do not "
        "perform a complete rewrite, add unrelated frameworks, change role identity, "
        "or memorize examples. Minimal edit is guidance, not an output-length gate."
        if evolution_variant in {
            M2B_DIAGNOSIS_MINIMAL_EDIT,
            M2C_DIAGNOSIS_MINIMAL_EDIT_RELEVANCE_CRITIC,
        } else ""
    )
    return (
        "Implement the approved repair plan as complete replacement decision procedures. "
        "Generate only the mutable reasoning procedure and decision process. Each candidate must "
        "stand alone, contain no training example or answer, and be no longer than the "
        "stated limit. Do not include, quote, imitate, or describe the solver output "
        "interface. Do not include any FINAL_ANSWER line or fixed answer value. Do not "
        "append an answer letter, label, placeholder, example output, JSON answer field, "
        "or formatting instruction for the final response. The immutable output interface "
        "for the Solver is appended later by the program and is not part of the candidate "
        "prompt. Do not return a patch or diagnosis metadata. Clean example: Identify the "
        "pronoun, enumerate grammatically compatible antecedents, compare contextual "
        "evidence, and select the best-supported interpretation. Invalid example (do not "
        "copy): Identify the pronoun and select the best interpretation. FINAL_ANSWER: A. "
        + lane_instruction
        + minimal_instruction
        + " "
        "Return strict JSON with the sole field candidate_prompts.\n"
        f"StudentPromptContractVersion: {STUDENT_PROMPT_CONTRACT_VERSION}\n"
        f"ParentPrompt:\n{parent_prompt}\n"
        f"ApprovedRepairPlan:\n{json.dumps(teacher_repair_plan_payload(approved_plan), ensure_ascii=False, sort_keys=True)}\n"
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
    requirements = [
        "Return candidate_prompts as a JSON array.",
        "Every candidate must be a non-empty string.",
        "Do not return null, objects, nested arrays, or empty strings.",
        "Each candidate must differ from the parent prompt and other candidates.",
        "Return only the required schema.",
    ]
    if "output_contract_contamination" in previous_rejection_classes:
        requirements.append(
            "A candidate included the immutable solver output interface. Return only "
            "the mutable decision procedure and do not include FINAL_ANSWER or any "
            "fixed answer."
        )
    feedback = {
        "student_recovery": True,
        "previous_rejection_classes": list(previous_rejection_classes),
        "required_candidate_count": int(required_candidate_count),
        "parent_prompt_hash": str(parent_prompt_hash),
        "approved_repair_plan_hash": str(approved_repair_plan_hash),
        "requirements": requirements,
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
    evolution_variant: str = M20_CURRENT_V15,
) -> str:
    payload = {
        "student_upstream_regeneration": True,
        "previous_approved_plan_hash": str(previous_plan_hash),
        "student_rejection_classes": list(student_rejection_classes),
        "requirements": [
            "Produce a materially different complete repair plan.",
            (
                "Return failure_pattern, repair_rule, preservation_rule, and all "
                "five residual-diagnosis/edit-plan fields."
                if evolution_variant != M20_CURRENT_V15
                else "Return failure_pattern, repair_rule, and preservation_rule."
            ),
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
    evolution_variant: str = M20_CURRENT_V15,
) -> TeacherRepairPlan:
    base = {"failure_pattern", "repair_rule", "preservation_rule"}
    diagnosis = {
        "diagnosis_primary_failure_mode", "diagnosis_evidence_patterns",
        "diagnosis_peer_contrast", "diagnosis_desired_behavior_changes",
        "edit_plan",
    }
    expected = base if evolution_variant == M20_CURRENT_V15 else base | diagnosis
    if set(payload) != expected:
        raise ValueError("teacher response must contain exactly three repair-plan fields")
    values: dict[str, Any] = {}
    for field in sorted(base):
        value = payload[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"teacher field {field} must be a non-empty string")
        value = value.strip()
        if len(value) > field_max_chars:
            raise ValueError(f"teacher field {field} exceeds character limit")
        values[field] = value
    if evolution_variant != M20_CURRENT_V15:
        for field in ("diagnosis_primary_failure_mode", "diagnosis_peer_contrast"):
            value = payload[field]
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > field_max_chars:
                raise ValueError(f"teacher diagnosis field {field} is invalid")
            values[field] = value.strip()
        for field in ("diagnosis_evidence_patterns", "diagnosis_desired_behavior_changes", "edit_plan"):
            value = payload[field]
            if not isinstance(value, list) or not 1 <= len(value) <= 2:
                raise ValueError(f"teacher diagnosis field {field} must contain one or two items")
            if any(not isinstance(row, str) or not row.strip() or len(row.strip()) > field_max_chars for row in value):
                raise ValueError(f"teacher diagnosis field {field} contains an invalid item")
            values[field] = tuple(row.strip() for row in value)
        generic = " ".join(values["diagnosis_primary_failure_mode"].lower().split())
        if generic in {"reason more carefully", "improve accuracy", "be more careful", "think carefully"}:
            raise ValueError("teacher residual diagnosis is generic")
    total_chars = sum(
        len(item) if isinstance(item, str) else sum(map(len, item))
        for item in values.values()
    )
    if total_chars > total_max_chars:
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
    evolution_variant: str = M20_CURRENT_V15,
) -> CriticDecision:
    expected = {"failed_checks", "risk_case_ids", "feedback"}
    if set(payload) != expected:
        raise ValueError("critic response must contain exactly three fields")
    failed = payload["failed_checks"]
    risk_ids = payload["risk_case_ids"]
    feedback = payload["feedback"]
    if not isinstance(failed, list) or any(not isinstance(row, str) for row in failed):
        raise ValueError("failed_checks must be a list of strings")
    allowed_checks = (
        RELEVANCE_CRITIC_FAILED_CHECKS
        if evolution_variant == M2C_DIAGNOSIS_MINIMAL_EDIT_RELEVANCE_CRITIC
        else CRITIC_FAILED_CHECKS
    )
    if len(set(failed)) != len(failed) or any(row not in allowed_checks for row in failed):
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
            if mutable_prompt_violation_reasons(prompt):
                reasons.append("output_contract_contamination")
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
