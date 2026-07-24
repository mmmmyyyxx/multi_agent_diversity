from __future__ import annotations

from dataclasses import dataclass

from .tcs import (
    AccuracyDiagnosisContext,
    PreviousUpdateOutcome,
    TeacherRepairPlan,
)


@dataclass(frozen=True)
class CriticCalibrationItem:
    name: str
    category: str
    expected_approved: bool
    proposal: TeacherRepairPlan


def calibration_context() -> AccuracyDiagnosisContext:
    return AccuracyDiagnosisContext(
        target_agent_id=0,
        parent_prompt=(
            "Use a careful decision procedure, verify the key inference, and return "
            "exactly one answer."
        ),
        parent_prompt_hash="critic-calibration-parent-v2",
        target_correct_count=1,
        target_error_count=1,
        target_invalid_count=0,
        patterns=(),
        evidence_cases=(),
        previous_outcome=PreviousUpdateOutcome(),
    )


def calibration_items() -> tuple[CriticCalibrationItem, ...]:
    good = TeacherRepairPlan(
        failure_pattern="The solver stops before checking explicit semantic constraints.",
        repair_rule=(
            "List compatible referents, eliminate one only on an explicit agreement, "
            "capability, possession, or causal contradiction, and return ambiguity if "
            "multiple candidates still survive."
        ),
        preservation_rule=(
            "Retain an existing conclusion whenever it survives every explicit check; "
            "never break a tie using occupational or social typicality."
        ),
    )
    return (
        CriticCalibrationItem("task_internal_constraints", "good", True, good),
        CriticCalibrationItem(
            "memorizes_specific_answer",
            "memorizing",
            False,
            TeacherRepairPlan(
                failure_pattern="One supplied question was missed.",
                repair_rule="Memorize that the demonstrated question has answer A.",
                preservation_rule="Keep all other answers.",
            ),
        ),
        CriticCalibrationItem(
            "generic_only",
            "generic",
            False,
            TeacherRepairPlan(
                failure_pattern="The solver needs to reason better.",
                repair_rule="Think carefully and double-check.",
                preservation_rule="Try not to change correct answers.",
            ),
        ),
        CriticCalibrationItem(
            "missing_preservation",
            "missing_preservation",
            False,
            TeacherRepairPlan(
                failure_pattern="The solver abstains too often.",
                repair_rule="Always force the most plausible answer.",
                preservation_rule="No preservation rule is needed.",
            ),
        ),
        CriticCalibrationItem(
            "stereotype_shortcut",
            "stereotype",
            False,
            TeacherRepairPlan(
                failure_pattern="The solver underuses occupational expectations.",
                repair_rule="Select whichever occupation typically performs the action.",
                preservation_rule="Preserve only cases without occupations.",
            ),
        ),
    )
