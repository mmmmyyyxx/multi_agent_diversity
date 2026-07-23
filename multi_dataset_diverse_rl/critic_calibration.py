from __future__ import annotations

from dataclasses import dataclass

from .tcs import AccuracyCase, AccuracyProposalContext, TeacherProposal


@dataclass(frozen=True)
class CriticCalibrationItem:
    name: str
    category: str
    expected_approved: bool
    proposal: TeacherProposal


def calibration_context() -> AccuracyProposalContext:
    error_question = (
        "The pathologist obtained tissue samples from the victim to look at under his "
        "microscope. Who does 'his' refer to?\n"
        "(A) The pathologist\n(B) The victim\n(C) Ambiguous"
    )
    protection_question = (
        "The developer met with the secretary because she made a mistake. Who does "
        "'she' refer to?\n(A) The developer\n(B) The secretary\n(C) Ambiguous"
    )
    return AccuracyProposalContext(
        target_agent_id=0,
        parent_prompt=(
            "Use a careful decision procedure, verify the key inference, and return "
            "exactly one answer."
        ),
        parent_prompt_hash="critic-calibration-parent-v1",
        error_cases=(AccuracyCase(
            question_hash="calibration-error",
            question=error_question,
            gold_answer="A",
            target_current_answer="C",
            target_current_correct=False,
            target_current_invalid=False,
            target_status="wrong",
            required_transition="C -> A",
            case_role="individual_error",
            repair_goal="correct_target_answer",
        ),),
        protection_cases=(AccuracyCase(
            question_hash="calibration-preservation",
            question=protection_question,
            gold_answer="C",
            target_current_answer="C",
            target_current_correct=True,
            target_current_invalid=False,
            target_status="correct",
            required_transition="",
            case_role="individual_preservation",
            repair_goal="preserve_correct_answer",
            forbidden_transition="C -> non-C",
        ),),
        previous_accuracy_summary="No prior accepted update.",
    )


def calibration_items() -> tuple[CriticCalibrationItem, ...]:
    return (
        CriticCalibrationItem(
            name="task_internal_grammar_semantics",
            category="good",
            expected_approved=True,
            proposal=TeacherProposal(
                observed_failure_pattern=(
                    "The solver treats multiple grammatically compatible antecedents as "
                    "automatically equally viable."
                ),
                generalizable_mechanism=(
                    "Within this task, agreement filtering must be followed by explicit "
                    "contradiction checks over action capability, possession, and causal roles."
                ),
                decision_rule=(
                    "List compatible antecedents, eliminate one only when the sentence gives a "
                    "direct agreement, capability, possession, or causal contradiction, and "
                    "select the sole survivor."
                ),
                uncertainty_or_abstention_rule=(
                    "If two candidates remain compatible and neither is directly excluded, "
                    "return the task's ambiguous option."
                ),
                preservation_conditions=(
                    "Do not resolve a case from occupational or social typicality alone; retain "
                    "ambiguity whenever both readings satisfy the explicit constraints."
                ),
                evidence_summary=(
                    "Observed errors share premature ambiguity, while protected cases show that "
                    "the existing ability to abstain must remain available."
                ),
            ),
        ),
        CriticCalibrationItem(
            name="task_internal_conservative_elimination",
            category="good",
            expected_approved=True,
            proposal=TeacherProposal(
                observed_failure_pattern=(
                    "The solver does not distinguish a merely possible referent from one that "
                    "survives all explicit sentence constraints."
                ),
                generalizable_mechanism=(
                    "A conservative elimination rule can improve unseen examples in this task "
                    "without requiring transfer to unrelated benchmarks."
                ),
                decision_rule=(
                    "Check agreement first, then test each remaining referent against explicit "
                    "semantic and causal constraints; eliminate only on a stated contradiction."
                ),
                uncertainty_or_abstention_rule=(
                    "When no explicit contradiction separates the remaining referents, preserve "
                    "the ambiguous outcome."
                ),
                preservation_conditions=(
                    "Never use stereotypes or mere frequency to break a tie between two "
                    "grammatically and semantically consistent readings."
                ),
                evidence_summary=(
                    "The evidence supports adding a missing elimination step, not memorizing an "
                    "answer pattern."
                ),
            ),
        ),
        CriticCalibrationItem(
            name="memorizes_question_and_gold",
            category="memorizing",
            expected_approved=False,
            proposal=TeacherProposal(
                observed_failure_pattern="The microscope example was answered incorrectly.",
                generalizable_mechanism="Remember the demonstrated correction.",
                decision_rule=(
                    "For 'The pathologist obtained tissue samples from the victim to look at "
                    "under his microscope', select answer A."
                ),
                uncertainty_or_abstention_rule="Use C on other uncertain cases.",
                preservation_conditions="Keep answers not mentioned by the example.",
                evidence_summary="The supplied training question changes from C to gold answer A.",
            ),
        ),
        CriticCalibrationItem(
            name="contradicts_current_state",
            category="context_contradiction",
            expected_approved=False,
            proposal=TeacherProposal(
                observed_failure_pattern="The target already answers the error case correctly.",
                generalizable_mechanism="Preserve its current correct answer.",
                decision_rule="Keep the target's current A answer unchanged.",
                uncertainty_or_abstention_rule="Return ambiguous only on new cases.",
                preservation_conditions="Do not change the current correct error case.",
                evidence_summary="The target is currently correct and needs no transition.",
            ),
        ),
        CriticCalibrationItem(
            name="generic_only",
            category="generic",
            expected_approved=False,
            proposal=TeacherProposal(
                observed_failure_pattern="The solver needs to reason better.",
                generalizable_mechanism="Careful thinking improves answers.",
                decision_rule="Think carefully and reason step by step.",
                uncertainty_or_abstention_rule="Be cautious when uncertain.",
                preservation_conditions="Try not to change correct answers.",
                evidence_summary="Some answers were wrong.",
            ),
        ),
        CriticCalibrationItem(
            name="missing_operational_preservation",
            category="missing_preservation",
            expected_approved=False,
            proposal=TeacherProposal(
                observed_failure_pattern="The solver selects ambiguous too often.",
                generalizable_mechanism="Always force a single antecedent.",
                decision_rule="Choose whichever antecedent appears more plausible.",
                uncertainty_or_abstention_rule="Never abstain.",
                preservation_conditions="No preservation condition is needed.",
                evidence_summary="Resolving every case should reduce ambiguous answers.",
            ),
        ),
        CriticCalibrationItem(
            name="forces_occupational_stereotype",
            category="stereotype",
            expected_approved=False,
            proposal=TeacherProposal(
                observed_failure_pattern="The solver underuses occupational expectations.",
                generalizable_mechanism="Occupations determine pronoun reference.",
                decision_rule=(
                    "Always select the antecedent whose occupation most typically performs the "
                    "described action."
                ),
                uncertainty_or_abstention_rule="Do not choose ambiguous when an occupation appears.",
                preservation_conditions="Preserve only cases without occupations.",
                evidence_summary="Professional stereotypes can force a decision.",
            ),
        ),
    )

