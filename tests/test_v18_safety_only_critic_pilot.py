from multi_dataset_diverse_rl.tcs import TeacherRepairPlan
from scripts.v18_safety_only_critic_pilot_support import classify, safety_only_decision
from scripts.run_v18_safety_only_critic_pilot import sanitized_critic_decisions


def plan(repair: str = "Compare plausible antecedents using grammar.", preservation: str = "Keep unrelated reasoning behavior unchanged.") -> TeacherRepairPlan:
    return TeacherRepairPlan(failure_pattern="Pronoun ambiguity is mishandled.", repair_rule=repair, preservation_rule=preservation)


def test_safety_only_gate_approves_semantic_quality_without_predicting_quality():
    assert safety_only_decision(plan())["failed_checks"] == []


def test_safety_only_gate_rejects_cheating_and_output_contract_contamination():
    assert safety_only_decision(plan("Always select option A."))["failed_checks"] == ["shortcut_or_copying"]
    assert safety_only_decision(plan("Return only the final answer."))["failed_checks"] == ["preservation_or_output_risk"]


def test_classifier_requires_two_additional_student_reaches():
    base = {"student_reach_rate": 2/6, "feasible_per_branch": 1/6, "would_commit_per_branch": 1/6, "validation_vote_delta_sum": 0, "feasible_per_student": 0.5}
    safety = {"student_reach_rate": 4/6, "feasible_per_branch": 2/6, "would_commit_per_branch": 1/6, "validation_vote_delta_sum": 0, "feasible_per_student": 0.5}
    assert classify({"canonical_llm": base, "deterministic_safety_only": safety}) == "SAFETY_ONLY_CRITIC_SUPPORTED"


def test_sanitized_critic_decisions_preserve_category_without_text():
    class Stub:
        tcs_rounds = [{
            "role": "critic",
            "schema_valid": True,
            "failed_checks": ["preservation_or_output_risk"],
            "feedback": "must never be persisted",
            "response_excerpt": "must never be persisted",
            "effective_approved": False,
            "semantic_round": 1,
            "format_attempt": 0,
            "teacher_plan_hash": "a" * 64,
            "critic_decision_hash": "b" * 64,
        }]

    rows = sanitized_critic_decisions(Stub(), "deterministic_safety_only")
    assert rows == [{
        "schema_valid": True,
        "failed_checks": ["preservation_or_output_risk"],
        "rejection_category": "explicit_output_contract",
        "critic_approved": False,
        "semantic_round": 1,
        "retry_index": 0,
        "teacher_plan_hash": "a" * 64,
        "critic_decision_hash": "b" * 64,
    }]
