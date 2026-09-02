import json

from multi_dataset_diverse_rl.llm_client import LLMCallResult
from scripts.v18_shadow_raw_critic_support import classify, shadow_transform


def result(payload: dict) -> LLMCallResult:
    return LLMCallResult(json.dumps(payload), 1, 1, 2, 0.1, "stop")


def critic_prompt() -> str:
    return "TeacherRepairPlan:\n" + json.dumps({
        "failure_pattern": "Pronoun ambiguity is mishandled.",
        "repair_rule": "Compare plausible antecedents using grammar.",
        "preservation_rule": "Keep unrelated reasoning unchanged.",
    })


def test_shadow_transforms_only_valid_semantic_rejection():
    original = result({"failed_checks": ["actionable_specificity"], "risk_case_ids": [], "feedback": "too broad"})
    from multi_dataset_diverse_rl.tcs import TeacherRepairPlan, teacher_repair_plan_hash
    plan_hash = teacher_repair_plan_hash(TeacherRepairPlan(
        failure_pattern="Pronoun ambiguity is mishandled.",
        repair_rule="Compare plausible antecedents using grammar.",
        preservation_rule="Keep unrelated reasoning unchanged.",
    ))
    transformed, event = shadow_transform(original, critic_prompt(), {plan_hash})
    assert event is not None
    assert event["original_rejected"] is True
    assert event["original_failed_checks"] == ["actionable_specificity"]
    assert json.loads(transformed.text)["failed_checks"] == []
    assert "too broad" not in json.dumps(event)


def test_shadow_does_not_transform_approval_or_malformed_response():
    approved = result({"failed_checks": [], "risk_case_ids": [], "feedback": ""})
    assert shadow_transform(approved, critic_prompt(), set()) == (approved, None)
    malformed = result({"failed_checks": ["x"]})
    assert shadow_transform(malformed, critic_prompt(), set()) == (malformed, None)


def test_shadow_requires_control_parser_validated_rejection():
    rejected = result({"failed_checks": ["actionable_specificity"], "risk_case_ids": [], "feedback": "too broad"})
    assert shadow_transform(rejected, critic_prompt(), set()) == (rejected, None)


def test_classifier_is_frozen_before_results():
    assert classify(rejected_witnesses=2, feasible_branches=2, would_commit_branches=1, validation_vote_delta_sum=0) == "NO_CLEAR_SIGNAL"
    assert classify(rejected_witnesses=5, feasible_branches=0, would_commit_branches=0, validation_vote_delta_sum=0) == "CRITIC_FILTERING_JUSTIFIED"
    assert classify(rejected_witnesses=5, feasible_branches=2, would_commit_branches=1, validation_vote_delta_sum=-1) == "CRITIC_OVER_FILTERING_CAUSALLY_SUPPORTED"
    assert classify(rejected_witnesses=5, feasible_branches=1, would_commit_branches=1, validation_vote_delta_sum=0) == "MIXED_SHADOW_RAW_SIGNAL"
