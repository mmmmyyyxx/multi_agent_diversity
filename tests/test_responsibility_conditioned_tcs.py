import json

import pytest

from multi_dataset_diverse_rl.diagnosis_aggregation import (
    AggregatedFailurePattern,
    CompactEvidenceCase,
    CompactLaneEvidenceCase,
    FailurePatternKey,
    LanePatternKey,
    LanePatternSummary,
)
from multi_dataset_diverse_rl.tcs import (
    AccuracyDiagnosisContext,
    AssignedResidualDiagnosisContext,
    CompactPreviousOutcome,
    PeerStateDiagnosisContext,
    PreviousUpdateOutcome,
    SingleLaneDiagnosisContext,
    TeacherRepairPlan,
    build_critic_request,
    build_student_request,
    build_teacher_request,
    build_teacher_revision_request,
    changed_teacher_plan_fields,
    compact_previous_outcome,
    context_payload,
    critic_decision_hash,
    parse_critic_decision,
    parse_student_candidates,
    parse_teacher_repair_plan,
    limit_diagnosis_context,
    serialize_context,
    teacher_repair_plan_hash,
)


def pattern() -> AggregatedFailurePattern:
    key = FailurePatternKey(
        case_family="conversion_failure",
        target_status="wrong",
        team_vote_status="wrong",
        target_answer_role="W1",
        gold_vote_count=2,
        largest_wrong_vote_count=3,
        plurality_margin=-1,
        peer_gold_vote_count=2,
        peer_largest_wrong_vote_count=2,
        peer_margin=0,
        vote_flip_gain=1,
        margin_gain=2,
        dominant_wrong_member=True,
        unique_correct=False,
        pivotal_correct=False,
    )
    return AggregatedFailurePattern(
        pattern_id="p1",
        key=key,
        case_count=4,
        assigned_case_count=2,
        direct_vote_fix_count=4,
        margin_gain_sum=8,
        dominant_wrong_count=4,
        mean_oracle_soft_utility_gain=0.25,
        max_oracle_soft_utility_gain=0.5,
        repair_goal="convert_existing_gold_coverage",
        represented_question_hashes=("q1", "q2", "q3", "q4"),
    )


def evidence() -> CompactEvidenceCase:
    return CompactEvidenceCase(
        case_id="c1",
        pattern_id="p1",
        case_family="conversion_failure",
        question_hash="q1",
        question="A sufficiently long representative question used for isolation checks?",
        gold_answer="A",
        target_current_answer="B",
        answer_role_signature=("W1", "G", "G", "W1", "W1"),
        target_answer_role="W1",
        gold_vote_count=2,
        largest_wrong_vote_count=3,
        plurality_margin=-1,
        peer_gold_vote_count=2,
        peer_largest_wrong_vote_count=2,
        peer_margin=0,
        vote_flip_gain=1,
        margin_gain=2,
        dominant_wrong_member=True,
        unique_correct=False,
        pivotal_correct=False,
        repair_goal="convert_existing_gold_coverage",
    )


def single_lane_context(parent_prompt="parent") -> SingleLaneDiagnosisContext:
    key = LanePatternKey("direct_flip", "dominant_wrong")
    pattern = LanePatternSummary("lane-p1", key, 5, 10)
    cases = tuple(
        CompactLaneEvidenceCase(
            case_id=f"lane-c{index}",
            question_hash=f"q{index}",
            question=f"Representative lane question {index} " + "x" * 80,
            gold_answer="A",
            target_current_answer="B",
            vote_flip_gain=1,
            margin_gain=2,
            dominant_wrong_member=True,
        )
        for index in range(2)
    )
    preservation = CompactLaneEvidenceCase(
        case_id="preserve-c1",
        question_hash="qp",
        question="Representative preservation question " + "y" * 80,
        gold_answer="A",
        target_current_answer="A",
        vote_flip_gain=0,
        margin_gain=0,
        dominant_wrong_member=False,
        unique_correct=True,
    )
    return SingleLaneDiagnosisContext(
        parent_prompt=parent_prompt,
        parent_prompt_hash="parent-hash",
        repair_lane="direct_flip",
        repair_goal="Convert existing correct coverage into a correct plurality.",
        active_residual_count=7,
        dominant_target_role="dominant_wrong",
        dominant_pattern_case_count=5,
        dominant_pattern=pattern,
        repair_cases=cases,
        preservation_case=preservation,
        previous_outcome=CompactPreviousOutcome(
            status="rejected", main_rejection="team_vote_regression"
        ),
    )


def test_single_lane_context_exposes_only_compact_directional_fields():
    context = single_lane_context()
    payload = context_payload(context)
    serialized = json.dumps(payload, sort_keys=True)
    assert payload["repair_lane"] == "direct_flip"
    assert payload["pattern_summary"] == {
        "repair_lane": "direct_flip",
        "repair_goal": context.repair_goal,
        "target_error_role": "dominant_wrong",
        "case_count": 5,
    }
    assert len(payload["repair_cases"]) == 2
    assert payload["preservation_case"] is not None
    forbidden = (
        "target_agent_id", "target_member_gain", "uplift_deficit",
        "direct_fix_responsibility_count", "margin_gain_responsibility_sum",
        "coverage_residual_count", "conversion_residual_count",
        "mean_oracle_soft_utility_gain", "answer_role_signature",
        "peer_gold_vote_count", "proposal_failure_feedback", "question_hash",
        "vote_flip_gain", "margin_gain", "dominant_wrong_member",
    )
    assert all(token not in serialized for token in forbidden)


def test_single_lane_context_limit_drops_preservation_then_second_repair_case():
    context = single_lane_context(parent_prompt="p" * 5200)
    bounded, diagnostics = limit_diagnosis_context(
        context,
        max_chars=10000,
        full_probe_case_count=20,
        available_pattern_count=2,
    )
    assert diagnostics.context_characters <= 6000
    assert diagnostics.selected_pattern_count == 1
    assert len(bounded.repair_cases) <= 2
    assert bounded.preservation_case is None
    assert len(serialize_context(bounded)) <= 6000


def test_single_lane_requests_enforce_one_direction_and_compact_outcome():
    context = single_lane_context()
    teacher = build_teacher_request(context)
    critic = build_critic_request(
        context,
        TeacherRepairPlan("direct flip", "check constraints", "keep valid rules"),
    )
    student = build_student_request(
        parent_prompt="parent",
        approved_plan=TeacherRepairPlan(
            "direct flip", "check constraints", "keep valid rules"
        ),
        answer_format="option_letter",
        candidate_count=2,
        candidate_prompt_max_chars=3000,
        single_lane=True,
    )
    assert "sole RepairLane" in teacher
    assert "more than the supplied repair lane" in critic
    assert "one core rule" in student
    assert compact_previous_outcome(PreviousUpdateOutcome()).status == "none"
    rejected = compact_previous_outcome(PreviousUpdateOutcome(
        attempted=True,
        empirical_evaluation_completed=True,
        rejection_reasons=("member_objective_regression", "team_vote_regression"),
    ))
    assert rejected == CompactPreviousOutcome(
        status="rejected", main_rejection="team_vote_regression"
    )
    semantic = compact_previous_outcome(PreviousUpdateOutcome(
        attempted=True,
        rejection_reasons=("semantic_rejection",),
    ))
    assert semantic == CompactPreviousOutcome(
        status="rejected", main_rejection="semantic_rejection"
    )


def contexts():
    common = dict(
        target_agent_id=0,
        parent_prompt="parent",
        parent_prompt_hash="parent-hash",
        patterns=(pattern(),),
        evidence_cases=(evidence(),),
        previous_outcome=PreviousUpdateOutcome(),
    )
    return (
        AccuracyDiagnosisContext(
            **common,
            target_correct_count=3,
            target_error_count=2,
            target_invalid_count=0,
        ),
        PeerStateDiagnosisContext(
            **common,
            vote_wrong_count=2,
            coverage_failure_count=0,
            conversion_failure_count=2,
            preservation_count=1,
        ),
        AssignedResidualDiagnosisContext(
            **common,
            assigned_residual_count=2,
            target_member_gain=1,
            uplift_deficit=3,
            direct_fix_responsibility_count=2,
            margin_gain_responsibility_sum=4,
            coverage_residual_count=0,
            conversion_residual_count=2,
            preservation_count=1,
        ),
    )


def test_context_serialization_isolates_accuracy_peer_and_member_fields():
    accuracy, peer, member = (context_payload(row) for row in contexts())
    accuracy_text = json.dumps(accuracy, sort_keys=True)
    peer_text = json.dumps(peer, sort_keys=True)
    member_text = json.dumps(member, sort_keys=True)
    for token in (
        "gold_vote_count", "plurality_margin", "peer_", "assigned",
        "member_gain", "improvement_need", "answer_role",
    ):
        assert token not in accuracy_text
    assert "gold_vote_count" in peer_text and "peer_gold_vote_count" in peer_text
    assert "assigned_case_count" not in peer_text
    assert "member_gains_from_initial" not in peer_text
    assert "\"target_member_gain\": 1" in member_text
    assert "\"uplift_deficit\": 3" in member_text
    assert "\"margin_gain_responsibility_sum\": 4" in member_text
    assert "assigned_case_count" in member_text
    for payload_text in (accuracy_text, peer_text, member_text):
        assert "represented_question_hashes" not in payload_text
        assert "question_hash" not in payload_text
        assert "parent_prompt_hash" not in payload_text
    assert pattern().represented_question_hashes == ("q1", "q2", "q3", "q4")

    plan = TeacherRepairPlan(
        "residual failure",
        "Apply the explicit rule before committing.",
        "Preserve outputs that pass the rule.",
    )
    for context in contexts():
        teacher_request = build_teacher_request(context)
        critic_request = build_critic_request(context, plan)
        for request in (teacher_request, critic_request):
            assert "represented_question_hashes" not in request
            assert "question_hash" not in request
            assert "parent_prompt_hash" not in request
            assert "pattern_id" in request
            assert "case_id" in request


def test_previous_outcome_hides_rollout_feedback_when_evaluation_did_not_run():
    accuracy, _, _ = contexts()
    failed_pipeline = type(accuracy)(
        **{
            **accuracy.__dict__,
            "previous_outcome": PreviousUpdateOutcome(attempted=True),
        }
    )
    assert context_payload(failed_pipeline)["previous_outcome"] == {
        "attempted": True,
        "empirical_feedback_available": False,
    }

    evaluated = type(accuracy)(
        **{
            **accuracy.__dict__,
            "previous_outcome": PreviousUpdateOutcome(
                attempted=True,
                empirical_evaluation_completed=True,
                accepted=False,
                rejection_reasons=("no_target_or_vote_progress",),
            ),
        }
    )
    assert context_payload(evaluated)["previous_outcome"] == {
        "attempted": True,
        "empirical_feedback_available": True,
        "accepted": False,
        "target_correct_delta": 0,
        "rejection_reasons": ("no_target_or_vote_progress",),
    }


def test_role_requests_state_configured_structural_character_limits():
    accuracy, _, _ = contexts()
    plan = TeacherRepairPlan(
        "residual failure",
        "Apply the explicit rule before committing.",
        "Preserve outputs that pass the rule.",
    )
    teacher_request = build_teacher_request(
        accuracy,
        field_max_chars=321,
        total_max_chars=654,
    )
    critic_request = build_critic_request(
        accuracy,
        plan,
        feedback_max_chars=123,
    )
    assert "TeacherFieldMaxCharacters: 321" in teacher_request
    assert "TeacherTotalMaxCharacters: 654" in teacher_request
    assert "CriticFeedbackMaxCharacters: 123" in critic_request
    assert "at most 123 characters" in critic_request


def test_teacher_revision_request_contains_full_prior_plan_and_critic_decision():
    accuracy, _, _ = contexts()
    previous = TeacherRepairPlan(
        "premature commitment",
        "Compare each option before committing.",
        "Preserve conclusions that pass every explicit check.",
    )
    critic = parse_critic_decision(
        {
            "failed_checks": ["evidence_mismatch"],
            "risk_case_ids": ["c1"],
            "feedback": "Use only observable task-input checks.",
        },
        allowed_case_ids={"c1"},
    )
    request = build_teacher_revision_request(
        context=accuracy,
        previous_plan=previous,
        critic_decision=critic,
    )
    assert "PreviousTeacherRepairPlan:" in request
    assert '"failure_pattern": "premature commitment"' in request
    assert '"repair_rule": "Compare each option before committing."' in request
    assert '"preservation_rule": "Preserve conclusions that pass every explicit check."' in request
    assert '"failed_checks": ["evidence_mismatch"]' in request
    assert '"risk_case_ids": ["c1"]' in request
    assert "Use only observable task-input checks." in request
    assert "DiagnosisContextHash:" in request
    assert evidence().question not in request
    assert "all four hard checks cumulatively" in request
    assert teacher_repair_plan_hash(previous)
    assert critic_decision_hash(critic)
    revised = TeacherRepairPlan(
        previous.failure_pattern,
        "Compare each option in order before committing.",
        previous.preservation_rule,
    )
    assert changed_teacher_plan_fields(previous, revised) == ("repair_rule",)


def test_teacher_schema_is_exact_nonempty_and_bounded():
    valid = {
        "failure_pattern": "premature commitment",
        "repair_rule": "Compare every option and abstain if evidence remains tied.",
        "preservation_rule": "Keep conclusions that still pass every explicit check.",
    }
    assert parse_teacher_repair_plan(valid).repair_rule.startswith("Compare")
    for mutation in (
        valid | {"evidence_summary": "old field"},
        valid | {"uncertainty_rule": "old field"},
        valid | {"failure_pattern": ""},
        valid | {"repair_rule": "x" * 801},
    ):
        with pytest.raises(ValueError):
            parse_teacher_repair_plan(mutation)
    with pytest.raises(ValueError, match="generic"):
        parse_teacher_repair_plan(valid | {"repair_rule": "think carefully"})


def test_critic_schema_computes_approval_and_rejects_old_or_unknown_fields():
    approved = parse_critic_decision(
        {"failed_checks": [], "risk_case_ids": [], "feedback": ""},
        allowed_case_ids={"c1"},
    )
    assert approved.approved
    rejected = parse_critic_decision(
        {
            "failed_checks": ["actionable_specificity"],
            "risk_case_ids": ["c1"],
            "feedback": "Make the rule executable.",
        },
        allowed_case_ids={"c1"},
    )
    assert not rejected.approved
    with pytest.raises(ValueError):
        parse_critic_decision(
            {"failed_checks": ["unknown"], "risk_case_ids": [], "feedback": "fix"},
            allowed_case_ids={"c1"},
        )
    with pytest.raises(ValueError):
        parse_critic_decision(
            {"failed_checks": [], "risk_case_ids": ["bad"], "feedback": ""},
            allowed_case_ids={"c1"},
        )
    for old_field in ("score", "soft_" + "concerns", "case_fact_" + "restatements"):
        with pytest.raises(ValueError):
            parse_critic_decision(
                {
                    "failed_checks": [],
                    "risk_case_ids": [],
                    "feedback": "",
                    old_field: [],
                },
                allowed_case_ids={"c1"},
            )
    with pytest.raises(ValueError):
        parse_critic_decision(
            {
                "failed_checks": ["evidence_mismatch"],
                "risk_case_ids": [],
                "feedback": "",
            },
            allowed_case_ids={"c1"},
        )


def test_student_request_is_context_free_and_partial_validity_is_retained():
    accuracy, _, _ = contexts()
    plan = TeacherRepairPlan(
        "premature commitment",
        "Compare each option and abstain only when explicit evidence remains tied.",
        "Preserve conclusions that pass every explicit check.",
    )
    request = build_student_request(
        parent_prompt="parent",
        approved_plan=plan,
        answer_format="option_letter",
        candidate_count=2,
        candidate_prompt_max_chars=6000,
    )
    for forbidden in ("question_hash", "gold_answer", "case_id", "gold_vote_count"):
        assert forbidden not in request
    assert "mutable reasoning procedure" in request
    assert "immutable output interface" in request
    assert "Do not include, quote, imitate, or describe" in request
    assert "OutputContract:" in request
    parsed = parse_student_candidates(
        {
            "candidate_prompts": [
                "parent",
                "A complete replacement procedure with an explicit verification step.",
                "A complete replacement procedure with an explicit verification step.",
            ]
        },
        parent_prompt="parent",
        context=accuracy,
        expected_count=3,
    )
    assert len(parsed.candidates) == 1
    assert parsed.rejection_reasons[0] == ("parent_identical",)
    assert parsed.rejection_reasons[2] == ("duplicate_candidate",)
    with pytest.raises(ValueError):
        parse_student_candidates(
            {"candidate_prompts": ["valid"], "metadata": "not allowed"},
            parent_prompt="parent",
            context=accuracy,
            expected_count=1,
        )


def test_student_rejects_excess_count_and_total_characters_without_truncating():
    accuracy, _, _ = contexts()
    with pytest.raises(ValueError, match="schema_invalid"):
        parse_student_candidates(
            {"candidate_prompts": ["first", "second", "third"]},
            parent_prompt="parent",
            context=accuracy,
            expected_count=2,
        )
    with pytest.raises(ValueError, match="too_long"):
        parse_student_candidates(
            {"candidate_prompts": ["a" * 6, "b" * 5]},
            parent_prompt="parent",
            context=accuracy,
            expected_count=2,
            candidate_prompt_max_chars=10,
            total_candidate_prompt_max_chars=10,
        )
    parsed = parse_student_candidates(
        {"candidate_prompts": ["valid replacement"]},
        parent_prompt="parent",
        context=accuracy,
        expected_count=2,
    )
    assert parsed.raw_count == 1
    assert [row.candidate_prompt for row in parsed.candidates] == [
        "valid replacement"
    ]
    parsed = parse_student_candidates(
        {"candidate_prompts": ["first valid", "second valid"]},
        parent_prompt="parent",
        context=accuracy,
        expected_count=2,
    )
    assert len(parsed.candidates) == 2
    parsed = parse_student_candidates(
        {"candidate_prompts": ["valid", "x" * 11]},
        parent_prompt="parent",
        context=accuracy,
        expected_count=2,
        candidate_prompt_max_chars=10,
        total_candidate_prompt_max_chars=20,
    )
    assert [row.candidate_prompt for row in parsed.candidates] == ["valid"]
    assert parsed.rejection_reasons[1] == ("too_long",)
