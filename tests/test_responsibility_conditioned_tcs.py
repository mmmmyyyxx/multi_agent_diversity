from dataclasses import asdict, replace

import pytest

from multi_dataset_diverse_rl.tcs import (
    AccuracyCase,
    AccuracyProposalContext,
    PeerStateCase,
    PeerStateProposalContext,
    PreservationCase,
    RepresentativeCase,
    MemberAwareResponsibilityCase,
    MemberAwareResponsibilityProposalContext,
    TCSContextLimits,
    TeacherProposal,
    build_critic_request,
    build_student_request,
    build_teacher_request,
    limit_proposal_context,
    parse_critic_decision,
    parse_student_candidates,
    proposal_context_case_facts,
)


def member_responsibility_case(question_hash="q", owner_age=2, gain=0.4):
    return MemberAwareResponsibilityCase(
        question_hash=question_hash,
        question=f"question-{question_hash}",
        gold_answer="A",
        target_current_answer="D",
        team_G=1,
        team_H=3,
        team_M=-2,
        team_wrong_histogram=(("B", 3), ("D", 1)),
        peer_G=1,
        peer_H=3,
        peer_M=-2,
        peer_wrong_histogram=(("B", 3),),
        direct_vote_fix=True,
        oracle_soft_utility_gain=gain,
        dominant_wrong_member=False,
        target_status="wrong",
        required_transition="D -> A",
        team_vote_status="wrong",
        case_role="assigned_coverage",
        repair_goal="introduce_first_gold_vote",
        responsibility_reason="assigned residual owner",
        member_correct_count=4,
        team_correct_count_sum=30,
        improvement_need=10,
        owner_age=owner_age,
    )


def member_responsibility_context(count=1):
    cases = tuple(member_responsibility_case(f"q{i}", owner_age=i, gain=i / 10) for i in range(count))
    conversion_cases = tuple(replace(
        row,
        case_role="assigned_conversion",
        repair_goal="convert_plurality_vote_to_gold",
    ) for row in cases)
    return MemberAwareResponsibilityProposalContext(
        target_agent_id=4,
        parent_prompt="parent decision procedure",
        parent_prompt_hash="parent-hash",
        member_correct_counts=(8, 7, 6, 5, 4),
        member_gains_from_initial=(0, 0, 1, 0, -1),
        target_improvement_need=10,
        assigned_coverage_cases=cases,
        assigned_conversion_cases=conversion_cases,
        member_error_cases=cases,
        preservation_cases=(PreservationCase(
            question_hash="protected",
            question="protected question",
            gold_answer="A",
            target_current_answer="A",
            unique_correct=True,
            pivotal_correct=True,
            team_margin=1,
            peer_G=0,
            peer_H=2,
            peer_M=-2,
            peer_wrong_histogram=(("B", 2),),
            target_status="correct",
            required_transition="",
            team_vote_status="correct",
            case_role="pivotal_preservation",
            repair_goal="preserve_correct_vote",
            forbidden_transition="A -> non-A",
        ),),
        representative_cases=(RepresentativeCase(
            question_hash="representative",
            question="representative question",
            gold_answer="A",
            target_current_answer="B",
            target_current_correct=False,
            target_current_invalid=False,
            target_status="wrong",
            required_transition="B -> A",
            team_vote_status="wrong",
            case_role="representative_error",
            repair_goal="repair_representative_error",
        ),),
        responsibility_summary="Agent 4 owns one residual.",
        previous_member_summary="No prior assigned repair.",
    )


def peer_context():
    row = member_responsibility_case()
    peer = PeerStateCase(**{
        key: value for key, value in row.__dict__.items()
        if key not in {
            "responsibility_reason",
            "member_correct_count",
            "team_correct_count_sum",
            "improvement_need",
            "owner_age",
        }
    })
    peer = replace(peer, case_role="coverage", repair_goal="introduce_first_gold_vote")
    protected = member_responsibility_context().preservation_cases
    representative = member_responsibility_context().representative_cases
    return PeerStateProposalContext(
        target_agent_id=4,
        parent_prompt="parent decision procedure",
        parent_prompt_hash="parent-hash",
        coverage_cases=(peer,),
        conversion_cases=(peer,),
        preservation_cases=protected,
        representative_cases=representative,
        previous_vote_competence_summary="Vote unchanged; competence improved by one.",
    )


def accuracy_context():
    row = AccuracyCase(
        "q", "question", "A", "B", False, False,
        "wrong", "B -> A", "individual_error", "correct_target_answer",
    )
    protected = AccuracyCase(
        "p", "protected", "A", "A", True, False,
        "correct", "", "individual_preservation", "preserve_correct_answer",
        "A -> non-A",
    )
    return AccuracyProposalContext(
        target_agent_id=4,
        parent_prompt="parent decision procedure",
        parent_prompt_hash="parent-hash",
        error_cases=(row,),
        protection_cases=(protected,),
        previous_accuracy_summary="Correct count improved by one; invalid count unchanged.",
    )


def proposal():
    return TeacherProposal(
        observed_failure_pattern="ambiguous reference",
        generalizable_mechanism="candidate referents are not filtered by explicit constraints",
        decision_rule="compare candidate referents and verify constraints",
        uncertainty_or_abstention_rule="retain ambiguity when no candidate is excluded",
        preservation_conditions="keep established answers unless a contradiction is found",
        evidence_summary="assigned cases share a missing referent check",
    )


def critic_payload(context, **overrides):
    payload = {
        "case_fact_restatements": [
            asdict(row) for row in proposal_context_case_facts(context)
        ],
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
        "score": 0.2,
        "feedback": "worth testing",
    }
    payload.update(overrides)
    return payload


def test_teacher_critic_student_share_parent_and_typed_context():
    proposal_context = member_responsibility_context()
    teacher = build_teacher_request(proposal_context)
    critic = build_critic_request(proposal_context, proposal())
    student = build_student_request(proposal_context, proposal(), 2)
    for request in (teacher, critic, student):
        assert "parent decision procedure" in request
        assert "parent-hash" in request
        assert '"question_hash":"q0"' in request
        assert '"peer_wrong_histogram":[["B",3]]' in request
        assert '"D"' not in request.split('"peer_wrong_histogram"', 1)[1].split("]", 2)[0]
    assert "TeacherProposal" in critic
    assert "ApprovedTeacherProposal" in student
    assert "Stage A/B rollout decides effectiveness" in critic
    assert "unseen examples within the current task" in teacher
    assert "unrelated benchmarks" in student


def test_context_types_enforce_ablation_information_boundaries():
    proposal_context = peer_context()
    teacher = build_teacher_request(proposal_context)
    critic = build_critic_request(proposal_context, proposal())
    student = build_student_request(proposal_context, proposal(), 2)
    assert "peer-state failure" in teacher
    assert "assigned residual targeting" not in critic
    assert "assigned responsibilities" not in student
    assert "peer-state evidence" in student
    for forbidden in ("assigned_", "owner_age", "responsibility_reason", "responsibility_summary"):
        assert forbidden not in teacher

    accuracy = build_teacher_request(accuracy_context())
    for forbidden in ('"team_G"', '"peer_G"', "vote", "responsibility", "owner_age"):
        assert forbidden not in accuracy

    responsibility = build_teacher_request(member_responsibility_context())
    assert "assigned_coverage_cases" in responsibility
    assert "owner_age" in responsibility
    assert "responsibility_reason" in responsibility


def test_critic_approval_uses_hard_checks_not_score():
    context = accuracy_context()
    valid = critic_payload(context, score=0.01)
    decision = parse_critic_decision(valid, context)
    assert decision.approved is True
    assert decision.score == 0.01
    assert decision.hard_checks["context_consistent"] is True

    blocked = critic_payload(
        context,
        executable_change=False,
        blocking_reasons=["no executable behavior"],
        score=1.0,
    )
    assert parse_critic_decision(blocked, context).approved is False


def test_critic_hard_check_boolean_and_fact_restatement_are_strict():
    context = accuracy_context()
    valid = critic_payload(context)
    with pytest.raises(ValueError, match="JSON boolean"):
        parse_critic_decision({**valid, "context_consistent": "true"}, context)
    wrong_facts = [dict(row) for row in valid["case_fact_restatements"]]
    wrong_facts[0]["target_status"] = "misread"
    with pytest.raises(ValueError, match="restatement mismatch"):
        parse_critic_decision({**valid, "case_fact_restatements": wrong_facts}, context)
    with pytest.raises(ValueError, match="disagree"):
        parse_critic_decision(
            {**valid, "context_consistent": False, "blocking_reasons": []},
            context,
        )


def test_student_schema_is_typed_and_missing_fields_fail():
    payload = {"candidates": [{
        "candidate_prompt": "new prompt",
        "observed_failure_pattern": "failure",
        "generalizable_mechanism": "mechanism",
        "decision_rule": "repair",
        "uncertainty_or_abstention_rule": "abstain",
        "preservation_conditions": "preserve",
        "evidence_summary": "evidence",
    }]}
    assert parse_student_candidates(payload, expected_count=1)[0].candidate_prompt == "new prompt"
    del payload["candidates"][0]["decision_rule"]
    with pytest.raises(KeyError):
        parse_student_candidates(payload, expected_count=1)


@pytest.mark.parametrize("count", [0, 2])
def test_student_schema_requires_exact_candidate_count(count):
    row = {
        "candidate_prompt": "new prompt",
        "observed_failure_pattern": "failure",
        "generalizable_mechanism": "mechanism",
        "decision_rule": "repair",
        "uncertainty_or_abstention_rule": "abstain",
        "preservation_conditions": "preserve",
        "evidence_summary": "evidence",
    }
    with pytest.raises(ValueError, match="candidate count"):
        parse_student_candidates({"candidates": [row] * count}, expected_count=1)


def test_context_limits_are_deterministic_and_report_truncation():
    original = member_responsibility_context(8)
    left, left_diagnostics = limit_proposal_context(
        original,
        TCSContextLimits(
            assigned_coverage=3,
            assigned_conversion=2,
            preservation=1,
            representative=1,
            member_error=4,
            max_chars=5000,
        ),
    )
    right, right_diagnostics = limit_proposal_context(
        original,
        TCSContextLimits(
            assigned_coverage=3,
            assigned_conversion=2,
            preservation=1,
            representative=1,
            member_error=4,
            max_chars=5000,
        ),
    )
    assert left == right
    assert left_diagnostics == right_diagnostics
    assert len(left.assigned_coverage_cases) <= 3
    assert len(left.assigned_conversion_cases) <= 2
    assert len(left.member_error_cases) <= 4
    assert left_diagnostics.truncated_cases["coverage"] >= 5
    assert left_diagnostics.truncated_cases["member_error"] >= 4
    assert left_diagnostics.context_characters <= 5000
