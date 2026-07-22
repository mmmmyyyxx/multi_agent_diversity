import pytest

from multi_dataset_diverse_rl.tcs import (
    AccuracyCase,
    AccuracyProposalContext,
    PeerStateCase,
    PeerStateProposalContext,
    PreservationCase,
    RepresentativeCase,
    ResponsibilityCase,
    ResponsibilityProposalContext,
    TCSContextLimits,
    TeacherProposal,
    build_critic_request,
    build_student_request,
    build_teacher_request,
    limit_proposal_context,
    parse_critic_decision,
    parse_student_candidates,
)


def responsibility_case(question_hash="q", owner_age=2, gain=0.4):
    return ResponsibilityCase(
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
        responsibility_reason="assigned residual owner",
        owner_age=owner_age,
    )


def responsibility_context(count=1):
    cases = tuple(responsibility_case(f"q{i}", owner_age=i, gain=i / 10) for i in range(count))
    return ResponsibilityProposalContext(
        target_agent_id=4,
        parent_prompt="parent decision procedure",
        parent_prompt_hash="parent-hash",
        assigned_coverage_cases=cases,
        assigned_conversion_cases=cases,
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
        ),),
        representative_cases=(RepresentativeCase(
            question_hash="representative",
            question="representative question",
            gold_answer="A",
            target_current_answer="B",
            target_current_correct=False,
            target_current_invalid=False,
        ),),
        responsibility_summary="Agent 4 owns one residual.",
        assigned_repair_history="No prior assigned repair.",
    )


def peer_context():
    row = responsibility_case()
    peer = PeerStateCase(**{
        key: value for key, value in row.__dict__.items()
        if key not in {"responsibility_reason", "owner_age"}
    })
    protected = responsibility_context().preservation_cases
    representative = responsibility_context().representative_cases
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
    row = AccuracyCase("q", "question", "A", "B", False, False)
    protected = AccuracyCase("p", "protected", "A", "A", True, False)
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
        target_failure_mechanism="ambiguous reference",
        repair_procedure="compare candidate referents and verify constraints",
        preservation_rule="keep established answers unless a contradiction is found",
        expected_effect="repair the assigned conversion case",
    )


def test_teacher_critic_student_share_parent_and_typed_context():
    proposal_context = responsibility_context()
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

    responsibility = build_teacher_request(responsibility_context())
    assert "assigned_coverage_cases" in responsibility
    assert "owner_age" in responsibility
    assert "responsibility_reason" in responsibility


def test_critic_bool_and_threshold_are_strict():
    valid = {"approved": True, "score": 0.8, "feedback": "sound", "rejection_reasons": []}
    assert parse_critic_decision(valid, 0.75).approved is True
    assert parse_critic_decision({**valid, "score": 0.7}, 0.75).approved is False
    with pytest.raises(ValueError, match="JSON boolean"):
        parse_critic_decision({**valid, "approved": "true"}, 0.75)
    with pytest.raises(KeyError):
        parse_critic_decision({"approved": True, "score": 1.0}, 0.75)


def test_student_schema_is_typed_and_missing_fields_fail():
    payload = {"candidates": [{
        "candidate_prompt": "new prompt",
        "target_failure_mechanism": "failure",
        "repair_procedure": "repair",
        "preservation_rule": "preserve",
        "expected_effect": "effect",
    }]}
    assert parse_student_candidates(payload, expected_count=1)[0].candidate_prompt == "new prompt"
    del payload["candidates"][0]["repair_procedure"]
    with pytest.raises(KeyError):
        parse_student_candidates(payload, expected_count=1)


@pytest.mark.parametrize("count", [0, 2])
def test_student_schema_requires_exact_candidate_count(count):
    row = {
        "candidate_prompt": "new prompt",
        "target_failure_mechanism": "failure",
        "repair_procedure": "repair",
        "preservation_rule": "preserve",
        "expected_effect": "effect",
    }
    with pytest.raises(ValueError, match="candidate count"):
        parse_student_candidates({"candidates": [row] * count}, expected_count=1)


def test_context_limits_are_deterministic_and_report_truncation():
    original = responsibility_context(8)
    left, left_diagnostics = limit_proposal_context(
        original,
        TCSContextLimits(assigned_coverage=3, assigned_conversion=2, preservation=1, representative=1, max_chars=5000),
    )
    right, right_diagnostics = limit_proposal_context(
        original,
        TCSContextLimits(assigned_coverage=3, assigned_conversion=2, preservation=1, representative=1, max_chars=5000),
    )
    assert left == right
    assert left_diagnostics == right_diagnostics
    assert len(left.assigned_coverage_cases) <= 3
    assert len(left.assigned_conversion_cases) <= 2
    assert left_diagnostics.truncated_cases["coverage"] >= 5
    assert left_diagnostics.context_characters <= 5000
