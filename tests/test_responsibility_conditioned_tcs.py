from multi_dataset_diverse_rl.tcs import (
    STUDENT_REQUIRED_FIELDS, build_responsibility_context, critic_rejects_surface_rewrite,
    teacher_instruction, validate_student_candidate,
)


def test_teacher_context_contains_assigned_peer_state_and_preservation():
    assigned = [{
        "question": "q", "gold_answer": "A", "target_current_answer": "B",
        "peer_answer_histogram": {"B": 3}, "G": 1, "H": 3, "M": -2,
        "direct_vote_fix": 1, "fix_soft_utility_gain": 0.4, "peer_gold_vote_count": 1,
    }]
    preservation = [{"question": "p", "unique_correct": True, "pivotal_vote_correct": True}]
    context = build_responsibility_context(
        target_agent_id=2, assigned_cases=assigned, preservation_cases=preservation,
        representative_cases=[],
    )
    serialized = str(context)
    assert all(token in serialized for token in ("peer_answer_histogram", "'G': 1", "'H': 3", "'M': -2"))
    assert context["preservation_unique_correct_cases"]
    assert context["preservation_pivotal_vote_cases"]
    instruction = teacher_instruction(context, responsibility_conditioned=True, accuracy_only=False).lower()
    assert "preset role" in instruction
    assert "preserve" in instruction


def test_student_schema_has_only_responsibility_fields_and_surface_cot_is_rejected():
    candidate = {field: field for field in STUDENT_REQUIRED_FIELDS}
    assert validate_student_candidate(candidate) == []
    assert "diversity_contribution" not in STUDENT_REQUIRED_FIELDS
    assert "role_description" not in STUDENT_REQUIRED_FIELDS
    assert critic_rejects_surface_rewrite("Think step by step and be careful") is True
    assert critic_rejects_surface_rewrite("When ambiguity remains, compare each referent and verify agreement") is False


def test_accuracy_teacher_instruction_has_no_peer_credit_language():
    instruction = teacher_instruction(
        {"representative_cases": [{"question": "q", "gold_answer": "A"}]},
        responsibility_conditioned=False,
        accuracy_only=True,
    )
    assert "peer answer histogram" not in instruction.lower()
    assert "soft-utility" not in instruction.lower()
