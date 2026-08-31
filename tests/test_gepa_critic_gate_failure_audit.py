from __future__ import annotations

from scripts.audit_gepa_critic_gate_failure import critic_group_status


def test_semantic_rejections_are_not_schema_failures() -> None:
    rows = [
        {
            "role": "critic",
            "schema_valid": True,
            "response_truncated": False,
            "failure_class": "semantic_rejection",
            "effective_approved": False,
        },
        {
            "role": "critic",
            "schema_valid": True,
            "response_truncated": False,
            "failure_class": "semantic_rejection",
            "effective_approved": False,
        },
    ]
    result = critic_group_status(rows)
    assert result["critic_semantic_rejection_count"] == 2
    assert result["critic_schema_invalid_count"] == 0
    assert result["critic_truncated_count"] == 0
    assert result["semantic_gate_exhausted"] is True
    assert result["student_reached"] is False


def test_second_round_recovery_reaches_student() -> None:
    rows = [
        {
            "role": "critic",
            "schema_valid": True,
            "response_truncated": False,
            "failure_class": "semantic_rejection",
            "effective_approved": False,
        },
        {
            "role": "critic",
            "schema_valid": True,
            "response_truncated": False,
            "failure_class": "",
            "effective_approved": True,
        },
        {"role": "student"},
    ]
    result = critic_group_status(rows)
    assert result["critic_semantic_rejection_count"] == 1
    assert result["critic_approved_count"] == 1
    assert result["semantic_gate_exhausted"] is False
    assert result["student_reached"] is True


def test_schema_failure_is_not_misclassified_as_semantic_exhaustion() -> None:
    rows = [{
        "role": "critic",
        "schema_valid": False,
        "response_truncated": False,
        "failure_class": "schema_error",
        "effective_approved": False,
    }]
    result = critic_group_status(rows)
    assert result["critic_schema_invalid_count"] == 1
    assert result["critic_semantic_rejection_count"] == 0
    assert result["semantic_gate_exhausted"] is False
