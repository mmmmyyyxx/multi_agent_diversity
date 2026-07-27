from scripts.build_sanitized_pilot_report import specialization_rows


def test_legacy_specialization_alias_is_not_presented_as_repair_attribution():
    row = specialization_rows([{
        "artifact_schema_version": "specialization_trajectory_v1",
        "update_index": 1,
        "agent_id": 2,
        "selected_pattern_ids": ["pattern-a"],
        "accepted_repair_pattern_ids": ["pattern-a"],
        "prompt_hash_before": "before",
        "prompt_hash_after": "after",
    }])[0]
    assert row["legacy_selected_pattern_alias"] == ["pattern-a"]
    assert row["repaired_selected_pattern_ids"] == "unavailable"
    assert "accepted_repair_pattern_ids" not in row


def test_v2_specialization_retains_only_verified_structural_repair_fields():
    row = specialization_rows([{
        "artifact_schema_version": "specialization_trajectory_v2",
        "update_index": 1,
        "agent_id": 2,
        "selected_context_pattern_ids": ["pattern-a"],
        "selected_context_pattern_question_hashes": {"pattern-a": ["question-hash"]},
        "repaired_selected_pattern_ids": ["pattern-a"],
        "assigned_repaired_pattern_ids": ["pattern-a"],
    }])[0]
    assert row["repaired_selected_pattern_ids"] == ["pattern-a"]
    assert row["assigned_repaired_pattern_ids"] == ["pattern-a"]
