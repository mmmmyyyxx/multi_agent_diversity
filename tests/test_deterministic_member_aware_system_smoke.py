import asyncio

from scripts.deterministic_member_aware_system_smoke import run_smoke


def test_real_system_smoke_covers_eight_updates_and_pareto_gates():
    report = asyncio.run(run_smoke())
    assert len(report["target_sequence"]) == 8
    assert report["team_transition_count"] == 8
    assert report["one_refresh_per_team_transition"] is True
    assert report["all_eligible_selected_within_8"] is True
    assert report["vote_positive_member_regressing_rejected"] is True
    assert report["vote_neutral_worst_member_positive_accepted"] is True
    assert report["single_agent_replacement_preserves_other_member_counts"] is True
    assert report["real_validation_key_is_feasible"] is True
    assert report["typical_role_call_count"] == 3
    assert report["max_selected_pattern_count"] <= 3
    assert report["max_selected_case_count"] <= 3
    assert report["student_raw_context_fields_seen"] == 0
    assert report["fault_smokes"] == {
        "critic_truncation": True,
        "critic_semantic_rejection": True,
        "student_partial_validity": True,
    }
