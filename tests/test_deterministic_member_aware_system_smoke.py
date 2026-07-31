import asyncio

from scripts.deterministic_member_aware_system_smoke import run_smoke


def test_real_system_smoke_enforces_repair_only_targeting_and_pareto_gates():
    report = asyncio.run(run_smoke())
    assert len(report["target_sequence"]) >= 1
    assert (
        report["all_selected_targets_have_responsibility_portfolios"]
        is True
    )
    assert report["default_catchup_disabled"] is True
    assert report["default_proposal_memory_disabled"] is True
    assert report["no_actionable_responsibility_is_noop"] is True
    assert report["target_neutral_vote_positive_accepted"] is True
    assert report["vote_positive_member_regressing_rejected"] is True
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
