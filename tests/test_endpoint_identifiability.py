from multi_dataset_diverse_rl.evaluation.endpoint_identifiability import (
    endpoint_structural_identifiability,
    target_row_is_plurality_pivotal_capable,
)


DOMAIN = ("A", "B", "C", None)


def test_four_identical_peers_lock_single_member_plurality():
    assert not target_row_is_plurality_pivotal_capable(
        ["A", "A", "A", "A"], legal_target_outputs=DOMAIN,
    )


def test_two_two_and_two_one_one_peer_states_are_pivotal_capable():
    assert target_row_is_plurality_pivotal_capable(
        ["A", "A", "B", "B"], legal_target_outputs=DOMAIN,
    )
    assert target_row_is_plurality_pivotal_capable(
        ["A", "A", "B", "C"], legal_target_outputs=DOMAIN,
    )


def test_invalid_votes_follow_frozen_abstention_semantics():
    assert target_row_is_plurality_pivotal_capable(
        ["A", "B", None, None], legal_target_outputs=DOMAIN,
    )


def test_profile_audit_reports_member_opportunities():
    profiles = [
        ["A", "A"],
        ["A", "A"],
        ["A", "B"],
        ["A", "B"],
        ["A", "C"],
    ]
    result = endpoint_structural_identifiability(
        profiles, legal_target_outputs_by_row=[DOMAIN, DOMAIN],
    )
    assert result["by_member"]["0"]["pivotal_capable_count"] == 1
    assert result["total_member_row_opportunities"] > 0
    assert result["endpoint_structurally_identifiable"] is True
