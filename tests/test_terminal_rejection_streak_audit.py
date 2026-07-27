from scripts.audit_terminal_rejection_streak import state_by_update, strictly_dominates


def test_objective_dominance_requires_one_strict_dimension() -> None:
    assert strictly_dominates([50, 12, 102], [49, 12, 98])
    assert not strictly_dominates([49, 12, 102], [49, 12, 102])
    assert not strictly_dominates([50, 11, 102], [49, 12, 98])


def test_state_index_advances_only_after_accepted_updates() -> None:
    decisions = [
        {"update_index": 0, "accepted_prompt_hash": ""},
        {"update_index": 1, "accepted_prompt_hash": "a"},
        {"update_index": 2, "accepted_prompt_hash": ""},
    ]
    assert state_by_update(decisions) == {0: 0, 1: 0, 2: 1}
