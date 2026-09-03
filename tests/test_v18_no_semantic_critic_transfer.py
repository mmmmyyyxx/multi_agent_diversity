from scripts.analyze_v18_no_semantic_critic_transfer import (
    ARMS,
    SEEDS,
    _validate_inputs,
    classify_gain_persistence,
    classify_loss_origin,
)


def test_scope_is_frozen_to_three_seed_a_c_decomposition():
    assert SEEDS == (68, 69, 70)
    assert ARMS == ("A_CANONICAL", "C_NO_SEMANTIC_CRITIC")


def test_gain_persistence_classes_cover_overwrite_and_recovery():
    assert classify_gain_persistence([True, True, True])["persistence_class"] == "retained_to_final"
    assert classify_gain_persistence([True, False, False])["persistence_class"] == "overwritten_not_recovered"
    assert classify_gain_persistence([True, False, True])["persistence_class"] == "overwritten_then_recovered_to_final"


def test_loss_origin_separates_initial_capability_from_prior_conversion():
    assert classify_loss_origin([True, True]) == "new_collateral_regression"
    assert classify_loss_origin([False, True]) == "prior_conversion_overwritten"


def test_current_summary_schema_uses_test_evaluation_count():
    states = [{"state_index": 0, "examples": [{"example_id_hash": "x"}]}]
    summary = {"test_evaluation_count": 0, "accepted_commit_count": 0}
    _validate_inputs(states, [], summary)
