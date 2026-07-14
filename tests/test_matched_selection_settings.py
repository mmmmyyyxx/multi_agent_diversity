from dataclasses import asdict

from scripts.experiment_config import select_settings


def test_scalar_and_pareto_settings_are_matched_except_selection_mode():
    scalar, pareto = select_settings("shared_scalar_tcs_vote_first,shared_vote_pareto_tcs")
    scalar_values = asdict(scalar)
    pareto_values = asdict(pareto)
    scalar_values.pop("name")
    pareto_values.pop("name")
    assert scalar_values.pop("candidate_selection_mode") == "scalar_reward"
    assert pareto_values.pop("candidate_selection_mode") == "vote_pareto"
    assert scalar_values == pareto_values


def test_reward_ablation_settings_share_the_tcs_execution_contract():
    names = (
        "shared_scalar_tcs_vote_first,"
        "shared_accuracy_only_tcs_vote_first,"
        "shared_guarded_diversity_tcs_vote_first,"
        "shared_vote_no_margin_tcs_vote_first,"
        "shared_vote_no_boundary_tcs_vote_first"
    )
    full, accuracy, guarded, no_margin, no_boundary = select_settings(names)
    expected_rewards = {
        full.name: "vote_useful_diversity",
        accuracy.name: "accuracy_only",
        guarded.name: "guarded_diversity",
        no_margin.name: "vote_useful_diversity",
        no_boundary.name: "vote_useful_diversity",
    }
    reference = asdict(full)
    reference.pop("name")
    reference.pop("reward_mode")
    for setting in (accuracy, guarded, no_margin, no_boundary):
        values = asdict(setting)
        assert values.pop("name") in expected_rewards
        assert values.pop("reward_mode") == expected_rewards[setting.name]
        assert values == reference
