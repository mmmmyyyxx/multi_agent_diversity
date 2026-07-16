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


def test_v7_ablation_settings_keep_the_matched_search_budget():
    names = (
        "shared_vote_pareto_tcs_static,"
        "shared_vote_pareto_tcs_boundary_selector,"
        "shared_vote_error_pareto_tcs,"
        "shared_vote_error_pareto_tcs_residual_specialization,"
        "shared_vote_error_pareto_tcs_residual_cycle_guard"
    )
    settings = select_settings(names)
    matched_fields = (
        "init_mode", "baseline_only", "reward_mode", "best_state_selection_mode",
        "optimizer_architecture", "optimizer_fallback_mode", "teacher_critic_use_voting_failure",
        "candidate_eval_strategy", "candidate_eval_pool_size", "candidate_eval_batch_size",
        "candidate_eval_execution_mode", "solver_rollout_singleflight",
        "candidate_eval_prompt_dedup", "candidate_eval_cache_logging",
        "reward_schedule_mode",
    )
    reference = asdict(settings[0])
    for setting in settings[1:]:
        values = asdict(setting)
        assert {key: values[key] for key in matched_fields} == {
            key: reference[key] for key in matched_fields
        }


def test_static_vote_pareto_differs_from_historical_setting_only_by_schedule():
    historical, static = select_settings(
        "shared_vote_pareto_tcs,shared_vote_pareto_tcs_static"
    )
    historical_values = asdict(historical)
    static_values = asdict(static)
    historical_values.pop("name")
    static_values.pop("name")
    assert historical_values.pop("reward_schedule_mode") == ""
    assert static_values.pop("reward_schedule_mode") == "static"
    assert historical_values == static_values
