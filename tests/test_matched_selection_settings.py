from dataclasses import asdict

from scripts.experiment_config import select_settings


def test_scalar_and_pareto_settings_are_matched_except_selection_mode():
    scalar, pareto = select_settings("shared_scalar_tcs_oracle_first,shared_oracle_pareto_tcs")
    scalar_values = asdict(scalar)
    pareto_values = asdict(pareto)
    scalar_values.pop("name")
    pareto_values.pop("name")
    assert scalar_values.pop("candidate_selection_mode") == "scalar_reward"
    assert pareto_values.pop("candidate_selection_mode") == "oracle_pareto"
    assert scalar_values == pareto_values
