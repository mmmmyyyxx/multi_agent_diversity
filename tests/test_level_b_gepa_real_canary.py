from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / "scripts/run_level_b_gepa_real_canary.py"
    spec = importlib.util.spec_from_file_location("level_b_gepa_real_canary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canary_preflight_is_zero_api_and_narrow() -> None:
    module = load()
    result = module.preflight()
    assert result["gate"] == "PASS"
    assert result["api_calls"] == 0
    assert result["validation_calls"] == 0
    assert result["test_calls"] == 0
    assert result["checks"] == {
        "one_opportunity": True,
        "one_target": True,
        "unchanged_gepa_budget": True,
        "decision_procedure_component": True,
        "level_b": True,
        "validation_zero": True,
        "test_zero": True,
        "no_retry_resume": True,
    }


def test_canary_classifier_distinguishes_empirical_path_states() -> None:
    module = load()

    class Cost:
        local_optimizer_solver_calls = 18
        team_minibatch_solver_calls = 12

    class Outcome:
        cost = Cost()
        funnel = {"local_candidates": 1}

    assert module.classify({"proposal_attempts": 1}, Outcome()) == (
        "BACKEND_EMPIRICAL_PATH_CONFIRMED"
    )
    Outcome.cost.local_optimizer_solver_calls = 12
    assert module.classify({"proposal_attempts": 1}, Outcome()) == (
        "PROPOSAL_CONTRACT_STILL_BLOCKS_EMPIRICAL_SEARCH"
    )
