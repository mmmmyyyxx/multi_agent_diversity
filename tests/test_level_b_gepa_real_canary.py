from __future__ import annotations

import importlib.util
import json
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
        "manifest_preflight_pass": True,
        "manifest_schema": True,
        "attempt_identity": True,
        "launch_transaction": True,
        "proposer_diagnostics": True,
        "classifier_version": True,
        "preregistration_hash": True,
        "no_retry_resume": True,
    }


def test_run_lifecycle_is_atomic_and_failed_start_is_durable(tmp_path: Path) -> None:
    module = load()
    prep = tmp_path / "prep"
    prep.mkdir()
    (prep / "source_freeze.json").write_text(
        json.dumps(
            {
                "execution_commit": "a" * 40,
                "protocol_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    lifecycle = module.start_run_attempt(prep, run_root)
    assert run_root.is_dir()
    assert lifecycle["status"] == "RUNNING"
    assert lifecycle["provider_call_boundary_reached"] is False
    assert not list(tmp_path.glob("*.starting"))

    module.transition_run_attempt(run_root, provider_boundary_reached=True)
    terminal = module.transition_run_attempt(
        run_root,
        status="FAILED_START",
        provider_calls_observed=0,
        failure_category="SimulatedBoundaryAbort",
    )
    assert terminal["status"] == "FAILED_START"
    assert terminal["provider_call_boundary_reached"] is True
    assert terminal["provider_calls_observed"] == 0
    assert [row["status"] for row in terminal["events"]] == [
        "RUNNING",
        "FAILED_START",
    ]
    assert terminal["events"][-1]["failure_category"] == "SimulatedBoundaryAbort"


def test_canary_classifier_distinguishes_empirical_path_states() -> None:
    module = load()

    class Outcome:
        pass

    assert module.classify(
        {"proposer_diagnostics": {"proposal_attempts": 1, "solver_reached": 1}},
        Outcome(),
    ) == (
        "LOCAL_EMPIRICAL_PATH_CONFIRMED"
    )
    assert module.classify(
        {"proposer_diagnostics": {"proposal_attempts": 1, "solver_reached": 0}},
        Outcome(),
    ) == (
        "PROPOSAL_CONTRACT_STILL_BLOCKS_EMPIRICAL_SEARCH"
    )
    assert module.classify(
        {"proposer_diagnostics": {"proposal_attempts": 0, "solver_reached": 0}},
        Outcome(),
    ) == "NO_REAL_PROPOSAL_ATTEMPT"
