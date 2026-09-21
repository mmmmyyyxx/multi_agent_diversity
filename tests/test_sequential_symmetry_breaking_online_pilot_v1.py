from __future__ import annotations

import json

import pytest

from multi_dataset_diverse_rl.provider_credentials import (
    DASHSCOPE_API_KEY_ENV,
    DASHSCOPE_BASE_URL_ENV,
)
from scripts import run_sequential_symmetry_breaking_online_pilot_v1 as pilot
from scripts.run_seed78_primary_responsibility_ab import Seed78System


def test_preflight_freezes_zero_api_scope() -> None:
    result = pilot.preflight()
    assert result["gate"] == "PASS"
    assert result["ready_to_run"] is False
    assert result["api_calls"] == 0
    assert result["validation50_calls"] == 0
    assert result["test50_calls"] == 0
    assert all(result["checks"].values())


def test_frozen_baseline_materialization_makes_no_provider_call(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(DASHSCOPE_API_KEY_ENV, "test-key")
    monkeypatch.setenv(DASHSCOPE_BASE_URL_ENV, "https://example.invalid/v1")
    _task, rows, parent_prompt = pilot._load_frozen_parent()
    optimize = tmp_path / "optimize.csv"
    shadow = tmp_path / "shadow.csv"
    optimize.write_text("question,answer\n", encoding="utf-8")
    shadow.write_text("question,answer\n", encoding="utf-8")
    ledger = pilot.BoundedLedger(tmp_path / "run/ledger.jsonl")
    system = Seed78System(
        pilot._config(tmp_path / "run", optimize_path=optimize, shadow_path=shadow),
        arm=pilot.ARM,
        ledger=ledger,
        raw_cache={},
    )

    snapshot = pilot._materialize_frozen_baseline(
        system,
        optimize_rows=rows,
        parent_prompt=parent_prompt,
    )

    assert snapshot["p_i"] == {str(member): 0 for member in range(5)}
    assert snapshot["total_member_row_opportunities"] == 0
    assert len(system.prompt_question_evaluator.cache) == 100
    assert not ledger.path.exists()
    assert not ledger.reservation_path.exists()


def test_bounded_ledger_fails_before_an_extra_transport_attempt(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(pilot, "TRANSPORT_ATTEMPT_CEILING", 2)
    ledger = pilot.BoundedLedger(tmp_path / "ledger.jsonl")
    ledger.reserve_provider_attempt("solver")
    ledger.reserve_provider_attempt("reflection")
    with pytest.raises(RuntimeError, match="transport_attempt_ceiling_exhausted"):
        ledger.reserve_provider_attempt("solver")
    rows = [json.loads(line) for line in ledger.reservation_path.read_text().splitlines()]
    assert rows == [
        {"attempt": 1, "role": "solver"},
        {"attempt": 2, "role": "reflection"},
    ]
