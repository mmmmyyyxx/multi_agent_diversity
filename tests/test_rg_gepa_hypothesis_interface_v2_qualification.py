from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import run_rg_gepa_hypothesis_interface_v2_qualification as runner


def test_qualification_cases_are_synthetic_and_balanced() -> None:
    cases = runner.qualification_cases()
    assert len(cases) == 12
    assert len({case["qualification_id"] for case in cases}) == 12
    assert {case["lane"] for case in cases} == {"coverage", "margin_support", "direct_flip"}
    assert all(set(case) == {"qualification_id", "lane", "symbolic_evidence", "mutation_index"} for case in cases)


def test_request_exposes_only_fixed_ids_and_symbolic_context() -> None:
    system, user = runner.build_request(runner.qualification_cases()[0])
    assert "exact JSON array" in system
    assert '["F1","E1","P1","A1"]' in user
    assert "Question:" not in user
    assert "Gold:" not in user


def test_qualification_ledger_persists_attempts(tmp_path: Path) -> None:
    ledger = runner.QualificationLedger(tmp_path / "ledger.jsonl")
    fake = SimpleNamespace(calls=[{
        "client_role": "optimizer",
        "model": "qwen3.7-flash",
        "attempt": 1,
        "success": True,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }])
    runner._persist_calls(fake, 0, runner.qualification_cases()[0], ledger)
    row = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert row["logical_role"] == "schema_qualification"
    assert row["total_tokens"] == 15


def test_qualification_ledger_preserves_provider_retry(tmp_path: Path) -> None:
    ledger = runner.QualificationLedger(tmp_path / "ledger.jsonl")
    fake = SimpleNamespace(calls=[
        {
            "client_role": "optimizer", "model": "qwen3.7-flash", "attempt": 1,
            "success": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        },
        {
            "client_role": "optimizer", "model": "qwen3.7-flash", "attempt": 2,
            "success": True, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        },
    ])
    runner._persist_calls(fake, 0, runner.qualification_cases()[0], ledger)
    rows = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert [row["success"] for row in rows] == [False, True]
    assert [row["attempt_index"] for row in rows] == [1, 2]
