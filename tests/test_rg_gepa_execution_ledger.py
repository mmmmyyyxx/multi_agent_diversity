"""Zero-API ledger coverage tests for the RG-GEPA fixed-parent executor."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_responsibility_guided_gepa_fixed_parent_pilot.py"


def _load():
    spec = importlib.util.spec_from_file_location("rg_gepa_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(module, *, parent_id: str, stage: str, ordinal: int, cache_hit: bool = False):
    row = {
        "seed": 76,
        "parent_id": parent_id,
        "update_index": 0,
        "candidate_id": "parent",
        "proposal_engine": "current",
        "evaluation_stage": stage,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "provider_attempt_id": f"{parent_id}:{stage}:{ordinal}",
        "cache_hit": cache_hit,
        "logical_role": "solver",
        "client_role": "solver",
        "success": True,
    }
    module.validate_ledger_record(row)
    return row


def _run(tmp_path: Path, module, *, include_full_team: bool) -> Path:
    cases = [f"case-{index}" for index in range(6)]
    (tmp_path / "result_sanitized.json").write_text(json.dumps({"results": [
        {"case_id": case, "actual_commits": 0, "test_calls": 0, "candidates": [{}, {}, {}, {}]}
        for case in cases
    ]}), encoding="utf-8")
    ledger = []
    for index, case in enumerate(cases):
        if include_full_team:
            ledger.append(_row(module, parent_id=case, stage="full_team", ordinal=index))
        ledger.extend(_row(module, parent_id=case, stage="minibatch_parent", ordinal=index * 12 + offset, cache_hit=True) for offset in range(12))
    with (tmp_path / "api_ledger_private.jsonl").open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row) + "\n")
    return tmp_path


def test_audit_requires_parent_full_team_ledger_coverage(tmp_path: Path):
    module = _load()
    with pytest.raises(RuntimeError, match="parent full-team ledger coverage"):
        module.audit(_run(tmp_path, module, include_full_team=False))


def test_audit_accepts_full_parent_and_paired_reuse_coverage(tmp_path: Path):
    module = _load()
    result = module.audit(_run(tmp_path, module, include_full_team=True))
    assert result["status"] == "PASS"
    assert result["minibatch_parent_cache_reuses"] == 72
