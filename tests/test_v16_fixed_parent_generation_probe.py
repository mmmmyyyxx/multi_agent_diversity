from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_registry_is_fixed_parent_balanced_and_outcome_blind():
    module = load("probe_registry", ROOT / "scripts" / "build_v16_fixed_parent_probe_registry.py")
    payload = module.build_registry(module.DEFAULT_RUN)
    assert len(payload["cases"]) == 3
    assert payload["case_selection_uses_candidate_outcomes"] is False
    assert len({row["parent_team_hash"] for row in payload["cases"]}) == 1
    assert sorted(item for row in payload["cases"] for item in row["cell_order"]) == sorted(payload["variants"] * 3)
    assert all(row["assigned_question_hashes"] for row in payload["cases"])
    assert payload["commit_enabled"] is False
    assert payload["validation_enabled"] is False
    assert payload["final_test_enabled"] is False


def test_runner_requires_explicit_api_authorization(tmp_path, monkeypatch):
    module = load("probe_runner", ROOT / "scripts" / "run_v16_fixed_parent_generation_probe.py")
    registry = tmp_path / "registry.json"
    registry.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("V16_FIXED_PARENT_PROBE_AUTHORIZED", raising=False)
    args = type("Args", (), {"registry": registry, "out_root": tmp_path / "out"})()
    with pytest.raises(SystemExit, match="API execution blocked"):
        import asyncio
        asyncio.run(module.main_async(args))


def test_cell_path_must_be_fresh(tmp_path):
    module = load("probe_runner_cell_path", ROOT / "scripts" / "run_v16_fixed_parent_generation_probe.py")
    cell = tmp_path / "cell"
    module.require_fresh_cell_path(cell)
    cell.mkdir()
    with pytest.raises(FileExistsError, match="must be fresh"):
        module.require_fresh_cell_path(cell)


def test_offline_preflight_reconstructs_all_cells_without_api():
    registry_module = load("probe_registry_preflight", ROOT / "scripts" / "build_v16_fixed_parent_probe_registry.py")
    preflight_module = load("probe_preflight", ROOT / "scripts" / "preflight_v16_fixed_parent_generation_probe.py")
    result = preflight_module.preflight(registry_module.build_registry(registry_module.DEFAULT_RUN))
    assert result["status"] == "PASS"
    assert result["api_calls"] == 0
    assert result["cell_count"] == 9


def test_auditor_rejects_parent_mutation(tmp_path, monkeypatch):
    registry = {"registry_content_hash": "h", "variants": ["c0", "c2", "c3"], "cases": [{"case_id": "x"}]}
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    (run / "probe_summary.json").write_text(json.dumps({
        "registry_hash": "h", "validation_calls": 0, "test_calls": 0,
        "cells": [{"case_id": "x", "variant": variant, "commit_performed": variant == "c2",
                   "parent_state_hash_before": "a", "parent_state_hash_after": "a",
                   "validation_calls": 0, "test_calls": 0, "generated_candidate_count": 2}
                  for variant in ("c0", "c2", "c3")],
    }), encoding="utf-8")
    module = load("probe_audit", ROOT / "scripts" / "audit_v16_fixed_parent_generation_probe.py")
    monkeypatch.setattr("sys.argv", ["audit", "--registry", str(tmp_path / "registry.json"), "--run_root", str(run), "--out", str(tmp_path / "audit.json")])
    with pytest.raises(SystemExit):
        module.main()
    assert json.loads((tmp_path / "audit.json").read_text())["gate"] == "FAIL"
