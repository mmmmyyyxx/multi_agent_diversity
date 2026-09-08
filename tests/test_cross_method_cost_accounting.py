from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_cross_method_cost_accounting.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cost_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_accounting_constants_and_arithmetic():
    module = load_module()
    assert sum(module.EXPECTED_OPPORTUNITIES.values()) == 53
    assert sum(module.REPORTED_ACCEPTED_STATES.values()) == 22
    assert sum(module.ACTUAL_COMMITS.values()) == 19
    assert all(v["input_tokens"] + v["output_tokens"] == v["total_tokens"] for v in module.GEPA_REPLAY.values())
    assert all(v["input_tokens"] + v["output_tokens"] == v["total_tokens"] for v in module.COMMON_REPLAY.values())


def test_generated_report_is_sanitized_and_reconciled():
    report = ROOT / "reports/cross_method_cost_accounting_20260907"
    if not report.exists():
        load_module().run(report)
    assert json.loads((report / "sanitization_manifest.json").read_text(encoding="utf-8"))["status"] == "PASS"
    reconciliation = json.loads((report / "reconciliation.json").read_text(encoding="utf-8"))
    assert reconciliation["commit_count_correction"] == {
        "actual_commits": 19, "initial_states_included": 3, "published_state_counts": 22
    }
    rows = list(csv.DictReader((report / "diversity_p1_usage_by_seed.csv").open(encoding="utf-8")))
    assert [int(row["actual_commits"]) for row in rows] == [6, 8, 5]
    assert rows[0]["total_tokens"] == ""
    assert int(rows[1]["total_tokens"]) + int(rows[2]["total_tokens"]) == 10_666_948
