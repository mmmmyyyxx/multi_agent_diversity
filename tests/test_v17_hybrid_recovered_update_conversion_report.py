from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "v17_hybrid_recovered_update_conversion_audit_20260822"


def load_json(name: str) -> dict:
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_conversion_audit_published_facts() -> None:
    summary = load_json("summary.json")
    assert summary["status"] == "PASS"
    assert summary["api_calls"] == 0
    assert summary["new_test_calls"] == 0
    assert summary["conceptual_would_commit_count"] == 5
    assert summary["hybrid_conceptual_would_commit_count"] == 2
    assert summary["rr_conceptual_would_commit_count"] == 3
    assert summary["deduplicated_transition_count"] == 3
    assert summary["all_oracle_gains_are_G0_to_G1"] is True
    assert summary["all_oracle_gains_remain_nonwinning"] is True
    assert summary["simultaneous_vote_gain_and_loss"] is False
    metrics = summary["aggregate_unique_transition_metrics"]
    assert metrics["oracle_gain_count"] == 4
    assert metrics["oracle_loss_count"] == 0
    assert metrics["vote_gain_count"] == metrics["vote_loss_count"] == 0
    assert metrics["oracle_gain_coverage_role_after_counts"] == {"singleton_correct": 4}
    assert metrics["oracle_gain_minimum_dominant_flips_needed_counts"] == {"2": 4}
    assert summary["diagnosis"] == "SINGLETON_COVERAGE_RECOVERY_WITHOUT_VOTE_CONVERSION"


def test_conversion_rows_preserve_conceptual_and_unique_denominators() -> None:
    with (REPORT / "conceptual_update_level.csv").open(encoding="utf-8", newline="") as handle:
        conceptual = list(csv.DictReader(handle))
    with (REPORT / "unique_update_level.csv").open(encoding="utf-8", newline="") as handle:
        unique = list(csv.DictReader(handle))
    with (REPORT / "transition_level.csv").open(encoding="utf-8", newline="") as handle:
        transitions = list(csv.DictReader(handle))
    assert len(conceptual) == 5
    assert len(unique) == 3
    assert len(transitions) == 33
    assert len({row["transition_id"] for row in conceptual}) == 3
    oracle_gains = [row for row in transitions if row["oracle_gain"] == "1"]
    assert len(oracle_gains) == 4
    assert all(row["g_transition"] == "G0_to_G1" for row in oracle_gains)
    assert all(row["wrong_coalition_direction"] == "reduced" for row in oracle_gains)


def test_conversion_report_manifest_and_sanitization() -> None:
    manifest = load_json("sha256_manifest.json")
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((REPORT / name).read_bytes()).hexdigest() == expected
    patterns = (
        re.compile(r"(?i)[a-z]:[\\/]"),
        re.compile(r"(?i)\b(?:https?|file|sqlite)://?"),
        re.compile(r"(?i)\.sqlite(?:\b|-)"),
        re.compile(r"(?i)(?:api[_-]?key|authorization)\s*[:=]"),
        re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    )
    for path in REPORT.iterdir():
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert all(not pattern.search(text) for pattern in patterns), path.name
