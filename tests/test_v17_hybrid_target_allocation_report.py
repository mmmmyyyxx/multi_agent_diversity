from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "v17_hybrid_target_allocation_pilot_20260821"


def _json(name: str) -> dict:
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_hybrid_report_facts_match_frozen_classifier() -> None:
    summary = _json("summary.json")
    classifier = _json("classifier.json")
    assert summary["phase_a_gate"] == "PASS"
    assert summary["phase_b_gate"] == "PASS"
    assert summary["phase_a_zero_api"] is True
    assert summary["prospective_parent_count"] == 6
    assert summary["cell_count"] == 18
    assert summary["conceptual_branch_count"] == 36
    assert summary["deduplicated_branch_count"] == 20
    assert summary["new_test_calls"] == 0
    assert summary["actual_prompt_commits"] == 0
    assert summary["trajectory_mutations"] == 0
    assert summary["funnel"]["W1_TOP2"]["feasible_branch_count"] == 0
    assert summary["funnel"]["RR_TOP2"]["feasible_branch_count"] == 3
    assert summary["funnel"]["HYBRID_EXPLOIT_EXPLORE"]["feasible_branch_count"] == 2
    assert classifier["hybrid_vote_benefit_supported"] is False
    assert classifier["hybrid_feasibility_recovery_supported"] is True
    assert classifier["hybrid_oracle_benefit_supported"] is True
    assert classifier["final_pilot_diagnosis"] == "HYBRID_THROUGHPUT_ONLY"


def test_hybrid_report_candidate_schema_and_parent_overlap() -> None:
    with (REPORT / "candidate_level.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 36
    forbidden = {"prompt", "question", "answer", "raw_response", "endpoint"}
    assert forbidden.isdisjoint(rows[0])
    assert all(row["candidate_stage"] in {"source", "revision"} for row in rows)
    with (REPORT / "parent_level.csv").open(encoding="utf-8", newline="") as fh:
        parents = list(csv.DictReader(fh))
    assert len(parents) == 6
    assert all(row["hybrid_explore_equals_w1_rank2"] == "0" for row in parents)


def test_hybrid_report_manifest_is_exact() -> None:
    manifest = _json("sha256_manifest.json")
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((REPORT / name).read_bytes()).hexdigest()
        assert actual == expected


def test_hybrid_report_is_sanitized() -> None:
    forbidden_patterns = {
        "absolute_windows_path": re.compile(r"(?i)[a-z]:[\\/]"),
        "network_or_file_uri": re.compile(r"(?i)\b(?:https?|file|sqlite)://?"),
        "sqlite_filename": re.compile(r"(?i)\.sqlite(?:\b|-)"),
        "credential_assignment": re.compile(r"(?i)(?:api[_-]?key|authorization)\s*[:=]"),
        "raw_secret": re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    }
    for path in REPORT.iterdir():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in forbidden_patterns.items():
            assert not pattern.search(text), f"{label}: {path.name}"
