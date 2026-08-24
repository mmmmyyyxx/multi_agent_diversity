from __future__ import annotations

import csv
import json
from pathlib import Path


REPORT = Path("reports/v18_hybrid_online_accumulation_pilot_20260822")


def load_json(name: str):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def load_csv(name: str):
    with (REPORT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_gate_provenance_retains_original_hold_and_admits_corrected_analysis():
    gate = load_json("gate_provenance.json")
    assert gate["original_frozen_gate_status"] == "FAIL/HOLD"
    assert gate["independent_semantics_audit_status"] == "PASS"
    assert gate["post_hoc_corrected_gate_status"] == "PASS"
    assert gate["corrected_gate_version"] == "post_hoc_corrected_gate_v1"
    assert gate["scientific_analysis_admitted"] is True
    assert gate["experiment_rerun"] is False
    assert gate["invalid_revisions_retried"] is False
    assert gate["new_api_calls"] == gate["new_model_calls"] == gate["new_test_calls"] == 0


def test_revision_attempt_and_evaluable_rows_remain_distinct():
    rows = load_csv("revision_attempts.csv")
    assert len(rows) == 54
    invalid = [row for row in rows if row["valid_output"] == "False"]
    assert len(invalid) == 4
    assert all(row["attempted"] == "True" for row in invalid)
    assert all(row["evaluable_row"] == "False" for row in invalid)
    assert all(row["opportunity_consumed"] == "True" for row in invalid)
    summary = load_json("summary.json")["revision_accounting"]
    assert summary["W1_TOP2"]["revision_attempt_count"] == 24
    assert summary["W1_TOP2"]["evaluable_revision_row_count"] == 24
    assert summary["HYBRID_BASE"]["revision_attempt_count"] == 30
    assert summary["HYBRID_BASE"]["evaluable_revision_row_count"] == 26


def test_primary_counts_trace_to_trajectory_table():
    rows = load_csv("trajectory_level.csv")
    assert len(rows) == 6
    summary = load_json("summary.json")
    for arm in ("W1_TOP2", "HYBRID_BASE"):
        selected = [row for row in rows if row["arm"] == arm]
        aggregate = summary["aggregate"][arm]
        assert aggregate["accepted_commits"] == sum(int(row["accepted_commit_count"]) for row in selected)
        assert aggregate["feasible_branches"] == sum(int(row["feasible_branch_count"]) for row in selected)
        assert aggregate["recoveries_0_to_1"] == sum(int(row["recovered_singleton_count"]) for row in selected)
        assert aggregate["deepenings_0_to_1_to_2_plus"] == sum(int(row["longitudinal_deepened_coverage_count"]) for row in selected)
        assert aggregate["persistent_singletons"] == sum(int(row["persistent_singleton_count"]) for row in selected)
        assert aggregate["cross_member_accumulations"] == sum(int(row["cross_member_support_accumulation_count"]) for row in selected)
        assert aggregate["recovered_coverage_vote_conversions"] == sum(int(row["recovered_coverage_to_vote_count"]) for row in selected)


def test_paired_table_and_frozen_classifier_are_deterministic():
    paired = load_csv("paired_comparison.csv")
    assert [int(row["deepening_hybrid_minus_w1"]) for row in paired] == [2, -1, 1]
    assert [int(row["vote_conversion_hybrid_minus_w1"]) for row in paired] == [2, 0, 0]
    assert [int(row["commit_hybrid_minus_w1"]) for row in paired] == [3, 0, 1]
    classifier = load_json("classifier.json")
    assert classifier["online_accumulation_supported"] is True
    assert classifier["online_vote_conversion_signal"] is True
    assert classifier["hybrid_throughput_recovery_reproduced"] is True
    assert classifier["persistent_singleton_reduced"] is True
    assert classifier["final_validation_vote_signal"] is False
    assert classifier["final_diagnosis"] == "LONGITUDINAL_ACCUMULATION_WITH_VOTE_CONVERSION"


def test_scientific_report_is_validation_only_and_sanitized():
    summary = load_json("summary.json")
    assert summary["new_test_calls"] == 0
    assert summary["scientific_scope"]["validation_only"] is True
    assert summary["scientific_scope"]["new_api_calls"] == 0
    assert summary["scientific_scope"]["new_test_calls"] == 0
    forbidden_headers = {
        "prompt", "question", "answer", "raw_response", "endpoint",
        "credential", "checkpoint",
    }
    for path in REPORT.glob("*.csv"):
        rows = load_csv(path.name)
        if rows:
            assert not forbidden_headers.intersection(rows[0])
    text = (REPORT / "README.md").read_text(encoding="utf-8")
    assert "Original frozen execution audit: FAIL / HOLD" in text
    assert "formal test or" in text
    assert "statistically significant" not in text
