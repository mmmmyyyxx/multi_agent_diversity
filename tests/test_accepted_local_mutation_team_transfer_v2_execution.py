from __future__ import annotations

import json
from pathlib import Path

from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/accepted_local_mutation_team_transfer_v2_execution_20260919"


def read(name: str):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_primary_result_is_five_mandatory_full_equal_outcomes():
    summary = read("summary.json")
    assert summary["status"] == "VALID_SINGLE_BASELINE_STATE_REPLAY"
    assert summary["primary_estimand"] == {"team_positive_count": 0, "denominator": 5}
    assert summary["team_equal_count"] == 5
    assert summary["team_negative_count"] == 0
    rows = read("transfer_table.json")
    assert len(rows) == 5
    assert all(row["full_team_result"] == "TEAM_EQUAL" for row in rows)
    assert all(row["team_vote_delta"] == 0 for row in rows)


def test_target_and_oracle_gains_are_preserved_in_five_row_mapping():
    rows = read("transfer_table.json")
    assert [row["target_member_full_delta"] for row in rows] == [16, 20, 1, 11, 4]
    assert [row["oracle_coverage_delta"] for row in rows] == [28, 26, 17, 25, 20]
    assert all(row["local_delta"] == 1 for row in rows)
    assert all(row["newly_fixed"] == 1 and row["newly_broken"] == 0 for row in rows)


def test_minibatch_and_secondary_diagnostics_are_reported_separately():
    gate = read("team_minibatch_gate_audit.json")
    assert gate["correct_promotion"] == 0
    assert gate["false_negative_filtering"] == 0
    assert gate["false_positive_promotion"] == 5
    assert gate["correct_filtering"] == 0
    rows = read("transfer_table.json")
    assert [row["common_safe_status"] for row in rows].count("PASS") == 4
    assert [row["shadow_status"] for row in rows].count("PASS") == 4


def test_integrity_budget_isolation_and_baseline_lock_audits_pass():
    assert all(read("integrity_audit.json").values())
    ledger = read("api_ledger_summary.json")
    assert ledger["provider_successes"] == ledger["provider_attempts"] == 750
    assert ledger["provider_failures"] == 0
    assert ledger["cache_hits"] == 60
    access = read("evaluation_access_summary.json")
    assert access["GEPA"] == access["Reflection"] == 0
    assert access["Validation50"] == access["Test50"] == 0
    assert access["write_back"] == access["persistent_realizability_update"] == 0
    baseline = read("baseline_redundancy_audit.json")
    assert baseline["unique_baseline_prompt_hashes"] == 1
    assert baseline["unique_optimize_profile_hashes"] == 1
    assert baseline["single_member_plurality_change_structurally_possible_rows"] == 0


def test_execution_report_is_sanitized_and_hash_replayable():
    assert scan_sanitized_artifacts(REPORT) == []
    assert read("sanitization_manifest.json")["status"] == "PASS"
    assert read("sha256_manifest.json") == build_sha256_manifest(REPORT)
    assert all(read("fact_assertions.json").values())
