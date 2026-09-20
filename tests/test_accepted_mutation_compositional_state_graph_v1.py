from __future__ import annotations

import json
from pathlib import Path

import yaml

from multi_dataset_diverse_rl.governance.artifacts import (
    build_sha256_manifest,
    scan_sanitized_artifacts,
)
from multi_dataset_diverse_rl.governance.manifest import (
    preregistration_hash,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/accepted_mutation_compositional_state_graph_v1_20260920"
MANIFEST = ROOT / "experiments/manifests/accepted_mutation_compositional_state_graph_v1.yaml"


def read(name: str):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_all_24_states_are_enumerated_without_fabricating_exact_profiles():
    states = read("state_catalog_24.json")
    assert len(states) == 24
    assert len({row["state_id"] for row in states}) == 24
    assert all(row["exact_output_diversity"] == "NOT_IDENTIFIED" for row in states)
    assert all(row["exact_pivotal_rows"] == "NOT_IDENTIFIED" for row in states)
    evidence = read("evidence_availability_audit.json")
    assert evidence["candidate_full_output_profiles_persisted"] is False
    assert evidence["exact_24_state_reconstruction"] == "NOT_AVAILABLE"


def test_baseline_endpoint_is_structurally_locked():
    audit = read("baseline_endpoint_identifiability.json")
    assert audit["total_member_row_opportunities"] == 0
    assert audit["endpoint_structurally_identifiable"] is False
    assert [audit["by_member"][str(i)]["pivotal_capable_count"] for i in range(5)] == [0] * 5


def test_existential_common_safe_positive_path_proof_is_preserved():
    proof = read("existential_reachability_proof.json")
    assert proof["gain_memberships_across_40_baseline_wrong_rows"] == 91
    assert proof["sum_of_four_triple_gain_counts_lower"] == 11
    assert proof["sum_of_four_triple_vote_loss_eligible_counts_exact"] == 8
    assert proof["sum_of_four_triple_vote_loss_counts_upper"] == 8
    assert proof["sum_of_four_triple_vote_deltas_lower"] == 3
    assert proof["at_least_one_common_safe_path_to_team_positive_triple_exists"] is True
    assert proof["exact_positive_triple_identity"].startswith("NOT_IDENTIFIED")


def test_full_composition_bounds_prove_positive_vote_delta():
    rows = read("full_composition_bounds.json")
    assert len(rows) == 2
    by_m4 = {row["selected_mutations"][-1]: row for row in rows}
    assert by_m4["member4_proposal2_c537ca25a8aa"]["vote_delta_lower"] == 3
    assert by_m4["member4_proposal4_f5d945233dc9"]["vote_delta_lower"] == 1


def test_zero_api_manifest_and_report_are_replayable():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "infrastructure/experiment_manifest.schema.json").read_text())
    assert validate_manifest(manifest, schema) == []
    assert manifest["artifacts"]["preregistration"]["sha256"] == preregistration_hash(manifest)
    assert manifest["api_authorization"]["authorized"] is False
    assert manifest["budget"]["limit"] == {
        "provider_calls": 0,
        "Validation50": 0,
        "Test50": 0,
    }
    assert read("api_ledger_summary.json") == {
        "GEPA": 0,
        "Reflection": 0,
        "Test50": 0,
        "Validation50": 0,
        "provider_calls": 0,
        "solver_calls": 0,
    }
    assert scan_sanitized_artifacts(REPORT) == []
    assert read("sanitization_manifest.json")["status"] == "PASS"
    assert read("sha256_manifest.json") == build_sha256_manifest(REPORT)
    assert all(read("fact_assertions.json").values())
