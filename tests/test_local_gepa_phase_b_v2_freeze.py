from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash, validate_manifest
from multi_dataset_diverse_rl.parent_acquisition import select_parents

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/local_gepa_acceptance_rate_pilot_phase_b_v2_prep_20260918"
MANIFEST = ROOT / "experiments/manifests/local_gepa_acceptance_rate_pilot_phase_b_v2.yaml"


def read(name):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_exact_single_state_parent_freeze_and_fifth_parent_audit():
    freeze = read("PARENT_FREEZE.json")
    selected = freeze["selected_parents"]
    assert [row["target_member"] for row in selected] == [1, 2, 3, 4]
    assert len({row["source_state_hash"] for row in selected}) == 1
    assert all(row["primary_responsibility_lane"] == "coverage" for row in selected)
    assert freeze["unselected_eligible_parent"]["target_member"] == 0
    assert freeze["unselected_eligible_parent"]["exclusion_reason"].startswith("stable_subset_sha256")
    catalog = json.loads((ROOT / "reports/local_gepa_parent_acquisition_v1/eligible_parent_catalog.json").read_text())
    assert select_parents(catalog) == [row["parent_task_id"] for row in selected]


def test_budget_scope_and_authorization_are_frozen():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "infrastructure/experiment_manifest.schema.json").read_text())
    assert validate_manifest(manifest, schema) == []
    assert manifest["artifacts"]["preregistration"]["sha256"] == preregistration_hash(manifest)
    assert manifest["api_authorization"]["authorized"] is True
    assert manifest["api_authorization"]["allowed_roles"] == ["solver", "reflection"]
    assert manifest["api_authorization"]["allowed_phases"] == ["phase_b_local_optimizer"]
    assert manifest["budget"]["limit"]["total_proposals"] == 32
    assert manifest["budget"]["limit"]["total_metric_rows"] == 820
    assert manifest["data"]["validation_policy"].endswith("calls=0")
    assert manifest["data"]["test_policy"] == "Test50 calls=0"
    assert not (ROOT / manifest["design"]["formal_run_root"]).exists()
    handoff = json.loads((ROOT / "experiments/local_gepa_acceptance_rate_pilot_phase_b_v2/EXPERIMENT_HANDOFF.json").read_text())
    assert handoff["READY_TO_RUN"] is False
    assert handoff["execution"]["exact_runner_command"] is None


def test_non_iid_reporting_and_equal_negative_decomposition_are_explicit():
    telemetry = read("telemetry_contract.json")
    assert telemetry["proposal_outcomes"] == ["STRICT_POSITIVE", "EXACT_EQUAL", "STRICT_NEGATIVE"]
    assert telemetry["iid_inference_permitted"] is False
    assert telemetry["optional_interval_label"] == "NAIVE_IID_REFERENCE_ONLY"
    assert telemetry["zero_delta_decomposition"] == ["BEHAVIORAL_NO_OP", "REPAIR_PRESERVATION_CANCELLATION"]
    template = read("final_report_template.json")
    assert template["cross_state_generalization"] == "PROHIBITED"
    assert template["team_transfer"] == "TEAM_TRANSFER_NOT_EVALUATED"
    assert len(template["parent_summaries_required"]) == 4


def test_report_hashes_sanitization_and_zero_api_evidence():
    assert scan_sanitized_artifacts(REPORT) == []
    assert read("sha256_manifest.json") == build_sha256_manifest(REPORT)
    facts = read("fact_assertions.json")
    assert all(facts.values())
    assert read("api_ledger_summary.json")["phase_b_provider_calls"] == 0


def test_preparation_script_has_no_provider_or_optimizer_execution():
    source = (ROOT / "scripts/prepare_local_gepa_acceptance_rate_phase_b_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not {"AsyncOpenAI", "OpenAI", "run_parent", "optimize"}.intersection(names)


def test_exact_private_parent_identity_replay():
    from scripts.preflight_local_gepa_acceptance_rate_phase_b_v2 import run
    result = run()
    assert result["status"] == "PASS"
    assert result["exact_parent_replay"] is True
    assert result["selected_parent_count"] == 4
    assert result["shared_state_count"] == 1
    assert result["execution_gate"] == "EXECUTABLE_HANDOFF_REQUIRED"


def test_execution_runner_fails_before_freeze_or_provider_without_runtime_authorization(tmp_path, monkeypatch):
    import asyncio
    from scripts.run_local_gepa_acceptance_rate_phase_b_v2 import execute
    monkeypatch.delenv("LOCAL_GEPA_PHASE_B_V2_AUTHORIZED", raising=False)
    try:
        asyncio.run(execute(tmp_path / "missing", tmp_path / "formal"))
    except PermissionError as exc:
        assert "LOCAL_GEPA_PHASE_B_V2_AUTHORIZED" in str(exc)
    else:
        raise AssertionError("execution must fail before freeze/provider construction")
    assert not (tmp_path / "formal").exists()
