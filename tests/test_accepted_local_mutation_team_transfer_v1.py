from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash, validate_manifest
from multi_dataset_diverse_rl.team_search.accepted_mutation_replay import classify_transfer


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/accepted_local_mutation_team_transfer_v1_prep_20260919"
MANIFEST = ROOT / "experiments/manifests/accepted_local_mutation_team_transfer_v1.yaml"


def read(name):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_all_five_accepted_mutations_are_frozen_without_subset_selection():
    freeze = read("accepted_mutation_freeze.json")
    rows = freeze["mutations"]
    assert freeze["baseline_team_hash"] == "faa0fc81ebe71a686554355b9c5946a1765fa3913e026ab65e59f07ae01052dd"
    assert [(row["target_member"], row["source_proposal_index"]) for row in rows] == [
        (1, 2), (2, 8), (3, 5), (4, 2), (4, 4),
    ]
    assert len({row["accepted_candidate_hash"] for row in rows}) == 5
    assert all(row["local_delta"] == 1 for row in rows)
    assert all(row["newly_fixed"] == 1 and row["newly_broken"] == 0 for row in rows)
    assert all(row["team_minibatch_group_counts"] == {
        "responsibility": 4, "coalition": 4, "preservation": 4,
    } for row in rows)
    assert freeze["selection_rule"].startswith("all Phase-B")


def test_manifest_is_valid_unauthorized_and_read_only():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "infrastructure/experiment_manifest.schema.json").read_text())
    assert validate_manifest(manifest, schema) == []
    assert manifest["artifacts"]["preregistration"]["sha256"] == preregistration_hash(manifest)
    assert manifest["api_authorization"] == {
        "authorized": False,
        "authorization_scope": "pending explicit team-replay authorization",
        "allowed_roles": [],
        "allowed_phases": [],
    }
    assert manifest["budget"]["limit"]["successful_solver_provider_calls"] == 800
    assert manifest["budget"]["limit"]["Reflection"] == 0
    assert manifest["budget"]["limit"]["GEPA"] == 0
    assert manifest["budget"]["limit"]["writeback"] == 0


def test_preparation_is_zero_api_and_handoff_is_blocked():
    tree = ast.parse((ROOT / "scripts/prepare_accepted_local_mutation_team_transfer_v1.py").read_text())
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not {"AsyncOpenAI", "OpenAI", "run_parent", "optimize"}.intersection(called)
    handoff = read("EXPERIMENT_HANDOFF.json")
    assert handoff["READY_TO_RUN"] is False
    assert handoff["exact_runner_command"] is None
    assert "api_authorization_pending" in handoff["blockers"]
    assert read("api_ledger_summary.json")["provider_calls"] == 0


def test_runner_has_no_local_search_or_commit_path():
    source = (ROOT / "scripts/run_accepted_local_mutation_team_transfer_v1.py").read_text()
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported_from = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name and ("gepa" in name.lower() or "committer" in name.lower())
                   for name in imported | imported_from)
    assert "ReflectionLM" not in source
    assert "SystemTeamCommitter" not in source
    assert ".commit(" not in source


def test_report_is_sanitized_and_hash_replayable():
    assert scan_sanitized_artifacts(REPORT) == []
    assert read("sanitization_manifest.json")["status"] == "PASS"
    assert read("sha256_manifest.json") == build_sha256_manifest(REPORT)
    assert all(read("fact_assertions.json").values())


def test_classifier_is_frozen_for_all_branches():
    base = [{"execution_status": "COMPLETE", "full_status": "PASS",
             "team_vote_delta": 0, "team_minibatch_status": "PASS"} for _ in range(5)]
    multiple = [dict(row) for row in base]
    multiple[0]["team_vote_delta"] = multiple[1]["team_vote_delta"] = 1
    assert classify_transfer(multiple) == "LOCAL_IMPROVEMENT_CAN_TRANSFER_TO_TEAM_GAIN"
    single = [dict(row) for row in base]
    single[0]["team_vote_delta"] = 1
    assert classify_transfer(single) == "SINGLE_TEAM_GAIN_OBSERVED_REPLICATION_NEEDED"
    assert classify_transfer(base) == "NO_FULL_TEAM_GAIN_WITH_SOME_MINIBATCH_SIGNAL"
    none = [{**row, "team_minibatch_status": "FAIL", "full_status": "NOT_REACHED"} for row in base]
    assert classify_transfer(none) == "LOCAL_TEAM_OBJECTIVE_MISALIGNMENT_SIGNAL"
    assert classify_transfer(base[:4]) == "TEAM_TRANSFER_REPLAY_NOT_EVALUABLE"
