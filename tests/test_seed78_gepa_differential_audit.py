from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "seed78_gepa_differential_audit_20260914"


def load(name: str):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_differential_facts_and_classifier() -> None:
    summary = load("summary.json")
    diversity = summary["diversity"]
    capacity = summary["independent_capacity"]
    assert summary["gate"] == "PASS"
    assert (summary["api_calls"], summary["validation_calls"], summary["test_calls"]) == (0, 0, 0)
    assert diversity["branches"] == 24
    assert diversity["proposal_attempts"] == diversity["changed_proposals"] == 89
    assert diversity["accepted_mutations"] == diversity["positive_minibatch_delta"] == 0
    assert diversity["local_optimizer_solver_calls"] == 0
    assert diversity["primary_pre_solver_rejection"] == {
        "append_only_mutation": 4,
        "compact_prompt_limit": 3,
        "output_contract_contamination": 82,
    }
    assert capacity["proposal_attempts"] == 66
    assert capacity["accepted_mutations"] == 29
    assert capacity["accepted_mutations_failing_diversity_contract"] == 29
    assert summary["classifier"]["primary"] == "LOCAL_GEPA_PROPOSER_MUTABLE_CONTRACT_MISMATCH_CONFIRMED"
    assert summary["classifier"]["team_minibatch_status"] == "TEAMMINIBATCH_TRANSFER_NOT_EVALUATED"


def test_branch_table_and_comparison_matrix() -> None:
    with (REPORT / "diversity_branch_audit.csv").open(encoding="utf-8", newline="") as handle:
        branches = list(csv.DictReader(handle))
    assert len(branches) == 24
    assert {row["train_examples"] for row in branches} == {"100"}
    assert {row["local_validation_examples"] for row in branches} == {"12"}
    assert sum(int(row["proposal_attempts"]) for row in branches) == 89
    assert sum(int(row["retained_program_candidates"]) for row in branches) == 24
    with (REPORT / "comparison_matrix.csv").open(encoding="utf-8", newline="") as handle:
        matrix = {row["item"]: row for row in csv.DictReader(handle)}
    assert matrix["official_gepa"]["causal_status"] == "MATCH"
    assert matrix["pre_solver_mutable_contract"]["causal_status"] == "DECISIVE_DIFFERENCE"
    assert matrix["main_rejection_location"]["causal_status"] == "DECISIVE_DIFFERENCE"


def test_sanitization_and_hash_manifest() -> None:
    sanitization = load("sanitization_manifest.json")
    assert sanitization["gate"] == "PASS"
    assert sanitization["forbidden_findings"] == []
    manifest = load("sha256_manifest.json")
    for name, expected in manifest.items():
        assert hashlib.sha256((REPORT / name).read_bytes()).hexdigest() == expected
