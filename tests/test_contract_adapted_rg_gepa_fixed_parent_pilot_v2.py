from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.experimental_contract_adapted_rg_gepa import (
    ContractAdaptedProtocolV2,
    parse_hypothesis_selection_v2,
)
from scripts import run_contract_adapted_rg_gepa_fixed_parent_pilot_v2 as runner


def test_v2_scientific_protocol_binds_decoding_and_semantic_hashes() -> None:
    protocol = ContractAdaptedProtocolV2()
    payload = runner._protocol_payload()
    assert payload["optimizer_model"] == "qwen3.7-flash"
    assert payload["reflection_temperature"] == 0.0
    assert payload["reflection_max_tokens"] == 64
    assert payload["hypothesis_interface_hash"] == protocol.hypothesis_interface_hash
    assert payload["renderer_vocabulary_hash"] == protocol.renderer_vocabulary_hash


def test_v2_production_request_uses_shared_wire_builder() -> None:
    case = {
        "responsibility_type": "coverage",
        "target_member": 0,
        "parent_prompts": ["careful procedure"],
        "parent_profile": [{
            "question_hash": "q1", "team_answers": ["b"], "team_correctness": [False]
        }],
        "minibatch_private": [{
            "example_id": "q1", "question": "synthetic question", "answer": "a", "source_type": "coverage"
        }],
    }
    system, user = runner._request(case, 0)
    assert "exact JSON array" in system
    assert "Parent procedure" in user
    assert "target_prediction=b" in user
    assert parse_hypothesis_selection_v2('["F1","E1","P1","A1"]').failure_id == "F1"


def test_source_freeze_verifies_hash_equality(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.py"
    source.write_text("stable", encoding="utf-8")
    prep = tmp_path / "prep"
    prep.mkdir()
    (prep / "PRE_API_FREEZE.json").write_text(json.dumps({
        "source_files": [{
            "path": "source.py",
            "sha256": hashlib.sha256(b"stable").hexdigest(),
        }]
    }), encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner._verify_source_freeze(prep)
    source.write_text("drift", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source freeze mismatch"):
        runner._verify_source_freeze(prep)


def test_v2_ledger_adds_operator_provenance(tmp_path: Path) -> None:
    ledger = runner.V2ExecutionLedger(tmp_path / "ledger.jsonl")
    record = {
        "ledger_version": runner.base.RG_GEPA_LEDGER_VERSION,
        "seed": 76, "parent_id": "p", "update_index": 0, "candidate_id": "B_V2_0",
        "proposal_engine": "gepa_reflection", "evaluation_stage": "reflection",
        "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
        "provider_attempt_id": "attempt", "logical_call_id": "logical", "attempt_index": 1,
        "record_kind": "optimizer_provider_attempt", "provider_attempts": 1,
        "successful_provider_calls": 1, "cache_hit": False, "logical_role": "reflection",
        "client_role": "optimizer", "success": True,
    }
    ledger.append(record)
    saved = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert saved["proposal_operator_version"] == "contract_adapted_rg_gepa_v2"


def test_summary_separates_true_yield_from_progressive_recall() -> None:
    candidates = [
        {"candidate_id": "a", "schema_valid": True, "hard_gate_passed": True,
         "promoted": True, "feasible": True},
        {"candidate_id": "b", "schema_valid": True, "hard_gate_passed": True,
         "promoted": False, "feasible": True},
    ]
    result = {
        "candidates": candidates,
        "winners": {"B0_PRIME": "a", "B1_PRIME": "a"},
    }
    summary = runner.summarize([result], [])
    assert summary["true_feasible_yield"] == 2
    assert summary["promoted_feasible"] == 1
    assert summary["missed_feasible_by_progressive"] == 1
    assert summary["progressive_feasible_recall"] == 0.5
