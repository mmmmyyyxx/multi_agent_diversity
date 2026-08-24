from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_v18_revision_parity_semantics import audit_trajectory


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def source(candidate_id: str, *, slot: int = 1) -> dict:
    return {
        "update_index": 0,
        "parent_team_hash": "a" * 64,
        "target_member": 1,
        "candidate_id": candidate_id,
        "candidate_stage": "source",
        "source_slot": slot,
        "valid": True,
    }


def event(source_id: str, *, valid: bool, revised_id: str = "") -> dict:
    row = {
        "update_index": 0,
        "parent_team_hash": "a" * 64,
        "target_agent_id": 1,
        "source_candidate_hash": source_id,
        "revision_attempted": True,
        "revision_output_valid": valid,
    }
    if valid:
        row["revised_candidate_hash"] = revised_id
    else:
        row["terminal_failure_class"] = "ValueError"
    return row


def revision(candidate_id: str, *, slot: int = 1) -> dict:
    return {
        "update_index": 0,
        "parent_team_hash": "a" * 64,
        "target_member": 1,
        "candidate_id": candidate_id,
        "candidate_stage": "revision",
        "source_slot": slot,
        "valid": True,
    }


def fixture_run(tmp_path: Path, candidates: list[dict], events: list[dict]) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    write_jsonl(run / "candidate_level_sanitized.jsonl", candidates)
    write_jsonl(run / "loss_blind_generic_revision_events.jsonl", events)
    write_jsonl(
        run / "candidate_decisions.jsonl",
        [{"selected_target_ids": [1]}],
    )
    return run


def test_invalid_revision_consumes_attempt_without_evaluable_row(tmp_path: Path):
    run = fixture_run(
        tmp_path,
        [source("s1"), source("s2", slot=2), revision("r1")],
        [event("s1", valid=True, revised_id="r1"), event("s2", valid=False)],
    )
    row, blockers = audit_trajectory(run, 59, "HYBRID_BASE")
    assert blockers == []
    assert row["valid_source_count"] == 2
    assert row["revision_attempt_count"] == 2
    assert row["revision_output_valid_count"] == 1
    assert row["revision_output_invalid_count"] == 1
    assert row["evaluable_revision_row_count"] == 1


def test_missing_revision_attempt_is_a_blocker(tmp_path: Path):
    run = fixture_run(tmp_path, [source("s1")], [])
    _, blockers = audit_trajectory(run, 59, "W1_TOP2")
    assert "attempts_equal_valid_sources:59:W1_TOP2" in blockers
    assert "source_attempt_join_is_one_to_one:59:W1_TOP2" in blockers


def test_valid_output_without_evaluable_row_is_a_blocker(tmp_path: Path):
    run = fixture_run(
        tmp_path,
        [source("s1")],
        [event("s1", valid=True, revised_id="r1")],
    )
    _, blockers = audit_trajectory(run, 60, "HYBRID_BASE")
    assert "valid_outputs_equal_evaluable_rows:60:HYBRID_BASE" in blockers
    assert "valid_output_row_join_is_one_to_one:60:HYBRID_BASE" in blockers


def test_duplicate_attempt_is_a_blocker(tmp_path: Path):
    run = fixture_run(
        tmp_path,
        [source("s1")],
        [event("s1", valid=False), event("s1", valid=False)],
    )
    _, blockers = audit_trajectory(run, 61, "W1_TOP2")
    assert "attempts_equal_valid_sources:61:W1_TOP2" in blockers
    assert "source_attempt_join_is_one_to_one:61:W1_TOP2" in blockers


def test_invalid_output_with_candidate_payload_is_a_blocker(tmp_path: Path):
    invalid = event("s1", valid=False)
    invalid["revised_candidate_hash"] = "r1"
    run = fixture_run(tmp_path, [source("s1")], [invalid])
    _, blockers = audit_trajectory(run, 59, "HYBRID_BASE")
    assert "invalid_outputs_have_no_evaluable_payload:59:HYBRID_BASE" in blockers


def test_unexpected_invalid_failure_class_is_a_blocker(tmp_path: Path):
    invalid = event("s1", valid=False)
    invalid["terminal_failure_class"] = "TransportError"
    run = fixture_run(tmp_path, [source("s1")], [invalid])
    _, blockers = audit_trajectory(run, 59, "HYBRID_BASE")
    assert "invalid_outputs_match_frozen_failure_semantics:59:HYBRID_BASE" in blockers
