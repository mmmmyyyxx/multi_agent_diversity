from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_v16_module2_pilot.py"
SPEC = importlib.util.spec_from_file_location("audit_v16_module2_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def decision(update=0, parent="parent", targets=(1, 3)):
    return {
        "update_index": update,
        "parent_team_hash": parent,
        "selected_target_ids": list(targets),
    }


def context(update=0, target=1, **extra):
    return {"update_index": update, "target_agent_id": target, **extra}


def test_missing_context_parent_is_reconciled_from_candidate_decision():
    row = context()
    indexed = AUDIT.index_decisions_by_update([decision()])
    resolved, source = AUDIT.reconcile_context_parent_hash(row, indexed)
    assert resolved == "parent"
    assert source == "candidate_decision_by_update"
    assert "parent_team_hash" not in row


def test_present_context_parent_is_verified_without_rewriting():
    row = context(parent_team_hash="parent")
    resolved, source = AUDIT.reconcile_context_parent_hash(
        row, AUDIT.index_decisions_by_update([decision()])
    )
    assert resolved == "parent"
    assert source == "context_row_verified"
    assert row["parent_team_hash"] == "parent"


def test_conflicting_context_parent_is_rejected():
    with pytest.raises(ValueError, match="conflicts"):
        AUDIT.reconcile_context_parent_hash(
            context(parent_team_hash="other"),
            AUDIT.index_decisions_by_update([decision()]),
        )


def test_missing_update_provenance_is_rejected():
    with pytest.raises(ValueError, match="no candidate decision provenance"):
        AUDIT.reconcile_context_parent_hash(context(update=2), {})


def test_unselected_context_target_is_rejected():
    with pytest.raises(ValueError, match="is not selected"):
        AUDIT.reconcile_context_parent_hash(
            context(target=4), AUDIT.index_decisions_by_update([decision()])
        )


def test_duplicate_decision_update_is_rejected():
    with pytest.raises(ValueError, match="duplicate candidate decision"):
        AUDIT.index_decisions_by_update([decision(), decision()])


def test_decision_without_parent_hash_is_rejected():
    row = decision()
    row["parent_team_hash"] = ""
    with pytest.raises(ValueError, match="lacks parent_team_hash"):
        AUDIT.index_decisions_by_update([row])
