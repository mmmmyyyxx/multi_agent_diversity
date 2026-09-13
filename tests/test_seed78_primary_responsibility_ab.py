from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from multi_dataset_diverse_rl.governance.manifest import (
    preregistration_hash,
    validate_manifest,
)
from multi_dataset_diverse_rl.responsibility import MemberAwareRepairOpportunity
from multi_dataset_diverse_rl.team_search.system_runtime import FrozenResponsibilitySnapshot
from scripts.run_seed78_primary_responsibility_ab import (
    ARM_A,
    ARM_B,
    AUTH_ENV,
    _authorize,
    _arm_a_selection,
    _classify,
    _mechanism_metrics,
    preflight,
    protocol_document,
)


ROOT = Path(__file__).parents[1]


def _opportunity(member: int) -> MemberAwareRepairOpportunity:
    return MemberAwareRepairOpportunity(
        agent_id=member,
        question_hash=f"q-{member}",
        vote_flip_gain=1,
        margin_gain=0,
        member_error=True,
        coverage_opportunity=False,
        conversion_opportunity=False,
        dominant_wrong_member=False,
        unique_correct=False,
        pivotal_correct=False,
        oracle_soft_utility_gain=0.0,
    )


def test_protocol_freezes_seed75_split_and_delayed_validation() -> None:
    protocol = protocol_document()
    assert protocol["seed"] == 78
    assert protocol["split_identity"] == {
        "optimize100": "anti_overfitting_split_v1/fold_a+fold_b",
        "shadow50": "anti_overfitting_split_v1/fold_c",
        "validation50": "anti_overfitting_split_v1/validation",
        "test50": "blocked_zero_calls",
    }
    assert protocol["only_difference"] == "target_scheduler"
    assert protocol["shared"]["always_two_targets"] is True
    assert "after_both_training_trajectories_freeze" in protocol["data_roles"]["validation50"]
    assert preflight()["gate"] == "PASS"
    assert preflight()["api_calls"] == 0


def test_execution_entry_point_fails_closed_without_runtime_authorization(monkeypatch) -> None:
    monkeypatch.delenv(AUTH_ENV, raising=False)
    with pytest.raises(PermissionError, match=AUTH_ENV):
        _authorize()


def test_arm_a_zero_score_fill_is_two_distinct_legal_fallback_targets() -> None:
    selected, lanes, _, rows = _arm_a_selection(
        FrozenResponsibilitySnapshot({}, {}, {}),
        update_index=4,
        cursor={},
    )
    assert len(selected) == len(set(selected)) == 2
    assert set(lanes.values()) == {"fallback"}
    assert all(row["fallback_reason"] == "always_two_targets_zero_score_fill" for row in rows)


def test_arm_a_preserves_scored_target_and_fills_only_second_slot() -> None:
    opportunity = _opportunity(3)
    selected, lanes, _, _ = _arm_a_selection(
        FrozenResponsibilitySnapshot(
            {3: (opportunity,)},
            {},
            {opportunity.question_hash: -1},
        ),
        update_index=0,
        cursor={},
    )
    assert selected[0] == 3
    assert len(selected) == len(set(selected)) == 2
    assert lanes[3] == "direct_flip"
    assert lanes[selected[1]] == "fallback"


def test_mechanism_accounting_and_frozen_classifier() -> None:
    def row(targets, committed):
        return {"selected_target_ids": targets, "committed_member_id": committed}

    a_traj = {"events": [row([0, 1], 0), row([0, 1], 0)]}
    b_traj = {"events": [row([0, 1], 0), row([0, 1], 1)]}
    a = _mechanism_metrics(a_traj, {"total_tokens": 200})
    b = _mechanism_metrics(b_traj, {"total_tokens": 100})
    assert a["target_commit_distribution_mismatch"] == pytest.approx(0.5)
    assert b["target_commit_distribution_mismatch"] == pytest.approx(0.0)
    assert a["max_unresolved_target_streak_by_member"]["1"] == 2
    assert b["max_unresolved_target_streak_by_member"]["1"] == 1
    a["validation50"] = {"vote_accuracy": 0.60}
    b["validation50"] = {"vote_accuracy": 0.60}
    assert _classify({ARM_A: a, ARM_B: b}) == "PRIMARY_RESPONSIBILITY_SCHEDULER_SUPPORTED"


def test_manifest_schema_and_preregistration_hash() -> None:
    manifest_path = ROOT / "experiments/manifests/seed78_primary_responsibility_ab_v1.yaml"
    schema = json.loads((ROOT / "infrastructure/experiment_manifest.schema.json").read_text())
    manifest = yaml.safe_load(manifest_path.read_text())
    recorded = manifest["artifacts"]["preregistration"]["sha256"]
    assert recorded == preregistration_hash(manifest)
    assert validate_manifest(manifest, schema) == []
