from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.cli import (
    _verify_frozen_initialization,
    pending_epoch_indices,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.termination import (
    COMPLETED_BY_BUDGET,
    COMPLETED_BY_EARLY_STOP,
    INCOMPLETE,
    USER_ABORTED_INCOMPLETE,
    assess_trajectory_termination,
)


def _records(count: int, *, final_no_commit_streak: int = 0):
    approvals = [True] * (count - final_no_commit_streak) + [
        False
    ] * final_no_commit_streak
    return [
        {"update_index": index, "writeback_approved": approved}
        for index, approved in enumerate(approvals)
    ]


def test_full_budget_is_completed_by_budget() -> None:
    result = assess_trajectory_termination(
        planned_update_opportunities=32,
        executed_update_records=_records(32, final_no_commit_streak=2),
        stored_early_stop_reason="",
        completed_update_count=32,
    )

    assert result.status == COMPLETED_BY_BUDGET
    assert result.training_completed is True
    assert result.remaining_unexecuted == 0
    assert result.terminal_update_index == 31


def test_seed75_p0_structural_fixture_is_completed_by_early_stop() -> None:
    result = assess_trajectory_termination(
        planned_update_opportunities=32,
        executed_update_records=_records(21, final_no_commit_streak=6),
        stored_early_stop_reason="no_shadow_approved_commit_streak_6",
        completed_update_count=21,
    )

    assert result.status == COMPLETED_BY_EARLY_STOP
    assert result.training_completed is True
    assert result.executed_update_opportunities == 21
    assert result.remaining_unexecuted == 11
    assert result.terminal_update_index == 20
    assert result.terminal_update_ordinal == 21
    assert result.final_no_commit_streak == 6


def test_five_failures_do_not_satisfy_six_failure_stop() -> None:
    result = assess_trajectory_termination(
        planned_update_opportunities=32,
        executed_update_records=_records(21, final_no_commit_streak=5),
        stored_early_stop_reason="",
        completed_update_count=21,
    )

    assert result.status == INCOMPLETE
    assert result.training_completed is False


def test_early_stop_label_without_supporting_history_fails_closed() -> None:
    result = assess_trajectory_termination(
        planned_update_opportunities=32,
        executed_update_records=_records(21, final_no_commit_streak=5),
        stored_early_stop_reason="no_shadow_approved_commit_streak_6",
        completed_update_count=21,
    )

    assert result.status == INCOMPLETE
    assert "early_stop_streak_mismatch" in result.errors


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        (_records(32), ""),
        (_records(21, final_no_commit_streak=6), "no_shadow_approved_commit_streak_6"),
    ],
)
def test_resume_schedules_zero_epochs_for_terminal_trajectory(records, reason) -> None:
    result = assess_trajectory_termination(
        planned_update_opportunities=32,
        executed_update_records=records,
        stored_early_stop_reason=reason,
        completed_update_count=len(records),
    )

    assert result.training_completed
    assert list(
        pending_epoch_indices(2, 4, training_completed=result.training_completed)
    ) == []


def test_user_abort_without_valid_early_stop_is_not_completion() -> None:
    result = assess_trajectory_termination(
        planned_update_opportunities=32,
        executed_update_records=_records(21, final_no_commit_streak=5),
        stored_early_stop_reason="",
        completed_update_count=21,
        user_aborted=True,
    )

    assert result.status == USER_ABORTED_INCOMPLETE
    assert result.training_completed is False


@dataclass
class _Protocol:
    legacy_protocol: bool = False


class _SystemStub:
    def __init__(self, records, reason: str):
        self.candidate_decisions = records
        self.completed_update_count = len(records)
        self.early_stop_reason = reason
        self.protocol = _Protocol()
        self.planned_update_count = 0
        self.training_completed = False
        self.termination_status = INCOMPLETE


def test_system_completion_contract_accepts_valid_early_stop() -> None:
    system = _SystemStub(
        _records(21, final_no_commit_streak=6),
        "no_shadow_approved_commit_streak_6",
    )

    PromptEnsembleOptimizationSystem.mark_training_complete(system, 32)

    assert system.training_completed is True
    assert system.termination_status == COMPLETED_BY_EARLY_STOP


def test_system_completion_contract_rejects_unproven_early_stop() -> None:
    system = _SystemStub(
        _records(21, final_no_commit_streak=5),
        "no_shadow_approved_commit_streak_6",
    )

    with pytest.raises(RuntimeError, match="before every planned update"):
        PromptEnsembleOptimizationSystem.mark_training_complete(system, 32)


class _Artifacts:
    def __init__(self):
        self.values = {}

    def write_json(self, name, value):
        self.values[name] = value


class _InitializationSystem:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.artifacts = _Artifacts()

    def frozen_initialization_snapshot(self):
        return self.snapshot


def _identity(commit: str, manifest: str):
    return {
        "git_commit": commit,
        "git_dirty": False,
        "manifest_sha256": manifest,
        "train_file_sha256": "train-file",
        "val_file_sha256": "val-file",
        "test_file_sha256": "test-file",
        "train_question_set_hash": "train-questions",
        "val_question_set_hash": "val-questions",
        "test_question_set_hash": "test-questions",
    }


def test_frozen_initialization_allows_only_explicit_provenance_transition(
    tmp_path,
) -> None:
    scientific_fields = [
        "initial_prompt_hashes",
        "initial_member_correct_counts",
        "initial_team_outcome",
        "initial_vote_oracle_ghm_hash",
        "probe_hash",
        "solver_request_identity",
        "solver_identity",
    ]
    source_identity = _identity("source-commit", "source-manifest")
    target_identity = _identity("target-commit", "target-manifest")
    expected = {
        "initial_prompt_hashes": ["prompt"] * 5,
        "initial_member_correct_counts": [59] * 5,
        "initial_team_outcome": {"team_vote_correct_count": 59},
        "initial_vote_oracle_ghm_hash": "ghm",
        "initial_train_state_hash": "source-state-hash",
        "probe_hash": "probe",
        "solver_request_identity": "request",
        "solver_identity": ["solver"],
        "immutable_run_identity": source_identity,
    }
    actual = {
        **expected,
        "initial_train_state_hash": "target-state-hash",
        "immutable_run_identity": target_identity,
    }
    manifest = {
        "manifest_version": "fixture",
        "initialization_snapshot": expected,
        "execution_identity_transition": {
            "schema_version": "frozen_initialization_execution_identity_transition_v1",
            "source_immutable_run_identity": source_identity,
            "target_immutable_run_identity": target_identity,
            "scientific_state_fields": scientific_fields,
        },
    }
    path = tmp_path / "initialization.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    system = _InitializationSystem(actual)
    cfg = SimpleNamespace(
        persistence=SimpleNamespace(frozen_initialization_manifest_path=str(path))
    )

    _verify_frozen_initialization(system, cfg)

    audit = system.artifacts.values["frozen_initialization_match.json"]
    assert audit["matched"] is True
    assert audit["execution_identity_transition_applied"] is True


def test_frozen_initialization_transition_never_masks_scientific_mismatch(
    tmp_path,
) -> None:
    source_identity = _identity("source-commit", "source-manifest")
    target_identity = _identity("target-commit", "target-manifest")
    expected = {
        "initial_prompt_hashes": ["prompt"] * 5,
        "initial_member_correct_counts": [59] * 5,
        "initial_team_outcome": {"team_vote_correct_count": 59},
        "initial_vote_oracle_ghm_hash": "ghm",
        "initial_train_state_hash": "source-state-hash",
        "probe_hash": "probe",
        "solver_request_identity": "request",
        "solver_identity": ["solver"],
        "immutable_run_identity": source_identity,
    }
    actual = {
        **expected,
        "initial_member_correct_counts": [58, 59, 59, 59, 59],
        "initial_train_state_hash": "target-state-hash",
        "immutable_run_identity": target_identity,
    }
    manifest = {
        "manifest_version": "fixture",
        "initialization_snapshot": expected,
        "execution_identity_transition": {
            "schema_version": "frozen_initialization_execution_identity_transition_v1",
            "source_immutable_run_identity": source_identity,
            "target_immutable_run_identity": target_identity,
            "scientific_state_fields": [
                "initial_prompt_hashes",
                "initial_member_correct_counts",
                "initial_team_outcome",
                "initial_vote_oracle_ghm_hash",
                "probe_hash",
                "solver_request_identity",
                "solver_identity",
            ],
        },
    }
    path = tmp_path / "initialization.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    system = _InitializationSystem(actual)
    cfg = SimpleNamespace(
        persistence=SimpleNamespace(frozen_initialization_manifest_path=str(path))
    )

    with pytest.raises(RuntimeError, match="frozen initialization mismatch"):
        _verify_frozen_initialization(system, cfg)
