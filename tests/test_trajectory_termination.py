from __future__ import annotations

from dataclasses import dataclass

import pytest

from multi_dataset_diverse_rl.cli import pending_epoch_indices
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

