"""Backend-neutral stopping semantics for fixed-budget and saturation runs.

This module is deliberately observational.  It does not choose examples,
targets, candidates, or write-backs.  Backends report complete native/local
units and Layer 2 reports completed opportunities; this module classifies the
resulting stop without changing the search policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import time
from typing import Any, Iterable, Mapping, Sequence

from .versions import (
    LAYER2_EVIDENCE_EPOCH_POLICY_VERSION,
    TEAM_EPOCH_SEMANTICS_VERSION,
)


LAYER2_EVIDENCE_EPOCH_POLICY_V1 = LAYER2_EVIDENCE_EPOCH_POLICY_VERSION
TEAM_EPOCH_SEMANTICS_V1 = TEAM_EPOCH_SEMANTICS_VERSION


class RunMode(str, Enum):
    FIXED_BUDGET = "fixed_budget"
    SATURATION = "saturation"


class StopReason(str, Enum):
    SATURATION_REACHED = "SATURATION_REACHED"
    SCIENTIFIC_BUDGET_REACHED = "SCIENTIFIC_BUDGET_REACHED"
    EMERGENCY_PROVIDER_CALL_CEILING = "EMERGENCY_PROVIDER_CALL_CEILING"
    EMERGENCY_OPTIMIZER_STEP_CEILING = "EMERGENCY_OPTIMIZER_STEP_CEILING"
    EMERGENCY_TEAM_EPOCH_CEILING = "EMERGENCY_TEAM_EPOCH_CEILING"
    EMERGENCY_WALL_TIME_CEILING = "EMERGENCY_WALL_TIME_CEILING"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    OPERATIONAL_ABORT = "OPERATIONAL_ABORT"
    INVALID_STATE = "INVALID_STATE"
    INSUFFICIENT_LAYER2_EVIDENCE = "INSUFFICIENT_LAYER2_EVIDENCE"


class OptimizationUnitType(str, Enum):
    GEPA_NATIVE_EPOCH = "GEPA_NATIVE_EPOCH"
    LAYER2_EVIDENCE_EPOCH = "LAYER2_EVIDENCE_EPOCH"
    MARS_NATIVE_ROUND = "MARS_NATIVE_ROUND"
    MARS_LAYER2_ROUND = "MARS_LAYER2_ROUND"
    TEAM_EPOCH = "TEAM_EPOCH"


@dataclass(frozen=True)
class SaturationConfig:
    """Shared run-regime configuration.

    Emergency values are intentionally high operational limits.  They remain
    mandatory in both regimes and never imply scientific convergence.
    """

    enabled: bool = False
    scientific_budget_enabled: bool = True
    local_no_update_patience: int | None = None
    team_no_update_patience: int | None = None
    emergency_max_provider_calls: int = 100_000
    emergency_max_optimizer_steps: int = 100_000
    emergency_max_team_epochs: int = 10_000
    emergency_max_wall_seconds: int | None = 86_400

    def __post_init__(self) -> None:
        if self.enabled and self.scientific_budget_enabled:
            raise ValueError("saturation mode must disable ordinary scientific budgets")
        if self.enabled and self.local_no_update_patience is None:
            raise ValueError("saturation mode requires local no-update patience")
        for name, value in (
            ("local_no_update_patience", self.local_no_update_patience),
            ("team_no_update_patience", self.team_no_update_patience),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when configured")
        for name, value in (
            ("emergency_max_provider_calls", self.emergency_max_provider_calls),
            ("emergency_max_optimizer_steps", self.emergency_max_optimizer_steps),
            ("emergency_max_team_epochs", self.emergency_max_team_epochs),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.emergency_max_wall_seconds is not None and self.emergency_max_wall_seconds <= 0:
            raise ValueError("emergency_max_wall_seconds must be positive when configured")

    @property
    def run_mode(self) -> RunMode:
        return RunMode.SATURATION if self.enabled else RunMode.FIXED_BUDGET

    def identity(self) -> str:
        return state_hash(asdict(self))


@dataclass(frozen=True)
class SaturationUnitRecord:
    unit_index: int
    unit_type: str
    backend: str
    mode: str
    start_state_hash: str
    end_state_hash: str
    accepted_local_update: bool
    team_commit: bool | None
    local_objective_before: float | None
    local_objective_after: float | None
    vote_acc: float | None
    oracle_acc: float | None
    no_update_counter_before: int
    no_update_counter_after: int
    stop_triggered: str | None


@dataclass
class SaturationCostAccounting:
    solver_calls: int = 0
    optimizer_calls: int = 0
    reflection_calls: int = 0
    planner_calls: int = 0
    teacher_calls: int = 0
    critic_calls: int = 0
    student_calls: int = 0
    team_evaluation_calls: int = 0
    provider_attempts: int = 0
    provider_successes: int = 0
    provider_failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, **deltas: int) -> None:
        unknown = set(deltas) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown saturation accounting fields: {sorted(unknown)}")
        if any(int(value) < 0 for value in deltas.values()):
            raise ValueError("saturation accounting deltas cannot be negative")
        for name, value in deltas.items():
            setattr(self, name, int(getattr(self, name)) + int(value))


@dataclass
class SaturationState:
    config: SaturationConfig
    backend: str
    mode: str
    local_no_update_counter: int = 0
    team_no_update_counter: int = 0
    local_epoch_or_round_count: int = 0
    team_epoch_count: int = 0
    local_accepted_update_count: int = 0
    team_commit_count: int = 0
    provider_calls: int = 0
    optimizer_steps: int = 0
    stop_reason: StopReason | None = None
    emergency_ceiling_triggered: bool = False
    emergency_ceiling_type: str | None = None
    state_hashes: list[str] = field(default_factory=list)
    accepted_update_hashes: list[str] = field(default_factory=list)
    revisited_state_count: int = 0
    trajectory: list[SaturationUnitRecord] = field(default_factory=list)
    cost: SaturationCostAccounting = field(default_factory=SaturationCostAccounting)
    _started_at: float = field(default_factory=time.monotonic, repr=False)

    @property
    def scientific_stopping_condition_reached(self) -> bool:
        return self.stop_reason in {
            StopReason.SATURATION_REACHED,
            StopReason.SCIENTIFIC_BUDGET_REACHED,
        }

    @property
    def run_status(self) -> str:
        if self.stop_reason is None:
            return "RUNNING"
        if self.stop_reason in {
            StopReason.SATURATION_REACHED,
            StopReason.SCIENTIFIC_BUDGET_REACHED,
        }:
            return "COMPLETE"
        return "EXECUTION_ABORTED"

    @property
    def oscillatory_search_suspected(self) -> bool:
        return self.revisited_state_count > 0 and self.stop_reason in {
            StopReason.EMERGENCY_PROVIDER_CALL_CEILING,
            StopReason.EMERGENCY_OPTIMIZER_STEP_CEILING,
            StopReason.EMERGENCY_TEAM_EPOCH_CEILING,
            StopReason.EMERGENCY_WALL_TIME_CEILING,
        }

    def _observe_hash(self, state_hash: str, *, accepted: bool) -> None:
        if not state_hash:
            raise ValueError("state hashes must be non-empty")
        if state_hash in self.state_hashes:
            self.revisited_state_count += 1
        self.state_hashes.append(state_hash)
        if accepted:
            self.accepted_update_hashes.append(state_hash)

    def add_usage(self, *, provider_calls: int = 0, optimizer_steps: int = 0) -> None:
        if provider_calls < 0 or optimizer_steps < 0:
            raise ValueError("usage deltas cannot be negative")
        self.provider_calls += provider_calls
        self.optimizer_steps += optimizer_steps
        self.cost.add(
            provider_attempts=provider_calls,
            provider_successes=provider_calls,
        )

    def add_cost(self, **deltas: int) -> None:
        self.cost.add(**deltas)

    def _emergency_reason(self) -> StopReason | None:
        cfg = self.config
        if self.provider_calls >= cfg.emergency_max_provider_calls:
            return StopReason.EMERGENCY_PROVIDER_CALL_CEILING
        if self.optimizer_steps >= cfg.emergency_max_optimizer_steps:
            return StopReason.EMERGENCY_OPTIMIZER_STEP_CEILING
        if self.team_epoch_count >= cfg.emergency_max_team_epochs:
            return StopReason.EMERGENCY_TEAM_EPOCH_CEILING
        if (
            cfg.emergency_max_wall_seconds is not None
            and time.monotonic() - self._started_at >= cfg.emergency_max_wall_seconds
        ):
            return StopReason.EMERGENCY_WALL_TIME_CEILING
        return None

    def check_emergency(self) -> StopReason | None:
        reason = self._emergency_reason()
        if reason is not None:
            self.stop_reason = reason
            self.emergency_ceiling_triggered = True
            self.emergency_ceiling_type = reason.value
        return reason

    def observe_local_unit(
        self,
        *,
        unit_type: OptimizationUnitType,
        start_state_hash: str,
        end_state_hash: str,
        accepted_update: bool,
        local_objective_before: float | None = None,
        local_objective_after: float | None = None,
        vote_acc: float | None = None,
        oracle_acc: float | None = None,
        scientific_budget_reached: bool = False,
    ) -> StopReason | None:
        if self.stop_reason is not None:
            raise RuntimeError("cannot observe a unit after termination")
        if accepted_update and start_state_hash == end_state_hash:
            self.stop_reason = StopReason.INVALID_STATE
            return self.stop_reason
        before = self.local_no_update_counter
        self.local_epoch_or_round_count += 1
        if accepted_update:
            self.local_accepted_update_count += 1
            self.local_no_update_counter = 0
        else:
            self.local_no_update_counter += 1
        self._observe_hash(end_state_hash, accepted=accepted_update)
        stop = self.check_emergency()
        if stop is None and scientific_budget_reached and self.config.scientific_budget_enabled:
            stop = self.stop_reason = StopReason.SCIENTIFIC_BUDGET_REACHED
        if (
            stop is None
            and self.config.enabled
            and self.local_no_update_counter >= int(self.config.local_no_update_patience or 0)
        ):
            stop = self.stop_reason = StopReason.SATURATION_REACHED
        self.trajectory.append(SaturationUnitRecord(
            unit_index=len(self.trajectory) + 1,
            unit_type=unit_type.value,
            backend=self.backend,
            mode=self.mode,
            start_state_hash=start_state_hash,
            end_state_hash=end_state_hash,
            accepted_local_update=accepted_update,
            team_commit=None,
            local_objective_before=local_objective_before,
            local_objective_after=local_objective_after,
            vote_acc=vote_acc,
            oracle_acc=oracle_acc,
            no_update_counter_before=before,
            no_update_counter_after=self.local_no_update_counter,
            stop_triggered=stop.value if stop else None,
        ))
        return stop

    def observe_team_epoch(
        self,
        *,
        start_state_hash: str,
        end_state_hash: str,
        team_commit: bool,
        accepted_local_update: bool,
        vote_acc: float | None = None,
        oracle_acc: float | None = None,
        scientific_budget_reached: bool = False,
    ) -> StopReason | None:
        if self.stop_reason is not None:
            raise RuntimeError("cannot observe a team epoch after termination")
        if team_commit and start_state_hash == end_state_hash:
            self.stop_reason = StopReason.INVALID_STATE
            return self.stop_reason
        before = self.team_no_update_counter
        self.team_epoch_count += 1
        if team_commit:
            self.team_commit_count += 1
            self.team_no_update_counter = 0
        else:
            self.team_no_update_counter += 1
        self._observe_hash(end_state_hash, accepted=team_commit)
        stop = self.check_emergency()
        if stop is None and scientific_budget_reached and self.config.scientific_budget_enabled:
            stop = self.stop_reason = StopReason.SCIENTIFIC_BUDGET_REACHED
        if (
            stop is None
            and self.config.enabled
            and self.config.team_no_update_patience is not None
            and self.team_no_update_counter >= self.config.team_no_update_patience
        ):
            stop = self.stop_reason = StopReason.SATURATION_REACHED
        self.trajectory.append(SaturationUnitRecord(
            unit_index=len(self.trajectory) + 1,
            unit_type=OptimizationUnitType.TEAM_EPOCH.value,
            backend=self.backend,
            mode=self.mode,
            start_state_hash=start_state_hash,
            end_state_hash=end_state_hash,
            accepted_local_update=accepted_local_update,
            team_commit=team_commit,
            local_objective_before=None,
            local_objective_after=None,
            vote_acc=vote_acc,
            oracle_acc=oracle_acc,
            no_update_counter_before=before,
            no_update_counter_after=self.team_no_update_counter,
            stop_triggered=stop.value if stop else None,
        ))
        return stop

    def telemetry(self) -> Mapping[str, Any]:
        return {
            "saturation_mode_enabled": self.config.enabled,
            "scientific_budget_enabled": self.config.scientific_budget_enabled,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "run_status": self.run_status,
            "scientific_stopping_condition_reached": self.scientific_stopping_condition_reached,
            "local_no_update_patience": self.config.local_no_update_patience,
            "team_no_update_patience": self.config.team_no_update_patience,
            "local_no_update_counter": self.local_no_update_counter,
            "team_no_update_counter": self.team_no_update_counter,
            "local_epoch_or_round_count": self.local_epoch_or_round_count,
            "team_epoch_count": self.team_epoch_count if "layer2" in self.mode.lower() else None,
            "local_accepted_update_count": self.local_accepted_update_count,
            "team_commit_count": self.team_commit_count if "layer2" in self.mode.lower() else None,
            "emergency_ceiling_triggered": self.emergency_ceiling_triggered,
            "emergency_ceiling_type": self.emergency_ceiling_type,
            "provider_calls": self.provider_calls,
            "optimizer_steps": self.optimizer_steps,
            "cost_accounting": asdict(self.cost),
            "wall_seconds": max(0.0, time.monotonic() - self._started_at),
            "state_hashes": list(self.state_hashes),
            "accepted_update_hashes": list(self.accepted_update_hashes),
            "revisited_state_count": self.revisited_state_count,
            "oscillation_diagnostic": (
                "OSCILLATORY_SEARCH_SUSPECTED" if self.oscillatory_search_suspected else None
            ),
            "trajectory": [asdict(row) for row in self.trajectory],
        }


@dataclass
class TeamEpochTracker:
    """Observe one epoch without changing scheduler choices.

    Eligibility is frozen at epoch start from scheduler summaries.  Existing
    scheduler decisions are then observed until every eligible member has
    received at least one opportunity.  A member may appear multiple times;
    zero-score members remain eligible when the scheduler's always-two/fallback
    policy includes them.  Persistent realizability is owned by the scheduler
    and is intentionally not reset here.
    """

    eligible_member_ids: tuple[int, ...]
    seen_member_ids: set[int] = field(default_factory=set)
    any_local_update: bool = False
    any_team_commit: bool = False
    opportunity_count: int = 0

    def __post_init__(self) -> None:
        if not self.eligible_member_ids or len(set(self.eligible_member_ids)) != len(
            self.eligible_member_ids
        ):
            raise ValueError("team epoch requires unique eligible members")

    def observe_opportunity(
        self,
        *,
        selected_member_ids: Sequence[int],
        local_accepted_update: bool,
        team_commit: bool,
    ) -> bool:
        selected = tuple(map(int, selected_member_ids))
        if not selected or not set(selected).issubset(set(self.eligible_member_ids)):
            raise ValueError("selected members must belong to the frozen eligible set")
        self.opportunity_count += 1
        self.seen_member_ids.update(selected)
        self.any_local_update = self.any_local_update or bool(local_accepted_update)
        self.any_team_commit = self.any_team_commit or bool(team_commit)
        return set(self.eligible_member_ids).issubset(self.seen_member_ids)


class Layer2TeamSaturationController:
    """Shared GEPA/MARS outer-loop saturation observer.

    The caller continues to use its existing scheduler and opportunity runner.
    This class merely groups those decisions into deterministic coverage epochs
    and updates the common team no-commit counter.
    """

    def __init__(self, state: SaturationState) -> None:
        if not state.config.enabled or state.config.team_no_update_patience is None:
            raise ValueError("Layer-2 team saturation requires enabled team patience")
        if "layer2" not in state.mode.lower():
            raise ValueError("team saturation is only valid for Layer-2 modes")
        self.state = state
        self.current_epoch: TeamEpochTracker | None = None
        self.epoch_start_state_hash: str | None = None

    def begin_epoch(
        self, *, eligible_member_ids: Sequence[int], start_state_hash: str
    ) -> None:
        if self.current_epoch is not None:
            raise RuntimeError("cannot begin a new team epoch before completing the current one")
        self.current_epoch = TeamEpochTracker(tuple(map(int, eligible_member_ids)))
        self.epoch_start_state_hash = start_state_hash

    def observe_opportunity(
        self,
        *,
        selected_member_ids: Sequence[int],
        local_accepted_update: bool,
        team_commit: bool,
        end_state_hash: str,
        vote_acc: float | None = None,
        oracle_acc: float | None = None,
        scientific_budget_reached: bool = False,
        provider_calls: int = 0,
    ) -> StopReason | None:
        if self.current_epoch is None or self.epoch_start_state_hash is None:
            raise RuntimeError("team epoch has not been started")
        self.state.add_usage(provider_calls=provider_calls, optimizer_steps=1)
        complete = self.current_epoch.observe_opportunity(
            selected_member_ids=selected_member_ids,
            local_accepted_update=local_accepted_update,
            team_commit=team_commit,
        )
        if not complete:
            return self.state.check_emergency()
        stop = self.state.observe_team_epoch(
            start_state_hash=self.epoch_start_state_hash,
            end_state_hash=end_state_hash,
            team_commit=self.current_epoch.any_team_commit,
            accepted_local_update=self.current_epoch.any_local_update,
            vote_acc=vote_acc,
            oracle_acc=oracle_acc,
            scientific_budget_reached=scientific_budget_reached,
        )
        self.current_epoch = None
        self.epoch_start_state_hash = None
        return stop


def state_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def eligible_members_from_scheduler_summaries(
    summaries: Iterable[Any],
) -> tuple[int, ...]:
    """Return the current scheduler's member universe without re-ranking it."""

    members = tuple(int(row.member_id) for row in summaries)
    if not members or len(set(members)) != len(members):
        raise ValueError("scheduler summaries must contain unique members")
    return members
