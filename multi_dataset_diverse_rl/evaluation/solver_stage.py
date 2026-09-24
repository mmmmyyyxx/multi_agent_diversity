"""Fail-closed attribution schema for runtime contexts that can call a Solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


SOLVER_PHASES = frozenset(
    {
        "initialization",
        "local_optimizer_solver_eval",
        "team_minibatch_eval",
        "team_full_eval",
        "diagnostic_full_eval",
        "team_shadow_eval",
        "final_validation",
    }
)


def validate_solver_stage_attribution(stage: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy only when the producer supplied an explicit valid phase."""

    phase = stage.get("phase")
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("Solver stage attribution requires a non-empty phase")
    if phase not in SOLVER_PHASES:
        raise ValueError(f"unknown Solver stage phase: {phase}")
    evaluation_stage = stage.get("evaluation_stage")
    if evaluation_stage is not None and evaluation_stage != phase:
        raise ValueError("Solver phase and evaluation_stage must match")
    return dict(stage)


@dataclass(frozen=True)
class TeamSolverStageAttribution:
    """Complete stage record for Layer-2 candidate evaluation paths."""

    phase: str
    seed: int
    parent_id: str
    update_index: int
    target_member: int
    candidate_id: str
    proposal_engine: str
    evaluation_stage: str

    def __post_init__(self) -> None:
        validate_solver_stage_attribution(asdict(self))

    def as_mapping(self) -> dict[str, Any]:
        return validate_solver_stage_attribution(asdict(self))


def team_solver_stage_attribution(
    *,
    phase: str,
    seed: int,
    parent_id: str,
    update_index: int,
    target_member: int,
    candidate_id: str,
    proposal_engine: str,
) -> dict[str, Any]:
    return TeamSolverStageAttribution(
        phase=phase,
        seed=seed,
        parent_id=parent_id,
        update_index=update_index,
        target_member=target_member,
        candidate_id=candidate_id,
        proposal_engine=proposal_engine,
        evaluation_stage=phase,
    ).as_mapping()
