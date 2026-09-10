"""The sole responsibility-to-local-task translation bridge."""

from __future__ import annotations

from dataclasses import dataclass

from ..local_optimizers.schemas import LocalEvidenceExample, LocalOptimizationTask, LocalOptimizerBudget
from .schemas import TeamEvidenceCase, TeamSearchAssignment, TeamSearchRequest


@dataclass(frozen=True)
class TeamMiniBatchQuota:
    responsibility: int = 4
    coalition: int = 4
    preservation: int = 4

    @property
    def total(self) -> int:
        return self.responsibility + self.coalition + self.preservation


class LocalTaskBuilder:
    def __init__(self, *, quota: TeamMiniBatchQuota | None = None, local_return_budget: int = 4) -> None:
        self.quota = quota or TeamMiniBatchQuota()
        self.local_return_budget = local_return_budget
        if self.quota.total != 12:
            raise ValueError("two_layer_rg_gepa_v1 requires TeamMiniBatch12")

    @staticmethod
    def _priority(row: TeamEvidenceCase) -> tuple[int, str]:
        lane_order = {"direct_flip": 0, "near_margin": 1, "coverage": 2}
        lane = min((lane_order[tag] for tag in row.tags if tag in lane_order), default=3)
        return lane, row.example_id

    def select_team_minibatch(self, evidence: tuple[TeamEvidenceCase, ...]) -> tuple[TeamEvidenceCase, ...]:
        limits = {
            "responsibility": self.quota.responsibility,
            "coalition": self.quota.coalition,
            "preservation": self.quota.preservation,
        }
        chosen: list[TeamEvidenceCase] = []
        seen: set[str] = set()
        for group in ("responsibility", "coalition", "preservation"):
            rows = sorted((row for row in evidence if row.evidence_group == group), key=self._priority)
            for row in rows[: limits[group]]:
                if row.example_id not in seen:
                    chosen.append(row)
                    seen.add(row.example_id)
        for row in sorted(evidence, key=lambda value: (value.evidence_group, *self._priority(value))):
            if len(chosen) >= self.quota.total:
                break
            if row.example_id not in seen:
                chosen.append(row)
                seen.add(row.example_id)
        return tuple(chosen)

    @staticmethod
    def _local(row: TeamEvidenceCase) -> LocalEvidenceExample:
        return LocalEvidenceExample(
            example_id=row.example_id,
            input_payload=row.input_payload,
            gold=row.gold,
            parent_output=row.target_output,
            textual_feedback=row.feedback,
            tags=row.tags,
        )

    def build(self, request: TeamSearchRequest, assignment: TeamSearchAssignment) -> LocalOptimizationTask:
        if not assignment.evidence:
            raise ValueError("team assignment contains no Optimize evidence")
        search = tuple(self._local(row) for row in assignment.evidence)
        by_id = {row.example_id: row for row in assignment.evidence}
        validation_ids = assignment.local_validation_example_ids or tuple(by_id)
        if any(example_id not in by_id for example_id in validation_ids):
            raise ValueError("local validation ids must refer to assigned Optimize evidence")
        local_validation = tuple(self._local(by_id[example_id]) for example_id in validation_ids)
        return LocalOptimizationTask(
            task_id=f"seed{request.seed}_update{request.update_index}_member{assignment.target_member}",
            parent_prompt=assignment.parent_prompt,
            search_examples=search,
            local_validation_examples=local_validation,
            optimization_context=assignment.optimization_context,
            solver_contract_id=request.solver_contract_id,
            output_contract_id=request.output_contract_id,
            seed=request.seed * 100_000 + request.update_index * 10 + assignment.target_member,
            budget=LocalOptimizerBudget(
                max_metric_calls=request.local_metric_budget,
                reflection_minibatch_size=3,
                max_returned_candidates=self.local_return_budget,
            ),
        )
