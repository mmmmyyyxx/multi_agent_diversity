"""The sole responsibility-to-local-task translation bridge."""

from __future__ import annotations

from dataclasses import dataclass

from ..local_optimizers.schemas import LocalEvidenceExample, LocalOptimizationTask, LocalOptimizerBudget
from .schemas import TeamEvidenceCase, TeamSearchAssignment, TeamSearchRequest
from ..versions import TEAM_MINIBATCH_CONTRACT_VERSION


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

    @staticmethod
    def _matches_primary_lane(row: TeamEvidenceCase, primary_lane: str | None) -> bool:
        if row.evidence_group != "responsibility":
            return True
        if primary_lane in (None, "fallback"):
            return True
        if primary_lane == "coverage":
            return "coverage" in row.tags or "pure_coverage" in row.tags
        return primary_lane in row.tags

    def select_team_minibatch(
        self,
        evidence: tuple[TeamEvidenceCase, ...],
        *,
        primary_responsibility_lane: str | None = None,
    ) -> tuple[TeamEvidenceCase, ...]:
        if primary_responsibility_lane not in {
            None, "direct_flip", "near_margin", "coverage", "fallback"
        }:
            raise ValueError("unknown primary responsibility lane")
        limits = {
            "responsibility": self.quota.responsibility,
            "coalition": self.quota.coalition,
            "preservation": self.quota.preservation,
        }
        chosen: list[TeamEvidenceCase] = []
        seen: set[str] = set()
        for group in ("responsibility", "coalition", "preservation"):
            rows = sorted(
                (
                    row
                    for row in evidence
                    if row.evidence_group == group
                    and self._matches_primary_lane(row, primary_responsibility_lane)
                ),
                key=self._priority,
            )
            unique_rows = []
            for row in rows:
                if row.example_id not in seen:
                    unique_rows.append(row)
                    seen.add(row.example_id)
                if len(unique_rows) == limits[group]:
                    break
            if len(unique_rows) != limits[group]:
                raise ValueError(
                    f"{TEAM_MINIBATCH_CONTRACT_VERSION} requires exactly "
                    f"{limits[group]} unique {group} examples"
                )
            chosen.extend(unique_rows)
        if len(chosen) != self.quota.total or len({row.example_id for row in chosen}) != self.quota.total:
            raise ValueError(f"{TEAM_MINIBATCH_CONTRACT_VERSION} requires exactly 12 unique examples")
        return tuple(chosen)

    @staticmethod
    def team_minibatch_telemetry(
        rows: tuple[TeamEvidenceCase, ...],
    ) -> dict[str, int | str]:
        counts = {
            group: sum(row.evidence_group == group for row in rows)
            for group in ("responsibility", "coalition", "preservation")
        }
        return {
            "contract_version": TEAM_MINIBATCH_CONTRACT_VERSION,
            "total_count": len(rows),
            "responsibility_count": counts["responsibility"],
            "coalition_count": counts["coalition"],
            "preservation_count": counts["preservation"],
            "backfill_count": 0,
        }

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
        evidence = assignment.evidence
        all_by_id = {row.example_id: row for row in evidence}
        primary_lane = assignment.primary_responsibility_lane
        if primary_lane is not None:
            if primary_lane not in {"direct_flip", "near_margin", "coverage", "fallback"}:
                raise ValueError("unknown primary responsibility lane")
            responsibility = tuple(
                row for row in evidence
                if row.evidence_group == "responsibility"
                and self._matches_primary_lane(row, primary_lane)
            )
            if primary_lane != "fallback" and not responsibility:
                raise ValueError("primary lane has no responsibility evidence")
            evidence = tuple(
                row for row in evidence
                if row.evidence_group != "responsibility" or row in responsibility
            )
        search = tuple(self._local(row) for row in evidence)
        by_id = {row.example_id: row for row in evidence}
        requested_validation_ids = assignment.local_validation_example_ids or tuple(by_id)
        if any(example_id not in all_by_id for example_id in requested_validation_ids):
            raise ValueError("local validation ids must refer to assigned Optimize evidence")
        validation_ids = tuple(example_id for example_id in requested_validation_ids if example_id in by_id)
        if not validation_ids:
            raise ValueError("primary-lane filtering removed all local validation evidence")
        local_validation = tuple(self._local(by_id[example_id]) for example_id in validation_ids)
        context = assignment.optimization_context
        if primary_lane is not None:
            context = (
                f"primary_responsibility_lane={primary_lane}\n"
                "Responsibility evidence is restricted to that lane; preservation and "
                f"general context remain non-target evidence.\n{context}"
            )
        return LocalOptimizationTask(
            task_id=f"seed{request.seed}_update{request.update_index}_member{assignment.target_member}",
            parent_prompt=assignment.parent_prompt,
            search_examples=search,
            local_validation_examples=local_validation,
            optimization_context=context,
            solver_contract_id=request.solver_contract_id,
            output_contract_id=request.output_contract_id,
            seed=request.seed * 100_000 + request.update_index * 10 + assignment.target_member,
            budget=LocalOptimizerBudget(
                max_metric_calls=request.local_metric_budget,
                reflection_minibatch_size=3,
                max_returned_candidates=self.local_return_budget,
            ),
        )
