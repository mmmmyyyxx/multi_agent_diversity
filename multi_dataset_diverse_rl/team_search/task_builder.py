"""The sole responsibility-to-local-task translation bridge."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from ..local_optimizers.schemas import LocalEvidenceExample, LocalOptimizationTask, LocalOptimizerBudget
from ..native_feed import (
    Layer2OptimizationRequest,
    NativeOptimizationRequest,
    NativeResourceBudget,
    PacketEvidenceExample,
    ResponsibilityEvidencePacket,
    ResponsibilityContext,
)
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


class NativeFeedRequestBuilder(LocalTaskBuilder):
    """Build a native control request without Layer-2 evidence/responsibility.

    The target and parent are frozen by the outer experiment. The backend owns
    its Optimize-only examples; Layer-2 residuals, lanes and evidence are not
    transmitted to the native optimizer.
    """

    def build(
        self, request: TeamSearchRequest, assignment: TeamSearchAssignment
    ) -> NativeOptimizationRequest:
        if not assignment.evidence:
            raise ValueError("team assignment contains no Optimize diagnosis")
        if any(row.source_split != "optimize" for row in assignment.evidence):
            raise ValueError("responsibility diagnosis must be Optimize-derived")
        responsibility = ResponsibilityContext(
            primary_lane="generic",
            responsibility_identity="native_control_no_layer2_responsibility_v1",
            responsibility_value=0.0,
            team_failure_summary={},
            coverage_summary={},
            peer_structure_summary={
                "team_size": 5,
                "aggregation": "equal_weight_plurality",
                "tie_policy": "abstain_incorrect",
            },
        )
        return NativeOptimizationRequest(
            request_id=(
                f"seed{request.seed}_update{request.update_index}_"
                f"member{assignment.target_member}_native"
            ),
            parent_decision_procedure=assignment.parent_prompt,
            target_member=assignment.target_member,
            responsibility=responsibility,
            team_state_identity=request.team_state_hash,
            optimize_universe_id=request.optimize_universe_id,
            solver_contract_id=request.solver_contract_id,
            output_contract_id=request.output_contract_id,
            seed=request.seed * 100_000 + request.update_index * 10 + assignment.target_member,
            budget=NativeResourceBudget(
                native_unit_limit=max(1, request.local_metric_budget),
                metric_call_limit=request.local_metric_budget,
                optimizer_call_limit=max(1, request.local_metric_budget),
                max_returned_candidates=self.local_return_budget,
            ),
            provenance={
                "builder": "NativeFeedRequestBuilder",
                "source_split": "optimize_only",
                "responsibility_semantics": "absent_native_control",
            },
        )


class Layer2EvidenceRequestBuilder(LocalTaskBuilder):
    """Freeze the complete local-search curriculum before Layer 1 starts."""

    packet_selection_policy = "responsibility_plus_latest_transition_frozen_eval_v2"

    @staticmethod
    def _packet_row(
        row: TeamEvidenceCase, *, role: str, lane: str
    ) -> PacketEvidenceExample:
        return PacketEvidenceExample(
            example_id=row.example_id,
            input_payload=row.input_payload,
            gold=row.gold,
            parent_output=row.target_output,
            textual_feedback=row.feedback,
            responsibility_role=role,
            lane=lane,
            metadata=(
                ("source_split", row.source_split),
                ("team_evidence_group", row.evidence_group),
            ),
        )

    @staticmethod
    def _universe_hash(rows: tuple[TeamEvidenceCase, ...]) -> str:
        import hashlib
        import json

        payload = [
            {
                "id": row.example_id,
                "input": row.input_payload,
                "gold": row.gold,
                "target_output": row.target_output,
                "feedback": row.feedback,
                "group": row.evidence_group,
                "tags": list(row.tags),
                "split": row.source_split,
            }
            for row in sorted(rows, key=lambda item: item.example_id)
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _schedule(
        responsibility_ids: tuple[str, ...],
        focus_ids: tuple[str, ...],
        anchor_ids: tuple[str, ...],
        *,
        batch_count: int,
        batch_size: int = 3,
    ) -> tuple[tuple[str, ...], ...]:
        roles = (responsibility_ids, focus_ids, anchor_ids)
        ordered: list[str] = []
        max_count = max(map(len, roles))
        for index in range(max_count):
            for values in roles:
                if index < len(values):
                    ordered.append(values[index])
        return tuple(
            tuple(ordered[(step * batch_size + offset) % len(ordered)] for offset in range(batch_size))
            for step in range(batch_count)
        )

    def build(
        self, request: TeamSearchRequest, assignment: TeamSearchAssignment
    ) -> Layer2OptimizationRequest:
        if not assignment.evidence:
            raise ValueError("team assignment contains no Optimize evidence")
        if any(row.source_split != "optimize" for row in assignment.evidence):
            raise ValueError("Layer-2 packet may contain only Optimize-derived evidence")
        lane = assignment.primary_responsibility_lane or "fallback"
        if lane not in {"direct_flip", "near_margin", "coverage", "fallback"}:
            raise ValueError("unknown primary responsibility lane")
        by_id = {row.example_id: row for row in assignment.evidence}
        if len(by_id) != len(assignment.evidence):
            raise ValueError("Layer-2 Optimize evidence ids must be unique")
        responsibility_rows = tuple(
            sorted(
                (
                    row
                    for row in assignment.evidence
                    if row.evidence_group == "responsibility"
                    and self._matches_primary_lane(row, lane)
                ),
                key=self._priority,
            )
        )
        if assignment.local_validation_example_ids:
            if len(assignment.local_validation_example_ids) != len(
                set(assignment.local_validation_example_ids)
            ):
                raise ValueError("local-eval example ids must be unique")
            missing_local_eval = tuple(
                row_id
                for row_id in assignment.local_validation_example_ids
                if row_id not in by_id
            )
            if missing_local_eval:
                raise ValueError(
                    "INSUFFICIENT_LAYER2_EVIDENCE: frozen local-eval example absent "
                    "from Optimize evidence"
                )
            local_eval_rows = tuple(
                by_id[row_id] for row_id in assignment.local_validation_example_ids
            )
            local_eval_source = "assignment_frozen_ids"
        else:
            # Compatibility path for direct packet construction. This is still a
            # deterministic Layer-2 decision and never asks the backend/global pool
            # to fill missing evidence.
            local_eval_rows = tuple(
                sorted(
                    (
                        row
                        for row in assignment.evidence
                        if row.evidence_group == "coalition"
                    ),
                    key=self._priority,
                )
            )
            local_eval_source = "deterministic_coalition_rows"
        if not responsibility_rows:
            raise ValueError("INSUFFICIENT_LAYER2_EVIDENCE: no aligned responsibility examples")
        if not local_eval_rows:
            raise ValueError("INSUFFICIENT_LAYER2_EVIDENCE: no local-eval examples")
        transition = assignment.latest_transition
        focus_rows = () if transition is None else tuple(
            by_id[row_id] for row_id in transition.newly_broken_ids if row_id in by_id
        )
        anchor_rows = () if transition is None else tuple(
            by_id[row_id] for row_id in transition.newly_fixed_ids if row_id in by_id
        )
        if transition is not None and (
            len(focus_rows) != len(transition.newly_broken_ids)
            or len(anchor_rows) != len(transition.newly_fixed_ids)
        ):
            raise ValueError("INSUFFICIENT_LAYER2_EVIDENCE: transition example absent from frozen universe")
        budget = NativeResourceBudget(
            native_unit_limit=max(1, request.local_metric_budget // 3),
            metric_call_limit=request.local_metric_budget,
            optimizer_call_limit=max(1, request.local_metric_budget),
            max_returned_candidates=self.local_return_budget,
        )
        responsibility = tuple(
            self._packet_row(row, role="responsibility", lane=lane)
            for row in responsibility_rows
        )
        focus = tuple(
            self._packet_row(row, role="focus", lane="global") for row in focus_rows
        )
        anchor = tuple(
            self._packet_row(row, role="anchor", lane="global") for row in anchor_rows
        )
        local_eval = tuple(
            self._packet_row(row, role="local_eval", lane="global")
            for row in local_eval_rows
        )
        schedule = self._schedule(
            tuple(row.packet_item_id for row in responsibility),
            tuple(row.packet_item_id for row in focus),
            tuple(row.packet_item_id for row in anchor),
            batch_count=budget.native_unit_limit,
        )
        parent_candidate_hash = hashlib.sha256(
            assignment.parent_prompt.encode("utf-8")
        ).hexdigest()
        if transition is not None and transition.child_candidate_hash != parent_candidate_hash:
            raise ValueError("latest transition child is not the current parent prompt")
        packet = ResponsibilityEvidencePacket(
            source_team_state_hash=request.team_state_hash,
            target_member=assignment.target_member,
            primary_responsibility_lane=lane,
            responsibility_value=assignment.responsibility_value,
            responsibility_context=assignment.optimization_context,
            responsibility_examples=responsibility,
            focus_examples=focus,
            anchor_examples=anchor,
            local_eval_examples=local_eval,
            ordered_batch_schedule=schedule,
            parent_candidate_hash=parent_candidate_hash,
            lineage_parent_hash=(
                transition.parent_candidate_hash if transition is not None else None
            ),
            data_universe_hash=self._universe_hash(assignment.evidence),
            budget=budget,
            provenance=(
                ("builder", "Layer2EvidenceRequestBuilder"),
                ("source_split", "optimize_only"),
                ("selection_policy", self.packet_selection_policy),
                ("local_eval_source", local_eval_source),
                ("batch_fill_policy", "cyclic_repeat_within_frozen_packet_v1"),
            ),
            latest_transition=transition,
        )
        return Layer2OptimizationRequest(
            request_id=(
                f"seed{request.seed}_update{request.update_index}_"
                f"member{assignment.target_member}_layer2_evidence"
            ),
            parent_decision_procedure=assignment.parent_prompt,
            packet=packet,
            solver_contract_id=request.solver_contract_id,
            output_contract_id=request.output_contract_id,
            seed=request.seed * 100_000 + request.update_index * 10 + assignment.target_member,
        )


def task_builder_for_optimization_mode(
    optimization_mode: str, *, local_return_budget: int = 4
) -> NativeFeedRequestBuilder | Layer2EvidenceRequestBuilder:
    """Bind controller/data-flow mode independently from the optimizer backend."""

    if optimization_mode == "native":
        return NativeFeedRequestBuilder(local_return_budget=local_return_budget)
    if optimization_mode == "layer2":
        return Layer2EvidenceRequestBuilder(local_return_budget=local_return_budget)
    raise ValueError("optimization_mode must be native or layer2")
