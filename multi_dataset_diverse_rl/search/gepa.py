"""Versioned GEPA-derived main-method configuration.

Historical Formal V3 uses the exact locked GEPAOptimizerConfig directly.
This type intentionally has no exact-tuple freeze, so a future scientific
change can receive a new method/config identity without changing baseline code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from ..local_optimizers.gepa_optimizer import GEPAOptimizerConfig
from ..native_feed import (
    CandidateTransitionAudit, Layer2OptimizationRequest, NativeResourceBudget,
    PacketEvidenceExample, ResponsibilityEvidencePacket,
)
from ..versions import (
    LAYER2_EVIDENCE_PACKET_V4_VERSION,
    LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION,
)
from .history import HistoryState
from .schemas import EvidenceItem, OptimizationOpportunity, SearchContractError


@dataclass(frozen=True)
class GEPADerivedConfig(GEPAOptimizerConfig):
    optimizer_fidelity_level: str = "GEPA_DERIVED_SEARCH_V1"

    def __post_init__(self) -> None:
        if self.reflection_minibatch_size <= 0 or self.k_local_return <= 0:
            raise SearchContractError("GEPA search capacities must be positive")
        if self.max_prompt_chars <= 0 or not self.engine_acceptance_semantics:
            raise SearchContractError("GEPA search contract incomplete")

    def identity(self) -> str:
        return hashlib.sha256(("gepa_derived_v1:" + super().identity()).encode("ascii")).hexdigest()


class CurrentGEPAPacketAdapter:
    """Translate role-qualified unified evidence to the frozen GEPA API packet.

    This adapter owns no example selection. It only serializes the roles that
    the opportunity builder has already frozen.
    """

    identity = "current_gepa_packet_adapter_v1"
    responsibility_context = (
        "Use only general reasoning rules. Preserve broad competence, do not copy "
        "examples, and do not modify or discuss the output interface."
    )

    @staticmethod
    def _row(row: EvidenceItem, *, role: str, lane: str) -> PacketEvidenceExample:
        data = row.signals
        if row.source_split != "optimize":
            raise SearchContractError("GEPA packet may use only Optimize evidence")
        return PacketEvidenceExample(
            example_id=row.example_id,
            input_payload=str(data["input_payload"]),
            gold=str(data["gold"]),
            parent_output=data.get("target_output"),
            textual_feedback=data.get("feedback"),
            responsibility_role=role,
            lane=lane,
            metadata=(("source_split", row.source_split),
                      ("team_evidence_group", str(data["legacy_group"]))),
        )

    @staticmethod
    def _ids_json(rows: tuple[EvidenceItem, ...]) -> str:
        return json.dumps([row.example_id for row in rows], separators=(",", ":"))

    @classmethod
    def _ids_hash(cls, rows: tuple[EvidenceItem, ...]) -> str:
        return hashlib.sha256(cls._ids_json(rows).encode("utf-8")).hexdigest()

    @staticmethod
    def _universe_hash(rows: tuple[EvidenceItem, ...]) -> str:
        payload = [{
            "id": row.example_id,
            "input": row.signals["input_payload"],
            "gold": row.signals["gold"],
            "target_output": row.signals.get("target_output"),
            "feedback": row.signals.get("feedback"),
            "group": row.signals["legacy_group"],
            "tags": list(row.signals["legacy_tags"]),
            "split": row.source_split,
        } for row in sorted(rows, key=lambda item: item.example_id)]
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()

    @staticmethod
    def _schedule(roles: tuple[tuple[str, ...], ...], batch_count: int) -> tuple[tuple[str, ...], ...]:
        ordered = [item for index in range(max(map(len, roles)))
                   for values in roles for item in values[index:index + 1]]
        if not ordered:
            raise SearchContractError("GEPA mutation schedule is empty")
        return tuple(tuple(ordered[(step * 3 + offset) % len(ordered)]
                           for offset in range(3)) for step in range(batch_count))

    def build(
        self, opportunity: OptimizationOpportunity, *, history: HistoryState,
        seed: int, update_index: int, solver_contract_id: str,
        output_contract_id: str, max_returned_candidates: int = 4,
    ) -> Layer2OptimizationRequest:
        if int(opportunity.search_budget.get("metric_calls", 0)) != 36:
            raise SearchContractError("current GEPA packet requires 36 local metric calls")
        lane = opportunity.diagnosis.responsibility[opportunity.target_member].primary_lane
        if lane not in {"direct_flip", "near_margin", "coverage", "fallback"}:
            raise SearchContractError("unknown current GEPA primary lane")
        universe = tuple(opportunity.evaluation_plan["evidence_universe"])
        if len({row.example_id for row in universe}) != len(universe):
            raise SearchContractError("duplicate Optimize evidence identity")
        all_repair = tuple(sorted((row for row in universe if
                                  row.signals.get("legacy_group") == "repair" and
                                  (lane == "fallback" or lane in row.signals.get("legacy_tags", ()) or
                                   lane == "coverage" and "pure_coverage" in row.signals.get("legacy_tags", ()))),
                                  key=lambda row: ({"direct_flip": 0, "near_margin": 1,
                                                    "coverage": 2}.get(str(row.signals.get("lane")), 3),
                                                   hashlib.sha256(row.example_id.encode()).hexdigest(),
                                                   row.example_id)))
        mutation = opportunity.evidence.mutation_evidence
        repair = tuple(row for row in mutation if "RESPONSIBILITY" in row.roles)
        focus = tuple(row for row in mutation if "TRANSITION_FOCUS" in row.roles)
        anchor = tuple(row for row in mutation if "TRANSITION_ANCHOR" in row.roles)
        local_eval = opportunity.evidence.search_validation_evidence
        if len(local_eval) != 12 or len(repair) < 4 or len(repair) + len(focus) + len(anchor) > 36:
            raise SearchContractError("current GEPA frozen evidence capacity mismatch")
        if tuple(row.example_id for row in repair) != tuple(row.example_id for row in all_repair[:36 - len(focus) - len(anchor)]):
            raise SearchContractError("GEPA scheduled repair rows differ from responsibility universe")
        latest = history.latest_member_transition(opportunity.target_member)
        transition = None
        if latest is not None:
            if (not latest.parent_correctness or not latest.child_correctness or
                    not latest.parent_prompt_hash or not latest.child_prompt_hash):
                raise SearchContractError("latest transition lacks correctness profiles")
            if latest.child_prompt_hash != hashlib.sha256(opportunity.parent_prompt.encode("utf-8")).hexdigest():
                raise SearchContractError("latest transition child differs from current parent prompt")
            transition = CandidateTransitionAudit(
                latest.parent_prompt_hash,
                latest.child_prompt_hash,
                latest.parent_correctness, latest.child_correctness,
            )
            if tuple(row.example_id for row in focus) != transition.newly_broken_ids or tuple(row.example_id for row in anchor) != transition.newly_fixed_ids:
                raise SearchContractError("GEPA transition focus or anchor differs from history")
        budget = NativeResourceBudget(12, 36, 36, max_returned_candidates)
        responsibility = tuple(self._row(row, role="responsibility", lane=lane) for row in repair)
        focus_rows = tuple(self._row(row, role="focus", lane="global") for row in focus)
        anchor_rows = tuple(self._row(row, role="anchor", lane="global") for row in anchor)
        eval_rows = tuple(self._row(row, role="local_eval", lane="global") for row in local_eval)
        packet = ResponsibilityEvidencePacket(
            source_team_state_hash=opportunity.parent_state_id,
            target_member=opportunity.target_member,
            primary_responsibility_lane=lane,
            responsibility_value=float(opportunity.objective["raw_responsibility"]),
            responsibility_context=self.responsibility_context,
            responsibility_examples=responsibility,
            focus_examples=focus_rows,
            anchor_examples=anchor_rows,
            local_eval_examples=eval_rows,
            ordered_batch_schedule=self._schedule((
                tuple(row.packet_item_id for row in responsibility),
                tuple(row.packet_item_id for row in focus_rows),
                tuple(row.packet_item_id for row in anchor_rows)), 12),
            parent_candidate_hash=hashlib.sha256(opportunity.parent_prompt.encode("utf-8")).hexdigest(),
            lineage_parent_hash=transition.parent_candidate_hash if transition else None,
            data_universe_hash=self._universe_hash(universe),
            budget=budget,
            provenance=(
                ("builder", "Layer2EvidenceRequestBuilder"),
                ("source_split", "optimize_only"),
                ("selection_policy", LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION),
                ("responsibility_universe_count", str(len(all_repair))),
                ("responsibility_universe_ids_json", self._ids_json(all_repair)),
                ("responsibility_universe_ids_sha256", self._ids_hash(all_repair)),
                ("responsibility_scheduled_count", str(len(repair))),
                ("responsibility_scheduled_ids_json", self._ids_json(repair)),
                ("responsibility_scheduled_ids_sha256", self._ids_hash(repair)),
                ("local_eval_source", "assignment_frozen_ids"),
                ("batch_fill_policy", "cyclic_repeat_within_frozen_packet_v1"),
            ),
            latest_transition=transition,
            packet_version=LAYER2_EVIDENCE_PACKET_V4_VERSION,
            selection_policy_version=LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION,
        )
        member = opportunity.target_member
        return Layer2OptimizationRequest(
            request_id=f"seed{seed}_update{update_index}_member{member}_layer2_evidence",
            parent_decision_procedure=opportunity.parent_prompt,
            packet=packet,
            solver_contract_id=solver_contract_id,
            output_contract_id=output_contract_id,
            seed=seed * 100_000 + update_index * 10 + member,
        )


class CurrentGEPABridge:
    """Run the current GEPA search over an already frozen unified opportunity."""

    identity = "current_gepa_bridge_v1"

    def __init__(self, *, optimizer: object, history: HistoryState,
                 seed: int, solver_contract_id: str,
                 output_contract_id: str) -> None:
        self.optimizer = optimizer
        self.history = history
        self.seed = seed
        self.solver_contract_id = solver_contract_id
        self.output_contract_id = output_contract_id
        self.packet_adapter = CurrentGEPAPacketAdapter()
        self.raw_candidates: dict[str, object] = {}
        self.last_packet_hash: str | None = None

    def make_task(self, opportunity: OptimizationOpportunity, context: object) -> Layer2OptimizationRequest:
        del context
        try:
            update_index = int(opportunity.opportunity_id.rsplit(":", 2)[1])
        except (ValueError, IndexError) as exc:
            raise SearchContractError("opportunity lacks update index") from exc
        task = self.packet_adapter.build(
            opportunity, history=self.history, seed=self.seed,
            update_index=update_index,
            solver_contract_id=self.solver_contract_id,
            output_contract_id=self.output_contract_id,
        )
        self.last_packet_hash = task.packet.packet_hash
        return task

    async def optimize(self, task: Layer2OptimizationRequest) -> object:
        result = await self.optimizer.optimize_layer2(task)
        self.raw_candidates = {row.candidate_id: row for row in result.candidates}
        return result
