"""Native-data-feed wrapper around the frozen official GEPA engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Sequence

from ..native_feed import (
    Layer2OptimizationRequest,
    NativeOptimizationRequest,
    PacketEvidenceExample,
)
from ..versions import (
    GEPA_LAYER2_EVIDENCE_BACKEND_VERSION,
    GEPA_LAYER2_RESPONSIBILITY_OVERLAY_VERSION,
    GEPA_NATIVE_FEED_VERSION,
)
from .gepa_adapter import GEPAAdapter
from .gepa_optimizer import GEPALocalPromptOptimizer
from .schemas import (
    LocalEvidenceExample,
    LocalOptimizationResult,
    LocalOptimizationTask,
    LocalOptimizerBudget,
    OpaqueOptimizerState,
)


@dataclass(frozen=True)
class GEPANativeSplitConfig:
    optimizer_train_count: int
    optimizer_val_count: int
    split_seed: int = 20260918
    identity: str = "gepa_optimize_only_train75_pareto25_v1"

    def __post_init__(self) -> None:
        if self.optimizer_train_count <= 0 or self.optimizer_val_count <= 0:
            raise ValueError("GEPA native train and optimizer-val sets must be non-empty")


@dataclass(frozen=True)
class GEPANativeDataset:
    trainset: tuple[LocalEvidenceExample, ...]
    valset: tuple[LocalEvidenceExample, ...]
    universe_identity: str
    split_identity: str


class GEPANativeDataBuilder:
    """Backend-owned deterministic split and feed construction."""

    def __init__(
        self,
        *,
        optimize_examples: Sequence[LocalEvidenceExample],
        optimize_universe_id: str,
        config: GEPANativeSplitConfig,
    ) -> None:
        self.examples = tuple(optimize_examples)
        self.optimize_universe_id = optimize_universe_id
        self.config = config
        if len(self.examples) != config.optimizer_train_count + config.optimizer_val_count:
            raise ValueError("GEPA native split counts do not exhaust Optimize universe")
        ids = [row.example_id for row in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError("Optimize universe example ids must be unique")

    def build(self, request: NativeOptimizationRequest) -> GEPANativeDataset:
        if request.optimize_universe_id != self.optimize_universe_id:
            raise ValueError("GEPA native Optimize-universe identity mismatch")
        ranked = sorted(
            self.examples,
            key=lambda row: hashlib.sha256(
                f"{self.config.split_seed}:{row.example_id}".encode("utf-8")
            ).hexdigest(),
        )
        train = tuple(ranked[: self.config.optimizer_train_count])
        val = tuple(ranked[self.config.optimizer_train_count :])
        return GEPANativeDataset(train, val, self.optimize_universe_id, self.config.identity)

    def identity(self) -> str:
        payload = {
            "backend": GEPA_NATIVE_FEED_VERSION,
            "universe": self.optimize_universe_id,
            "config": self.config.__dict__,
            "ordered_example_ids": [row.example_id for row in self.examples],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class NativeFeedGEPAAdapter(GEPAAdapter):
    """Add responsibility metadata to native GEPA trajectory evidence."""

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        dataset = super().make_reflective_dataset(
            candidate, eval_batch, components_to_update
        )
        if not self.optimization_context:
            return dataset
        return {
            component: [
                {**record, "Layer2 Responsibility Overlay": self.optimization_context}
                for record in records
            ]
            for component, records in dataset.items()
        }


class GEPANativeFeedOptimizer:
    """Same official GEPA search with optional Layer-2 reflection overlay."""

    backend_name = "gepa_native_feed"
    backend_version = GEPA_NATIVE_FEED_VERSION

    def __init__(
        self,
        *,
        engine: GEPALocalPromptOptimizer,
        data_builder: GEPANativeDataBuilder,
        layer2_overlay_enabled: bool,
    ) -> None:
        self.engine = engine
        self.data_builder = data_builder
        self.layer2_overlay_enabled = layer2_overlay_enabled

    def task_for(self, request: NativeOptimizationRequest) -> LocalOptimizationTask:
        dataset = self.data_builder.build(request)
        overlay = ""
        if self.layer2_overlay_enabled:
            overlay = (
                f"overlay_version={GEPA_LAYER2_RESPONSIBILITY_OVERLAY_VERSION}\n"
                f"{request.responsibility.canonical_overlay()}"
            )
        if request.budget.max_returned_candidates != self.engine.config.k_local_return:
            raise ValueError("native request/GEPA return budget mismatch")
        return LocalOptimizationTask(
            task_id=request.request_id,
            parent_prompt=request.parent_decision_procedure,
            search_examples=dataset.trainset,
            local_validation_examples=dataset.valset,
            optimization_context=overlay,
            solver_contract_id=request.solver_contract_id,
            output_contract_id=request.output_contract_id,
            seed=request.seed,
            budget=LocalOptimizerBudget(
                max_metric_calls=request.budget.metric_call_limit,
                reflection_minibatch_size=3,
                max_returned_candidates=request.budget.max_returned_candidates,
            ),
        )

    async def optimize_native(
        self, request: NativeOptimizationRequest
    ) -> LocalOptimizationResult:
        # The opaque result crosses the common optimizer boundary unchanged;
        # Layer 2 cannot inspect native split state to steer GEPA's search.
        return await self.engine.optimize(self.task_for(request))

    def parity_identity(self, request: NativeOptimizationRequest) -> str:
        payload = {
            "data_builder": self.data_builder.identity(),
            "engine_protocol": self.engine.config.identity(),
            "request_budget": request.budget.__dict__,
            "solver_contract": request.solver_contract_id,
            "output_contract": request.output_contract_id,
            "candidate_component": "decision_procedure",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class Layer2EvidenceScheduleExhausted(RuntimeError):
    pass


class Layer2FrozenBatchSampler:
    """Deliver the exact precomputed Layer-2 schedule through GEPA's public API."""

    def __init__(
        self,
        *,
        ordered_batch_schedule: tuple[tuple[str, ...], ...],
        ordered_example_ids: tuple[str, ...],
    ) -> None:
        self.schedule = ordered_batch_schedule
        self.ordered_example_ids = ordered_example_ids
        self.allowed_ids = frozenset(ordered_example_ids)
        self.index_by_id = {
            example_id: index for index, example_id in enumerate(ordered_example_ids)
        }
        self.cursor = 0
        self.delivery_calls = 0
        self.backend_example_selection_calls = 0
        self.delivered_ids: list[tuple[str, ...]] = []

    def next_minibatch_ids(self, loader, state):
        del state
        loader_ids = tuple(loader.all_ids())
        if loader_ids != tuple(range(len(self.ordered_example_ids))):
            raise RuntimeError("GEPA treatment loader differs from Layer-2 packet")
        if self.cursor >= len(self.schedule):
            raise Layer2EvidenceScheduleExhausted(
                "Layer-2 evidence schedule exhausted; native fallback is forbidden"
            )
        batch = self.schedule[self.cursor]
        self.cursor += 1
        self.delivery_calls += 1
        if not set(batch).issubset(self.allowed_ids):
            raise RuntimeError("scheduled GEPA batch contains an unlisted example")
        self.delivered_ids.append(batch)
        return [self.index_by_id[example_id] for example_id in batch]


def _local(row: PacketEvidenceExample) -> LocalEvidenceExample:
    return LocalEvidenceExample(
        example_id=row.packet_item_id,
        input_payload=row.input_payload,
        gold=row.gold,
        parent_output=row.parent_output,
        textual_feedback=row.textual_feedback,
        tags=(row.lane, row.responsibility_role),
    )


class GEPALayer2EvidenceOptimizer:
    """Official GEPA search core over a fully Layer-2-owned curriculum."""

    backend_name = "gepa_search_core_layer2_evidence"
    backend_version = GEPA_LAYER2_EVIDENCE_BACKEND_VERSION

    def __init__(self, *, engine: GEPALocalPromptOptimizer) -> None:
        self.engine = engine
        self.last_sampler: Layer2FrozenBatchSampler | None = None

    def task_for(self, request: Layer2OptimizationRequest) -> LocalOptimizationTask:
        packet = request.packet
        if packet.budget.max_returned_candidates != self.engine.config.k_local_return:
            raise ValueError("Layer-2 request/GEPA return budget mismatch")
        search = tuple(
            _local(row)
            for row in (
                *packet.responsibility_examples,
                *packet.focus_examples,
                *packet.anchor_examples,
            )
        )
        local_eval = tuple(_local(row) for row in packet.local_eval_examples)
        context = json.dumps(
            {
                "packet_version": packet.packet_version,
                "packet_hash": packet.packet_hash,
                "primary_responsibility_lane": packet.primary_responsibility_lane,
                "responsibility_value": packet.responsibility_value,
                "responsibility_context": packet.responsibility_context,
                "TEAM RESPONSIBILITY EVIDENCE": [
                    {"example_id": row.example_id, "role": "responsibility", "lane": row.lane}
                    for row in packet.responsibility_examples
                ],
                "RECENT REGRESSION / FOCUS EVIDENCE": [
                    {"example_id": row.example_id, "role": "focus", "lane": row.lane}
                    for row in packet.focus_examples
                ],
                "RECENT GAIN / ANCHOR EVIDENCE": [
                    {"example_id": row.example_id, "role": "anchor", "lane": row.lane}
                    for row in packet.anchor_examples
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return LocalOptimizationTask(
            task_id=request.request_id,
            parent_prompt=request.parent_decision_procedure,
            search_examples=search,
            local_validation_examples=local_eval,
            optimization_context=context,
            solver_contract_id=request.solver_contract_id,
            output_contract_id=request.output_contract_id,
            seed=request.seed,
            budget=LocalOptimizerBudget(
                max_metric_calls=packet.budget.metric_call_limit,
                reflection_minibatch_size=3,
                max_returned_candidates=packet.budget.max_returned_candidates,
            ),
        )

    async def optimize_layer2(
        self, request: Layer2OptimizationRequest
    ) -> LocalOptimizationResult:
        packet = request.packet
        before = packet.packet_hash
        ordered_ids = tuple(
            row.packet_item_id
            for row in (
                *packet.responsibility_examples,
                *packet.focus_examples,
                *packet.anchor_examples,
            )
        )
        sampler = Layer2FrozenBatchSampler(
            ordered_batch_schedule=packet.ordered_batch_schedule,
            ordered_example_ids=ordered_ids,
        )
        self.last_sampler = sampler
        result = await self.engine.optimize_with_batch_sampler(
            self.task_for(request), sampler
        )
        if packet.packet_hash != before:
            raise RuntimeError("GEPA mutated the immutable Layer-2 packet")
        provenance = request.candidate_provenance(
            backend=self.backend_name, backend_version=self.backend_version
        )
        candidates = tuple(
            replace(
                candidate,
                backend_metadata={**candidate.backend_metadata, **provenance},
            )
            for candidate in result.candidates
        )
        prior_payload = (
            dict(result.optimizer_state.payload)
            if result.optimizer_state is not None
            else {}
        )
        telemetry = dict(prior_payload.get("telemetry", {}))
        telemetry.update(
            {
                "responsibility_packet_hash": packet.packet_hash,
                "scheduled_batch_count": len(packet.ordered_batch_schedule),
                "batch_delivery_calls": sampler.delivery_calls,
                "delivered_batch_ids": [list(batch) for batch in sampler.delivered_ids],
                "backend_example_selection_calls": 0,
                "native_sampler_called": False,
                "official_gepa_search_core_modified": False,
                "responsibility_count": len(packet.responsibility_examples),
                "focus_count": len(packet.focus_examples),
                "anchor_count": len(packet.anchor_examples),
                "local_eval_count": len(packet.local_eval_examples),
                "role_intersection_counts": dict(packet.role_intersection_counts),
                "transition_effect_hash": (
                    packet.latest_transition.transition_effect_hash
                    if packet.latest_transition is not None else None
                ),
            }
        )
        state = OpaqueOptimizerState(
            self.backend_name,
            self.backend_version,
            {**prior_payload, "telemetry": telemetry},
        )
        return replace(
            result,
            candidates=candidates,
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            optimizer_state=state,
        )
