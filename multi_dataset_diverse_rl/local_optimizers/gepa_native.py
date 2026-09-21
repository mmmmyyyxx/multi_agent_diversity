"""Native-data-feed wrapper around the frozen official GEPA engine."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

from ..native_feed import NativeOptimizationRequest
from ..versions import (
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
