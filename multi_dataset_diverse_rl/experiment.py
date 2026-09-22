"""Minimal production experiment contract and orchestration engine.

The engine selects only three orthogonal dimensions: Layer-1 backend,
native/Layer-2 scope, and fixed-budget/saturation stopping. Scientific search,
responsibility, evidence, admission and provider mechanics remain in their
existing owners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Callable, Mapping, Protocol

from .local_optimizers.base import LocalOptimizerBackend
from .local_optimizers.schemas import LocalOptimizationResult
from .native_feed import Layer2OptimizationRequest, NativeOptimizationRequest
from .saturation import SaturationConfig, SaturationState
from .team_search.controller import TeamSearchController
from .team_search.schemas import TeamSearchOutcome, TeamSearchRequest


class ExperimentContractError(ValueError):
    """Typed fail-closed error for invalid production experiment contracts."""


class OptimizerBackend(str, Enum):
    GEPA = "gepa"
    MARS = "mars"


class OptimizationScope(str, Enum):
    NATIVE = "native"
    LAYER2 = "layer2"


class StoppingRegime(str, Enum):
    FIXED_BUDGET = "fixed_budget"
    SATURATION = "saturation"


@dataclass(frozen=True)
class ExperimentSpec:
    backend: OptimizerBackend
    optimization_scope: OptimizationScope
    stopping_regime: StoppingRegime
    task_identity: str
    data_identity: str
    fixed_budget_units: int = 1
    local_no_update_patience: int = 3
    team_no_update_patience: int = 2
    agent_count: int = 5
    solver_contract_id: str = "COMMON_SOLVER_CONTRACT_V1"
    output_contract_id: str = "task_output_contract_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.backend, OptimizerBackend):
            raise ExperimentContractError("backend must be an OptimizerBackend")
        if not isinstance(self.optimization_scope, OptimizationScope):
            raise ExperimentContractError("optimization_scope must be an OptimizationScope")
        if not isinstance(self.stopping_regime, StoppingRegime):
            raise ExperimentContractError("stopping_regime must be a StoppingRegime")
        for name in ("task_identity", "data_identity", "solver_contract_id", "output_contract_id"):
            if not getattr(self, name):
                raise ExperimentContractError(f"{name} is required")
        if self.agent_count != 5:
            raise ExperimentContractError("production team must contain exactly five agents")
        if self.fixed_budget_units <= 0:
            raise ExperimentContractError("fixed_budget_units must be positive")
        if self.local_no_update_patience != 3 or self.team_no_update_patience != 2:
            raise ExperimentContractError("production saturation patience must remain 3/2")

    @property
    def mode_id(self) -> str:
        return f"{self.backend.value}_{self.optimization_scope.value}".upper()

    def identity(self) -> str:
        payload = {
            "backend": self.backend.value,
            "optimization_scope": self.optimization_scope.value,
            "stopping_regime": self.stopping_regime.value,
            "task_identity": self.task_identity,
            "data_identity": self.data_identity,
            "fixed_budget_units": self.fixed_budget_units,
            "local_no_update_patience": self.local_no_update_patience,
            "team_no_update_patience": self.team_no_update_patience,
            "agent_count": self.agent_count,
            "solver_contract_id": self.solver_contract_id,
            "output_contract_id": self.output_contract_id,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class RuntimeContext:
    seed: int
    provider_profile: str
    solver_model: str
    optimizer_model: str
    evaluator_model: str
    run_identity_sha256: str
    authorization_identity: str
    cache_identity: str
    ledger_identity: str

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ExperimentContractError("seed must be non-negative")
        for name in (
            "provider_profile", "solver_model", "optimizer_model", "evaluator_model",
            "run_identity_sha256", "authorization_identity", "cache_identity", "ledger_identity",
        ):
            if not getattr(self, name):
                raise ExperimentContractError(f"runtime {name} is required")


OptimizationProblem = NativeOptimizationRequest | Layer2OptimizationRequest


@dataclass(frozen=True)
class LocalOptimizationRequest:
    target_member: int
    parent_prompt: str
    optimization_scope: str
    problem: OptimizationProblem
    seed: int
    local_seed: int
    rng_identity: str
    local_no_update_patience: int
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_member < 0 or not self.parent_prompt or not self.rng_identity:
            raise ExperimentContractError("local target, parent and RNG identity are required")
        if self.optimization_scope not in {"native", "layer2"}:
            raise ExperimentContractError("local optimization scope must be native or layer2")
        expected = NativeOptimizationRequest if self.optimization_scope == "native" else Layer2OptimizationRequest
        if not isinstance(self.problem, expected):
            raise ExperimentContractError(
                f"{self.optimization_scope} local request has the wrong problem type"
            )
        if self.problem.seed != self.local_seed:
            raise ExperimentContractError("local RNG seed does not match problem seed")
        if self.parent_prompt != self.problem.parent_decision_procedure:
            raise ExperimentContractError("local parent does not match problem parent")
        problem_target = (
            self.problem.target_member
            if isinstance(self.problem, NativeOptimizationRequest)
            else self.problem.packet.target_member
        )
        if self.target_member != problem_target:
            raise ExperimentContractError("local target does not match problem target")

    def validate_against(self, context: RuntimeContext) -> None:
        if self.seed != context.seed:
            raise ExperimentContractError("local request seed does not match RuntimeContext")

    @classmethod
    def from_problem(
        cls,
        problem: OptimizationProblem,
        *,
        scope: OptimizationScope,
        context: RuntimeContext,
        local_no_update_patience: int,
        provenance: Mapping[str, str] | None = None,
    ) -> "LocalOptimizationRequest":
        target_member = (
            problem.target_member
            if isinstance(problem, NativeOptimizationRequest)
            else problem.packet.target_member
        )
        return cls(
            target_member=target_member,
            parent_prompt=problem.parent_decision_procedure,
            optimization_scope=scope.value,
            problem=problem,
            seed=context.seed,
            local_seed=problem.seed,
            rng_identity=(
                f"run-seed:{context.seed}:local-seed:{problem.seed}:"
                f"request:{problem.request_id}"
            ),
            local_no_update_patience=local_no_update_patience,
            provenance=dict(provenance or {}),
        )


@dataclass(frozen=True)
class Layer2Opportunity:
    request: TeamSearchRequest
    completes_team_epoch: bool = True


@dataclass(frozen=True)
class ExperimentInputs:
    initial_state_hash: str
    native_problem: NativeOptimizationRequest | None = None
    layer2_opportunities: tuple[Layer2Opportunity, ...] = ()

    def __post_init__(self) -> None:
        if not self.initial_state_hash:
            raise ExperimentContractError("initial state hash is required")


class Layer2ControllerFactory(Protocol):
    def __call__(self, backend) -> TeamSearchController:
        ...


@dataclass(frozen=True)
class ExperimentServices:
    backend: LocalOptimizerBackend
    layer2_controller_factory: Layer2ControllerFactory | None = None
    team_state_hash_reader: Callable[[], str] | None = None


@dataclass(frozen=True)
class EngineEvent:
    index: int
    kind: str
    request_identity: str
    candidate_ids: tuple[str, ...]
    committed_candidate_id: str | None
    state_hash: str
    stop_reason: str | None
    telemetry: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    mode_id: str
    stopping_regime: str
    initial_state_hash: str
    final_state_hash: str
    stop_reason: str
    events: tuple[EngineEvent, ...]
    local_result: LocalOptimizationResult | None
    team_outcomes: tuple[TeamSearchOutcome, ...]


class _ContextBoundLayer2Backend:
    """Adapts the production backend to the backend-neutral team controller."""

    optimizer_backend: str
    optimization_mode = "layer2"
    backend_fidelity: str

    def __init__(
        self,
        backend: LocalOptimizerBackend,
        context: RuntimeContext,
        spec: ExperimentSpec,
    ) -> None:
        self.backend = backend
        self.context = context
        self.spec = spec
        self.optimizer_backend = backend.name
        self.backend_fidelity = backend.fidelity

    async def optimize(self, problem: OptimizationProblem) -> LocalOptimizationResult:
        request = LocalOptimizationRequest.from_problem(
            problem,
            scope=OptimizationScope.LAYER2,
            context=self.context,
            local_no_update_patience=self.spec.local_no_update_patience,
            provenance={"engine": "production_v1"},
        )
        return await self.backend.optimize(request, self.context)


class ExperimentEngine:
    """The only production native/Layer-2 orchestration dispatcher."""

    async def run(
        self,
        spec: ExperimentSpec,
        runtime: RuntimeContext,
        inputs: ExperimentInputs,
        services: ExperimentServices,
    ) -> ExperimentResult:
        if services.backend.name != spec.backend.value:
            raise ExperimentContractError("configured backend does not match ExperimentSpec")
        if spec.optimization_scope is OptimizationScope.NATIVE:
            return await self._run_native(spec, runtime, inputs, services)
        return await self._run_layer2(spec, runtime, inputs, services)

    async def _run_native(self, spec, runtime, inputs, services) -> ExperimentResult:
        if inputs.native_problem is None or inputs.layer2_opportunities:
            raise ExperimentContractError("native execution requires exactly one native problem")
        request = LocalOptimizationRequest.from_problem(
            inputs.native_problem,
            scope=OptimizationScope.NATIVE,
            context=runtime,
            local_no_update_patience=spec.local_no_update_patience,
            provenance={"engine": "production_v1"},
        )
        result = await services.backend.optimize(request, runtime)
        stop_reason = (
            "SATURATION_REACHED"
            if spec.stopping_regime is StoppingRegime.SATURATION
            else "SCIENTIFIC_BUDGET_REACHED"
        )
        final_hash = (
            result.candidates[0].candidate_id if result.candidates else inputs.initial_state_hash
        )
        event = EngineEvent(
            index=1,
            kind="LOCAL_OPTIMIZATION",
            request_identity=inputs.native_problem.identity(),
            candidate_ids=tuple(row.candidate_id for row in result.candidates),
            committed_candidate_id=None,
            state_hash=final_hash,
            stop_reason=stop_reason,
            telemetry={"backend_termination_reason": result.termination_reason},
        )
        return ExperimentResult(
            spec.mode_id, spec.stopping_regime.value, inputs.initial_state_hash,
            final_hash, stop_reason, (event,), result, (),
        )

    async def _run_layer2(self, spec, runtime, inputs, services) -> ExperimentResult:
        if inputs.native_problem is not None or not inputs.layer2_opportunities:
            raise ExperimentContractError("Layer2 execution requires Layer2 opportunities only")
        if services.layer2_controller_factory is None or services.team_state_hash_reader is None:
            raise ExperimentContractError("Layer2 controller and state reader are required")
        bridge = _ContextBoundLayer2Backend(services.backend, runtime, spec)
        controller = services.layer2_controller_factory(bridge)
        state = SaturationState(
            SaturationConfig(
                enabled=spec.stopping_regime is StoppingRegime.SATURATION,
                scientific_budget_enabled=spec.stopping_regime is StoppingRegime.FIXED_BUDGET,
                local_no_update_patience=(
                    spec.local_no_update_patience
                    if spec.stopping_regime is StoppingRegime.SATURATION else None
                ),
                team_no_update_patience=(
                    spec.team_no_update_patience
                    if spec.stopping_regime is StoppingRegime.SATURATION else None
                ),
            ),
            backend=spec.backend.value,
            mode=spec.mode_id.lower(),
        )
        outcomes: list[TeamSearchOutcome] = []
        events: list[EngineEvent] = []
        current_hash = inputs.initial_state_hash
        completed_epochs = 0
        for index, opportunity in enumerate(inputs.layer2_opportunities, start=1):
            if opportunity.request.seed != runtime.seed:
                raise ExperimentContractError("Layer2 opportunity seed does not match RuntimeContext")
            outcome = await controller.run_opportunity(opportunity.request)
            outcomes.append(outcome)
            current_hash = services.team_state_hash_reader()
            stop = None
            if opportunity.completes_team_epoch:
                completed_epochs += 1
                stop = state.observe_team_epoch(
                    start_state_hash=(
                        inputs.initial_state_hash if completed_epochs == 1
                        else events[-1].state_hash
                    ),
                    end_state_hash=current_hash,
                    team_commit=outcome.committed_candidate_id is not None,
                    accepted_local_update=bool(outcome.candidates),
                    scientific_budget_reached=(
                        spec.stopping_regime is StoppingRegime.FIXED_BUDGET
                        and completed_epochs >= spec.fixed_budget_units
                    ),
                )
            events.append(EngineEvent(
                index=index,
                kind="TEAM_OPPORTUNITY",
                request_identity=(
                    f"seed:{opportunity.request.seed}:update:{opportunity.request.update_index}:"
                    f"team:{opportunity.request.team_state_hash}"
                ),
                candidate_ids=tuple(row.local_candidate.candidate_id for row in outcome.candidates),
                committed_candidate_id=outcome.committed_candidate_id,
                state_hash=current_hash,
                stop_reason=stop.value if stop else None,
                telemetry={"funnel": dict(outcome.funnel), "audit": dict(outcome.audit_metadata)},
            ))
            if stop is not None:
                break
        stop_reason = state.stop_reason.value if state.stop_reason else "INPUT_EXHAUSTED"
        return ExperimentResult(
            spec.mode_id, spec.stopping_regime.value, inputs.initial_state_hash,
            current_hash, stop_reason, tuple(events), None, tuple(outcomes),
        )


async def run_experiment(
    spec: ExperimentSpec,
    runtime: RuntimeContext,
    inputs: ExperimentInputs,
    services: ExperimentServices,
) -> ExperimentResult:
    """Public production API for every new experiment."""

    return await ExperimentEngine().run(spec, runtime, inputs, services)


def experiment_spec_from_mapping(payload: Mapping[str, object]) -> ExperimentSpec:
    """Parse the scientific section of a production manifest."""

    try:
        return ExperimentSpec(
            backend=OptimizerBackend(str(payload["backend"])),
            optimization_scope=OptimizationScope(str(payload["optimization_scope"])),
            stopping_regime=StoppingRegime(str(payload["stopping_regime"])),
            task_identity=str(payload["task_identity"]),
            data_identity=str(payload["data_identity"]),
            fixed_budget_units=int(payload.get("fixed_budget_units", 1)),
            local_no_update_patience=int(payload.get("local_no_update_patience", 3)),
            team_no_update_patience=int(payload.get("team_no_update_patience", 2)),
            agent_count=int(payload.get("agent_count", 5)),
            solver_contract_id=str(payload.get("solver_contract_id", "COMMON_SOLVER_CONTRACT_V1")),
            output_contract_id=str(payload.get("output_contract_id", "task_output_contract_v1")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentContractError(f"invalid scientific manifest: {exc}") from exc


def runtime_context_from_mapping(payload: Mapping[str, object]) -> RuntimeContext:
    """Parse execution identity without exposing provider secrets to science code."""

    try:
        return RuntimeContext(
            seed=int(payload["seed"]),
            provider_profile=str(payload["provider_profile"]),
            solver_model=str(payload["solver_model"]),
            optimizer_model=str(payload["optimizer_model"]),
            evaluator_model=str(payload["evaluator_model"]),
            run_identity_sha256=str(payload["run_identity_sha256"]),
            authorization_identity=str(payload["authorization_identity"]),
            cache_identity=str(payload["cache_identity"]),
            ledger_identity=str(payload["ledger_identity"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentContractError(f"invalid runtime manifest: {exc}") from exc
