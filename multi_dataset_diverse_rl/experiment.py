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
import itertools
import json
from typing import Callable, Mapping, Protocol

from . import versions
from .local_optimizers.base import LocalOptimizerBackend
from .local_optimizers.schemas import LocalOptimizationResult
from .native_feed import Layer2OptimizationRequest, NativeOptimizationRequest
from .saturation import SaturationConfig, SaturationEmergency, SaturationState, StopReason, TeamEpochTracker
from .team_search.controller import TeamSearchController
from .team_search.schemas import TeamSearchOutcome, TeamSearchRequest


class ExperimentContractError(ValueError):
    """Typed fail-closed error for invalid production experiment contracts."""


class ExperimentEarlyStop(RuntimeError):
    """Stop before a new dynamic opportunity when no valid local task exists."""

    def __init__(self, reason: str) -> None:
        if not reason:
            raise ValueError("early-stop reason is required")
        self.reason = reason
        super().__init__(reason)


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
    layer2_protocol_version: str | None = None
    emergency_max_provider_calls: int = 100_000
    emergency_max_optimizer_steps: int = 100_000
    emergency_max_team_epochs: int = 10_000
    emergency_max_wall_seconds: int | None = 86_400

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
        if self.layer2_protocol_version is not None and self.optimization_scope is not OptimizationScope.LAYER2:
            raise ExperimentContractError("Layer2 protocol identity requires Layer2 scope")
        if self.layer2_protocol_version not in {
            None, versions.LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
        }:
            raise ExperimentContractError("unsupported opt-in Layer2 protocol identity")
        for value in (self.emergency_max_provider_calls, self.emergency_max_optimizer_steps,
                      self.emergency_max_team_epochs):
            if value <= 0:
                raise ExperimentContractError("emergency ceilings must be positive")

    @property
    def mode_id(self) -> str:
        return f"{self.backend.value}_{self.optimization_scope.value}".upper()

    @property
    def method_identity(self) -> str:
        """Opt-in method identity, distinct from the historical v15 runtime."""

        suffix = "_V4" if self.layer2_protocol_version == versions.LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION else ""
        return f"{versions.UNIFIED_EXPERIMENT_ENGINE_VERSION}:{self.mode_id}{suffix}"

    @property
    def is_v4_layer2(self) -> bool:
        return self.layer2_protocol_version == versions.LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION

    def identity(self) -> str:
        payload = {
            "method_identity": self.method_identity,
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
            "layer2_protocol_version": self.layer2_protocol_version,
            "emergency_max_provider_calls": self.emergency_max_provider_calls,
            "emergency_max_optimizer_steps": self.emergency_max_optimizer_steps,
            "emergency_max_team_epochs": self.emergency_max_team_epochs,
            "emergency_max_wall_seconds": self.emergency_max_wall_seconds,
            "backend_feed_version": {
                (OptimizerBackend.GEPA, OptimizationScope.NATIVE): versions.GEPA_NATIVE_FEED_VERSION,
                (OptimizerBackend.GEPA, OptimizationScope.LAYER2): versions.GEPA_LAYER2_EVIDENCE_BACKEND_VERSION,
                (OptimizerBackend.MARS, OptimizationScope.NATIVE): versions.MARS_NATIVE_FEED_VERSION,
                (OptimizerBackend.MARS, OptimizationScope.LAYER2): versions.MARS_LAYER2_EVIDENCE_BACKEND_VERSION,
            }[(self.backend, self.optimization_scope)],
            "layer2_packet_version": (
                (versions.LAYER2_EVIDENCE_PACKET_V4_VERSION if self.is_v4_layer2
                 else versions.LAYER2_EVIDENCE_PACKET_VERSION)
                if self.optimization_scope is OptimizationScope.LAYER2 else None
            ),
            "layer2_policy_versions": (
                {
                    "responsibility": versions.PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
                    "realizability": versions.PERSISTENT_REALIZABILITY_SEMANTICS_VERSION,
                    "team_minibatch": versions.TEAM_MINIBATCH_CONTRACT_VERSION,
                    "team_admission": versions.CANDIDATE_SELECTION_VERSION,
                    **({
                        "responsibility_source": versions.LAYER2_RESPONSIBILITY_SOURCE_VERSION,
                        "evidence_selection": versions.LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION,
                        "target_feasibility": versions.LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION,
                        "team_search": versions.LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
                        "team_epoch": versions.TEAM_EPOCH_SEMANTICS_V4_VERSION,
                    } if self.is_v4_layer2 else {}),
                }
                if self.optimization_scope is OptimizationScope.LAYER2 else None
            ),
            "stopping_contract_version": versions.SATURATION_STOPPING_CONTRACT_VERSION,
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
    eligible_member_ids: tuple[int, ...] | None = None


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
    technical_local_canary_only: bool = False
    layer2_opportunity_factory: Callable[[int, str], Layer2Opportunity] | None = None
    layer2_outcome_observer: Callable[[int, TeamSearchOutcome], str | None] | None = None
    durable_usage_reader: Callable[[], Mapping[str, int]] | None = None


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
        if services.durable_usage_reader is not None:
            usage = services.durable_usage_reader()
            if int(usage["provider_attempts"]) >= spec.emergency_max_provider_calls:
                stop_reason = StopReason.EMERGENCY_PROVIDER_CALL_CEILING.value
                event = EngineEvent(
                    index=1, kind="LOCAL_OPTIMIZATION",
                    request_identity=inputs.native_problem.identity(),
                    candidate_ids=(), committed_candidate_id=None,
                    state_hash=inputs.initial_state_hash, stop_reason=stop_reason,
                    telemetry={"canonical_stop_reason": stop_reason,
                               "backend_termination_reason": "pre_native_durable_ceiling"},
                )
                return ExperimentResult(
                    spec.mode_id, spec.stopping_regime.value,
                    inputs.initial_state_hash, inputs.initial_state_hash,
                    stop_reason, (event,), None, (),
                )
        try:
            result = await services.backend.optimize(request, runtime)
        except SaturationEmergency as exc:
            stop_reason = exc.reason.value
            event = EngineEvent(
                index=1, kind="LOCAL_OPTIMIZATION",
                request_identity=inputs.native_problem.identity(),
                candidate_ids=(), committed_candidate_id=None,
                state_hash=inputs.initial_state_hash, stop_reason=stop_reason,
                telemetry={"canonical_stop_reason": stop_reason,
                           "backend_termination_reason": stop_reason},
            )
            return ExperimentResult(
                spec.mode_id, spec.stopping_regime.value,
                inputs.initial_state_hash, inputs.initial_state_hash,
                stop_reason, (event,), None, (),
            )
        except RuntimeError as exc:
            if str(exc) not in {"transport_attempt_emergency_ceiling",
                                "successful_provider_emergency_ceiling"}:
                raise
            stop_reason = StopReason.EMERGENCY_PROVIDER_CALL_CEILING.value
            event = EngineEvent(
                index=1, kind="LOCAL_OPTIMIZATION",
                request_identity=inputs.native_problem.identity(),
                candidate_ids=(), committed_candidate_id=None,
                state_hash=inputs.initial_state_hash, stop_reason=stop_reason,
                telemetry={"canonical_stop_reason": stop_reason,
                           "backend_termination_reason": str(exc)},
            )
            return ExperimentResult(
                spec.mode_id, spec.stopping_regime.value,
                inputs.initial_state_hash, inputs.initial_state_hash,
                stop_reason, (event,), None, (),
            )
        saturation = (
            result.optimizer_state.payload.get("saturation")
            if result.optimizer_state is not None else None
        )
        if spec.stopping_regime is StoppingRegime.SATURATION:
            backend_stop = result.termination_reason
            if backend_stop == StopReason.SATURATION_REACHED.value and isinstance(saturation, Mapping) and (
                saturation.get("stop_reason") == backend_stop
                and int(saturation.get("local_no_update_counter", -1)) >= spec.local_no_update_patience
            ):
                stop_reason = backend_stop
            elif backend_stop in {reason.value for reason in StopReason if reason.name.startswith("EMERGENCY_")}:
                stop_reason = backend_stop
            else:
                stop_reason = StopReason.OPERATIONAL_ABORT.value
        else:
            stop_reason = "SCIENTIFIC_BUDGET_REACHED"
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
            telemetry={"backend_termination_reason": result.termination_reason,
                       "backend_saturation": saturation,
                       "canonical_stop_reason": stop_reason},
        )
        return ExperimentResult(
            spec.mode_id, spec.stopping_regime.value, inputs.initial_state_hash,
            final_hash, stop_reason, (event,), result, (),
        )

    async def _run_layer2(self, spec, runtime, inputs, services) -> ExperimentResult:
        dynamic = services.layer2_opportunity_factory is not None
        if inputs.native_problem is not None or (not dynamic and not inputs.layer2_opportunities):
            raise ExperimentContractError("Layer2 execution requires Layer2 opportunities only")
        if dynamic and inputs.layer2_opportunities:
            raise ExperimentContractError("dynamic and frozen Layer2 opportunities cannot be mixed")
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
                emergency_max_provider_calls=spec.emergency_max_provider_calls,
                emergency_max_optimizer_steps=spec.emergency_max_optimizer_steps,
                emergency_max_team_epochs=spec.emergency_max_team_epochs,
                emergency_max_wall_seconds=spec.emergency_max_wall_seconds,
            ),
            backend=spec.backend.value,
            mode=spec.mode_id.lower(),
        )
        outcomes: list[TeamSearchOutcome] = []
        events: list[EngineEvent] = []
        current_hash = inputs.initial_state_hash
        completed_epochs = 0
        opportunity_source = (
            itertools.count(1) if dynamic and spec.stopping_regime is StoppingRegime.SATURATION
            else range(1, spec.fixed_budget_units + 1) if dynamic
            else range(1, len(inputs.layer2_opportunities) + 1)
        )
        external_stop: str | None = None
        epoch: TeamEpochTracker | None = None
        epoch_parent_hash: str | None = None
        v4_saturation = spec.is_v4_layer2 and spec.stopping_regime is StoppingRegime.SATURATION
        for index in opportunity_source:
            if services.durable_usage_reader is not None:
                state.sync_durable_provider_usage(services.durable_usage_reader())
            emergency = state.check_emergency()
            if emergency is not None:
                external_stop = emergency.value
                break
            if dynamic:
                try:
                    opportunity = services.layer2_opportunity_factory(index, current_hash)
                except ExperimentEarlyStop as exc:
                    external_stop = exc.reason
                    break
            else:
                opportunity = inputs.layer2_opportunities[index - 1]
            if opportunity.request.seed != runtime.seed:
                raise ExperimentContractError("Layer2 opportunity seed does not match RuntimeContext")
            if dynamic and (
                opportunity.request.team_state_hash != current_hash
                or opportunity.request.update_index != index - 1
            ):
                raise ExperimentContractError("dynamic Layer2 parent/update identity mismatch")
            if v4_saturation:
                eligible = opportunity.eligible_member_ids
                if eligible is None or not eligible or len(set(eligible)) != len(eligible):
                    raise ExperimentContractError("V4 saturation requires a nonempty frozen feasible eligible set")
                if epoch is None:
                    epoch = TeamEpochTracker(eligible)
                    epoch_parent_hash = current_hash
                elif eligible != epoch.eligible_member_ids or current_hash != epoch_parent_hash:
                    raise ExperimentContractError("V4 epoch eligibility/parent changed without commit")
            if services.technical_local_canary_only:
                if (spec.backend is not OptimizerBackend.GEPA
                    or spec.stopping_regime is not StoppingRegime.FIXED_BUDGET
                    or spec.fixed_budget_units != 1 or index != 1):
                    raise ExperimentContractError("local canary requires one fixed-budget GEPA opportunity")
                outcome = await controller.run_local_empirical_canary(opportunity.request)
            else:
                try:
                    outcome = await controller.run_opportunity(opportunity.request)
                except SaturationEmergency as exc:
                    if services.durable_usage_reader is not None:
                        state.sync_durable_provider_usage(services.durable_usage_reader())
                    external_stop = exc.reason.value
                    break
                except RuntimeError as exc:
                    if str(exc) not in {"transport_attempt_emergency_ceiling",
                                        "successful_provider_emergency_ceiling"}:
                        raise
                    if services.durable_usage_reader is not None:
                        state.sync_durable_provider_usage(services.durable_usage_reader())
                    external_stop = StopReason.EMERGENCY_PROVIDER_CALL_CEILING.value
                    break
            outcomes.append(outcome)
            if services.durable_usage_reader is not None:
                state.sync_durable_provider_usage(services.durable_usage_reader())
            state.optimizer_steps += sum(
                int(branch.get("local_saturation", {}).get("optimizer_steps", 0))
                for branch in outcome.audit_metadata.get("branch_metadata", ())
                if isinstance(branch, Mapping)
            )
            current_hash = services.team_state_hash_reader()
            committed = outcome.committed_candidate_id is not None
            if v4_saturation and ((not committed and current_hash != epoch_parent_hash)
                                  or (committed and current_hash == epoch_parent_hash)):
                raise ExperimentContractError("V4 parent state changed without matching atomic commit")
            if services.layer2_outcome_observer is not None:
                external_stop = services.layer2_outcome_observer(index, outcome)
            stop = None
            coverage_complete: bool | None = None
            if v4_saturation:
                selected = tuple(outcome.audit_metadata.get("selected_target_ids", ()))
                if not selected or epoch is None or epoch_parent_hash is None:
                    raise ExperimentContractError("V4 opportunity lacks selected target/epoch")
                coverage_complete = epoch.observe_opportunity(
                    selected_member_ids=selected,
                    local_accepted_update=bool(outcome.candidates),
                    team_commit=committed,
                )
                if committed:
                    stop = state.observe_team_commit(
                        start_state_hash=epoch_parent_hash,
                        end_state_hash=current_hash,
                        coverage_complete=coverage_complete,
                    )
                    epoch = None
                    epoch_parent_hash = None
                elif coverage_complete:
                    completed_epochs += 1
                    stop = state.observe_team_epoch(
                        start_state_hash=epoch_parent_hash,
                        end_state_hash=current_hash,
                        team_commit=False,
                        accepted_local_update=epoch.any_local_update,
                    )
                    epoch = None
                    epoch_parent_hash = None
            elif opportunity.completes_team_epoch:
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
            if stop is None and state.stop_reason is None:
                stop = state.check_emergency()
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
                stop_reason=external_stop or (stop.value if stop else None),
                telemetry={"funnel": dict(outcome.funnel), "audit": dict(outcome.audit_metadata),
                           "coverage_complete": coverage_complete,
                           "epoch_end_reason": ("TEAM_COMMIT" if committed else
                                                "COVERAGE_COMPLETE_NO_COMMIT" if coverage_complete
                                                else None),
                           "saturation": state.telemetry()},
            ))
            if stop is not None or external_stop is not None:
                break
        stop_reason = external_stop or (state.stop_reason.value if state.stop_reason else "INPUT_EXHAUSTED")
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
            layer2_protocol_version=(str(payload["layer2_protocol_version"])
                                     if payload.get("layer2_protocol_version") is not None else None),
            emergency_max_provider_calls=int(payload.get("emergency_max_provider_calls", 100_000)),
            emergency_max_optimizer_steps=int(payload.get("emergency_max_optimizer_steps", 100_000)),
            emergency_max_team_epochs=int(payload.get("emergency_max_team_epochs", 10_000)),
            emergency_max_wall_seconds=(int(payload.get("emergency_max_wall_seconds", 86_400))
                                        if payload.get("emergency_max_wall_seconds", 86_400) is not None else None),
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
