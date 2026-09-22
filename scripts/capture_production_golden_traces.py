"""Capture deterministic pre-refactor semantic traces without provider access."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.local_optimizers.backend_registry import (  # noqa: E402
    BackendRuntimeConfig,
    Layer1BackendRegistry,
)
from multi_dataset_diverse_rl.local_optimizers.schemas import (  # noqa: E402
    LocalOptimizationResult,
    LocalPromptCandidate,
)
from multi_dataset_diverse_rl.native_feed import (  # noqa: E402
    Layer2OptimizationRequest,
    NativeOptimizationRequest,
)
from multi_dataset_diverse_rl.saturation import (  # noqa: E402
    OptimizationUnitType,
    SaturationConfig,
    SaturationState,
)
from multi_dataset_diverse_rl.team_search.candidate_evaluator import (  # noqa: E402
    EvaluationCost,
)
from multi_dataset_diverse_rl.team_search.controller import TeamSearchController  # noqa: E402
from multi_dataset_diverse_rl.team_search.schemas import (  # noqa: E402
    TeamEvidenceCase,
    TeamMiniBatchMetrics,
    TeamSearchAssignment,
    TeamSearchRequest,
)
from multi_dataset_diverse_rl.team_search.task_builder import (  # noqa: E402
    Layer2EvidenceRequestBuilder,
    NativeFeedRequestBuilder,
)


BACKENDS = ("gepa", "mars")
REGIMES = ("fixed_budget", "saturation")


def _result(*, candidate: bool, reason: str) -> LocalOptimizationResult:
    candidates = (
        LocalPromptCandidate(
            "candidate-1", "Use semantic compatibility.", 1.0,
            {"local-0": 1.0}, ("parent",), 1,
        ),
    ) if candidate else ()
    return LocalOptimizationResult(
        candidates=candidates,
        backend_name="fixture",
        backend_version="fixture-v1",
        optimizer_state=None,
        solver_calls=2,
        optimizer_calls=1,
        input_tokens=3,
        output_tokens=2,
        total_tokens=5,
        termination_reason=reason,
    )


class _NativeBackend:
    async def optimize_native(self, _request: NativeOptimizationRequest) -> LocalOptimizationResult:
        return _result(candidate=True, reason="complete")


class _Layer2Backend:
    async def optimize_layer2(self, request: Layer2OptimizationRequest) -> LocalOptimizationResult:
        assert request.packet.packet_hash
        return _result(candidate=True, reason="complete")


class _Responsibility:
    def __init__(self, assignment: TeamSearchAssignment) -> None:
        self.assignment = assignment

    def assign(self, _request: TeamSearchRequest) -> TeamSearchAssignment:
        return self.assignment


class _Evaluator:
    def __init__(self, *, shadow_passed: bool) -> None:
        self.shadow_passed = shadow_passed
        self.stages: list[str] = []

    def active_evaluation(self, _assignment: TeamSearchAssignment) -> object:
        return object()

    def evaluate_minibatch(self, _assignment, _candidate, minibatch):
        assert len(minibatch) == 12
        self.stages.append("TeamMiniBatch")
        return TeamMiniBatchMetrics(target_delta=1), EvaluationCost(1, 1, 1)

    def evaluate_full(self, _assignment, _candidate):
        self.stages.append("Full")
        return object(), EvaluationCost(1, 1, 1)

    def evaluate_shadow(self, _assignment, _candidate):
        self.stages.append("Shadow")
        return SimpleNamespace(passed=self.shadow_passed), EvaluationCost(1, 1, 1)


class _Selector:
    def annotate(self, records, *, active):
        del active
        return records

    def select(self, records):
        return next((row for row in records if row.promoted), None)


class _Committer:
    def __init__(self) -> None:
        self.ids: list[str] = []

    def commit(self, *, assignment, candidate, evaluation) -> None:
        del assignment, evaluation
        self.ids.append(candidate.candidate_id)


def _assignment() -> TeamSearchAssignment:
    evidence = tuple(
        TeamEvidenceCase(
            f"{group}-{index}", f"payload-{group}-{index}", "A", "B",
            "sanitized", group,
            (("direct_flip",) if group == "responsibility" else ()),
        )
        for group in ("responsibility", "coalition", "preservation")
        for index in range(4)
    )
    return TeamSearchAssignment(
        2, "Use semantic evidence.", evidence, "repair", "responsibility-v1",
        primary_responsibility_lane="direct_flip", responsibility_value=8.0,
    )


def _request(update: int = 1) -> TeamSearchRequest:
    return TeamSearchRequest(
        78, update, "team-initial", 36,
        "COMMON_SOLVER_CONTRACT_V1", "output-v1", "optimize-fixture",
    )


def _registry() -> Layer1BackendRegistry:
    registry = Layer1BackendRegistry()
    for backend in BACKENDS:
        registry.register(
            backend=backend, mode="native", factory=_NativeBackend,
            fidelity=f"{backend}-native-fixture",
        )
        registry.register(
            backend=backend, mode="layer2", factory=_Layer2Backend,
            fidelity=f"{backend}-layer2-fixture",
        )
    return registry


def _config(backend: str, mode: str, regime: str) -> BackendRuntimeConfig:
    saturation = SaturationConfig(
        enabled=regime == "saturation",
        scientific_budget_enabled=regime == "fixed_budget",
        local_no_update_patience=3 if regime == "saturation" else None,
        team_no_update_patience=2 if regime == "saturation" else None,
    )
    return BackendRuntimeConfig(
        optimizer_backend=backend, optimization_mode=mode, seed=78,
        solver_model="solver-fixture", optimizer_model="optimizer-fixture",
        budget="fixture-budget", data_split_manifest="optimize-fixture",
        initial_state="team-initial", run_mode=regime,
        saturation_config_identity=(saturation.identity() if regime == "saturation" else "none"),
    )


def _base_trace(backend: str, scope: str, regime: str) -> dict[str, Any]:
    return {
        "mode": f"{backend}_{scope}".upper(),
        "stopping_regime": regime,
        "initial_state_hash": "team-initial",
        "target_member": 2,
        "responsibility_lane": "direct_flip",
        "focus_ids": [],
        "anchor_ids": [],
        "persistent_realizability": {"member": 2, "failure_count": 0},
        "pivotality": {"P_0": 1, "P_1": 2, "P_2": 3, "P_3": 4, "P_4": 2},
    }


async def _native_trace(backend: str, regime: str) -> dict[str, Any]:
    cfg = _config(backend, "native", regime)
    runtime = _registry().create(cfg)
    problem = NativeFeedRequestBuilder().build(_request(), _assignment())
    result = await runtime.optimize(problem)
    trace = _base_trace(backend, "native", regime)
    trace.update({
        "packet": None,
        "layer1_request_identity": problem.identity(),
        "candidate_ids": [row.candidate_id for row in result.candidates],
        "local_scores": [row.local_score for row in result.candidates],
        "team_stages": [],
        "commit": None,
        "stopping": {
            "local_no_update_patience": 3 if regime == "saturation" else None,
            "team_no_update_patience": None,
            "stop_reason": "SATURATION_REACHED" if regime == "saturation" else "SCIENTIFIC_BUDGET_REACHED",
        },
        "final_state_hash": "candidate-1",
    })
    return trace


async def _layer2_trace(backend: str, regime: str) -> dict[str, Any]:
    cfg = _config(backend, "layer2", regime)
    runtime = _registry().create(cfg)
    assignment = _assignment()
    builder = Layer2EvidenceRequestBuilder()
    packet_request = builder.build(_request(), assignment)
    evaluator = _Evaluator(shadow_passed=regime == "fixed_budget")
    committer = _Committer()
    controller = TeamSearchController(
        responsibility=_Responsibility(assignment), task_builder=builder,
        local_optimizer=runtime, evaluator=evaluator, selector=_Selector(),
        committer=committer,
    )
    outcomes = []
    count = 1 if regime == "fixed_budget" else 2
    state = SaturationState(
        SaturationConfig(
            enabled=regime == "saturation",
            scientific_budget_enabled=regime == "fixed_budget",
            local_no_update_patience=3 if regime == "saturation" else None,
            team_no_update_patience=2 if regime == "saturation" else None,
        ),
        backend=backend,
        mode=f"{backend}_layer2",
    )
    for index in range(count):
        outcome = await controller.run_opportunity(_request(index + 1))
        outcomes.append(outcome)
        state.observe_team_epoch(
            start_state_hash="team-initial",
            end_state_hash=("team-committed" if outcome.committed_candidate_id else "team-initial"),
            team_commit=outcome.committed_candidate_id is not None,
            accepted_local_update=bool(outcome.candidates),
            scientific_budget_reached=regime == "fixed_budget",
        )
    final = outcomes[-1]
    trace = _base_trace(backend, "layer2", regime)
    trace.update({
        "packet": {
            "hash": packet_request.packet.packet_hash,
            "responsibility": [row.example_id for row in packet_request.packet.responsibility_examples],
            "focus": [row.example_id for row in packet_request.packet.focus_examples],
            "anchor": [row.example_id for row in packet_request.packet.anchor_examples],
            "local_eval": [row.example_id for row in packet_request.packet.local_eval_examples],
            "schedule": [list(row) for row in packet_request.packet.ordered_batch_schedule],
        },
        "layer1_request_identity": packet_request.identity(),
        "candidate_ids": [row.local_candidate.candidate_id for row in final.candidates],
        "local_scores": [row.local_candidate.local_score for row in final.candidates],
        "team_stages": evaluator.stages,
        "team_funnel": dict(final.funnel),
        "commit": committer.ids[-1] if committer.ids else None,
        "stopping": {
            "local_no_update_patience": 3 if regime == "saturation" else None,
            "team_no_update_patience": 2 if regime == "saturation" else None,
            "team_no_update_counter": state.team_no_update_counter,
            "stop_reason": state.stop_reason.value if state.stop_reason else None,
        },
        "final_state_hash": "team-committed" if committer.ids else "team-initial",
    })
    return trace


async def capture() -> dict[str, Any]:
    traces: dict[str, Any] = {}
    for backend in BACKENDS:
        for regime in REGIMES:
            traces[f"{backend}_native_{regime}"] = await _native_trace(backend, regime)
            traces[f"{backend}_layer2_{regime}"] = await _layer2_trace(backend, regime)
    return {"schema_version": "production_golden_trace_v1", "traces": traces}


async def _engine_trace(backend: str, scope: str, regime: str) -> dict[str, Any]:
    """Capture the same normalized trace through the production engine."""

    from multi_dataset_diverse_rl.experiment import (
        ExperimentInputs,
        ExperimentServices,
        ExperimentSpec,
        Layer2Opportunity,
        OptimizationScope,
        OptimizerBackend,
        RuntimeContext,
        StoppingRegime,
        run_experiment,
    )
    from multi_dataset_diverse_rl.local_optimizers.production_backends import (
        GEPABackend,
        MARSBackend,
    )

    spec = ExperimentSpec(
        backend=OptimizerBackend(backend),
        optimization_scope=OptimizationScope(scope),
        stopping_regime=StoppingRegime(regime),
        task_identity="fixture-task",
        data_identity="optimize-fixture",
    )
    runtime = RuntimeContext(
        seed=78,
        provider_profile="offline-fixture",
        solver_model="solver-fixture",
        optimizer_model="optimizer-fixture",
        evaluator_model="evaluator-fixture",
        run_identity_sha256="run-fixture",
        authorization_identity="zero-api-fixture",
        cache_identity="cache-fixture",
        ledger_identity="ledger-fixture",
    )
    backend_type = GEPABackend if backend == "gepa" else MARSBackend
    adapter = backend_type(native=_NativeBackend(), layer2=_Layer2Backend())
    assignment = _assignment()
    outer = _request()
    trace = _base_trace(backend, scope, regime)
    if scope == "native":
        problem = NativeFeedRequestBuilder().build(outer, assignment)
        result = await run_experiment(
            spec,
            runtime,
            ExperimentInputs("team-initial", native_problem=problem),
            ExperimentServices(adapter),
        )
        local = result.local_result
        assert local is not None
        trace.update({
            "packet": None,
            "layer1_request_identity": problem.identity(),
            "candidate_ids": [row.candidate_id for row in local.candidates],
            "local_scores": [row.local_score for row in local.candidates],
            "team_stages": [],
            "commit": None,
            "stopping": {
                "local_no_update_patience": 3 if regime == "saturation" else None,
                "team_no_update_patience": None,
                "stop_reason": result.stop_reason,
            },
            "final_state_hash": result.final_state_hash,
        })
        return trace

    builder = Layer2EvidenceRequestBuilder()
    packet_request = builder.build(outer, assignment)
    evaluator = _Evaluator(shadow_passed=regime == "fixed_budget")
    committer = _Committer()

    def controller_factory(bound_backend):
        return TeamSearchController(
            responsibility=_Responsibility(assignment),
            task_builder=builder,
            local_optimizer=bound_backend,
            evaluator=evaluator,
            selector=_Selector(),
            committer=committer,
        )

    count = 1 if regime == "fixed_budget" else 2
    result = await run_experiment(
        spec,
        runtime,
        ExperimentInputs(
            "team-initial",
            layer2_opportunities=tuple(
                Layer2Opportunity(_request(index + 1)) for index in range(count)
            ),
        ),
        ExperimentServices(
            adapter,
            layer2_controller_factory=controller_factory,
            team_state_hash_reader=lambda: (
                "team-committed" if committer.ids else "team-initial"
            ),
        ),
    )
    final = result.team_outcomes[-1]
    trace.update({
        "packet": {
            "hash": packet_request.packet.packet_hash,
            "responsibility": [
                row.example_id for row in packet_request.packet.responsibility_examples
            ],
            "focus": [row.example_id for row in packet_request.packet.focus_examples],
            "anchor": [row.example_id for row in packet_request.packet.anchor_examples],
            "local_eval": [row.example_id for row in packet_request.packet.local_eval_examples],
            "schedule": [list(row) for row in packet_request.packet.ordered_batch_schedule],
        },
        "layer1_request_identity": packet_request.identity(),
        "candidate_ids": [row.local_candidate.candidate_id for row in final.candidates],
        "local_scores": [row.local_candidate.local_score for row in final.candidates],
        "team_stages": evaluator.stages,
        "team_funnel": dict(final.funnel),
        "commit": committer.ids[-1] if committer.ids else None,
        "stopping": {
            "local_no_update_patience": 3 if regime == "saturation" else None,
            "team_no_update_patience": 2 if regime == "saturation" else None,
            "team_no_update_counter": 0 if committer.ids else count,
            "stop_reason": result.stop_reason,
        },
        "final_state_hash": result.final_state_hash,
    })
    return trace


async def capture_engine() -> dict[str, Any]:
    traces: dict[str, Any] = {}
    for backend in BACKENDS:
        for regime in REGIMES:
            traces[f"{backend}_native_{regime}"] = await _engine_trace(
                backend, "native", regime
            )
            traces[f"{backend}_layer2_{regime}"] = await _engine_trace(
                backend, "layer2", regime
            )
    return {"schema_version": "production_golden_trace_v1", "traces": traces}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--implementation", choices=("before", "after"), default="before"
    )
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    payload = asyncio.run(capture() if args.implementation == "before" else capture_engine())
    if args.compare is not None:
        expected = json.loads(args.compare.read_text(encoding="utf-8"))
        if payload != expected:
            raise SystemExit("golden semantic trace mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "trace_count": len(payload["traces"])}, sort_keys=True))


if __name__ == "__main__":
    main()
