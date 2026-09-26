from __future__ import annotations

import ast
import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.experiment import (
    ExperimentContractError,
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
from multi_dataset_diverse_rl.local_optimizers.base import LocalOptimizerBackend
from multi_dataset_diverse_rl.local_optimizers.production_backends import (
    GEPABackend,
    MARSBackend,
)
from multi_dataset_diverse_rl.team_search.controller import TeamSearchController
from multi_dataset_diverse_rl.team_search.task_builder import (
    Layer2EvidenceRequestBuilder,
    NativeFeedRequestBuilder,
)
from scripts import capture_production_golden_traces as baseline
from scripts.run_experiment import _execute, preflight


ROOT = Path(__file__).parents[1]
def _runtime() -> RuntimeContext:
    return RuntimeContext(
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


def _spec(backend: str, scope: str, regime: str) -> ExperimentSpec:
    return ExperimentSpec(
        backend=OptimizerBackend(backend),
        optimization_scope=OptimizationScope(scope),
        stopping_regime=StoppingRegime(regime),
        task_identity="fixture-task",
        data_identity="optimize-fixture",
    )


def _backend(name: str):
    cls = GEPABackend if name == "gepa" else MARSBackend
    return cls(native=baseline._NativeBackend(), layer2=baseline._Layer2Backend())


async def _after_trace(backend: str, scope: str, regime: str):
    spec = _spec(backend, scope, regime)
    runtime = _runtime()
    assignment = baseline._assignment()
    outer = baseline._request()
    trace = baseline._base_trace(backend, scope, regime)
    adapter = _backend(backend)
    if scope == "native":
        problem = NativeFeedRequestBuilder().build(outer, assignment)
        result = await run_experiment(
            spec,
            runtime,
            ExperimentInputs("team-initial", native_problem=problem),
            ExperimentServices(adapter),
        )
        assert result.local_result is not None
        trace.update({
            "packet": None,
            "layer1_request_identity": problem.identity(),
            "candidate_ids": [row.candidate_id for row in result.local_result.candidates],
            "local_scores": [row.local_score for row in result.local_result.candidates],
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
    evaluator = baseline._Evaluator(shadow_passed=regime == "fixed_budget")
    committer = baseline._Committer()

    def controller_factory(bound_backend):
        return TeamSearchController(
            responsibility=baseline._Responsibility(assignment),
            task_builder=builder,
            local_optimizer=bound_backend,
            evaluator=evaluator,
            selector=baseline._Selector(),
            committer=committer,
        )

    count = 1 if regime == "fixed_budget" else 2
    opportunities = tuple(
        Layer2Opportunity(baseline._request(index + 1)) for index in range(count)
    )
    result = await run_experiment(
        spec,
        runtime,
        ExperimentInputs("team-initial", layer2_opportunities=opportunities),
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
            "local_eval": [
                row.example_id for row in packet_request.packet.local_eval_examples
            ],
            "schedule": [
                list(row) for row in packet_request.packet.ordered_batch_schedule
            ],
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


@pytest.mark.parametrize("backend", ["gepa", "mars"])
@pytest.mark.parametrize("scope", ["native", "layer2"])
@pytest.mark.parametrize("regime", ["fixed_budget", "saturation"])
def test_eight_final_semantic_traces_replay_exactly(backend, scope, regime) -> None:
    actual = asyncio.run(_after_trace(backend, scope, regime))
    assert actual == asyncio.run(_after_trace(backend, scope, regime))
    if scope == "layer2":
        expected = [f"repair-{index}" for index in range(4)]
        expected.sort(key=lambda value: (hashlib.sha256(value.encode("utf-8")).hexdigest(), value))
        assert actual["packet"]["responsibility"] == expected


def test_layer2_is_backend_independent_for_equal_local_results() -> None:
    gepa = asyncio.run(_after_trace("gepa", "layer2", "fixed_budget"))
    mars = asyncio.run(_after_trace("mars", "layer2", "fixed_budget"))
    gepa.pop("mode")
    mars.pop("mode")
    assert gepa == mars


def test_both_production_adapters_implement_one_backend_contract() -> None:
    assert isinstance(_backend("gepa"), LocalOptimizerBackend)
    assert isinstance(_backend("mars"), LocalOptimizerBackend)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provider_profile", "", "provider_profile"),
        ("solver_model", "", "solver_model"),
        ("run_identity_sha256", "", "run_identity"),
    ],
)
def test_runtime_contract_fails_with_typed_errors(field, value, message) -> None:
    with pytest.raises(ExperimentContractError, match=message):
        replace(_runtime(), **{field: value})


def test_scientific_contract_fails_before_dispatch() -> None:
    with pytest.raises(ExperimentContractError, match="backend"):
        ExperimentSpec(  # type: ignore[arg-type]
            backend="gepa",
            optimization_scope=OptimizationScope.NATIVE,
            stopping_regime=StoppingRegime.FIXED_BUDGET,
            task_identity="task",
            data_identity="data",
        )
    with pytest.raises(ExperimentContractError, match="patience"):
        replace(_spec("gepa", "layer2", "saturation"), local_no_update_patience=4)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            values.add(node.module)
    return values


def test_production_dependency_direction_and_historical_isolation() -> None:
    core = ROOT / "multi_dataset_diverse_rl/experiment.py"
    backends = ROOT / "multi_dataset_diverse_rl/local_optimizers/production_backends.py"
    entrypoint = ROOT / "scripts/run_experiment.py"
    assert not any(name == "scripts" or name.startswith("scripts.") for name in _imports(core))
    assert "multi_dataset_diverse_rl.config" not in _imports(core)
    assert "multi_dataset_diverse_rl.config" not in _imports(backends)
    source = entrypoint.read_text(encoding="utf-8")
    assert "FROZEN_EXECUTION_GOVERNANCE_NOT_BOUND" in source
    assert "seed75" not in source.casefold()
    assert "seed78" not in source.casefold()
    assert "primary_responsibility_ab" not in source
    runtime_source = (ROOT / "multi_dataset_diverse_rl/team_search/system_runtime.py").read_text(
        encoding="utf-8"
    )
    assert 'parent_id=f"seed78_' not in runtime_source


def test_layer2_does_not_import_production_concrete_backends() -> None:
    forbidden = {"gepa", "mars", "production_backends"}
    for path in (ROOT / "multi_dataset_diverse_rl/team_search").glob("*.py"):
        imports = _imports(path)
        assert not any(any(token in name for token in forbidden) for name in imports), path


def test_new_experiment_is_configuration_only() -> None:
    base = _spec("gepa", "layer2", "saturation")
    future = replace(base, task_identity="future-task", data_identity="future-data")
    mars = replace(future, backend=OptimizerBackend.MARS)
    assert future.mode_id == "GEPA_LAYER2"
    assert mars.mode_id == "MARS_LAYER2"
    assert future.identity() != base.identity()


def test_opt_in_method_identity_never_uses_historical_v15_version(monkeypatch) -> None:
    from multi_dataset_diverse_rl import versions

    identities = {
        _spec(backend, scope, "fixed_budget").method_identity
        for backend in ("gepa", "mars")
        for scope in ("native", "layer2")
    }
    assert len(identities) == 4
    assert all(versions.UNIFIED_EXPERIMENT_ENGINE_VERSION in item for item in identities)
    assert all(versions.METHOD_VERSION not in item for item in identities)
    assert versions.METHOD_VERSION == "member_aware_peer_state_v15"
    spec = _spec("gepa", "layer2", "saturation")
    original = spec.identity()
    monkeypatch.setattr(versions, "TEAM_MINIBATCH_CONTRACT_VERSION", "changed-fixture")
    assert spec.identity() != original


def test_cli_never_treats_factory_string_as_execution_authorization() -> None:
    manifest = {
        "scientific": {
            "backend": "gepa",
            "optimization_scope": "layer2",
            "stopping_regime": "saturation",
            "task_identity": "offline-fixture",
            "data_identity": "optimize-fixture",
        },
        "runtime": {
            "seed": 78,
            "provider_profile": "offline-fixture",
            "solver_model": "solver-fixture",
            "optimizer_model": "optimizer-fixture",
            "evaluator_model": "evaluator-fixture",
            "run_identity_sha256": "unverified-run",
            "authorization_identity": "unverified-authorization",
            "cache_identity": "cache-fixture",
            "ledger_identity": "ledger-fixture",
        },
        "execution_factory": "untrusted.module:construct_provider",
    }
    result = preflight(manifest)
    assert result["gate"] == "HOLD"
    assert result["blockers"] == ["FROZEN_EXECUTION_GOVERNANCE_NOT_BOUND"]
    assert result["provider_attempts"] == result["validation_calls"] == result["test_calls"] == 0
    with pytest.raises(ExperimentContractError, match="ABORT_PRE_PROVIDER"):
        asyncio.run(_execute(manifest))
