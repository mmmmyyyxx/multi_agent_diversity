"""Zero-provider engine traces for the frozen parent-scoped V4 team epoch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import pytest

from multi_dataset_diverse_rl.experiment import (
    ExperimentEarlyStop, ExperimentInputs, ExperimentServices, ExperimentSpec,
    Layer2Opportunity, OptimizationScope, OptimizerBackend, RuntimeContext,
    StoppingRegime, run_experiment,
)
from multi_dataset_diverse_rl.local_optimizers.schemas import (
    LocalOptimizationResult, OpaqueOptimizerState,
)
from multi_dataset_diverse_rl.native_feed import (
    NativeOptimizationRequest, NativeResourceBudget, ResponsibilityContext,
)
from multi_dataset_diverse_rl.saturation import SaturationEmergency, StopReason
from multi_dataset_diverse_rl.team_search.schemas import TeamSearchRequest
from multi_dataset_diverse_rl import versions
from multi_dataset_diverse_rl.versions import LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION


def test_formal_v4_identity_is_explicit_and_unknown_protocol_fails_closed():
    common = dict(
        backend=OptimizerBackend.GEPA, optimization_scope=OptimizationScope.LAYER2,
        stopping_regime=StoppingRegime.SATURATION, task_identity="fake-task",
        data_identity="fake-data",
    )
    v4 = ExperimentSpec(**common, layer2_protocol_version=LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION)
    legacy = ExperimentSpec(**common)
    assert v4.method_identity.endswith("GEPA_LAYER2_V4")
    assert v4.identity() != legacy.identity()
    assert versions.LAYER2_EVIDENCE_PACKET_V4_VERSION
    with pytest.raises(ValueError, match="unsupported opt-in Layer2 protocol"):
        ExperimentSpec(**common, layer2_protocol_version="unfrozen-v5")


def _run(plan, eligible_by_parent, *, empty_at_start=False):
    state = {"hash": "s0", "calls": 0}
    runtime = RuntimeContext(
        seed=80, provider_profile="fake", solver_model="fake", optimizer_model="fake",
        evaluator_model="fake", run_identity_sha256="fake-run",
        authorization_identity="fake-auth", cache_identity="fake-cache",
        ledger_identity="fake-ledger",
    )
    spec = ExperimentSpec(
        backend=OptimizerBackend.GEPA, optimization_scope=OptimizationScope.LAYER2,
        stopping_regime=StoppingRegime.SATURATION, task_identity="fake-task",
        data_identity="fake-data", fixed_budget_units=1,
        layer2_protocol_version=LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
    )

    class _Backend:
        name = "gepa"
        fidelity = "fake"

    class _Controller:
        async def run_opportunity(self, request):
            selected, commit, successor = plan[request.update_index]
            state["calls"] += 1
            if commit:
                state["hash"] = successor
            return SimpleNamespace(
                candidates=(), committed_candidate_id=("child" if commit else None),
                funnel={"local_candidates": 0},
                audit_metadata={"selected_target_ids": (selected,), "branch_metadata": ()},
            )

    def next_opportunity(index, parent):
        if empty_at_start or index > len(plan):
            raise ExperimentEarlyStop("NO_FEASIBLE_LAYER2_OPPORTUNITY")
        eligible = eligible_by_parent[parent]
        request = TeamSearchRequest(
            seed=80, update_index=index - 1, team_state_hash=parent,
            local_metric_budget=36, solver_contract_id="COMMON_SOLVER_CONTRACT_V1",
            output_contract_id="output-v1", optimize_universe_id="fake-data",
        )
        return Layer2Opportunity(request, eligible_member_ids=eligible)

    result = asyncio.run(run_experiment(
        spec, runtime, ExperimentInputs("s0"),
        ExperimentServices(
            backend=_Backend(), layer2_controller_factory=lambda _: _Controller(),
            team_state_hash_reader=lambda: state["hash"],
            layer2_opportunity_factory=next_opportunity,
            durable_usage_reader=lambda: {
                "provider_attempts": state["calls"],
                "successful_provider_calls": state["calls"],
                "failed_provider_attempts": 0, "cache_hits": 0,
            },
        ),
    ))
    return result, state


def _counter(event):
    return event.telemetry["saturation"]["team_no_update_counter"]


def test_commit_interrupts_old_parent_epoch_and_recomputes_eligibility():
    result, state = _run(
        [(0, False, None), (1, True, "s1"), (3, False, None)],
        {"s0": (0, 1, 2), "s1": (3,)},
    )
    assert [row.telemetry["audit"]["selected_target_ids"] for row in result.events] == [
        (0,), (1,), (3,),
    ]
    assert [row.telemetry["epoch_end_reason"] for row in result.events] == [
        None, "TEAM_COMMIT", "COVERAGE_COMPLETE_NO_COMMIT",
    ]
    assert [_counter(row) for row in result.events] == [0, 0, 1]
    assert result.events[1].telemetry["coverage_complete"] is False
    assert result.events[1].telemetry["saturation"]["team_commit_count"] == 1
    assert result.final_state_hash == "s1" and state["calls"] == 3


def test_two_same_parent_complete_no_commit_epochs_trigger_saturation():
    result, _ = _run(
        [(0, False, None), (1, False, None), (0, False, None), (1, False, None)],
        {"s0": (0, 1)},
    )
    assert result.stop_reason == "SATURATION_REACHED"
    assert [_counter(row) for row in result.events] == [0, 1, 1, 2]
    assert result.events[-1].telemetry["saturation"]["team_epoch_count"] == 2


def test_partial_next_epoch_commit_resets_prior_no_update_patience():
    result, _ = _run(
        [(0, False, None), (1, False, None), (0, True, "s1"), (2, False, None)],
        {"s0": (0, 1), "s1": (2,)},
    )
    assert [_counter(row) for row in result.events] == [0, 1, 0, 1]
    assert result.events[2].telemetry["coverage_complete"] is False
    assert result.events[2].telemetry["saturation"]["team_epoch_count"] == 1
    assert result.events[2].telemetry["saturation"]["team_commit_count"] == 1


def test_empty_v4_eligible_set_stops_before_epoch_or_provider():
    result, state = _run([], {"s0": ()}, empty_at_start=True)
    assert result.stop_reason == "NO_FEASIBLE_LAYER2_OPPORTUNITY"
    assert result.events == () and result.team_outcomes == ()
    assert state["calls"] == 0


def test_native_saturation_requires_backend_counter_evidence():
    problem = NativeOptimizationRequest(
        request_id="native-fake", parent_decision_procedure="Reason from context.",
        target_member=0, responsibility=ResponsibilityContext("generic", "native", 1.0),
        team_state_identity="s0", optimize_universe_id="fake-data",
        solver_contract_id="COMMON_SOLVER_CONTRACT_V1", output_contract_id="output-v1",
        seed=80, budget=NativeResourceBudget(1, 1, 1), provenance={},
    )
    spec = ExperimentSpec(
        backend=OptimizerBackend.GEPA, optimization_scope=OptimizationScope.NATIVE,
        stopping_regime=StoppingRegime.SATURATION, task_identity="fake-task",
        data_identity="fake-data",
    )
    runtime = RuntimeContext(80, "fake", "fake", "fake", "fake", "run", "auth", "cache", "ledger")

    for backend_reason, counter, expected in (
        ("SATURATION_REACHED", 3, "SATURATION_REACHED"),
        ("SATURATION_REACHED", 2, "OPERATIONAL_ABORT"),
        ("EMERGENCY_OPTIMIZER_STEP_CEILING", 1, "EMERGENCY_OPTIMIZER_STEP_CEILING"),
        ("gepa_metric_budget_exhausted", 3, "OPERATIONAL_ABORT"),
    ):
        class Backend:
            name = "gepa"
            fidelity = "fake"

            async def optimize(self, request, context):
                return LocalOptimizationResult(
                    candidates=(), backend_name="gepa", backend_version="fake",
                    optimizer_state=OpaqueOptimizerState("gepa", "fake", {
                        "saturation": {
                            "stop_reason": backend_reason,
                            "local_no_update_counter": counter,
                        },
                    }),
                    solver_calls=0, optimizer_calls=0,
                    input_tokens=0, output_tokens=0, total_tokens=0,
                    termination_reason=backend_reason,
                )

        result = asyncio.run(run_experiment(
            spec, runtime, ExperimentInputs("s0", native_problem=problem),
            ExperimentServices(backend=Backend()),
        ))
        assert result.stop_reason == expected
        assert result.events[0].telemetry["backend_termination_reason"] == backend_reason
        assert result.events[0].telemetry["canonical_stop_reason"] == expected


def test_local_emergency_aborts_before_team_outcome_or_epoch():
    state = {"hash": "s0", "calls": 0}
    spec = ExperimentSpec(
        backend=OptimizerBackend.GEPA, optimization_scope=OptimizationScope.LAYER2,
        stopping_regime=StoppingRegime.SATURATION, task_identity="fake-task",
        data_identity="fake-data", layer2_protocol_version=LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
    )
    runtime = RuntimeContext(80, "fake", "fake", "fake", "fake", "run", "auth", "cache", "ledger")

    class Backend:
        name = "gepa"
        fidelity = "fake"

    class Controller:
        async def run_opportunity(self, request):
            state["calls"] += 1
            raise SaturationEmergency(StopReason.EMERGENCY_OPTIMIZER_STEP_CEILING)

    result = asyncio.run(run_experiment(
        spec, runtime, ExperimentInputs("s0"),
        ExperimentServices(
            backend=Backend(), layer2_controller_factory=lambda _: Controller(),
            team_state_hash_reader=lambda: state["hash"],
            layer2_opportunity_factory=lambda index, parent: Layer2Opportunity(
                TeamSearchRequest(80, index - 1, parent, 36,
                                  "COMMON_SOLVER_CONTRACT_V1", "output-v1"),
                eligible_member_ids=(0,),
            ),
        ),
    ))
    assert result.stop_reason == "EMERGENCY_OPTIMIZER_STEP_CEILING"
    assert result.events == () and result.team_outcomes == ()
    assert state["hash"] == "s0" and state["calls"] == 1
