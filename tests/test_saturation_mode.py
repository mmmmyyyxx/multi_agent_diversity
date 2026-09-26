from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from multi_dataset_diverse_rl.local_optimizers.base import LocalSolverObservation
from multi_dataset_diverse_rl.local_optimizers.gepa_native import Layer2FrozenBatchSampler
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPALocalPromptOptimizer
from multi_dataset_diverse_rl.local_optimizers.mars_native import (
    MARSLayer2EvidenceOptimizer,
    MARSNativeDataBuilder,
    MARSNativeFeedOptimizer,
    MARSRoleResponse,
)
from multi_dataset_diverse_rl.local_optimizers.schemas import (
    LocalEvidenceExample,
    LocalOptimizationTask,
    LocalOptimizerBudget,
)
from multi_dataset_diverse_rl.native_feed import (
    NativeOptimizationRequest,
    NativeResourceBudget,
    ResponsibilityContext,
)
from multi_dataset_diverse_rl.saturation import (
    LAYER2_EVIDENCE_EPOCH_POLICY_V1,
    Layer2TeamSaturationController,
    OptimizationUnitType,
    SaturationConfig,
    SaturationState,
    StopReason,
    TeamEpochTracker,
)
from multi_dataset_diverse_rl.team_search.schemas import (
    TeamEvidenceCase,
    TeamSearchAssignment,
    TeamSearchRequest,
)
from multi_dataset_diverse_rl.team_search.task_builder import Layer2EvidenceRequestBuilder


@pytest.fixture(autouse=True)
def _provider_isolation(monkeypatch):
    for name in (
        "DASHSCOPE_API_KEY",
        "LWJ_DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "LWJ_DASHSCOPE_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def sat(*, local: int = 2, team: int | None = None, **overrides) -> SaturationConfig:
    values = dict(
        enabled=True,
        scientific_budget_enabled=False,
        local_no_update_patience=local,
        team_no_update_patience=team,
        emergency_max_provider_calls=1000,
        emergency_max_optimizer_steps=1000,
        emergency_max_team_epochs=1000,
        emergency_max_wall_seconds=1000,
    )
    values.update(overrides)
    return SaturationConfig(**values)


@pytest.mark.parametrize(
    ("mode", "unit"),
    [
        ("gepa_native", OptimizationUnitType.GEPA_NATIVE_EPOCH),
        ("gepa_layer2", OptimizationUnitType.LAYER2_EVIDENCE_EPOCH),
        ("mars_native", OptimizationUnitType.MARS_NATIVE_ROUND),
        ("mars_layer2", OptimizationUnitType.MARS_LAYER2_ROUND),
    ],
)
def test_four_modes_share_no_update_and_reset_semantics(mode, unit) -> None:
    state = SaturationState(sat(local=2), backend=mode.split("_")[0], mode=mode)
    assert state.observe_local_unit(
        unit_type=unit,
        start_state_hash="a",
        end_state_hash="b",
        accepted_update=True,
        scientific_budget_reached=True,
    ) is None
    assert state.local_no_update_counter == 0
    assert state.observe_local_unit(
        unit_type=unit, start_state_hash="b", end_state_hash="b", accepted_update=False,
        scientific_budget_reached=True,
    ) is None
    assert state.observe_local_unit(
        unit_type=unit, start_state_hash="b", end_state_hash="b", accepted_update=False,
        scientific_budget_reached=True,
    ) is StopReason.SATURATION_REACHED
    assert state.scientific_stopping_condition_reached is True


def test_hidden_scientific_budget_is_ignored_only_in_saturation_mode() -> None:
    saturation = SaturationState(sat(local=3), backend="gepa", mode="gepa_native")
    assert saturation.observe_local_unit(
        unit_type=OptimizationUnitType.GEPA_NATIVE_EPOCH,
        start_state_hash="a", end_state_hash="a", accepted_update=False,
        scientific_budget_reached=True,
    ) is None
    fixed = SaturationState(
        SaturationConfig(enabled=False, scientific_budget_enabled=True),
        backend="gepa", mode="gepa_native",
    )
    assert fixed.observe_local_unit(
        unit_type=OptimizationUnitType.GEPA_NATIVE_EPOCH,
        start_state_hash="a", end_state_hash="a", accepted_update=False,
        scientific_budget_reached=True,
    ) is StopReason.SCIENTIFIC_BUDGET_REACHED


def test_emergency_ceiling_never_masquerades_as_convergence() -> None:
    state = SaturationState(
        sat(local=5, emergency_max_provider_calls=1),
        backend="gepa", mode="gepa_native",
    )
    state.add_usage(provider_calls=1)
    assert state.check_emergency() is StopReason.EMERGENCY_PROVIDER_CALL_CEILING
    assert state.scientific_stopping_condition_reached is False
    assert state.telemetry()["emergency_ceiling_triggered"] is True


class _Loader:
    def all_ids(self):
        return [0]


def test_one_batch_layer2_packet_replays_as_next_evidence_epoch() -> None:
    sampler = Layer2FrozenBatchSampler(
        ordered_batch_schedule=(("responsibility:e0",),),
        ordered_example_ids=("responsibility:e0",),
        replay_epochs=True,
    )
    assert sampler.next_minibatch_ids(_Loader(), object()) == [0]
    assert sampler.last_delivery_completed_epoch is True
    sampler.last_delivery_completed_epoch = False
    assert sampler.next_minibatch_ids(_Loader(), object()) == [0]
    assert sampler.completed_epoch_count == 2
    assert sampler.epoch_policy == LAYER2_EVIDENCE_EPOCH_POLICY_V1
    assert sampler.backend_example_selection_calls == 0


def test_layer2_local_update_does_not_reset_team_patience_but_vote_neutral_commit_does() -> None:
    state = SaturationState(sat(local=2, team=2), backend="gepa", mode="gepa_layer2")
    outer = Layer2TeamSaturationController(state)
    outer.begin_epoch(eligible_member_ids=(0, 1), start_state_hash="team-a")
    assert outer.observe_opportunity(
        selected_member_ids=(0, 1), local_accepted_update=True, team_commit=False,
        end_state_hash="team-a", vote_acc=0.6,
    ) is None
    assert state.team_no_update_counter == 1
    outer.begin_epoch(eligible_member_ids=(0, 1), start_state_hash="team-a")
    assert outer.observe_opportunity(
        selected_member_ids=(0, 1), local_accepted_update=True, team_commit=True,
        end_state_hash="team-b", vote_acc=0.6,
    ) is None
    assert state.team_no_update_counter == 0
    assert state.team_commit_count == 1


def test_team_epoch_observes_existing_scheduler_choices_and_allows_repeats() -> None:
    epoch = TeamEpochTracker((0, 1, 2))
    assert epoch.observe_opportunity(
        selected_member_ids=(0, 1), local_accepted_update=False, team_commit=False
    ) is False
    assert epoch.observe_opportunity(
        selected_member_ids=(0, 1), local_accepted_update=False, team_commit=False
    ) is False
    assert epoch.observe_opportunity(
        selected_member_ids=(2, 0), local_accepted_update=False, team_commit=False
    ) is True
    assert epoch.opportunity_count == 3


def _row(index: int) -> LocalEvidenceExample:
    return LocalEvidenceExample(f"e{index}", f"problem {index}", "A")


class _Evaluator:
    solver_contract_id = "COMMON_SOLVER_CONTRACT_V1"
    output_contract_id = "output-v1"

    def evaluate(self, procedure, example):
        correct = "semantic" in procedure.casefold()
        return LocalSolverObservation(
            "A" if correct else "B", "", correct, True, input_tokens=1, output_tokens=1
        )


class _Roles:
    async def complete(self, *, role, context):
        del context
        payloads = {
            "planner": {"steps": ["one", "two", "three", "four"]},
            "teacher": {"question": "improve"},
            "critic": {"socratic_valid": True},
            "student": {"decision_procedure": "Use semantic relations."},
        }
        return MARSRoleResponse(payloads[role], provider_called=False)


def _native_request() -> NativeOptimizationRequest:
    return NativeOptimizationRequest(
        request_id="sat-mars-native",
        parent_decision_procedure="Use a generic procedure.",
        target_member=0,
        responsibility=ResponsibilityContext("generic", "r", 1.0),
        team_state_identity="team",
        optimize_universe_id="opt",
        solver_contract_id="COMMON_SOLVER_CONTRACT_V1",
        output_contract_id="output-v1",
        seed=1,
        # Poison every old scientific cap. Saturation must still complete one
        # accepted round plus two unchanged no-update rounds.
        budget=NativeResourceBudget(1, 1, 1, 2),
        provenance={"source_split": "optimize_only"},
    )


def test_mars_native_round_saturation_ignores_old_fixed_caps() -> None:
    optimizer = MARSNativeFeedOptimizer(
        evaluator=_Evaluator(), role_client=_Roles(),
        data_builder=MARSNativeDataBuilder(
            optimize_examples=tuple(_row(i) for i in range(3)), optimize_universe_id="opt"
        ),
        task_definition="solve", layer2_overlay_enabled=False,
        max_stable_rounds=99, saturation_config=sat(local=2),
    )
    result = asyncio.run(optimizer.optimize_native(_native_request()))
    telemetry = result.optimizer_state.payload["saturation"]
    assert result.termination_reason == StopReason.SATURATION_REACHED.value
    assert telemetry["local_epoch_or_round_count"] == 3
    assert telemetry["local_accepted_update_count"] == 1
    assert telemetry["local_no_update_counter"] == 2


def test_mars_saturation_requests_new_native_plans_until_patience() -> None:
    class _OneStepRoles(_Roles):
        def __init__(self):
            self.planner_calls = 0

        async def complete(self, *, role, context):
            response = await super().complete(role=role, context=context)
            if role == "planner":
                self.planner_calls += 1
                return MARSRoleResponse({"steps": ["one"]}, provider_called=False)
            return response

    roles = _OneStepRoles()
    optimizer = MARSNativeFeedOptimizer(
        evaluator=_Evaluator(), role_client=roles,
        data_builder=MARSNativeDataBuilder(
            optimize_examples=tuple(_row(i) for i in range(3)), optimize_universe_id="opt"
        ),
        task_definition="solve", layer2_overlay_enabled=False,
        max_stable_rounds=99, saturation_config=sat(local=2),
    )
    result = asyncio.run(optimizer.optimize_native(_native_request()))
    assert result.termination_reason == StopReason.SATURATION_REACHED.value
    assert roles.planner_calls == 3


def test_gepa_native_epoch_stopper_uses_complete_epochs_not_metric_budget(tmp_path) -> None:
    parent = "Use a generic procedure."

    class _Result:
        per_val_instance_best_candidates = {0: {0}}
        val_aggregate_scores = [0.0]
        candidates = [{"decision_procedure": parent}]
        num_candidates = 1
        val_subscores = [{0: 0.0}]
        parents = [[None]]
        discovery_eval_counts = [1]

        @staticmethod
        def to_dict():
            return {"candidate_count": 1}

    class _State:
        program_candidates = _Result.candidates

    class _Sampler:
        last_delivery_completed_epoch = False

    captured = {}

    def _optimize(**kwargs):
        captured.update(kwargs)
        sampler = kwargs["batch_sampler"]
        callback = kwargs["callbacks"][1]
        for iteration in range(2):
            sampler.last_delivery_completed_epoch = True
            callback.on_iteration_end({
                "iteration": iteration + 1,
                "state": _State(),
                "proposal_accepted": False,
            })
            if kwargs["stop_callbacks"](_State()):
                break
        return _Result()

    accounting = {"successful_calls": 0, "input_tokens": 0, "output_tokens": 0}
    engine = GEPALocalPromptOptimizer(
        evaluator=_Evaluator(),
        reflection_lm=lambda prompt: parent,
        accounting_reader=lambda: accounting,
        run_root=tmp_path,
        optimize_fn=_optimize,
    )
    task = LocalOptimizationTask(
        task_id="gepa-saturation",
        parent_prompt=parent,
        search_examples=(_row(0),),
        local_validation_examples=(_row(0),),
        optimization_context="",
        solver_contract_id="COMMON_SOLVER_CONTRACT_V1",
        output_contract_id="output-v1",
        seed=1,
        budget=LocalOptimizerBudget(1, 3, 4),
    )
    result = asyncio.run(engine.optimize_saturation(
        task,
        batch_sampler=_Sampler(),
        config=sat(local=2),
        unit_type=OptimizationUnitType.GEPA_NATIVE_EPOCH,
        mode_name="gepa_native",
    ))
    assert captured["max_metric_calls"] is None
    assert result.termination_reason == StopReason.SATURATION_REACHED.value
    telemetry = result.optimizer_state.payload["saturation"]
    assert telemetry["local_epoch_or_round_count"] == 2
    assert telemetry["local_no_update_counter"] == 2


def _layer2_request():
    evidence = tuple(
        TeamEvidenceCase(
            f"{group}-{index}", f"problem {group} {index}", "A", "B", "feedback",
            group, (("direct_flip",) if group == "repair" else ()),
        )
        for group in ("repair", "team_hard", "preservation")
        for index in range(4)
    )
    assignment = TeamSearchAssignment(
        0, "Use a generic procedure.", evidence, "repair", "r",
        primary_responsibility_lane="direct_flip", responsibility_value=1.0,
    )
    built = Layer2EvidenceRequestBuilder().build(
        TeamSearchRequest(
            1, 1, "team", 36, "COMMON_SOLVER_CONTRACT_V1", "output-v1", "opt"
        ),
        assignment,
    )
    # Preserve the valid frozen packet schedule while poisoning every legacy
    # per-backend resource cap. Saturation must ignore these values.
    return replace(
        built,
        packet=replace(
            built.packet,
            budget=NativeResourceBudget(1, 1, 1, 2),
        ),
    )


class _PoisonGlobal:
    def build(self, request):
        raise AssertionError(f"global dataset accessed: {request}")

    def identity(self):
        return "poison"


def test_mars_layer2_saturation_uses_packet_only_and_separates_local_state() -> None:
    optimizer = MARSLayer2EvidenceOptimizer(
        evaluator=_Evaluator(), role_client=_Roles(), data_builder=_PoisonGlobal(),
        task_definition="solve", layer2_overlay_enabled=True,
        max_stable_rounds=99, saturation_config=sat(local=2, team=2),
    )
    result = asyncio.run(optimizer.optimize_layer2(_layer2_request()))
    telemetry = result.optimizer_state.payload["saturation"]
    assert result.termination_reason == StopReason.SATURATION_REACHED.value
    assert telemetry["local_accepted_update_count"] == 1
    assert telemetry["team_commit_count"] == 0
    assert result.optimizer_state.payload["native_global_dataset_accessed"] is False
