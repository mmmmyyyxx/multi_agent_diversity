from __future__ import annotations

import asyncio
from pathlib import Path

from multi_dataset_diverse_rl.local_optimizers.base import LocalSolverObservation
from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import GEPAAdapter
from multi_dataset_diverse_rl.local_optimizers.gepa_native import (
    GEPANativeDataBuilder,
    GEPANativeFeedOptimizer,
    GEPANativeSplitConfig,
    GEPALayer2EvidenceOptimizer,
    NativeFeedGEPAAdapter,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPALocalPromptOptimizer
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample
from multi_dataset_diverse_rl.native_feed import (
    Layer2OptimizationRequest,
    NativeOptimizationRequest,
    NativeResourceBudget,
    ResponsibilityContext,
)
from multi_dataset_diverse_rl.team_search.schemas import (
    TeamEvidenceCase,
    TeamSearchAssignment,
    TeamSearchRequest,
)
from multi_dataset_diverse_rl.team_search.task_builder import (
    Layer2EvidenceRequestBuilder,
)


def row(index: int) -> LocalEvidenceExample:
    return LocalEvidenceExample(
        example_id=f"e{index}",
        input_payload=f"Disambiguation problem {index}",
        gold="A",
        tags=("coverage",),
    )


def request() -> NativeOptimizationRequest:
    return NativeOptimizationRequest(
        request_id="native-gepa-fixture",
        parent_decision_procedure="Use a generic decision procedure.",
        target_member=2,
        responsibility=ResponsibilityContext("coverage", "r1", 3.0),
        team_state_identity="team",
        optimize_universe_id="opt9",
        solver_contract_id="COMMON_SOLVER_CONTRACT_V1",
        output_contract_id="output-v1",
        seed=9,
        budget=NativeResourceBudget(12, 12, 4, 4),
        provenance={"source_split": "optimize_only"},
    )


class Evaluator:
    solver_contract_id = "COMMON_SOLVER_CONTRACT_V1"
    output_contract_id = "output-v1"

    def __init__(self):
        self.ids = []

    def evaluate(self, procedure, example):
        self.ids.append(example.example_id)
        correct = "distinguish" in procedure.casefold()
        return LocalSolverObservation(
            "A" if correct else "B", "reasoning", correct, True, input_tokens=1,
            output_tokens=1,
        )


class Reflection:
    def __init__(self):
        self.accounting = {"successful_calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        self.accounting["successful_calls"] += 1
        return "```Distinguish referents using semantic compatibility evidence.```"


def build_optimizer(tmp_path: Path, overlay: bool):
    reflection = Reflection()
    engine = GEPALocalPromptOptimizer(
        evaluator=Evaluator(),
        reflection_lm=reflection,
        accounting_reader=lambda: reflection.accounting,
        run_root=tmp_path,
        adapter_factory=NativeFeedGEPAAdapter,
    )
    data = GEPANativeDataBuilder(
        optimize_examples=tuple(row(i) for i in range(9)),
        optimize_universe_id="opt9",
        config=GEPANativeSplitConfig(6, 3),
    )
    return GEPANativeFeedOptimizer(
        engine=engine, data_builder=data, layer2_overlay_enabled=overlay
    ), reflection


def test_gepa_owns_native_train_val_and_layer2_does_not_filter(tmp_path: Path) -> None:
    treatment, reflection = build_optimizer(tmp_path / "treatment", True)
    control, _ = build_optimizer(tmp_path / "control", False)
    treatment_task = treatment.task_for(request())
    control_task = control.task_for(request())
    assert len(treatment_task.search_examples) == 6
    assert len(treatment_task.local_validation_examples) == 3
    assert treatment_task.search_examples == control_task.search_examples
    assert treatment_task.local_validation_examples == control_task.local_validation_examples
    assert "LAYER2_RESPONSIBILITY_OVERLAY_V1" in treatment_task.optimization_context
    assert control_task.optimization_context == ""
    assert treatment.parity_identity(request()) == control.parity_identity(request())
    result = asyncio.run(treatment.optimize_native(request()))
    assert result.candidates
    assert result.solver_calls > 0
    assert result.optimizer_calls > 0
    assert reflection.prompts


def test_overlay_is_additive_to_native_reflective_record() -> None:
    adapter = NativeFeedGEPAAdapter(
        Evaluator(),
        parent_prompt="parent",
        all_examples=(row(0),),
        optimization_context="overlay_version=GEPA_LAYER2_RESPONSIBILITY_OVERLAY_V1",
        output_contract_id="output-v1",
    )
    evaluated = adapter.evaluate(
        [row(0)], {"decision_procedure": "Distinguish semantic referents."}, True
    )
    record = adapter.make_reflective_dataset(
        {"decision_procedure": "Distinguish semantic referents."},
        evaluated,
        ["decision_procedure"],
    )["decision_procedure"][0]
    assert "Problem" in record and "Reasoning Evidence" in record
    assert record["Layer2 Responsibility Overlay"].startswith("overlay_version=")


def layer2_request(suffix: str = "") -> Layer2OptimizationRequest:
    evidence = tuple(
        TeamEvidenceCase(
            f"{group}-{index}{suffix}",
            f"problem {group} {index}{suffix}",
            "A",
            "B",
            "sanitized outcome",
            group,
            (("direct_flip",) if group == "responsibility" else ()),
        )
        for group in ("responsibility", "coalition", "preservation")
        for index in range(4)
    )
    assignment = TeamSearchAssignment(
        2,
        "Use a generic decision procedure.",
        evidence,
        f"repair responsibility{suffix}",
        f"resp{suffix}",
        primary_responsibility_lane="direct_flip",
        responsibility_value=8.0,
    )
    outer = TeamSearchRequest(
        80, 1, "team-state", 36, "COMMON_SOLVER_CONTRACT_V1", "output-v1",
        "optimize-only-v1",
    )
    return Layer2EvidenceRequestBuilder().build(outer, assignment)


def build_layer2_optimizer(tmp_path: Path):
    evaluator = Evaluator()
    reflection = Reflection()
    engine = GEPALocalPromptOptimizer(
        evaluator=evaluator,
        reflection_lm=reflection,
        accounting_reader=lambda: reflection.accounting,
        run_root=tmp_path,
        adapter_factory=NativeFeedGEPAAdapter,
    )
    return GEPALayer2EvidenceOptimizer(engine=engine), evaluator, reflection


def test_layer2_schedule_replaces_native_sampler_and_reaches_solver(
    tmp_path: Path, monkeypatch
) -> None:
    from gepa.strategies.batch_sampler import EpochShuffledBatchSampler

    monkeypatch.setattr(
        EpochShuffledBatchSampler,
        "next_minibatch_ids",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native sampler must not run in treatment")
        ),
    )
    optimizer, evaluator, reflection = build_layer2_optimizer(tmp_path)
    request = layer2_request()
    before = request.packet.packet_hash
    result = asyncio.run(optimizer.optimize_layer2(request))
    assert request.packet.packet_hash == before
    assert optimizer.last_sampler is not None
    assert optimizer.last_sampler.backend_example_selection_calls == 0
    assert optimizer.last_sampler.delivery_calls >= 1
    assert optimizer.last_sampler.delivered_ids == list(
        request.packet.ordered_batch_schedule[
            : optimizer.last_sampler.delivery_calls
        ]
    )
    packet_ids = {
        row.packet_item_id
        for row in (
            *request.packet.responsibility_examples,
            *request.packet.focus_examples,
            *request.packet.anchor_examples,
            *request.packet.local_eval_examples,
        )
    }
    assert set(evaluator.ids).issubset(packet_ids)
    assert result.candidates
    assert all(
        row.backend_metadata["responsibility_packet_hash"]
        == request.packet.packet_hash
        for row in result.candidates
    )
    telemetry = result.optimizer_state.payload["telemetry"]
    assert telemetry["backend_example_selection_calls"] == 0
    assert telemetry["native_sampler_called"] is False
    assert telemetry["local_eval_count"] == len(request.packet.local_eval_examples)
    assert telemetry["role_intersection_counts"] == dict(
        request.packet.role_intersection_counts
    )
    assert reflection.prompts
    assert request.packet.packet_hash in reflection.prompts[0]
    assert "TEAM RESPONSIBILITY EVIDENCE" in reflection.prompts[0]
    assert "RECENT REGRESSION / FOCUS EVIDENCE" in reflection.prompts[0]
    assert "RECENT GAIN / ANCHOR EVIDENCE" in reflection.prompts[0]


def test_layer2_packet_changes_gepa_reflection_input(tmp_path: Path) -> None:
    optimizer_a, _, reflection_a = build_layer2_optimizer(tmp_path / "a")
    optimizer_b, _, reflection_b = build_layer2_optimizer(tmp_path / "b")
    asyncio.run(optimizer_a.optimize_layer2(layer2_request("-a")))
    asyncio.run(optimizer_b.optimize_layer2(layer2_request("-b")))
    assert reflection_a.prompts and reflection_b.prompts
    assert reflection_a.prompts[0] != reflection_b.prompts[0]
