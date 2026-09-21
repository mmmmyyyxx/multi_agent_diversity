from __future__ import annotations

import asyncio
from pathlib import Path

from multi_dataset_diverse_rl.local_optimizers.base import LocalSolverObservation
from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import GEPAAdapter
from multi_dataset_diverse_rl.local_optimizers.gepa_native import (
    GEPANativeDataBuilder,
    GEPANativeFeedOptimizer,
    GEPANativeSplitConfig,
    NativeFeedGEPAAdapter,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPALocalPromptOptimizer
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample
from multi_dataset_diverse_rl.native_feed import (
    NativeOptimizationRequest,
    NativeResourceBudget,
    ResponsibilityContext,
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

    def evaluate(self, procedure, example):
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
