from __future__ import annotations

import asyncio

from multi_dataset_diverse_rl.local_optimizers.base import LocalSolverObservation
from multi_dataset_diverse_rl.local_optimizers.mars_native import (
    MARSNativeDataBuilder,
    MARSNativeFeedOptimizer,
    MARSRoleResponse,
)
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample
from multi_dataset_diverse_rl.native_feed import (
    NativeOptimizationRequest,
    NativeResourceBudget,
    ResponsibilityContext,
)


def row(index: int) -> LocalEvidenceExample:
    return LocalEvidenceExample(f"e{index}", f"problem {index}", "A")


def request() -> NativeOptimizationRequest:
    return NativeOptimizationRequest(
        request_id="mars-native-fixture",
        parent_decision_procedure="Use a generic decision procedure.",
        target_member=1,
        responsibility=ResponsibilityContext("near_margin", "r2", 4.0),
        team_state_identity="team",
        optimize_universe_id="opt5",
        solver_contract_id="COMMON_SOLVER_CONTRACT_V1",
        output_contract_id="output-v1",
        seed=8,
        budget=NativeResourceBudget(1, 10, 4, 2),
        provenance={"source_split": "optimize_only"},
    )


class Evaluator:
    solver_contract_id = "COMMON_SOLVER_CONTRACT_V1"
    output_contract_id = "output-v1"

    def __init__(self):
        self.ids = []

    def evaluate(self, procedure, example):
        self.ids.append(example.example_id)
        correct = "semantic" in procedure.casefold()
        return LocalSolverObservation(
            "A" if correct else "B", "", correct, True, input_tokens=1, output_tokens=1
        )


class Roles:
    def __init__(self):
        self.calls = []

    async def complete(self, *, role, context):
        self.calls.append((role, context))
        payloads = {
            "planner": {"steps": ["improve semantic disambiguation"]},
            "teacher": {"question": "Which semantic relation resolves ambiguity?"},
            "critic": {"socratic_valid": True},
            "student": {"decision_procedure": "Use semantic relations to resolve ambiguity."},
        }
        return MARSRoleResponse(payloads[role], input_tokens=1, output_tokens=1)


def build(overlay: bool):
    evaluator = Evaluator()
    roles = Roles()
    data = MARSNativeDataBuilder(
        optimize_examples=tuple(row(i) for i in range(5)),
        optimize_universe_id="opt5",
    )
    optimizer = MARSNativeFeedOptimizer(
        evaluator=evaluator,
        role_client=roles,
        data_builder=data,
        task_definition="Solve disambiguation QA.",
        layer2_overlay_enabled=overlay,
    )
    return optimizer, evaluator, roles


def test_official_code_flow_uses_full_target_dataset_and_returns_candidate() -> None:
    optimizer, evaluator, roles = build(True)
    result = asyncio.run(optimizer.optimize_native(request()))
    assert [role for role, _ in roles.calls] == ["planner", "teacher", "critic", "student"]
    assert evaluator.ids == [f"e{i}" for i in range(5)] * 2
    assert result.solver_calls == 10
    assert result.optimizer_calls == 4
    assert result.candidates and result.candidates[0].local_score == 1.0
    state = result.optimizer_state
    assert state is not None
    assert state.payload["target_dataset_size"] == 5
    assert state.payload["responsibility_overlay_present"] is True
    assert len(state.payload["history"]) == 2
    assert state.payload["completed_rounds"] == 1
    assert state.payload["termination_reason"] == "candidate_returned"
    assert all(
        "search_examples" not in context and "local_validation_examples" not in context
        for _, context in roles.calls
    )


def test_control_treatment_data_and_native_contract_are_identical() -> None:
    treatment, _, treatment_roles = build(True)
    control, _, control_roles = build(False)
    assert treatment.parity_identity(request()) == control.parity_identity(request())
    asyncio.run(treatment.optimize_native(request()))
    asyncio.run(control.optimize_native(request()))
    assert treatment.data_builder.examples == control.data_builder.examples
    assert treatment_roles.calls[0][1]["responsibility_overlay"] is not None
    assert control_roles.calls[0][1]["responsibility_overlay"] is None
