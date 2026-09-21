from __future__ import annotations

import asyncio

from multi_dataset_diverse_rl.local_optimizers.base import LocalSolverObservation
from multi_dataset_diverse_rl.local_optimizers.mars_native import (
    MARSLayer2EvidenceOptimizer,
    MARSNativeDataBuilder,
    MARSNativeFeedOptimizer,
    MARSRoleResponse,
)
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
        1,
        "Use a generic decision procedure.",
        evidence,
        f"repair direct flips{suffix}",
        f"responsibility{suffix}",
        primary_responsibility_lane="direct_flip",
        responsibility_value=8.0,
    )
    outer = TeamSearchRequest(
        80, 1, "team-state", 36, "COMMON_SOLVER_CONTRACT_V1", "output-v1",
        "optimize-only-v1",
    )
    return Layer2EvidenceRequestBuilder().build(outer, assignment)


class PoisonGlobalDataBuilder:
    def build(self, _request):
        raise AssertionError("MARS global/native dataset must not be read in treatment")

    def identity(self):
        return "poison-global-data"


def build_layer2():
    evaluator = Evaluator()
    roles = Roles()
    optimizer = MARSLayer2EvidenceOptimizer(
        evaluator=evaluator,
        role_client=roles,
        data_builder=PoisonGlobalDataBuilder(),
        task_definition="Solve disambiguation QA.",
        layer2_overlay_enabled=True,
    )
    return optimizer, evaluator, roles


def test_layer2_evidence_reaches_mars_roles_and_exact_target_set() -> None:
    optimizer, evaluator, roles = build_layer2()
    request = layer2_request()
    before = request.packet.packet_hash
    result = asyncio.run(optimizer.optimize_layer2(request))
    assert request.packet.packet_hash == before
    assert [role for role, _ in roles.calls] == [
        "planner", "teacher", "critic", "student"
    ]
    expected_eval_ids = [
        row.packet_item_id for row in request.packet.local_eval_examples
    ] * 2
    assert evaluator.ids == expected_eval_ids
    assert all(
        (
            "layer2_evidence_packet" in context
            or role == "critic" and context["packet_hash"] == request.packet.packet_hash
        )
        for role, context in roles.calls
    )
    planner_packet = roles.calls[0][1]["layer2_evidence_packet"]
    assert len(planner_packet["TEAM RESPONSIBILITY EVIDENCE"]) == 4
    assert planner_packet["RECENT REGRESSION / FOCUS EVIDENCE"] == []
    assert planner_packet["RECENT GAIN / ANCHOR EVIDENCE"] == []
    assert all(
        "layer2_evidence_packet" in context
        for _, context in roles.calls
    )
    assert result.candidates
    assert result.optimizer_state.payload["backend_example_selection_calls"] == 0
    assert result.optimizer_state.payload["native_global_dataset_accessed"] is False
    assert result.optimizer_state.payload["local_eval_count"] == len(
        request.packet.local_eval_examples
    )
    assert result.optimizer_state.payload["role_intersection_counts"] == dict(
        request.packet.role_intersection_counts
    )
    assert result.candidates[0].backend_metadata["responsibility_packet_hash"] == before


def test_layer2_packet_changes_mars_planner_and_tcs_input() -> None:
    optimizer_a, _, roles_a = build_layer2()
    optimizer_b, _, roles_b = build_layer2()
    asyncio.run(optimizer_a.optimize_layer2(layer2_request("-a")))
    asyncio.run(optimizer_b.optimize_layer2(layer2_request("-b")))
    assert roles_a.calls[0][1] != roles_b.calls[0][1]
    assert roles_a.calls[1][1] != roles_b.calls[1][1]
