from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from multi_dataset_diverse_rl.native_feed import NativeOptimizationRequest
from multi_dataset_diverse_rl.native_feed_audit import layer2_contract_manifest
from multi_dataset_diverse_rl.team_search.schemas import (
    TeamEvidenceCase,
    TeamSearchAssignment,
    TeamSearchRequest,
)
from multi_dataset_diverse_rl.team_search.task_builder import NativeFeedRequestBuilder


ROOT = Path(__file__).parents[1]


def _evidence(index: int, group: str, tag: str) -> TeamEvidenceCase:
    return TeamEvidenceCase(
        example_id=f"private-{index}",
        input_payload=f"private question {index}",
        gold="A",
        target_output="B",
        feedback="private feedback",
        evidence_group=group,
        tags=(tag,),
    )


def test_native_request_contains_outer_control_but_no_backend_samples() -> None:
    rows = tuple(
        _evidence(i, group, tag)
        for i, (group, tag) in enumerate(
            [
                ("responsibility", "direct_flip"),
                ("coalition", "near_margin"),
                ("preservation", "coverage"),
            ]
        )
    )
    assignment = TeamSearchAssignment(
        target_member=2,
        parent_prompt="Use semantic evidence.",
        evidence=rows,
        optimization_context="legacy context must not select rows",
        responsibility_identity="resp-hash",
        primary_responsibility_lane="direct_flip",
        responsibility_value=8.0,
    )
    request = TeamSearchRequest(
        seed=80,
        update_index=1,
        team_state_hash="team-hash",
        local_metric_budget=24,
        solver_contract_id="COMMON_SOLVER_CONTRACT_V1",
        output_contract_id="output-v1",
        optimize_universe_id="optimize100-native-v1",
    )
    native = NativeFeedRequestBuilder().build(request, assignment)
    assert isinstance(native, NativeOptimizationRequest)
    assert native.target_member == 2
    assert native.responsibility.primary_lane == "direct_flip"
    assert native.optimize_universe_id == "optimize100-native-v1"
    names = {field.name for field in fields(native)}
    assert "search_examples" not in names
    assert "local_validation_examples" not in names
    serialized = repr(native)
    assert "private question" not in serialized
    assert "private feedback" not in serialized
    assert "private-" not in serialized


def test_layer2_contract_manifest_is_deterministic() -> None:
    assert layer2_contract_manifest(ROOT) == layer2_contract_manifest(ROOT)

