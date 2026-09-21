from __future__ import annotations

import asyncio
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from multi_dataset_diverse_rl.local_optimizers.schemas import (
    LocalOptimizationResult,
    LocalPromptCandidate,
)
from multi_dataset_diverse_rl.native_feed import NativeOptimizationRequest
from multi_dataset_diverse_rl.native_feed_audit import layer2_contract_manifest
from multi_dataset_diverse_rl.team_search.candidate_evaluator import EvaluationCost
from multi_dataset_diverse_rl.team_search.controller import TeamSearchController
from multi_dataset_diverse_rl.team_search.schemas import (
    TeamEvidenceCase,
    TeamMiniBatchMetrics,
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
    assert native.responsibility.primary_lane == "generic"
    assert native.responsibility.responsibility_value == 0.0
    assert native.responsibility.team_failure_summary == {}
    assert native.responsibility.coverage_summary == {}
    assert native.provenance["responsibility_semantics"] == "absent_native_control"
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


class _Responsibility:
    def __init__(self, assignment: TeamSearchAssignment) -> None:
        self.assignment = assignment

    def assign(self, _request: TeamSearchRequest) -> TeamSearchAssignment:
        return self.assignment


class _NativeOptimizer:
    def __init__(self) -> None:
        self.requests: list[NativeOptimizationRequest] = []

    async def optimize_native(
        self, request: NativeOptimizationRequest
    ) -> LocalOptimizationResult:
        self.requests.append(request)
        candidate = LocalPromptCandidate(
            "native-candidate", "Use semantic compatibility.", 1.0, {}, (), 1
        )
        return LocalOptimizationResult(
            (candidate,), "native-fixture", "v1", None, 2, 1, 3, 2, 5, "complete"
        )


class _TeamEvaluator:
    def active_evaluation(self, _assignment):
        return object()

    def evaluate_minibatch(self, _assignment, _candidate, minibatch):
        assert len(minibatch) == 12
        return TeamMiniBatchMetrics(target_delta=1), EvaluationCost(1, 1, 1)

    def evaluate_full(self, _assignment, _candidate):
        return object(), EvaluationCost(1, 1, 1)

    def evaluate_shadow(self, _assignment, _candidate):
        return SimpleNamespace(passed=True), EvaluationCost(1, 1, 1)


class _Selector:
    def annotate(self, records, *, active):
        del active
        return records

    def select(self, records):
        return next((row for row in records if row.promoted), None)


class _Committer:
    def __init__(self) -> None:
        self.candidate_ids: list[str] = []

    def commit(self, *, assignment, candidate, evaluation) -> None:
        del assignment, evaluation
        self.candidate_ids.append(candidate.candidate_id)


def test_team_controller_consumes_native_feed_candidate_end_to_end() -> None:
    evidence = tuple(
        _evidence(index + group_index * 4, group, "direct_flip")
        for group_index, group in enumerate(
            ("responsibility", "coalition", "preservation")
        )
        for index in range(4)
    )
    assignment = TeamSearchAssignment(
        2,
        "Use semantic evidence.",
        evidence,
        "",
        "resp-hash",
        primary_responsibility_lane="direct_flip",
        responsibility_value=8.0,
    )
    optimizer = _NativeOptimizer()
    committer = _Committer()
    controller = TeamSearchController(
        responsibility=_Responsibility(assignment),
        task_builder=NativeFeedRequestBuilder(),
        local_optimizer=optimizer,
        evaluator=_TeamEvaluator(),
        selector=_Selector(),
        committer=committer,
    )
    request = TeamSearchRequest(
        80,
        1,
        "team-hash",
        24,
        "COMMON_SOLVER_CONTRACT_V1",
        "output-v1",
        "optimize100-native-v1",
    )
    outcome = asyncio.run(controller.run_opportunity(request))
    assert len(optimizer.requests) == 1
    assert optimizer.requests[0].optimize_universe_id == "optimize100-native-v1"
    assert outcome.committed_candidate_id == "native-candidate"
    assert outcome.funnel == {
        "target_branches": 1,
        "local_candidates": 1,
        "team_minibatch_survivors": 1,
        "full_team_evaluated_candidates": 1,
        "feasible_candidates": 0,
        "committed_candidates": 1,
    }
    assert committer.candidate_ids == ["native-candidate"]

