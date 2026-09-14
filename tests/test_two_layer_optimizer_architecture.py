from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.local_optimizers.base import NoOpContextProvider
from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import (
    GEPAAdapter,
    LocalSolverObservation,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPAOptimizerConfig, GEPALocalPromptOptimizer
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import GEPA_COMMIT, verify_frozen_gepa
from multi_dataset_diverse_rl.local_optimizers.legacy_tcs import LegacyTCSLocalOptimizer
from multi_dataset_diverse_rl.local_optimizers.registry import default_registry
from multi_dataset_diverse_rl.local_optimizers.schemas import (
    LocalEvidenceExample,
    LocalOptimizationResult,
    LocalOptimizationTask,
    LocalOptimizerBudget,
    LocalPromptCandidate,
)
from multi_dataset_diverse_rl.responsibility import MemberAwareRepairOpportunity
from multi_dataset_diverse_rl.team_search.candidate_evaluator import EvaluationCost
from multi_dataset_diverse_rl.team_search.controller import TeamSearchController
from multi_dataset_diverse_rl.team_search.primary_responsibility_binding import (
    PrimaryResponsibilityOnlineBinding,
)
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (
    DIRECT_FLIP,
    PrimaryResponsibilityPersistentRealizabilityScheduler,
)
from multi_dataset_diverse_rl.team_search.protocol import TeamSearchContract, TwoLayerProtocolIdentity
from multi_dataset_diverse_rl.team_search.schemas import (
    TeamEvidenceCase,
    TeamMiniBatchMetrics,
    TeamSearchAssignment,
    TeamSearchRequest,
)
from multi_dataset_diverse_rl.team_search.task_builder import LocalTaskBuilder
from multi_dataset_diverse_rl.versions import (
    COMMON_SOLVER_CONTRACT_V1_ID,
    LOCAL_GEPA_RESULT_SEMANTICS_VERSION,
)


ROOT = Path(__file__).parents[1]


def example(index: int) -> LocalEvidenceExample:
    return LocalEvidenceExample(
        example_id=f"e{index}",
        input_payload=f"Choose the semantically compatible referent in case {index}.",
        gold="A",
        tags=("coverage",),
    )


def task() -> LocalOptimizationTask:
    rows = tuple(example(index) for index in range(3))
    return LocalOptimizationTask(
        task_id="mock_gepa_lifecycle",
        parent_prompt="Use a generic decision procedure.",
        search_examples=rows,
        local_validation_examples=rows,
        optimization_context="Improve semantic disambiguation without copying examples.",
        solver_contract_id=COMMON_SOLVER_CONTRACT_V1_ID,
        output_contract_id="task_output_contract_v1",
        seed=7,
        budget=LocalOptimizerBudget(12, 3, 4),
    )


class FakeLocalEvaluator:
    solver_contract_id = COMMON_SOLVER_CONTRACT_V1_ID
    output_contract_id = "task_output_contract_v1"

    def evaluate(self, prompt: str, row: LocalEvidenceExample) -> LocalSolverObservation:
        correct = "distinguish" in prompt.casefold()
        return LocalSolverObservation(
            parsed_answer="A" if correct else "B",
            raw_output="answer",
            correct=correct,
            valid=True,
            input_tokens=2,
            output_tokens=1,
        )


class FakeReflection:
    def __init__(self) -> None:
        self.accounting = {"successful_calls": 0, "input_tokens": 0, "output_tokens": 0}

    def __call__(self, prompt):
        del prompt
        self.accounting["successful_calls"] += 1
        self.accounting["input_tokens"] += 10
        self.accounting["output_tokens"] += 5
        return "```Use semantic compatibility to distinguish candidate referents carefully.```"


def test_official_gepa_full_lifecycle_and_lineage(tmp_path: Path) -> None:
    reflection = FakeReflection()
    optimizer = GEPALocalPromptOptimizer(
        evaluator=FakeLocalEvaluator(),
        reflection_lm=reflection,
        accounting_reader=lambda: reflection.accounting,
        run_root=tmp_path,
    )
    result = asyncio.run(optimizer.optimize(task()))
    assert optimizer.GEPA_GROUNDED is True
    assert result.backend_name == "gepa"
    assert result.optimizer_calls == 1
    assert result.solver_calls == 12
    assert result.candidates
    assert all(row.local_rank_metadata["local_gepa_frontier"] for row in result.candidates)
    state = result.optimizer_state
    assert state is not None
    assert len(state.payload["gepa_result"]["candidates"]) >= 2
    assert state.payload["gepa_result"]["parents"][1] == [0]
    assert any(row["event_type"] == "candidate_accepted" for row in state.payload["callback_events"])
    assert (tmp_path / "mock_gepa_lifecycle.lineage.jsonl").is_file()


def test_local_gepa_does_not_return_unchanged_seed_candidate(tmp_path: Path) -> None:
    reflection = FakeReflection()

    class RootOnlyResult:
        per_val_instance_best_candidates = {index: {0} for index in range(3)}
        val_aggregate_scores = [1.0]
        candidates = [{"system_prompt": task().parent_prompt}]
        num_candidates = 1
        val_subscores = [{0: 1.0, 1: 1.0, 2: 1.0}]
        parents = [[None]]
        discovery_eval_counts = [3]

        @staticmethod
        def to_dict():
            return {
                "candidates": RootOnlyResult.candidates,
                "parents": RootOnlyResult.parents,
            }

    optimizer = GEPALocalPromptOptimizer(
        evaluator=FakeLocalEvaluator(),
        reflection_lm=reflection,
        accounting_reader=lambda: reflection.accounting,
        run_root=tmp_path,
        optimize_fn=lambda **_kwargs: RootOnlyResult(),
    )
    result = asyncio.run(optimizer.optimize(task()))
    assert result.candidates == ()
    assert result.termination_reason == "no_local_improvement"
    assert result.optimizer_state is not None
    assert result.optimizer_state.payload["local_gepa_frontier_indices"] == [0]
    assert result.optimizer_state.payload["changed_frontier_indices"] == []
    assert result.optimizer_state.payload["returned_candidate_indices"] == []
    assert result.optimizer_state.payload["result_semantics"] == "changed_candidates_only_v1"


def test_gepa_adapter_rejects_unsafe_prompt_before_solver_call() -> None:
    evaluator = FakeLocalEvaluator()
    adapter = GEPAAdapter(
        evaluator,
        parent_prompt="parent",
        all_examples=(example(0),),
        optimization_context="",
        output_contract_id="task_output_contract_v1",
    )
    result = adapter.evaluate(
        [example(0)], {"system_prompt": "Return FINAL_ANSWER: A"}, capture_traces=True
    )
    assert result.scores == [0.0]
    assert adapter.solver_calls == 0
    assert result.trajectories is not None
    assert result.trajectories[0].observation.provider_called is False


def test_frozen_official_gepa_dependency_identity() -> None:
    identity = verify_frozen_gepa()
    assert identity["version"] == "v0.1.1"
    assert identity["commit"] == GEPA_COMMIT


def test_registry_has_no_silent_future_backend_fallback() -> None:
    registry = default_registry()
    assert registry.registered == ("gepa", "legacy_tcs")
    with pytest.raises(NotImplementedError):
        registry.create("sepo")
    with pytest.raises(NotImplementedError):
        registry.create("espo")


def test_legacy_facade_preserves_result_bytes() -> None:
    expected = LocalOptimizationResult(
        candidates=(LocalPromptCandidate("c", "replacement", 1.0, {"e": 1.0}, (), 0),),
        backend_name="legacy_tcs",
        backend_version="v15",
        optimizer_state=None,
        solver_calls=2,
        optimizer_calls=3,
        input_tokens=7,
        output_tokens=5,
        total_tokens=12,
        termination_reason="complete",
    )

    async def bridge(_task):
        return expected

    actual = asyncio.run(LegacyTCSLocalOptimizer(bridge, backend_version="v15").optimize(task()))
    assert actual == expected


def test_quota_builder_and_optimize_only_governance() -> None:
    rows = tuple(
        TeamEvidenceCase(
            example_id=f"{group}-{index}",
            input_payload=f"payload {group} {index}",
            gold="A",
            target_output=None,
            feedback=None,
            evidence_group=group,
            tags=("direct_flip",) if group == "responsibility" else (),
        )
        for group in ("responsibility", "coalition", "preservation")
        for index in range(5)
    )
    assignment = TeamSearchAssignment(0, "parent", rows, "context", "r")
    request = TeamSearchRequest(1, 2, "team", 20, COMMON_SOLVER_CONTRACT_V1_ID, "output")
    builder = LocalTaskBuilder()
    built = builder.build(request, assignment)
    counts = {group: 0 for group in ("responsibility", "coalition", "preservation")}
    by_id = {row.example_id: row for row in rows}
    assert len(built.local_validation_examples) == 15
    for row in builder.select_team_minibatch(rows):
        counts[by_id[row.example_id].evidence_group] += 1
    assert counts == {"responsibility": 4, "coalition": 4, "preservation": 4}
    with pytest.raises(ValueError):
        TeamEvidenceCase("x", "payload", "A", None, None, "responsibility", (), "test")


class StubResponsibility:
    def __init__(self, assignment):
        self.assignment = assignment

    def assign(self, request):
        del request
        return self.assignment


class StubOptimizer:
    async def optimize(self, local_task):
        del local_task
        candidate = LocalPromptCandidate("stub", "replacement prompt", 1.0, {}, (), 0)
        return LocalOptimizationResult(
            (candidate,), "stub", "1", None, 1, 1, 3, 2, 5, "complete"
        )


class StubEvaluator:
    def __init__(self):
        self.shadow_targets = []

    def active_evaluation(self, assignment):
        del assignment
        return object()

    def evaluate_minibatch(self, assignment, candidate, minibatch):
        del assignment, candidate
        assert minibatch
        return TeamMiniBatchMetrics(target_delta=1), EvaluationCost(1, 2, 1)

    def evaluate_full(self, assignment, candidate):
        del assignment, candidate
        return object(), EvaluationCost(1, 2, 1)

    def evaluate_shadow(self, assignment, candidate):
        del candidate
        self.shadow_targets.append(assignment.target_member)
        return SimpleNamespace(passed=True), EvaluationCost(1, 2, 1)


class StubSelector:
    def annotate(self, records, *, active):
        del active
        return records

    def select(self, records):
        return next(row for row in records if row.promoted)


class StubCommitter:
    def __init__(self):
        self.ids = []

    def commit(self, *, assignment, candidate, evaluation):
        del assignment, evaluation
        self.ids.append(candidate.candidate_id)


def test_team_controller_is_backend_agnostic() -> None:
    evidence = tuple(
        TeamEvidenceCase(f"e{i}", f"payload {i}", "A", None, None, "responsibility", ())
        for i in range(3)
    )
    assignment = TeamSearchAssignment(0, "parent", evidence, "", "r")
    committer = StubCommitter()
    controller = TeamSearchController(
        responsibility=StubResponsibility(assignment),
        task_builder=LocalTaskBuilder(),
        local_optimizer=StubOptimizer(),
        evaluator=StubEvaluator(),
        selector=StubSelector(),
        committer=committer,
    )
    request = TeamSearchRequest(1, 0, "team", 10, COMMON_SOLVER_CONTRACT_V1_ID, "output")
    outcome = asyncio.run(controller.run_opportunity(request))
    assert outcome.committed_candidate_id == "stub"
    assert outcome.funnel["committed_candidates"] == 1
    assert committer.ids == ["stub"]


class StubPrimaryAssignmentFactory:
    def __init__(self, evidence):
        self.evidence = evidence

    def build(self, *, request, summary):
        del request
        return TeamSearchAssignment(
            summary.member_id,
            f"parent-{summary.member_id}",
            self.evidence,
            "context",
            f"responsibility-{summary.member_id}",
            primary_responsibility_lane=summary.primary_lane,
        )


def test_primary_binding_runs_two_frozen_branches_and_records_once() -> None:
    evidence = tuple(
        TeamEvidenceCase(
            f"e{i}", f"payload {i}", "A", None, None, "responsibility", (DIRECT_FLIP,)
        )
        for i in range(3)
    )
    evaluator = StubEvaluator()
    committer = StubCommitter()
    controller = TeamSearchController(
        responsibility=StubResponsibility(None),
        task_builder=LocalTaskBuilder(),
        local_optimizer=StubOptimizer(),
        evaluator=evaluator,
        selector=StubSelector(),
        committer=committer,
    )
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler()
    binding = PrimaryResponsibilityOnlineBinding(
        scheduler=scheduler,
        assignment_factory=StubPrimaryAssignmentFactory(evidence),
        controller=controller,
    )
    request = TeamSearchRequest(1, 0, "team", 10, COMMON_SOLVER_CONTRACT_V1_ID, "output")
    opportunities = {
        member: (
            MemberAwareRepairOpportunity(
                agent_id=member,
                question_hash=f"q-{member}",
                vote_flip_gain=1,
                margin_gain=0,
                member_error=True,
                coverage_opportunity=False,
                conversion_opportunity=False,
                dominant_wrong_member=False,
                unique_correct=False,
                pivotal_correct=False,
                oracle_soft_utility_gain=0.0,
            ),
        )
        for member in range(5)
    }
    outcome = asyncio.run(
        binding.run_opportunity(
            request,
            assigned=opportunities,
            current_margin_by_question={f"q-{member}": -1 for member in range(5)},
        )
    )
    selected = outcome.decision.selected_member_ids
    assert len(selected) == 2
    assert outcome.team_outcome.funnel["target_branches"] == 2
    assert outcome.team_outcome.audit_metadata["selected_target_ids"] == selected
    assert outcome.team_outcome.audit_metadata["committed_member_id"] == selected[0]
    assert evaluator.shadow_targets == [selected[0]]
    assert committer.ids == ["stub"]
    assert scheduler.state.target_count_by_member[selected[0]] == 1
    assert scheduler.state.commit_count_by_member[selected[0]] == 1
    assert scheduler.state.failure_count_by_member[selected[0]] == 0
    assert scheduler.state.failure_count_by_member[selected[1]] == 1
    assert scheduler.state.completed_update_indices == {0}


class FailingFrozenController:
    async def run_frozen_opportunity(self, request, assignments):
        del request, assignments
        raise RuntimeError("synthetic operational abort")


class NoWinnerSelector(StubSelector):
    def select(self, records):
        del records
        return None


def test_primary_binding_records_valid_no_commit_for_both_targets() -> None:
    evidence = tuple(
        TeamEvidenceCase(
            f"e{i}", f"payload {i}", "A", None, None, "responsibility", (DIRECT_FLIP,)
        )
        for i in range(3)
    )
    controller = TeamSearchController(
        responsibility=StubResponsibility(None),
        task_builder=LocalTaskBuilder(),
        local_optimizer=StubOptimizer(),
        evaluator=StubEvaluator(),
        selector=NoWinnerSelector(),
        committer=StubCommitter(),
    )
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler()
    binding = PrimaryResponsibilityOnlineBinding(
        scheduler=scheduler,
        assignment_factory=StubPrimaryAssignmentFactory(evidence),
        controller=controller,
    )
    request = TeamSearchRequest(3, 1, "team", 10, COMMON_SOLVER_CONTRACT_V1_ID, "output")
    opportunities = {
        member: (
            MemberAwareRepairOpportunity(
                agent_id=member,
                question_hash=f"no-commit-{member}",
                vote_flip_gain=1,
                margin_gain=0,
                member_error=True,
                coverage_opportunity=False,
                conversion_opportunity=False,
                dominant_wrong_member=False,
                unique_correct=False,
                pivotal_correct=False,
                oracle_soft_utility_gain=0.0,
            ),
        )
        for member in range(5)
    }
    result = asyncio.run(
        binding.run_opportunity(
            request,
            assigned=opportunities,
            current_margin_by_question={f"no-commit-{member}": -1 for member in range(5)},
        )
    )
    assert result.team_outcome.committed_candidate_id is None
    assert result.team_outcome.cost.team_shadow_solver_calls == 0
    assert all(
        scheduler.state.failure_count_by_member[member] == 1
        for member in result.decision.selected_member_ids
    )
    assert scheduler.state.completed_update_indices == {1}


def test_primary_binding_records_operational_abort_without_counter_change() -> None:
    evidence = (
        TeamEvidenceCase("e", "payload", "A", None, None, "preservation", ()),
    )
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler()
    binding = PrimaryResponsibilityOnlineBinding(
        scheduler=scheduler,
        assignment_factory=StubPrimaryAssignmentFactory(evidence),
        controller=FailingFrozenController(),  # type: ignore[arg-type]
    )
    request = TeamSearchRequest(2, 4, "team", 10, COMMON_SOLVER_CONTRACT_V1_ID, "output")
    before = dict(scheduler.state.failure_count_by_member)
    with pytest.raises(RuntimeError, match="synthetic"):
        asyncio.run(
            binding.run_opportunity(
                request,
                assigned={},
                current_margin_by_question={},
            )
        )
    assert scheduler.state.failure_count_by_member == before
    assert scheduler.state.completed_update_indices == {4}
    with pytest.raises(ValueError, match="already recorded"):
        asyncio.run(
            binding.run_opportunity(
                request,
                assigned={},
                current_margin_by_question={},
            )
        )


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_import_boundaries() -> None:
    local_root = ROOT / "multi_dataset_diverse_rl" / "local_optimizers"
    team_root = ROOT / "multi_dataset_diverse_rl" / "team_search"
    local_forbidden = ("responsibility", "vote_aligned_scheduler", "shadow_gate", "system")
    for path in local_root.glob("*.py"):
        imports = _imports(path)
        assert not any(any(token in name for token in local_forbidden) for name in imports), path
    for path in team_root.glob("*.py"):
        imports = _imports(path)
        assert not any(name == "gepa" or name.startswith("gepa.") for name in imports), path


def test_layered_protocol_hashes_are_independent() -> None:
    local_a = GEPAOptimizerConfig().identity()
    local_b = "future-sepo-hash"
    team = TeamSearchContract().identity()
    a = TwoLayerProtocolIdentity("solver", "data", local_a, team)
    b = TwoLayerProtocolIdentity("solver", "data", local_b, team)
    assert a.team_search_contract_hash == b.team_search_contract_hash
    assert a.local_optimizer_contract_hash != b.local_optimizer_contract_hash
    assert a.full_protocol_hash != b.full_protocol_hash
    assert NoOpContextProvider().build_context(task_id="x") == ""
    assert GEPAOptimizerConfig().result_semantics == LOCAL_GEPA_RESULT_SEMANTICS_VERSION
