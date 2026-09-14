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
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import (
    GEPAOptimizerConfig,
    GEPALocalPromptOptimizer,
    local_gepa_budget_capacity,
    verify_frozen_gepa_engine_contract,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_proposer_contract import (
    DECISION_PROCEDURE_REFLECTION_TEMPLATE,
    DECISION_PROCEDURE_REFLECTION_TEMPLATE_SHA256,
)
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
    LOCAL_GEPA_CANDIDATE_COMPONENT,
    LOCAL_GEPA_RESULT_SEMANTICS_VERSION,
    LOCAL_OPTIMIZER_FIDELITY_LEVEL,
    METHOD_VERSION,
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

    def __init__(self) -> None:
        self.calls = 0
        self.procedures: list[str] = []

    def evaluate(
        self, decision_procedure: str, row: LocalEvidenceExample
    ) -> LocalSolverObservation:
        self.calls += 1
        self.procedures.append(decision_procedure)
        correct = "distinguish" in decision_procedure.casefold()
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
    assert state.payload["telemetry"]["proposal_attempts"] >= 1
    assert state.payload["telemetry"]["positive_minibatch_deltas"] >= 1
    assert state.payload["telemetry"]["accepted_mutations"] >= 1
    assert state.payload["telemetry"]["full_local_evaluations"] >= 1
    assert (tmp_path / "mock_gepa_lifecycle.lineage.jsonl").is_file()


def test_gepa_contract_freezes_real_engine_controls_and_budget_arithmetic() -> None:
    config = GEPAOptimizerConfig()
    assert not hasattr(config, "acceptance_criterion")
    assert config.optimizer_fidelity_level == "LEVEL_B_API_COMPATIBLE_ADAPTATION"
    assert config.optimizer_fidelity_level == LOCAL_OPTIMIZER_FIDELITY_LEVEL
    assert config.candidate_component_name == LOCAL_GEPA_CANDIDATE_COMPONENT == "decision_procedure"
    assert config.engine_acceptance_semantics == "pinned_gepa_v0.1.1_strict_improvement"
    assert config.skip_perfect_score is True
    assert config.perfect_score == 1.0
    assert config.batch_sampler == "epoch_shuffled"
    assert config.val_evaluation_policy == "full_eval"
    assert config.max_merge_invocations == 5
    assert config.merge_val_overlap_floor == 5
    assert config.max_prompt_chars == 3000
    assert config.proposer_contract_version == "decision_procedure_proposer_v1"
    assert config.reflection_prompt_template_sha256 == DECISION_PROCEDURE_REFLECTION_TEMPLATE_SHA256
    assert DECISION_PROCEDURE_REFLECTION_TEMPLATE.count("<curr_param>") == 1
    assert DECISION_PROCEDURE_REFLECTION_TEMPLATE.count("<side_info>") == 1
    verify_frozen_gepa_engine_contract()
    capacity = local_gepa_budget_capacity(
        metric_budget=36, validation_size=12, reflection_minibatch_size=3
    )
    assert capacity.seed_evaluation_calls == 12
    assert capacity.proposal_attempt_calls == 6
    assert capacity.accepted_full_evaluation_calls == 12
    assert capacity.max_rejected_proposals == 4
    assert capacity.max_accepted_children == capacity.max_accepted_generations == 1


def test_parent_and_weight_contracts_fail_before_optimizer_entry(tmp_path: Path) -> None:
    called = False

    def forbidden_optimize(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("optimizer must not start")

    reflection = FakeReflection()
    optimizer = GEPALocalPromptOptimizer(
        evaluator=FakeLocalEvaluator(),
        reflection_lm=reflection,
        accounting_reader=lambda: reflection.accounting,
        run_root=tmp_path,
        optimize_fn=forbidden_optimize,
    )
    unsafe = LocalOptimizationTask(
        **{**task().__dict__, "task_id": "unsafe-parent", "parent_prompt": "Return FINAL_ANSWER: A"}
    )
    with pytest.raises(ValueError, match="PARENT_DECISION_PROCEDURE_CONTRACT_VIOLATION"):
        asyncio.run(optimizer.optimize(unsafe))
    weighted_rows = tuple(
        LocalEvidenceExample(**{**row.__dict__, "weight": 2.0}) for row in task().search_examples
    )
    weighted = LocalOptimizationTask(
        **{**task().__dict__, "task_id": "weighted", "search_examples": weighted_rows,
           "local_validation_examples": weighted_rows}
    )
    with pytest.raises(ValueError, match="unit-weight"):
        asyncio.run(optimizer.optimize(weighted))
    assert called is False
    assert reflection.accounting["successful_calls"] == 0


def test_cross_split_identity_requires_exact_content() -> None:
    search = (example(0),)
    validation = (
        LocalEvidenceExample("e0", "different payload", "A", tags=("coverage",)),
    )
    with pytest.raises(ValueError, match="cross-split"):
        LocalOptimizationTask(
            task_id="mismatch",
            parent_prompt="Use a general decision procedure.",
            search_examples=search,
            local_validation_examples=validation,
            optimization_context="",
            solver_contract_id=COMMON_SOLVER_CONTRACT_V1_ID,
            output_contract_id="task_output_contract_v1",
            seed=1,
            budget=LocalOptimizerBudget(12, 3, 4),
        )


def test_changed_frontier_is_validated_and_deduplicated_before_top_k(tmp_path: Path) -> None:
    parent = task().parent_prompt
    valid_a = "Use semantic compatibility to distinguish referents and resolve ambiguity."
    valid_b = "Compare the candidate interpretations and choose the coherent referent."
    prompts = [
        parent,
        "Return FINAL_ANSWER: A",
        valid_a,
        valid_a,
        "x" * 3001,
        valid_b,
    ]

    class Result:
        per_val_instance_best_candidates = {index: set(range(6)) for index in range(3)}
        val_aggregate_scores = [0.0, 10.0, 9.0, 8.0, 7.0, 6.0]
        candidates = [{"decision_procedure": prompt} for prompt in prompts]
        num_candidates = 6
        val_subscores = [{0: 0.0, 1: 0.0, 2: 0.0} for _ in prompts]
        parents = [[None], [0], [0], [0], [0], [0]]
        discovery_eval_counts = [3] * 6

        @staticmethod
        def to_dict():
            return {"candidate_count": 6}

    captured: dict[str, object] = {}

    def optimize(**kwargs):
        captured.update(kwargs)
        return Result()

    reflection = FakeReflection()
    optimizer = GEPALocalPromptOptimizer(
        evaluator=FakeLocalEvaluator(),
        reflection_lm=reflection,
        accounting_reader=lambda: reflection.accounting,
        run_root=tmp_path,
        optimize_fn=optimize,
    )
    result = asyncio.run(optimizer.optimize(task()))
    assert [row.prompt for row in result.candidates] == [valid_a, valid_b]
    assert result.optimizer_state is not None
    state = result.optimizer_state.payload
    assert state["invalid_prompt_indices"] == [1, 4]
    assert state["duplicate_prompt_indices"] == [3]
    assert state["returned_candidate_indices"] == [2, 5]
    assert captured["reflection_prompt_template"] == DECISION_PROCEDURE_REFLECTION_TEMPLATE
    assert captured["skip_perfect_score"] is True
    assert captured["perfect_score"] == 1.0
    assert captured["batch_sampler"] == "epoch_shuffled"
    assert captured["val_evaluation_policy"] == "full_eval"
    assert captured["use_merge"] is False
    assert captured["max_merge_invocations"] == 5
    assert captured["merge_val_overlap_floor"] == 5
    assert captured["custom_candidate_proposer"] is None
    assert captured["seed_candidate"] == {"decision_procedure": task().parent_prompt}


def test_local_gepa_does_not_return_unchanged_seed_candidate(tmp_path: Path) -> None:
    reflection = FakeReflection()

    class RootOnlyResult:
        per_val_instance_best_candidates = {index: {0} for index in range(3)}
        val_aggregate_scores = [1.0]
        candidates = [{"decision_procedure": task().parent_prompt}]
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


def test_decision_procedure_is_sole_gepa_component() -> None:
    evaluator = FakeLocalEvaluator()
    adapter = GEPAAdapter(
        evaluator,
        parent_prompt="parent",
        all_examples=(example(0),),
        optimization_context="",
        output_contract_id="task_output_contract_v1",
    )
    with pytest.raises(ValueError, match="exactly one decision_procedure"):
        adapter.evaluate([example(0)], {"system_prompt": "legacy"})
    assert adapter.propose_new_texts is None
    valid = "Use semantic compatibility to distinguish referents carefully."
    result = adapter.evaluate([example(0)], {"decision_procedure": valid}, capture_traces=True)
    assert result.scores == [1.0]
    assert evaluator.calls == 1
    assert evaluator.procedures == [valid]
    assert adapter.make_reflective_dataset(
        {"decision_procedure": valid}, result, ["decision_procedure"]
    ).keys() == {"decision_procedure"}


@pytest.mark.parametrize(
    "candidate,examples",
    [
        ("Return FINAL_ANSWER: A", (example(0),)),
        ("Use a general decision procedure.\nAdd a patch.", (example(0),)),
        ("x" * 3001, (example(0),)),
        (
            "First compare the deeply ambiguous candidate referents using contextual grammar clues.",
            (
                LocalEvidenceExample(
                    "copy",
                    "First compare the deeply ambiguous candidate referents using contextual grammar clues before choosing any option.",
                    "A",
                ),
            ),
        ),
    ],
)
def test_invalid_decision_procedure_never_reaches_solver(candidate, examples) -> None:
    evaluator = FakeLocalEvaluator()
    parent = "Use a general decision procedure."
    adapter = GEPAAdapter(
        evaluator,
        parent_prompt=parent,
        all_examples=examples,
        optimization_context="",
        output_contract_id="task_output_contract_v1",
    )
    result = adapter.evaluate(
        list(examples), {"decision_procedure": candidate}, capture_traces=True
    )
    assert result.scores == [0.0]
    assert adapter.solver_calls == 0
    assert evaluator.calls == 0
    assert result.trajectories is not None
    assert result.trajectories[0].observation.provider_called is False


def test_frozen_official_gepa_dependency_identity() -> None:
    identity = verify_frozen_gepa()
    assert identity["version"] == "v0.1.1"
    assert identity["commit"] == GEPA_COMMIT
    assert identity["source_sha256"] == "84c3c7e5f80fd272f0841357ec9327e3b0ea8ee53cd8107d1ab8d4cdb36ff1f8"


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
    assert builder.team_minibatch_telemetry(builder.select_team_minibatch(rows)) == {
        "contract_version": "primary_lane_strict_4_4_4_v1",
        "total_count": 12,
        "responsibility_count": 4,
        "coalition_count": 4,
        "preservation_count": 4,
        "backfill_count": 0,
    }
    with pytest.raises(ValueError, match="exactly 4 unique preservation"):
        builder.select_team_minibatch(
            tuple(row for row in rows if row.example_id not in {"preservation-3", "preservation-4"})
        )
    with pytest.raises(ValueError):
        TeamEvidenceCase("x", "payload", "A", None, None, "responsibility", (), "test")


def test_team_minibatch_responsibility_quota_matches_primary_lane() -> None:
    responsibility = tuple(
        TeamEvidenceCase(
            f"{lane}-{index}", f"payload {lane} {index}", "A", None, None,
            "responsibility", (lane,),
        )
        for lane in ("direct_flip", "near_margin")
        for index in range(4)
    )
    global_rows = tuple(
        TeamEvidenceCase(
            f"{group}-{index}", f"payload {group} {index}", "A", None, None,
            group, (),
        )
        for group in ("coalition", "preservation")
        for index in range(4)
    )
    selected = LocalTaskBuilder().select_team_minibatch(
        responsibility + global_rows,
        primary_responsibility_lane="near_margin",
    )
    responsibility_rows = [row for row in selected if row.evidence_group == "responsibility"]
    assert len(selected) == len({row.example_id for row in selected}) == 12
    assert len(responsibility_rows) == 4
    assert all("near_margin" in row.tags for row in responsibility_rows)


class StubResponsibility:
    def __init__(self, assignment):
        self.assignment = assignment

    def assign(self, request):
        del request
        return self.assignment


def strict_team_evidence(lane: str = DIRECT_FLIP) -> tuple[TeamEvidenceCase, ...]:
    return tuple(
        TeamEvidenceCase(
            f"{group}-{index}",
            f"payload {group} {index}",
            "A",
            None,
            None,
            group,
            (lane,) if group == "responsibility" else (),
        )
        for group in ("responsibility", "coalition", "preservation")
        for index in range(4)
    )


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
    evidence = strict_team_evidence()
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
    assert outcome.audit_metadata["team_minibatch"] == {
        "contract_version": "primary_lane_strict_4_4_4_v1",
        "total_count": 12,
        "responsibility_count": 4,
        "coalition_count": 4,
        "preservation_count": 4,
        "backfill_count": 0,
    }


def test_fake_provider_end_to_end_positive_path_commits_once(tmp_path: Path) -> None:
    evidence = strict_team_evidence("near_margin")
    assignment = TeamSearchAssignment(
        0,
        "Use a general compatibility procedure.",
        evidence,
        "Improve the general reasoning procedure.",
        "synthetic-near-margin",
        primary_responsibility_lane="near_margin",
    )
    reflection = FakeReflection()
    local_optimizer = GEPALocalPromptOptimizer(
        evaluator=FakeLocalEvaluator(),
        reflection_lm=reflection,
        accounting_reader=lambda: reflection.accounting,
        run_root=tmp_path,
    )
    evaluator = StubEvaluator()
    committer = StubCommitter()
    controller = TeamSearchController(
        responsibility=StubResponsibility(assignment),
        task_builder=LocalTaskBuilder(),
        local_optimizer=local_optimizer,
        evaluator=evaluator,
        selector=StubSelector(),
        committer=committer,
    )
    request = TeamSearchRequest(
        78, 0, "synthetic-team", 36, COMMON_SOLVER_CONTRACT_V1_ID,
        "task_output_contract_v1",
    )
    outcome = asyncio.run(controller.run_opportunity(request))
    assert outcome.cost.local_optimizer_solver_calls > 0
    assert outcome.cost.local_optimizer_meta_calls > 0
    assert outcome.funnel["local_candidates"] >= 1
    assert outcome.funnel["team_minibatch_survivors"] >= 1
    assert outcome.funnel["full_team_evaluated_candidates"] >= 1
    assert outcome.funnel["committed_candidates"] == 1
    assert len(committer.ids) == 1
    assert evaluator.shadow_targets == [0]
    assert all(
        candidate.local_candidate.prompt != assignment.parent_prompt
        for candidate in outcome.candidates
    )
    minibatch_ids = outcome.audit_metadata["team_minibatch_example_ids"]
    assert len(minibatch_ids) == len(set(minibatch_ids)) == 12
    assert set(minibatch_ids[:4]) == {f"responsibility-{index}" for index in range(4)}
    assert outcome.audit_metadata["primary_responsibility_lane"] == "near_margin"
    telemetry = outcome.audit_metadata["local_optimizer_telemetry"]
    assert telemetry["accepted_mutations"] >= 1
    assert telemetry["positive_minibatch_deltas"] >= 1
    assert telemetry["full_local_evaluations"] >= 1


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
    evidence = strict_team_evidence()
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
    evidence = strict_team_evidence()
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
        source = path.read_text(encoding="utf-8")
        assert not any(
            symbol in source
            for symbol in (
                "GEPAEngine", "ParetoCandidateSelector", "ReflectiveMutationProposer",
                "GEPAState", "candidate_parent_selector",
            )
        ), path
    forbidden_layer2_symbols = (
        "CommonSafe", "ShadowGate", "PersistentRealizability",
        "PrimaryResponsibility", "TeamCandidateSelector", "TeamCommitter",
    )
    for path in local_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(symbol in source for symbol in forbidden_layer2_symbols), path


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
    assert METHOD_VERSION == "member_aware_peer_state_v15"
