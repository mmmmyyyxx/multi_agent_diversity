"""Current BBH one-opportunity semantic replay across the control-flow move."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

from multi_dataset_diverse_rl.local_optimizers.schemas import (
    LocalOptimizationResult, LocalPromptCandidate,
)
from multi_dataset_diverse_rl.local_optimizers.base import LocalSolverObservation
from multi_dataset_diverse_rl.local_optimizers.gepa_native import GEPALayer2EvidenceOptimizer
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPALocalPromptOptimizer
from multi_dataset_diverse_rl.search.aggregation import PluralityAggregation
from multi_dataset_diverse_rl.search.current_bbh import (
    CurrentBBHEvidenceSource, CurrentBBHStateSource, PluralityResponsibilityAnalyzer,
)
from multi_dataset_diverse_rl.search.current_bbh_runtime import (
    CurrentBBHAssignmentAdapter, build_current_bbh_orchestrator,
)
from multi_dataset_diverse_rl.search.engines import GEPASearchEngine
from multi_dataset_diverse_rl.search.gepa import (
    CurrentGEPABridge, CurrentGEPAPacketAdapter, GEPADerivedConfig,
)
from multi_dataset_diverse_rl.search.evaluation import CandidateEvaluationPipeline
from multi_dataset_diverse_rl.search.history import NullMemoryProvider
from multi_dataset_diverse_rl.search.history import HistoryState
from multi_dataset_diverse_rl.search.legacy_bbh_replay import (
    FrozenBBHCommitter, FrozenBBHGate, FrozenBBHGEPARequestBridge,
    FrozenBBHOpportunityBuilder, FrozenBBHPromotion,
    FrozenBBHTeamEvaluationProvider, FrozenBBHTransitionPolicy,
)
from multi_dataset_diverse_rl.search.orchestrator import (
    OpportunityBuilder, StateAnalyzer, UnifiedSearchOrchestrator,
)
from multi_dataset_diverse_rl.search.policies import (
    CurrentEvidenceFeasibility, CurrentRoleEvidencePolicy, GlobalStopPolicy,
    TargetPolicyV1,
)
from multi_dataset_diverse_rl.search.schemas import (
    BenchmarkCapabilities, Diagnosis, SearchMethodConfig, TeamStateSnapshot,
)
from multi_dataset_diverse_rl.team_search.candidate_evaluator import EvaluationCost
from multi_dataset_diverse_rl.team_search.candidate_selector import CommonSafeTeamCandidateSelector
from multi_dataset_diverse_rl.team_search.controller import TeamSearchController
from multi_dataset_diverse_rl.team_search.schemas import (
    TeamEvidenceCase, TeamMiniBatchMetrics, TeamSearchAssignment,
    TeamSearchRequest,
)
from multi_dataset_diverse_rl.team_search.task_builder import Layer2EvidenceRequestBuilder
from multi_dataset_diverse_rl.team_search.system_runtime import (
    SystemResponsibilityAssignmentFactory, freeze_current_responsibility,
)
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (
    PrimaryResponsibilityPersistentRealizabilityScheduler,
    build_primary_responsibility_summaries,
)
from multi_dataset_diverse_rl.team_search.v4_opportunity import select_v4_feasible_opportunity
from multi_dataset_diverse_rl.peer_state import build_peer_vote_context, build_team_vote_state
from multi_dataset_diverse_rl.responsibility import (
    ResponsibilityState, compute_member_aware_repair_opportunity,
)
from multi_dataset_diverse_rl.versions import PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION
from multi_dataset_diverse_rl.saturation import SaturationConfig
from multi_dataset_diverse_rl.member_objectives import member_gain_metrics
from multi_dataset_diverse_rl.native_feed import CandidateTransitionAudit


class FakeBenchmark:
    capabilities = BenchmarkCapabilities(True, True, True)


class FrozenResponsibility:
    def __init__(self, assignment):
        self.assignment = assignment

    def assign(self, request):
        del request
        return self.assignment


class FakeLocalOptimizer:
    def __init__(self):
        self.packet_hashes = []
        self.local_eval_ids = []

    async def optimize_layer2(self, task):
        self.packet_hashes.append(task.packet.packet_hash)
        self.local_eval_ids.append(tuple(row.example_id for row in task.packet.local_eval_examples))
        row = LocalPromptCandidate(
            "child", "better procedure", 0.9, {}, (), 1,
            backend_metadata={"local_acceptance_delta": 0.1},
        )
        return LocalOptimizationResult(
            (row,), "gepa", "fake", None, 0, 0, 0, 0, 0, "SATURATION_REACHED",
        )


class FakeTeamEvaluator:
    def __init__(self, *, shadow_passed):
        self.events = []
        self.shadow_passed = shadow_passed

    def active_evaluation(self, assignment):
        del assignment
        return "parent_eval"

    def evaluate_minibatch(self, assignment, candidate, minibatch):
        del assignment
        self.events.append(("team_probe", candidate.candidate_id,
                            tuple(row.example_id for row in minibatch)))
        return TeamMiniBatchMetrics(target_delta=1), EvaluationCost(0, 0, 0)

    def evaluate_full(self, assignment, candidate):
        del assignment
        self.events.append(("full", candidate.candidate_id))
        return "child_full", EvaluationCost(0, 0, 0)

    def evaluate_shadow(self, assignment, candidate):
        del assignment
        self.events.append(("shadow", candidate.candidate_id))
        return SimpleNamespace(passed=self.shadow_passed), EvaluationCost(0, 0, 0)


class FakeSelector:
    def annotate(self, records, *, active):
        assert active == "parent_eval"
        return records

    def select(self, records):
        return next((row for row in records if row.promoted), None)


class FakeTeamState:
    def __init__(self, prompt="parent"):
        self.prompts = (prompt,) * 5

    def snapshot(self):
        return TeamStateSnapshot(hashlib.sha256(repr(self.prompts).encode()).hexdigest(),
                                 self.prompts)


class FakeCommitter:
    def __init__(self, state):
        self.state = state

    def commit(self, *, assignment, candidate, evaluation):
        assert evaluation == "child_full"
        self.state.prompts = tuple(candidate.prompt if i == assignment.target_member
                                   else prompt for i, prompt in enumerate(self.state.prompts))


class FakeDiagnosis:
    def analyze(self, state, history):
        del state, history
        return Diagnosis()


def _assignment_and_request(parent_id, parent_prompt="parent"):
    rows = []
    for group, count in (("repair", 8), ("preservation", 4)):
        for index in range(count):
            tags = ("direct_flip", "team_hard") if group == "repair" else ()
            rows.append(TeamEvidenceCase(
                f"{group}-{index}", f"input {index}", "A", "B", "feedback",
                group, tags,
            ))
    assignment = TeamSearchAssignment(
        0, parent_prompt, tuple(rows), "context", "raw-v4",
        primary_responsibility_lane="direct_flip", responsibility_value=32,
    )
    builder = Layer2EvidenceRequestBuilder(bounded_search_view=True)
    minibatch = builder.select_team_minibatch(
        assignment.evidence, primary_responsibility_lane="direct_flip",
    )
    assignment = TeamSearchAssignment(
        0, parent_prompt, tuple(rows), "context", "raw-v4",
        local_validation_example_ids=tuple(row.example_id for row in minibatch),
        primary_responsibility_lane="direct_flip", responsibility_value=32,
    )
    request = TeamSearchRequest(81, 0, parent_id, 36, "COMMON_SOLVER_CONTRACT_V1", "output")
    return assignment, request, builder


def _old_path(shadow_passed):
    state = FakeTeamState()
    assignment, request, builder = _assignment_and_request(state.snapshot().team_state_id)
    optimizer = FakeLocalOptimizer()
    evaluator = FakeTeamEvaluator(shadow_passed=shadow_passed)
    controller = TeamSearchController(
        responsibility=FrozenResponsibility(assignment), task_builder=builder,
        local_optimizer=optimizer, evaluator=evaluator,
        selector=FakeSelector(), committer=FakeCommitter(state),
    )
    outcome = asyncio.run(controller.run_opportunity(request))
    return state, assignment, optimizer, evaluator, outcome


def _new_path(shadow_passed):
    state = FakeTeamState()
    assignment, request, builder = _assignment_and_request(state.snapshot().team_state_id)
    frozen = FrozenBBHOpportunityBuilder(request, assignment, builder)
    optimizer = FakeLocalOptimizer()
    bridge = FrozenBBHGEPARequestBridge(frozen, optimizer)
    evaluator = FakeTeamEvaluator(shadow_passed=shadow_passed)
    provider = FrozenBBHTeamEvaluationProvider(frozen, bridge, evaluator)
    orchestrator = UnifiedSearchOrchestrator(
        method=SearchMethodConfig(), benchmark=FakeBenchmark(),
        aggregation=PluralityAggregation(), state=state,
        analyzer=StateAnalyzer(FakeDiagnosis()), opportunities=frozen,
        engine=GEPASearchEngine(bridge),
        evaluation=CandidateEvaluationPipeline(provider, FrozenBBHPromotion()),
        transition=FrozenBBHTransitionPolicy(bridge, FakeSelector()),
        gate=FrozenBBHGate(frozen, bridge, evaluator),
        committer=FrozenBBHCommitter(frozen, bridge, FakeCommitter(state), state),
        memory=NullMemoryProvider(), stop=GlobalStopPolicy(),
    )
    result = asyncio.run(orchestrator.run(max_opportunities=1))
    return state, assignment, optimizer, evaluator, result


def test_bbh_v4_one_opportunity_old_new_golden_commit_and_shadow_reject():
    for shadow_passed in (True, False):
        old_state, old_assignment, old_optimizer, old_eval, old = _old_path(shadow_passed)
        new_state, new_assignment, new_optimizer, new_eval, new = _new_path(shadow_passed)
        assert old_assignment == new_assignment
        assert old_optimizer.packet_hashes == new_optimizer.packet_hashes
        assert old_optimizer.local_eval_ids == new_optimizer.local_eval_ids
        assert old_eval.events == new_eval.events
        assert old.committed_candidate_id == new.trace[0].committed_candidate_id
        assert old_state.snapshot().team_state_id == new_state.snapshot().team_state_id
        assert old.funnel["team_minibatch_survivors"] == len(new.trace[0].promoted_ids)
        assert old.funnel["full_team_evaluated_candidates"] == sum(
            event[0] == "full" for event in new_eval.events
        )


class FakeLocalSolver:
    solver_contract_id = "COMMON_SOLVER_CONTRACT_V1"
    output_contract_id = "output"

    def __init__(self):
        self.calls = 0

    def evaluate(self, decision_procedure, row):
        del row
        self.calls += 1
        correct = "distinguish" in decision_procedure.casefold()
        return LocalSolverObservation(
            "A" if correct else "B", "fake response", correct, True,
            input_tokens=2, output_tokens=1,
        )


class FakeReflection:
    def __init__(self):
        self.accounting = {"successful_calls": 0, "input_tokens": 0,
                           "output_tokens": 0}

    def __call__(self, prompt):
        del prompt
        self.accounting["successful_calls"] += 1
        self.accounting["input_tokens"] += 10
        self.accounting["output_tokens"] += 5
        return "```Use semantic compatibility to distinguish candidate referents carefully.```"


def _real_fake_gepa(root):
    solver = FakeLocalSolver()
    reflection = FakeReflection()
    engine = GEPALocalPromptOptimizer(
        evaluator=solver, reflection_lm=reflection,
        accounting_reader=lambda: reflection.accounting,
        run_root=root,
    )
    return GEPALayer2EvidenceOptimizer(engine=engine), solver, reflection


def test_bbh_v4_official_gepa_fake_provider_golden(tmp_path):
    prompt = "Use a general compatibility procedure."
    old_state = FakeTeamState(prompt)
    new_state = FakeTeamState(prompt)
    assignment, request, builder = _assignment_and_request(
        old_state.snapshot().team_state_id, prompt,
    )
    old_gepa, old_solver, old_reflection = _real_fake_gepa(tmp_path / "old")
    new_gepa, new_solver, new_reflection = _real_fake_gepa(tmp_path / "new")
    old_eval = FakeTeamEvaluator(shadow_passed=True)
    old = asyncio.run(TeamSearchController(
        responsibility=FrozenResponsibility(assignment), task_builder=builder,
        local_optimizer=old_gepa, evaluator=old_eval,
        selector=FakeSelector(), committer=FakeCommitter(old_state),
    ).run_opportunity(request))

    frozen = FrozenBBHOpportunityBuilder(request, assignment, builder)
    bridge = FrozenBBHGEPARequestBridge(frozen, new_gepa)
    new_eval = FakeTeamEvaluator(shadow_passed=True)
    provider = FrozenBBHTeamEvaluationProvider(frozen, bridge, new_eval)
    unified = asyncio.run(UnifiedSearchOrchestrator(
        method=SearchMethodConfig(), benchmark=FakeBenchmark(),
        aggregation=PluralityAggregation(), state=new_state,
        analyzer=StateAnalyzer(FakeDiagnosis()), opportunities=frozen,
        engine=GEPASearchEngine(bridge),
        evaluation=CandidateEvaluationPipeline(provider, FrozenBBHPromotion()),
        transition=FrozenBBHTransitionPolicy(bridge, FakeSelector()),
        gate=FrozenBBHGate(frozen, bridge, new_eval),
        committer=FrozenBBHCommitter(frozen, bridge, FakeCommitter(new_state), new_state),
    ).run(max_opportunities=1))
    assert old_solver.calls == new_solver.calls > 0
    assert old_reflection.accounting == new_reflection.accounting
    assert old_eval.events == new_eval.events
    assert tuple(row.local_candidate.prompt for row in old.candidates) == tuple(
        row.prompt for row in bridge.raw_candidates.values()
    )
    assert old.committed_candidate_id == unified.trace[0].committed_candidate_id
    assert old_state.snapshot().team_state_id == new_state.snapshot().team_state_id
    assert old.funnel["local_candidates"] == len(unified.trace[0].candidate_ids)


def test_current_bbh_diagnosis_target_and_evidence_match_frozen_v4(tmp_path):
    ids = tuple(f"wrong-{index:02d}" for index in range(8)) + tuple(
        f"right-{index:02d}" for index in range(4)
    )
    states = tuple(build_team_vote_state(
        question_hash=row_id, gold_answer="A",
        answers=("B",) * 5 if index < 8 else ("A",) * 5,
        valid_vector=(True,) * 5,
    ) for index, row_id in enumerate(ids))
    opportunities = {state.question_hash: tuple(
        compute_member_aware_repair_opportunity(
            team_state=state, peer_context=build_peer_vote_context(state, member),
        ) for member in range(5)
    ) for state in states}

    class System:
        responsibility_state = ResponsibilityState(
            updates_since_selected_by_agent={member: 0 for member in range(5)}
        )
        fixed_probe = SimpleNamespace(examples=tuple(
            SimpleNamespace(question_hash=row_id, question="public case", gold_answer="A")
            for row_id in ids
        ))
        agents = tuple(SimpleNamespace(current_prompt="parent") for _ in range(5))
        active_profiles = tuple(tuple(SimpleNamespace(
            answer="B" if index < 8 else "A", valid=True,
        ) for index in range(12)) for _ in range(5))
        protocol = SimpleNamespace(tie_policy="abstain")
        normalize_answer = staticmethod(lambda answer: answer)
        match_answer = staticmethod(lambda answer, gold: answer == gold)

        def __init__(self):
            self.agents = tuple(SimpleNamespace(current_prompt="parent") for _ in range(5))
            self.active_profiles = tuple(tuple(SimpleNamespace(
                answer="B" if index < 8 else "A", valid=True,
            ) for index in range(12)) for _ in range(5))

        def team_prompt_state_hash(self):
            return hashlib.sha256(repr(tuple(row.current_prompt for row in self.agents)).encode()).hexdigest()

        def current_states_and_opportunities(self):
            return states, {}, opportunities

    system = System()
    history = HistoryState()
    new_state = CurrentBBHStateSource(system).snapshot()
    new_diagnosis = PluralityResponsibilityAnalyzer(system).analyze(new_state, history)
    old_snapshot = freeze_current_responsibility(system, update_index=0)
    old_summaries = build_primary_responsibility_summaries(
        assigned=old_snapshot.assigned,
        current_margin_by_question=old_snapshot.current_margin_by_question,
        failure_count_by_member={},
    )
    assert tuple((new_diagnosis.responsibility[i].direct_count,
                  new_diagnosis.responsibility[i].near_margin_count,
                  new_diagnosis.responsibility[i].coverage_count,
                  new_diagnosis.responsibility[i].raw_value,
                  new_diagnosis.responsibility[i].primary_lane)
                 for i in range(5)) == tuple(
                     (row.direct_count, row.near_margin_count, row.coverage_count,
                      row.primary_score, row.primary_lane) for row in old_summaries
                 )
    request = TeamSearchRequest(81, 0, new_state.team_state_id, 36,
                                "COMMON_SOLVER_CONTRACT_V1", "output")
    old_builder = Layer2EvidenceRequestBuilder(bounded_search_view=True)
    old_factory = SystemResponsibilityAssignmentFactory(
        system=system, snapshot_reader=lambda: old_snapshot, task_builder=old_builder,
    )
    old_selected = select_v4_feasible_opportunity(
        snapshot=old_snapshot,
        scheduler=PrimaryResponsibilityPersistentRealizabilityScheduler(
            version=PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION,
        ), factory=old_factory, task_builder=old_builder, request=request,
    )
    new_opportunity = OpportunityBuilder(
        source=CurrentBBHEvidenceSource(system, history),
        feasibility=CurrentEvidenceFeasibility(), target=TargetPolicyV1(),
        evidence=CurrentRoleEvidencePolicy(),
    ).build(state=new_state, diagnosis=new_diagnosis, history=history, update_index=0)
    assert new_opportunity is not None
    assert new_opportunity.target_member == old_selected.assignment.target_member
    assert new_opportunity.objective["raw_responsibility"] == old_selected.assignment.responsibility_value
    assert tuple(row.example_id for row in new_opportunity.evidence.team_probe_evidence) == (
        old_selected.assignment.local_validation_example_ids
    )
    assert tuple(row.example_id for row in new_opportunity.evidence.search_validation_evidence) == (
        old_selected.assignment.local_validation_example_ids
    )
    assert tuple(row.example_id for row in new_opportunity.evidence.mutation_evidence
                 if "RESPONSIBILITY" in row.roles) == tuple(
                     row.example_id for row in old_selected.packet.responsibility_examples
                 )
    new_packet = CurrentGEPAPacketAdapter().build(
        new_opportunity, history=history, seed=81, update_index=0,
        solver_contract_id=request.solver_contract_id,
        output_contract_id=request.output_contract_id,
    ).packet
    assert tuple((n.example_id, n.metadata, o.metadata) for n, o in zip(
        new_packet.local_eval_examples, old_selected.packet.local_eval_examples, strict=True
    ) if n != o) == ()
    assert new_packet.__dict__ == old_selected.packet.__dict__
    assert new_packet.packet_hash == old_selected.packet.packet_hash
    assert new_packet.ordered_batch_schedule == old_selected.packet.ordered_batch_schedule
    adapted = CurrentBBHAssignmentAdapter(history).build(new_opportunity)
    assert adapted.evidence == old_selected.assignment.evidence
    assert adapted.local_validation_example_ids == old_selected.assignment.local_validation_example_ids
    old_optimizer, old_solver, old_reflection = _real_fake_gepa(tmp_path / "packet_old")
    new_optimizer, new_solver, new_reflection = _real_fake_gepa(tmp_path / "packet_new")
    old_result = asyncio.run(old_optimizer.optimize_layer2(
        old_builder.build(request, old_selected.assignment)
    ))
    bridge = CurrentGEPABridge(
        optimizer=new_optimizer, history=history, seed=81,
        solver_contract_id=request.solver_contract_id,
        output_contract_id=request.output_contract_id,
    )
    new_result = asyncio.run(bridge.optimize(bridge.make_task(new_opportunity, None)))
    assert old_solver.calls == new_solver.calls
    assert old_reflection.accounting == new_reflection.accounting
    assert tuple(row.prompt for row in old_result.candidates) == tuple(
        row.prompt for row in new_result.candidates
    )
    derived_solver = FakeLocalSolver()
    derived_reflection = FakeReflection()
    derived_engine = GEPALocalPromptOptimizer(
        evaluator=derived_solver, reflection_lm=derived_reflection,
        accounting_reader=lambda: derived_reflection.accounting,
        run_root=tmp_path / "derived_saturation",
        config=GEPADerivedConfig(),
    )
    derived_optimizer = GEPALayer2EvidenceOptimizer(
        engine=derived_engine,
        saturation_config=SaturationConfig(
            enabled=True, scientific_budget_enabled=False,
            local_no_update_patience=3, team_no_update_patience=2,
        ),
    )
    derived_result = asyncio.run(derived_optimizer.optimize_layer2(
        bridge.make_task(new_opportunity, None)
    ))
    assert derived_result.termination_reason == "SATURATION_REACHED"
    assert derived_solver.calls > 0

    class EmptyOptimizer:
        async def optimize_layer2(self, task):
            assert task.packet.packet_hash == old_selected.packet.packet_hash
            return LocalOptimizationResult((), "gepa", "fake", None, 0, 0, 0,
                                           0, 0, "SATURATION_REACHED")

    composed = build_current_bbh_orchestrator(
        system=system, benchmark=FakeBenchmark(), optimizer=EmptyOptimizer(),
        evaluator=FakeTeamEvaluator(shadow_passed=True), committer=object(),
        seed=81, solver_contract_id=request.solver_contract_id,
        output_contract_id=request.output_contract_id,
    )
    composed_result = asyncio.run(composed.run(max_opportunities=1))
    assert composed_result.trace[0].target_member == old_selected.assignment.target_member
    assert composed_result.trace[0].candidate_ids == ()
    assert composed_result.final_state_id == composed_result.initial_state_id

    def candidate_evaluation(prompt, member, *, improved):
        counts = tuple(5 if improved and i == member else 4 for i in range(5))
        return SimpleNamespace(
            prompt=prompt, prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            competence=SimpleNamespace(correct_count=counts[member],
                                       terminal_invalid_count=0, invalid_count=0),
            team_outcome=SimpleNamespace(vote_correct_count=5 if improved else 4,
                                         mean_soft_vote_utility=0.0),
            member_gain=member_gain_metrics((4,) * 5, (4,) * 5, counts, member),
            marginal=SimpleNamespace(vote_gain_count=int(improved), vote_loss_count=0,
                                     net_vote_delta=int(improved)),
            protection=SimpleNamespace(unique_correct_gain_count=0,
                                       unique_correct_loss_count=0,
                                       pivotal_correct_gain_count=0,
                                       pivotal_correct_loss_count=0),
        )

    class MeasuredEvaluator:
        def __init__(self, measured_system, *, shadow_passed=True):
            self.system = measured_system
            self.shadow_passed = shadow_passed
            self.events = []
            self.transition_audits = {}

        def active_evaluation(self, assignment):
            return candidate_evaluation(assignment.parent_prompt,
                                        assignment.target_member, improved=False)

        def evaluate_minibatch(self, assignment, candidate, minibatch):
            self.events.append(("team_probe", tuple(row.example_id for row in minibatch)))
            return TeamMiniBatchMetrics(target_delta=1, vote_delta=1,
                                        responsibility_delta=1), EvaluationCost(0, 0, 0)

        def evaluate_full(self, assignment, candidate):
            self.events.append(("full", candidate.candidate_id))
            member = assignment.target_member
            parent = self.system.team_prompt_state_hash()
            parent_correct = tuple((row_id, index >= 8) for index, row_id in enumerate(ids))
            child_correct = tuple((row_id, index >= 8 or index == 0)
                                  for index, row_id in enumerate(ids))
            self.transition_audits[(member, candidate.candidate_id, parent)] = (
                CandidateTransitionAudit(
                    hashlib.sha256(assignment.parent_prompt.encode()).hexdigest(),
                    hashlib.sha256(candidate.prompt.encode()).hexdigest(),
                    parent_correct, child_correct,
                )
            )
            return candidate_evaluation(candidate.prompt, member, improved=True), EvaluationCost(0, 0, 0)

        def evaluate_shadow(self, assignment, candidate):
            self.events.append(("shadow", candidate.candidate_id))
            return SimpleNamespace(passed=self.shadow_passed), EvaluationCost(0, 0, 0)

    class MeasuredCommitter:
        def __init__(self, measured_system):
            self.system = measured_system

        def commit(self, *, assignment, candidate, evaluation):
            assert evaluation.prompt_hash == hashlib.sha256(candidate.prompt.encode()).hexdigest()
            self.system.agents[assignment.target_member].current_prompt = candidate.prompt

    old_system = System()
    new_system = System()
    old_eval = MeasuredEvaluator(old_system)
    new_eval = MeasuredEvaluator(new_system)
    old_selected_assignment = old_selected.assignment
    old_controller = TeamSearchController(
        responsibility=FrozenResponsibility(old_selected_assignment),
        task_builder=old_builder, local_optimizer=FakeLocalOptimizer(),
        evaluator=old_eval, selector=CommonSafeTeamCandidateSelector(),
        committer=MeasuredCommitter(old_system),
    )
    old_outcome = asyncio.run(old_controller.run_opportunity(request))
    new_runtime = build_current_bbh_orchestrator(
        system=new_system, benchmark=FakeBenchmark(), optimizer=FakeLocalOptimizer(),
        evaluator=new_eval, committer=MeasuredCommitter(new_system),
        seed=81, solver_contract_id=request.solver_contract_id,
        output_contract_id=request.output_contract_id,
    )
    new_outcome = asyncio.run(new_runtime.run(max_opportunities=1))
    assert old_eval.events == new_eval.events
    assert old_outcome.committed_candidate_id == new_outcome.trace[0].committed_candidate_id
    assert old_system.team_prompt_state_hash() == new_system.team_prompt_state_hash()
    assert new_outcome.transitions[0].newly_fixed_ids == ("wrong-00",)
    successor_state = CurrentBBHStateSource(new_system).snapshot()
    successor_diagnosis = PluralityResponsibilityAnalyzer(new_system).analyze(
        successor_state, new_runtime.history,
    )
    successor_opportunity = OpportunityBuilder(
        source=CurrentBBHEvidenceSource(new_system, new_runtime.history),
        feasibility=CurrentEvidenceFeasibility(), target=TargetPolicyV1(),
        evidence=CurrentRoleEvidencePolicy(),
    ).build(state=successor_state, diagnosis=successor_diagnosis,
            history=new_runtime.history, update_index=1)
    assert successor_opportunity is not None
    old_successor = old_builder.build(
        TeamSearchRequest(81, 1, successor_state.team_state_id, 36,
                          request.solver_contract_id, request.output_contract_id),
        CurrentBBHAssignmentAdapter(new_runtime.history).build(successor_opportunity),
    )
    new_successor = CurrentGEPAPacketAdapter().build(
        successor_opportunity, history=new_runtime.history, seed=81,
        update_index=1, solver_contract_id=request.solver_contract_id,
        output_contract_id=request.output_contract_id,
    )
    assert new_successor.packet.packet_hash == old_successor.packet.packet_hash
    assert tuple(row.example_id for row in new_successor.packet.anchor_examples) == ("wrong-00",)

    rejected_system = System()
    rejected_evaluator = MeasuredEvaluator(rejected_system, shadow_passed=False)
    rejected = build_current_bbh_orchestrator(
        system=rejected_system, benchmark=FakeBenchmark(),
        optimizer=FakeLocalOptimizer(), evaluator=rejected_evaluator,
        committer=MeasuredCommitter(rejected_system), seed=81,
        solver_contract_id=request.solver_contract_id,
        output_contract_id=request.output_contract_id,
    )
    rejected_result = asyncio.run(rejected.run(max_opportunities=1))
    assert rejected_result.trace[0].selected_candidate_id == "child"
    assert rejected_result.trace[0].committed_candidate_id is None
    assert rejected_result.final_state_id == rejected_result.initial_state_id
    assert rejected.history.transitions == []
