"""Zero-API semantic-contract proofs, including the pinned GEPA reflection seam."""
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl import versions
from multi_dataset_diverse_rl.search.semantic_contract import (
    FocusedPatternDiagnosticV2, FocusedEvidencePolicyV2, InitialCompetenceTransitionV2,
    mechanism_identity,
)
from multi_dataset_diverse_rl.search.experience import StrategyExperienceMemoryV2, abstract_actions
from multi_dataset_diverse_rl.search.patterns import PatternHypothesis
from multi_dataset_diverse_rl.search.memory import OpportunityOutcome
from multi_dataset_diverse_rl.search.schemas import (
    SearchMethodConfig, SearchCandidate, SearchContractError, TeamEvaluation, EvaluatedCandidate,
    Diagnosis, BenchmarkCapabilities,
)
from multi_dataset_diverse_rl.search.history import HistoryState
from multi_dataset_diverse_rl.search.policies import TargetPolicyV1, ResponsibilitySignal
from multi_dataset_diverse_rl.search.variable_evidence import VariableEvidenceFeasibilityV1
from multi_dataset_diverse_rl.search.runtime_v2 import V2OpportunityBuilder
from multi_dataset_diverse_rl.search.orchestrator import StateAnalyzer, UnifiedSearchOrchestrator, UnifiedSearchContext
from multi_dataset_diverse_rl.search.evaluation import CandidateEvaluationPipeline
from multi_dataset_diverse_rl.search.transition import TeamStateCommitter
from multi_dataset_diverse_rl.search.aggregation import PluralityAggregation
from multi_dataset_diverse_rl.search.gepa_v2 import V2ReflectionAdapter
from test_unified_search_v2 import (
    rows, Store, Analyzer, Source, TeamProvider, Gate, make_engine, Solver,
    PARENT, A, B, LIMITS, opportunity, CurrentBBHPromotion,
)

ROOT = Path(__file__).resolve().parents[1]


class Patterns:
    def __init__(self, patterns):
        self.patterns = patterns
        self.calls = []

    def diagnose(self, request):
        self.calls.append(request)
        return self.patterns


def diagnose(patterns, values=None):
    values = values if values is not None else rows(10)
    return FocusedPatternDiagnosticV2(Patterns(patterns)).analyze(
        Store().snapshot(), Analyzer().analyze(None, None), 2, values, HistoryState())


def test_denominators_and_low_coverage_never_gate_focus():
    context = diagnose((PatternHypothesis("label", "Missed constraint", "Check explicit constraints", ("e0", "e1")),))
    assert context["DPR_all"] == .2
    assert context["ConditionalDPR"] == 1
    assert context["Coverage"] == .2
    assert context["dominant_pattern_ratio"] == .2
    assert context["normalized_entropy"] == 0
    assert context["focus_mechanism_id"]


def test_focus_is_lane_aligned_and_independent_of_label():
    values = tuple(replace(r, signals={**r.signals, "lane": "near_margin"},
        roles=frozenset({"REPAIR", "near_margin"})) if i >= 2 else r for i, r in enumerate(rows(7)))
    hypotheses = (PatternHypothesis("small", "Missed constraint", "Check constraints", ("e0", "e1"), confidence=.5),
                  PatternHypothesis("large", "Missed cases", "Enumerate cases", tuple(f"e{i}" for i in range(2, 7)), confidence=.9))
    a = diagnose(hypotheses, values)
    b = diagnose(tuple(replace(p, pattern_id="renamed_" + p.pattern_id) for p in hypotheses), values)
    assert a == b
    assert a["support_count"] == 2
    assert a["mixed_pattern_count"] == 2 and a["focus_mechanism_id"]
    assert a["DPR_all"] == pytest.approx(2/7)


def test_mechanism_identity_uses_description_and_merges_duplicate_mechanisms():
    assert mechanism_identity(" missed  CHECK ", "Verify Constraints") == mechanism_identity("MISSED CHECK", "verify constraints")
    assert mechanism_identity("missed check", "verify constraints") != mechanism_identity("missed case", "verify constraints")
    context = diagnose((PatternHypothesis("one", "Missed check", "Verify constraints", ("e0",)),
                        PatternHypothesis("two", "missed check", "verify constraints", ("e1",))))
    assert len(context["patterns"]) == 1 and context["support_count"] == 2


def test_empty_analyzer_partition_fails_without_generic_mixed_fallback():
    with pytest.raises(SearchContractError, match="FOCUS_NOT_IDENTIFIED"):
        diagnose(())


def focused_view(values, hypotheses):
    context = diagnose(hypotheses, values)
    view, audit = FocusedEvidencePolicyV2().compose(Store().snapshot(), Analyzer().analyze(None, None), 2, values, context)
    return view, audit, context


def test_one_focus_with_genuine_boundary_backfill_excludes_other_repairs():
    values = rows(8, repairs=5)
    view, audit, context = focused_view(values, (
        PatternHypothesis("focus", "Missed constraints", "Check constraints", ("e0",)),
        PatternHypothesis("other", "Missed cases", "Enumerate cases", ("e1",)),))
    # Equal support gives a stable mechanism tie; whichever wins must exclude
    # ALL other repairs, including unassigned residuals e2/e3/e4.
    support = next(p["support_ids"] for p in context["patterns"] if p["pattern_id"] == context["focus_mechanism_id"])
    mutation_ids = {r.example_id for r in view.mutation_evidence}
    assert mutation_ids & {f"e{i}" for i in range(5)} == set(support)
    assert len(mutation_ids) == 3
    assert audit["nonfocus_repair_count"] == 0 and audit["boundary_count"] == 2
    assert all("REPAIR" not in r.roles for r in view.mutation_evidence if r.example_id not in support)


def test_declared_counterexample_is_boundary_not_second_repair_gradient():
    view, audit, _ = focused_view(rows(5), (PatternHypothesis("focus", "Missed constraints", "Check constraints",
        ("e0",), counterexample_ids=("e1",), risk_ids=("e2",)),))
    assert {r.example_id for r in view.mutation_evidence} == {"e0", "e1", "e2"}
    assert audit["pattern_risk_count"] == 2
    assert next(r for r in view.mutation_evidence if r.example_id == "e1").signals["legacy_tags"] == ("safety_boundary_v2",)


def test_no_disguised_repair_backfill_when_no_legal_boundaries():
    with pytest.raises(SearchContractError, match="MINIMUM_WITHOUT_LEGAL_BOUNDARIES"):
        focused_view(rows(5), (PatternHypothesis("focus", "Missed constraint", "Check constraints", ("e0",)),))


def test_reflection_marks_even_wrong_boundaries_as_nonrepair(tmp_path):
    view, _, context = focused_view(rows(5), (PatternHypothesis("focus", "Missed constraint", "Check constraints",
        ("e0",), counterexample_ids=("e1",), risk_ids=("e2",)),))
    engine, _, _ = make_engine(tmp_path)
    o = replace(opportunity(), evidence=view, pattern_context=context)
    task = engine.bridge.make_task(o, UnifiedSearchContext(None, None, HistoryState(), context, {}))
    adapter = V2ReflectionAdapter(Solver(), parent_prompt=PARENT, all_examples=task.search_examples,
        optimization_context=task.optimization_context, output_contract_id="synthetic")
    batch = adapter.evaluate(list(task.search_examples), {"decision_procedure": PARENT}, True)
    records = adapter.make_reflective_dataset({"decision_procedure": PARENT}, batch, ["decision_procedure"])["decision_procedure"]
    for row in records:
        if row["example_id"] != "e0":
            assert "BOUNDARY ONLY" in row["Reasoning Focus"] and "Do not repair" in row["Reasoning Focus"]
        else:
            assert "single selected focus" in row["Reasoning Focus"]
    assert len(records) == 3


def test_responsibility_can_concentrate_without_round_robin_and_exposes_other_member():
    diagnosis = Diagnosis({0: ResponsibilitySignal(0, 20, 0, 0, "direct_flip"),
                           1: ResponsibilitySignal(1, 1, 0, 0, "direct_flip")})
    history = HistoryState()
    selected = []
    for _ in range(21):
        decision = TargetPolicyV1().select(Store().snapshot(), diagnosis, (0, 1), history)
        selected.append(decision.selected_member)
        history.observe_opportunity(decision.selected_member, committed=False)
    assert selected[:20] == [0] * 20 and selected[20] == 1
    assert diagnosis.responsibility[0].raw_value == 80


def candidate(target_score, team_score, *, invalid=0, local_score=-100):
    full = TeamEvaluation(team_score, None, (target_score, 65, 65, 65, 65),
        aggregation_diagnostics={"terminal_invalid_delta": invalid})
    return EvaluatedCandidate(SearchCandidate("c", "Check explicit constraints before reasoning.", local_score),
                              None, full, True, False, {"target_member": 0})


@pytest.mark.parametrize("member,team,allowed", [(68, 87, True), (65, 87, True), (64, 87, False),
    (68, 82, False), (68, 81, False), (73, 82, False), (73, 87, True)])
def test_initial_floor_and_strict_team_gain(member, team, allowed):
    policy = InitialCompetenceTransitionV2()
    policy.bind_initial((65,) * 5, "initial")
    parent = TeamEvaluation(82, None, (72,) * 5)
    row = candidate(member, team)
    assert (policy.select(parent, (row,)).candidate is row) == allowed
    assert policy.select(parent, (candidate(member, team, invalid=1),)).candidate is None


def test_floor_cannot_be_rebased_or_inferred_from_incumbent():
    policy = InitialCompetenceTransitionV2()
    with pytest.raises(SearchContractError, match="NOT_FROZEN"):
        policy.select(TeamEvaluation(82, None, (72,) * 5), ())
    policy.bind_initial((65,) * 5, "initial")
    with pytest.raises(SearchContractError, match="CANNOT_REBASE"):
        policy.bind_initial((72,) * 5, "child")


def memory_outcome(*, success=True, prompt="Check explicit constraints and enumerate cases.", context=None):
    o = replace(opportunity(), parent_prompt="Reason carefully.", pattern_context=context or {})
    row = candidate(68, 87)
    row = replace(row, candidate=replace(row.candidate, prompt=prompt), diagnostics={"target_member": 2,
        **({"scientific_risk_code": "COMMON_SAFE_REJECTION"} if not success else {})})
    return OpportunityOutcome(o, (row,), row.candidate.candidate_id if success else None, success, True if success else None, 0)


def test_experience_is_grounded_private_strategy_and_shared_failed_action():
    memory = StrategyExperienceMemoryV2(**LIMITS)
    delta = memory.prepare_outcome(memory_outcome())
    memory.validate_delta(delta); memory.apply_outcome(delta)
    private = memory.private[0]
    assert private.owner_member == 2
    assert private.action["added_checks"] == ("case_analysis", "explicit_constraint_check")
    assert all(getattr(private, k) for k in ("situation", "action", "outcome", "lesson"))
    assert private.lesson["successful_strategy"] and private.lesson["avoid_strategy"] is None
    delta = memory.prepare_outcome(memory_outcome(success=False))
    memory.validate_delta(delta); memory.apply_outcome(delta)
    assert memory.shared[0].owner_member is None
    assert memory.shared[0].lesson["avoid_strategy"] and memory.shared[0].lesson["successful_strategy"] is None
    o = memory_outcome().opportunity
    assert memory.read_for_opportunity(o)["private"]
    assert memory.read_for_opportunity(replace(o, target_member=1,
        diagnosis=Diagnosis({1: ResponsibilitySignal(1, 1, 0, 0, "direct_flip")})))["private"] == []


def test_unknown_edits_operational_failure_and_incomplete_do_not_manufacture_experience():
    memory = StrategyExperienceMemoryV2(**LIMITS)
    for outcome in (memory_outcome(prompt="Think carefully in a new way."),
                    replace(memory_outcome(), operational_failure=True), replace(memory_outcome(), complete=False)):
        delta = memory.prepare_outcome(outcome)
        memory.validate_delta(delta); memory.apply_outcome(delta)
    assert not memory.private and not memory.shared and memory.write_count == 0


def test_negated_strategy_is_not_misreported_as_successful_added_check():
    assert "explicit_constraint_check" not in abstract_actions("Never check constraints.")
    assert "case_analysis" not in abstract_actions("Do not enumerate cases.")
    memory = StrategyExperienceMemoryV2(**LIMITS)
    outcome = memory_outcome(prompt="Never check constraints.")
    delta = memory.prepare_outcome(outcome)
    assert not delta.private
    outcome = replace(outcome, opportunity=replace(outcome.opportunity, parent_prompt="Check explicit constraints."))
    delta = memory.prepare_outcome(outcome)
    assert delta.private[0].action["removed_checks"] == ("explicit_constraint_check",)
    assert not delta.private[0].action["added_checks"]


def test_memory_stores_no_provider_copied_question_answer_or_prompt():
    context = diagnose((PatternHypothesis("raw-provider-label", "Secret raw question.",
        "Secret gold answer. Check constraints.", ("e0", "e1", "e2")),))
    memory = StrategyExperienceMemoryV2(**LIMITS)
    delta = memory.prepare_outcome(memory_outcome(context=context))
    memory.validate_delta(delta); memory.apply_outcome(delta)
    raw = json.dumps(asdict(memory.private[0]))
    assert "Secret" not in raw and "raw-provider-label" not in raw
    assert "Check explicit constraints and enumerate cases." not in raw
    assert memory.private[0].pattern_id == context["focus_mechanism_id"]


def test_bounded_memory_context_contains_real_experience_not_just_identifiers():
    memory = StrategyExperienceMemoryV2(**{**LIMITS, "max_context_chars": 1200})
    outcome = memory_outcome()
    delta = memory.prepare_outcome(outcome)
    memory.validate_delta(delta); memory.apply_outcome(delta)
    view = memory.read_for_opportunity(outcome.opportunity)
    assert len(json.dumps(view, sort_keys=True)) <= 1200
    assert view["private"]
    assert all(view["private"][0][k] for k in ("situation", "action", "outcome", "lesson"))


def test_current_manifest_rejects_old_transition_pattern_and_unbound_memory():
    import yaml
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    from scripts.replay_experiment import preflight
    draft = yaml.safe_load((ROOT / "experiments/templates/unified_experiment_v2_1.yaml").read_text(encoding="utf-8"))
    assert validate_manifest_v2(ROOT, draft) == []
    assert preflight(draft)["gate"] == "HOLD"
    for field, value in (("transition_identity", "common_safe_v1"), ("evidence_identity", "variable_pattern_capable_evidence_v1"),
                         ("pattern_identity", "pattern_diagnostic_v1"), ("memory_identity", "strategy_experience_memory_v2"),
                         ("models", {"solver": "qwen3.7-flash", "optimizer": "qwen3.7-flash", "solver_thinking": False})):
        assert validate_manifest_v2(ROOT, {**draft, field: value})


class InitialStore(Store):
    def __init__(self):
        super().__init__()
        self.initial_member_scores = TeamProvider.evaluation(PARENT).member_scores
        self.initial_state_id = super().snapshot().team_state_id

    def snapshot(self):
        return replace(super().snapshot(), member_scores=TeamProvider.evaluation(self.prompts[2]).member_scores)


class TeamProviderV21(TeamProvider):
    async def full(self, opp, candidate):
        result = await super().full(opp, candidate)
        return replace(result, aggregation_diagnostics={**result.aggregation_diagnostics, "terminal_invalid_delta": 0})


@pytest.mark.parametrize("pattern_enabled,memory_enabled", [(False, False), (True, False), (False, True), (True, True)])
def test_four_current_arms_on_single_orchestrator_local_rejected_team_good_commits(tmp_path, pattern_enabled, memory_enabled):
    engine, solver, reflection = make_engine(tmp_path)
    solver.structural = True
    store, history, provider = InitialStore(), HistoryState(), TeamProviderV21()
    patterns = FocusedPatternDiagnosticV2(Patterns((PatternHypothesis("provider-id", "Missing relational check",
        "Check relational consistency", ("e0", "e1", "e2")),))) if pattern_enabled else None
    memory = StrategyExperienceMemoryV2(**LIMITS) if memory_enabled else None
    mechanism = {}
    if patterns:
        mechanism["pattern_provider_binding"] = "fake_bound_v2"
    if memory:
        mechanism["memory"] = LIMITS
    method = SearchMethodConfig.v2_1(pattern_policy=patterns.identity if patterns else versions.UNIFIED_NULL_PATTERN_VERSION,
        memory_policy=memory.identity if memory else versions.UNIFIED_NULL_MEMORY_VERSION, mechanism_config=mechanism)
    built = V2OpportunityBuilder(source=Source(rows()), feasibility=VariableEvidenceFeasibilityV1(),
        target=TargetPolicyV1(), evidence=FocusedEvidencePolicyV2(), patterns=patterns)
    run = UnifiedSearchOrchestrator(method=method, benchmark=SimpleNamespace(capabilities=BenchmarkCapabilities(True, True, True)),
        aggregation=PluralityAggregation(), state=store, analyzer=StateAnalyzer(Analyzer()), opportunities=built,
        engine=engine, evaluation=CandidateEvaluationPipeline(provider, CurrentBBHPromotion()),
        transition=InitialCompetenceTransitionV2(), gate=Gate(), committer=TeamStateCommitter(store), history=history, memory=memory)
    result = asyncio.run(run.run(max_opportunities=1))
    assert len(result.transitions) == 1 and store.prompts[2] == B
    rejected = [c for c in engine.bridge.raw_candidates.values() if not c.backend_metadata["gepa_search_survived"]]
    assert rejected[0].candidate_id == result.trace[0].committed_candidate_id
    assert result.trace[0].allocation_audit["realized_team_gain"] == 1
    assert result.trace[0].allocation_audit["initial_member_scores"] == store.initial_member_scores
    assert all("Optional Search Context" not in problem and '"private"' not in problem for _, problem in solver.requests)
    assert engine.bridge.optimizer.config.reflection_minibatch_size == 3


def test_new_identity_and_cli_are_closed_until_new_freeze():
    from scripts.replay_experiment import preflight
    assert SearchMethodConfig.v2_1().identity() != SearchMethodConfig.v2().identity()
    assert SearchMethodConfig.from_mapping({"method": versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION}) == SearchMethodConfig.v2_1()
    result = preflight({"scientific": {"method": versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION}, "runtime": {
        "seed": 81, "provider_profile": "fake", "solver_model": "qwen3-8b", "optimizer_model": "qwen3.7-flash",
        "evaluator_model": "qwen3.7-flash", "run_identity_sha256": "fake", "authorization_identity": "none",
        "cache_identity": "fake", "ledger_identity": "fake"}})
    assert result["gate"] == "HOLD"
    assert result["provider_attempts"] == result["validation_calls"] == result["test_calls"] == 0


@pytest.mark.parametrize("arm", ["A1", "A2", "A3", "A4"])
def test_current_binary_production_composition_actual_pinned_gepa_four_arms(tmp_path, arm):
    import os
    import test_math_preexecution as fixtures
    from multi_dataset_diverse_rl.search.binary_composition import build_binary_orchestrator
    from multi_dataset_diverse_rl.search.gepa_v2 import GEPATeamExposureOptimizer
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker,BenchmarkSolver,ReflectionProvider
    from multi_dataset_diverse_rl.search.legacy.pattern_provider import PatternProvider
    c = fixtures.contract()
    binding = fixtures.MATHExecutionBinding(ROOT, c)
    old = binding.method(arm)
    method = replace(old, method=versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION,
        evidence_policy=versions.UNIFIED_FOCUSED_EVIDENCE_VERSION,
        transition_policy=versions.UNIFIED_COMPETENCE_TRANSITION_VERSION,
        pattern_policy=versions.UNIFIED_FOCUSED_PATTERN_VERSION if c["arms"][arm][0] else versions.UNIFIED_NULL_PATTERN_VERSION,
        memory_policy=versions.UNIFIED_EXPERIENCE_MEMORY_VERSION if c["arms"][arm][1] else versions.UNIFIED_NULL_MEMORY_VERSION)
    prompts = tuple(m["prompt"] for m in json.loads((ROOT / c["initial_team_path"]).read_bytes())["members"])
    adapter = fixtures.MATHBenchmarkAdapter()
    def examples(prefix, count):
        return tuple(fixtures.CorrectnessExample(fixtures.BenchmarkInput(prefix + str(i),
            f"Synthetic {prefix} arithmetic {i}.", adapter.output_contract, benchmark_id="math"), "1") for i in range(count))
    optimize, shadow = examples("optimize", 6), examples("shadow", 300)
    good = "Check explicit constraints and verify relational consistency before deriving the mathematical result."
    other = "Enumerate cases carefully before deriving the mathematical result."
    calls, contexts = [], []
    def transport(req):
        calls.append(req)
        if req["model"] == "qwen3-8b":
            assert req["messages"][0]["content"] == adapter.output_contract
            prompt, problem = req["messages"][1]["content"].split("\n\n", 1)
            assert '"memory"' not in problem and '"pattern"' not in problem
            i = int(problem.rsplit(" ", 1)[1].rstrip("."))
            if problem.startswith("Synthetic shadow"):
                correct = prompt in {prompts[1], prompts[2], good, other}
            elif prompt in prompts:
                m = prompts.index(prompt)
                correct = m in {1, 2} or m == 0 and i >= 3
            else:
                assert prompt in {good, other}
                correct = i != 2 if prompt == good else i >= 2
            return dict(text="Synthetic reasoning.\nFINAL_ANSWER: " + ("1" if correct else "2"), input_tokens=2, output_tokens=2)
        assert req["model"] == "qwen3.7-flash"
        if len(req["messages"]) == 2:
            data = json.loads(req["messages"][1]["content"])
            return dict(text=json.dumps({"patterns": [dict(pattern_id="provider-label", failure_mechanism="Missed constraints",
                corrective_principle="Check explicit constraints", support_ids=data["residual_ids"], counterexample_ids=[], risk_ids=[], confidence=.9)]}), input_tokens=2, output_tokens=2)
        return dict(text="```" + (good if len(contexts) == 1 else other) + "```", input_tokens=2, output_tokens=2)
    broker = RequestBroker(contract=c, transport=transport, arm=arm, seed=81, ledger_writer=lambda row: None)
    solver, reflection = BenchmarkSolver(adapter, broker), ReflectionProvider(broker)
    pattern = PatternProvider(broker, json.loads((ROOT / c["pattern_prompt_path"]).read_bytes())["prompt"]) if c["arms"][arm][0] else None
    def official(**kwargs):
        contexts.append(kwargs["adapter"].optimization_context)
        return fixtures.import_frozen_gepa().optimize(**kwargs)
    optimizer = GEPATeamExposureOptimizer(evaluator=solver, reflection_lm=reflection,
        accounting_reader=reflection.accounting, run_root=tmp_path, optimize_fn=official)
    run = build_binary_orchestrator(benchmark=adapter, aggregation=fixtures.EquivalencePluralityAggregation(),
        examples=optimize, prompts=prompts, solver=solver, optimizer=optimizer, method=method, seed=81,
        shadow_loader=lambda: shadow, shadow_count=300, runtime_readiness=binding.blockers, pattern_provider=pattern)
    run.state.initialize()
    initial = run.state.snapshot()
    solver_calls_before = len(calls)
    with pytest.raises(SearchContractError, match="CANNOT_REBASE"):
        run.state.initialize()
    assert len(calls) == solver_calls_before
    result = asyncio.run(run.run(max_opportunities=3))
    assert result.transitions and result.trace[0].committed_candidate_id
    assert all(t.allocation_audit["realized_team_gain"] > 0 for t in result.trace if t.committed_candidate_id)
    assert run.state.initial_member_scores == initial.member_scores
    assert broker.usage["validation"] == broker.usage["test"] == 0
    if c["arms"][arm][1]:
        assert run.memory.private and all(e.action and e.lesson["successful_strategy"] for e in run.memory.private)
        assert all(e.outcome["team_score_delta"] > 0 and e.outcome["target_initial_margin"] >= 0 for e in run.memory.private)
    assert bool(broker.usage["pattern"]) == c["arms"][arm][0]
    assert all(("failure_mechanism" in context) == c["arms"][arm][0] for context in contexts)
    evidence = dict(arm=arm, method_identity=method.identity(), atomic_commits=len(result.transitions),
        opportunity_count=len(result.trace), ledger_counts=broker.usage, allocation_audits=[t.allocation_audit for t in result.trace],
        real_provider_calls=0, shadow_raw_search_leakage=0,
        initial_floor_preserved=True, official_gepa=True)
    destination = os.environ.get("FORMAL_V3_EVIDENCE_CAPTURE_DIR")
    if destination:
        path = Path(destination) / ("semantic_v21_" + arm + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
