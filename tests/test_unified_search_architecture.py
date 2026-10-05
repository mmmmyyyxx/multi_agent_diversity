"""Zero-provider contracts for the new team-search control flow."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.search.aggregation import (
    AggregationRequest, AggregationResponse, LLMAggregation,
    PluralityAggregation, RuntimeModelIdentity,
)
from multi_dataset_diverse_rl.search.benchmark import BenchmarkInput
from multi_dataset_diverse_rl.search.evaluation import CandidateEvaluationPipeline, TeamEvaluator
from multi_dataset_diverse_rl.search.history import HistoryState, NullMemoryProvider, NullPatternAnalyzer
from multi_dataset_diverse_rl.search.gepa import GEPADerivedConfig
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPAOptimizerConfig
from multi_dataset_diverse_rl.search.orchestrator import (
    OpportunityBuilder, StateAnalyzer, UnifiedSearchOrchestrator,
)
from multi_dataset_diverse_rl.search.policies import (
    CurrentEvidenceFeasibility, CurrentRoleEvidencePolicy, GlobalStopPolicy,
    ResponsibilitySignal, SearchStopPolicy, TargetDecision, TargetPolicyV1,
)
from multi_dataset_diverse_rl.search.schemas import (
    BenchmarkCapabilities, Diagnosis, EvidenceItem, ParsedOutput, SearchCandidate,
    SearchContractError, SearchMethodConfig, SearchResult, TeamEvaluation,
    TeamStateSnapshot, TransitionDecision,
)
from multi_dataset_diverse_rl.search.transition import (
    CommonSafeTransitionPolicy, TeamStateCommitter,
)
from multi_dataset_diverse_rl.experiment import (
    RuntimeContext, UnifiedExperimentInputs, UnifiedExperimentServices,
    run_experiment,
)
from scripts.replay_experiment import preflight


class FakeBenchmark:
    capabilities = BenchmarkCapabilities(True, True, True)
    preferred_aggregation = "plurality"

    def format_input(self, item):
        return item.problem

    def parse_member_output(self, raw, item):
        del item
        return ParsedOutput(raw.removeprefix("FINAL_ANSWER: ").strip(), bool(raw.strip()))

    parse_output = parse_member_output

    def score_member_output(self, parsed, gold):
        return float(parsed.valid and parsed.answer == gold)

    def build_task_feedback(self, parsed, gold):
        return "correct" if self.score_member_output(parsed, gold) else "incorrect"


def test_plurality_and_llm_use_one_benchmark_evaluator_without_gold_leak() -> None:
    item = BenchmarkInput("q1", "public problem", "one answer")
    outputs = ("red", "blue", "red", "green", "blue")
    benchmark = FakeBenchmark()
    seen = []
    usage = []

    async def fake_provider(request):
        seen.append(request)
        return AggregationResponse("red", 10, 2)

    llm = LLMAggregation(
        runtime=RuntimeModelIdentity("optimizer-model", 81, "greedy-v1"),
        provider=fake_provider, usage_observer=lambda *row: usage.append(row),
    )
    plurality = asyncio.run(TeamEvaluator(benchmark, PluralityAggregation()).evaluate_row(
        item=item, member_outputs=outputs, gold="red",
    ))
    llm_row = asyncio.run(TeamEvaluator(benchmark, llm).evaluate_row(
        item=item, member_outputs=outputs, gold="red",
    ))
    assert plurality.aggregate_score == 0  # two-way tie abstains
    assert llm_row.aggregate_score == 1
    assert plurality.aggregation_calls == 0
    assert llm_row.aggregation_calls == 1 and llm_row.aggregation_tokens == 12
    assert len(seen) == 1
    assert seen[0].model == "optimizer-model" and seen[0].role == "aggregator"
    assert seen[0].prompt.count("red") == 2
    assert "public problem" in seen[0].prompt
    assert "gold" not in seen[0].prompt.lower()
    assert "reward" not in seen[0].prompt.lower()
    assert "correct_vector" not in seen[0].prompt.lower()
    assert "preferred" not in seen[0].prompt.lower()
    assert usage == [("team_aggregation", 10, 2)]
    solver_key = replace(seen[0], role="solver").cache_identity()
    assert solver_key != seen[0].cache_identity()
    assert replace(seen[0], model="other-model").cache_identity() != seen[0].cache_identity()
    with pytest.raises(ValueError, match="evaluation-only"):
        BenchmarkInput("q2", "public problem", "one answer",
                       public_context={"gold": "red"})
    runtime = RuntimeContext(
        81, "fake", "solver", "optimizer-model", "optimizer-model",
        "run-sha", "offline-fake", "cache", "ledger",
    )
    assert RuntimeModelIdentity.from_runtime_context(
        runtime, decoding_identity="greedy-v1",
    ).optimizer_model == seen[0].model

    async def cached_provider(request):
        del request
        return AggregationResponse("red", provider_called=False)

    cached = asyncio.run(TeamEvaluator(benchmark, LLMAggregation(
        runtime=RuntimeModelIdentity.from_runtime_context(
            runtime, decoding_identity="greedy-v1",
        ), provider=cached_provider,
    )).evaluate_row(item=item, member_outputs=outputs, gold="red"))
    assert cached.aggregation_calls == cached.aggregation_tokens == 0


def test_llm_aggregation_capability_does_not_invent_responsibility() -> None:
    class FreeForm(FakeBenchmark):
        capabilities = BenchmarkCapabilities(False, False, False)
        preferred_aggregation = "llm"

    class State:
        def snapshot(self):
            return TeamStateSnapshot("root", ("p",) * 5)

    orchestrator = UnifiedSearchOrchestrator(
        method=SearchMethodConfig(aggregation_policy="equal_status_llm_aggregation_v1"),
        benchmark=FreeForm(), aggregation=object(), state=State(),
        analyzer=object(), opportunities=object(), engine=object(),
        evaluation=object(), transition=object(), gate=object(),
        committer=object(),
    )
    with pytest.raises(SearchContractError, match="SCIENTIFIC_DECISION_REQUIRED"):
        asyncio.run(orchestrator.run(max_opportunities=1))

    async def fake_provider(request):
        del request
        return AggregationResponse("answer")

    bbh_orchestrator, _, _ = _orchestrator()
    bbh_orchestrator.method = replace(
        bbh_orchestrator.method,
        aggregation_policy="equal_status_llm_aggregation_v1",
    )
    bbh_orchestrator.aggregation = LLMAggregation(
        runtime=RuntimeModelIdentity("optimizer", 81, "greedy"),
        provider=fake_provider,
    )
    with pytest.raises(SearchContractError, match="SCIENTIFIC_DECISION_REQUIRED"):
        asyncio.run(bbh_orchestrator.run(max_opportunities=1))


def _rows():
    rows = []
    for group, n in (("REPAIR", 8), ("PRESERVATION", 4)):
        for index in range(n):
            roles = {group}
            if group == "REPAIR":
                roles.add("TEAM_HARD")
                roles.add("direct_flip")
            rows.append(EvidenceItem(
                f"{group}-{index}", "optimize", frozenset(roles),
                {"lane": "direct_flip", "team_disagreement": index % 3},
            ))
    return tuple(rows)


class FakeState:
    def __init__(self):
        self.prompts = ("parent",) * 5

    def snapshot(self):
        identity = hashlib.sha256(repr(self.prompts).encode()).hexdigest()
        return TeamStateSnapshot(identity, self.prompts)

    def replace_member(self, member_id, prompt):
        self.prompts = tuple(prompt if i == member_id else value
                             for i, value in enumerate(self.prompts))
        return self.snapshot()

    def restore(self, state):
        self.prompts = state.member_prompts


class FakeAnalyzer:
    def analyze(self, state, history):
        del state, history
        return Diagnosis(responsibility={0: ResponsibilitySignal(0, 2, 0, 0, "direct_flip")})


class FakeSource:
    def for_member(self, state, diagnosis, member_id):
        del state, diagnosis, member_id
        return _rows()


class FakeEngine:
    async def search(self, opportunity, context):
        assert context.memory_view == {} and context.pattern_view == {}
        assert len(opportunity.evidence.search_validation_evidence) == 12
        assert len(opportunity.evidence.team_probe_evidence) == 12
        assert opportunity.evidence.search_validation_evidence is not opportunity.evidence.team_probe_evidence
        return SearchResult((SearchCandidate("child", "improved"),), "SATURATION_REACHED")


class FakeEvaluator:
    async def active(self, opportunity):
        del opportunity
        return TeamEvaluation(0.4, None, (0.4,) * 5)

    async def team_probe(self, opportunity, candidate):
        del opportunity, candidate
        return TeamEvaluation(0.5, None, (0.5,) * 5)

    async def full(self, opportunity, candidate):
        del opportunity, candidate
        return TeamEvaluation(0.6, None, (0.5, 0.4, 0.4, 0.4, 0.4))


class FakePromotion:
    def select(self, rows):
        return tuple(row.candidate_id for row, _ in rows)


class FakeGate:
    async def check(self, opportunity, candidate):
        del opportunity, candidate
        return True


def _orchestrator(*, target=None, evidence=None, transition=None, memory=None, state=None):
    state = state or FakeState()
    history = HistoryState()
    builder = OpportunityBuilder(
        source=FakeSource(), feasibility=CurrentEvidenceFeasibility(),
        target=target or TargetPolicyV1(),
        evidence=evidence or CurrentRoleEvidencePolicy(),
    )
    orchestrator = UnifiedSearchOrchestrator(
        method=SearchMethodConfig(), benchmark=FakeBenchmark(),
        aggregation=PluralityAggregation(), state=state,
        analyzer=StateAnalyzer(FakeAnalyzer(), NullPatternAnalyzer()),
        opportunities=builder, engine=FakeEngine(),
        evaluation=CandidateEvaluationPipeline(FakeEvaluator(), FakePromotion()),
        transition=transition or CommonSafeTransitionPolicy(),
        gate=FakeGate(), committer=TeamStateCommitter(state),
        history=history, memory=memory or NullMemoryProvider(),
        stop=GlobalStopPolicy(),
    )
    return orchestrator, state, history


def test_replaceable_components_and_atomic_commit() -> None:
    orchestrator, state, history = _orchestrator()
    result = asyncio.run(orchestrator.run(max_opportunities=1))
    assert len(result.trace) == len(result.transitions) == 1
    assert result.trace[0].committed_candidate_id == "child"
    assert state.prompts == ("improved", "parent", "parent", "parent", "parent")
    assert history.latest_member_transition(0) == result.transitions[0]
    assert history.commit_counts == {0: 1}

    class AlternateTarget(TargetPolicyV1):
        def select(self, state, diagnosis, feasible_members, history):
            answer = super().select(state, diagnosis, feasible_members, history)
            return TargetDecision(answer.selected_member, answer.eligible_members,
                                  answer.target_scores, "alternate")

    class AlternateEvidence(CurrentRoleEvidencePolicy):
        def build(self, state, diagnosis, member_id, rows):
            return super().build(state, diagnosis, member_id, rows)

    class AlternateTransition(CommonSafeTransitionPolicy):
        def select(self, parent, candidates):
            return TransitionDecision(None, "test veto")

    replacement, state2, _ = _orchestrator(
        target=AlternateTarget(), evidence=AlternateEvidence(),
        transition=AlternateTransition(),
    )
    result2 = asyncio.run(replacement.run(max_opportunities=1))
    assert result2.trace[0].committed_candidate_id is None
    assert state2.prompts == ("parent",) * 5


def test_unified_production_entrypoint_uses_single_search_loop() -> None:
    orchestrator, state, _ = _orchestrator()
    runtime = RuntimeContext(
        81, "fake", "solver", "optimizer", "optimizer",
        "run-sha", "offline-fake", "cache", "ledger",
    )
    result = asyncio.run(run_experiment(
        SearchMethodConfig(), runtime, UnifiedExperimentInputs(1),
        UnifiedExperimentServices(orchestrator),
    ))
    assert result.trace[0].committed_candidate_id == "child"
    assert state.prompts[0] == "improved"


def test_unified_cli_preflight_holds_before_provider() -> None:
    manifest = {
        "scientific": {"method": "unified_team_prompt_search_v1"},
        "runtime": {
            "seed": 81, "provider_profile": "fake", "solver_model": "solver",
            "optimizer_model": "optimizer", "evaluator_model": "optimizer",
            "run_identity_sha256": "fake-run", "authorization_identity": "none",
            "cache_identity": "fake-cache", "ledger_identity": "fake-ledger",
        },
    }
    result = preflight(manifest)
    assert result["gate"] == "HOLD"
    assert result["aggregation_model"] == "optimizer"
    assert result["provider_attempts"] == result["validation_calls"] == result["test_calls"] == 0


def test_component_identity_changes_with_policy_and_rejects_unknown_fields() -> None:
    default = SearchMethodConfig()
    assert replace(default, transition_policy="new_transition_v2").identity() != default.identity()
    assert replace(default, aggregation_policy="equal_status_llm_aggregation_v1").identity() != default.identity()
    with pytest.raises(SearchContractError, match="unknown unified method component"):
        SearchMethodConfig.from_mapping({"method": default.method, "layer2_mode": True})
    assert GEPADerivedConfig().identity() != GEPAOptimizerConfig().identity()
    assert GEPADerivedConfig(k_local_return=5).identity() != GEPADerivedConfig().identity()
    with pytest.raises(ValueError):
        GEPAOptimizerConfig(k_local_return=5)


def test_global_saturation_counts_complete_no_commit_epochs_only() -> None:
    stop = GlobalStopPolicy(no_commit_patience=2)
    assert stop.observe_opportunity(parent_state_id="root", eligible_members=(0, 1),
                                    selected_member=0, local_update=False,
                                    committed=False) is None
    assert stop.no_commit_epochs == 0
    assert stop.observe_opportunity(parent_state_id="root", eligible_members=(0, 1),
                                    selected_member=1, local_update=True,
                                    committed=False) is None
    assert stop.no_commit_epochs == 1
    assert stop.observe_opportunity(parent_state_id="root", eligible_members=(0, 1),
                                    selected_member=0, local_update=True,
                                    committed=True) is None
    assert stop.no_commit_epochs == 0
    assert stop.observe_opportunity(parent_state_id="child", eligible_members=(0,),
                                    selected_member=0, local_update=False,
                                    committed=False) is None
    assert stop.observe_opportunity(parent_state_id="child", eligible_members=(0,),
                                    selected_member=0, local_update=False,
                                    committed=False) == "SATURATION_REACHED"
    assert SearchStopPolicy(3).no_update_patience == 3


def test_provider_emergency_ceiling_stops_before_search() -> None:
    orchestrator, state, history = _orchestrator()
    orchestrator.provider_call_reader = lambda: (
        orchestrator.method.global_stop.emergency_max_provider_calls
    )
    result = asyncio.run(orchestrator.run(max_opportunities=1))
    assert result.stop_reason == "EMERGENCY_PROVIDER_CALL_CEILING"
    assert result.trace == () and history.transitions == []
    assert state.prompts == ("parent",) * 5


def test_atomic_rollback_when_memory_observation_fails() -> None:
    class BrokenMemory(NullMemoryProvider):
        def observe_transition(self, transition):
            raise RuntimeError("fake memory failure")

    orchestrator, state, history = _orchestrator(memory=BrokenMemory())
    before = state.snapshot().team_state_id
    with pytest.raises(RuntimeError, match="fake memory failure"):
        asyncio.run(orchestrator.run(max_opportunities=1))
    assert state.snapshot().team_state_id == before
    assert history.transitions == [] and history.target_counts == {}


def test_full_dynamic_fake_saturation_replay() -> None:
    """Two commits change the parent; two full no-commit epochs then saturate."""
    class SequentialEngine:
        def __init__(self):
            self.parents = []

        async def search(self, opportunity, context):
            self.parents.append((opportunity.parent_state_id,
                                 opportunity.parent_prompt,
                                 context.history.latest_member_transition(0)))
            index = len(self.parents)
            candidates = ((SearchCandidate(f"child-{index}", f"improved-{index}"),)
                          if index <= 2 else ())
            return SearchResult(candidates, "SATURATION_REACHED")

    orchestrator, state, history = _orchestrator()
    class RecordingAnalyzer(FakeAnalyzer):
        def __init__(self):
            self.parents = []

        def analyze(self, state, history):
            self.parents.append(state.team_state_id)
            return super().analyze(state, history)

    responsibility = RecordingAnalyzer()
    orchestrator.analyzer = StateAnalyzer(responsibility)
    engine = SequentialEngine()
    orchestrator.engine = engine
    first = asyncio.run(orchestrator.run(max_opportunities=10))
    assert first.stop_reason == "SATURATION_REACHED"
    assert len(first.trace) == 4 and len(first.transitions) == 2
    assert [row.committed_candidate_id for row in first.trace] == [
        "child-1", "child-2", None, None,
    ]
    assert first.trace[0].child_state_id == first.trace[1].parent_state_id
    assert all(row.parent_state_id == first.trace[1].child_state_id
               for row in first.trace[2:])
    assert [row[1] for row in engine.parents] == [
        "parent", "improved-1", "improved-2", "improved-2",
    ]
    assert responsibility.parents == [row.parent_state_id for row in first.trace]
    assert engine.parents[0][2] is None
    assert engine.parents[1][2] == first.transitions[0]
    assert engine.parents[2][2] == first.transitions[1]
    assert history.commit_counts == {0: 2} and history.failure_counts == {0: 2}
    assert orchestrator.stop.no_commit_epochs == 2
    assert len({record.child_state_id for record in first.transitions}) == 2
    assert first.final_state_id == state.snapshot().team_state_id
    replay, _, _ = _orchestrator()
    replay.engine = SequentialEngine()
    second = asyncio.run(replay.run(max_opportunities=10))
    assert second.final_state_id == first.final_state_id
    assert second.stop_reason == first.stop_reason


def test_new_search_package_has_no_historical_ownership_interfaces() -> None:
    root = Path(__file__).resolve().parents[1] / "multi_dataset_diverse_rl" / "search"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py")
                       if path.name not in {"legacy_bbh_replay.py", "gepa.py",
                                            "current_bbh_runtime.py"})
    for forbidden in ("Layer1Backend", "Layer2OptimizationRequest",
                      "Layer2EvidencePromptOptimizer", "TeamSearchController"):
        assert forbidden not in source
    for adapter in ("gepa.py", "current_bbh_runtime.py"):
        adapter_source = (root / adapter).read_text(encoding="utf-8")
        assert "TeamSearchController" not in adapter_source
        assert "Layer2EvidenceRequestBuilder" not in adapter_source.replace(
            '"Layer2EvidenceRequestBuilder"', ""
        )
