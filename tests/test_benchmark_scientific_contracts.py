"""Synthetic scientific-contract replay. No raw datasets, retrieval or real clients."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.benchmarks import (
    BENCHMARKS, HotpotQAAnswerAdapter, HoVerBenchmarkAdapter, IFBenchBenchmarkAdapter,
    MATHBenchmarkAdapter, PUPABenchmarkAdapter, benchmark_preflight,
)
from multi_dataset_diverse_rl.benchmarks.hover import RetrievedEvidence, normalize_title
from multi_dataset_diverse_rl.benchmarks.ifbench import ConstraintReference, response_variants
from multi_dataset_diverse_rl.benchmarks.math_domain_v2 import domain_matrix as equivalence_matrix
from multi_dataset_diverse_rl.benchmarks.protocols import PROTOCOLS, protocol_input
from multi_dataset_diverse_rl.benchmarks.pupa import PAPILLONMemberPipeline, fake_pupa_judge_score
from multi_dataset_diverse_rl.peer_state import build_team_vote_state, build_peer_vote_context
from multi_dataset_diverse_rl.responsibility import ResponsibilityState, compute_member_aware_repair_opportunity
from multi_dataset_diverse_rl.team_search.system_runtime import freeze_current_responsibility
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import build_primary_responsibility_summaries
from multi_dataset_diverse_rl.search.aggregation import AggregationResponse, LLMAggregation, RuntimeModelIdentity
from multi_dataset_diverse_rl.search.benchmark import BenchmarkInput
from multi_dataset_diverse_rl.search.binary_responsibility import (
    BinaryPluralityResponsibilityAnalyzer, BinaryPluralityObservation, binary_plurality_snapshot,
    observation_from_outputs,
)
from multi_dataset_diverse_rl.search.current_bbh import CurrentBBHStateSource, PluralityResponsibilityAnalyzer
from multi_dataset_diverse_rl.search.evaluation import TeamEvaluator
from multi_dataset_diverse_rl.search.history import HistoryState
from multi_dataset_diverse_rl.search.information_firewall import EVALUATOR_ONLY_KEYS
from multi_dataset_diverse_rl.search.policies import TargetPolicyV1
from multi_dataset_diverse_rl.search.schemas import BenchmarkCapabilities, ParsedOutput, SearchContractError
from multi_dataset_diverse_rl.search.scientific_aggregation import (
    NormalizedAnswerPluralityAggregation, HoVerEvidenceAggregation, IFBenchRawResponseAggregation,
    EquivalencePluralityAggregation, equivalence_classes,
)


def item(adapter, **hidden):
    field, = PROTOCOLS[adapter.benchmark_id].public_solver_fields
    return protocol_input(adapter.benchmark_id, "synthetic-id", {field: "Synthetic public input", **hidden},
                          adapter.output_contract)


class ExactChecker:
    def __init__(self, identity):
        self.identity = identity
    def build_description(self, expected=None, prompt=None):
        if expected is not None:
            self.expected = expected
        return "Synthetic checker description"
    def get_instruction_args(self):
        return {}
    def check_following(self, response):
        return response == self.expected


def binary_diagnosis(adapter, parsed, successes, classes=None):
    state = binary_plurality_snapshot(state_id="synthetic-state", member_prompts=("procedure",) * 5,
        observations=(BinaryPluralityObservation("synthetic-id",
            tuple(classes or [p.answer for p in parsed]), tuple(p.valid for p in parsed),
            tuple(bool(s) for s in successes)),), capabilities=adapter.capabilities)
    return BinaryPluralityResponsibilityAnalyzer(adapter.capabilities).analyze(state, HistoryState())


def test_hotpotqa_task_protocol_and_normalization():
    adapter = HotpotQAAnswerAdapter()
    q = item(adapter, answer="HIDDEN_REFERENCE_SENTINEL", supporting_facts="HIDDEN_FACTS_SENTINEL")
    assert PROTOCOLS["hotpotqa"].task_contract_id == "HOTPotQA_GEPA_ANSWER_V1"
    assert "HIDDEN" not in adapter.format_input(q)
    raw = ("FINAL_ANSWER: The Atlas!", "FINAL_ANSWER: atlas", "FINAL_ANSWER: Atlas",
           "FINAL_ANSWER: other", "invalid")
    result = asyncio.run(TeamEvaluator(adapter, NormalizedAnswerPluralityAggregation()).evaluate_row(
        item=q, member_outputs=raw, gold="Atlas"))
    assert result.aggregate_score == 1 and result.member_scores == (1, 1, 1, 0, 0)
    parsed = tuple(adapter.parse_member_output(r, q) for r in raw)
    assert len(binary_diagnosis(adapter, parsed, result.member_scores).responsibility) == 5
    assert PROTOCOLS["hotpotqa"].system_dependencies[0].blocker == "SYSTEM_DEPENDENCY_RETRIEVAL_NOT_FROZEN"


def test_hotpotqa_no_supporting_fact_primary_metric_and_f1_not_binary_success():
    a = HotpotQAAnswerAdapter()
    p = a.parse_member_output("FINAL_ANSWER: red blue", item(a))
    assert a.answer_f1(p, "red green") == .5
    assert a.score_member_output(p, "red green") == 0


def test_hover_retrieval_coverage_not_verdict():
    a = HoVerBenchmarkAdapter()
    q = item(a)
    p = a.parse_member_output('{"titles": ["The Atlas! | passage", "Beta"]}', q)
    coverage = a.evaluate(p, RetrievedEvidence(("Atlas", "Gamma")))
    assert not coverage.coverage_success and coverage.missing_gold_titles == ("gamma",)
    assert normalize_title("Caf\u00e9") == normalize_title("Cafe\u0301")
    for verdict in ('{"verdict": "SUPPORTED"}', '{"label": "REFUTED", "titles": ["Atlas"]}'):
        assert not a.parse_member_output(verdict, q).valid


def test_hover_evidence_aggregation_union_guard_and_coverage_e2e():
    a = HoVerBenchmarkAdapter()
    q = item(a, gold_titles="HIDDEN_TITLE_SENTINEL")
    seen = []
    async def fake(request):
        seen.append(request)
        return AggregationResponse('{"titles": ["atlas", "beta"]}', 2, 1)
    agg = HoVerEvidenceAggregation(runtime=RuntimeModelIdentity("optimizer", 1, "greedy"), provider=fake)
    outputs = ('{"titles": ["atlas"]}', '{"titles": ["beta"]}', '{"titles": ["atlas"]}',
               '{"titles": ["beta"]}', '{"titles": ["gamma"]}')
    result = asyncio.run(TeamEvaluator(a, agg).evaluate_row(item=q, member_outputs=outputs,
                         gold=RetrievedEvidence(("Atlas", "Beta"))))
    assert result.aggregate_score == 1 and result.aggregation_calls == 1
    assert seen[0].role == "team_aggregation" and seen[0].model == "optimizer"
    assert "HIDDEN_TITLE_SENTINEL" not in seen[0].prompt
    assert replace(seen[0], role="solver").cache_identity() != seen[0].cache_identity()
    async def invented(request):
        return AggregationResponse('{"titles": ["unproposed"]}')
    bad = asyncio.run(HoVerEvidenceAggregation(runtime=agg.runtime, provider=invented).aggregate(
        item=q, member_outputs=outputs, benchmark=a))
    assert not bad.parsed_output.valid and bad.diagnostics["invalid_reason"].endswith("OUTSIDE_MEMBER_UNION")
    async def too_many(request):
        return AggregationResponse(json.dumps({"titles": [f"title {i}" for i in range(25)]}))
    large = (json.dumps({"titles": [f"title {i}" for i in range(25)]}),) * 5
    bad = asyncio.run(HoVerEvidenceAggregation(runtime=agg.runtime, provider=too_many).aggregate(
        item=q, member_outputs=large, benchmark=a))
    assert not bad.parsed_output.valid and bad.diagnostics["invalid_reason"].endswith("CEILING_EXCEEDED")


def test_ifbench_raw_response_output_and_eight_variants():
    a = IFBenchBenchmarkAdapter({"fixture": ExactChecker})
    raw = "header\n*core*\nfooter"
    parsed = a.parse_member_output(raw, item(a))
    assert parsed.answer == raw and parsed.valid
    assert response_variants("header\n*core*\nfooter") == (
        "header\n*core*\nfooter", "header\ncore\nfooter", "*core*\nfooter", "header\n*core*",
        "*core*", "core\nfooter", "header\ncore", "core")
    for variant in response_variants(raw):
        if variant.strip():
            reference = ConstraintReference("public", ("fixture",), ({"expected": variant},))
            assert a.evaluate(parsed, reference).score == 1
    reference = ConstraintReference("public", ("fixture", "fixture"),
                                    ({"expected": "core", "unused": None}, {"expected": "not a variant"}))
    receipt = a.evaluate(parsed, reference)
    assert (receipt.score, receipt.success_vector, receipt.num_satisfied, receipt.num_constraints) == (.5, (True, False), 1, 2)
    assert reference.kwargs[0]["unused"] is None
    assert not a.parse_member_output(" \n ", item(a)).valid


def test_ifbench_hidden_fields_not_in_solver_input_or_aggregator_request():
    a = IFBenchBenchmarkAdapter({"fixture": ExactChecker})
    q = item(a, instruction_id_list=["HIDDEN_INSTRUCTION_SENTINEL"],
             kwargs=[{"private": "HIDDEN_KWARGS_SENTINEL"}], success_vector=[True])
    seen = []
    async def fake(request):
        seen.append(request)
        return AggregationResponse("core\n", 2, 1)
    agg = IFBenchRawResponseAggregation(runtime=RuntimeModelIdentity("optimizer", 1, "greedy"), provider=fake)
    reference = ConstraintReference("Synthetic public input", ("fixture", "fixture"),
                                    ({"expected": "core"}, {"expected": "absent"}))
    result = asyncio.run(TeamEvaluator(a, agg).evaluate_row(item=q, member_outputs=("core",) * 5, gold=reference))
    assert result.aggregate_score == .5 and result.aggregate_valid
    assert "HIDDEN" not in seen[0].prompt and seen[0].role == "team_aggregation"
    final = asyncio.run(agg.aggregate(item=q, member_outputs=("core",) * 5, benchmark=a))
    assert final.raw_output == final.parsed_output.answer == "core\n"
    with pytest.raises(SearchContractError, match="IFBENCH_LOCAL_EVALUATOR_DEPENDENCIES_NOT_FROZEN"):
        IFBenchBenchmarkAdapter().evaluate(a.parse_member_output("core", q), reference)


@pytest.mark.parametrize("left,right", [
    ("1/2", ".5"), ("x=2", "x=2"), (r"\sqrt{4}", "2"), (r"\frac{1}{2}", ".5"),
    (r"\boxed{2}", "2"), (r"\{1,2\}", r"\{2,1\}"),
])
def test_math_equivalence_cases(left, right):
    a = MATHBenchmarkAdapter()
    p = a.parse_member_output("reasoning not scored\n### " + left, item(a))
    assert p.valid and a.score_member_output(p, right) == 1


def test_math_strict_set_tuple_and_wrong_answer():
    a = MATHBenchmarkAdapter()
    assert not a.parse_member_output("### (1,2)", item(a)).valid
    assert not a.equivalent(r"\{1,2\}", "(1,2)")
    assert not a.equivalent("1/2", "3")


def test_math_invalid_parse_and_timeout_fail_closed(monkeypatch):
    a = MATHBenchmarkAdapter()
    q = item(a)
    for raw in ("### nonsense", "###", "### 2\nafter",
                "### 2\n### 3", "Unmarked prose 2", "### unknown prose 2",
                "### 2+", r"### \frac{1}{"):
        assert not a.parse_member_output(raw, q).valid
    import multi_dataset_diverse_rl.benchmarks.math_domain_v2 as module
    equivalence_matrix.cache_clear()
    def timed_out(*args, **kwargs):
        import subprocess
        raise subprocess.TimeoutExpired("offline-worker", 8)
    monkeypatch.setattr(module.subprocess, "run", timed_out)
    assert not a.parse_member_output("### 938", q).valid
    with pytest.raises(SearchContractError, match="REFERENCE_UNSCORABLE"):
        a.score_member_output(ParsedOutput("938", True), "938")
    with pytest.raises(SearchContractError, match="MATH_EVALUATOR_TIMEOUT"):
        equivalence_matrix(("938",))


def test_math_equivalence_plurality_tie_invalid_and_e2e():
    a = MATHBenchmarkAdapter()
    q = item(a)
    agg = EquivalencePluralityAggregation()
    outputs = ("### 1/2", "### .5", r"### \frac{1}{2}",
               "### 3", "invalid")
    result = asyncio.run(agg.aggregate(item=q, member_outputs=outputs, benchmark=a))
    assert result.raw_output == outputs[0] and result.diagnostics["winner_member"] == 0
    assert result.diagnostics["invalid_abstentions"] == 1
    assert a.score_member_output(result.parsed_output, ".5") == 1
    parsed = tuple(a.parse_member_output(raw, q) for raw in outputs)
    success = tuple(a.score_member_output(p, ".5") for p in parsed)
    classes = ("half", "half", "half", "three", "invalid")
    assert len(binary_diagnosis(a, parsed, success, classes).responsibility) == 5
    observation = observation_from_outputs(benchmark=a, item=q, member_outputs=outputs, gold=".5")
    assert observation.member_success == (True, True, True, False, False)
    assert observation.vote_classes[:3] == ("class:0",) * 3
    tie = ("### 1/2", "### .5", "### 3", "### 3", "invalid")
    assert not asyncio.run(agg.aggregate(item=q, member_outputs=tie, benchmark=a)).parsed_output.valid


def test_equivalence_non_transitive_guard():
    relation = lambda a, b: abs(int(a) - int(b)) <= 1
    with pytest.raises(SearchContractError, match="EQUIVALENCE_RELATION_INCONSISTENT"):
        equivalence_classes(("1", "2", "3"), relation)
    class Adapter:
        capabilities = BenchmarkCapabilities(True, True, True)
        def parse_member_output(self, raw, item):
            return ParsedOutput(raw, True)
    result = asyncio.run(EquivalencePluralityAggregation(relation).aggregate(
        item=item(MATHBenchmarkAdapter()), member_outputs=("1", "2", "3", "3", "3"), benchmark=Adapter()))
    assert not result.parsed_output.valid
    assert result.diagnostics["abstention_reason"] == "EQUIVALENCE_RELATION_INCONSISTENT"
    with pytest.raises(SearchContractError, match="EQUIVALENCE_EVALUATOR_FAILURE"):
        equivalence_classes(("1",), lambda a, b: 1 / 0)


def test_math_worker_real_deadline_and_failure(monkeypatch):
    import multi_dataset_diverse_rl.benchmarks.math_domain_v2 as module
    # A real subprocess is started and killed on deadline, including on Windows.
    monkeypatch.setitem(module.SETTINGS, "process_deadline", 0.000001)
    with pytest.raises(SearchContractError, match="MATH_EVALUATOR_TIMEOUT"):
        equivalence_matrix(("23917",))
    import subprocess
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 1, "{}", ""))
    with pytest.raises(SearchContractError, match="MATH_EVALUATOR_FAILURE"):
        equivalence_matrix(("23918",))


@pytest.mark.parametrize("key", ["hover", "ifbench", "pupa"])
def test_binary_plurality_responsibility_capability_gate(key):
    with pytest.raises(SearchContractError, match="BINARY_PLURALITY_RESPONSIBILITY_CAPABILITY_REQUIRED"):
        BinaryPluralityResponsibilityAnalyzer(BENCHMARKS[key].capabilities)


def test_binary_plurality_requires_every_capability_and_boolean_not_fraction():
    for flag in ("supports_vote_classes", "supports_plurality_margin", "supports_boolean_member_success",
                 "supports_current_responsibility"):
        with pytest.raises(SearchContractError, match="CAPABILITY_REQUIRED"):
            BinaryPluralityResponsibilityAnalyzer(replace(BenchmarkCapabilities(True, True, True), **{flag: False}))
    with pytest.raises(SearchContractError, match="OBSERVATION_INVALID"):
        binary_plurality_snapshot(state_id="synthetic", member_prompts=("p",) * 5,
            observations=(BinaryPluralityObservation("row", ("a",) * 5, (True,) * 5, (.5,) * 5),),
            capabilities=BenchmarkCapabilities(True, True, True))


def bbh_parity_receipt():
    ids = tuple(f"synthetic-{i:04d}" for i in range(4 ** 5))
    states = tuple(build_team_vote_state(question_hash=row_id, gold_answer="A",
        answers=answers, valid_vector=tuple(bool(x) for x in answers))
        for row_id, answers in zip(ids, itertools.product(("A", "B", "C", ""), repeat=5), strict=True))
    opportunities = {row.question_hash: tuple(compute_member_aware_repair_opportunity(
        team_state=row, peer_context=build_peer_vote_context(row, m)) for m in range(5)) for row in states}
    class System:
        responsibility_state = ResponsibilityState(updates_since_selected_by_agent={m: 0 for m in range(5)})
        agents = tuple(SimpleNamespace(current_prompt="synthetic") for _ in range(5))
        def team_prompt_state_hash(self):
            return "synthetic-parent"
        def current_states_and_opportunities(self):
            return states, {}, opportunities
    system = System()
    history = HistoryState()
    history.failure_counts.update({0: 0, 1: 1, 2: 2, 3: 3, 4: 4})
    snapshot = CurrentBBHStateSource(system).snapshot()
    diagnosis = PluralityResponsibilityAnalyzer(system).analyze(snapshot, history)
    old = freeze_current_responsibility(system, update_index=0)
    reference = build_primary_responsibility_summaries(assigned=old.assigned,
        current_margin_by_question=old.current_margin_by_question,
        failure_count_by_member=history.failure_counts)
    new_vectors = [(s.direct_count, s.near_margin_count, s.coverage_count, s.raw_value, s.primary_lane)
                   for _, s in sorted(diagnosis.responsibility.items())]
    old_vectors = [(s.direct_count, s.near_margin_count, s.coverage_count, s.primary_score, s.primary_lane)
                   for s in reference]
    assert new_vectors == old_vectors
    for member in range(5):
        assert diagnosis.benchmark_signals["assigned"][member] == tuple(old.assigned[member])
    receipts = []
    for mask in itertools.product((False, True), repeat=5):
        feasible = tuple(i for i in range(5) if mask[i])
        decision = TargetPolicyV1().select(state=snapshot, diagnosis=diagnosis, feasible_members=feasible, history=history)
        scores = {s.member_id: s.primary_score / (1 + history.failure_counts[s.member_id])
                  for s in reference if s.member_id in feasible}
        assert decision.target_scores == scores
        expected = min(scores, key=lambda m: (-scores[m], m)) if scores else None
        assert decision.selected_member == expected
        receipts.append(asdict(decision))
    return {"pass": True, "vote_states": len(states), "feasibility_masks": len(receipts),
        "fields": ["D", "N", "C", "V", "lane", "raw_legal_portfolios", "target_scores", "selected_member"],
        "diagnosis_sha256": hashlib.sha256(json.dumps(new_vectors, sort_keys=True).encode()).hexdigest(),
        "target_sha256": hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest(),
        "reference": "frozen V4 freeze_current_responsibility / build_primary_responsibility_summaries"}


def test_binary_plurality_responsibility_bbh_parity():
    assert bbh_parity_receipt()["pass"]


def test_pupa_member_pipeline_preserves_artifacts_scorer_and_team_hold():
    calls = []
    pipeline = PAPILLONMemberPipeline(
        trusted_redactor=lambda query: calls.append("redact") or "synthetic redacted request",
        untrusted_model=lambda request: calls.append("untrusted") or "synthetic untrusted response",
        trusted_synthesis=lambda query, request, response: calls.append("synthesize") or "synthetic synthesis")
    artifacts = pipeline.run("synthetic private query")
    assert calls == ["redact", "untrusted", "synthesize"]
    assert PROTOCOLS["pupa"].public_solver_fields == ("llm_request",)
    assert PROTOCOLS["pupa"].trusted_pipeline_input_fields == ("user_query",)
    assert all(getattr(artifacts, k) for k in ("llm_request", "llm_response", "response"))
    quality_calls = []
    def quality(a, b):
        quality_calls.append((a, b))
        return False
    score = fake_pupa_judge_score(artifacts, reference_response="synthetic reference", pii_str="one||one||two",
        quality_judge=quality, leakage_judge=lambda request, pii: 1)
    assert len(quality_calls) == 2 and quality_calls[0] == quality_calls[1][::-1] and score == .75
    assert fake_pupa_judge_score(artifacts, reference_response="synthetic reference", pii_str="",
        quality_judge=lambda a, b: True, leakage_judge=lambda request, pii: 0, trace=True) is True
    seen = []
    async def forbidden(request):
        seen.append(request)
        raise AssertionError("provider accessed")
    with pytest.raises(SearchContractError, match="PUPA_TEAM_AGGREGATION_POLICY_NOT_FROZEN"):
        asyncio.run(LLMAggregation(runtime=RuntimeModelIdentity("optimizer", 1, "greedy"), provider=forbidden).aggregate(
            item=item(PUPABenchmarkAdapter()), member_outputs=(artifacts.response,) * 5, benchmark=PUPABenchmarkAdapter()))
    assert not seen
    broken = PAPILLONMemberPipeline(trusted_redactor=lambda query: 1 / 0,
        untrusted_model=lambda request: (_ for _ in ()).throw(AssertionError("untrusted should not run")),
        trusted_synthesis=lambda *args: "unreachable").run("synthetic private query")
    assert (broken.llm_request, broken.llm_response, broken.response) == ("", "", "")
    assert broken.pipeline_status == "INVALID_PIPELINE_EXCEPTION"


@pytest.mark.parametrize("key", sorted(EVALUATOR_ONLY_KEYS))
def test_recursive_information_firewall(key):
    with pytest.raises(SearchContractError, match="INFERENCE_EVALUATOR_CONTEXT_FORBIDDEN"):
        BenchmarkInput("synthetic", "public", "contract", public_context={"nested": [{key: "HIDDEN_SENTINEL"}]})


def test_all_aggregator_requests_no_gold_and_no_hidden_values():
    for key, protocol in PROTOCOLS.items():
        hidden = {field: f"HIDDEN_SENTINEL_{i}" for i, field in enumerate(protocol.hidden_evaluator_fields)}
        a = {"hotpotqa": HotpotQAAnswerAdapter, "hover": HoVerBenchmarkAdapter,
             "ifbench": IFBenchBenchmarkAdapter, "math": MATHBenchmarkAdapter, "pupa": PUPABenchmarkAdapter}[key]()
        q = item(a, **hidden)
        assert all(value not in q.problem for value in hidden.values()) and not q.public_context
    assert all(benchmark_preflight(key)["provider_attempts"] == 0 for key in PROTOCOLS)


def test_no_dataset_download_no_provider_call_or_upstream_import(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("download/materialization/provider attempted")
    import multi_dataset_diverse_rl.benchmarks.data_freeze as freeze
    # The public protocol layer must not construct or call the data plane.
    from multi_dataset_diverse_rl.benchmarks.data_sources import PublicDataDownloader
    monkeypatch.setattr(PublicDataDownloader, "fetch", forbidden)
    monkeypatch.setattr(freeze, "freeze_local", forbidden)
    monkeypatch.setattr(freeze, "build_manifest", forbidden)
    for key in PROTOCOLS:
        assert benchmark_preflight(key)["gate"] == "HOLD_PRE_PROVIDER"
    root = Path("multi_dataset_diverse_rl/benchmarks/_vendor/ifbench")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "nltk.download(" not in text and "from spacy.cli import download" not in text
    import sys
    assert not any(name.startswith("gepa_artifact.benchmarks") for name in sys.modules)
