"""Offline benchmark contracts and explicit scientific holds."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json

import pytest

from multi_dataset_diverse_rl.benchmarks import (
    BENCHMARKS, HotpotQAAnswerAdapter, HoVerBenchmarkAdapter,
    IFBenchBenchmarkAdapter, PUPABenchmarkAdapter, MATHBenchmarkAdapter,
    benchmark_preflight,
)
from multi_dataset_diverse_rl.benchmarks.splits import SplitManifest, load_jsonl
from multi_dataset_diverse_rl.search.aggregation import (
    AggregationRequest, AggregationResponse, LLMAggregation, PluralityAggregation,
    RuntimeModelIdentity,
)
from multi_dataset_diverse_rl.search.benchmark import BenchmarkInput
from multi_dataset_diverse_rl.search.evaluation import TeamEvaluator
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.orchestrator import UnifiedSearchOrchestrator
from multi_dataset_diverse_rl.search.schemas import SearchMethodConfig
from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import (
    validate_mutable_decision_procedure,
)
from scripts.run_experiment import preflight


def _item(benchmark_id="hotpotqa"):
    return BenchmarkInput("same-id", "Who wrote it?", HotpotQAAnswerAdapter.output_contract,
                          benchmark_id=benchmark_id, benchmark_version="synthetic-v1",
                          parser_contract="hotpotqa-answer-em-v1")


def test_registry_and_preprovider_holds() -> None:
    assert set(BENCHMARKS) == {"hotpotqa", "hover", "ifbench", "pupa", "math"}
    for key, spec in BENCHMARKS.items():
        assert spec.benchmark_id == key and spec.upstream
        assert spec.provenance_frozen == (key == "ifbench")
        assert spec.split_frozen == (key == "ifbench")
        assert not spec.unified_search_ready
        if key != "ifbench":
            assert "BENCHMARK_PROVENANCE_NOT_FROZEN" in spec.blockers()
            assert "BENCHMARK_SPLIT_NOT_FROZEN" in spec.blockers()
        assert benchmark_preflight(key)["gate"] == "HOLD_PRE_PROVIDER"
    with pytest.raises(SearchContractError, match="BENCHMARK_UNKNOWN"):
        benchmark_preflight("not-a-benchmark")
    runtime = {"seed": 81, "provider_profile": "fake", "solver_model": "solver",
               "optimizer_model": "optimizer", "evaluator_model": "optimizer",
               "run_identity_sha256": "fake", "authorization_identity": "none",
               "cache_identity": "fake", "ledger_identity": "fake"}
    for key in BENCHMARKS:
        result = preflight({"scientific": {"method": "unified_team_prompt_search_v1",
                                             "benchmark_id": key}, "runtime": runtime})
        assert result["gate"] == "HOLD" and result["provider_attempts"] == 0
        assert "RESPONSIBILITY_POLICY_NOT_FROZEN" in result["blockers"]


def test_unselected_adapters_fail_closed_before_parsing_or_scoring() -> None:
    for kind in (HoVerBenchmarkAdapter, IFBenchBenchmarkAdapter,
                 PUPABenchmarkAdapter, MATHBenchmarkAdapter):
        adapter = kind()
        assert not adapter.capabilities.supports_current_responsibility
        with pytest.raises(SearchContractError, match="BENCHMARK_TASK_EVALUATOR_NOT_FROZEN"):
            adapter.parse_member_output("some response", _item())
        with pytest.raises(SearchContractError, match="BENCHMARK_TASK_EVALUATOR_NOT_FROZEN"):
            adapter.format_input(_item())
        with pytest.raises(SearchContractError, match="BENCHMARK_TASK_EVALUATOR_NOT_FROZEN"):
            adapter.score_member_output(None, None)


@pytest.mark.parametrize("kind", [HotpotQAAnswerAdapter, HoVerBenchmarkAdapter,
                                    IFBenchBenchmarkAdapter, PUPABenchmarkAdapter,
                                    MATHBenchmarkAdapter])
def test_each_unfrozen_benchmark_refuses_search_before_any_component(kind) -> None:
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError(f"component accessed before capability gate: {name}")

    forbidden = Forbidden()
    search = UnifiedSearchOrchestrator(
        method=SearchMethodConfig(), benchmark=kind(), aggregation=forbidden,
        state=forbidden, analyzer=forbidden, opportunities=forbidden,
        engine=forbidden, evaluation=forbidden, transition=forbidden,
        gate=forbidden, committer=forbidden,
    )
    with pytest.raises(SearchContractError, match="HOLD_PRE_PROVIDER"):
        asyncio.run(search.run(max_opportunities=1))


def test_output_contract_remains_outside_mutable_prompt() -> None:
    validate_mutable_decision_procedure("Reason carefully about the question.")
    with pytest.raises(ValueError, match="output_contract_contamination"):
        validate_mutable_decision_procedure(HotpotQAAnswerAdapter.output_contract)


def test_hotpota_answer_metric_and_plurality_fixture() -> None:
    adapter = HotpotQAAnswerAdapter()
    item = _item()
    assert adapter.parse_member_output("FINAL_ANSWER: The, Beatles!", item).answer == "beatles"
    assert adapter.score_member_output(
        adapter.parse_member_output("FINAL_ANSWER: BEATLES", item), "The Beatles!",
    ) == 1
    assert adapter.answer_f1(adapter.parse_member_output("FINAL_ANSWER: John Paul", item),
                             "John George") == .5
    assert adapter.answer_f1(adapter.parse_member_output("FINAL_ANSWER: yes maybe", item),
                             "yes") == 0
    assert adapter.score_member_output(adapter.parse_member_output("FINAL_ANSWER: Queen", item),
                                       "The Beatles") == 0
    for bad in ("no final line", "FINAL_ANSWER: ", "FINAL_ANSWER: yes\ntrailing",
                "FINAL_ANSWER: yes\nFINAL_ANSWER: no"):
        assert not adapter.parse_member_output(bad, item).valid
    row = asyncio.run(TeamEvaluator(adapter, PluralityAggregation()).evaluate_row(
        item=item, member_outputs=("FINAL_ANSWER: The Beatles!", "FINAL_ANSWER: beatles",
                                   "FINAL_ANSWER: Queen", "FINAL_ANSWER: Queen",
                                   "FINAL_ANSWER: Beatles"), gold="The Beatles",
    ))
    assert row.aggregate_score == 1 and row.aggregation_calls == 0
    assert row.member_scores == (1, 1, 0, 0, 1)


def test_materialization_hash_order_and_explicit_split_disjointness(tmp_path) -> None:
    data = b'{"id":"b","question":"second"}\n{"id":"a","question":"first"}\n'
    path = tmp_path / "fixture.jsonl"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    first = load_jsonl(path, benchmark_id="hotpotqa", upstream_identity="fixture-v1",
                       upstream_split="synthetic", expected_sha256=digest, id_field="id")
    second = load_jsonl(path, benchmark_id="hotpotqa", upstream_identity="fixture-v1",
                        upstream_split="synthetic", expected_sha256=digest, id_field="id")
    assert first == second and first.ordered_ids == ("b", "a")
    manifest = SplitManifest("hotpotqa", "fixture-v1", "synthetic", digest,
                             "explicit fixture assignment", None, first.ordered_ids,
                             ("b",), (), ("a",), ())
    assert manifest.identity() == replace(manifest).identity()
    assert len(manifest.group_hashes()) == 4
    with pytest.raises(SearchContractError, match="BENCHMARK_SPLIT_NOT_FROZEN"):
        manifest.require_frozen()
    with pytest.raises(SearchContractError, match="BENCHMARK_SPLIT_OVERLAP"):
        replace(manifest, validation_ids=("b",))
    with pytest.raises(SearchContractError, match="BENCHMARK_SOURCE_HASH_MISMATCH"):
        load_jsonl(path, benchmark_id="hotpotqa", upstream_identity="fixture-v1",
                   upstream_split="synthetic", expected_sha256="0" * 64, id_field="id")


def test_llm_aggregation_request_is_public_and_benchmark_separated() -> None:
    seen = []
    async def fake(request):
        seen.append(request)
        return AggregationResponse("FINAL_ANSWER: beatles", 3, 1)

    aggregator = LLMAggregation(runtime=RuntimeModelIdentity("optimizer", 81, "greedy"),
                                provider=fake)
    row = asyncio.run(TeamEvaluator(HotpotQAAnswerAdapter(), aggregator).evaluate_row(
        item=_item(), member_outputs=("FINAL_ANSWER: beatles",) * 5,
        gold="private reference value",
    ))
    assert row.aggregation_calls == 1 and row.aggregation_tokens == 4
    assert row.aggregation_logical_evaluations == 1 and row.aggregation_cache_hits == 0
    request = seen[0]
    assert request.model == "optimizer" and request.role == "aggregator"
    assert "private reference value" not in request.prompt
    assert "gold" not in request.prompt.casefold()
    assert request.benchmark_id == "hotpotqa"
    assert replace(request, benchmark_id="math").cache_identity() != request.cache_identity()
    assert replace(request, benchmark_version="v2").cache_identity() != request.cache_identity()
    assert replace(request, role="solver").cache_identity() != request.cache_identity()
    assert replace(request, seed=82).cache_identity() != request.cache_identity()
    assert replace(request, parser_contract="different").cache_identity() != request.cache_identity()
    assert replace(request, output_contract_sha256="different").cache_identity() != request.cache_identity()
    base = _item().request_identity(role="solver", model_identity="solver-model",
                                    prompt_identity="prompt-sha", seed=81,
                                    decoding_identity="greedy")
    assert replace(_item(), benchmark_id="math").request_identity(
        role="solver", model_identity="solver-model", prompt_identity="prompt-sha",
        seed=81, decoding_identity="greedy") != base
    assert _item().request_identity(role="solver", model_identity="solver-model",
                                    prompt_identity="prompt-sha", seed=82,
                                    decoding_identity="greedy") != base
    legacy = AggregationRequest("model", "aggregator", "prompt", "q1", 81, "greedy")
    old_payload = (legacy.model, legacy.role, legacy.prompt, legacy.input_id,
                   legacy.seed, legacy.decoding_identity)
    expected = hashlib.sha256(json.dumps(old_payload, ensure_ascii=False,
                                         separators=(",", ":")).encode()).hexdigest()
    assert legacy.cache_identity() == expected

    async def cached(request):
        return AggregationResponse("FINAL_ANSWER: beatles", provider_called=False)
    cache_row = asyncio.run(TeamEvaluator(
        HotpotQAAnswerAdapter(), LLMAggregation(
            runtime=RuntimeModelIdentity("optimizer", 81, "greedy"), provider=cached,
        )).evaluate_row(item=_item(), member_outputs=("FINAL_ANSWER: beatles",) * 5,
                        gold="beatles"))
    assert cache_row.aggregation_logical_evaluations == 1
    assert cache_row.aggregation_calls == 0 and cache_row.aggregation_cache_hits == 1
