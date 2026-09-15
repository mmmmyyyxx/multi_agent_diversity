from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from infrastructure.common_solver_contract_v1.contract import (
    COMMON_SOLVER_CONTRACT_ID,
    CONTRACT_SPEC,
    canonical_json_bytes,
    contract_identity,
    parse_solver_output,
    serialize_solver_request,
)
from infrastructure.common_solver_contract_v1.entrypoints import (
    from_diversity,
    from_gepa,
    from_mars,
)
from infrastructure.common_solver_contract_v1.evaluator import CommonSolverEvaluator


ROOT = Path(__file__).resolve().parents[1]
QUESTION_LF = "Who left?\nOptions:\n(A) Alex\n(B) Blair\n(C) Ambiguous"
PROMPT = "Resolve the reference from grammar and context, then choose one listed option."


def test_three_boundary_adapters_have_identical_request_bytes() -> None:
    diversity = from_diversity(
        prompt=PROMPT,
        raw_question=QUESTION_LF.replace("\n", "\r\n"),
    )
    mars = from_mars(prompt=PROMPT, rendered_question=QUESTION_LF)
    gepa = from_gepa(
        prompt=PROMPT,
        example={
            "question": "Who left?",
            "option_labels": ["A", "B", "C"],
            "choices": ["Alex", "Blair", "Ambiguous"],
        },
    )
    assert canonical_json_bytes(diversity) == canonical_json_bytes(mars)
    assert canonical_json_bytes(mars) == canonical_json_bytes(gepa)
    assert list(diversity) == ["model", "messages", "temperature", "max_tokens", "extra_body"]
    assert diversity["model"] == "qwen3-8b"
    assert diversity["temperature"] == 0.0
    assert diversity["max_tokens"] == 1800
    assert diversity["extra_body"] == {"enable_thinking": False}
    assert not ({"seed", "top_p", "stop"} & set(diversity))


def test_same_decision_procedure_instantiates_same_complete_solver_request() -> None:
    first = serialize_solver_request(decision_procedure=PROMPT, question=QUESTION_LF)
    second = serialize_solver_request(decision_procedure=PROMPT, question=QUESTION_LF)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    system_text = first["messages"][0]["content"]
    assert system_text.count(PROMPT) == 1
    assert system_text.count("Mandatory output interface") == 1


def test_strict_parser_and_shared_exact_request_cache() -> None:
    calls: list[dict] = []

    async def transport(request: dict) -> str:
        calls.append(request)
        return "Reasoning.\nFINAL_ANSWER: B"

    async def scenario() -> None:
        evaluator = CommonSolverEvaluator(transport=transport)
        first = await evaluator.evaluate(decision_procedure=PROMPT, question=QUESTION_LF)
        second = await evaluator.evaluate(decision_procedure=PROMPT, question=QUESTION_LF)
        assert first.response.valid and first.response.answer == "B"
        assert not first.cache_hit and second.cache_hit
        assert evaluator.accounting() == {
            "logical_calls": 2,
            "provider_attempts": 1,
            "successful_provider_calls": 1,
            "failed_provider_attempts": 0,
            "cache_hits": 1,
            "cache_entries": 1,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    asyncio.run(scenario())
    assert len(calls) == 1
    assert not parse_solver_output(
        "FINAL_ANSWER: B because", question=QUESTION_LF
    ).valid
    assert not parse_solver_output(
        "FINAL_ANSWER: A\nFINAL_ANSWER: B", question=QUESTION_LF
    ).valid


def test_failed_attempt_observer_runs_before_retry_and_terminal_raise() -> None:
    events: list[dict[str, object]] = []
    calls = 0

    async def transport(_request: dict) -> str:
        nonlocal calls
        calls += 1
        raise ConnectionError("synthetic transport failure")

    async def scenario() -> None:
        evaluator = CommonSolverEvaluator(
            transport=transport,
            failed_attempt_observer=lambda event: events.append(dict(event)),
        )
        with pytest.raises(ConnectionError):
            await evaluator.evaluate(decision_procedure=PROMPT, question=QUESTION_LF)
        assert evaluator.accounting()["failed_provider_attempts"] == 4

    asyncio.run(scenario())
    assert calls == CONTRACT_SPEC.transport_attempt_cap == 4
    assert [event["attempt_index"] for event in events] == [1, 2, 3, 4]
    assert all(event["error_type"] == "ConnectionError" for event in events)
    assert all(event["request_identity"] for event in events)


def test_paired_evaluators_share_one_provider_realization() -> None:
    shared_cache: dict[str, str] = {}
    calls = {"a": 0, "b": 0}

    async def transport_a(_request: dict) -> str:
        calls["a"] += 1
        return "Reasoning A.\nFINAL_ANSWER: B"

    async def transport_b(_request: dict) -> str:
        calls["b"] += 1
        return "Reasoning B.\nFINAL_ANSWER: A"

    async def scenario() -> None:
        arm_a = CommonSolverEvaluator(transport=transport_a, cache=shared_cache)
        arm_b = CommonSolverEvaluator(transport=transport_b, cache=shared_cache)
        first = await arm_a.evaluate(decision_procedure=PROMPT, question=QUESTION_LF)
        second = await arm_b.evaluate(decision_procedure=PROMPT, question=QUESTION_LF)
        assert first.request_identity == second.request_identity
        assert first.response == second.response
        assert first.cache_hit is False
        assert second.cache_hit is True

    asyncio.run(scenario())
    assert calls == {"a": 1, "b": 0}


def test_frozen_report_and_replay_registry_are_execution_ready() -> None:
    report = ROOT / "reports/common_solver_contract_v1_prep_20260906"
    facts = json.loads((report / "fact_assertions.json").read_text(encoding="utf-8"))
    public = json.loads((report / "replay_registry_public.json").read_text(encoding="utf-8"))
    contract = json.loads((report / "contract_manifest.json").read_text(encoding="utf-8"))
    assert facts["gate"] == "PASS"
    assert facts["api_calls"] == 0
    assert facts["test50_accessed"] is False
    assert facts["serialization_exact_bytes_equal"] is True
    assert facts["serialization_smoke_case_count"] == 20
    assert public["status"] == "EXECUTION_READY_PENDING_EXPLICIT_API_AUTHORIZATION"
    assert public["case_count"] == 50
    assert [row["state_id"] for row in public["states"]] == [
        "P0_COMMON",
        "MARS_SEED76_FINAL",
        "GEPA_SEED76_FINAL",
        "DIVERSITY_SEED76_P1_FINAL",
    ]
    assert contract["spec"]["contract_id"] == COMMON_SOLVER_CONTRACT_ID
    assert contract["identity"] == contract_identity()
    assert contract["spec"]["transport_attempt_cap"] == CONTRACT_SPEC.transport_attempt_cap
