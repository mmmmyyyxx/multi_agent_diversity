import asyncio
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


class StatusError(Exception):
    def __init__(self, status_code, retry_after=None):
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        self.response = SimpleNamespace(
            status_code=status_code,
            headers={} if retry_after is None else {"retry-after": str(retry_after)},
        )


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def create(self, **_kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=outcome),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )


def fake_client(outcomes):
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_retryable_429_retries_and_logs_each_attempt(tmp_path, monkeypatch):
    cfg = Config.from_flat(
        out_dir=str(tmp_path), max_retries=1, max_transient_retries=2,
        retry_sleep=0.0, max_retry_backoff=0.0,
    )
    system = PromptEnsembleOptimizationSystem(cfg)
    client, completions = fake_client([StatusError(429, retry_after=2), "ok"])
    system.llm._client_or_raise = lambda _role: client
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("multi_dataset_diverse_rl.llm_client.asyncio.sleep", fake_sleep)
    result = asyncio.run(system._chat("model", "system", "user", 0.0, 10, "optimizer"))
    assert result.text == "ok"
    assert result.finish_reason == "stop"
    assert result.completion_token_limit == 10
    assert completions.calls == 2
    assert [row["success"] for row in system.llm.calls] == [False, True]
    assert system.llm.calls[0]["status_code"] == 429
    assert system.llm.calls[1]["total_tokens"] == 5
    assert system.llm.calls[1]["finish_reason"] == "stop"
    assert system.llm.calls[1]["completion_token_limit"] == 10
    assert system.llm.calls[1]["hit_completion_limit"] is False
    assert sleeps[0] >= 2.0


@pytest.mark.parametrize("failure", [
    StatusError(408), StatusError(409), StatusError(500), StatusError(502),
    StatusError(503), StatusError(504), TimeoutError("timeout"), ConnectionError("connection"),
])
def test_all_declared_transient_failures_retry(tmp_path, monkeypatch, failure):
    cfg = Config.from_flat(
        out_dir=str(tmp_path), max_retries=1, max_transient_retries=1,
        retry_sleep=0.0, max_retry_backoff=0.0,
    )
    system = PromptEnsembleOptimizationSystem(cfg)
    client, completions = fake_client([failure, "ok"])
    system.llm._client_or_raise = lambda _role: client

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("multi_dataset_diverse_rl.llm_client.asyncio.sleep", no_sleep)
    assert asyncio.run(
        system._chat("model", "system", "user", 0.0, 10, "optimizer")
    ).text == "ok"
    assert completions.calls == 2
    assert [row["success"] for row in system.llm.calls] == [False, True]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_retryable_client_status_fails_immediately(tmp_path, status):
    cfg = Config.from_flat(out_dir=str(tmp_path), max_retries=3, max_transient_retries=3)
    system = PromptEnsembleOptimizationSystem(cfg)
    client, completions = fake_client([StatusError(status)])
    system.llm._client_or_raise = lambda _role: client
    with pytest.raises(StatusError):
        asyncio.run(system._chat("model", "system", "user", 0.0, 10, "optimizer"))
    assert completions.calls == 1
    assert system.llm.calls[0]["status_code"] == status


def test_llm_call_budget_stops_before_extra_call(tmp_path):
    cfg = Config.from_flat(out_dir=str(tmp_path), max_total_llm_calls=1)
    system = PromptEnsembleOptimizationSystem(cfg)
    client, completions = fake_client(["first", "second"])
    system.llm._client_or_raise = lambda _role: client
    assert asyncio.run(
        system._chat("model", "system", "user", 0.0, 10, "optimizer")
    ).text == "first"
    with pytest.raises(RuntimeError, match="max_total_llm_calls"):
        asyncio.run(system._chat("model", "system", "user", 0.0, 10, "optimizer"))
    assert completions.calls == 1


def test_llm_token_budget_stops_after_over_budget_response(tmp_path):
    cfg = Config.from_flat(out_dir=str(tmp_path), max_total_tokens=4)
    system = PromptEnsembleOptimizationSystem(cfg)
    client, completions = fake_client(["five tokens", "unused"])
    system.llm._client_or_raise = lambda _role: client
    with pytest.raises(RuntimeError, match="max_total_tokens"):
        asyncio.run(system._chat("model", "system", "user", 0.0, 10, "optimizer"))
    assert completions.calls == 1
    assert system.llm.calls[0]["total_tokens"] == 5
