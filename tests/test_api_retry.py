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
        self.request_kwargs = []

    async def create(self, **kwargs):
        self.request_kwargs.append(kwargs)
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
    assert completions.calls == 2
    assert [row["success"] for row in system.llm.calls] == [False, True]
    assert system.llm.calls[0]["status_code"] == 429
    assert system.llm.calls[1]["total_tokens"] == 5
    assert system.llm.calls[1]["finish_reason"] == "stop"
    assert completions.request_kwargs[1]["max_tokens"] == 10
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


def test_many_calls_do_not_stop_and_cost_keeps_accumulating(tmp_path):
    cfg = Config.from_flat(out_dir=str(tmp_path))
    system = PromptEnsembleOptimizationSystem(cfg)
    client, completions = fake_client([f"result-{index}" for index in range(25)])
    system.llm._client_or_raise = lambda _role: client
    for index in range(25):
        assert asyncio.run(
            system._chat("model", "system", "user", 0.0, None, "optimizer", "teacher")
        ).text == f"result-{index}"
    assert completions.calls == 25
    assert all("max_tokens" not in kwargs for kwargs in completions.request_kwargs)
    assert system.cost_summary()["total_tokens"] == 125


def test_tcs_omits_completion_limit_while_solver_keeps_1800(tmp_path):
    system = PromptEnsembleOptimizationSystem(Config.from_flat(out_dir=str(tmp_path)))
    client, completions = fake_client(["teacher", "critic", "student", "solver"])
    system.llm._client_or_raise = lambda _role: client

    asyncio.run(system._chat("model", "teacher", "user", 0.0, None, "optimizer"))
    asyncio.run(system._chat("model", "critic", "user", 0.0, None, "evaluator"))
    asyncio.run(system._chat(
        "model", "Return strict JSON only.", "user", 0.0, None, "optimizer"
    ))
    asyncio.run(system.llm.chat_result(
        "model", "solver", "user", 0.0, system.cfg.models.solver_max_tokens,
        "solver",
    ))

    assert all(
        "max_tokens" not in kwargs for kwargs in completions.request_kwargs[:3]
    )
    assert completions.request_kwargs[3]["max_tokens"] == 1800
    assert [row["role"] for row in system.llm.calls] == [
        "teacher", "critic", "student", "solver"
    ]
