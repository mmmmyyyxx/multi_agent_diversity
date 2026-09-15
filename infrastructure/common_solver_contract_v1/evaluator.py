"""Shared evaluator engine with injectable transport and exact-request cache."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, MutableMapping

from .contract import (
    CONTRACT_SPEC,
    ParsedSolverOutput,
    canonical_json_bytes,
    parse_solver_output,
    request_identity,
    serialize_solver_request,
)


@dataclass(frozen=True)
class TransportResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""


Transport = Callable[[dict], Awaitable[str | TransportResponse]]
FailedAttemptObserver = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True)
class EvaluationResult:
    request_identity: str
    response: ParsedSolverOutput
    transport_attempts: int
    cache_hit: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""


class CommonSolverEvaluator:
    """One execution path for MARS, GEPA, and Diversity frozen-state replay."""

    def __init__(
        self,
        *,
        transport: Transport,
        cache: MutableMapping[str, str] | None = None,
        retryable: Callable[[Exception], bool] | None = None,
        failed_attempt_observer: FailedAttemptObserver | None = None,
    ) -> None:
        self.transport = transport
        self.cache = cache if cache is not None else {}
        self.retryable = retryable or (lambda exc: isinstance(exc, (TimeoutError, ConnectionError)))
        self.failed_attempt_observer = failed_attempt_observer
        self.logical_calls = 0
        self.provider_attempts = 0
        self.cache_hits = 0
        self.successful_provider_calls = 0
        self.failed_provider_attempts = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def evaluate(
        self, *, decision_procedure: str, question: str
    ) -> EvaluationResult:
        self.logical_calls += 1
        identity = request_identity(
            decision_procedure=decision_procedure, question=question
        )
        if identity in self.cache:
            self.cache_hits += 1
            raw = self.cache[identity]
            return EvaluationResult(
                identity,
                parse_solver_output(raw, question=question),
                0,
                True,
            )
        request = serialize_solver_request(
            decision_procedure=decision_procedure, question=question
        )
        last_error: Exception | None = None
        for attempt in range(1, CONTRACT_SPEC.transport_attempt_cap + 1):
            self.provider_attempts += 1
            try:
                response = await self.transport(request)
                if isinstance(response, TransportResponse):
                    raw = response.text
                    prompt_tokens = int(response.prompt_tokens)
                    completion_tokens = int(response.completion_tokens)
                    finish_reason = str(response.finish_reason)
                else:
                    raw = str(response)
                    prompt_tokens = completion_tokens = 0
                    finish_reason = ""
                self.successful_provider_calls += 1
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                self.cache[identity] = raw
                return EvaluationResult(
                    identity,
                    parse_solver_output(raw, question=question),
                    attempt,
                    False,
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                )
            except Exception as exc:
                last_error = exc
                self.failed_provider_attempts += 1
                if self.failed_attempt_observer is not None:
                    status_code = getattr(exc, "status_code", None)
                    if status_code is None and getattr(exc, "response", None) is not None:
                        status_code = getattr(exc.response, "status_code", None)
                    self.failed_attempt_observer({
                        "request_identity": identity,
                        "attempt_index": attempt,
                        "error_type": type(exc).__name__,
                        "status_code": status_code,
                    })
                if not self.retryable(exc) or attempt == CONTRACT_SPEC.transport_attempt_cap:
                    raise
                await asyncio.sleep(CONTRACT_SPEC.retry_backoff_seconds[attempt - 1])
        raise RuntimeError(f"unreachable transport failure: {last_error}")

    def accounting(self) -> dict[str, int]:
        return {
            "logical_calls": self.logical_calls,
            "provider_attempts": self.provider_attempts,
            "successful_provider_calls": self.successful_provider_calls,
            "failed_provider_attempts": self.failed_provider_attempts,
            "cache_hits": self.cache_hits,
            "cache_entries": len(self.cache),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


def canonical_request_bytes(*, decision_procedure: str, question: str) -> bytes:
    return canonical_json_bytes(
        serialize_solver_request(
            decision_procedure=decision_procedure, question=question
        )
    )
