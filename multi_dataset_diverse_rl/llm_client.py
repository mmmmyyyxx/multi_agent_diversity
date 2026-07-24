from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from .config import Config


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}


@dataclass(frozen=True)
class LLMCallResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    finish_reason: str


class RoleAwareLLMClient:
    def __init__(
        self,
        cfg: Config,
        override: Callable[[str, str, float, int | None], Awaitable[str]] | None = None,
    ):
        self.cfg = cfg
        self.override = override
        self.clients: dict[str, AsyncOpenAI] = {}
        self.calls: list[dict[str, Any]] = []

    def _role_credentials(self, role: str) -> tuple[str, str]:
        if role == "solver":
            return self.cfg.models.solver_api_key_env, self.cfg.models.solver_base_url_env
        if role == "optimizer":
            return self.cfg.models.optimizer_api_key_env, self.cfg.models.optimizer_base_url_env
        if role == "evaluator":
            return self.cfg.models.evaluator_api_key_env, self.cfg.models.evaluator_base_url_env
        raise ValueError(f"Unknown client role: {role}")

    def _client_or_raise(self, role: str) -> AsyncOpenAI:
        if role not in self.clients:
            key_env, base_env = self._role_credentials(role)
            key = os.getenv(key_env) if key_env else os.getenv("OPENAI_API_KEY")
            base = os.getenv(base_env) if base_env else (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
            if not key:
                raise ValueError(f"API key is not configured for role={role}")
            self.clients[role] = AsyncOpenAI(api_key=key, base_url=base)
        return self.clients[role]

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        status = getattr(exc, "status_code", None)
        if status is None and getattr(exc, "response", None) is not None:
            status = getattr(exc.response, "status_code", None)
        return int(status) if status is not None else None

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) if response is not None else {}
        value = headers.get("retry-after") if hasattr(headers, "get") else None
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                return max(0.0, parsedate_to_datetime(str(value)).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return None

    def _retryable(self, exc: Exception) -> bool:
        status = self._status_code(exc)
        if status in NON_RETRYABLE_STATUS_CODES:
            return False
        if status is not None:
            return status in RETRYABLE_STATUS_CODES
        name = type(exc).__name__.lower()
        return isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)) or any(
            marker in name for marker in ("timeout", "connection")
        )

    def record_override_solver(self, *, started: float) -> None:
        self.calls.append({
            "role": "solver",
            "client_role": "solver",
            "model": self.cfg.models.agent_model,
            "attempt": 1,
            "success": True,
            "status_code": 200,
            "error_type": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": time.time() - started,
            "finish_reason": "stop",
            "configured_solver_max_tokens": self.cfg.models.solver_max_tokens,
        })

    async def chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
        role: str,
        logical_role: str | None = None,
    ) -> str:
        return (
            await self.chat_result(
                model, system_prompt, user_prompt, temperature, max_tokens, role,
                logical_role,
            )
        ).text

    async def chat_result(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
        role: str,
        logical_role: str | None = None,
    ) -> LLMCallResult:
        max_attempts = max(1, self.cfg.persistence.max_retries + self.cfg.persistence.max_transient_retries)
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            started = time.time()
            try:
                if self.override is not None and role in {"optimizer", "evaluator"}:
                    text = await self.override(system_prompt, user_prompt, temperature, max_tokens)
                    prompt_tokens = completion_tokens = 0
                    finish_reason = "stop"
                else:
                    request_kwargs = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "timeout": self.cfg.persistence.llm_call_timeout,
                    }
                    if max_tokens is not None:
                        request_kwargs["max_tokens"] = max_tokens
                    response = await self._client_or_raise(role).chat.completions.create(
                        **request_kwargs,
                    )
                    text = response.choices[0].message.content or ""
                    usage = response.usage
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    finish_reason = str(response.choices[0].finish_reason or "")
                latency = time.time() - started
                call_record = {
                    "role": logical_role or role,
                    "client_role": role,
                    "model": model,
                    "attempt": attempt,
                    "success": True,
                    "status_code": 200,
                    "error_type": "",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "latency_seconds": latency,
                    "finish_reason": finish_reason,
                }
                if role == "solver" and max_tokens is not None:
                    call_record["configured_solver_max_tokens"] = max_tokens
                self.calls.append(call_record)
                return LLMCallResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_seconds=latency,
                    finish_reason=finish_reason,
                )
            except Exception as exc:
                last_error = exc
                status = self._status_code(exc)
                self.calls.append({
                    "role": logical_role or role,
                    "client_role": role,
                    "model": model,
                    "attempt": attempt,
                    "success": False,
                    "status_code": status,
                    "error_type": type(exc).__name__,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_seconds": time.time() - started,
                    "finish_reason": "",
                })
                if not self._retryable(exc) or attempt >= max_attempts:
                    raise
                exponential = self.cfg.persistence.retry_sleep * (2 ** min(attempt - 1, 8))
                retry_after = self._retry_after_seconds(exc)
                jitter = random.Random(
                    f"{self.cfg.training.seed}:{role}:{attempt}:{len(self.calls)}"
                ).uniform(0.0, 0.25)
                base_delay = retry_after if retry_after is not None else min(
                    self.cfg.persistence.max_retry_backoff, exponential,
                )
                await asyncio.sleep(base_delay + jitter)
        raise RuntimeError(f"LLM call failed: {last_error}")

    def cost_summary(self) -> dict[str, Any]:
        successful = [row for row in self.calls if row["success"]]
        tokens_by_role = {
            logical_role: sum(
                int(row["total_tokens"])
                for row in self.calls
                if row["role"] == logical_role
            )
            for logical_role in ("solver", "teacher", "critic", "student")
        }
        return {
            "solver_calls": sum(
                row.get("client_role", row["role"]) == "solver" for row in successful
            ),
            "optimizer_calls": sum(
                row.get("client_role", row["role"]) == "optimizer" for row in successful
            ),
            "evaluator_calls": sum(
                row.get("client_role", row["role"]) == "evaluator" for row in successful
            ),
            "total_llm_calls": len(self.calls),
            "successful_llm_calls": len(successful),
            "failed_llm_attempts": len(self.calls) - len(successful),
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in self.calls),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in self.calls),
            "total_tokens": sum(int(row["total_tokens"]) for row in self.calls),
            "tokens_by_role": tokens_by_role,
            "latency_seconds": sum(float(row["latency_seconds"]) for row in self.calls),
        }
