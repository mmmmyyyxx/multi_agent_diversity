from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Mapping, Protocol


@dataclass(frozen=True)
class PromptAnswer:
    answer: str
    trace: str
    valid: bool
    validity_status: str = ""
    raw_final_answer_payload: str = ""
    final_answer_line_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    response_hash: str = ""
    request_identity: str = ""
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.validity_status:
            object.__setattr__(self, "validity_status", "valid" if self.valid else "invalid_unspecified")
        if not self.response_hash:
            object.__setattr__(
                self,
                "response_hash",
                hashlib.sha256(str(self.trace).encode("utf-8")).hexdigest(),
            )
        if not self.created_at:
            object.__setattr__(self, "created_at", time.time())


class SharedSolverCache(Protocol):
    hits: int
    misses: int
    waits: int

    async def resolve(
        self,
        *,
        cache_key: str,
        metadata: Mapping[str, Any],
        producer: Callable[[], Awaitable[PromptAnswer]],
    ) -> PromptAnswer: ...


class PromptQuestionEvaluator:
    """Run-scoped prompt-question cache with sampling semantics independent of agent identity."""

    def __init__(
        self,
        *,
        model_request_identity: str,
        parser_version: str,
        temperature: float,
        decoding_seed: int,
        cache_metadata: Mapping[str, Any] | None = None,
        shared_cache: SharedSolverCache | None = None,
        observation_callback: Callable[[str, str, PromptAnswer], None] | None = None,
        version: str = "prompt_question_v1",
    ):
        self.version = str(version)
        self.model_request_identity = str(model_request_identity)
        self.parser_version = str(parser_version)
        self.temperature = float(temperature)
        self.decoding_seed = int(decoding_seed)
        self.cache_metadata = dict(cache_metadata or {})
        self.shared_cache = shared_cache
        self.observation_callback = observation_callback
        self.cache: dict[str, PromptAnswer] = {}
        self.inflight: dict[str, asyncio.Future[PromptAnswer]] = {}
        self.lock = asyncio.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def key(self, prompt_hash: str, question_hash: str) -> str:
        payload = {
            "version": self.version,
            "model_request_identity": self.model_request_identity,
            "prompt_hash": str(prompt_hash),
            "question_hash": str(question_hash),
            "parser_version": self.parser_version,
            "temperature": self.temperature,
            "evaluation_replica_seed": self.decoding_seed,
            **self.cache_metadata,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    async def evaluate(
        self,
        *,
        question: str,
        question_hash: str,
        prompt: str,
        prompt_hash: str,
        agent_id: int,
        solve: Callable[[str, int, str], Awaitable[PromptAnswer]],
    ) -> PromptAnswer:
        key = self.key(prompt_hash, question_hash)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        owner = False
        async with self.lock:
            future = self.inflight.get(key)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                self.inflight[key] = future
                owner = True
        if not owner:
            self.cache_hits += 1
            return await future
        try:
            self.cache_misses += 1
            async def produce() -> PromptAnswer:
                return await solve(question, agent_id, prompt)

            if self.shared_cache is None:
                answer = await produce()
            else:
                answer = await self.shared_cache.resolve(
                    cache_key=key,
                    metadata={
                        **self.cache_metadata,
                        "model_request_identity": self.model_request_identity,
                        "parser_version": self.parser_version,
                        "temperature": self.temperature,
                        "evaluation_replica_seed": self.decoding_seed,
                        "prompt_hash": str(prompt_hash),
                        "question_hash": str(question_hash),
                    },
                    producer=produce,
                )
            self.cache[key] = answer
            if self.observation_callback is not None:
                self.observation_callback(str(prompt_hash), str(question_hash), answer)
            future.set_result(answer)
            return answer
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            async with self.lock:
                self.inflight.pop(key, None)

    def identity(self) -> tuple[str, str, str, float, int]:
        return (
            self.version,
            self.model_request_identity,
            self.parser_version,
            self.temperature,
            self.decoding_seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_request_identity": self.model_request_identity,
            "parser_version": self.parser_version,
            "temperature": self.temperature,
            "decoding_seed": self.decoding_seed,
            "cache": {key: asdict(value) for key, value in self.cache.items()},
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        actual = (
            str(payload["version"]),
            str(payload["model_request_identity"]),
            str(payload["parser_version"]),
            float(payload["temperature"]),
            int(payload["decoding_seed"]),
        )
        if actual != self.identity():
            raise ValueError(
                "Prompt-question evaluator identity mismatch: "
                + json.dumps({"expected": self.identity(), "actual": actual})
            )
        raw_cache = payload["cache"]
        if not isinstance(raw_cache, Mapping):
            raise ValueError("prompt-question cache must be an object")
        self.cache = {str(key): PromptAnswer(**value) for key, value in raw_cache.items()}
        self.cache_hits = int(payload["cache_hits"])
        self.cache_misses = int(payload["cache_misses"])
