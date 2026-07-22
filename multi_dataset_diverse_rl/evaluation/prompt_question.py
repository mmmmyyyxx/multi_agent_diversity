from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Mapping


@dataclass(frozen=True)
class PromptAnswer:
    answer: str
    trace: str
    valid: bool
    validity_status: str = ""

    def __post_init__(self) -> None:
        if not self.validity_status:
            object.__setattr__(self, "validity_status", "valid" if self.valid else "invalid_unspecified")


class PromptQuestionEvaluator:
    """Run-scoped prompt-question cache with sampling semantics independent of agent identity."""

    def __init__(
        self,
        *,
        model_request_identity: str,
        parser_version: str,
        temperature: float,
        decoding_seed: int,
        version: str = "prompt_question_v1",
    ):
        self.version = str(version)
        self.model_request_identity = str(model_request_identity)
        self.parser_version = str(parser_version)
        self.temperature = float(temperature)
        self.decoding_seed = int(decoding_seed)
        self.cache: dict[str, PromptAnswer] = {}
        self.inflight: dict[str, asyncio.Future[PromptAnswer]] = {}
        self.lock = asyncio.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def key(self, prompt_hash: str, question_hash: str) -> str:
        payload = (
            self.version,
            self.model_request_identity,
            str(prompt_hash),
            str(question_hash),
            self.parser_version,
            repr(self.temperature),
            str(self.decoding_seed),
        )
        return hashlib.sha256("|".join(payload).encode("utf-8")).hexdigest()

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
            answer = await solve(question, agent_id, prompt)
            self.cache[key] = answer
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
