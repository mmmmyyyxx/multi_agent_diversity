from __future__ import annotations

import asyncio
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .fixed_probe import ProbeExample, PromptAnswer, fixed_probe_hash


@dataclass(frozen=True)
class DatasetEvaluationRow:
    question_hash: str
    vote_correct: bool
    top_tie: bool
    gold_vote_count: int
    largest_wrong_vote_count: int
    plurality_margin: int


@dataclass(frozen=True)
class DatasetMetrics:
    plurality_vote_acc: float
    vote_acc: float
    mean_individual_acc: float
    min_individual_acc: float
    per_agent_acc: tuple[float, ...]
    mean_soft_vote_utility: float
    c0_count: int
    mean_invalid_rate: float
    tie_count: int
    tie_rate: float
    rows: tuple[DatasetEvaluationRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ValidationProbeEvaluator:
    def __init__(
        self,
        examples: Sequence[ProbeExample],
        *,
        model_identity: str,
        parser_version: str,
        temperature: float,
        seed: int,
        version: str = "validation_probe_v1",
    ):
        self.examples = tuple(examples)
        self.version = str(version)
        self.probe_hash = fixed_probe_hash(self.examples, self.version)
        self.model_identity = str(model_identity)
        self.parser_version = str(parser_version)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.cache: dict[str, PromptAnswer] = {}
        self.inflight: dict[str, asyncio.Future[PromptAnswer]] = {}
        self.lock = asyncio.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def _key(self, prompt_hash: str, question_hash: str) -> str:
        raw = "|".join((
            self.probe_hash,
            self.model_identity,
            str(prompt_hash),
            str(question_hash),
            self.parser_version,
            repr(self.temperature),
            str(self.seed),
        ))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def evaluate_prompt(
        self,
        agent_id: int,
        prompt: str,
        prompt_hash: str,
        solve: Callable[[str, int, str], Awaitable[PromptAnswer]],
    ) -> tuple[PromptAnswer, ...]:
        async def one(example: ProbeExample) -> PromptAnswer:
            key = self._key(prompt_hash, example.question_hash)
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
                answer = await solve(example.question, agent_id, prompt)
                self.cache[key] = answer
                future.set_result(answer)
                return answer
            except Exception as exc:
                future.set_exception(exc)
                raise
            finally:
                async with self.lock:
                    self.inflight.pop(key, None)

        return tuple(await asyncio.gather(*(one(example) for example in self.examples)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "probe_hash": self.probe_hash,
            "model_identity": self.model_identity,
            "parser_version": self.parser_version,
            "temperature": self.temperature,
            "seed": self.seed,
            "cache": {key: asdict(value) for key, value in self.cache.items()},
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        identity = (
            str(payload["version"]),
            str(payload["probe_hash"]),
            str(payload["model_identity"]),
            str(payload["parser_version"]),
            float(payload["temperature"]),
            int(payload["seed"]),
        )
        expected = (
            self.version,
            self.probe_hash,
            self.model_identity,
            self.parser_version,
            self.temperature,
            self.seed,
        )
        if identity != expected:
            raise ValueError("Validation probe identity mismatch. Start a new run.")
        raw_cache = payload["cache"]
        if not isinstance(raw_cache, Mapping):
            raise ValueError("validation cache must be an object")
        self.cache = {str(key): PromptAnswer(**value) for key, value in raw_cache.items()}
        self.cache_hits = int(payload["cache_hits"])
        self.cache_misses = int(payload["cache_misses"])
