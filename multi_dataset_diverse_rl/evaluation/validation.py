from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .fixed_probe import ProbeExample, fixed_probe_hash
from .prompt_question import PromptAnswer, PromptQuestionEvaluator


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
    vote_correct_count: int
    per_agent_correct_counts: tuple[int, ...]
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
    validity_status_counts: dict[str, int] = field(default_factory=dict)
    terminal_invalid_count: int = 0

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
        prompt_question_evaluator: PromptQuestionEvaluator | None = None,
        version: str = "validation_probe_v1",
    ):
        self.examples = tuple(examples)
        self.version = str(version)
        self.probe_hash = fixed_probe_hash(self.examples, self.version)
        self.model_identity = str(model_identity)
        self.parser_version = str(parser_version)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.prompt_question_evaluator = prompt_question_evaluator or PromptQuestionEvaluator(
            model_request_identity=self.model_identity,
            parser_version=self.parser_version,
            temperature=self.temperature,
            decoding_seed=self.seed,
        )

    @property
    def cache_hits(self) -> int:
        return self.prompt_question_evaluator.cache_hits

    @property
    def cache_misses(self) -> int:
        return self.prompt_question_evaluator.cache_misses

    async def evaluate_prompt(
        self,
        agent_id: int,
        prompt: str,
        prompt_hash: str,
        solve: Callable[[str, int, str], Awaitable[PromptAnswer]],
    ) -> tuple[PromptAnswer, ...]:
        async def one(example: ProbeExample) -> PromptAnswer:
            return await self.prompt_question_evaluator.evaluate(
                question=example.question,
                question_hash=example.question_hash,
                prompt=prompt,
                prompt_hash=prompt_hash,
                agent_id=agent_id,
                solve=solve,
            )

        return tuple(await asyncio.gather(*(one(example) for example in self.examples)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "probe_hash": self.probe_hash,
            "model_identity": self.model_identity,
            "parser_version": self.parser_version,
            "temperature": self.temperature,
            "seed": self.seed,
            "prompt_question_evaluator": self.prompt_question_evaluator.to_dict(),
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
        evaluator_payload = payload["prompt_question_evaluator"]
        if not isinstance(evaluator_payload, dict):
            raise ValueError("validation prompt-question evaluator must be an object")
        self.prompt_question_evaluator.restore(evaluator_payload)
