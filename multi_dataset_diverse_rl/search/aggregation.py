"""Equal-status aggregation, isolated from scoring and gold answers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Awaitable, Callable, Mapping, Protocol

from .benchmark import BenchmarkAdapter, BenchmarkInput
from .schemas import ParsedOutput, SearchContractError
from .. import versions


@dataclass(frozen=True)
class RuntimeModelIdentity:
    optimizer_model: str
    seed: int
    decoding_identity: str

    def __post_init__(self) -> None:
        if not self.optimizer_model or not self.decoding_identity:
            raise SearchContractError("optimizer model and decoding identity are required")

    @classmethod
    def from_runtime_context(cls, runtime: object, *, decoding_identity: str) -> "RuntimeModelIdentity":
        return cls(str(getattr(runtime, "optimizer_model")),
                   int(getattr(runtime, "seed")), decoding_identity)


@dataclass(frozen=True)
class AggregationRequest:
    model: str
    role: str
    prompt: str
    input_id: str
    seed: int
    decoding_identity: str

    def cache_identity(self) -> str:
        payload = (self.model, self.role, self.prompt, self.input_id,
                   self.seed, self.decoding_identity)
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AggregationResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    provider_called: bool = True


@dataclass(frozen=True)
class AggregationResult:
    raw_output: str
    parsed_output: ParsedOutput
    role: str
    model: str | None
    calls: int
    tokens: int
    cache_identity: str | None
    diagnostics: Mapping[str, object] = field(default_factory=dict)


class AggregationPolicy(Protocol):
    async def aggregate(
        self, *, item: BenchmarkInput, member_outputs: tuple[str, ...],
        benchmark: BenchmarkAdapter,
    ) -> AggregationResult: ...


class PluralityAggregation:
    """One equal vote per valid parsed answer; a top-count tie abstains."""

    identity = versions.UNIFIED_PLURALITY_AGGREGATION_VERSION

    async def aggregate(
        self, *, item: BenchmarkInput, member_outputs: tuple[str, ...],
        benchmark: BenchmarkAdapter,
    ) -> AggregationResult:
        if not benchmark.capabilities.supports_plurality:
            raise SearchContractError("benchmark does not support plurality")
        counts: dict[str, int] = {}
        for raw in member_outputs:
            parsed = benchmark.parse_member_output(raw, item)
            if parsed.valid:
                counts[parsed.answer] = counts.get(parsed.answer, 0) + 1
        ordered = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
        tie = len(ordered) > 1 and ordered[0][1] == ordered[1][1]
        answer = ordered[0][0] if ordered and not tie else ""
        # Parsing remains the benchmark's authority, including the abstention.
        raw = f"FINAL_ANSWER: {answer}" if answer else ""
        parsed = benchmark.parse_output(raw, item) if answer else ParsedOutput("", False)
        margin = (ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else (ordered[0][1] if ordered else 0)
        return AggregationResult(
            raw, parsed, "team_aggregation", None, 0, 0, None,
            {"vote_counts": dict(ordered), "top_margin": margin,
             "tie_abstained": tie},
        )


AGGREGATION_INSTRUCTION_V1 = (
    "Given the problem and candidate responses from equal-status members, "
    "produce one final team response. Resolve disagreements using the reasoning "
    "in the responses. No member has special authority. Do not mention the "
    "aggregation process. Respect the stated output contract."
)
AGGREGATION_INSTRUCTION_SHA256 = hashlib.sha256(
    AGGREGATION_INSTRUCTION_V1.encode("utf-8")
).hexdigest()


AggregationProvider = Callable[[AggregationRequest], Awaitable[AggregationResponse]]
UsageObserver = Callable[[str, int, int], None]


class LLMAggregation:
    """Inference-time team aggregation using the optimizer model identity."""

    identity = versions.UNIFIED_LLM_AGGREGATION_VERSION

    def __init__(
        self, *, runtime: RuntimeModelIdentity, provider: AggregationProvider,
        usage_observer: UsageObserver | None = None,
    ) -> None:
        self.runtime = runtime
        self.provider = provider
        self.usage_observer = usage_observer

    async def aggregate(
        self, *, item: BenchmarkInput, member_outputs: tuple[str, ...],
        benchmark: BenchmarkAdapter,
    ) -> AggregationResult:
        if not member_outputs:
            raise SearchContractError("aggregation needs member outputs")
        public = {
            "instruction": AGGREGATION_INSTRUCTION_V1,
            "problem": benchmark.format_input(item),
            "public_context": dict(item.public_context),
            "output_contract": item.output_contract,
            "equal_status_responses": list(member_outputs),
        }
        request = AggregationRequest(
            model=self.runtime.optimizer_model,
            role="aggregator",
            prompt=json.dumps(public, ensure_ascii=False, sort_keys=True),
            input_id=item.input_id,
            seed=self.runtime.seed,
            decoding_identity=self.runtime.decoding_identity,
        )
        response = await self.provider(request)
        if not response.provider_called and (response.input_tokens or response.output_tokens):
            raise SearchContractError("cached aggregation response has physical token usage")
        if self.usage_observer is not None and response.provider_called:
            self.usage_observer("team_aggregation", response.input_tokens,
                                response.output_tokens)
        parsed = benchmark.parse_output(response.text, item)
        return AggregationResult(
            response.text, parsed, "team_aggregation", request.model,
            int(response.provider_called),
            response.input_tokens + response.output_tokens,
            request.cache_identity(),
            {"prompt_version": AGGREGATION_INSTRUCTION_SHA256},
        )
