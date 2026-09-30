"""Public benchmark semantics supplied to team search and aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .schemas import BenchmarkCapabilities, ParsedOutput


@dataclass(frozen=True)
class BenchmarkInput:
    """Inference-visible input. Gold and evaluation fields have no slot here."""

    input_id: str
    problem: str
    output_contract: str
    public_context: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        forbidden = {"gold", "reward", "correct_vector", "evaluation_result",
                     "label", "reference_answer"}
        if forbidden.intersection(key.casefold() for key in self.public_context):
            raise ValueError("inference input contains evaluation-only context")


class BenchmarkAdapter(Protocol):
    @property
    def capabilities(self) -> BenchmarkCapabilities: ...

    @property
    def preferred_aggregation(self) -> str: ...

    def format_input(self, item: BenchmarkInput) -> str: ...

    def parse_member_output(self, raw: str, item: BenchmarkInput) -> ParsedOutput: ...

    def parse_output(self, raw: str, item: BenchmarkInput) -> ParsedOutput: ...

    def score_member_output(self, parsed: ParsedOutput, gold: Any) -> float: ...

    def build_task_feedback(self, parsed: ParsedOutput, gold: Any) -> str: ...


class BBHBenchmarkAdapter:
    """Current strict BBH parser, kept behind the benchmark boundary."""

    capabilities = BenchmarkCapabilities(True, True, True)
    preferred_aggregation = "plurality"

    def __init__(self, task_spec: Any, *, answer_format: str = "option_letter") -> None:
        self.task_spec = task_spec
        self.answer_format = answer_format

    def format_input(self, item: BenchmarkInput) -> str:
        return item.problem

    def parse_member_output(self, raw: str, item: BenchmarkInput) -> ParsedOutput:
        from ..evaluation.solver_output import parse_solver_output

        parsed = parse_solver_output(raw, question=item.problem,
                                     task_spec=self.task_spec,
                                     answer_format=self.answer_format)
        return ParsedOutput(parsed.answer, parsed.valid)

    def parse_output(self, raw: str, item: BenchmarkInput) -> ParsedOutput:
        return self.parse_member_output(raw, item)

    def score_member_output(self, parsed: ParsedOutput, gold: Any) -> float:
        if not parsed.valid:
            return 0.0
        from ..answer_formats import match_answer

        return float(match_answer(parsed.answer, gold, self.answer_format))

    def build_task_feedback(self, parsed: ParsedOutput, gold: Any) -> str:
        return "correct" if self.score_member_output(parsed, gold) else "incorrect"
