"""Pinned GEPA eight-variant constraint metric with an injectable checker registry."""
from dataclasses import dataclass
from typing import Callable, Mapping

from ..search.benchmark import BenchmarkInput
from ..search.schemas import BenchmarkCapabilities, ParsedOutput, SearchContractError


@dataclass(frozen=True)
class ConstraintReference:
    prompt: str
    instruction_ids: tuple[str, ...]
    kwargs: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class ConstraintEvaluation:
    score: float
    instruction_ids: tuple[str, ...]
    success_vector: tuple[bool, ...]
    num_satisfied: int
    num_constraints: int


def response_variants(response: str) -> tuple[str, ...]:
    lines = response.split("\n")
    first = "\n".join(lines[1:]).strip()
    last = "\n".join(lines[:-1]).strip()
    both = "\n".join(lines[1:-1]).strip()
    return (response, response.replace("*", ""), first, last, both,
            first.replace("*", ""), last.replace("*", ""), both.replace("*", ""))


class IFBenchBenchmarkAdapter:
    benchmark_id = "ifbench"
    preferred_aggregation = "llm_raw_response"
    capabilities = BenchmarkCapabilities(False, False, False, False, False)
    output_contract = "Return the raw response to the prompt, without an added FINAL_ANSWER or TEAM_RESPONSE wrapper."

    def __init__(self, checker_registry: Mapping[str, Callable] | None = None) -> None:
        self.checker_registry = checker_registry

    def format_input(self, item: BenchmarkInput) -> str:
        return item.problem

    def parse_member_output(self, raw: str, item: BenchmarkInput) -> ParsedOutput:
        return ParsedOutput(raw, bool(item.benchmark_id == self.benchmark_id and raw.strip()))

    parse_output = parse_member_output

    def evaluate(self, parsed: ParsedOutput, reference: ConstraintReference) -> ConstraintEvaluation:
        if not reference.instruction_ids or len(reference.instruction_ids) != len(reference.kwargs):
            raise SearchContractError("IFBENCH_CONSTRAINT_REFERENCE_INVALID")
        registry = self.checker_registry
        if registry is None:
            from .ifbench_checkers import load_pinned_checkers
            registry = load_pinned_checkers()
        success = []
        variants = response_variants(parsed.answer) if parsed.valid else ()
        for instruction_id, arguments in zip(reference.instruction_ids, reference.kwargs, strict=True):
            if instruction_id not in registry:
                raise SearchContractError("IFBENCH_CHECKER_NOT_FROZEN")
            instruction = registry[instruction_id](instruction_id)
            # Copy/filter rather than mutating upstream example kwargs.
            instruction.build_description(**{k: v for k, v in arguments.items() if v is not None})
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=reference.prompt)
            success.append(any(bool(text.strip()) and instruction.check_following(text) for text in variants))
        satisfied = sum(success)
        return ConstraintEvaluation(satisfied / len(success), reference.instruction_ids,
                                    tuple(success), satisfied, len(success))

    def score_member_output(self, parsed: ParsedOutput, gold: ConstraintReference) -> float:
        return self.evaluate(parsed, gold).score

    def build_task_feedback(self, parsed: ParsedOutput, gold: ConstraintReference) -> str:
        from dataclasses import asdict
        import json
        return json.dumps(asdict(self.evaluate(parsed, gold)), sort_keys=True)
