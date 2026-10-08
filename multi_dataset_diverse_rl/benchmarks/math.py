"""Strict final-expression scoring with pinned, offline math-verify equivalence."""
from __future__ import annotations

from functools import lru_cache
import json
import re
import subprocess
import sys

from ..search.benchmark import BenchmarkInput
from ..search.schemas import BenchmarkCapabilities, ParsedOutput, SearchContractError
from .. import versions
from .math_interface import MATH_SOLVER_INTERFACE_V6, v6_interface_contract


MATH_PROCESS_DEADLINE_SECONDS = versions.MATH_EVALUATOR_PROCESS_DEADLINE_SECONDS


@lru_cache(maxsize=512)
def equivalence_matrix(expressions: tuple[str, ...]) -> tuple[tuple[bool, ...], tuple[tuple[bool, ...], ...]]:
    if not expressions or len(expressions) > 6 or any(len(x) > 2048 for x in expressions):
        raise SearchContractError("MATH_EXPRESSION_SHAPE_INVALID")
    try:
        result = subprocess.run([sys.executable, "-m", "multi_dataset_diverse_rl.benchmarks.math_worker"],
            input=json.dumps({"expressions": expressions}), capture_output=True, text=True,
            timeout=MATH_PROCESS_DEADLINE_SECONDS, check=False)
        if result.returncode:
            raise SearchContractError("MATH_EVALUATOR_FAILURE")
        row = json.loads(result.stdout)
        valid = tuple(row["valid"])
        matrix = tuple(tuple(x) for x in row["equivalence"])
        if (len(valid) != len(expressions) or len(matrix) != len(valid)
                or any(len(x) != len(valid) for x in matrix)
                or any(type(x) is not bool for x in valid)
                or any(type(x) is not bool for r in matrix for x in r)):
            raise SearchContractError("MATH_EVALUATOR_RESPONSE_INVALID")
        return valid, matrix
    except subprocess.TimeoutExpired as exc:
        raise SearchContractError("MATH_EVALUATOR_TIMEOUT") from exc
    except (ValueError, KeyError, OSError) as exc:
        raise SearchContractError("MATH_EVALUATOR_FAILURE") from exc


def payload_supported(text: str) -> bool:
    if not text.strip() or len(text) > 2048 or "\n" in text:
        return False
    # The pinned LaTeX extractor interprets comma-parentheses as FiniteSet.
    # Ordered tuple semantics are unsupported in V1, rather than silently lost.
    if re.search(r"\([^)]*,[^)]*\)", text):
        return False
    # Unknown prose is not an answer expression. Supported LaTeX commands
    # are delegated to the pinned parser, never repaired or guessed here.
    without_commands = re.sub(r"\\[A-Za-z]+", "", text)
    words = re.findall(r"[A-Za-z]{2,}", without_commands)
    return not any(word not in {"sqrt", "sin", "cos", "tan", "log", "ln", "pi", "oo"} for word in words)


class MATHBenchmarkAdapter:
    benchmark_id = "math"
    preferred_aggregation = "equivalence_plurality"
    capabilities = BenchmarkCapabilities(True, True, True, True, True)
    output_contract=MATH_SOLVER_INTERFACE_V6
    solver_interface_contract=staticmethod(v6_interface_contract)

    def format_input(self, item: BenchmarkInput) -> str:
        return item.problem

    def parse_member_output(self, raw: str, item: BenchmarkInput) -> ParsedOutput:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        marked = [line for line in lines if line.startswith("FINAL_ANSWER:")]
        if item.benchmark_id != self.benchmark_id or len(marked) != 1 or lines[-1] != marked[0]:
            return ParsedOutput("", False)
        expression = marked[0].removeprefix("FINAL_ANSWER:").strip()
        if not payload_supported(expression):
            return ParsedOutput("", False)
        try:
            valid, _ = equivalence_matrix((expression,))
            return ParsedOutput(expression, valid[0])
        except SearchContractError:
            return ParsedOutput("", False)

    parse_output = parse_member_output

    def equivalent(self, left: str, right: str) -> bool:
        if not all(payload_supported(s) for s in (left, right)):
            return False
        valid, matrix = equivalence_matrix((left, right))
        # Symmetric relation: an asymmetric grader result fails closed.
        if matrix[0][1] != matrix[1][0]:
            raise SearchContractError("EQUIVALENCE_RELATION_INCONSISTENT")
        return all(valid) and matrix[0][1]

    def score_member_output(self, parsed: ParsedOutput, gold: object) -> float:
        if not parsed.valid:
            return 0.0
        try:
            return float(self.equivalent(str(gold), parsed.answer))
        except SearchContractError:
            return 0.0

    def build_task_feedback(self, parsed: ParsedOutput, gold: object) -> str:
        return "correct" if self.score_member_output(parsed, gold) else "incorrect"
