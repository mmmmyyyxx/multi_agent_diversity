"""Explicit adapters for tasks whose project protocol cannot yet be inferred."""

from __future__ import annotations

from ..search.benchmark import BenchmarkInput
from ..search.schemas import BenchmarkCapabilities, ParsedOutput, SearchContractError


class _UnfrozenAdapter:
    capabilities = BenchmarkCapabilities(False, False, False)
    preferred_aggregation = "unfrozen"
    benchmark_id = ""

    def _hold(self) -> None:
        raise SearchContractError(f"BENCHMARK_PROVENANCE_NOT_FROZEN: {self.benchmark_id}")

    def format_input(self, item: BenchmarkInput) -> str:
        del item
        self._hold()

    def parse_member_output(self, raw: str, item: BenchmarkInput) -> ParsedOutput:
        del raw, item
        self._hold()

    parse_output = parse_member_output

    def score_member_output(self, parsed: ParsedOutput, gold: object) -> float:
        del parsed, gold
        self._hold()

    def build_task_feedback(self, parsed: ParsedOutput, gold: object) -> str:
        del parsed, gold
        self._hold()


class HoVerBenchmarkAdapter(_UnfrozenAdapter):
    benchmark_id = "hover"


class IFBenchBenchmarkAdapter(_UnfrozenAdapter):
    benchmark_id = "ifbench"


class PUPABenchmarkAdapter(_UnfrozenAdapter):
    benchmark_id = "pupa"


class MATHBenchmarkAdapter(_UnfrozenAdapter):
    benchmark_id = "math"
