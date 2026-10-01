"""Frozen benchmark aggregation contracts, independent of evaluation gold."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

from .. import versions
from ..benchmarks.hover import HoVerBenchmarkAdapter
from .aggregation import LLMAggregation, PluralityAggregation, AggregationResult
from .schemas import ParsedOutput, SearchContractError


def require_five(outputs: Sequence[object]) -> None:
    if len(outputs) != 5:
        raise SearchContractError("TEAM_REQUIRES_EXACTLY_FIVE_EQUAL_MEMBERS")


def equivalence_classes(answers: Sequence[str], relation: Callable[[str, str], bool]) -> tuple[tuple[int, ...], ...]:
    """Validate the entire finite relation before admitting any vote classes."""
    try:
        matrix = [[relation(a, b) for b in answers] for a in answers]
    except SearchContractError:
        raise
    except Exception as exc:
        raise SearchContractError("EQUIVALENCE_EVALUATOR_FAILURE") from exc
    n = len(answers)
    if (any(type(x) is not bool for r in matrix for x in r)
            or any(not matrix[i][i] for i in range(n))
            or any(matrix[i][j] != matrix[j][i] for i in range(n) for j in range(n))
            or any(matrix[i][j] and matrix[j][k] and not matrix[i][k]
                   for i in range(n) for j in range(n) for k in range(n))):
        raise SearchContractError("EQUIVALENCE_RELATION_INCONSISTENT")
    groups = []
    assigned: set[int] = set()
    for i in range(n):
        if i not in assigned:
            group = tuple(j for j in range(n) if matrix[i][j])
            groups.append(group)
            assigned.update(group)
    return tuple(groups)


class NormalizedAnswerPluralityAggregation(PluralityAggregation):
    identity = "normalized_equal_plurality_v1"

    async def aggregate(self, *, item, member_outputs, benchmark):
        require_five(member_outputs)
        if item.benchmark_id != "hotpotqa":
            raise SearchContractError("AGGREGATION_BENCHMARK_MISMATCH")
        return await super().aggregate(item=item, member_outputs=member_outputs, benchmark=benchmark)


class EquivalencePluralityAggregation:
    identity = versions.EQUIVALENCE_PLURALITY_VERSION

    def __init__(self, relation: Callable[[str, str], bool] | None = None) -> None:
        self.relation = relation

    async def aggregate(self, *, item, member_outputs, benchmark):
        return self.aggregate_sync(item=item, member_outputs=member_outputs, benchmark=benchmark)

    def aggregate_sync(self, *, item, member_outputs, benchmark):
        require_five(member_outputs)
        if not benchmark.capabilities.binary_plurality_responsibility:
            raise SearchContractError("EQUIVALENCE_PLURALITY_CAPABILITY_REQUIRED")
        parsed = tuple(benchmark.parse_member_output(raw, item) for raw in member_outputs)
        indices = tuple(i for i, row in enumerate(parsed) if row.valid)
        try:
            groups = equivalence_classes(tuple(parsed[i].answer for i in indices),
                                         self.relation or benchmark.equivalent)
        except SearchContractError as exc:
            return AggregationResult("", ParsedOutput("", False), "team_aggregation", None, 0, 0, None,
                {"abstention_reason": str(exc), "invalid_abstentions": 5 - len(indices)})
        ordered = sorted(groups, key=lambda g: (-len(g), g[0]))
        tie = len(ordered) > 1 and len(ordered[0]) == len(ordered[1])
        winner = indices[ordered[0][0]] if ordered and not tie else None
        return AggregationResult(member_outputs[winner] if winner is not None else "",
            parsed[winner] if winner is not None else ParsedOutput("", False), "team_aggregation",
            None, 0, 0, None, {"equivalence_classes": tuple(tuple(indices[i] for i in g) for g in groups),
                "tie_abstained": tie, "invalid_abstentions": 5 - len(indices), "winner_member": winner})


class HoVerEvidenceAggregation(LLMAggregation):
    identity = "llm_evidence_union_24_v1"
    request_role = "team_aggregation"
    instruction = (
        "Select evidence from five equal-status members for the claim. Return only JSON with titles. "
        "Every title must belong to the union of proposed member titles. Return at most 24 distinct titles. "
        "Do not produce a supported/refuted verdict. No member has special authority."
    )
    max_titles = versions.HOVER_AGGREGATE_MAX_TITLES  # Engineering ceiling, not official HoVer metric.

    async def aggregate(self, *, item, member_outputs, benchmark):
        require_five(member_outputs)
        if item.benchmark_id != "hover" or not isinstance(benchmark, HoVerBenchmarkAdapter):
            raise SearchContractError("AGGREGATION_BENCHMARK_MISMATCH")
        union = set()
        public = []
        for raw in member_outputs:
            parsed = benchmark.parse_member_output(raw, item)
            if parsed.valid:
                union.update(benchmark.evidence(parsed).titles)
                # Send evidence titles only; no evaluation state or hidden passages.
                import json
                public.append(json.dumps({"titles": benchmark.evidence(parsed).titles}))
            else:
                public.append('{"titles": []}')
        result = await super().aggregate(item=item, member_outputs=tuple(public), benchmark=benchmark)
        parsed = result.parsed_output
        reason = None
        if parsed.valid:
            evidence = benchmark.evidence(parsed)
            if not set(evidence.titles) <= union:
                reason = "HOVER_AGGREGATE_TITLE_OUTSIDE_MEMBER_UNION"
            elif len(evidence.titles) > self.max_titles:
                reason = "HOVER_AGGREGATE_EVIDENCE_CEILING_EXCEEDED"
        if reason:
            return replace(result, parsed_output=ParsedOutput("", False),
                           diagnostics={**result.diagnostics, "invalid_reason": reason})
        return result


class IFBenchRawResponseAggregation(LLMAggregation):
    identity = "llm_equal_raw_responses_v1"
    request_role = "team_aggregation"
    instruction = (
        "Produce one response to the original prompt using five equal-status raw member responses. "
        "Follow only the original prompt's requirements. No member has special authority. "
        "Return the raw response only. Do not add FINAL_ANSWER, TEAM_RESPONSE, or a markdown wrapper."
    )

    async def aggregate(self, *, item, member_outputs, benchmark):
        require_five(member_outputs)
        if item.benchmark_id != "ifbench":
            raise SearchContractError("AGGREGATION_BENCHMARK_MISMATCH")
        return await super().aggregate(item=item, member_outputs=member_outputs, benchmark=benchmark)
