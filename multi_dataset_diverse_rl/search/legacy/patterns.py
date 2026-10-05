"""Target-specific Optimize-only mechanism diagnosis, dormant without binding."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import math
from typing import Protocol, Sequence

from ... import versions
from ..schemas import Diagnosis, EvidenceItem, SearchContractError, TeamStateSnapshot

PATTERN_PROMPT_CONTRACT = (
    "Identify failure mechanisms, missing reasoning steps and corrective principles, "
    "not topics, entities or keywords. Reference only supplied Optimize example IDs. "
    "Do not rewrite the final prompt, choose the target or rank members."
)


from .pattern_records import PatternHypothesis


@dataclass(frozen=True)
class PatternInput:
    parent_state_id: str
    target_member: int
    primary_lane: str
    evidence_rows: tuple[EvidenceItem, ...]
    residual_ids: tuple[str, ...]
    structured_history: tuple[tuple[str, int], ...]
    prompt_contract: str = PATTERN_PROMPT_CONTRACT


class PatternDiagnosticProvider(Protocol):
    def diagnose(self, request: PatternInput) -> Sequence[PatternHypothesis]: ...


@dataclass(frozen=True)
class PatternDiagnostic:
    patterns: tuple[PatternHypothesis, ...]
    dominant_pattern_id: str | None
    support_count: int
    total_residual_count: int
    dominant_pattern_ratio: float
    normalized_entropy: float
    mixed_pattern_count: int
    unassigned_residual_ids: tuple[str, ...]

    def context(self):
        return asdict(self)


class PatternDiagnosticV1:
    identity = versions.UNIFIED_PATTERN_DIAGNOSTIC_VERSION

    def __init__(self, provider: PatternDiagnosticProvider | None = None):
        if provider is None:
            raise SearchContractError("PATTERN_PROVIDER_NOT_BOUND")
        self.provider = provider

    def analyze(self, state, diagnosis, target_member, evidence_rows, history):
        rows = tuple(evidence_rows)
        if any(r.source_split != "optimize" for r in rows):
            raise SearchContractError("pattern may read Optimize only")
        universe = {r.example_id for r in rows}
        if len(universe) != len(rows):
            raise SearchContractError("pattern universe duplicate IDs")
        residuals = {r.example_id for r in rows if "REPAIR" in r.roles}
        lane = diagnosis.responsibility[target_member].primary_lane
        primary = {r.example_id for r in rows if "REPAIR" in r.roles and
                   (lane == "fallback" or lane in r.roles or r.signals.get("lane") == lane)}
        # No full state object, gold outside Optimize, other-member outputs,
        # Shadow profile or future transition is passed to the provider.
        request = PatternInput(state.team_state_id, target_member, lane, rows,
                               tuple(sorted(residuals)),
                               (("target_count", history.target_counts.get(target_member, 0)),))
        patterns = tuple(self.provider.diagnose(request))
        if len({p.pattern_id for p in patterns}) != len(patterns):
            raise SearchContractError("duplicate pattern identity")
        assigned = set()
        for p in patterns:
            if not set(p.support_ids) <= residuals or not set((*p.counterexample_ids, *p.risk_ids)) <= universe:
                raise SearchContractError("pattern IDs outside target Optimize universe")
            if assigned & set(p.support_ids):
                raise SearchContractError("pattern supports must form disjoint mechanisms")
            assigned.update(p.support_ids)
        active = tuple(p for p in patterns if p.support_ids)
        focus = min(active, key=lambda p: (-len(primary & set(p.support_ids)), -len(p.support_ids),
                                          -p.confidence, hashlib.sha256(p.pattern_id.encode()).hexdigest(),
                                          p.pattern_id)) if active else None
        total = len(assigned)
        weights = [len(p.support_ids) / total for p in active] if total else []
        entropy = -sum(w * math.log(w) for w in weights) / math.log(len(weights)) if len(weights) > 1 else 0.0
        return PatternDiagnostic(patterns, focus.pattern_id if focus else None,
                                 len(focus.support_ids) if focus else 0, len(residuals),
                                 len(focus.support_ids) / total if focus and total else 0.0,
                                 min(1.0, max(0.0, entropy)), len(active),
                                 tuple(sorted(residuals - assigned))).context()
