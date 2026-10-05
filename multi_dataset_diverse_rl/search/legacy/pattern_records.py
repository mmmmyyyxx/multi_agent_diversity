"""Immutable Pattern observation schema shared with explicit replay."""
from dataclasses import dataclass, field
from typing import Mapping, Any
from ..schemas import SearchContractError

@dataclass(frozen=True)
class PatternHypothesis:
    pattern_id: str
    failure_mechanism: str
    corrective_principle: str
    support_ids: tuple[str, ...]
    counterexample_ids: tuple[str, ...] = ()
    risk_ids: tuple[str, ...] = ()
    confidence: float = 0.0

    def __post_init__(self):
        if not self.pattern_id or not self.failure_mechanism or not self.corrective_principle:
            raise SearchContractError("pattern mechanism and principle required")
        if not 0 <= self.confidence <= 1:
            raise SearchContractError("pattern confidence out of bounds")
        for ids in (self.support_ids, self.counterexample_ids, self.risk_ids):
            if len(ids) != len(set(ids)):
                raise SearchContractError("duplicate pattern IDs")
