"""Versioned, scientific Layer-2 evidence-admissibility outcomes."""

from __future__ import annotations

from enum import Enum


class Layer2FeasibilityReason(str, Enum):
    FEASIBLE = "FEASIBLE"
    NO_POSITIVE_RESPONSIBILITY = "NO_POSITIVE_RESPONSIBILITY"
    INSUFFICIENT_PRIMARY_REPAIR_QUOTA = "INSUFFICIENT_PRIMARY_REPAIR_QUOTA"
    INSUFFICIENT_PRESERVATION_QUOTA = "INSUFFICIENT_PRESERVATION_QUOTA"
    INSUFFICIENT_TEAM_HARD_QUOTA = "INSUFFICIENT_TEAM_HARD_QUOTA"
    INSUFFICIENT_PACKET_REPAIR_CAPACITY = "INSUFFICIENT_PACKET_REPAIR_CAPACITY"
    INVALID_PACKET_CONSTRUCTION = "INVALID_PACKET_CONSTRUCTION"


class Layer2EvidenceInfeasible(ValueError):
    """A valid parent state cannot admit this member under frozen V4 quotas."""

    def __init__(self, reason: Layer2FeasibilityReason) -> None:
        self.reason = reason
        super().__init__(reason.value)
