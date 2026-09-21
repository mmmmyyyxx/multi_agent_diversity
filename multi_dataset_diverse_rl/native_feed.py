"""Backend-neutral outer-control contract for native-feed optimizers.

Layer 2 chooses a member and describes the team responsibility.  It does not
choose optimizer examples.  Each backend resolves ``optimize_universe_id``
into its own native data-flow after receiving this request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping


NATIVE_FEED_REQUEST_VERSION = "backend_native_feed_request_v1"
LAYER2_RESPONSIBILITY_CONTEXT_VERSION = "layer2_responsibility_context_v1"


@dataclass(frozen=True)
class NativeResourceBudget:
    """Comparable envelope whose native unit is interpreted by the backend."""

    native_unit_limit: int
    metric_call_limit: int
    optimizer_call_limit: int
    max_returned_candidates: int = 4
    semantics: str = "backend_native_budget_v1"

    def __post_init__(self) -> None:
        if min(
            self.native_unit_limit,
            self.metric_call_limit,
            self.optimizer_call_limit,
            self.max_returned_candidates,
        ) <= 0:
            raise ValueError("native-feed budget fields must be positive")


@dataclass(frozen=True)
class ResponsibilityContext:
    """Sanitized team responsibility metadata; never a sample selector."""

    primary_lane: str
    responsibility_identity: str
    responsibility_value: float
    team_failure_summary: Mapping[str, int] = field(default_factory=dict)
    coverage_summary: Mapping[str, int] = field(default_factory=dict)
    peer_structure_summary: Mapping[str, int | str] = field(default_factory=dict)
    version: str = LAYER2_RESPONSIBILITY_CONTEXT_VERSION

    def __post_init__(self) -> None:
        if self.primary_lane not in {
            "direct_flip",
            "near_margin",
            "coverage",
            "fallback",
            "generic",
        }:
            raise ValueError("unknown responsibility lane")
        if not self.responsibility_identity:
            raise ValueError("responsibility identity is required")
        if self.responsibility_value < 0:
            raise ValueError("responsibility value cannot be negative")

    def overlay_payload(self) -> Mapping[str, Any]:
        return {
            "version": self.version,
            "primary_lane": self.primary_lane,
            "responsibility_identity": self.responsibility_identity,
            "responsibility_value": self.responsibility_value,
            "team_failure_summary": dict(self.team_failure_summary),
            "coverage_summary": dict(self.coverage_summary),
            "peer_structure_summary": dict(self.peer_structure_summary),
        }

    def canonical_overlay(self) -> str:
        return json.dumps(
            self.overlay_payload(), sort_keys=True, separators=(",", ":")
        )


@dataclass(frozen=True)
class NativeOptimizationRequest:
    """Minimal shared request; backend-specific examples are intentionally absent."""

    request_id: str
    parent_decision_procedure: str
    target_member: int
    responsibility: ResponsibilityContext
    team_state_identity: str
    optimize_universe_id: str
    solver_contract_id: str
    output_contract_id: str
    seed: int
    budget: NativeResourceBudget
    provenance: Mapping[str, str]
    backend_state: Mapping[str, Any] | None = None
    request_version: str = NATIVE_FEED_REQUEST_VERSION

    def __post_init__(self) -> None:
        if not self.request_id or not self.parent_decision_procedure:
            raise ValueError("native-feed request identity and parent are required")
        if self.target_member < 0:
            raise ValueError("target member cannot be negative")
        if not self.team_state_identity or not self.optimize_universe_id:
            raise ValueError("team state and Optimize-universe identities are required")
        if not self.solver_contract_id or not self.output_contract_id:
            raise ValueError("solver and output contracts are required")
        forbidden = {
            "search_examples",
            "local_validation_examples",
            "reflection_minibatch",
            "example_ids",
        }
        if forbidden.intersection(self.provenance):
            raise ValueError("Layer 2 provenance cannot materialize backend samples")

    def identity(self) -> str:
        payload = {
            "request_version": self.request_version,
            "request_id": self.request_id,
            "parent_sha256": hashlib.sha256(
                self.parent_decision_procedure.encode("utf-8")
            ).hexdigest(),
            "target_member": self.target_member,
            "responsibility": self.responsibility.overlay_payload(),
            "team_state_identity": self.team_state_identity,
            "optimize_universe_id": self.optimize_universe_id,
            "solver_contract_id": self.solver_contract_id,
            "output_contract_id": self.output_contract_id,
            "seed": self.seed,
            "budget": self.budget.__dict__,
            "provenance": dict(self.provenance),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

