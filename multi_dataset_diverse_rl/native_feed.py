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
LAYER2_EVIDENCE_PACKET_VERSION = "responsibility_evidence_packet_v1"
LAYER2_EVIDENCE_REQUEST_VERSION = "layer2_owned_evidence_request_v1"
LAYER2_EVIDENCE_SELECTION_POLICY_VERSION = "deterministic_repair_preservation_eval_v1"


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


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PacketEvidenceExample:
    """One immutable Optimize-derived example selected by Layer 2."""

    example_id: str
    input_payload: str
    gold: str
    parent_output: str | None
    textual_feedback: str | None
    responsibility_role: str
    lane: str
    metadata: tuple[tuple[str, str | int | bool], ...] = ()

    def __post_init__(self) -> None:
        if not self.example_id or not self.input_payload or not self.gold:
            raise ValueError("packet evidence identity, input, and gold are required")
        if self.responsibility_role not in {"repair", "preservation", "local_eval"}:
            raise ValueError("unknown packet evidence role")
        if self.lane not in {
            "direct_flip", "near_margin", "coverage", "fallback", "global"
        }:
            raise ValueError("unknown packet evidence lane")
        keys = [key for key, _ in self.metadata]
        if len(keys) != len(set(keys)):
            raise ValueError("packet evidence metadata keys must be unique")

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "example_id": self.example_id,
            "input_payload": self.input_payload,
            "gold": self.gold,
            "parent_output": self.parent_output,
            "textual_feedback": self.textual_feedback,
            "responsibility_role": self.responsibility_role,
            "lane": self.lane,
            "metadata": list(self.metadata),
        }


@dataclass(frozen=True)
class ResponsibilityEvidencePacket:
    """Complete, immutable Layer-2 curriculum for one local search."""

    source_team_state_hash: str
    target_member: int
    primary_responsibility_lane: str
    responsibility_value: float
    responsibility_context: str
    repair_examples: tuple[PacketEvidenceExample, ...]
    preservation_examples: tuple[PacketEvidenceExample, ...]
    local_eval_examples: tuple[PacketEvidenceExample, ...]
    ordered_batch_schedule: tuple[tuple[str, ...], ...]
    data_universe_hash: str
    budget: NativeResourceBudget
    provenance: tuple[tuple[str, str], ...]
    packet_version: str = LAYER2_EVIDENCE_PACKET_VERSION
    selection_policy_version: str = LAYER2_EVIDENCE_SELECTION_POLICY_VERSION
    packet_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.source_team_state_hash or not self.data_universe_hash:
            raise ValueError("packet source-state and universe hashes are required")
        if self.target_member < 0 or self.responsibility_value < 0:
            raise ValueError("invalid packet target or responsibility value")
        if not self.repair_examples:
            raise ValueError("packet requires repair evidence")
        if not self.preservation_examples:
            raise ValueError("packet requires preservation evidence")
        if not self.local_eval_examples:
            raise ValueError("packet requires local-evaluation evidence")
        role_sets = {
            "repair": {row.example_id for row in self.repair_examples},
            "preservation": {row.example_id for row in self.preservation_examples},
            "local_eval": {row.example_id for row in self.local_eval_examples},
        }
        if any(len(ids) != len(rows) for ids, rows in (
            (role_sets["repair"], self.repair_examples),
            (role_sets["preservation"], self.preservation_examples),
            (role_sets["local_eval"], self.local_eval_examples),
        )):
            raise ValueError("packet evidence ids must be unique within each role")
        if (
            role_sets["repair"] & role_sets["preservation"]
            or role_sets["repair"] & role_sets["local_eval"]
            or role_sets["preservation"] & role_sets["local_eval"]
        ):
            raise ValueError("default packet policy requires disjoint evidence roles")
        scheduled = role_sets["repair"] | role_sets["preservation"]
        if not self.ordered_batch_schedule or any(
            not batch or not set(batch).issubset(scheduled)
            for batch in self.ordered_batch_schedule
        ):
            raise ValueError("packet schedule must contain only selected search evidence")
        if set().union(*(set(batch) for batch in self.ordered_batch_schedule)) != scheduled:
            raise ValueError("packet schedule must cover every selected search example")
        if any(row.responsibility_role != "repair" for row in self.repair_examples):
            raise ValueError("repair evidence role mismatch")
        if any(
            row.responsibility_role != "preservation"
            for row in self.preservation_examples
        ):
            raise ValueError("preservation evidence role mismatch")
        if any(row.responsibility_role != "local_eval" for row in self.local_eval_examples):
            raise ValueError("local-evaluation evidence role mismatch")
        payload = self.identity_payload(include_hash=False)
        object.__setattr__(self, "packet_hash", _stable_hash(payload))

    def identity_payload(self, *, include_hash: bool = True) -> Mapping[str, Any]:
        payload = {
            "packet_version": self.packet_version,
            "source_team_state_hash": self.source_team_state_hash,
            "target_member": self.target_member,
            "primary_responsibility_lane": self.primary_responsibility_lane,
            "responsibility_value": self.responsibility_value,
            "responsibility_context": self.responsibility_context,
            "repair_examples": [row.identity_payload() for row in self.repair_examples],
            "preservation_examples": [
                row.identity_payload() for row in self.preservation_examples
            ],
            "local_eval_examples": [
                row.identity_payload() for row in self.local_eval_examples
            ],
            "ordered_batch_schedule": [list(batch) for batch in self.ordered_batch_schedule],
            "data_universe_hash": self.data_universe_hash,
            "selection_policy_version": self.selection_policy_version,
            "budget": self.budget.__dict__,
            "provenance": list(self.provenance),
        }
        return {**payload, **({"packet_hash": self.packet_hash} if include_hash else {})}

    def role_id_hash(self, role: str) -> str:
        rows = {
            "repair": self.repair_examples,
            "preservation": self.preservation_examples,
            "local_eval": self.local_eval_examples,
        }[role]
        return _stable_hash([row.example_id for row in rows])

    @property
    def batch_schedule_hash(self) -> str:
        return _stable_hash([list(batch) for batch in self.ordered_batch_schedule])


@dataclass(frozen=True)
class Layer2OptimizationRequest:
    """Treatment request whose complete evidence curriculum is Layer-2-owned."""

    request_id: str
    parent_decision_procedure: str
    packet: ResponsibilityEvidencePacket
    solver_contract_id: str
    output_contract_id: str
    seed: int
    request_version: str = LAYER2_EVIDENCE_REQUEST_VERSION

    def __post_init__(self) -> None:
        if not self.request_id or not self.parent_decision_procedure:
            raise ValueError("Layer-2 evidence request identity and parent are required")
        if not self.solver_contract_id or not self.output_contract_id:
            raise ValueError("solver and output contracts are required")

    def identity(self) -> str:
        return _stable_hash(
            {
                "request_version": self.request_version,
                "request_id": self.request_id,
                "parent_sha256": hashlib.sha256(
                    self.parent_decision_procedure.encode("utf-8")
                ).hexdigest(),
                "packet_hash": self.packet.packet_hash,
                "solver_contract_id": self.solver_contract_id,
                "output_contract_id": self.output_contract_id,
                "seed": self.seed,
            }
        )

    def candidate_provenance(self, *, backend: str, backend_version: str) -> Mapping[str, Any]:
        return {
            "source_team_state_hash": self.packet.source_team_state_hash,
            "target_member": self.packet.target_member,
            "responsibility_lane": self.packet.primary_responsibility_lane,
            "responsibility_packet_hash": self.packet.packet_hash,
            "repair_example_ids_hash": self.packet.role_id_hash("repair"),
            "preservation_example_ids_hash": self.packet.role_id_hash("preservation"),
            "local_eval_example_ids_hash": self.packet.role_id_hash("local_eval"),
            "batch_schedule_hash": self.packet.batch_schedule_hash,
            "backend": backend,
            "backend_version": backend_version,
        }

