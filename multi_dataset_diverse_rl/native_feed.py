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
LAYER2_EVIDENCE_PACKET_VERSION = "responsibility_evidence_packet_v2_transition_semantics"
LAYER2_EVIDENCE_REQUEST_VERSION = "layer2_owned_evidence_request_v2"
LAYER2_EVIDENCE_SELECTION_POLICY_VERSION = "responsibility_plus_latest_transition_eval_v1"


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
        if self.responsibility_role not in {
            "responsibility", "focus", "anchor", "local_eval"
        }:
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

    @property
    def packet_item_id(self) -> str:
        """Role-qualified identity used when one source example has multiple roles."""

        return f"{self.responsibility_role}:{self.example_id}"


@dataclass(frozen=True)
class CandidateTransitionAudit:
    """Sanitized one-step parent-to-child correctness transition."""

    parent_candidate_hash: str
    child_candidate_hash: str
    parent_correctness: tuple[tuple[str, bool], ...]
    child_correctness: tuple[tuple[str, bool], ...]
    newly_fixed_ids: tuple[str, ...] = field(init=False)
    newly_broken_ids: tuple[str, ...] = field(init=False)
    unchanged_correct_count: int = field(init=False)
    unchanged_wrong_count: int = field(init=False)
    parent_profile_hash: str = field(init=False)
    child_profile_hash: str = field(init=False)
    transition_effect_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.parent_candidate_hash or not self.child_candidate_hash:
            raise ValueError("transition candidate hashes are required")
        parent = dict(self.parent_correctness)
        child = dict(self.child_correctness)
        if (
            not parent
            or set(parent) != set(child)
            or len(parent) != len(self.parent_correctness)
            or len(child) != len(self.child_correctness)
        ):
            raise ValueError("transition profiles require identical unique example ids")
        ordered_ids = sorted(parent)
        fixed = tuple(row_id for row_id in ordered_ids if not parent[row_id] and child[row_id])
        broken = tuple(row_id for row_id in ordered_ids if parent[row_id] and not child[row_id])
        unchanged_correct = sum(parent[row_id] and child[row_id] for row_id in ordered_ids)
        unchanged_wrong = sum(not parent[row_id] and not child[row_id] for row_id in ordered_ids)
        parent_payload = [[row_id, bool(parent[row_id])] for row_id in ordered_ids]
        child_payload = [[row_id, bool(child[row_id])] for row_id in ordered_ids]
        parent_hash = _stable_hash(parent_payload)
        child_hash = _stable_hash(child_payload)
        effect_payload = {
            "parent_candidate_hash": self.parent_candidate_hash,
            "child_candidate_hash": self.child_candidate_hash,
            "parent_profile_hash": parent_hash,
            "child_profile_hash": child_hash,
            "newly_fixed_ids": list(fixed),
            "newly_broken_ids": list(broken),
            "unchanged_correct_count": unchanged_correct,
            "unchanged_wrong_count": unchanged_wrong,
        }
        object.__setattr__(self, "newly_fixed_ids", fixed)
        object.__setattr__(self, "newly_broken_ids", broken)
        object.__setattr__(self, "unchanged_correct_count", unchanged_correct)
        object.__setattr__(self, "unchanged_wrong_count", unchanged_wrong)
        object.__setattr__(self, "parent_profile_hash", parent_hash)
        object.__setattr__(self, "child_profile_hash", child_hash)
        object.__setattr__(self, "transition_effect_hash", _stable_hash(effect_payload))

    def sanitized_payload(self) -> Mapping[str, Any]:
        return {
            "parent_candidate_hash": self.parent_candidate_hash,
            "child_candidate_hash": self.child_candidate_hash,
            "newly_fixed_ids": list(self.newly_fixed_ids),
            "newly_broken_ids": list(self.newly_broken_ids),
            "unchanged_correct_count": self.unchanged_correct_count,
            "unchanged_wrong_count": self.unchanged_wrong_count,
            "parent_profile_hash": self.parent_profile_hash,
            "child_profile_hash": self.child_profile_hash,
            "transition_effect_hash": self.transition_effect_hash,
        }


def transition_audit_from_categorical_profiles(
    *,
    parent_candidate_hash: str,
    child_candidate_hash: str,
    parent_profile: tuple[Mapping[str, Any], ...],
    child_profile: tuple[Mapping[str, Any], ...],
) -> CandidateTransitionAudit:
    """Reconstruct transition roles from persisted sanitized categorical profiles."""

    def correctness(rows: tuple[Mapping[str, Any], ...]) -> tuple[tuple[str, bool], ...]:
        values = []
        for row in rows:
            row_id = row.get("example_id", row.get("example_id_hash"))
            if not isinstance(row_id, str) or not row_id or "correct" not in row:
                raise ValueError("categorical profile row lacks example identity/correctness")
            values.append((row_id, bool(row["correct"])))
        return tuple(values)

    return CandidateTransitionAudit(
        parent_candidate_hash=parent_candidate_hash,
        child_candidate_hash=child_candidate_hash,
        parent_correctness=correctness(parent_profile),
        child_correctness=correctness(child_profile),
    )


@dataclass(frozen=True)
class ResponsibilityEvidencePacket:
    """Complete, immutable Layer-2 curriculum for one local search."""

    source_team_state_hash: str
    target_member: int
    primary_responsibility_lane: str
    responsibility_value: float
    responsibility_context: str
    responsibility_examples: tuple[PacketEvidenceExample, ...]
    focus_examples: tuple[PacketEvidenceExample, ...]
    anchor_examples: tuple[PacketEvidenceExample, ...]
    local_eval_examples: tuple[PacketEvidenceExample, ...]
    ordered_batch_schedule: tuple[tuple[str, ...], ...]
    parent_candidate_hash: str
    lineage_parent_hash: str | None
    data_universe_hash: str
    budget: NativeResourceBudget
    provenance: tuple[tuple[str, str], ...]
    latest_transition: CandidateTransitionAudit | None = None
    packet_version: str = LAYER2_EVIDENCE_PACKET_VERSION
    selection_policy_version: str = LAYER2_EVIDENCE_SELECTION_POLICY_VERSION
    packet_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.source_team_state_hash or not self.data_universe_hash:
            raise ValueError("packet source-state and universe hashes are required")
        if self.target_member < 0 or self.responsibility_value < 0:
            raise ValueError("invalid packet target or responsibility value")
        if not self.responsibility_examples:
            raise ValueError("packet requires responsibility evidence")
        if not self.local_eval_examples:
            raise ValueError("packet requires local-evaluation evidence")
        if not self.parent_candidate_hash:
            raise ValueError("packet parent candidate hash is required")
        role_sets = {
            "responsibility": {row.example_id for row in self.responsibility_examples},
            "focus": {row.example_id for row in self.focus_examples},
            "anchor": {row.example_id for row in self.anchor_examples},
            "local_eval": {row.example_id for row in self.local_eval_examples},
        }
        if any(len(ids) != len(rows) for ids, rows in (
            (role_sets["responsibility"], self.responsibility_examples),
            (role_sets["focus"], self.focus_examples),
            (role_sets["anchor"], self.anchor_examples),
            (role_sets["local_eval"], self.local_eval_examples),
        )):
            raise ValueError("packet evidence ids must be unique within each role")
        scheduled = {
            row.packet_item_id
            for row in (
                *self.responsibility_examples, *self.focus_examples, *self.anchor_examples
            )
        }
        if not self.ordered_batch_schedule or any(
            not batch or not set(batch).issubset(scheduled)
            for batch in self.ordered_batch_schedule
        ):
            raise ValueError("packet schedule must contain only selected search evidence")
        if set().union(*(set(batch) for batch in self.ordered_batch_schedule)) != scheduled:
            raise ValueError("packet schedule must cover every selected search example")
        if any(row.responsibility_role != "responsibility" for row in self.responsibility_examples):
            raise ValueError("responsibility evidence role mismatch")
        if any(row.responsibility_role != "focus" for row in self.focus_examples):
            raise ValueError("focus evidence role mismatch")
        if any(row.responsibility_role != "anchor" for row in self.anchor_examples):
            raise ValueError("anchor evidence role mismatch")
        if any(row.responsibility_role != "local_eval" for row in self.local_eval_examples):
            raise ValueError("local-evaluation evidence role mismatch")
        if self.latest_transition is None:
            if self.focus_examples or self.anchor_examples or self.lineage_parent_hash is not None:
                raise ValueError("root packet cannot synthesize transition evidence")
        else:
            if self.lineage_parent_hash != self.latest_transition.parent_candidate_hash:
                raise ValueError("packet lineage parent differs from transition parent")
            if self.parent_candidate_hash != self.latest_transition.child_candidate_hash:
                raise ValueError("packet parent differs from transition child")
            if role_sets["focus"] != set(self.latest_transition.newly_broken_ids):
                raise ValueError("focus evidence is not the exact newly-broken set")
            if role_sets["anchor"] != set(self.latest_transition.newly_fixed_ids):
                raise ValueError("anchor evidence is not the exact newly-fixed set")
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
            "responsibility_examples": [
                row.identity_payload() for row in self.responsibility_examples
            ],
            "focus_examples": [row.identity_payload() for row in self.focus_examples],
            "anchor_examples": [row.identity_payload() for row in self.anchor_examples],
            "local_eval_examples": [
                row.identity_payload() for row in self.local_eval_examples
            ],
            "ordered_batch_schedule": [list(batch) for batch in self.ordered_batch_schedule],
            "parent_candidate_hash": self.parent_candidate_hash,
            "lineage_parent_hash": self.lineage_parent_hash,
            "data_universe_hash": self.data_universe_hash,
            "selection_policy_version": self.selection_policy_version,
            "budget": self.budget.__dict__,
            "provenance": list(self.provenance),
            "latest_transition": (
                self.latest_transition.sanitized_payload()
                if self.latest_transition is not None else None
            ),
        }
        return {**payload, **({"packet_hash": self.packet_hash} if include_hash else {})}

    def role_id_hash(self, role: str) -> str:
        rows = {
            "responsibility": self.responsibility_examples,
            "focus": self.focus_examples,
            "anchor": self.anchor_examples,
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
            "responsibility_example_ids_hash": self.packet.role_id_hash("responsibility"),
            "focus_example_ids_hash": self.packet.role_id_hash("focus"),
            "anchor_example_ids_hash": self.packet.role_id_hash("anchor"),
            "local_eval_example_ids_hash": self.packet.role_id_hash("local_eval"),
            "batch_schedule_hash": self.packet.batch_schedule_hash,
            "backend": backend,
            "backend_version": backend_version,
        }
