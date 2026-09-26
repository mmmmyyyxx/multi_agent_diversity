"""Layered protocol identities for two_layer_rg_gepa_v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..versions import (
    LAYER2_TEAM_SEARCH_PROTOCOL_VERSION,
    PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
)

def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TeamSearchContract:
    protocol_version: str = LAYER2_TEAM_SEARCH_PROTOCOL_VERSION
    target_policy: str = PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION
    team_minibatch_composition: str = "4_repair_4_preservation_4_team_hard"
    team_minibatch_size: int = 12
    team_minibatch_responsibility_lane: str = "primary_lane_aligned"
    team_minibatch_quota_semantics: str = "strict_no_backfill"
    max_full_candidates: int = 2
    team_selection: str = "current_common_safe"
    shadow_policy: str = "winner_only_shadow_gate_v1"
    aggregation: str = "equal_weight_plurality_tie_abstain"

    def identity(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class TwoLayerProtocolIdentity:
    solver_contract_hash: str
    dataset_contract_hash: str
    local_optimizer_contract_hash: str
    team_search_contract_hash: str

    @property
    def full_protocol_hash(self) -> str:
        return _hash(asdict(self))
