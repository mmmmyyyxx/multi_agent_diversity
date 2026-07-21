from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List


class BehaviorContext(str, Enum):
    INVALID = "invalid"
    TEAM_WRONG_PIVOTAL_FIX = "team_wrong_pivotal_fix"
    TEAM_WRONG_NONPIVOTAL = "team_wrong_nonpivotal"
    TEAM_CORRECT_DOMINANT_WRONG_REDUNDANCY = "team_correct_dominant_wrong_redundancy"
    TEAM_CORRECT_TARGET_WRONG_OTHER = "team_correct_target_wrong_other"
    TARGET_CORRECT_PIVOTAL_HOLD = "target_correct_pivotal_hold"
    TARGET_CORRECT_ROBUST = "target_correct_robust"


class CapabilityResidualFamily(str, Enum):
    ENTITY_BINDING = "entity_binding"
    RELATION_TRACKING = "relation_tracking"
    QUALIFIER_NEGATION = "qualifier_negation"
    TEMPORAL_ORDER = "temporal_order"
    OPTION_COMPARISON = "option_comparison"
    CONTRADICTION_CHECK = "contradiction_check"
    NUMERIC_SYMBOLIC = "numeric_symbolic"
    COMMONSENSE_CONSISTENCY = "commonsense_consistency"
    FINAL_VERIFICATION = "final_verification"
    OUTPUT_VALIDITY = "output_validity"
    UNKNOWN = "unknown"


BEHAVIOR_CONTEXT_NAMES = tuple(context.value for context in BehaviorContext)
CAPABILITY_RESIDUAL_FAMILY_NAMES = tuple(family.value for family in CapabilityResidualFamily)


def uniform_vote_context_profile() -> Dict[str, float]:
    value = 1.0 / float(len(BEHAVIOR_CONTEXT_NAMES))
    return {context: value for context in BEHAVIOR_CONTEXT_NAMES}


def empty_capability_profile() -> Dict[str, float]:
    return {family: 0.0 for family in CAPABILITY_RESIDUAL_FAMILY_NAMES}


@dataclass
class CapabilityEvidence:
    support: int = 0
    weighted_gain: float = 0.0
    weighted_loss: float = 0.0
    posterior_value: float = 0.0
    last_updated_epoch: int = -1

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CapabilityEvidence":
        return cls(
            support=int(payload.get("support", 0) or 0),
            weighted_gain=float(payload.get("weighted_gain", 0.0) or 0.0),
            weighted_loss=float(payload.get("weighted_loss", 0.0) or 0.0),
            posterior_value=float(payload.get("posterior_value", 0.0) or 0.0),
            last_updated_epoch=int(payload.get("last_updated_epoch", -1) if payload.get("last_updated_epoch") is not None else -1),
        )


@dataclass
class BehaviorFingerprintEntry:
    target_correct: bool
    target_answer_hash: str
    team_vote_correct: bool
    vote_margin_bucket: int
    behavior_context: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BehaviorFingerprintEntry":
        return cls(
            target_correct=bool(payload.get("target_correct", False)),
            target_answer_hash=str(payload.get("target_answer_hash", "")),
            team_vote_correct=bool(payload.get("team_vote_correct", False)),
            vote_margin_bucket=int(payload.get("vote_margin_bucket", 0) or 0),
            behavior_context=str(payload.get("behavior_context", BehaviorContext.INVALID.value)),
        )


@dataclass
class BehaviorStateSummary:
    state_id: str
    epoch: int
    prompt_hash: str
    behavior_fingerprint: Dict[str, BehaviorFingerprintEntry]
    transition_vector: Dict[str, float]
    target_accuracy: float
    team_vote_accuracy: float
    mean_vote_margin: float
    preserved_mechanisms: List[str] = None
    capability_profile: Dict[str, float] = field(default_factory=dict)
    paired_behavior_utility: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["behavior_fingerprint"] = {
            key: asdict(value) if isinstance(value, BehaviorFingerprintEntry) else dict(value)
            for key, value in self.behavior_fingerprint.items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BehaviorStateSummary":
        fingerprint = payload.get("behavior_fingerprint", {})
        return cls(
            state_id=str(payload.get("state_id", "")),
            epoch=int(payload.get("epoch", 0) or 0),
            prompt_hash=str(payload.get("prompt_hash", "")),
            behavior_fingerprint={
                str(key): BehaviorFingerprintEntry.from_dict(value)
                for key, value in fingerprint.items()
                if isinstance(value, dict)
            } if isinstance(fingerprint, dict) else {},
            transition_vector={str(key): float(value) for key, value in dict(payload.get("transition_vector", {})).items()},
            target_accuracy=float(payload.get("target_accuracy", 0.0) or 0.0),
            team_vote_accuracy=float(payload.get("team_vote_accuracy", 0.0) or 0.0),
            mean_vote_margin=float(payload.get("mean_vote_margin", 0.0) or 0.0),
            preserved_mechanisms=[str(value) for value in payload.get("preserved_mechanisms", [])] if isinstance(payload.get("preserved_mechanisms", []), list) else [],
            capability_profile={str(key): float(value) for key, value in dict(payload.get("capability_profile", {})).items()},
            paired_behavior_utility={str(key): float(value) for key, value in dict(payload.get("paired_behavior_utility", {})).items()},
        )


@dataclass
class RejectedBehaviorSummary:
    state_id: str
    epoch: int
    prompt_hash: str
    parent_prompt_hash: str
    rejection_reason: str
    prompt_change_ratio: float
    max_behavior_cycle_similarity: float
    behavior_cycle_overlap: int
    transition_vector: Dict[str, float]
    behavior_fingerprint: Dict[str, BehaviorFingerprintEntry] = field(default_factory=dict)
    paired_behavior_utility: Dict[str, float] = field(default_factory=dict)
    failure_signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["behavior_fingerprint"] = {
            key: asdict(value) if isinstance(value, BehaviorFingerprintEntry) else dict(value)
            for key, value in self.behavior_fingerprint.items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RejectedBehaviorSummary":
        return cls(
            state_id=str(payload.get("state_id", "")),
            epoch=int(payload.get("epoch", 0) or 0),
            prompt_hash=str(payload.get("prompt_hash", "")),
            parent_prompt_hash=str(payload.get("parent_prompt_hash", "")),
            rejection_reason=str(payload.get("rejection_reason", "")),
            prompt_change_ratio=float(payload.get("prompt_change_ratio", 0.0) or 0.0),
            max_behavior_cycle_similarity=float(payload.get("max_behavior_cycle_similarity", 0.0) or 0.0),
            behavior_cycle_overlap=int(payload.get("behavior_cycle_overlap", 0) or 0),
            transition_vector={str(key): float(value) for key, value in dict(payload.get("transition_vector", {})).items()},
            behavior_fingerprint={
                str(key): BehaviorFingerprintEntry.from_dict(value)
                for key, value in dict(payload.get("behavior_fingerprint", {})).items()
                if isinstance(value, dict)
            },
            paired_behavior_utility={str(key): float(value) for key, value in dict(payload.get("paired_behavior_utility", {})).items()},
            failure_signature=str(payload.get("failure_signature", "")),
        )


class AgentState:
    def __init__(self, initial_prompt: str, homogeneity_window: int = 50):
        self.initial_prompt = initial_prompt
        self.current_prompt = initial_prompt
        self.history = [initial_prompt]
        self.prompt_beam: List[Dict[str, Any]] = [
            {
                "id": "",
                "prompt": initial_prompt,
                "score": None,
                "metrics": {},
                "parent_id": None,
                "generation": 0,
            }
        ]
        self.homogeneity_window = max(1, int(homogeneity_window))
        self.recent_homogeneity_flags = deque(maxlen=self.homogeneity_window)
        self.homogeneity_count = 0
        self.accept_count = 0
        self.reject_count = 0
        self.last_update_record: Dict[str, Any] = {}
        self.vote_context_profile = uniform_vote_context_profile()
        self.capability_profile = empty_capability_profile()
        self.capability_evidence = {
            family: CapabilityEvidence() for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
        }
        self.pending_capability_evidence: List[Dict[str, Any]] = []
        self.pending_capability_update_count = 0
        self.capability_profile_update_count = 0
        self.accepted_behavior_archive: List[BehaviorStateSummary] = []
        self.rejected_behavior_archive: List[RejectedBehaviorSummary] = []
        self.cycle_reject_count = 0
        self.large_shift_reject_count = 0
        self.duplicate_prompt_reject_count = 0
        self.last_accepted_prompt_hash = None
        from .lineage import empty_lineage_state
        self.lineage_state = empty_lineage_state()
        self.safe_qd_archive: List[Dict[str, Any]] = []
        self.prompt_memory: List[Dict[str, Any]] = []
        self.probation_archive: List[Dict[str, Any]] = []
        self.per_niche_parent_count: Dict[str, int] = {}
        self.probation_parent_count = 0
        self.optimizer_update_count_by_epoch: Dict[str, int] = {}

    def observe_homogeneity_result(self, homogeneous_flag: int):
        flag = 1 if int(homogeneous_flag) > 0 else 0
        self.recent_homogeneity_flags.append(flag)
        self.homogeneity_count = int(sum(self.recent_homogeneity_flags))

    def trajectory_state_dict(self) -> Dict[str, Any]:
        return {
            "vote_context_profile": dict(self.vote_context_profile),
            "capability_profile": dict(self.capability_profile),
            "capability_evidence": {
                family: asdict(evidence) for family, evidence in self.capability_evidence.items()
            },
            "pending_capability_evidence": list(self.pending_capability_evidence),
            "pending_capability_update_count": int(self.pending_capability_update_count),
            "capability_profile_update_count": int(self.capability_profile_update_count),
            "accepted_behavior_archive": [item.to_dict() for item in self.accepted_behavior_archive],
            "rejected_behavior_archive": [item.to_dict() for item in self.rejected_behavior_archive],
            "cycle_reject_count": int(self.cycle_reject_count),
            "large_shift_reject_count": int(self.large_shift_reject_count),
            "duplicate_prompt_reject_count": int(self.duplicate_prompt_reject_count),
            "last_accepted_prompt_hash": self.last_accepted_prompt_hash,
            "lineage_state": dict(self.lineage_state),
            "safe_qd_archive": list(self.safe_qd_archive),
            "prompt_memory": list(self.prompt_memory),
            "probation_archive": list(self.probation_archive),
            "per_niche_parent_count": dict(self.per_niche_parent_count),
            "probation_parent_count": int(self.probation_parent_count),
            "optimizer_update_count_by_epoch": dict(self.optimizer_update_count_by_epoch),
        }

    def restore_trajectory_state(self, payload: Dict[str, Any]) -> None:
        vote_profile = payload.get("vote_context_profile", {})
        self.vote_context_profile = {
            context: float(vote_profile.get(context, 0.0)) for context in BEHAVIOR_CONTEXT_NAMES
        } if isinstance(vote_profile, dict) else uniform_vote_context_profile()
        vote_total = sum(max(0.0, value) for value in self.vote_context_profile.values())
        self.vote_context_profile = (
            {key: max(0.0, value) / vote_total for key, value in self.vote_context_profile.items()}
            if vote_total > 0.0 else uniform_vote_context_profile()
        )
        capability_profile = payload.get("capability_profile", {})
        self.capability_profile = {
            family: max(0.0, float(capability_profile.get(family, 0.0)))
            for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
        } if isinstance(capability_profile, dict) else empty_capability_profile()
        evidence_payload = payload.get("capability_evidence", {})
        self.capability_evidence = {
            family: CapabilityEvidence.from_dict(evidence_payload.get(family, {}))
            for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
        } if isinstance(evidence_payload, dict) else {
            family: CapabilityEvidence() for family in CAPABILITY_RESIDUAL_FAMILY_NAMES
        }
        pending = payload.get("pending_capability_evidence", [])
        self.pending_capability_evidence = [dict(item) for item in pending if isinstance(item, dict)] if isinstance(pending, list) else []
        self.pending_capability_update_count = int(payload.get("pending_capability_update_count", 0) or 0)
        self.capability_profile_update_count = int(payload.get("capability_profile_update_count", 0) or 0)
        accepted = payload.get("accepted_behavior_archive", [])
        rejected = payload.get("rejected_behavior_archive", [])
        self.accepted_behavior_archive = [
            BehaviorStateSummary.from_dict(item) for item in accepted if isinstance(item, dict)
        ] if isinstance(accepted, list) else []
        self.rejected_behavior_archive = [
            RejectedBehaviorSummary.from_dict(item) for item in rejected if isinstance(item, dict)
        ] if isinstance(rejected, list) else []
        self.cycle_reject_count = int(payload.get("cycle_reject_count", 0) or 0)
        self.large_shift_reject_count = int(payload.get("large_shift_reject_count", 0) or 0)
        self.duplicate_prompt_reject_count = int(payload.get("duplicate_prompt_reject_count", 0) or 0)
        value = payload.get("last_accepted_prompt_hash")
        self.last_accepted_prompt_hash = str(value) if value else None
        from .lineage import empty_lineage_state
        self.lineage_state = {**empty_lineage_state(), **dict(payload.get("lineage_state", {}) or {})}
        self.safe_qd_archive = [dict(item) for item in payload.get("safe_qd_archive", []) if isinstance(item, dict)]
        self.prompt_memory = [dict(item) for item in payload.get("prompt_memory", []) if isinstance(item, dict)]
        self.probation_archive = [dict(item) for item in payload.get("probation_archive", []) if isinstance(item, dict)]
        self.per_niche_parent_count = {str(key): int(value) for key, value in dict(payload.get("per_niche_parent_count", {}) or {}).items()}
        self.probation_parent_count = int(payload.get("probation_parent_count", 0) or 0)
        self.optimizer_update_count_by_epoch = {str(key): int(value) for key, value in dict(payload.get("optimizer_update_count_by_epoch", {}) or {}).items()}
