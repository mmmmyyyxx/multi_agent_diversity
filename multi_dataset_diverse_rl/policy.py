from collections import deque
from dataclasses import asdict, dataclass
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


BEHAVIOR_CONTEXT_NAMES = tuple(context.value for context in BehaviorContext)


def uniform_specialization_profile() -> Dict[str, float]:
    value = 1.0 / float(len(BEHAVIOR_CONTEXT_NAMES))
    return {context: value for context in BEHAVIOR_CONTEXT_NAMES}


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
    specialization_profile: Dict[str, float]
    target_accuracy: float
    team_vote_accuracy: float
    mean_vote_margin: float
    preserved_mechanisms: List[str] = None

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
            specialization_profile={str(key): float(value) for key, value in dict(payload.get("specialization_profile", {})).items()},
            target_accuracy=float(payload.get("target_accuracy", 0.0) or 0.0),
            team_vote_accuracy=float(payload.get("team_vote_accuracy", 0.0) or 0.0),
            mean_vote_margin=float(payload.get("mean_vote_margin", 0.0) or 0.0),
            preserved_mechanisms=[str(value) for value in payload.get("preserved_mechanisms", [])] if isinstance(payload.get("preserved_mechanisms", []), list) else [],
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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
        self.specialization_profile = uniform_specialization_profile()
        self.specialization_update_count = 0
        self.last_accepted_transition: Dict[str, float] = {}
        self.accepted_behavior_archive: List[BehaviorStateSummary] = []
        self.rejected_behavior_archive: List[RejectedBehaviorSummary] = []
        self.cycle_reject_count = 0
        self.large_shift_reject_count = 0
        self.duplicate_prompt_reject_count = 0
        self.last_accepted_prompt_hash = None

    def observe_homogeneity_result(self, homogeneous_flag: int):
        flag = 1 if int(homogeneous_flag) > 0 else 0
        self.recent_homogeneity_flags.append(flag)
        self.homogeneity_count = int(sum(self.recent_homogeneity_flags))

    def trajectory_state_dict(self) -> Dict[str, Any]:
        return {
            "specialization_profile": dict(self.specialization_profile),
            "specialization_update_count": int(self.specialization_update_count),
            "last_accepted_transition": dict(self.last_accepted_transition),
            "accepted_behavior_archive": [item.to_dict() for item in self.accepted_behavior_archive],
            "rejected_behavior_archive": [item.to_dict() for item in self.rejected_behavior_archive],
            "cycle_reject_count": int(self.cycle_reject_count),
            "large_shift_reject_count": int(self.large_shift_reject_count),
            "duplicate_prompt_reject_count": int(self.duplicate_prompt_reject_count),
            "last_accepted_prompt_hash": self.last_accepted_prompt_hash,
        }

    def restore_trajectory_state(self, payload: Dict[str, Any]) -> None:
        profile = payload.get("specialization_profile", {})
        self.specialization_profile = {
            context: float(profile.get(context, 0.0)) for context in BEHAVIOR_CONTEXT_NAMES
        } if isinstance(profile, dict) else uniform_specialization_profile()
        total = sum(max(0.0, value) for value in self.specialization_profile.values())
        self.specialization_profile = (
            {key: max(0.0, value) / total for key, value in self.specialization_profile.items()}
            if total > 0.0 else uniform_specialization_profile()
        )
        self.specialization_update_count = int(payload.get("specialization_update_count", 0) or 0)
        transition = payload.get("last_accepted_transition", {})
        self.last_accepted_transition = {str(key): float(value) for key, value in dict(transition).items()}
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
