from typing import Any, Dict, List

from .behavior_profiles import behavior_distance
from .mechanisms import levenshtein_sequence_distance, mechanism_distance


def empty_lineage_state() -> Dict[str, Any]:
    return {
        "lineage_status": "uncommitted",
        "lineage_anchor_prompt_hash": "",
        "lineage_anchor_prompt": "",
        "lineage_anchor_mechanism_signature": [],
        "lineage_anchor_mechanism_embedding": [],
        "lineage_anchor_correctness_vector": [],
        "lineage_anchor_rescue_vector": [],
        "lineage_anchor_accuracy": -1.0,
        "lineage_anchor_epoch": -1,
        "lineage_stability_count": 0,
        "pending_lineage_signature": [],
        "pending_lineage_count": 0,
        "lineage_commit_count": 0,
        "lineage_switch_attempt_count": 0,
        "lineage_switch_commit_count": 0,
        "lineage_switch_cancel_count": 0,
        "last_lineage_drift": 0.0,
    }


def _same_lineage(left: List[str], right: List[str]) -> bool:
    if not left or not right:
        return left == right
    return left[0] == right[0] and levenshtein_sequence_distance(left, right) <= 0.25


def lineage_drift(candidate: Dict[str, Any], state: Dict[str, Any], config: Any) -> Dict[str, float]:
    if state.get("lineage_status") != "committed":
        return {"anchor_mechanism_drift": 0.0, "anchor_behavior_drift": 0.0, "lineage_drift": 0.0, "lineage_drift_penalty": 0.0}
    anchor_mechanism = {
        "normalized_operation_sequence": state.get("lineage_anchor_mechanism_signature", []),
        "mechanism_embedding": state.get("lineage_anchor_mechanism_embedding", []),
    }
    anchor_behavior = {
        "correctness_vector": state.get("lineage_anchor_correctness_vector", []),
        "error_vector": [1 - int(value) for value in state.get("lineage_anchor_correctness_vector", [])],
        "rescue_vector": state.get("lineage_anchor_rescue_vector", []),
    }
    mechanism = mechanism_distance(candidate.get("mechanism_representation", {}), anchor_mechanism,
        sequence_weight=config.mechanism_sequence_distance_weight,
        embedding_weight=config.mechanism_embedding_distance_weight)["mechanism_distance"]
    behavior = behavior_distance(candidate.get("behavior_profile", {}), anchor_behavior,
        correct_set_weight=config.behavior_correct_set_weight,
        rescue_weight=config.behavior_rescue_weight,
        shared_wrong_weight=config.behavior_error_overlap_weight,
        wrong_answer_dispersion_weight=config.behavior_wrong_answer_dispersion_weight,
        support_shrinkage=config.behavior_support_shrinkage)["behavior_distance"]
    drift = config.lineage_mechanism_drift_weight * mechanism + config.lineage_behavior_drift_weight * behavior
    return {
        "anchor_mechanism_drift": mechanism,
        "anchor_behavior_drift": behavior,
        "lineage_drift": drift,
        "lineage_drift_penalty": max(0.0, drift - config.lineage_soft_drift_threshold),
    }


def update_lineage_state(state: Dict[str, Any], selected: Dict[str, Any], *, epoch: int, quality_gate_passed: bool, config: Any) -> Dict[str, Any]:
    state = {**empty_lineage_state(), **dict(state)}
    old_status = state["lineage_status"]
    signature = list(selected.get("mechanism_representation", {}).get("normalized_operation_sequence", []))
    same = _same_lineage(signature, list(state.get("pending_lineage_signature", [])))
    fold_quality_passed = bool(selected.get("fold_quality_gate_passed", True))
    fold_stable = (
        bool(selected.get("fold_behavior_stable", True))
    )
    if not quality_gate_passed or not fold_quality_passed or not fold_stable:
        state["pending_lineage_signature"] = []
        state["pending_lineage_count"] = 0
        return {
            **state,
            "old_status": old_status,
            "new_status": state["lineage_status"],
            "reason": "unstable_single_fold_specialization" if not fold_stable else "quality_gate_failed",
        }
    if old_status != "committed":
        state["pending_lineage_signature"] = signature
        state["pending_lineage_count"] = int(state.get("pending_lineage_count", 0)) + 1 if same else 1
        state["lineage_stability_count"] = state["pending_lineage_count"]
        if old_status == "uncommitted" and state["lineage_stability_count"] >= int(getattr(config, "lineage_commit_required_snapshots", config.lineage_provisional_epochs)) - 1:
            state["lineage_status"] = "provisional"
            _set_anchor(state, selected, epoch)
        if state["lineage_stability_count"] >= int(getattr(config, "lineage_commit_required_snapshots", config.lineage_commit_epochs)):
            anchor_behavior = {
                "correctness_vector": state.get("lineage_anchor_correctness_vector", []),
                "error_vector": [1 - int(value) for value in state.get("lineage_anchor_correctness_vector", [])],
                "rescue_vector": state.get("lineage_anchor_rescue_vector", []),
            }
            selected_behavior = selected.get("behavior_profile", {})
            behavior_stable = behavior_distance(
                selected_behavior,
                anchor_behavior,
                correct_set_weight=config.behavior_correct_set_weight,
                rescue_weight=config.behavior_rescue_weight,
                shared_wrong_weight=config.behavior_error_overlap_weight,
                wrong_answer_dispersion_weight=config.behavior_wrong_answer_dispersion_weight,
                support_shrinkage=config.behavior_support_shrinkage,
            )["behavior_distance"] <= config.lineage_soft_drift_threshold
            rescue_preserved = not sum(anchor_behavior["rescue_vector"]) or bool(sum(selected_behavior.get("rescue_vector", [])))
            if behavior_stable and rescue_preserved:
                state["lineage_status"] = "committed"
                state["lineage_commit_count"] += int(old_status != "committed")
                _set_anchor(state, selected, epoch)
        return {**state, "old_status": old_status, "new_status": state["lineage_status"], "reason": "stable_lineage_observed"}
    anchor_signature = list(state.get("lineage_anchor_mechanism_signature", []))
    if _same_lineage(signature, anchor_signature):
        if state.get("pending_lineage_count", 0):
            state["lineage_switch_cancel_count"] += 1
        state["pending_lineage_signature"] = []
        state["pending_lineage_count"] = 0
        state["lineage_stability_count"] += 1
        return {**state, "old_status": old_status, "new_status": old_status, "reason": "anchor_retained"}
    state["lineage_switch_attempt_count"] += int(not same)
    state["pending_lineage_signature"] = signature
    state["pending_lineage_count"] = int(state.get("pending_lineage_count", 0)) + 1 if same else 1
    if state["pending_lineage_count"] >= int(getattr(config, "lineage_switch_confirmation_snapshots", config.lineage_switch_confirmation_epochs)):
        _set_anchor(state, selected, epoch)
        state["lineage_switch_commit_count"] += 1
        state["pending_lineage_signature"] = []
        state["pending_lineage_count"] = 0
        reason = "lineage_switch_committed"
    else:
        reason = "lineage_switch_pending"
    return {**state, "old_status": old_status, "new_status": state["lineage_status"], "reason": reason}


def _set_anchor(state: Dict[str, Any], selected: Dict[str, Any], epoch: int) -> None:
    representation = selected.get("mechanism_representation", {})
    behavior = selected.get("behavior_profile", {})
    state.update({
        "lineage_anchor_prompt_hash": selected.get("prompt_hash", ""),
        "lineage_anchor_prompt": selected.get("prompt", ""),
        "lineage_anchor_mechanism_signature": list(representation.get("normalized_operation_sequence", [])),
        "lineage_anchor_mechanism_embedding": list(representation.get("mechanism_embedding", [])),
        "lineage_anchor_correctness_vector": list(behavior.get("correctness_vector", [])),
        "lineage_anchor_rescue_vector": list(behavior.get("rescue_vector", [])),
        "lineage_anchor_accuracy": float(behavior.get("accuracy", 0.0)),
        "lineage_anchor_epoch": int(epoch),
    })
