"""Accuracy-first sequential state optimization used by V9."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any, Dict, Mapping, Sequence

import numpy as np


STATE_NAMES = ("C0", "C1", "C2", "C3", "C4", "C5")
PROMPT_MEMORY_SLOTS = (
    "active",
    "accuracy_best",
    "state_vote_best",
    "safe_diversity_parent",
    "recent_safe_parent",
    "rollback_or_recent_success",
    "quality_fill",
)


def question_state(gold_vote_count: Any) -> str:
    return f"C{max(0, min(5, int(gold_vote_count or 0)))}"


def state_histogram(gold_vote_counts: Sequence[Any]) -> Dict[str, Any]:
    counts = {name.lower() + "_count": 0 for name in STATE_NAMES}
    for value in gold_vote_counts:
        counts[question_state(value).lower() + "_count"] += 1
    total = len(gold_vote_counts)
    rates = {
        key.replace("_count", "_rate"): (float(value) / total if total else 0.0)
        for key, value in counts.items()
    }
    return {
        **counts,
        **rates,
        "c3plus_count": counts["c3_count"] + counts["c4_count"] + counts["c5_count"],
    }


def state_potentials(config: Any) -> Dict[int, float]:
    return {
        index: float(getattr(config, f"state_potential_c{index}"))
        for index in range(6)
    }


def state_vote_reward(
    active_correctness: Sequence[Sequence[Any]],
    candidate_correctness: Sequence[Sequence[Any]],
    active_vote_correct: Sequence[Any],
    candidate_vote_correct: Sequence[Any],
    config: Any,
) -> Dict[str, Any]:
    if len(active_correctness) != len(candidate_correctness):
        raise ValueError("active and candidate probe rows must align")
    potentials = state_potentials(config)
    active_g = [sum(bool(value) for value in row) for row in active_correctness]
    candidate_g = [sum(bool(value) for value in row) for row in candidate_correctness]
    distribution = float(np.mean([
        potentials[max(0, min(5, after))] - potentials[max(0, min(5, before))]
        for before, after in zip(active_g, candidate_g)
    ])) if active_g else 0.0
    vote_delta = (
        float(np.mean([bool(value) for value in candidate_vote_correct]))
        - float(np.mean([bool(value) for value in active_vote_correct]))
    ) if active_vote_correct else 0.0
    active_agent_acc = np.mean(np.asarray(active_correctness, dtype=float), axis=0).tolist() if active_correctness else []
    candidate_agent_acc = np.mean(np.asarray(candidate_correctness, dtype=float), axis=0).tolist() if candidate_correctness else []
    active_bottom2 = float(np.mean(sorted(active_agent_acc)[:2])) if active_agent_acc else 0.0
    candidate_bottom2 = float(np.mean(sorted(candidate_agent_acc)[:2])) if candidate_agent_acc else 0.0
    bottom2_delta = candidate_bottom2 - active_bottom2
    distribution_component = distribution if bool(getattr(config, "state_distribution_reward_enabled", True)) else 0.0
    vote_component = (
        float(getattr(config, "state_reward_vote_weight", 2.0)) * vote_delta
        if bool(getattr(config, "state_vote_reward_enabled", True)) else 0.0
    )
    bottom2_component = (
        float(getattr(config, "state_reward_bottom2_weight", 0.25)) * bottom2_delta
        if bool(getattr(config, "state_bottom2_reward_enabled", False)) else 0.0
    )
    transitions = {
        f"c{before}_to_c{after}_count": sum(
            int(left == before and right == after)
            for left, right in zip(active_g, candidate_g)
        )
        for before, after in [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (1, 0), (2, 1), (3, 2), (4, 3), (5, 4),
        ]
    }
    return {
        "state_reward_total": distribution_component + vote_component + bottom2_component,
        "state_vote_reward": distribution_component + vote_component + bottom2_component,
        "state_reward_distribution_component": distribution_component,
        "state_reward_vote_component": vote_component,
        "state_reward_bottom2_component": bottom2_component,
        "vote_accuracy_delta": vote_delta,
        "vote_gain_count": sum(int(not bool(a) and bool(b)) for a, b in zip(active_vote_correct, candidate_vote_correct)),
        "vote_loss_count": sum(int(bool(a) and not bool(b)) for a, b in zip(active_vote_correct, candidate_vote_correct)),
        "diagnostic_raw_vote_accuracy_delta": vote_delta,
        "active_bottom2_accuracy": active_bottom2,
        "candidate_bottom2_accuracy": candidate_bottom2,
        "active_gold_vote_counts": active_g,
        "candidate_gold_vote_counts": candidate_g,
        **transitions,
        **{f"active_{key}": value for key, value in state_histogram(active_g).items()},
        **{f"candidate_{key}": value for key, value in state_histogram(candidate_g).items()},
    }


def _jaccard_distance(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 - (len(left & right) / len(union)) if union else 0.0


def correct_set_diversity(profiles: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    correct_sets = [
        {
            index
            for index, value in enumerate(profile.get("correctness_vector", []))
            if bool(value)
            and not bool(
                profile.get("invalid_vector", [])[index]
                if index < len(profile.get("invalid_vector", []))
                else 1
            )
        }
        for profile in profiles
    ]
    distances = [_jaccard_distance(left, right) for left, right in combinations(correct_sets, 2)]
    return {
        "correct_set_diversity_mean": float(np.mean(distances)) if distances else 0.0,
        "correct_set_diversity_min": min(distances, default=0.0),
    }


def sequential_team_metrics(
    profiles: Sequence[Mapping[str, Any]],
    gold_answers: Sequence[str],
    question_hashes: Sequence[str],
    target_agent_id: int,
    config: Any,
    *,
    vote_fn: Any,
    match_fn: Any,
) -> Dict[str, Any]:
    row_count = len(gold_answers)
    correctness_by_agent = [list(profile.get("correctness_vector", [])) for profile in profiles]
    correctness_rows = [
        [int(vector[index]) if index < len(vector) else 0 for vector in correctness_by_agent]
        for index in range(row_count)
    ]
    vote_correct = []
    for index, gold in enumerate(gold_answers):
        answers = [
            str(profile.get("answer_vector", [])[index])
            if index < len(profile.get("answer_vector", [])) else ""
            for profile in profiles
        ]
        vote = vote_fn(
            answers,
            tie_break_method=config.vote_tie_break,
            seed=config.seed,
            question_hash=str(question_hashes[index]),
        )
        vote_correct.append(int(match_fn(str(vote.get("vote_answer", "")), str(gold))))
    target = profiles[target_agent_id]
    target_correctness = [int(bool(value)) for value in target.get("correctness_vector", [])]
    target_invalid = [int(bool(value)) for value in target.get("invalid_vector", [])]
    return {
        "full_probe_size": row_count,
        "candidate_target_correct_count": sum(target_correctness),
        "candidate_target_accuracy": sum(target_correctness) / max(1, row_count),
        "candidate_invalid_count": sum(target_invalid),
        "candidate_invalid_rate": sum(target_invalid) / max(1, row_count),
        "vote_correct_vector": vote_correct,
        "plurality_vote_accuracy": sum(vote_correct) / max(1, row_count),
        "correctness_rows": correctness_rows,
        **state_histogram([sum(row) for row in correctness_rows]),
        **correct_set_diversity(profiles),
        **safe_trace_diversity_c4c5(profiles, target_agent_id, config),
    }


def _cosine_distance(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        return None
    return max(0.0, min(2.0, 1.0 - float(np.dot(a, b) / denominator)))


def safe_trace_diversity_c4c5(
    profiles: Sequence[Mapping[str, Any]], target_agent_id: int, config: Any
) -> Dict[str, Any]:
    if not profiles or target_agent_id < 0 or target_agent_id >= len(profiles):
        return {"safe_trace_diversity_c4c5": 0.0, "safe_trace_pair_count": 0, "safe_trace_constraint_available": False}
    row_count = min((len(profile.get("correctness_vector", [])) for profile in profiles), default=0)
    weighted_distances = []
    weights = []
    for question_index in range(row_count):
        g = sum(bool(profile.get("correctness_vector", [])[question_index]) for profile in profiles)
        if g not in {4, 5}:
            continue
        target = profiles[target_agent_id]
        if not bool(target.get("correctness_vector", [])[question_index]) or bool(target.get("invalid_vector", [1] * row_count)[question_index]):
            continue
        target_embeddings = target.get("trace_embedding_vector_per_question", [])
        if question_index >= len(target_embeddings):
            continue
        for peer_id, peer in enumerate(profiles):
            if peer_id == target_agent_id:
                continue
            if not bool(peer.get("correctness_vector", [])[question_index]) or bool(peer.get("invalid_vector", [1] * row_count)[question_index]):
                continue
            peer_embeddings = peer.get("trace_embedding_vector_per_question", [])
            if question_index >= len(peer_embeddings):
                continue
            distance = _cosine_distance(target_embeddings[question_index], peer_embeddings[question_index])
            if distance is None:
                continue
            weight = float(getattr(config, f"state_safe_trace_weight_c{g}", 1.0))
            weighted_distances.append(distance * weight)
            weights.append(weight)
    return {
        "safe_trace_diversity_c4c5": (sum(weighted_distances) / sum(weights)) if weights else 0.0,
        "safe_trace_pair_count": len(weights),
        "safe_trace_constraint_available": bool(weights),
    }


def paired_safe_trace_diversity_c4c5(
    active_profiles: Sequence[Mapping[str, Any]],
    candidate_profiles: Sequence[Mapping[str, Any]],
    target_agent_id: int,
    config: Any,
) -> Dict[str, Any]:
    if (
        not active_profiles
        or len(active_profiles) != len(candidate_profiles)
        or target_agent_id < 0
        or target_agent_id >= len(active_profiles)
    ):
        return {
            "active_paired_safe_trace_diversity": 0.0,
            "candidate_paired_safe_trace_diversity": 0.0,
            "paired_safe_trace_delta": 0.0,
            "paired_safe_trace_pair_count": 0,
            "paired_safe_trace_constraint_available": False,
        }
    row_count = min(
        min((len(profile.get("correctness_vector", [])) for profile in active_profiles), default=0),
        min((len(profile.get("correctness_vector", [])) for profile in candidate_profiles), default=0),
    )
    active_weighted = []
    candidate_weighted = []
    weights = []
    active_target = active_profiles[target_agent_id]
    candidate_target = candidate_profiles[target_agent_id]
    for question_index in range(row_count):
        active_g = sum(bool(profile.get("correctness_vector", [])[question_index]) for profile in active_profiles)
        candidate_g = sum(bool(profile.get("correctness_vector", [])[question_index]) for profile in candidate_profiles)
        if active_g not in {4, 5} or candidate_g not in {4, 5}:
            continue
        if not bool(active_target.get("correctness_vector", [])[question_index]):
            continue
        if not bool(candidate_target.get("correctness_vector", [])[question_index]):
            continue
        if bool(active_target.get("invalid_vector", [1] * row_count)[question_index]):
            continue
        if bool(candidate_target.get("invalid_vector", [1] * row_count)[question_index]):
            continue
        active_target_embeddings = active_target.get("trace_embedding_vector_per_question", [])
        candidate_target_embeddings = candidate_target.get("trace_embedding_vector_per_question", [])
        if question_index >= len(active_target_embeddings) or question_index >= len(candidate_target_embeddings):
            continue
        for peer_id, (active_peer, candidate_peer) in enumerate(zip(active_profiles, candidate_profiles)):
            if peer_id == target_agent_id:
                continue
            if not bool(active_peer.get("correctness_vector", [])[question_index]):
                continue
            if not bool(candidate_peer.get("correctness_vector", [])[question_index]):
                continue
            if bool(active_peer.get("invalid_vector", [1] * row_count)[question_index]):
                continue
            if bool(candidate_peer.get("invalid_vector", [1] * row_count)[question_index]):
                continue
            active_peer_embeddings = active_peer.get("trace_embedding_vector_per_question", [])
            candidate_peer_embeddings = candidate_peer.get("trace_embedding_vector_per_question", [])
            if question_index >= len(active_peer_embeddings) or question_index >= len(candidate_peer_embeddings):
                continue
            active_distance = _cosine_distance(
                active_target_embeddings[question_index], active_peer_embeddings[question_index]
            )
            candidate_distance = _cosine_distance(
                candidate_target_embeddings[question_index], candidate_peer_embeddings[question_index]
            )
            if active_distance is None or candidate_distance is None:
                continue
            weight = float(getattr(config, f"state_safe_trace_weight_c{active_g}", 1.0))
            active_weighted.append(active_distance * weight)
            candidate_weighted.append(candidate_distance * weight)
            weights.append(weight)
    active_mean = sum(active_weighted) / sum(weights) if weights else 0.0
    candidate_mean = sum(candidate_weighted) / sum(weights) if weights else 0.0
    return {
        "active_paired_safe_trace_diversity": active_mean,
        "candidate_paired_safe_trace_diversity": candidate_mean,
        "paired_safe_trace_delta": candidate_mean - active_mean,
        "paired_safe_trace_pair_count": len(weights),
        "paired_safe_trace_constraint_available": bool(weights),
    }


def outcome_signature(
    profile: Mapping[str, Any],
    version: str,
    probe_hash: str = "",
    question_hashes: Sequence[str] = (),
) -> str:
    payload = {
        "version": version,
        "probe_hash": str(probe_hash or profile.get("fixed_probe_hash", "")),
        "question_hashes": [
            str(value) for value in (question_hashes or profile.get("question_hashes", []))
        ],
        "correctness": [int(bool(value)) for value in profile.get("correctness_vector", [])],
        "invalid": [int(bool(value)) for value in profile.get("invalid_vector", [])],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _quantized_normalized_embedding(vector: Sequence[Any]) -> list[float]:
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        return []
    return [round(float(component), 3) for component in (values / norm)]


def safe_trace_signature(
    profiles: Sequence[Mapping[str, Any]],
    target_agent_id: int,
    version: str,
    probe_hash: str = "",
    question_hashes: Sequence[str] = (),
) -> str:
    signature_question_hashes = list(
        question_hashes or (profiles[0].get("question_hashes", []) if profiles else [])
    )
    if not profiles or target_agent_id < 0 or target_agent_id >= len(profiles):
        pairs = []
    else:
        target = profiles[target_agent_id]
        row_count = min((len(profile.get("correctness_vector", [])) for profile in profiles), default=0)
        pairs = []
        for question_index in range(row_count):
            state = sum(bool(profile.get("correctness_vector", [])[question_index]) for profile in profiles)
            if state not in {4, 5}:
                continue
            if not bool(target.get("correctness_vector", [])[question_index]):
                continue
            if bool(target.get("invalid_vector", [1] * row_count)[question_index]):
                continue
            target_embeddings = target.get("trace_embedding_vector_per_question", [])
            if question_index >= len(target_embeddings):
                continue
            target_embedding = _quantized_normalized_embedding(target_embeddings[question_index])
            if not target_embedding:
                continue
            for peer_id, peer in enumerate(profiles):
                if peer_id == target_agent_id:
                    continue
                if not bool(peer.get("correctness_vector", [])[question_index]):
                    continue
                if bool(peer.get("invalid_vector", [1] * row_count)[question_index]):
                    continue
                peer_embeddings = peer.get("trace_embedding_vector_per_question", [])
                if question_index >= len(peer_embeddings):
                    continue
                peer_embedding = _quantized_normalized_embedding(peer_embeddings[question_index])
                if not peer_embedding:
                    continue
                pairs.append({
                    "question_hash": str(
                        signature_question_hashes[question_index]
                        if question_index < len(signature_question_hashes)
                        else question_index
                    ),
                    "state": f"C{state}",
                    "target_agent_id": int(target_agent_id),
                    "peer_agent_id": int(peer_id),
                    "target_embedding": target_embedding,
                    "peer_embedding": peer_embedding,
                })
    payload = {
        "version": version,
        "probe_hash": str(probe_hash or (profiles[0].get("fixed_probe_hash", "") if profiles else "")),
        "question_hashes": [str(value) for value in signature_question_hashes],
        "pairs": pairs,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def full_probe_constraints(
    metrics: Mapping[str, Any], active_metrics: Mapping[str, Any], initial_metrics: Mapping[str, Any], config: Any
) -> Dict[str, Any]:
    candidate_correct = int(metrics.get("candidate_target_correct_count", 0) or 0)
    active_correct = int(active_metrics.get("candidate_target_correct_count", active_metrics.get("target_correct_count", 0)) or 0)
    initial_correct = int(initial_metrics.get("candidate_target_correct_count", initial_metrics.get("target_correct_count", 0)) or 0)
    probe_size = max(1, int(metrics.get("full_probe_size", 1) or 1))
    local_allowed = max(
        int(getattr(config, "state_accuracy_band_allowed_loss_questions", 0)),
        int(float(getattr(config, "state_local_accuracy_loss_epsilon", 0.0)) * probe_size + 1e-12),
    )
    global_allowed = max(
        int(getattr(config, "state_accuracy_band_allowed_loss_questions", 0)),
        int(float(getattr(config, "state_global_accuracy_loss_epsilon", 0.0)) * probe_size + 1e-12),
    )
    local_loss = max(0, active_correct - candidate_correct)
    global_loss = max(0, initial_correct - candidate_correct)
    invalid_passed = int(metrics.get("candidate_invalid_count", probe_size) or 0) <= (
        int(active_metrics.get("candidate_invalid_count", active_metrics.get("invalid_count", probe_size)) or 0)
        + int(float(getattr(config, "invalid_guard_epsilon", 0.0)) * probe_size + 1e-12)
    )
    diversity_enabled = bool(getattr(config, "state_diversity_constraints_enabled", True))
    mean_floor = max(
        float(active_metrics.get("correct_set_diversity_mean", 0.0)) - float(getattr(config, "state_correct_set_diversity_local_epsilon", 0.03)),
        float(initial_metrics.get("correct_set_diversity_mean", 0.0)) - float(getattr(config, "state_correct_set_diversity_global_epsilon", 0.05)),
    )
    min_floor = max(
        float(active_metrics.get("correct_set_diversity_min", 0.0)) - float(getattr(config, "state_min_pairwise_diversity_local_epsilon", 0.05)),
        float(initial_metrics.get("correct_set_diversity_min", 0.0)) - float(getattr(config, "state_min_pairwise_diversity_global_epsilon", 0.08)),
    )
    correct_mean_slack = float(metrics.get("correct_set_diversity_mean", 0.0)) - mean_floor
    correct_min_slack = float(metrics.get("correct_set_diversity_min", 0.0)) - min_floor
    safe_available = bool(metrics.get("paired_safe_trace_constraint_available", False))
    safe_floor = (
        float(metrics.get("active_paired_safe_trace_diversity", 0.0))
        - float(getattr(config, "state_safe_trace_local_epsilon", 0.05))
    )
    safe_slack = float(metrics.get("candidate_paired_safe_trace_diversity", 0.0)) - safe_floor
    safe_passed = (
        (not diversity_enabled)
        or (not safe_available)
        or safe_slack >= -1e-12
    )
    correct_passed = (not diversity_enabled) or (correct_mean_slack >= -1e-12 and correct_min_slack >= -1e-12)
    catastrophic_limit = int(getattr(config, "state_catastrophic_vote_loss_limit", -1))
    vote_loss_passed = catastrophic_limit < 0 or int(metrics.get("vote_loss_count", 0) or 0) <= catastrophic_limit
    passed = local_loss <= local_allowed and global_loss <= global_allowed and invalid_passed and correct_passed and safe_passed and vote_loss_passed
    slacks = [correct_mean_slack, correct_min_slack] + ([safe_slack] if safe_available else [])
    binding_tolerance = float(getattr(config, "state_diversity_binding_tolerance", 0.01))
    accuracy_invalid_passed = local_loss <= local_allowed and global_loss <= global_allowed and invalid_passed
    correct_rejected = bool(diversity_enabled and accuracy_invalid_passed and not correct_passed)
    safe_rejected = bool(diversity_enabled and accuracy_invalid_passed and correct_passed and not safe_passed)
    return {
        "active_target_correct_count": active_correct,
        "candidate_target_correct_count": candidate_correct,
        "initial_target_correct_count": initial_correct,
        "local_accuracy_loss_count": local_loss,
        "global_accuracy_loss_count": global_loss,
        "local_accuracy_constraint_passed": local_loss <= local_allowed,
        "global_accuracy_constraint_passed": global_loss <= global_allowed,
        "invalid_guard_passed": invalid_passed,
        "accuracy_constraint_passed": local_loss <= local_allowed and global_loss <= global_allowed,
        "invalid_constraint_passed": invalid_passed,
        "active_correct_set_diversity_mean": float(active_metrics.get("correct_set_diversity_mean", 0.0)),
        "candidate_correct_set_diversity_mean": float(metrics.get("correct_set_diversity_mean", 0.0)),
        "initial_correct_set_diversity_mean": float(initial_metrics.get("correct_set_diversity_mean", 0.0)),
        "active_min_pairwise_diversity": float(active_metrics.get("correct_set_diversity_min", 0.0)),
        "candidate_min_pairwise_diversity": float(metrics.get("correct_set_diversity_min", 0.0)),
        "initial_min_pairwise_diversity": float(initial_metrics.get("correct_set_diversity_min", 0.0)),
        "active_safe_trace_diversity_c4c5": float(active_metrics.get("safe_trace_diversity_c4c5", 0.0)),
        "candidate_safe_trace_diversity_c4c5": float(metrics.get("safe_trace_diversity_c4c5", 0.0)),
        "initial_safe_trace_diversity_c4c5": float(initial_metrics.get("safe_trace_diversity_c4c5", 0.0)),
        "correct_set_diversity_constraint_passed": correct_passed,
        "safe_trace_constraint_passed": safe_passed,
        "safe_trace_constraint_available": safe_available,
        "candidate_safe_trace_constraint_available": safe_available,
        "paired_safe_trace_constraint_available": safe_available,
        "paired_safe_trace_pair_count": int(metrics.get("paired_safe_trace_pair_count", 0) or 0),
        "catastrophic_vote_loss_guard_passed": vote_loss_passed,
        "diversity_constraint_slack": min(slacks, default=0.0),
        "diversity_constraint_evaluated": bool(diversity_enabled and accuracy_invalid_passed),
        "correct_set_constraint_rejected": correct_rejected,
        "safe_trace_constraint_rejected": safe_rejected,
        "correct_set_constraint_binding": bool(
            diversity_enabled
            and accuracy_invalid_passed
            and min(correct_mean_slack, correct_min_slack) <= binding_tolerance
        ),
        "safe_trace_constraint_binding": bool(
            diversity_enabled
            and accuracy_invalid_passed
            and safe_available
            and safe_slack <= binding_tolerance
        ),
        "sequential_constraints_passed": passed,
        "candidate_feasible": passed,
        "rejection_reason": "" if passed else "sequential_full_probe_constraint",
    }


def accuracy_first_key(item: Mapping[str, Any], config: Any) -> tuple:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), Mapping) else {}
    diversity_key_value = (
        float(metrics.get("diversity_constraint_slack", 0.0) or 0.0)
        if bool(getattr(config, "state_diversity_constraints_enabled", False))
        else 0.0
    )
    return (
        int(metrics.get("candidate_target_correct_count", 0) or 0),
        float(metrics.get("candidate_target_accuracy", 0.0) or 0.0),
        float(metrics.get("state_vote_reward", metrics.get("state_reward_total", 0.0)) or 0.0),
        -int(metrics.get("candidate_invalid_count", 0) or 0),
        diversity_key_value,
        -int(item.get("generation", 0) or 0),
        str(item.get("prompt_hash", "")),
    )


def candidate_strictly_beats_incumbent(candidate: Mapping[str, Any], incumbent: Mapping[str, Any], config: Any) -> bool:
    candidate_metrics = candidate.get("metrics", {})
    incumbent_metrics = incumbent.get("metrics", {})
    candidate_correct = int(candidate_metrics.get("candidate_target_correct_count", 0) or 0)
    incumbent_correct = int(incumbent_metrics.get("candidate_target_correct_count", 0) or 0)
    if candidate_correct > incumbent_correct:
        return True
    if candidate_correct < incumbent_correct:
        return False
    gain = float(candidate_metrics.get("state_vote_reward", 0.0) or 0.0) - float(incumbent_metrics.get("state_vote_reward", 0.0) or 0.0)
    if gain < float(getattr(config, "state_min_secondary_reward_gain", 0.0)) - 1e-12:
        return False
    return accuracy_first_key(candidate, config) > accuracy_first_key(incumbent, config)


def epoch_agent_order(epoch: int, num_agents: int = 5) -> list[int]:
    if num_agents <= 0:
        return []
    offset = int(epoch) % int(num_agents)
    return list(range(offset, num_agents)) + list(range(offset))


def rebuild_prompt_memory(
    items: Sequence[Mapping[str, Any]],
    active_prompt_hash: str,
    capacity: int = 5,
    *,
    config: Any,
    previous_active_item: Mapping[str, Any] | None = None,
    return_diagnostics: bool = False,
) -> list[Dict[str, Any]] | tuple[list[Dict[str, Any]], Dict[str, Any]]:
    unique_by_hash: Dict[str, Dict[str, Any]] = {}
    for raw in items:
        item = dict(raw)
        prompt_hash = str(item.get("prompt_hash", ""))
        if prompt_hash:
            unique_by_hash.setdefault(prompt_hash, item)
    values = list(unique_by_hash.values())
    active = next((item for item in values if str(item.get("prompt_hash", "")) == active_prompt_hash), None)
    if active is None:
        raise ValueError("prompt memory requires the active prompt")
    capacity = max(1, int(capacity))
    diversity_enabled = bool(getattr(config, "state_diversity_constraints_enabled", False))
    rollback_hash = ""
    rollback_first = None
    if previous_active_item is not None:
        previous_hash = str(previous_active_item.get("prompt_hash", ""))
        if previous_hash and previous_hash != active_prompt_hash:
            rollback_hash = previous_hash
            rollback_first = unique_by_hash.get(previous_hash, dict(previous_active_item))
    safe_values = [
        item for item in values
        if bool(item.get("metrics", {}).get("accuracy_constraint_passed", True))
        and bool(item.get("metrics", {}).get("invalid_constraint_passed", True))
    ]
    slot_values = [
        item for item in safe_values
        if str(item.get("prompt_hash", "")) != rollback_hash
    ]
    accuracy_ranking = sorted(slot_values, key=lambda item: accuracy_first_key(item, config), reverse=True)
    state_vote_ranking = sorted(slot_values, key=lambda item: (
        float(item.get("metrics", {}).get("state_vote_reward", 0.0) or 0.0),
        *accuracy_first_key(item, config),
    ), reverse=True)
    if diversity_enabled:
        parent_slot = "safe_diversity_parent"
        parent_ranking = sorted(slot_values, key=lambda item: (
            float(item.get("metrics", {}).get("diversity_constraint_slack", 0.0) or 0.0),
            float(item.get("metrics", {}).get("paired_safe_trace_delta", 0.0) or 0.0),
            *accuracy_first_key(item, config),
        ), reverse=True)
    else:
        parent_slot = "recent_safe_parent"
        parent_ranking = sorted(slot_values, key=lambda item: (
            int(item.get("accepted_update_index", -1) or -1),
            *accuracy_first_key(item, config),
        ), reverse=True)
    rollback_ranking = [rollback_first] if rollback_first is not None else []
    rollback_ranking.extend(sorted(
        (item for item in safe_values if str(item.get("prompt_hash", "")) != rollback_hash),
        key=lambda item: (
            int(item.get("accepted_update_index", -1) or -1),
            int(item.get("generation", 0) or 0),
            *accuracy_first_key(item, config),
        ),
        reverse=True,
    ))
    slot_rankings: list[tuple[str, list[Dict[str, Any]]]] = [
        ("active", [active]),
        ("accuracy_best", accuracy_ranking),
        ("state_vote_best", state_vote_ranking),
        (parent_slot, parent_ranking),
        ("rollback_or_recent_success", rollback_ranking),
    ]
    memory = []
    seen_hashes = set()
    seen_outcomes = set()
    seen_safe_traces = set()
    duplicate_skips = 0
    slot_candidate_count = {slot: len(ranking) for slot, ranking in slot_rankings}

    def add_candidate(slot: str, candidate: Mapping[str, Any], *, enforce_outcome: bool) -> bool:
        nonlocal duplicate_skips
        prompt_hash = str(candidate.get("prompt_hash", ""))
        outcome = str(candidate.get("outcome_signature_hash", candidate.get("outcome_signature", "")))
        safe_trace = str(candidate.get("safe_trace_signature_hash", candidate.get("safe_trace_signature", "")))
        if not prompt_hash or prompt_hash in seen_hashes:
            duplicate_skips += 1
            return False
        if enforce_outcome and outcome and outcome in seen_outcomes:
            allows_safe_variant = (
                slot == "safe_diversity_parent"
                and bool(safe_trace)
                and safe_trace not in seen_safe_traces
            )
            if not allows_safe_variant:
                duplicate_skips += 1
                return False
        row = dict(candidate)
        row["prompt_memory_slot"] = slot
        row["memory_slot"] = slot
        memory.append(row)
        seen_hashes.add(prompt_hash)
        if outcome:
            seen_outcomes.add(outcome)
        if safe_trace:
            seen_safe_traces.add(safe_trace)
        return True

    for slot, ranking in slot_rankings:
        if len(memory) >= capacity:
            break
        for candidate in ranking:
            is_explicit_rollback = (
                slot == "rollback_or_recent_success"
                and str(candidate.get("prompt_hash", "")) == rollback_hash
            )
            if add_candidate(
                slot,
                candidate,
                enforce_outcome=slot != "active" and not is_explicit_rollback,
            ):
                break

    quality_fill_count = 0
    for candidate in accuracy_ranking:
        if len(memory) >= capacity:
            break
        if add_candidate("quality_fill", candidate, enforce_outcome=False):
            quality_fill_count += 1
    diagnostics = {
        "memory_capacity": capacity,
        "memory_occupancy": len(memory),
        "memory_underfilled": len(memory) < capacity,
        "memory_underfilled_reason": (
            "insufficient_distinct_safe_prompts" if len(memory) < capacity else ""
        ),
        "slot_candidate_count": slot_candidate_count,
        "slot_duplicate_skip_count": duplicate_skips,
        "quality_fill_count": quality_fill_count,
        "rollback_prompt_hash": rollback_hash,
    }
    return (memory, diagnostics) if return_diagnostics else memory


def select_memory_parents(
    memory: Sequence[Mapping[str, Any]], count: int, rotation_offset: int = 0
) -> tuple[list[Dict[str, Any]], list[str]]:
    order = {slot: index for index, slot in enumerate(PROMPT_MEMORY_SLOTS)}
    ranked = sorted(
        (dict(item) for item in memory),
        key=lambda item: (order.get(str(item.get("prompt_memory_slot", "")), len(order)), str(item.get("prompt_hash", ""))),
    )
    active = [item for item in ranked if str(item.get("prompt_memory_slot", "")) == "active"]
    remaining = [item for item in ranked if str(item.get("prompt_memory_slot", "")) != "active"]
    if remaining:
        offset = int(rotation_offset) % len(remaining)
        remaining = remaining[offset:] + remaining[:offset]
    selected = (active[:1] + remaining)[:max(1, int(count))]
    return selected, [str(item.get("prompt_memory_slot", "memory")) for item in selected]
