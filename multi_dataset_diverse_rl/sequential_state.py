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
    "rollback_or_recent_success",
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
    # A wrong-to-wrong answer change has no optimization value even if it
    # happens to alter a plurality tie. Vote credit requires a change in G.
    reward_candidate_vote = [
        candidate if before_g != after_g else active
        for active, candidate, before_g, after_g in zip(
            active_vote_correct, candidate_vote_correct, active_g, candidate_g
        )
    ]
    vote_delta = (
        float(np.mean([bool(value) for value in reward_candidate_vote]))
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
    secondary_enabled = bool(getattr(config, "state_distribution_reward_enabled", True)) or bool(
        getattr(config, "state_vote_reward_enabled", True)
    )
    bottom2_component = (
        float(getattr(config, "state_reward_bottom2_weight", 0.25)) * bottom2_delta
        if secondary_enabled else 0.0
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
        "vote_gain_count": sum(int(not bool(a) and bool(b)) for a, b in zip(active_vote_correct, reward_candidate_vote)),
        "vote_loss_count": sum(int(bool(a) and not bool(b)) for a, b in zip(active_vote_correct, reward_candidate_vote)),
        "diagnostic_raw_vote_accuracy_delta": (
            float(np.mean([bool(value) for value in candidate_vote_correct]))
            - float(np.mean([bool(value) for value in active_vote_correct]))
        ) if active_vote_correct else 0.0,
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
        {index for index, value in enumerate(profile.get("correctness_vector", [])) if bool(value)}
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


def outcome_signature(profile: Mapping[str, Any], version: str) -> str:
    payload = {
        "version": version,
        "correctness": [int(bool(value)) for value in profile.get("correctness_vector", [])],
        "invalid": [int(bool(value)) for value in profile.get("invalid_vector", [])],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def safe_trace_signature(profile: Mapping[str, Any], version: str) -> str:
    embeddings = profile.get("trace_embedding_vector_per_question", [])
    payload = {
        "version": version,
        "correctness": [int(bool(value)) for value in profile.get("correctness_vector", [])],
        "valid_trace_embeddings": [
            [round(float(component), 6) for component in vector]
            if index < len(profile.get("invalid_vector", [])) and not bool(profile.get("invalid_vector", [])[index]) else []
            for index, vector in enumerate(embeddings)
        ],
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
    active_safe_available = bool(active_metrics.get("safe_trace_constraint_available", False))
    candidate_safe_available = bool(metrics.get("safe_trace_constraint_available", False))
    safe_available = active_safe_available
    safe_floor = max(
        float(active_metrics.get("safe_trace_diversity_c4c5", 0.0)) - float(getattr(config, "state_safe_trace_local_epsilon", 0.05)),
        float(initial_metrics.get("safe_trace_diversity_c4c5", 0.0)) - float(getattr(config, "state_safe_trace_global_epsilon", 0.08)),
    )
    safe_slack = float(metrics.get("safe_trace_diversity_c4c5", 0.0)) - safe_floor
    safe_passed = (
        (not diversity_enabled)
        or (not safe_available)
        or (candidate_safe_available and safe_slack >= -1e-12)
    )
    correct_passed = (not diversity_enabled) or (correct_mean_slack >= -1e-12 and correct_min_slack >= -1e-12)
    catastrophic_limit = int(getattr(config, "state_catastrophic_vote_loss_limit", -1))
    vote_loss_passed = catastrophic_limit < 0 or int(metrics.get("vote_loss_count", 0) or 0) <= catastrophic_limit
    passed = local_loss <= local_allowed and global_loss <= global_allowed and invalid_passed and correct_passed and safe_passed and vote_loss_passed
    slacks = [correct_mean_slack, correct_min_slack] + ([safe_slack] if safe_available else [])
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
        "candidate_safe_trace_constraint_available": candidate_safe_available,
        "catastrophic_vote_loss_guard_passed": vote_loss_passed,
        "diversity_constraint_slack": min(slacks, default=0.0),
        "sequential_constraints_passed": passed,
        "candidate_feasible": passed,
        "rejection_reason": "" if passed else "sequential_full_probe_constraint",
    }


def accuracy_first_key(item: Mapping[str, Any]) -> tuple:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), Mapping) else {}
    return (
        int(metrics.get("candidate_target_correct_count", 0) or 0),
        float(metrics.get("candidate_target_accuracy", 0.0) or 0.0),
        float(metrics.get("state_vote_reward", metrics.get("state_reward_total", 0.0)) or 0.0),
        -int(metrics.get("candidate_invalid_count", 0) or 0),
        float(metrics.get("diversity_constraint_slack", 0.0) or 0.0),
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
    return accuracy_first_key(candidate) > accuracy_first_key(incumbent)


def epoch_agent_order(epoch: int, num_agents: int = 5) -> list[int]:
    if num_agents <= 0:
        return []
    offset = int(epoch) % int(num_agents)
    return list(range(offset, num_agents)) + list(range(offset))


def rebuild_prompt_memory(
    items: Sequence[Mapping[str, Any]], active_prompt_hash: str, capacity: int = 5
) -> list[Dict[str, Any]]:
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
    slots: list[tuple[str, Dict[str, Any] | None]] = [
        ("active", active),
        ("accuracy_best", max(values, key=lambda item: (
            int(item.get("metrics", {}).get("candidate_target_correct_count", 0) or 0),
            -int(item.get("generation", 0) or 0), str(item.get("prompt_hash", ""))), default=None)),
        ("state_vote_best", max(values, key=lambda item: (
            float(item.get("metrics", {}).get("state_vote_reward", 0.0) or 0.0),
            int(item.get("metrics", {}).get("candidate_target_correct_count", 0) or 0),
            str(item.get("prompt_hash", ""))), default=None)),
        ("safe_diversity_parent", max(values, key=lambda item: (
            float(item.get("metrics", {}).get("diversity_constraint_slack", 0.0) or 0.0),
            float(item.get("metrics", {}).get("safe_trace_diversity_c4c5", 0.0) or 0.0),
            str(item.get("prompt_hash", ""))), default=None)),
        ("rollback_or_recent_success", max(values, key=lambda item: (
            int(item.get("accepted_update_index", -1) or -1),
            int(item.get("generation", 0) or 0), str(item.get("prompt_hash", ""))), default=None)),
    ]
    memory = []
    seen_hashes = set()
    seen_outcomes = set()
    seen_safe_traces = set()
    for slot, candidate in slots:
        if candidate is None:
            continue
        prompt_hash = str(candidate.get("prompt_hash", ""))
        outcome = str(candidate.get("outcome_signature_hash", candidate.get("outcome_signature", "")))
        safe_trace = str(candidate.get("safe_trace_signature_hash", candidate.get("safe_trace_signature", "")))
        if prompt_hash in seen_hashes:
            continue
        if outcome and outcome in seen_outcomes:
            if slot != "safe_diversity_parent" or not safe_trace or safe_trace in seen_safe_traces:
                continue
        row = dict(candidate)
        row["prompt_memory_slot"] = slot
        row["memory_slot"] = slot
        memory.append(row)
        seen_hashes.add(prompt_hash)
        if outcome:
            seen_outcomes.add(outcome)
        if safe_trace:
            seen_safe_traces.add(safe_trace)
        if len(memory) >= max(1, int(capacity)):
            break
    return memory


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
