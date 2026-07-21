"""Rollout-only diversity and quality selection for the V8 rollout-QD methods."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np


ROLLOUT_QD_METHODS = frozenset({
    "v8_accuracy_rollout_embedding",
    "v8_rollout_qd_vote_ready",
    "v9_state_conditioned_error",
})


def is_rollout_qd_method(method_version: Any) -> bool:
    return str(method_version or "").strip().lower() in ROLLOUT_QD_METHODS


def is_vote_ready_rollout_method(method_version: Any) -> bool:
    return str(method_version or "").strip().lower() == "v8_rollout_qd_vote_ready"


def rollout_signature(profile: Mapping[str, Any]) -> str:
    payload = {
        "answers": [str(value or "").strip() for value in profile.get("answer_vector", [])],
        "correctness": [int(bool(value)) for value in profile.get("correctness_vector", [])],
        "invalid": [int(bool(value)) for value in profile.get("invalid_vector", [])],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def correctness_set_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_invalid = list(left.get("invalid_vector", []))
    right_invalid = list(right.get("invalid_vector", []))
    left_set = {
        index
        for index, value in enumerate(left.get("correctness_vector", []))
        if bool(value) and not bool(left_invalid[index] if index < len(left_invalid) else 1)
    }
    right_set = {
        index
        for index, value in enumerate(right.get("correctness_vector", []))
        if bool(value) and not bool(right_invalid[index] if index < len(right_invalid) else 1)
    }
    union = left_set | right_set
    return 0.0 if not union else float(1.0 - len(left_set & right_set) / len(union))


def wrong_diversity_is_useful(row: Mapping[str, Any], *, candidate: bool) -> bool:
    baseline_at_boundary = int(row.get("baseline_gold_vote_count", 0) or 0) == 2
    if not candidate:
        return baseline_at_boundary
    return bool(
        baseline_at_boundary
        or int(row.get("candidate_largest_wrong_vote_count", 0) or 0)
        < int(row.get("baseline_largest_wrong_vote_count", 0) or 0)
        or float(row.get("candidate_plurality_margin_votes", 0.0) or 0.0)
        > float(row.get("baseline_plurality_margin_votes", 0.0) or 0.0)
        or (not bool(row.get("baseline_vote_correct", False)) and bool(row.get("candidate_vote_correct", False)))
    )


def useful_wrong_answer_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_answers = list(left.get("answer_vector", []))
    right_answers = list(right.get("answer_vector", []))
    left_correct = list(left.get("correctness_vector", []))
    right_correct = list(right.get("correctness_vector", []))
    left_invalid = list(left.get("invalid_vector", []))
    right_invalid = list(right.get("invalid_vector", []))
    left_useful = list(left.get("wrong_diversity_useful_vector", []))
    right_useful = list(right.get("wrong_diversity_useful_vector", []))
    comparisons = []
    for index in range(min(len(left_answers), len(right_answers))):
        valid = not bool(left_invalid[index] if index < len(left_invalid) else 1)
        valid = valid and not bool(right_invalid[index] if index < len(right_invalid) else 1)
        both_wrong = not bool(left_correct[index] if index < len(left_correct) else 0)
        both_wrong = both_wrong and not bool(right_correct[index] if index < len(right_correct) else 0)
        enabled = bool(left_useful[index]) if index < len(left_useful) else False
        enabled = enabled or (bool(right_useful[index]) if index < len(right_useful) else False)
        if valid and both_wrong and enabled and str(left_answers[index]).strip() and str(right_answers[index]).strip():
            comparisons.append(int(str(left_answers[index]).strip() != str(right_answers[index]).strip()))
    return float(np.mean(comparisons)) if comparisons else 0.0


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(1.0 - float(np.dot(a, b) / denominator), 0.0, 1.0))


def valid_trace_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_embeddings = list(left.get("trace_embedding_vector_per_question", []))
    right_embeddings = list(right.get("trace_embedding_vector_per_question", []))
    left_correct = list(left.get("correctness_vector", []))
    right_correct = list(right.get("correctness_vector", []))
    left_invalid = list(left.get("invalid_vector", []))
    right_invalid = list(right.get("invalid_vector", []))
    distances = []
    for index in range(min(len(left_embeddings), len(right_embeddings))):
        if bool(left_invalid[index] if index < len(left_invalid) else 1):
            continue
        if bool(right_invalid[index] if index < len(right_invalid) else 1):
            continue
        if not (bool(left_correct[index] if index < len(left_correct) else 0) or bool(right_correct[index] if index < len(right_correct) else 0)):
            continue
        distances.append(_cosine_distance(left_embeddings[index], right_embeddings[index]))
    return float(np.mean(distances)) if distances else 0.0


def rollout_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    correctness_weight: float = 0.50,
    wrong_weight: float = 0.20,
    trace_weight: float = 0.30,
) -> Dict[str, float]:
    correct = correctness_set_distance(left, right)
    wrong = useful_wrong_answer_distance(left, right)
    trace = valid_trace_distance(left, right)
    total = correctness_weight * correct + wrong_weight * wrong + trace_weight * trace
    return {
        "correct_set_rollout_distance": float(correct),
        "useful_wrong_answer_dispersion": float(wrong),
        "rollout_trace_embedding_distance": float(trace),
        "rollout_distance": float(np.clip(total, 0.0, 1.0)),
    }


def candidate_transition_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {
        "c0_to_c1_count": 0, "c1_to_c2_count": 0, "c2_to_c3_count": 0,
        "c3_to_c2_count": 0, "c2_to_c1_count": 0, "c1_to_c0_count": 0,
        "vote_gain_count": 0, "vote_loss_count": 0,
        "dominant_wrong_break_count": 0, "dominant_wrong_create_count": 0,
    }
    margin_gain = margin_loss = 0.0
    for row in rows:
        before = sum(int(bool(value)) for value in row.get("baseline_individual_correct", []))
        after = sum(int(bool(value)) for value in row.get("candidate_individual_correct", []))
        if before == 0 and after >= 1:
            counts["c0_to_c1_count"] += 1
        if before == 1 and after >= 2:
            counts["c1_to_c2_count"] += 1
        if before == 2 and after >= 3:
            counts["c2_to_c3_count"] += 1
        if before >= 3 and after <= 2:
            counts["c3_to_c2_count"] += 1
        if before == 2 and after <= 1:
            counts["c2_to_c1_count"] += 1
        if before == 1 and after == 0:
            counts["c1_to_c0_count"] += 1
        before_vote = bool(row.get("baseline_vote_correct", False))
        after_vote = bool(row.get("candidate_vote_correct", False))
        counts["vote_gain_count"] += int(not before_vote and after_vote)
        counts["vote_loss_count"] += int(before_vote and not after_vote)
        delta = float(row.get("candidate_plurality_margin_votes", 0.0) or 0.0) - float(
            row.get("baseline_plurality_margin_votes", 0.0) or 0.0
        )
        margin_gain += max(0.0, delta)
        margin_loss += max(0.0, -delta)
        before_wrong = int(row.get("baseline_largest_wrong_vote_count", 0) or 0)
        after_wrong = int(row.get("candidate_largest_wrong_vote_count", 0) or 0)
        counts["dominant_wrong_break_count"] += int(after_wrong < before_wrong)
        counts["dominant_wrong_create_count"] += int(after_wrong > before_wrong)
    denominator = max(1, len(rows))
    return {
        **counts,
        "gold_margin_gain": float(margin_gain),
        "gold_margin_loss": float(margin_loss),
        "net_vote_count": int(counts["vote_gain_count"] - counts["vote_loss_count"]),
        "net_c3_count": int(counts["c2_to_c3_count"] - counts["c3_to_c2_count"]),
        "net_dominant_wrong_break_count": int(
            counts["dominant_wrong_break_count"] - counts["dominant_wrong_create_count"]
        ),
        "net_vote_rate": float((counts["vote_gain_count"] - counts["vote_loss_count"]) / denominator),
        "net_c3_rate": float((counts["c2_to_c3_count"] - counts["c3_to_c2_count"]) / denominator),
        "gold_margin_delta": float((margin_gain - margin_loss) / denominator),
        "dominant_wrong_net_rate": float(
            (counts["dominant_wrong_break_count"] - counts["dominant_wrong_create_count"]) / denominator
        ),
    }


def quality_guard(metrics: Mapping[str, Any], config: Any) -> Dict[str, Any]:
    accuracy_passed = float(metrics.get("candidate_target_accuracy", 0.0)) >= (
        float(metrics.get("baseline_target_accuracy", 0.0)) - float(config.accuracy_guard_epsilon)
    )
    invalid_passed = float(metrics.get("candidate_invalid_rate", 1.0)) <= (
        float(metrics.get("baseline_invalid_rate", 1.0)) + float(config.invalid_guard_epsilon)
    )
    c3_passed = int(metrics.get("c3_to_c2_count", 0) or 0) <= int(config.rollout_c3_loss_epsilon)
    vote_passed = int(metrics.get("vote_loss_count", 0) or 0) <= int(config.rollout_vote_loss_epsilon)
    passed = accuracy_passed and invalid_passed and c3_passed and vote_passed
    return {
        "accuracy_guard_passed": bool(accuracy_passed),
        "invalid_guard_passed": bool(invalid_passed),
        "c3_loss_guard_passed": bool(c3_passed),
        "vote_loss_guard_passed": bool(vote_passed),
        "rollout_quality_guard_passed": bool(passed),
        "rejection_reason": "" if passed else "rollout_quality_guard",
    }


def candidate_reward(metrics: Mapping[str, Any], config: Any, *, vote_ready: bool) -> float:
    invalid_delta = max(
        0.0,
        float(metrics.get("candidate_invalid_rate", 0.0)) - float(metrics.get("baseline_invalid_rate", 0.0)),
    )
    diversity_gain = float(metrics.get("rollout_diversity_delta", 0.0) or 0.0)
    target_accuracy = float(metrics.get("candidate_target_accuracy", 0.0) or 0.0)
    if not vote_ready:
        return float(
            config.rollout_simple_target_accuracy_weight * target_accuracy
            + config.rollout_simple_diversity_weight * diversity_gain
            - config.rollout_simple_invalid_weight * invalid_delta
        )
    return float(
        config.vote_ready_target_accuracy_weight * target_accuracy
        + config.vote_ready_vote_weight * float(metrics.get("net_vote_rate", 0.0) or 0.0)
        + config.vote_ready_c3_weight * float(metrics.get("net_c3_rate", 0.0) or 0.0)
        + config.vote_ready_margin_weight * float(metrics.get("gold_margin_delta", 0.0) or 0.0)
        + config.vote_ready_wrong_break_weight * float(metrics.get("dominant_wrong_net_rate", 0.0) or 0.0)
        + config.vote_ready_diversity_weight * diversity_gain
        - config.rollout_simple_invalid_weight * invalid_delta
    )


def vote_ready_candidate_key(item: Mapping[str, Any]) -> tuple:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), Mapping) else {}
    return (
        -int(metrics.get("vote_loss_count", 0) or 0),
        -int(metrics.get("c3_to_c2_count", 0) or 0),
        int(metrics.get("vote_gain_count", 0) or 0),
        int(metrics.get("c2_to_c3_count", 0) or 0),
        float(metrics.get("candidate_target_accuracy", 0.0) or 0.0),
        float(metrics.get("gold_margin_gain", 0.0) or 0.0),
        int(metrics.get("dominant_wrong_break_count", 0) or 0),
        float(metrics.get("candidate_rollout_diversity", 0.0) or 0.0),
        -float(metrics.get("candidate_invalid_rate", 1.0) or 1.0),
        -int(item.get("generation", 0) or 0),
        str(item.get("prompt_hash", "")),
    )


def rollout_quality_key(item: Mapping[str, Any], *, vote_ready: bool) -> tuple:
    if vote_ready:
        return vote_ready_candidate_key(item)
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), Mapping) else {}
    return (
        float(item.get("reward", metrics.get("reward", 0.0)) or 0.0),
        float(metrics.get("candidate_target_accuracy", 0.0) or 0.0),
        int(metrics.get("net_vote_count", 0) or 0),
        int(metrics.get("net_c3_count", 0) or 0),
        float(metrics.get("candidate_rollout_diversity", 0.0) or 0.0),
        -int(item.get("generation", 0) or 0),
        str(item.get("prompt_hash", "")),
    )


def _profile(item: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), Mapping) else {}
    profile = metrics.get("rollout_profile", {})
    return profile if isinstance(profile, Mapping) else {}


def select_rollout_archive(
    items: Sequence[Dict[str, Any]],
    incumbent_hash: str,
    capacity: int,
    config: Any,
    *,
    vote_ready: bool,
) -> list[Dict[str, Any]]:
    safe = [dict(item) for item in items if str(item.get("prompt_hash", "")) == incumbent_hash or bool(
        (item.get("metrics", {}) or {}).get("rollout_quality_guard_passed", False)
    )]
    by_prompt: Dict[str, Dict[str, Any]] = {}
    for item in safe:
        prompt_hash = str(item.get("prompt_hash", ""))
        previous = by_prompt.get(prompt_hash)
        if previous is None or (
            bool(_profile(item)) and not bool(_profile(previous))
        ) or rollout_quality_key(item, vote_ready=vote_ready) > rollout_quality_key(previous, vote_ready=vote_ready):
            by_prompt[prompt_hash] = item
    by_signature: Dict[str, Dict[str, Any]] = {}
    for item in by_prompt.values():
        signature = str(_profile(item).get("rollout_signature_hash", "") or rollout_signature(_profile(item)))
        key = signature or str(item.get("prompt_hash", ""))
        previous = by_signature.get(key)
        item_is_incumbent = str(item.get("prompt_hash", "")) == incumbent_hash
        previous_is_incumbent = bool(previous) and str(previous.get("prompt_hash", "")) == incumbent_hash
        if previous is None or (item_is_incumbent and not previous_is_incumbent) or (
            item_is_incumbent == previous_is_incumbent
            and rollout_quality_key(item, vote_ready=vote_ready) > rollout_quality_key(previous, vote_ready=vote_ready)
        ):
            by_signature[key] = item
    candidates = list(by_signature.values())
    candidates.sort(key=lambda item: rollout_quality_key(item, vote_ready=vote_ready), reverse=True)
    incumbent = next((item for item in candidates if str(item.get("prompt_hash", "")) == incumbent_hash), None)
    selected = [incumbent] if incumbent is not None else []
    for item in candidates:
        if item in selected:
            continue
        if len(selected) >= max(1, capacity - 1):
            break
        selected.append(item)
    if len(selected) < capacity:
        remaining = [item for item in candidates if item not in selected]
        while remaining and len(selected) < capacity:
            best = max(
                remaining,
                key=lambda item: min(
                    [rollout_distance(_profile(item), _profile(other),
                        correctness_weight=config.rollout_correct_distance_weight,
                        wrong_weight=config.rollout_wrong_distance_weight,
                        trace_weight=config.rollout_trace_distance_weight)["rollout_distance"] for other in selected]
                    or [0.0]
                ),
            )
            selected.append(best)
            remaining.remove(best)
    return selected[:capacity]


def select_rollout_representatives(
    archive: Sequence[Dict[str, Any]],
    incumbent_hash: str,
    capacity: int,
    config: Any,
    *,
    vote_ready: bool,
) -> list[Dict[str, Any]]:
    if not archive:
        return []
    incumbent = next((dict(item) for item in archive if str(item.get("prompt_hash", "")) == incumbent_hash), dict(archive[0]))
    quality = max(archive, key=lambda item: rollout_quality_key(item, vote_ready=vote_ready))
    selected = [incumbent]
    if str(quality.get("prompt_hash", "")) != str(incumbent.get("prompt_hash", "")):
        selected.append(dict(quality))
    remaining = [dict(item) for item in archive if str(item.get("prompt_hash", "")) not in {str(row.get("prompt_hash", "")) for row in selected}]
    while remaining and len(selected) < capacity:
        best = max(remaining, key=lambda item: min(
            rollout_distance(
                _profile(item), _profile(other),
                correctness_weight=config.rollout_correct_distance_weight,
                wrong_weight=config.rollout_wrong_distance_weight,
                trace_weight=config.rollout_trace_distance_weight,
            )["rollout_distance"] for other in selected
        ))
        selected.append(best)
        remaining.remove(best)
    return selected[:capacity]


def rollout_team_metrics(
    prompt_profiles: Sequence[Mapping[str, Any]],
    gold_answers: Sequence[str],
    question_hashes: Sequence[str],
    *,
    vote_fn: Callable[..., Mapping[str, Any]],
    match_fn: Callable[[str, str], bool],
    tie_break_method: str,
    seed: int,
    config: Any,
) -> Dict[str, Any]:
    answer_vectors = [list(profile.get("answer_vector", [])) for profile in prompt_profiles]
    correct_vectors = [list(profile.get("correctness_vector", [])) for profile in prompt_profiles]
    invalid_vectors = [list(profile.get("invalid_vector", [])) for profile in prompt_profiles]
    vote_correct, depths, margins, wrong_concentrations, same_wrong = [], [], [], [], []
    useful_wrong_pair_values = []
    for index, gold in enumerate(gold_answers):
        answers = [values[index] if index < len(values) else "" for values in answer_vectors]
        correct = [int(values[index]) if index < len(values) else 0 for values in correct_vectors]
        vote = vote_fn(answers, tie_break_method=tie_break_method, seed=seed, question_hash=str(question_hashes[index]))
        vote_correct.append(int(match_fn(str(vote.get("vote_answer", "")), str(gold))))
        depths.append(sum(correct))
        counts = dict(vote.get("vote_counts", {}))
        gold_count = sum(int(match_fn(str(answer), str(gold))) * int(count) for answer, count in counts.items())
        wrong_counts = [int(count) for answer, count in counts.items() if not match_fn(str(answer), str(gold))]
        max_wrong = max(wrong_counts, default=0)
        margins.append(float((gold_count - max_wrong) / max(1, len(prompt_profiles))))
        wrong_concentrations.append(float(max_wrong / max(1, sum(wrong_counts))))
        wrong_pairs = sum(count * (count - 1) / 2 for count in wrong_counts)
        total_pairs = max(1.0, (len(prompt_profiles) - gold_count) * (len(prompt_profiles) - gold_count - 1) / 2)
        same_wrong.append(float(wrong_pairs / total_pairs))
        if gold_count == 2:
            pair_values = []
            for left, right in itertools.combinations(range(len(answers)), 2):
                left_invalid = bool(invalid_vectors[left][index]) if index < len(invalid_vectors[left]) else True
                right_invalid = bool(invalid_vectors[right][index]) if index < len(invalid_vectors[right]) else True
                if left_invalid or right_invalid or correct[left] or correct[right]:
                    continue
                if str(answers[left]).strip() and str(answers[right]).strip():
                    pair_values.append(int(str(answers[left]).strip() != str(answers[right]).strip()))
            useful_wrong_pair_values.extend(pair_values)
    per_agent_counts = [sum(int(value) for value in vector) for vector in correct_vectors]
    pair_component_rows = [
        rollout_distance(
            prompt_profiles[left], prompt_profiles[right],
            correctness_weight=config.rollout_correct_distance_weight,
            wrong_weight=config.rollout_wrong_distance_weight,
            trace_weight=config.rollout_trace_distance_weight,
        )
        for left, right in itertools.combinations(range(len(prompt_profiles)), 2)
    ]
    correct_distance = float(np.mean([row["correct_set_rollout_distance"] for row in pair_component_rows])) if pair_component_rows else 0.0
    trace_distance = float(np.mean([row["rollout_trace_embedding_distance"] for row in pair_component_rows])) if pair_component_rows else 0.0
    useful_wrong_distance = float(np.mean(useful_wrong_pair_values)) if useful_wrong_pair_values else 0.0
    rollout_diversity_score = float(np.clip(
        config.rollout_correct_distance_weight * correct_distance
        + config.rollout_wrong_distance_weight * useful_wrong_distance
        + config.rollout_trace_distance_weight * trace_distance,
        0.0, 1.0,
    ))
    size = max(1, len(gold_answers))
    return {
        "vote_correct_count": int(sum(vote_correct)),
        "vote_acc": float(sum(vote_correct) / size),
        "c3_correct_count": int(sum(depth >= 3 for depth in depths)),
        "coverage_depth_c3": float(sum(depth >= 3 for depth in depths) / size),
        "coverage_depth_c2": float(sum(depth >= 2 for depth in depths) / size),
        "coverage_depth_c1": float(sum(depth >= 1 for depth in depths) / size),
        "total_agent_correct_count": int(sum(per_agent_counts)),
        "mean_individual_acc": float(sum(per_agent_counts) / max(1, size * len(prompt_profiles))),
        "bottom2_correct_count": int(sum(sorted(per_agent_counts)[:2])),
        "bottom2_mean_acc": float(sum(sorted(per_agent_counts)[:2]) / max(1, size * min(2, len(per_agent_counts)))),
        "per_agent_correct_count": per_agent_counts,
        "per_agent_acc": [float(value / size) for value in per_agent_counts],
        "mean_gold_plurality_margin": float(np.mean(margins)) if margins else 0.0,
        "dominant_wrong_concentration": float(np.mean(wrong_concentrations)) if wrong_concentrations else 0.0,
        "same_wrong_pair_rate": float(np.mean(same_wrong)) if same_wrong else 0.0,
        "rollout_diversity_score": rollout_diversity_score,
        "correct_set_rollout_distance": correct_distance,
        "useful_wrong_answer_dispersion": useful_wrong_distance,
        "rollout_trace_embedding_distance": trace_distance,
        "answer_vectors": answer_vectors,
        "correctness_vectors": correct_vectors,
        "invalid_vectors": invalid_vectors,
    }


def rollout_team_key(team: Mapping[str, Any]) -> tuple:
    return (
        int(team.get("vote_correct_count", 0) or 0),
        int(team.get("c3_correct_count", 0) or 0),
        int(team.get("total_agent_correct_count", 0) or 0),
        int(team.get("bottom2_correct_count", 0) or 0),
        float(team.get("mean_gold_plurality_margin", 0.0) or 0.0),
        -float(team.get("dominant_wrong_concentration", 1.0) or 0.0),
        float(team.get("rollout_diversity_score", 0.0) or 0.0),
        float(team.get("coverage_depth_c2", 0.0) or 0.0),
        float(team.get("coverage_depth_c1", 0.0) or 0.0),
        json.dumps(team.get("prompt_hashes", []), separators=(",", ":")),
    )


def accuracy_rollout_team_key(team: Mapping[str, Any]) -> tuple:
    return (
        int(team.get("total_agent_correct_count", 0) or 0),
        int(team.get("vote_correct_count", 0) or 0),
        int(team.get("c3_correct_count", 0) or 0),
        int(team.get("bottom2_correct_count", 0) or 0),
        float(team.get("mean_gold_plurality_margin", 0.0) or 0.0),
        float(team.get("rollout_diversity_score", 0.0) or 0.0),
        -float(team.get("dominant_wrong_concentration", 1.0) or 0.0),
        json.dumps(team.get("prompt_hashes", []), separators=(",", ":")),
    )


def enumerate_rollout_teams(
    beams: Sequence[Sequence[Mapping[str, Any]]],
    gold_answers: Sequence[str],
    question_hashes: Sequence[str],
    **kwargs: Any,
) -> list[Dict[str, Any]]:
    teams = []
    for indices in itertools.product(*[range(len(beam)) for beam in beams]):
        profiles = [beams[agent_id][index] for agent_id, index in enumerate(indices)]
        metrics = rollout_team_metrics(profiles, gold_answers, question_hashes, **kwargs)
        teams.append({
            "beam_indices": list(indices),
            "prompt_profiles": profiles,
            "prompt_hashes": [str(profile.get("prompt_hash", "")) for profile in profiles],
            **metrics,
        })
    return teams
