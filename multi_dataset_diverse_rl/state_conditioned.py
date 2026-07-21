"""State-conditioned correlated-error optimization for the V9 method."""

from __future__ import annotations

import itertools
import hashlib
import json
import math
from collections import Counter
from typing import Any, Callable, Dict, Mapping, Sequence


STATE_CONDITIONED_METHOD = "v9_state_conditioned_error"
STATE_CONDITIONED_CHECKPOINT_VERSION = 1
STATE_NAMES = ("C0", "C1", "C2", "C3PLUS")


def coverage_case_assignees(
    question_hash: str,
    num_agents: int,
    *,
    seed: int = 0,
    assignment_count: int = 2,
) -> list[int]:
    """Assign a hard coverage case deterministically without fixed agent roles."""
    count = max(0, min(int(assignment_count), int(num_agents)))
    if count == 0:
        return []
    ranked = []
    for agent_id in range(int(num_agents)):
        digest = hashlib.sha256(
            f"{int(seed)}|{str(question_hash)}|{agent_id}".encode("utf-8")
        ).hexdigest()
        ranked.append((digest, agent_id))
    return [agent_id for _, agent_id in sorted(ranked)[:count]]


def is_state_conditioned_method(method_version: Any) -> bool:
    return str(method_version or "").strip().lower() == STATE_CONDITIONED_METHOD


def question_state(gold_vote_count: Any) -> str:
    count = max(0, int(gold_vote_count or 0))
    if count == 0:
        return "C0"
    if count == 1:
        return "C1"
    if count == 2:
        return "C2"
    return "C3PLUS"


def c2_dispersion_rescuability(option_count: Any, *, wrong_agent_count: int = 3) -> Dict[str, Any]:
    options = max(0, int(option_count or 0))
    wrong_options = max(0, options - 1)
    if wrong_options <= 0:
        minimum_largest_wrong = int(wrong_agent_count)
        label = "unrescuable"
    else:
        minimum_largest_wrong = int(math.ceil(int(wrong_agent_count) / wrong_options))
        label = (
            "strict" if minimum_largest_wrong < 2
            else "tie_only" if minimum_largest_wrong == 2
            else "unrescuable"
        )
    return {
        "option_count": options,
        "wrong_option_count": wrong_options,
        "c2_minimum_largest_wrong_vote": minimum_largest_wrong,
        "c2_strictly_rescuable_by_dispersion": label == "strict",
        "c2_tie_only_rescuable_by_dispersion": label == "tie_only",
        "c2_unrescuable_by_dispersion": label == "unrescuable",
        "c2_dispersion_rescuability": label,
    }


def _dominant_wrong_answer(answers: Sequence[Any], correct: Sequence[Any]) -> str:
    counts = Counter(
        str(answer or "").strip()
        for answer, is_correct in zip(answers, correct)
        if str(answer or "").strip() and not bool(is_correct)
    )
    if not counts:
        return ""
    best = max(counts.values())
    return sorted(answer for answer, count in counts.items() if count == best)[0]


def candidate_row_state_fields(row: Mapping[str, Any], option_count: Any = 0) -> Dict[str, Any]:
    baseline_g = int(row.get("baseline_gold_vote_count", 0) or 0)
    candidate_g = int(row.get("candidate_gold_vote_count", 0) or 0)
    baseline_h = int(row.get("baseline_largest_wrong_vote_count", 0) or 0)
    candidate_h = int(row.get("candidate_largest_wrong_vote_count", 0) or 0)
    baseline_correct = list(row.get("baseline_individual_correct", []))
    candidate_correct = list(row.get("candidate_individual_correct", []))
    return {
        "baseline_state": question_state(baseline_g),
        "candidate_state": question_state(candidate_g),
        "baseline_G": baseline_g,
        "candidate_G": candidate_g,
        "baseline_H": baseline_h,
        "candidate_H": candidate_h,
        "baseline_M": baseline_g - baseline_h,
        "candidate_M": candidate_g - candidate_h,
        "option_count": max(0, int(option_count or row.get("option_count", 0) or 0)),
        "baseline_target_correct": int(bool(row.get("baseline_target_correct", False))),
        "candidate_target_correct": int(bool(row.get(
            "candidate_target_correct", row.get("target_agent_correct", False)
        ))),
        "target_answer": str(row.get("target_answer", "") or ""),
        "baseline_dominant_wrong_answer": _dominant_wrong_answer(
            list(row.get("baseline_answers", [])), baseline_correct
        ),
        "candidate_dominant_wrong_answer": _dominant_wrong_answer(
            list(row.get("candidate_answers", [])), candidate_correct
        ),
    }


def compute_c2_wrong_split_metrics(row: Mapping[str, Any]) -> Dict[str, Any]:
    baseline_g = int(row.get("baseline_G", row.get("baseline_gold_vote_count", 0)) or 0)
    candidate_g = int(row.get("candidate_G", row.get("candidate_gold_vote_count", 0)) or 0)
    baseline_h = int(row.get("baseline_H", row.get("baseline_largest_wrong_vote_count", 0)) or 0)
    candidate_h = int(row.get("candidate_H", row.get("candidate_largest_wrong_vote_count", 0)) or 0)
    enabled = baseline_g == 2 and candidate_g == 2
    reduction = max(0, baseline_h - candidate_h) if enabled else 0
    creation = max(0, candidate_h - baseline_h) if enabled else 0
    baseline_vote = bool(row.get("baseline_vote_correct", False))
    candidate_vote = bool(row.get("candidate_vote_correct", False))
    candidate_margin = int(row.get("candidate_M", candidate_g - candidate_h) or 0)
    vote_gain = int(enabled and not baseline_vote and candidate_vote)
    vote_loss = int(enabled and baseline_vote and not candidate_vote)
    tie_gain = int(enabled and candidate_margin == 0 and reduction > 0)
    strict_gain = int(enabled and candidate_margin > 0 and not baseline_vote and candidate_vote)
    return {
        "c2_wrong_split_enabled": bool(enabled),
        "c2_wrong_cluster_reduction": int(reduction),
        "c2_wrong_cluster_creation": int(creation),
        "c2_wrong_split_vote_gain_count": vote_gain,
        "c2_wrong_split_vote_loss_count": vote_loss,
        "c2_wrong_split_tie_gain_count": tie_gain,
        "c2_wrong_split_strict_gain_count": strict_gain,
        "c2_dominant_wrong_break_count": int(enabled and reduction > 0),
        "c2_dominant_wrong_create_count": int(enabled and creation > 0),
        "wrong_answer_diversity_task_gain": float(reduction) if enabled else 0.0,
    }


def state_conditioned_transition_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {
        "c0_to_c1_count": 0,
        "c1_to_c2_count": 0,
        "c2_to_c3_count": 0,
        "c3plus_additional_correct_count": 0,
        "c1_to_c0_count": 0,
        "c2_to_c1_count": 0,
        "c3_to_c2_count": 0,
        "target_wrong_to_correct_count": 0,
        "target_correct_to_wrong_count": 0,
        "vote_gain_count": 0,
        "vote_loss_count": 0,
        "c2_wrong_split_vote_gain_count": 0,
        "c2_wrong_split_vote_loss_count": 0,
        "c2_wrong_split_tie_gain_count": 0,
        "c2_wrong_split_strict_gain_count": 0,
        "c2_dominant_wrong_break_count": 0,
        "c2_dominant_wrong_create_count": 0,
        "c2_wrong_cluster_reduction": 0,
        "c2_wrong_cluster_creation": 0,
        "c2_strictly_rescuable_count": 0,
        "c2_tie_only_rescuable_count": 0,
        "c2_unrescuable_by_dispersion_count": 0,
    }
    paired_rows = []
    c0_rescue_vector = []
    c1_deepening_vector = []
    for raw in rows:
        row = {**dict(raw), **candidate_row_state_fields(raw, raw.get("option_count", 0))}
        baseline_g = int(row["baseline_G"])
        candidate_g = int(row["candidate_G"])
        counts["c0_to_c1_count"] += int(baseline_g == 0 and candidate_g >= 1)
        counts["c1_to_c2_count"] += int(baseline_g == 1 and candidate_g >= 2)
        c0_rescue_vector.append(int(baseline_g == 0 and candidate_g >= 1))
        c1_deepening_vector.append(int(baseline_g == 1 and candidate_g >= 2))
        counts["c2_to_c3_count"] += int(baseline_g == 2 and candidate_g >= 3)
        counts["c3plus_additional_correct_count"] += max(
            0, candidate_g - baseline_g
        ) if baseline_g >= 3 else 0
        counts["c1_to_c0_count"] += int(baseline_g == 1 and candidate_g == 0)
        counts["c2_to_c1_count"] += int(baseline_g == 2 and candidate_g <= 1)
        counts["c3_to_c2_count"] += int(baseline_g >= 3 and candidate_g <= 2)
        before_target = bool(row.get("baseline_target_correct", False))
        after_target = bool(row.get("candidate_target_correct", False))
        counts["target_wrong_to_correct_count"] += int(not before_target and after_target)
        counts["target_correct_to_wrong_count"] += int(before_target and not after_target)
        before_vote = bool(row.get("baseline_vote_correct", False))
        after_vote = bool(row.get("candidate_vote_correct", False))
        counts["vote_gain_count"] += int(not before_vote and after_vote)
        counts["vote_loss_count"] += int(before_vote and not after_vote)
        split = compute_c2_wrong_split_metrics(row)
        for key in (
            "c2_wrong_split_vote_gain_count", "c2_wrong_split_vote_loss_count",
            "c2_wrong_split_tie_gain_count", "c2_wrong_split_strict_gain_count",
            "c2_dominant_wrong_break_count", "c2_dominant_wrong_create_count",
            "c2_wrong_cluster_reduction", "c2_wrong_cluster_creation",
        ):
            counts[key] += int(split[key])
        rescue = c2_dispersion_rescuability(row.get("option_count", 0))
        if baseline_g == 2:
            counts["c2_strictly_rescuable_count"] += int(rescue["c2_strictly_rescuable_by_dispersion"])
            counts["c2_tie_only_rescuable_count"] += int(rescue["c2_tie_only_rescuable_by_dispersion"])
            counts["c2_unrescuable_by_dispersion_count"] += int(rescue["c2_unrescuable_by_dispersion"])
        paired_rows.append({**row, **split, **rescue})
    return {
        **counts,
        "coverage_utility_key": (
            counts["c0_to_c1_count"], counts["c1_to_c2_count"],
            -counts["c1_to_c0_count"], -counts["c2_to_c1_count"],
        ),
        "conversion_utility_key": (
            counts["c2_to_c3_count"], counts["c2_wrong_split_strict_gain_count"],
            counts["c2_wrong_split_vote_gain_count"], counts["c2_wrong_cluster_reduction"],
            counts["c2_wrong_split_tie_gain_count"], -counts["c2_dominant_wrong_create_count"],
        ),
        "state_transition_rows": paired_rows,
        "c0_rescue_vector_per_prompt": c0_rescue_vector,
        "c1_deepening_vector_per_prompt": c1_deepening_vector,
    }


def state_quality_guard(metrics: Mapping[str, Any], config: Any) -> Dict[str, Any]:
    accuracy_passed = float(metrics.get("candidate_target_accuracy", 0.0)) >= (
        float(metrics.get("baseline_target_accuracy", 0.0)) - float(config.accuracy_guard_epsilon)
    )
    invalid_passed = float(metrics.get("candidate_invalid_rate", 1.0)) <= (
        float(metrics.get("baseline_invalid_rate", 1.0)) + float(config.invalid_guard_epsilon)
    )
    loss_fields = {
        "c1_to_c0": ("c1_to_c0_count", "state_c1_to_c0_loss_epsilon"),
        "c2_to_c1": ("c2_to_c1_count", "state_c2_to_c1_loss_epsilon"),
        "c3_to_c2": ("c3_to_c2_count", "state_c3_to_c2_loss_epsilon"),
        "vote": ("vote_loss_count", "state_vote_loss_epsilon"),
    }
    loss_passed = {
        name: int(metrics.get(metric, 0) or 0) <= int(getattr(config, epsilon, 0) or 0)
        for name, (metric, epsilon) in loss_fields.items()
    }
    passed = accuracy_passed and invalid_passed and all(loss_passed.values())
    return {
        "accuracy_guard_passed": bool(accuracy_passed),
        "invalid_guard_passed": bool(invalid_passed),
        "c1_to_c0_guard_passed": bool(loss_passed["c1_to_c0"]),
        "c2_to_c1_guard_passed": bool(loss_passed["c2_to_c1"]),
        "c3_to_c2_guard_passed": bool(loss_passed["c3_to_c2"]),
        "vote_loss_guard_passed": bool(loss_passed["vote"]),
        "state_quality_guard_passed": bool(passed),
        "rollout_quality_guard_passed": bool(passed),
        "rejection_reason": "" if passed else "state_quality_guard",
    }


def _metrics(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("metrics", {})
    return value if isinstance(value, Mapping) else {}


def global_accuracy_quality_key(item: Mapping[str, Any], *, trace_tiebreak: bool = False) -> tuple:
    metrics = _metrics(item)
    return (
        float(metrics.get("candidate_target_accuracy", 0.0) or 0.0),
        -float(metrics.get("candidate_invalid_rate", 1.0) or 1.0),
        int(metrics.get("target_wrong_to_correct_count", 0) or 0),
        -int(metrics.get("target_correct_to_wrong_count", 0) or 0),
        float(metrics.get("trace_embedding_distance", 0.0) or 0.0) if trace_tiebreak else 0.0,
        -int(item.get("generation", 0) or 0),
        str(item.get("prompt_hash", "")),
    )


def coverage_utility_key(item: Mapping[str, Any], *, trace_tiebreak: bool = False) -> tuple:
    metrics = _metrics(item)
    return (
        int(metrics.get("c0_to_c1_count", 0) or 0),
        int(metrics.get("c1_to_c2_count", 0) or 0),
        -int(metrics.get("c1_to_c0_count", 0) or 0),
        -int(metrics.get("c2_to_c1_count", 0) or 0),
        float(metrics.get("candidate_target_accuracy", 0.0) or 0.0),
        -float(metrics.get("candidate_invalid_rate", 1.0) or 1.0),
        float(metrics.get("trace_embedding_distance", 0.0) or 0.0) if trace_tiebreak else 0.0,
        str(item.get("prompt_hash", "")),
    )


def conversion_utility_key(item: Mapping[str, Any], *, trace_tiebreak: bool = False) -> tuple:
    metrics = _metrics(item)
    return (
        int(metrics.get("c2_to_c3_count", 0) or 0),
        int(metrics.get("c2_wrong_split_strict_gain_count", 0) or 0),
        int(metrics.get("c2_wrong_split_vote_gain_count", 0) or 0),
        int(metrics.get("c2_wrong_cluster_reduction", 0) or 0),
        int(metrics.get("c2_wrong_split_tie_gain_count", 0) or 0),
        -int(metrics.get("c2_dominant_wrong_create_count", 0) or 0),
        float(metrics.get("candidate_target_accuracy", 0.0) or 0.0),
        -float(metrics.get("candidate_invalid_rate", 1.0) or 1.0),
        float(metrics.get("trace_embedding_distance", 0.0) or 0.0) if trace_tiebreak else 0.0,
        str(item.get("prompt_hash", "")),
    )


def state_conditioned_candidate_key(item: Mapping[str, Any], config: Any) -> tuple:
    metrics = _metrics(item)
    route = str(item.get("optimization_route", metrics.get("optimization_route", "general_accuracy")))
    trace = bool(getattr(config, "state_trace_tiebreak_enabled", True))
    route_key = (
        conversion_utility_key(item, trace_tiebreak=trace)
        if route == "vote_conversion" else
        coverage_utility_key(item, trace_tiebreak=trace)
        if route == "coverage_repair" else
        global_accuracy_quality_key(item, trace_tiebreak=trace)
    )
    return (
        float(metrics.get("candidate_target_accuracy", 0.0) or 0.0),
        route_key,
    )


def _rollout_signature(item: Mapping[str, Any]) -> str:
    profile = _metrics(item).get("rollout_profile", {})
    if not isinstance(profile, Mapping):
        return ""
    return str(profile.get("rollout_signature_hash", "") or "")


def select_state_conditioned_archive(
    items: Sequence[Mapping[str, Any]],
    incumbent_hash: str,
    capacity: int,
    config: Any,
) -> list[Dict[str, Any]]:
    safe = [
        dict(item) for item in items
        if str(item.get("prompt_hash", "")) == incumbent_hash
        or bool(_metrics(item).get("state_quality_guard_passed", False))
    ]
    by_prompt: Dict[str, Dict[str, Any]] = {}
    for item in safe:
        prompt_hash = str(item.get("prompt_hash", ""))
        previous = by_prompt.get(prompt_hash)
        if previous is None or state_conditioned_candidate_key(item, config) > state_conditioned_candidate_key(previous, config):
            by_prompt[prompt_hash] = item
    by_signature: Dict[str, Dict[str, Any]] = {}
    for item in by_prompt.values():
        key = _rollout_signature(item) or str(item.get("prompt_hash", ""))
        previous = by_signature.get(key)
        incumbent = str(item.get("prompt_hash", "")) == incumbent_hash
        previous_incumbent = bool(previous) and str(previous.get("prompt_hash", "")) == incumbent_hash
        if previous is None or (incumbent and not previous_incumbent) or (
            incumbent == previous_incumbent
            and state_conditioned_candidate_key(item, config) > state_conditioned_candidate_key(previous, config)
        ):
            by_signature[key] = item
    candidates = list(by_signature.values())
    non_incumbent_safe = [
        item for item in candidates
        if str(item.get("prompt_hash", "")) != incumbent_hash
        and bool(_metrics(item).get("state_quality_guard_passed", False))
    ]
    best_accuracy = max(
        (float(_metrics(item).get("candidate_target_accuracy", 0.0) or 0.0) for item in non_incumbent_safe),
        default=None,
    )
    epsilon = max(0.0, float(getattr(config, "state_accuracy_tie_epsilon", 0.02) or 0.0))
    accuracy_band = [
        item for item in non_incumbent_safe
        if best_accuracy is not None
        and float(_metrics(item).get("candidate_target_accuracy", 0.0) or 0.0) >= best_accuracy - epsilon
    ]
    slots = []

    def add(item: Mapping[str, Any] | None, slot: str) -> None:
        if item is None:
            return
        prompt_hash = str(item.get("prompt_hash", ""))
        if any(str(existing.get("prompt_hash", "")) == prompt_hash for existing in slots):
            return
        candidate = dict(item)
        candidate["state_archive_slot"] = slot
        candidate.setdefault("metrics", {})["state_archive_slot"] = slot
        slots.append(candidate)

    add(next((item for item in candidates if str(item.get("prompt_hash", "")) == incumbent_hash), None), "incumbent")
    trace = bool(getattr(config, "state_trace_tiebreak_enabled", True))
    add(max(accuracy_band, key=lambda item: global_accuracy_quality_key(item, trace_tiebreak=trace), default=None), "overall_accuracy")
    if bool(getattr(config, "state_coverage_enabled", True)):
        add(max(accuracy_band, key=lambda item: coverage_utility_key(item, trace_tiebreak=trace), default=None), "coverage_repair")
    if bool(getattr(config, "state_c2_wrong_split_enabled", True)):
        add(max(accuracy_band, key=lambda item: conversion_utility_key(item, trace_tiebreak=trace), default=None), "vote_conversion")
    for item in sorted(accuracy_band, key=lambda value: state_conditioned_candidate_key(value, config), reverse=True):
        add(item, "quality_fill")
        if len(slots) >= max(1, int(capacity)):
            break
    return slots[: max(1, int(capacity))]


def select_state_conditioned_representatives(
    archive: Sequence[Mapping[str, Any]], incumbent_hash: str, capacity: int, config: Any
) -> list[Dict[str, Any]]:
    ordered = sorted(
        (dict(item) for item in archive),
        key=lambda item: (
            str(item.get("prompt_hash", "")) == incumbent_hash,
            {"overall_accuracy": 3, "coverage_repair": 2, "vote_conversion": 1}.get(
                str(item.get("state_archive_slot", "")), 0
            ),
            state_conditioned_candidate_key(item, config),
        ),
        reverse=True,
    )
    return ordered[: max(1, int(capacity))]


def state_team_metrics(
    prompt_profiles: Sequence[Mapping[str, Any]],
    gold_answers: Sequence[str],
    question_hashes: Sequence[str],
    *,
    vote_fn: Callable[..., Mapping[str, Any]],
    match_fn: Callable[[str, str], bool],
    tie_break_method: str,
    seed: int,
) -> Dict[str, Any]:
    answers = [list(profile.get("answer_vector", [])) for profile in prompt_profiles]
    correctness = [list(profile.get("correctness_vector", [])) for profile in prompt_profiles]
    invalids = [list(profile.get("invalid_vector", [])) for profile in prompt_profiles]
    trace_distances = []
    c0 = c1 = c2 = c3plus = vote_correct = c2_vote_correct = c2_strict = c2_tie = 0
    margins = []
    c2_wrong = []
    for index, gold in enumerate(gold_answers):
        question_answers = [vector[index] if index < len(vector) else "" for vector in answers]
        question_correct = [int(vector[index]) if index < len(vector) else 0 for vector in correctness]
        g = sum(question_correct)
        vote = vote_fn(question_answers, tie_break_method=tie_break_method, seed=seed, question_hash=str(question_hashes[index]))
        is_vote_correct = int(match_fn(str(vote.get("vote_answer", "")), str(gold)))
        counts = dict(vote.get("vote_counts", {}))
        wrong_counts = [int(count) for answer, count in counts.items() if not match_fn(str(answer), str(gold))]
        h = max(wrong_counts, default=0)
        margin = g - h
        c0 += int(g == 0)
        c1 += int(g == 1)
        c2 += int(g == 2)
        c3plus += int(g >= 3)
        vote_correct += is_vote_correct
        if g == 2:
            c2_vote_correct += is_vote_correct
            c2_strict += int(margin > 0)
            c2_tie += int(margin == 0)
            c2_wrong.append(h)
        margins.append(margin)
    per_agent_counts = [sum(int(value) for value in vector) for vector in correctness]
    invalid_count = sum(int(value) for vector in invalids for value in vector)
    for left, right in itertools.combinations(range(len(prompt_profiles)), 2):
        left_embeddings = list(prompt_profiles[left].get("trace_embedding_vector_per_question", []))
        right_embeddings = list(prompt_profiles[right].get("trace_embedding_vector_per_question", []))
        for a, b in zip(left_embeddings, right_embeddings):
            if not a or not b or len(a) != len(b):
                continue
            dot = sum(float(x) * float(y) for x, y in zip(a, b))
            left_norm = math.sqrt(sum(float(x) * float(x) for x in a))
            right_norm = math.sqrt(sum(float(y) * float(y) for y in b))
            if left_norm > 0 and right_norm > 0:
                trace_distances.append(max(0.0, min(1.0, 1.0 - dot / (left_norm * right_norm))))
    return {
        "vote_correct_count": vote_correct,
        "total_agent_correct_count": sum(per_agent_counts),
        "bottom2_correct_count": sum(sorted(per_agent_counts)[: min(2, len(per_agent_counts))]),
        "per_agent_correct_count": per_agent_counts,
        "c0_count": c0,
        "c1_count": c1,
        "c2_count": c2,
        "c3plus_count": c3plus,
        "coverage_depth_c2": sum(g >= 2 for g in [
            sum(int(vector[index]) if index < len(vector) else 0 for vector in correctness)
            for index in range(len(gold_answers))
        ]),
        "c2_vote_correct_count": c2_vote_correct,
        "c2_strict_vote_correct_count": c2_strict,
        "c2_tie_count": c2_tie,
        "mean_gold_plurality_margin": sum(margins) / max(1, len(margins)),
        "c2_mean_largest_wrong_vote": sum(c2_wrong) / max(1, len(c2_wrong)),
        "invalid_count": invalid_count,
        "trace_diversity_tiebreak": sum(trace_distances) / max(1, len(trace_distances)),
    }


def select_state_conditioned_team(
    teams: Sequence[Mapping[str, Any]], config: Any, *, probe_size: int, num_agents: int
) -> Dict[str, Any]:
    if not teams:
        raise ValueError("state-conditioned team selection requires at least one team")
    best_total = max(int(team.get("total_agent_correct_count", 0) or 0) for team in teams)
    slack = int(round(max(0, int(probe_size)) * max(1, int(num_agents)) * float(config.state_joint_total_correct_slack_rate)))
    band = [team for team in teams if int(team.get("total_agent_correct_count", 0) or 0) >= best_total - slack]
    trace_enabled = bool(getattr(config, "state_trace_tiebreak_enabled", True))
    coverage_enabled = bool(getattr(config, "state_coverage_enabled", True))
    c2_split_enabled = bool(getattr(config, "state_c2_wrong_split_enabled", True))

    def key(team: Mapping[str, Any]) -> tuple:
        coverage_key = (
            -int(team.get("c0_count", 0) or 0),
            int(team.get("vote_correct_count", 0) or 0),
            int(team.get("coverage_depth_c2", 0) or 0),
        ) if coverage_enabled else (0, 0, 0)
        conversion_key = (
            int(team.get("c2_strict_vote_correct_count", 0) or 0),
            int(team.get("c2_vote_correct_count", 0) or 0),
            -float(team.get("c2_mean_largest_wrong_vote", 0.0) or 0.0),
        ) if c2_split_enabled else (0, 0, 0.0)
        return (
            *coverage_key,
            *conversion_key,
            int(team.get("bottom2_correct_count", 0) or 0),
            float(team.get("mean_gold_plurality_margin", 0.0) or 0.0),
            -int(team.get("invalid_count", 0) or 0),
            float(team.get("trace_diversity_tiebreak", 0.0) or 0.0) if trace_enabled else 0.0,
            json.dumps(team.get("prompt_hashes", []), separators=(",", ":")),
        )

    selected = max(band, key=key)
    return {
        "selected": dict(selected),
        "joint_best_total_correct": best_total,
        "joint_total_correct_slack": slack,
        "joint_quality_band_count": len(band),
        "joint_selection_key": list(key(selected)),
    }


def state_conditioned_validation_key(epoch_record: Mapping[str, Any]) -> tuple:
    val = epoch_record.get("val", {}) if isinstance(epoch_record.get("val", {}), Mapping) else {}
    c2_count = int(val.get("c2_count", val.get("correct_agent_count_2", 0)) or 0)
    c2_vote_correct = int(val.get("c2_vote_correct_count", 0) or 0)
    c2_rate = c2_vote_correct / max(1, c2_count)
    return (
        -float(val.get("plurality_vote_acc", val.get("vote_acc", 0.0)) or 0.0),
        -float(val.get("mean_individual_acc", 0.0) or 0.0),
        float(val.get("c0_rate", val.get("all_wrong_rate", 1.0)) or 0.0),
        -float(c2_rate),
        -float(val.get("bottom2_mean_acc", 0.0) or 0.0),
        -float(val.get("mean_gold_plurality_margin", val.get("mean_plurality_margin_votes", 0.0)) or 0.0),
        float(val.get("mean_invalid_rate", 0.0) or 0.0),
        int(epoch_record.get("epoch", 0) or 0),
    )


def state_dataset_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    size = len(rows)
    counts = {name: 0 for name in STATE_NAMES}
    c2_vote_correct = c2_vote_fail = c2_strict = c2_tie = 0
    all_same_wrong = 0
    c2_strictly_rescuable = c2_tie_only_rescuable = c2_unrescuable = 0
    c2_largest_wrong_total = 0
    state_by_question_hash = {}
    for row in rows:
        g = int(row.get("gold_vote_count", sum(int(value) for value in row.get("individual_correct", []))) or 0)
        h = int(row.get("largest_wrong_vote_count", 0) or 0)
        state = question_state(g)
        counts[state] += 1
        question_hash = str(row.get("question_hash", ""))
        if question_hash:
            state_by_question_hash[question_hash] = state
        if g == 2:
            vote = int(bool(row.get("vote_correct", False)))
            c2_vote_correct += vote
            c2_vote_fail += 1 - vote
            c2_strict += int(g - h > 0)
            c2_tie += int(g - h == 0)
            c2_largest_wrong_total += h
            rescuability = c2_dispersion_rescuability(row.get("option_count", 0))
            c2_strictly_rescuable += int(rescuability["c2_strictly_rescuable_by_dispersion"])
            c2_tie_only_rescuable += int(rescuability["c2_tie_only_rescuable_by_dispersion"])
            c2_unrescuable += int(rescuability["c2_unrescuable_by_dispersion"])
        answers = list(row.get("vote_counts", {}).values()) if isinstance(row.get("vote_counts", {}), Mapping) else []
        all_same_wrong += int(g == 0 and bool(answers) and max([int(value) for value in answers], default=0) == sum(int(value) for value in answers))
    return {
        "c0_count": counts["C0"],
        "c1_count": counts["C1"],
        "c2_count": counts["C2"],
        "c3plus_count": counts["C3PLUS"],
        "c0_rate": counts["C0"] / max(1, size),
        "c1_rate": counts["C1"] / max(1, size),
        "c2_rate": counts["C2"] / max(1, size),
        "c3plus_rate": counts["C3PLUS"] / max(1, size),
        "c2_vote_correct_count": c2_vote_correct,
        "c2_vote_fail_count": c2_vote_fail,
        "c2_strict_win_count": c2_strict,
        "c2_tie_count": c2_tie,
        "c2_mean_largest_wrong_vote": c2_largest_wrong_total / max(1, counts["C2"]),
        "c2_strictly_rescuable_count": c2_strictly_rescuable,
        "c2_tie_only_rescuable_count": c2_tie_only_rescuable,
        "c2_unrescuable_by_dispersion_count": c2_unrescuable,
        "all_agents_same_wrong_count": all_same_wrong,
        "all_agents_same_wrong_rate": all_same_wrong / max(1, size),
        "persistent_c0_count": 0,
        "new_c0_count": 0,
        "resolved_c0_count": 0,
        "state_by_question_hash": state_by_question_hash,
    }


def paired_c0_metrics(
    initial_state_by_question_hash: Mapping[str, str],
    current_state_by_question_hash: Mapping[str, str],
) -> Dict[str, int]:
    shared = sorted(set(initial_state_by_question_hash) & set(current_state_by_question_hash))
    return {
        "persistent_c0_count": sum(
            initial_state_by_question_hash[key] == "C0" and current_state_by_question_hash[key] == "C0"
            for key in shared
        ),
        "new_c0_count": sum(
            initial_state_by_question_hash[key] != "C0" and current_state_by_question_hash[key] == "C0"
            for key in shared
        ),
        "resolved_c0_count": sum(
            initial_state_by_question_hash[key] == "C0" and current_state_by_question_hash[key] != "C0"
            for key in shared
        ),
    }
