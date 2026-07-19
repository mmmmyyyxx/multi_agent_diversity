import itertools
import json
import hashlib
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .behavior_profiles import behavior_distance, build_team_behavior_profiles
from .lineage import lineage_drift
from .mechanisms import mechanism_distance, mechanism_niche_key, mechanisms_are_near_duplicate


QUALITY_KEYS = ("vote_acc", "mean_individual_acc", "bottom2_mean_acc", "coverage_depth_c1", "coverage_depth_c2")


def _candidate_quality_key(item: Dict[str, Any]) -> tuple:
    metrics = item.get("metrics", {})
    return (
        float(metrics.get("candidate_target_accuracy", metrics.get("target_agent_accuracy", 0.0)) or 0.0),
        float(metrics.get("depth1_net_delta", 0.0) or 0.0),
        float(metrics.get("depth2_net_delta", 0.0) or 0.0),
        float(metrics.get("plurality_vote_gain_rate", metrics.get("vote_gain_rate", 0.0)) or 0.0),
        float(metrics.get("penalized_reward", item.get("reward", 0.0)) or 0.0),
        -int(item.get("generation", 0) or 0),
        str(item.get("candidate_id", item.get("id", ""))),
    )


def select_quality_diversity_archive(
    candidates: Sequence[Dict[str, Any]],
    beam_size: int,
    incumbent_prompt_hash: str,
    config: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    feasible = [item for item in candidates if not str(item.get("metrics", {}).get("rejection_reason", ""))]
    by_niche: Dict[tuple, Dict[str, Any]] = {}
    for item in feasible:
        representation = dict(item.get("metrics", {}).get("mechanism_representation", {}) or {})
        niche = mechanism_niche_key(representation)
        item["qd_niche_key"] = [niche[0], list(niche[1])]
        item.setdefault("metrics", {})["qd_niche_key"] = list(item["qd_niche_key"])
        incumbent = by_niche.get(niche)
        if incumbent is None or _candidate_quality_key(item) > _candidate_quality_key(incumbent):
            by_niche[niche] = item
    elites = list(by_niche.values())
    incumbent = next((item for item in feasible if str(item.get("prompt_hash", "")) == incumbent_prompt_hash), None)
    if incumbent is None:
        incumbent = max(elites, key=_candidate_quality_key, default=None)
    retained: List[Dict[str, Any]] = []
    if incumbent is not None:
        incumbent["beam_slot"] = "incumbent"
        incumbent.setdefault("metrics", {})["beam_slot"] = "incumbent"
        retained.append(incumbent)
    different = [item for item in elites if item not in retained and item.get("qd_niche_key") != (retained[0].get("qd_niche_key") if retained else None)]
    repair_candidates = [
        item for item in different
        if str(item.get("metrics", {}).get("candidate_type", item.get("candidate_type", ""))) == "task_specific_repair"
    ]
    repair = max(repair_candidates or different, key=_candidate_quality_key, default=None)
    if repair is not None:
        repair["beam_slot"] = "task_repair_niche"
        repair.setdefault("metrics", {})["beam_slot"] = "task_repair_niche"
        retained.append(repair)
    remaining = [item for item in elites if item not in retained]
    if remaining and len(retained) < beam_size:
        def distinct_key(item):
            rep = item.get("metrics", {}).get("mechanism_representation", {})
            distances = [
                mechanism_distance(
                    rep,
                    kept.get("metrics", {}).get("mechanism_representation", {}),
                    sequence_weight=float(getattr(config, "mechanism_sequence_distance_weight", 0.5)),
                    embedding_weight=float(getattr(config, "mechanism_embedding_distance_weight", 0.5)),
                )["mechanism_distance"]
                for kept in retained
            ]
            candidate_behavior = item.get("metrics", {}).get("behavior_profile", {})
            behavior_distances = [
                behavior_distance(
                    candidate_behavior,
                    kept.get("metrics", {}).get("behavior_profile", {}),
                    correct_set_weight=float(getattr(config, "behavior_correct_set_weight", 0.50)),
                    rescue_weight=float(getattr(config, "behavior_rescue_weight", 0.35)),
                    shared_wrong_weight=float(getattr(config, "behavior_error_overlap_weight", 0.15)),
                    wrong_answer_dispersion_weight=float(getattr(config, "behavior_wrong_answer_dispersion_weight", 0.15)),
                    support_shrinkage=float(getattr(config, "behavior_support_shrinkage", 5.0)),
                )["behavior_distance"]
                for kept in retained
                if candidate_behavior and kept.get("metrics", {}).get("behavior_profile")
            ]
            return (sum(distances) + sum(behavior_distances), _candidate_quality_key(item))
        distinct = max(remaining, key=distinct_key)
        distinct["beam_slot"] = "mechanism_niche"
        distinct.setdefault("metrics", {})["beam_slot"] = "mechanism_niche"
        retained.append(distinct)
    for item in sorted(feasible, key=_candidate_quality_key, reverse=True):
        if len(retained) >= beam_size:
            break
        representation = item.get("metrics", {}).get("mechanism_representation", {})
        if (
            item not in retained
            and all(item.get("prompt_hash") != kept.get("prompt_hash") for kept in retained)
            and not any(
                mechanisms_are_near_duplicate(
                    representation,
                    kept.get("metrics", {}).get("mechanism_representation", {}),
                    float(getattr(config, "mechanism_near_duplicate_similarity_threshold", 0.97)),
                )
                for kept in retained
            )
        ):
            item["beam_slot"] = "task_repair_niche"
            item.setdefault("metrics", {})["beam_slot"] = "task_repair_niche"
            retained.append(item)
    for item in candidates:
        if item not in retained:
            item["beam_slot"] = "not_retained"
            item.setdefault("metrics", {})["beam_slot"] = "not_retained"
    return retained[:beam_size], {
        "niche_count": len(by_niche),
        "incumbent_slot_occupancy": int(any(item.get("beam_slot") == "incumbent" for item in retained)),
        "task_repair_niche_occupancy": int(any(item.get("beam_slot") == "task_repair_niche" for item in retained)),
        "mechanism_niche_occupancy": int(any(item.get("beam_slot") == "mechanism_niche" for item in retained)),
    }


def team_quality_metrics(
    prompt_profiles: Sequence[Dict[str, Any]],
    gold_answers: Sequence[str],
    question_hashes: Sequence[str],
    *,
    vote_fn: Callable[..., Dict[str, Any]],
    match_fn: Callable[[str, str], bool],
    tie_break_method: str,
    seed: int,
) -> Dict[str, Any]:
    answers = [list(profile.get("answer_vector", [])) for profile in prompt_profiles]
    correctness = [list(profile.get("correctness_vector", [])) for profile in prompt_profiles]
    per_agent = [float(np.mean(values)) if values else 0.0 for values in correctness]
    depths, votes, margins = [], [], []
    for index, gold in enumerate(gold_answers):
        row_answers = [values[index] if index < len(values) else "" for values in answers]
        row_correct = [values[index] if index < len(values) else 0 for values in correctness]
        vote = vote_fn(row_answers, tie_break_method=tie_break_method, seed=seed, question_hash=str(question_hashes[index]))
        votes.append(int(match_fn(str(vote.get("vote_answer", "")), str(gold))))
        depths.append(sum(int(value) for value in row_correct))
        counts = vote.get("vote_counts", {})
        gold_count = sum(int(match_fn(answer, str(gold))) * int(count) for answer, count in counts.items())
        wrong_count = max((int(count) for answer, count in counts.items() if not match_fn(answer, str(gold))), default=0)
        margins.append((gold_count - wrong_count) / max(len(prompt_profiles), 1))
    sorted_acc = sorted(per_agent)
    total_correct = int(sum(sum(int(value) for value in row) for row in correctness))
    bottom2_correct = int(sum(sorted([sum(int(value) for value in row) for row in correctness])[:2])) if correctness else 0
    return {
        "vote_acc": float(np.mean(votes)) if votes else 0.0,
        "mean_individual_acc": float(np.mean(per_agent)) if per_agent else 0.0,
        "bottom2_mean_acc": float(np.mean(sorted_acc[:2])) if sorted_acc else 0.0,
        "coverage_depth_c1": float(np.mean([depth >= 1 for depth in depths])) if depths else 0.0,
        "coverage_depth_c2": float(np.mean([depth >= 2 for depth in depths])) if depths else 0.0,
        "coverage_depth_c1_correct_count": int(sum(depth >= 1 for depth in depths)),
        "coverage_depth_c2_correct_count": int(sum(depth >= 2 for depth in depths)),
        "mean_normalized_plurality_margin": float(np.mean(margins)) if margins else 0.0,
        "per_agent_acc": per_agent,
        "vote_correct_count": int(sum(votes)),
        "total_agent_correct_count": total_correct,
        "bottom2_correct_count": bottom2_correct,
        "per_agent_correct_count": [int(sum(int(value) for value in row)) for row in correctness],
        "answer_vectors": answers,
        "correctness_vectors": correctness,
    }


def quality_feasible(
    candidate: Dict[str, Any],
    incumbent: Dict[str, Any],
    initial_per_agent: Sequence[float],
    anchor_accuracies: Sequence[float],
    epsilons: Dict[str, float],
    per_agent_epsilon: float,
) -> bool:
    if any(float(candidate.get(key, 0.0)) < float(incumbent.get(key, 0.0)) - float(epsilons.get(key, 0.0)) for key in QUALITY_KEYS):
        return False
    for index, accuracy in enumerate(candidate.get("per_agent_acc", [])):
        lower = max(
            float(initial_per_agent[index]) if index < len(initial_per_agent) else 0.0,
            float(incumbent.get("per_agent_acc", [0.0] * (index + 1))[index]),
            float(anchor_accuracies[index]) if index < len(anchor_accuracies) and anchor_accuracies[index] >= 0.0 else 0.0,
        ) - float(per_agent_epsilon)
        if float(accuracy) < lower:
            return False
    return True


def epsilon_dominates(left: Dict[str, Any], right: Dict[str, Any], epsilons: Dict[str, float]) -> bool:
    return all(float(left[key]) >= float(right[key]) - float(epsilons.get(key, 0.0)) for key in QUALITY_KEYS) and any(
        float(left[key]) > float(right[key]) + 1e-12 for key in QUALITY_KEYS
    )


def epsilon_quality_frontier(teams: Sequence[Dict[str, Any]], epsilons: Dict[str, float]) -> List[Dict[str, Any]]:
    return [team for index, team in enumerate(teams) if not any(
        other_index != index and epsilon_dominates(other, team, epsilons)
        for other_index, other in enumerate(teams)
    )]


def team_diversity_metrics(team: Dict[str, Any], config: Any) -> Dict[str, float]:
    if any(profile.get("profile_kind") == "team_combination_profile" for profile in team["prompt_profiles"]):
        raise AssertionError("team-dependent metrics cannot be loaded from prompt-level cache")
    profiles = build_team_behavior_profiles(team["answer_vectors"], team["correctness_vectors"])
    behavior_values, mechanism_values = [], []
    for left in range(len(profiles)):
        for right in range(left + 1, len(profiles)):
            behavior_values.append(behavior_distance(
                profiles[left], profiles[right],
                correct_set_weight=config.behavior_correct_set_weight,
                rescue_weight=config.behavior_rescue_weight,
                shared_wrong_weight=config.behavior_error_overlap_weight,
                wrong_answer_dispersion_weight=config.behavior_wrong_answer_dispersion_weight,
                support_shrinkage=config.behavior_support_shrinkage,
                wrong_support_shrinkage=config.behavior_wrong_support_shrinkage,
            )["behavior_distance"])
            mechanism_values.append(mechanism_distance(
                team["prompt_profiles"][left].get("mechanism_representation", {}),
                team["prompt_profiles"][right].get("mechanism_representation", {}),
                sequence_weight=config.mechanism_sequence_distance_weight,
                embedding_weight=config.mechanism_embedding_distance_weight,
            )["mechanism_distance"])
    rescue_counts = [sum(profile["rescue_vector"]) for profile in profiles]
    total_rescue = sum(rescue_counts)
    rescue_balance = 0.0 if not total_rescue else 1.0 - sum((count / total_rescue) ** 2 for count in rescue_counts)
    mean_behavior = float(np.mean(behavior_values)) if behavior_values else 0.0
    min_behavior = min(behavior_values, default=0.0)
    mean_mechanism = float(np.mean(mechanism_values)) if mechanism_values else 0.0
    score = (
        config.team_diversity_mean_behavior_weight * mean_behavior
        + config.team_diversity_min_behavior_weight * min_behavior
        + config.team_diversity_mechanism_weight * mean_mechanism
        + config.team_diversity_rescue_balance_weight * rescue_balance
    )
    return {
        "mean_behavior_distance": mean_behavior,
        "min_behavior_distance": min_behavior,
        "mean_mechanism_distance": mean_mechanism,
        "rescue_balance_score": rescue_balance,
        "team_diversity_score": float(score),
        "behavior_profiles": profiles,
    }


def enumerate_joint_teams(
    beams: Sequence[Sequence[Dict[str, Any]]],
    gold_answers: Sequence[str],
    question_hashes: Sequence[str],
    *, vote_fn, match_fn, tie_break_method: str, seed: int,
) -> List[Dict[str, Any]]:
    teams = []
    for indices in itertools.product(*[range(len(beam)) for beam in beams]):
        prompt_profiles = [beams[agent_id][beam_index] for agent_id, beam_index in enumerate(indices)]
        metrics = team_quality_metrics(
            prompt_profiles, gold_answers, question_hashes,
            vote_fn=vote_fn, match_fn=match_fn, tie_break_method=tie_break_method, seed=seed,
        )
        teams.append({"beam_indices": list(indices), "prompt_profiles": prompt_profiles, **metrics})
    return teams


def deterministic_probe_folds(question_hashes: Sequence[str], *, seed: int, seed_offset: int = 9100) -> List[List[int]]:
    ordered = sorted(
        range(len(question_hashes)),
        key=lambda index: hashlib.sha256(f"{seed + seed_offset}:{question_hashes[index]}".encode("utf-8")).hexdigest(),
    )
    return [ordered[::2], ordered[1::2]]


def team_metrics_for_indices(team: Dict[str, Any], indices: Sequence[int], gold_answers: Sequence[str], question_hashes: Sequence[str], *, vote_fn, match_fn, tie_break_method: str, seed: int) -> Dict[str, Any]:
    profiles = []
    for profile in team["prompt_profiles"]:
        profiles.append({
            **profile,
            "answer_vector": [profile["answer_vector"][index] for index in indices],
            "correctness_vector": [profile["correctness_vector"][index] for index in indices],
        })
    return team_quality_metrics(
        profiles, [gold_answers[index] for index in indices], [question_hashes[index] for index in indices],
        vote_fn=vote_fn, match_fn=match_fn, tie_break_method=tie_break_method, seed=seed,
    )


def hierarchical_quality_bands(
    teams: Sequence[Dict[str, Any]],
    incumbent: Dict[str, Any],
    config: Any,
    quality_anchor: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    def count(team: Dict[str, Any], key: str) -> int:
        size = max(len(team.get("answer_vectors", [[0]])[0]), 1)
        if key in team:
            return int(team[key])
        if key == "vote_correct_count":
            return int(round(float(team.get("vote_acc", 0.0)) * size))
        if key == "total_agent_correct_count":
            return int(round(float(team.get("mean_individual_acc", 0.0)) * size * len(team.get("per_agent_acc", []))))
        if key == "bottom2_correct_count":
            return int(round(float(team.get("bottom2_mean_acc", 0.0)) * size * 2))
        if key == "coverage_depth_c1_correct_count":
            return int(round(float(team.get("coverage_depth_c1", 0.0)) * size))
        if key == "coverage_depth_c2_correct_count":
            return int(round(float(team.get("coverage_depth_c2", 0.0)) * size))
        return 0
    floor = quality_anchor or incumbent

    def feasible(team: Dict[str, Any]) -> bool:
        if count(team, "vote_correct_count") < count(floor, "vote_correct_count") - int(config.joint_allowed_vote_loss_questions):
            return False
        if count(team, "total_agent_correct_count") < count(floor, "total_agent_correct_count") - int(config.joint_allowed_total_agent_correct_loss):
            return False
        if count(team, "bottom2_correct_count") < count(floor, "bottom2_correct_count") - int(config.joint_allowed_bottom2_correct_loss):
            return False
        if count(team, "coverage_depth_c1_correct_count") < count(floor, "coverage_depth_c1_correct_count") - int(config.joint_allowed_c1_loss_questions):
            return False
        if count(team, "coverage_depth_c2_correct_count") < count(floor, "coverage_depth_c2_correct_count") - int(config.joint_allowed_c2_loss_questions):
            return False
        team_counts = team.get("per_agent_correct_count", [
            int(round(value * max(len(team.get("answer_vectors", [[0]])[0]), 1)))
            for value in team.get("per_agent_acc", [])
        ])
        incumbent_counts = floor.get("per_agent_correct_count", [
            int(round(value * max(len(incumbent.get("answer_vectors", [[0]])[0]), 1)))
            for value in floor.get("per_agent_acc", [])
        ])
        return all(
            value >= (incumbent_counts[index] if index < len(incumbent_counts) else 0) - int(config.joint_allowed_per_agent_correct_loss)
            for index, value in enumerate(team_counts)
        )
    levels = [[team for team in teams if feasible(team)]]
    rules = (
        ("vote_correct_count", int(config.joint_vote_band_questions)),
        ("total_agent_correct_count", int(config.joint_mean_band_correct_count)),
        ("bottom2_correct_count", int(config.joint_bottom2_band_correct_count)),
        ("c1_correct_count", int(config.joint_c1_band_questions)),
        ("c2_correct_count", int(config.joint_c2_band_questions)),
    )
    for key, band in rules:
        previous = levels[-1]
        if not previous:
            levels.append(previous)
            continue
        values = [
            int(round(team["coverage_depth_c1"] * len(team["answer_vectors"][0]))) if key == "c1_correct_count"
            else int(round(team["coverage_depth_c2"] * len(team["answer_vectors"][0]))) if key == "c2_correct_count"
            else count(team, key)
            for team in previous
        ]
        best = max(values)
        narrowed = [team for team, value in zip(previous, values) if best - value <= band]
        levels.append(narrowed or previous)
    return {
        "quality_floor": levels[0],
        "bands": levels[1:],
        "band_names": ["vote", "mean", "bottom2", "c1", "c2"],
        "final": levels[-1] or [incumbent],
    }


def deterministic_team_key(team: Dict[str, Any]) -> tuple:
    return (
        float(team.get("stable_team_score", 0.0)),
        float(team.get("vote_acc", 0.0)),
        float(team.get("mean_individual_acc", 0.0)),
        float(team.get("bottom2_mean_acc", 0.0)),
        float(team.get("coverage_depth_c1", 0.0)),
        float(team.get("coverage_depth_c2", 0.0)),
        -int(team.get("active_prompt_changed_count", 0)),
        json.dumps(team.get("prompt_hashes", []), separators=(",", ":")),
    )


def fold_specialization_profiles(fold_metrics: Dict[str, Any]) -> List[List[float]]:
    """Summarize per-agent specialization relative to peers on one fold."""
    if not fold_metrics:
        return []
    correctness = [list(values) for values in fold_metrics.get("correctness_vectors", [])]
    agent_count = len(correctness)
    profiles: List[List[float]] = []
    for agent_id, own in enumerate(correctness):
        residuals, rescues, unique = [], [], []
        for index, own_correct in enumerate(own):
            peer_values = [
                int(correctness[peer][index])
                for peer in range(agent_count)
                if peer != agent_id and index < len(correctness[peer])
            ]
            peer_mean = float(np.mean(peer_values)) if peer_values else 0.0
            peer_correct = sum(peer_values)
            residuals.append(float(own_correct) - peer_mean)
            rescues.append(int(own_correct and peer_correct <= 1))
            unique.append(int(own_correct and peer_correct == 0))
        profiles.append([
            float(np.mean(residuals)) if residuals else 0.0,
            float(np.mean(rescues)) if rescues else 0.0,
            float(np.mean(unique)) if unique else 0.0,
        ])
    return profiles


def team_prompt_hashes(team: Dict[str, Any]) -> List[str]:
    return [str(profile.get("prompt_hash", "")) for profile in team.get("prompt_profiles", [])]


def active_prompt_change_count(team: Dict[str, Any], incumbent: Dict[str, Any]) -> int:
    return sum(left != right for left, right in zip(team_prompt_hashes(team), team_prompt_hashes(incumbent)))


def fold_quality_feasible(candidate: Dict[str, Any], incumbent: Dict[str, Any], config: Any) -> bool:
    """Reject only catastrophic fold regressions, using integer evidence counts."""
    if int(candidate.get("total_agent_correct_count", 0)) < int(incumbent.get("total_agent_correct_count", 0)) - int(config.joint_allowed_total_agent_correct_loss):
        return False
    size = max(len(candidate.get("answer_vectors", [[0]])[0]), 1)
    incumbent_size = max(len(incumbent.get("answer_vectors", [[0]])[0]), 1)
    candidate_c1 = int(round(float(candidate.get("coverage_depth_c1", 0.0)) * size))
    incumbent_c1 = int(round(float(incumbent.get("coverage_depth_c1", 0.0)) * incumbent_size))
    candidate_c2 = int(round(float(candidate.get("coverage_depth_c2", 0.0)) * size))
    incumbent_c2 = int(round(float(incumbent.get("coverage_depth_c2", 0.0)) * incumbent_size))
    return (
        candidate_c1 >= incumbent_c1 - int(config.joint_allowed_c1_loss_questions)
        and candidate_c2 >= incumbent_c2 - int(config.joint_allowed_c2_loss_questions)
    )


def select_stable_joint_team(
    teams: Sequence[Dict[str, Any]],
    incumbent: Dict[str, Any],
    initial_per_agent: Sequence[float],
    lineage_states: Sequence[Dict[str, Any]],
    probe_size: int,
    config: Any,
    *,
    gold_answers: Sequence[str] = (),
    question_hashes: Sequence[str] = (),
    vote_fn=None,
    match_fn=None,
    tie_break_method: str = "random",
    seed: int = 0,
    change_limit: int | None = None,
    quality_anchor: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    change_rejected = 0
    eligible = []
    for team in teams:
        changes = active_prompt_change_count(team, incumbent)
        if change_limit is not None and changes > int(change_limit):
            change_rejected += 1
            continue
        eligible.append(team)
    bands = hierarchical_quality_bands(eligible, incumbent, config, quality_anchor=quality_anchor)
    feasible = bands["quality_floor"]
    frontier = bands["final"]
    stable_frontier = []
    hard_rejections = 0
    fold_quality_rejections = 0
    folds = (
        deterministic_probe_folds(question_hashes, seed=seed, seed_offset=int(config.probe_stability_seed_offset))
        if gold_answers and question_hashes and vote_fn is not None and match_fn is not None
        else []
    )
    incumbent_fold_metrics = [
        team_metrics_for_indices(
            incumbent, indices, gold_answers, question_hashes,
            vote_fn=vote_fn, match_fn=match_fn, tie_break_method=tie_break_method, seed=seed,
        )
        for indices in folds if indices
    ]
    for team in frontier:
        diversity = team_diversity_metrics(team, config)
        team.update(diversity)
        if gold_answers and question_hashes and vote_fn is not None and match_fn is not None:
            fold_diversities = []
            fold_metrics_rows = []
            fold_quality_passed = True
            incumbent_fold_index = 0
            for indices in folds:
                if not indices:
                    fold_diversities.append(0.0)
                    continue
                fold_metrics = team_metrics_for_indices(
                    team, indices, gold_answers, question_hashes,
                    vote_fn=vote_fn, match_fn=match_fn, tie_break_method=tie_break_method, seed=seed,
                )
                fold_metrics_rows.append(fold_metrics)
                fold_quality_passed = fold_quality_passed and fold_quality_feasible(
                    fold_metrics, incumbent_fold_metrics[incumbent_fold_index], config,
                )
                incumbent_fold_index += 1
                fold_team = {**fold_metrics, "prompt_profiles": team["prompt_profiles"]}
                fold_diversities.append(team_diversity_metrics(fold_team, config)["team_diversity_score"])
            team["fold_quality_metrics"] = fold_metrics_rows
            team["fold_quality_gate_passed"] = bool(fold_quality_passed)
            if len(fold_metrics_rows) >= 2:
                left_profiles = fold_specialization_profiles(fold_metrics_rows[0])
                right_profiles = fold_specialization_profiles(fold_metrics_rows[1])
                team["per_agent_fold_specialization_profiles"] = [left_profiles, right_profiles]
                team["per_agent_cross_fold_behavior_gap"] = [
                    float(np.mean(np.abs(np.asarray(left_profiles[index]) - np.asarray(right_profiles[index]))))
                    for index in range(min(len(left_profiles), len(right_profiles)))
                ]
            else:
                team["per_agent_fold_specialization_profiles"] = []
                team["per_agent_cross_fold_behavior_gap"] = [0.0 for _ in team.get("prompt_profiles", [])]
            team["fold_diversities"] = fold_diversities
            team["cross_fold_diversity_mean"] = float(np.mean(fold_diversities)) if fold_diversities else 0.0
            team["cross_fold_diversity_gap"] = float(max(fold_diversities) - min(fold_diversities)) if len(fold_diversities) >= 2 else 0.0
            team["stable_diversity_score"] = team["cross_fold_diversity_mean"] - 0.50 * team["cross_fold_diversity_gap"]
            if not fold_quality_passed:
                fold_quality_rejections += 1
                continue
        else:
            team["fold_quality_metrics"] = []
            team["fold_quality_gate_passed"] = True
            team["per_agent_fold_specialization_profiles"] = []
            team["per_agent_cross_fold_behavior_gap"] = [0.0 for _ in team.get("prompt_profiles", [])]
            team["fold_diversities"] = []
            team["cross_fold_diversity_mean"] = team["team_diversity_score"]
            team["cross_fold_diversity_gap"] = 0.0
            team["stable_diversity_score"] = team["team_diversity_score"]
        lineage_penalties, hard_jump = [], False
        for agent_id, profile in enumerate(team["prompt_profiles"]):
            profile["behavior_profile"] = diversity["behavior_profiles"][agent_id]
            drift = lineage_drift(profile, lineage_states[agent_id], config)
            profile["lineage_drift"] = drift
            lineage_penalties.append(drift["lineage_drift_penalty"])
            state = lineage_states[agent_id]
            if state.get("lineage_status") == "committed" and drift["lineage_drift"] > config.lineage_hard_drift_threshold:
                accuracy_gain = profile["accuracy"] - float(state.get("lineage_anchor_accuracy", 0.0))
                vote_gain = team["vote_acc"] - incumbent["vote_acc"]
                hard_jump = hard_jump or (
                    accuracy_gain < config.lineage_switch_min_accuracy_gain
                    and vote_gain < config.lineage_switch_min_vote_gain
                )
        if hard_jump:
            hard_rejections += 1
            continue

        peer_penalties, peer_hard = [], False
        for left in range(len(team["prompt_profiles"])):
            for right in range(left + 1, len(team["prompt_profiles"])):
                left_profile = team["prompt_profiles"][left]
                right_profile = team["prompt_profiles"][right]
                pair_introduces_change = (
                    team["prompt_profiles"][left].get("prompt_hash") != incumbent["prompt_profiles"][left].get("prompt_hash")
                    or team["prompt_profiles"][right].get("prompt_hash") != incumbent["prompt_profiles"][right].get("prompt_hash")
                )
                if pair_introduces_change and mechanisms_are_near_duplicate(
                    left_profile["mechanism_representation"], right_profile["mechanism_representation"],
                    config.peer_collapse_hard_similarity,
                ):
                    peer_hard = peer_hard or behavior_distance(
                        diversity["behavior_profiles"][left], diversity["behavior_profiles"][right],
                        correct_set_weight=config.behavior_correct_set_weight,
                        rescue_weight=config.behavior_rescue_weight,
                        shared_wrong_weight=config.behavior_error_overlap_weight,
                        wrong_answer_dispersion_weight=config.behavior_wrong_answer_dispersion_weight,
                        support_shrinkage=config.behavior_support_shrinkage,
                    )["behavior_distance"] < 0.10

        for agent_id, profile in enumerate(team["prompt_profiles"]):
            if profile.get("prompt_hash") == incumbent["prompt_profiles"][agent_id].get("prompt_hash"):
                continue
            for peer_id, peer_state in enumerate(lineage_states):
                if peer_id == agent_id or peer_state.get("lineage_status") != "committed":
                    continue
                peer_representation = {
                    "normalized_operation_sequence": peer_state.get("lineage_anchor_mechanism_signature", []),
                    "mechanism_embedding": peer_state.get("lineage_anchor_mechanism_embedding", []),
                }
                distance = mechanism_distance(
                    profile["mechanism_representation"], peer_representation,
                    sequence_weight=config.mechanism_sequence_distance_weight,
                    embedding_weight=config.mechanism_embedding_distance_weight,
                )["mechanism_distance"]
                peer_penalties.append(max(0.0, 1.0 - distance - config.peer_collapse_soft_similarity))
                if mechanisms_are_near_duplicate(
                    profile["mechanism_representation"], peer_representation,
                    config.peer_collapse_hard_similarity,
                ):
                    peer_behavior = {
                        "correctness_vector": peer_state.get("lineage_anchor_correctness_vector", []),
                        "error_vector": [1 - int(value) for value in peer_state.get("lineage_anchor_correctness_vector", [])],
                        "rescue_vector": peer_state.get("lineage_anchor_rescue_vector", []),
                    }
                    peer_hard = peer_hard or behavior_distance(
                        diversity["behavior_profiles"][agent_id], peer_behavior,
                        correct_set_weight=config.behavior_correct_set_weight,
                        rescue_weight=config.behavior_rescue_weight,
                        shared_wrong_weight=config.behavior_error_overlap_weight,
                        wrong_answer_dispersion_weight=config.behavior_wrong_answer_dispersion_weight,
                        support_shrinkage=config.behavior_support_shrinkage,
                    )["behavior_distance"] < 0.10
        if peer_hard and team is not incumbent:
            hard_rejections += 1
            continue
        team["lineage_drift_penalty_mean"] = float(np.mean(lineage_penalties)) if lineage_penalties else 0.0
        team["peer_collapse_penalty_mean"] = float(np.mean(peer_penalties)) if peer_penalties else 0.0
        team["stable_team_score"] = (
            team["stable_diversity_score"]
            - team["lineage_drift_penalty_mean"]
            - team["peer_collapse_penalty_mean"]
        )
        team["active_prompt_changed_count"] = active_prompt_change_count(team, incumbent)
        team["prompt_hashes"] = team_prompt_hashes(team)
        stable_frontier.append(team)
    if not stable_frontier:
        incumbent.update(team_diversity_metrics(incumbent, config))
        incumbent.update({
            "lineage_drift_penalty_mean": 0.0,
            "peer_collapse_penalty_mean": 0.0,
            "stable_team_score": incumbent["team_diversity_score"],
            "cross_fold_diversity_mean": incumbent["team_diversity_score"],
            "cross_fold_diversity_gap": 0.0,
            "stable_diversity_score": incumbent["team_diversity_score"],
            "fold_quality_metrics": incumbent_fold_metrics,
            "fold_quality_gate_passed": True,
            "per_agent_cross_fold_behavior_gap": [0.0 for _ in incumbent.get("prompt_profiles", [])],
            "per_agent_fold_specialization_profiles": [],
            "active_prompt_changed_count": 0,
            "prompt_hashes": [profile["prompt_hash"] for profile in incumbent["prompt_profiles"]],
        })
        stable_frontier = [incumbent]
    selected = sorted(stable_frontier, key=lambda team: (
        -team["stable_team_score"], -team["vote_acc"], -team["mean_individual_acc"],
        -team["bottom2_mean_acc"], -team["coverage_depth_c1"], -team["coverage_depth_c2"],
        team["active_prompt_changed_count"], tuple(team["prompt_hashes"]),
    ))[0]
    return {
        "selected": selected,
        "feasible_count": len(feasible),
        "quality_floor_feasible_count": len(feasible),
        "quality_frontier_count": len(frontier),
        "final_candidate_team_count": len(frontier),
        "hierarchical_band_counts": [len(level) for level in bands["bands"]],
        "hierarchical_band_count_by_name": {
            name: len(level) for name, level in zip(bands["band_names"], bands["bands"])
        },
        "combination_rejected_by_change_limit_count": change_rejected,
        "fold_quality_rejection_count": fold_quality_rejections,
        "hard_rejection_count": hard_rejections,
        "selected_has_soft_peer_collapse": bool(float(selected.get("peer_collapse_penalty_mean", 0.0)) > 0.0),
    }
