from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Any, Callable, Sequence

from .evaluation.fixed_probe import ProbeExample, PromptAnswer
from .peer_state import build_team_vote_state


TEAM_DIFFERENTIATION_VERSION = "answer_behavior_differentiation_v1"
VOTE_TRANSITION_DECOMPOSITION_VERSION = "ghm_vote_transition_v1"


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _matrix(size: int, value: float | None = 0.0) -> list[list[float | None]]:
    return [[value for _ in range(size)] for _ in range(size)]


def _pearson_binary(left: Sequence[bool], right: Sequence[bool]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    x = [int(value) for value in left]
    y = [int(value) for value in right]
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    var_x = sum((value - mean_x) ** 2 for value in x)
    var_y = sum((value - mean_y) ** 2 for value in y)
    if var_x == 0 or var_y == 0:
        return None
    covariance = sum(
        (a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True)
    )
    return covariance / math.sqrt(var_x * var_y)


def _off_diagonal_values(matrix: Sequence[Sequence[float | None]]) -> list[float]:
    return [
        float(matrix[i][j])
        for i, j in combinations(range(len(matrix)), 2)
        if matrix[i][j] is not None
    ]


def team_behavior_metrics(
    *,
    examples: Sequence[ProbeExample],
    profiles: Sequence[Sequence[PromptAnswer]],
    normalize_answer: Callable[[Any], str],
    match_answer: Callable[[Any, Any], bool],
    tie_break: str,
    seed: int,
) -> dict[str, Any]:
    if len(profiles) != 5:
        raise ValueError("team differentiation requires exactly five profiles")
    if any(len(profile) != len(examples) for profile in profiles):
        raise ValueError("profile length must match the example count")
    n = len(examples)
    states = []
    correctness = [[] for _ in range(5)]
    normalized_answers = [[] for _ in range(5)]
    valid_vectors = [[] for _ in range(5)]
    wrong_answer_counts = [Counter() for _ in range(5)]
    terminal_invalid_count = 0
    for index, example in enumerate(examples):
        answers = [profiles[agent][index].answer for agent in range(5)]
        valid = [profiles[agent][index].valid for agent in range(5)]
        state = build_team_vote_state(
            question_hash=example.question_hash,
            gold_answer=example.gold_answer,
            answers=answers,
            valid_vector=valid,
            normalize_answer=normalize_answer,
            match_answer=match_answer,
            tie_break=tie_break,
            seed=seed,
        )
        states.append(state)
        for agent_id in range(5):
            is_correct = bool(state.team_correctness[agent_id])
            correctness[agent_id].append(is_correct)
            valid_vectors[agent_id].append(bool(valid[agent_id]))
            normalized = (
                normalize_answer(answers[agent_id]) if valid[agent_id] else "__INVALID__"
            )
            normalized_answers[agent_id].append(normalized)
            if not is_correct and valid[agent_id]:
                wrong_answer_counts[agent_id][normalized] += 1
            terminal_invalid_count += int(
                profiles[agent_id][index].terminal_invalid
            )

    correct_counts = [sum(row) for row in correctness]
    accuracies = [value / n if n else 0.0 for value in correct_counts]
    vote_correct_count = sum(state.vote_correct for state in states)
    oracle_correct_count = sum(state.gold_vote_count > 0 for state in states)
    g_values = [state.gold_vote_count for state in states]
    h_values = [state.largest_wrong_vote_count for state in states]
    m_values = [state.plurality_margin for state in states]
    boundary = [abs(value) <= 1 for value in m_values]
    concentrations = [
        h / (5 - g) for g, h in zip(g_values, h_values, strict=True) if 5 - g > 0
    ]
    boundary_concentrations = [
        h / (5 - g)
        for g, h, keep in zip(g_values, h_values, boundary, strict=True)
        if keep and 5 - g > 0
    ]
    wrong_concentrations = [
        h / (5 - g)
        for g, h, state in zip(g_values, h_values, states, strict=True)
        if not state.vote_correct and 5 - g > 0
    ]

    disagreement = _matrix(5)
    agreement = _matrix(5)
    double_fault = _matrix(5)
    same_wrong_unconditional = _matrix(5)
    same_wrong_conditional = _matrix(5)
    same_wrong_expected = _matrix(5)
    same_wrong_excess = _matrix(5)
    correlations = _matrix(5, None)
    for i in range(5):
        disagreement[i][i] = 0.0
        agreement[i][i] = 1.0
        double_fault[i][i] = sum(not value for value in correctness[i]) / n if n else 0.0
        correlations[i][i] = 1.0
        for j in range(i + 1, 5):
            unequal = sum(
                a != b
                for a, b in zip(
                    normalized_answers[i], normalized_answers[j], strict=True
                )
            )
            both_wrong = [
                not a and not b
                for a, b in zip(correctness[i], correctness[j], strict=True)
            ]
            same_wrong = [
                both_wrong[index]
                and valid_vectors[i][index]
                and valid_vectors[j][index]
                and normalized_answers[i][index] == normalized_answers[j][index]
                for index in range(n)
            ]
            both_wrong_count = sum(both_wrong)
            same_wrong_count = sum(same_wrong)
            expected = 0.0
            wrong_i = max(1, sum(not value for value in correctness[i]))
            wrong_j = max(1, sum(not value for value in correctness[j]))
            for answer in set(wrong_answer_counts[i]) | set(wrong_answer_counts[j]):
                expected += (
                    wrong_answer_counts[i][answer] / wrong_i
                ) * (
                    wrong_answer_counts[j][answer] / wrong_j
                )
            values = {
                "disagreement": unequal / n if n else 0.0,
                "agreement": 1.0 - (unequal / n if n else 0.0),
                "double_fault": both_wrong_count / n if n else 0.0,
                "same_wrong_unconditional": same_wrong_count / n if n else 0.0,
                "same_wrong_conditional": (
                    same_wrong_count / both_wrong_count if both_wrong_count else None
                ),
                "same_wrong_expected": expected,
            }
            disagreement[i][j] = disagreement[j][i] = values["disagreement"]
            agreement[i][j] = agreement[j][i] = values["agreement"]
            double_fault[i][j] = double_fault[j][i] = values["double_fault"]
            same_wrong_unconditional[i][j] = same_wrong_unconditional[j][i] = values[
                "same_wrong_unconditional"
            ]
            same_wrong_conditional[i][j] = same_wrong_conditional[j][i] = values[
                "same_wrong_conditional"
            ]
            same_wrong_expected[i][j] = same_wrong_expected[j][i] = expected
            excess = (
                values["same_wrong_conditional"] - expected
                if values["same_wrong_conditional"] is not None
                else None
            )
            same_wrong_excess[i][j] = same_wrong_excess[j][i] = excess
            correlation = _pearson_binary(correctness[i], correctness[j])
            correlations[i][j] = correlations[j][i] = correlation

    correlation_values = _off_diagonal_values(correlations)
    rho_bar = _mean(correlation_values)
    n_eff = (
        5 / (1 + 4 * rho_bar)
        if rho_bar is not None and 1 + 4 * rho_bar > 0
        else None
    )
    error_sizes = [5 - value for value in g_values]
    variance = (
        sum((value - sum(accuracies) / 5) ** 2 for value in accuracies) / 5
        if accuracies else 0.0
    )
    return {
        "version": TEAM_DIFFERENTIATION_VERSION,
        "example_count": n,
        "per_agent_correct_counts": correct_counts,
        "per_agent_accuracy": accuracies,
        "mean_member_accuracy": sum(accuracies) / 5,
        "minimum_member_accuracy": min(accuracies),
        "maximum_member_accuracy": max(accuracies),
        "member_accuracy_variance": variance,
        "team_vote_correct_count": vote_correct_count,
        "team_vote_accuracy": vote_correct_count / n if n else 0.0,
        "oracle_correct_count": oracle_correct_count,
        "oracle_accuracy": oracle_correct_count / n if n else 0.0,
        "terminal_invalid_count": terminal_invalid_count,
        "mean_G": _mean(g_values),
        "mean_H": _mean(h_values),
        "mean_M": _mean(m_values),
        "p_margin_positive": sum(value > 0 for value in m_values) / n if n else 0.0,
        "p_margin_zero": sum(value == 0 for value in m_values) / n if n else 0.0,
        "p_margin_negative": sum(value < 0 for value in m_values) / n if n else 0.0,
        "margin_histogram": dict(sorted(Counter(m_values).items())),
        "G_histogram": {str(value): g_values.count(value) for value in range(6)},
        "H_histogram": {str(value): h_values.count(value) for value in range(6)},
        "all_wrong_rate": sum(value == 0 for value in g_values) / n if n else 0.0,
        "oracle_covered_but_vote_wrong_rate": sum(
            0 < g <= h for g, h in zip(g_values, h_values, strict=True)
        ) / n if n else 0.0,
        "fragile_correct_rate": sum(value == 1 for value in m_values) / n if n else 0.0,
        "stable_correct_rate": sum(value >= 2 for value in m_values) / n if n else 0.0,
        "mean_dominant_wrong_concentration": _mean(concentrations),
        "boundary_dominant_wrong_concentration": _mean(boundary_concentrations),
        "team_wrong_dominant_wrong_concentration": _mean(wrong_concentrations),
        "answer_disagreement_matrix": disagreement,
        "exact_answer_agreement_matrix": agreement,
        "double_fault_matrix": double_fault,
        "mean_off_diagonal_double_fault": _mean(_off_diagonal_values(double_fault)),
        "maximum_pairwise_double_fault": max(_off_diagonal_values(double_fault), default=None),
        "minimum_pairwise_double_fault": min(_off_diagonal_values(double_fault), default=None),
        "same_wrong_unconditional_matrix": same_wrong_unconditional,
        "same_wrong_conditional_matrix": same_wrong_conditional,
        "same_wrong_expected_matrix": same_wrong_expected,
        "same_wrong_excess_matrix": same_wrong_excess,
        "mean_off_diagonal_same_wrong_excess": _mean(
            _off_diagonal_values(same_wrong_excess)
        ),
        "pairwise_correctness_correlation": correlations,
        "mean_pairwise_correctness_correlation": rho_bar,
        "n_eff": n_eff,
        "error_coalition_histogram": {
            str(value): error_sizes.count(value) for value in range(6)
        },
        "p_at_least_3_agents_wrong": sum(value >= 3 for value in error_sizes) / n if n else 0.0,
        "p_at_least_4_agents_wrong": sum(value >= 4 for value in error_sizes) / n if n else 0.0,
        "p_all_5_agents_wrong": sum(value == 5 for value in error_sizes) / n if n else 0.0,
        "dominant_wrong_coalition_histogram": {
            str(value): h_values.count(value) for value in range(6)
        },
    }


def vote_transition_decomposition(
    *,
    examples: Sequence[ProbeExample],
    incumbent_profiles: Sequence[Sequence[PromptAnswer]],
    candidate_profiles: Sequence[Sequence[PromptAnswer]],
    normalize_answer: Callable[[Any], str],
    match_answer: Callable[[Any, Any], bool],
    tie_break: str,
    seed: int,
) -> dict[str, Any]:
    rows = []
    for index, example in enumerate(examples):
        states = []
        for profiles in (incumbent_profiles, candidate_profiles):
            states.append(build_team_vote_state(
                question_hash=example.question_hash,
                gold_answer=example.gold_answer,
                answers=[profile[index].answer for profile in profiles],
                valid_vector=[profile[index].valid for profile in profiles],
                normalize_answer=normalize_answer,
                match_answer=match_answer,
                tie_break=tie_break,
                seed=seed,
            ))
        old, new = states
        rows.append({
            "old_G": old.gold_vote_count,
            "new_G": new.gold_vote_count,
            "old_H": old.largest_wrong_vote_count,
            "new_H": new.largest_wrong_vote_count,
            "old_M": old.plurality_margin,
            "new_M": new.plurality_margin,
            "delta_G": new.gold_vote_count - old.gold_vote_count,
            "delta_H": new.largest_wrong_vote_count - old.largest_wrong_vote_count,
            "delta_M": new.plurality_margin - old.plurality_margin,
            "old_vote_correct": old.vote_correct,
            "new_vote_correct": new.vote_correct,
            "boundary": abs(old.plurality_margin) <= 1,
        })
    gains = [row for row in rows if not row["old_vote_correct"] and row["new_vote_correct"]]
    losses = [row for row in rows if row["old_vote_correct"] and not row["new_vote_correct"]]
    def source(row: dict[str, Any]) -> str:
        if row["delta_G"] > 0 and row["delta_H"] >= 0:
            return "accuracy_driven"
        if row["delta_G"] == 0 and row["delta_H"] < 0:
            return "concentration_driven"
        if row["delta_G"] > 0 and row["delta_H"] < 0:
            return "joint"
        return "other_plurality_structure_transition"
    boundary_rows = [row for row in rows if row["boundary"]]
    return {
        "version": VOTE_TRANSITION_DECOMPOSITION_VERSION,
        "wrong_to_correct_vote_flip_count": len(gains),
        "correct_to_wrong_vote_flip_count": len(losses),
        "net_vote_flip_count": len(gains) - len(losses),
        "mean_delta_G": _mean([row["delta_G"] for row in rows]),
        "mean_delta_H": _mean([row["delta_H"] for row in rows]),
        "mean_delta_M": _mean([row["delta_M"] for row in rows]),
        "vote_gain_source_counts": dict(Counter(source(row) for row in gains)),
        "vote_loss_source_counts": dict(Counter(source(row) for row in losses)),
        "boundary_vote_gains": sum(
            not row["old_vote_correct"] and row["new_vote_correct"]
            for row in boundary_rows
        ),
        "boundary_vote_losses": sum(
            row["old_vote_correct"] and not row["new_vote_correct"]
            for row in boundary_rows
        ),
        "boundary_mean_delta_M": _mean([row["delta_M"] for row in boundary_rows]),
        "boundary_dominant_wrong_concentration_delta": _mean([
            (
                row["new_H"] / (5 - row["new_G"])
                if 5 - row["new_G"] > 0 else 0.0
            ) - (
                row["old_H"] / (5 - row["old_G"])
                if 5 - row["old_G"] > 0 else 0.0
            )
            for row in boundary_rows
        ]),
    }
