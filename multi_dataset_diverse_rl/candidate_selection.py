from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _metrics(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("metrics", {})
    return value if isinstance(value, Mapping) else {}


def _hash(item: Mapping[str, Any]) -> str:
    return str(item.get("prompt_hash", ""))


def stage_a_multichannel_shortlist(
    candidates: Sequence[Mapping[str, Any]], *, channel_top_k: int = 2, total_budget: int | None = None,
) -> list[dict[str, Any]]:
    rows = [dict(item) for item in candidates]
    channels = (
        lambda item: (
            int(_metrics(item).get("candidate_target_correct_count", 0)),
            -int(_metrics(item).get("candidate_invalid_count", 0)), _hash(item),
        ),
        lambda item: (
            int(_metrics(item).get("net_vote_delta", 0)),
            -int(_metrics(item).get("vote_loss_count", 0)),
            float(_metrics(item).get("soft_vote_utility_delta", 0.0)), _hash(item),
        ),
        lambda item: (
            int(_metrics(item).get("coverage_gain_count", 0)),
            float(_metrics(item).get("assigned_residual_utility_delta", 0.0)),
            -int(_metrics(item).get("unique_correct_loss_count", 0)),
            -int(_metrics(item).get("pivotal_vote_correct_loss_count", 0)), _hash(item),
        ),
    )
    selected: dict[str, dict[str, Any]] = {}
    for channel in channels:
        for item in sorted(rows, key=channel, reverse=True)[:max(0, int(channel_top_k))]:
            selected.setdefault(_hash(item), item)
    result = list(selected.values())
    if total_budget is not None:
        for item in sorted(rows, key=lambda row: _hash(row)):
            if len(result) >= max(0, int(total_budget)):
                break
            if _hash(item) not in selected:
                selected[_hash(item)] = item
                result.append(item)
        result = result[:max(0, int(total_budget))]
    return result


@dataclass(frozen=True)
class ConstraintLimits:
    local_accuracy_allowance: int = 0
    global_accuracy_allowance: int = 0
    invalid_allowance: int = 0
    vote_loss_limit: int = 0
    unique_correct_loss_limit: int = 0
    pivotal_loss_limit: int = 0
    min_soft_utility_gain: float = 0.005


def candidate_is_feasible(
    metrics: Mapping[str, Any], active_metrics: Mapping[str, Any], initial_metrics: Mapping[str, Any], limits: ConstraintLimits,
) -> bool:
    correct = int(metrics.get("candidate_target_correct_count", 0))
    invalid = int(metrics.get("candidate_invalid_count", 0))
    return bool(
        correct >= int(active_metrics.get("candidate_target_correct_count", 0)) - limits.local_accuracy_allowance
        and correct >= int(initial_metrics.get("candidate_target_correct_count", 0)) - limits.global_accuracy_allowance
        and invalid <= int(active_metrics.get("candidate_invalid_count", 0)) + limits.invalid_allowance
        and int(metrics.get("vote_loss_count", 0)) <= limits.vote_loss_limit
        and int(metrics.get("unique_correct_loss_count", 0)) <= limits.unique_correct_loss_limit
        and int(metrics.get("pivotal_vote_correct_loss_count", 0)) <= limits.pivotal_loss_limit
    )


def vote_first_key(item: Mapping[str, Any]) -> tuple:
    metrics = _metrics(item)
    return (
        int(metrics.get("net_vote_delta", 0)),
        -int(metrics.get("vote_loss_count", 0)),
        float(metrics.get("soft_vote_utility_delta", 0.0)),
        int(metrics.get("coverage_gain_count", 0)),
        float(metrics.get("assigned_residual_utility_delta", 0.0)),
        int(metrics.get("candidate_target_correct_count", 0)),
        -int(metrics.get("candidate_invalid_count", 0)),
        -int(item.get("generation", 0)),
        _hash(item),
    )


def candidate_is_acceptable(item: Mapping[str, Any], incumbent: Mapping[str, Any], limits: ConstraintLimits) -> bool:
    metrics = _metrics(item)
    incumbent_metrics = _metrics(incumbent)
    if vote_first_key(item) <= vote_first_key(incumbent):
        return False
    net_vote = int(metrics.get("net_vote_delta", 0))
    vote_loss = int(metrics.get("vote_loss_count", 0))
    if net_vote > 0:
        return True
    if net_vote != 0 or vote_loss != 0:
        return False
    if float(metrics.get("soft_vote_utility_delta", 0.0)) >= limits.min_soft_utility_gain:
        return True
    return bool(
        int(metrics.get("candidate_target_correct_count", 0))
        > int(incumbent_metrics.get("candidate_target_correct_count", 0))
        and int(metrics.get("unique_correct_loss_count", 0)) == 0
        and int(metrics.get("pivotal_vote_correct_loss_count", 0)) == 0
    )
