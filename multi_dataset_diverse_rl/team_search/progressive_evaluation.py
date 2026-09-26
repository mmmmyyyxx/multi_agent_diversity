"""High-recall TeamMiniBatch12 promotion; never final acceptance."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .schemas import TeamCandidateRecord, TeamMiniBatchMetrics


def is_catastrophic(metrics: TeamMiniBatchMetrics) -> bool:
    return metrics.invalid_delta > 0 or metrics.vote_delta <= -2 or metrics.team_net_vote_delta <= -3


def has_promotion_signal(metrics: TeamMiniBatchMetrics) -> bool:
    return any(
        value > 0
        for value in (
            metrics.responsibility_delta,
            metrics.target_delta,
            metrics.vote_delta,
            metrics.broad_delta,
            metrics.team_net_vote_delta,
        )
    )


def promotion_key(record: TeamCandidateRecord) -> tuple[int, int, int, int, int, str]:
    if record.minibatch_metrics is None:
        raise ValueError("team minibatch metrics are required before promotion")
    metrics = record.minibatch_metrics
    return (
        metrics.vote_delta,
        metrics.team_net_vote_delta,
        metrics.responsibility_delta,
        metrics.target_delta,
        metrics.broad_delta,
        record.local_candidate.candidate_id,
    )


def promote_team_candidates(
    records: Sequence[TeamCandidateRecord], *, max_promoted: int = 2
) -> tuple[TeamCandidateRecord, ...]:
    if max_promoted <= 0:
        raise ValueError("team promotion budget must be positive")
    eligible = [
        row
        for row in records
        if row.minibatch_metrics is not None
        and not is_catastrophic(row.minibatch_metrics)
        and has_promotion_signal(row.minibatch_metrics)
    ]
    selected = {
        row.local_candidate.candidate_id
        for row in sorted(eligible, key=promotion_key, reverse=True)[:max_promoted]
    }
    return tuple(
        replace(row, promoted=row.local_candidate.candidate_id in selected)
        for row in records
    )
