"""Real-team quality anchor frontier for Stable-QD joint selection."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

from ..core.models import QualityAnchor, QualityCounts


def quality_counts(team: dict[str, Any]) -> QualityCounts:
    return QualityCounts.from_dict(team)


def build_quality_anchor(team: dict[str, Any], *, epoch: int, created_order: int) -> QualityAnchor:
    prompt_hashes = [str(value) for value in team.get("prompt_hashes", [])]
    if not prompt_hashes:
        prompt_hashes = [str(item.get("prompt_hash", "")) for item in team.get("prompt_profiles", [])]
    digest = hashlib.sha256(json.dumps([int(epoch), prompt_hashes], separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    return QualityAnchor(
        anchor_id=f"team:{digest}", epoch=int(epoch), prompt_hashes=prompt_hashes,
        counts=quality_counts(team), created_order=int(created_order),
    )


def dominates(left: QualityAnchor, right: QualityAnchor) -> bool:
    left_values = _quality_vector(left.counts)
    right_values = _quality_vector(right.counts)
    return all(a >= b for a, b in zip(left_values, right_values)) and any(a > b for a, b in zip(left_values, right_values))


def _quality_vector(counts: QualityCounts) -> tuple[int, ...]:
    return (
        counts.vote, counts.total_agent_correct, counts.bottom2_correct,
        counts.c1, counts.c2, *counts.per_agent_correct,
    )


def update_quality_anchor_archive(
    archive: Sequence[QualityAnchor | dict[str, Any]],
    new_anchors: Iterable[QualityAnchor],
    *,
    capacity: int = 5,
) -> list[QualityAnchor]:
    by_team: dict[tuple[str, ...], QualityAnchor] = {}
    for value in [*archive, *new_anchors]:
        anchor = value if isinstance(value, QualityAnchor) else QualityAnchor.from_dict(value)
        key = tuple(anchor.prompt_hashes)
        current = by_team.get(key)
        if current is None or anchor.created_order >= current.created_order:
            by_team[key] = anchor
    values = list(by_team.values())
    frontier = [anchor for anchor in values if not any(other is not anchor and dominates(other, anchor) for other in values)]
    if len(frontier) <= int(capacity):
        return sorted(frontier, key=lambda item: (item.created_order, item.anchor_id))

    selectors = (
        lambda item: (item.counts.vote, item.created_order),
        lambda item: (item.counts.total_agent_correct, sum(item.counts.per_agent_correct), item.created_order),
        lambda item: (item.counts.bottom2_correct, item.created_order),
        lambda item: (min(item.counts.c1, item.counts.c2), item.counts.c1 + item.counts.c2, item.created_order),
        lambda item: (item.created_order,),
    )
    retained: list[QualityAnchor] = []
    for selector in selectors:
        chosen = max(frontier, key=selector)
        if chosen not in retained:
            retained.append(chosen)
        if len(retained) >= int(capacity):
            break
    for item in sorted(frontier, key=lambda value: (value.created_order, value.anchor_id), reverse=True):
        if len(retained) >= int(capacity):
            break
        if item not in retained:
            retained.append(item)
    return sorted(retained[: int(capacity)], key=lambda item: (item.created_order, item.anchor_id))


def counts_feasible(candidate: QualityCounts, floor: QualityCounts, config: Any) -> bool:
    if candidate.vote < floor.vote - int(config.joint_allowed_vote_loss_questions): return False
    if candidate.total_agent_correct < floor.total_agent_correct - int(config.joint_allowed_total_agent_correct_loss): return False
    if candidate.bottom2_correct < floor.bottom2_correct - int(config.joint_allowed_bottom2_correct_loss): return False
    if candidate.c1 < floor.c1 - int(config.joint_allowed_c1_loss_questions): return False
    if candidate.c2 < floor.c2 - int(config.joint_allowed_c2_loss_questions): return False
    return all(
        value >= (floor.per_agent_correct[index] if index < len(floor.per_agent_correct) else 0)
        - int(config.joint_allowed_per_agent_correct_loss)
        for index, value in enumerate(candidate.per_agent_correct)
    )


def real_anchor_feasible(candidate: dict[str, Any], anchors: Sequence[QualityAnchor | dict[str, Any]], config: Any) -> bool:
    counts = quality_counts(candidate)
    return any(counts_feasible(counts, anchor.counts if isinstance(anchor, QualityAnchor) else QualityAnchor.from_dict(anchor).counts, config) for anchor in anchors)
