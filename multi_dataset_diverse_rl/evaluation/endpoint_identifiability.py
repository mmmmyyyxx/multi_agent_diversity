"""Zero-API structural identifiability checks for plurality endpoints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..utils import plurality_vote_with_diagnostics


def plurality_outcome(answers: Sequence[str | None]) -> str:
    """Return the frozen tie-as-abstain plurality answer; invalid is an empty vote."""
    normalized = ["" if answer is None else str(answer) for answer in answers]
    return str(plurality_vote_with_diagnostics(
        normalized, tie_break_method="abstain"
    )["vote_answer"])


def target_row_is_plurality_pivotal_capable(
    peer_answers: Sequence[str | None],
    *,
    legal_target_outputs: Sequence[str | None],
) -> bool:
    """Whether two legal target outputs can produce different plurality answers."""
    outcomes = {
        plurality_outcome([*peer_answers, candidate])
        for candidate in legal_target_outputs
    }
    return len(outcomes) >= 2


def endpoint_structural_identifiability(
    profiles: Sequence[Sequence[str | None]],
    *,
    legal_target_outputs_by_row: Sequence[Sequence[str | None]],
) -> dict[str, Any]:
    """Enumerate per-member rows where changing only that member can alter plurality."""
    if len(profiles) != 5:
        raise ValueError("exactly five member profiles required")
    row_count = len(profiles[0])
    if any(len(profile) != row_count for profile in profiles):
        raise ValueError("member profile lengths differ")
    if len(legal_target_outputs_by_row) != row_count:
        raise ValueError("legal output domains do not match profile rows")
    by_member: dict[str, dict[str, Any]] = {}
    total = 0
    for target in range(5):
        capable_rows = []
        for row_index in range(row_count):
            peers = [profiles[member][row_index] for member in range(5) if member != target]
            if target_row_is_plurality_pivotal_capable(
                peers, legal_target_outputs=legal_target_outputs_by_row[row_index]
            ):
                capable_rows.append(row_index)
        total += len(capable_rows)
        by_member[str(target)] = {
            "pivotal_capable_count": len(capable_rows),
            "pivotal_capable_row_indices": capable_rows,
        }
    return {
        "row_count": row_count,
        "member_count": 5,
        "tie_break": "abstain",
        "by_member": by_member,
        "total_member_row_opportunities": total,
        "endpoint_structurally_identifiable": total > 0,
    }
