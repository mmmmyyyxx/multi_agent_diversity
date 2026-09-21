from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from multi_dataset_diverse_rl.native_feed import Layer2OptimizationRequest
from multi_dataset_diverse_rl.team_search.schemas import (
    TeamEvidenceCase,
    TeamSearchAssignment,
    TeamSearchRequest,
)
from multi_dataset_diverse_rl.team_search.task_builder import (
    Layer2EvidenceRequestBuilder,
)


def evidence(*, context_suffix: str = "") -> tuple[TeamEvidenceCase, ...]:
    rows = []
    for group, tag in (
        ("responsibility", "direct_flip"),
        ("coalition", "near_margin"),
        ("preservation", "coverage"),
    ):
        for index in range(4):
            rows.append(
                TeamEvidenceCase(
                    f"{group}-{index}{context_suffix}",
                    f"payload {group} {index}{context_suffix}",
                    "A",
                    "B",
                    "sanitized outcome",
                    group,
                    (tag,),
                )
            )
    return tuple(rows)


def build(*, context_suffix: str = "") -> Layer2OptimizationRequest:
    assignment = TeamSearchAssignment(
        target_member=2,
        parent_prompt="Use semantic evidence.",
        evidence=evidence(context_suffix=context_suffix),
        optimization_context=f"repair direct flips{context_suffix}",
        responsibility_identity=f"responsibility{context_suffix}",
        primary_responsibility_lane="direct_flip",
        responsibility_value=8.0,
    )
    request = TeamSearchRequest(
        80,
        1,
        "team-state",
        36,
        "COMMON_SOLVER_CONTRACT_V1",
        "output-v1",
        "optimize-only-v1",
    )
    return Layer2EvidenceRequestBuilder().build(request, assignment)


def test_packet_is_deterministic_immutable_aligned_and_complete() -> None:
    left = build()
    right = build()
    assert left.identity() == right.identity()
    assert left.packet.packet_hash == right.packet.packet_hash
    assert len(left.packet.repair_examples) == 4
    assert len(left.packet.preservation_examples) == 4
    assert len(left.packet.local_eval_examples) == 4
    assert all(row.lane == "direct_flip" for row in left.packet.repair_examples)
    assert all(
        row.responsibility_role == "preservation"
        for row in left.packet.preservation_examples
    )
    role_ids = [
        {row.example_id for row in rows}
        for rows in (
            left.packet.repair_examples,
            left.packet.preservation_examples,
            left.packet.local_eval_examples,
        )
    ]
    assert not role_ids[0] & role_ids[1]
    assert not role_ids[0] & role_ids[2]
    assert not role_ids[1] & role_ids[2]
    scheduled = set().union(*(set(batch) for batch in left.packet.ordered_batch_schedule))
    assert scheduled == role_ids[0] | role_ids[1]
    before = left.packet.packet_hash
    with pytest.raises(FrozenInstanceError):
        left.packet.target_member = 4  # type: ignore[misc]
    assert left.packet.packet_hash == before


def test_layer2_packet_content_directly_changes_optimizer_input() -> None:
    packet_a = build(context_suffix="-a")
    packet_b = build(context_suffix="-b")
    assert packet_a.parent_decision_procedure == packet_b.parent_decision_procedure
    assert packet_a.seed == packet_b.seed
    assert packet_a.packet.budget == packet_b.packet.budget
    assert packet_a.packet.packet_hash != packet_b.packet.packet_hash
    assert packet_a.identity() != packet_b.identity()


def test_missing_preservation_fails_closed() -> None:
    assignment = TeamSearchAssignment(
        1,
        "Use semantic evidence.",
        tuple(row for row in evidence() if row.evidence_group != "preservation"),
        "repair direct flips",
        "responsibility",
        primary_responsibility_lane="direct_flip",
        responsibility_value=4.0,
    )
    request = TeamSearchRequest(
        80, 1, "team", 36, "solver", "output", "optimize-only-v1"
    )
    with pytest.raises(ValueError, match="no preservation examples"):
        Layer2EvidenceRequestBuilder().build(request, assignment)


def test_heldout_rows_are_rejected_before_packet_creation() -> None:
    with pytest.raises(ValueError, match="Optimize-derived"):
        TeamEvidenceCase(
            "heldout",
            "private heldout payload",
            "A",
            None,
            None,
            "responsibility",
            ("direct_flip",),
            source_split="validation",
        )
