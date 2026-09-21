from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest

from multi_dataset_diverse_rl.native_feed import (
    CandidateTransitionAudit,
    Layer2OptimizationRequest,
    transition_audit_from_categorical_profiles,
)
from multi_dataset_diverse_rl.team_search.schemas import TeamEvidenceCase, TeamSearchAssignment, TeamSearchRequest
from multi_dataset_diverse_rl.team_search.system_runtime import LatestTransitionStore
from multi_dataset_diverse_rl.team_search.task_builder import Layer2EvidenceRequestBuilder


PARENT = "Use semantic evidence."
PARENT_HASH = hashlib.sha256(PARENT.encode("utf-8")).hexdigest()


def evidence(*, context_suffix: str = "") -> tuple[TeamEvidenceCase, ...]:
    rows = []
    for group, tag in (("responsibility", "direct_flip"), ("coalition", "near_margin"), ("preservation", "coverage")):
        for index in range(4):
            rows.append(TeamEvidenceCase(
                f"{group}-{index}{context_suffix}", f"payload {group} {index}{context_suffix}",
                "A", "B", "sanitized outcome", group, (tag,),
            ))
    return tuple(rows)


def transition(*, suffix: str = "") -> CandidateTransitionAudit:
    ids = tuple(row.example_id for row in evidence(context_suffix=suffix))
    parent = {row_id: False for row_id in ids}
    child = dict(parent)
    # Persistent responsibility residual: wrong before and after, therefore not focus.
    parent[f"responsibility-0{suffix}"] = False
    child[f"responsibility-0{suffix}"] = False
    parent[f"preservation-0{suffix}"] = True
    child[f"preservation-0{suffix}"] = False
    parent[f"preservation-1{suffix}"] = False
    child[f"preservation-1{suffix}"] = True
    parent[f"preservation-2{suffix}"] = True
    child[f"preservation-2{suffix}"] = True
    return CandidateTransitionAudit(
        parent_candidate_hash=f"lineage-parent{suffix}", child_candidate_hash=PARENT_HASH,
        parent_correctness=tuple(parent.items()), child_correctness=tuple(child.items()),
    )


def build(*, context_suffix: str = "", latest: CandidateTransitionAudit | None = None) -> Layer2OptimizationRequest:
    assignment = TeamSearchAssignment(
        target_member=2, parent_prompt=PARENT, evidence=evidence(context_suffix=context_suffix),
        optimization_context=f"repair direct flips{context_suffix}",
        responsibility_identity=f"responsibility{context_suffix}",
        primary_responsibility_lane="direct_flip", responsibility_value=8.0,
        latest_transition=latest,
    )
    request = TeamSearchRequest(
        80, 1, "team-state", 36, "COMMON_SOLVER_CONTRACT_V1", "output-v1", "optimize-only-v1",
    )
    return Layer2EvidenceRequestBuilder().build(request, assignment)


def test_root_packet_is_deterministic_immutable_aligned_and_has_no_transition() -> None:
    left = build()
    right = build()
    assert left.identity() == right.identity()
    assert left.packet.packet_hash == right.packet.packet_hash
    assert len(left.packet.responsibility_examples) == 4
    assert left.packet.focus_examples == ()
    assert left.packet.anchor_examples == ()
    assert len(left.packet.local_eval_examples) == 4
    assert all(row.lane == "direct_flip" for row in left.packet.responsibility_examples)
    assert left.packet.lineage_parent_hash is None
    scheduled = set().union(*(set(batch) for batch in left.packet.ordered_batch_schedule))
    assert scheduled == {row.packet_item_id for row in left.packet.responsibility_examples}
    before = left.packet.packet_hash
    with pytest.raises(FrozenInstanceError):
        left.packet.target_member = 4  # type: ignore[misc]
    assert left.packet.packet_hash == before


def test_transition_roles_distinguish_responsibility_focus_and_anchor() -> None:
    packet = build(latest=transition()).packet
    responsibility_ids = {row.example_id for row in packet.responsibility_examples}
    focus_ids = {row.example_id for row in packet.focus_examples}
    anchor_ids = {row.example_id for row in packet.anchor_examples}
    assert "responsibility-0" in responsibility_ids
    assert "responsibility-0" not in focus_ids
    assert focus_ids == {"preservation-0"}
    assert anchor_ids == {"preservation-1"}
    assert packet.latest_transition is not None
    assert packet.latest_transition.unchanged_correct_count == 1
    assert packet.parent_candidate_hash == PARENT_HASH


def test_transition_is_reproducible_from_categorical_profiles() -> None:
    expected = transition()
    parent = tuple({"example_id_hash": row_id, "correct": correct} for row_id, correct in expected.parent_correctness)
    child = tuple({"example_id_hash": row_id, "correct": correct} for row_id, correct in expected.child_correctness)
    rebuilt = transition_audit_from_categorical_profiles(
        parent_candidate_hash=expected.parent_candidate_hash,
        child_candidate_hash=expected.child_candidate_hash,
        parent_profile=parent, child_profile=child,
    )
    assert rebuilt.sanitized_payload() == expected.sanitized_payload()
    assert rebuilt.transition_effect_hash == expected.transition_effect_hash


def test_transition_store_exposes_only_latest_one_step_transition() -> None:
    store = LatestTransitionStore()
    first = transition()
    second = CandidateTransitionAudit(
        parent_candidate_hash="older-child", child_candidate_hash="newer-child",
        parent_correctness=(("x", True), ("y", False)),
        child_correctness=(("x", True), ("y", True)),
    )
    store.record(2, first)
    store.record(2, second)
    assert store.get(2) is second
    payload = store.sanitized_payload()
    assert payload["2"]["transition_effect_hash"] == second.transition_effect_hash
    assert first.transition_effect_hash not in str(payload)


def test_local_eval_is_frozen_separately_and_overlap_is_reportable() -> None:
    packet = build(latest=transition()).packet
    local_eval = {row.example_id for row in packet.local_eval_examples}
    assert local_eval == {f"coalition-{index}" for index in range(4)}
    assert not local_eval & {row.example_id for row in packet.responsibility_examples}
    assert not local_eval & {row.example_id for row in packet.focus_examples}
    assert not local_eval & {row.example_id for row in packet.anchor_examples}
    assert packet.role_intersection_counts == {
        "responsibility_x_focus": 0,
        "responsibility_x_anchor": 0,
        "responsibility_x_local_eval": 0,
        "focus_x_anchor": 0,
        "focus_x_local_eval": 0,
        "anchor_x_local_eval": 0,
    }


def test_explicit_layer2_local_eval_ids_are_honored_exactly_and_overlap_is_counted() -> None:
    assignment = TeamSearchAssignment(
        target_member=2,
        parent_prompt=PARENT,
        evidence=evidence(),
        optimization_context="repair direct flips",
        responsibility_identity="responsibility",
        local_validation_example_ids=("responsibility-0", "coalition-1"),
        primary_responsibility_lane="direct_flip",
        responsibility_value=8.0,
    )
    request = TeamSearchRequest(
        80, 1, "team-state", 36, "COMMON_SOLVER_CONTRACT_V1", "output-v1",
        "optimize-only-v1",
    )
    packet = Layer2EvidenceRequestBuilder().build(request, assignment).packet
    assert [row.example_id for row in packet.local_eval_examples] == [
        "responsibility-0", "coalition-1"
    ]
    assert packet.role_intersection_counts["responsibility_x_local_eval"] == 1
    assert dict(packet.provenance)["local_eval_source"] == "assignment_frozen_ids"


def test_explicit_layer2_local_eval_never_backfills_from_outside_assignment() -> None:
    assignment = TeamSearchAssignment(
        target_member=2,
        parent_prompt=PARENT,
        evidence=evidence(),
        optimization_context="repair direct flips",
        responsibility_identity="responsibility",
        local_validation_example_ids=("not-in-optimize-evidence",),
        primary_responsibility_lane="direct_flip",
        responsibility_value=8.0,
    )
    request = TeamSearchRequest(
        80, 1, "team-state", 36, "COMMON_SOLVER_CONTRACT_V1", "output-v1",
        "optimize-only-v1",
    )
    with pytest.raises(ValueError, match="frozen local-eval example absent"):
        Layer2EvidenceRequestBuilder().build(request, assignment)


def test_layer2_packet_content_directly_changes_optimizer_input() -> None:
    packet_a = build(context_suffix="-a")
    packet_b = build(context_suffix="-b")
    assert packet_a.parent_decision_procedure == packet_b.parent_decision_procedure
    assert packet_a.seed == packet_b.seed
    assert packet_a.packet.budget == packet_b.packet.budget
    assert packet_a.packet.packet_hash != packet_b.packet.packet_hash
    assert packet_a.identity() != packet_b.identity()


def test_transition_must_end_at_current_parent() -> None:
    wrong = CandidateTransitionAudit(
        parent_candidate_hash="old", child_candidate_hash="not-current-parent",
        parent_correctness=(("responsibility-0", True),),
        child_correctness=(("responsibility-0", False),),
    )
    with pytest.raises(ValueError, match="current parent prompt"):
        build(latest=wrong)


def test_heldout_rows_are_rejected_before_packet_creation() -> None:
    with pytest.raises(ValueError, match="Optimize-derived"):
        TeamEvidenceCase(
            "heldout", "private heldout payload", "A", None, None,
            "responsibility", ("direct_flip",), source_split="validation",
        )
