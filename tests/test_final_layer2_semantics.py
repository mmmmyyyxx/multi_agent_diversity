"""Final production Layer-2 contract; no provider calls or historical routing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from multi_dataset_diverse_rl.peer_state import build_peer_vote_context, build_team_vote_state
from multi_dataset_diverse_rl.responsibility import (
    ResponsibilityState, compute_member_aware_repair_opportunity,
)

from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (
    build_primary_responsibility_summaries,
    select_primary_responsibility_targets,
)
from multi_dataset_diverse_rl.team_search.schemas import (
    TeamEvidenceCase, TeamSearchAssignment, TeamSearchRequest,
)
from multi_dataset_diverse_rl.team_search.task_builder import (
    Layer2EvidenceRequestBuilder, LocalTaskBuilder,
)
from multi_dataset_diverse_rl.team_search.system_runtime import (
    SystemResponsibilityAssignmentFactory, freeze_current_responsibility,
)
from multi_dataset_diverse_rl.versions import TEAM_MINIBATCH_CONTRACT_VERSION


def _evidence() -> tuple[TeamEvidenceCase, ...]:
    residual = tuple(
        TeamEvidenceCase(
            f"wrong-{index:02d}", "case", "A", "B", None,
            "repair", ("repair", "coverage", "team_hard"),
            team_disagreement=index % 3, residual_frequency=5,
            team_margin=-5,
        )
        for index in range(8)
    )
    correct = tuple(
        TeamEvidenceCase(
            f"right-{index:02d}", "case", "A", "A", None,
            "preservation", ("preservation",),
            team_margin=5, mutation_sensitive=False,
        )
        for index in range(4)
    )
    return residual + correct


def test_overlap_and_seed_independent_primary_target_order() -> None:
    rows = build_primary_responsibility_summaries(
        assigned={}, current_margin_by_question={}, failure_count_by_member={}
    )
    decisions = [select_primary_responsibility_targets(
        rows, seed=seed, update_index=seed * 7, target_count=2
    ).selected_member_ids for seed in range(100)]
    assert set(decisions) == {(0, 1)}


def test_routing_policy_poison_cannot_change_raw_graph_or_scores() -> None:
    states = tuple(build_team_vote_state(
        question_hash=f"wrong-{index:02d}", gold_answer="A",
        answers=["B"] * 5, valid_vector=[True] * 5,
    ) for index in range(8)) + tuple(build_team_vote_state(
        question_hash=f"right-{index:02d}", gold_answer="A",
        answers=["A"] * 5, valid_vector=[True] * 5,
    ) for index in range(4))
    opportunities = {state.question_hash: tuple(
        compute_member_aware_repair_opportunity(
            team_state=state, peer_context=build_peer_vote_context(state, member)
        ) for member in range(5)
    ) for state in states}

    class System:
        responsibility_state = ResponsibilityState(
            updates_since_selected_by_agent={member: 0 for member in range(5)}
        )

        def __init__(self, routed_owner: int) -> None:
            self.routed_owner = routed_owner
            self.fixed_probe = SimpleNamespace(examples=tuple(
                SimpleNamespace(question_hash=state.question_hash, question="case", gold_answer="A")
                for state in states
            ))
            self.active_profiles = tuple(tuple(SimpleNamespace(
                answer="B" if index < 8 else "A", valid=True,
            ) for index in range(12)) for _ in range(5))
            self.agents = tuple(SimpleNamespace(current_prompt="same prompt") for _ in range(5))
            self.protocol = SimpleNamespace(tie_policy="abstain")

        def current_states_and_opportunities(self):
            return states, {}, opportunities

        def assign_responsibilities(self, *, update_index):
            raise AssertionError("historical routing must not be called")

    left = freeze_current_responsibility(System(0), update_index=0)
    right = freeze_current_responsibility(System(4), update_index=99)
    assert left.assigned == right.assigned
    assert all(len(left.assigned[member]) == 8 for member in range(5))
    margins = left.current_margin_by_question
    scores = build_primary_responsibility_summaries(
        assigned=left.assigned, current_margin_by_question=margins,
        failure_count_by_member={},
    )
    assert scores == build_primary_responsibility_summaries(
        assigned=right.assigned, current_margin_by_question=margins,
        failure_count_by_member={},
    )
    assert select_primary_responsibility_targets(
        scores, seed=0, update_index=0
    ).selected_member_ids == (0, 1)
    request = TeamSearchRequest(80, 0, "same-team", 36, "solver", "output", "global")
    assignments = tuple(SystemResponsibilityAssignmentFactory(
        system=system,
        snapshot_reader=lambda snapshot=snapshot: snapshot,
        task_builder=LocalTaskBuilder(),
    ).build_from_member(
        request=request, member_id=0, primary_lane="fallback",
        responsibility_identity="raw-legal",
    ) for system, snapshot in ((System(0), left), (System(4), right)))
    assert assignments[0].evidence == assignments[1].evidence
    assert assignments[0].local_validation_example_ids == assignments[1].local_validation_example_ids
    assert {row.example_id for row in assignments[0].evidence if "team_hard" in row.tags} == {
        f"wrong-{index:02d}" for index in range(8)
    }
    selected = LocalTaskBuilder().select_team_minibatch(assignments[0].evidence)
    assert assignments[0].local_validation_example_ids == tuple(row.example_id for row in selected)
    assert tuple(row.example_id for row in Layer2EvidenceRequestBuilder().build(
        request, assignments[0]
    ).packet.local_eval_examples) == assignments[0].local_validation_example_ids
    assert sum(row.evidence_group == "repair" for row in assignments[0].evidence) == 8
    assert sum(row.evidence_group == "preservation" for row in assignments[0].evidence) == 4


def test_identical_baseline_has_repair_preservation_team_hard_without_unassigned_rows() -> None:
    evidence = _evidence()
    builder = Layer2EvidenceRequestBuilder()
    outcomes = [builder.select_team_minibatch(
        evidence if index % 2 else tuple(reversed(evidence)),
        primary_responsibility_lane="coverage",
    ) for index in range(100)]
    identities = [tuple((row.example_id, row.evidence_group) for row in rows)
                  for rows in outcomes]
    assert all(value == identities[0] for value in identities)
    rows = outcomes[0]
    assert len(rows) == len({row.example_id for row in rows}) == 12
    hashed = lambda value: (hashlib.sha256(value.encode("utf-8")).hexdigest(), value)
    assert [row.example_id for row in rows if row.evidence_group == "repair"] == sorted(
        (f"wrong-{index:02d}" for index in range(8)), key=hashed,
    )[:4]
    assert builder.team_minibatch_telemetry(rows) == {
        "contract_version": TEAM_MINIBATCH_CONTRACT_VERSION,
        "total_count": 12,
        "repair_count": 4,
        "preservation_count": 4,
        "team_hard_count": 4,
        "backfill_count": 0,
    }
    assert {row.example_id for row in rows if row.evidence_group == "team_hard"} <= {
        row.example_id for row in evidence if row.evidence_group == "repair"
    }


def test_same_frozen_packet_ignores_external_global_dataset_identity() -> None:
    evidence = _evidence()
    builder = Layer2EvidenceRequestBuilder()
    assignment = TeamSearchAssignment(
        target_member=0, parent_prompt="reasoning", evidence=evidence,
        optimization_context="", responsibility_identity="raw-legal",
        local_validation_example_ids=tuple(row.example_id for row in evidence),
        primary_responsibility_lane="coverage",
    )
    left = builder.build(
        TeamSearchRequest(80, 0, "same-team", 36, "solver", "output", "global-a"),
        assignment,
    )
    right = builder.build(
        TeamSearchRequest(80, 0, "same-team", 36, "solver", "output", "global-b"),
        assignment,
    )
    assert left == right
    assert left.packet.packet_hash == right.packet.packet_hash


def test_production_layer2_does_not_reintroduce_unassigned_coalition_group() -> None:
    root = Path(__file__).resolve().parents[1] / "multi_dataset_diverse_rl"
    for path in (root / "team_search").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "coalition_examples" not in source
        assert "unassigned_residual" not in source
    runtime = (root / "team_search" / "system_runtime.py").read_text(encoding="utf-8")
    assert "refresh_responsibility_after_commit" not in runtime
    assert "service_routing_audit" not in runtime


def test_preservation_vulnerability_and_hash_tie_ordering() -> None:
    rows = tuple(
        TeamEvidenceCase(
            f"preserve-{index}", "case", "A", "A", None,
            "preservation", ("preservation",),
            team_margin=margin,
            team_disagreement=disagreement,
            mutation_sensitive=sensitive,
        )
        for index, (sensitive, margin, disagreement) in enumerate((
            (False, 1, 9), (True, 3, 9), (True, 1, 1),
            (True, 1, 3), (True, 1, 3),
        ))
    )
    ordered = sorted(rows, key=LocalTaskBuilder._preservation_priority)
    assert [row.example_id for row in ordered[:3]] == [
        "preserve-3" if hashlib.sha256(b"preserve-3").hexdigest() < hashlib.sha256(b"preserve-4").hexdigest() else "preserve-4",
        "preserve-4" if hashlib.sha256(b"preserve-3").hexdigest() < hashlib.sha256(b"preserve-4").hexdigest() else "preserve-3",
        "preserve-2",
    ]
    assert ordered[-1].example_id == "preserve-0"
