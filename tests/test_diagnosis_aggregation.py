from multi_dataset_diverse_rl.diagnosis_aggregation import (
    DIAGNOSIS_AGGREGATION_VERSION,
    answer_role_signature,
    aggregate_probe_diagnosis,
    aggregate_single_lane_diagnosis,
)
from multi_dataset_diverse_rl.evaluation.fixed_probe import ProbeExample
from multi_dataset_diverse_rl.peer_state import (
    build_peer_vote_context,
    build_team_vote_state,
)
from multi_dataset_diverse_rl.responsibility import (
    RepairLane,
    compute_member_aware_repair_opportunity,
)


def state(question_hash, answers, valid=None):
    return build_team_vote_state(
        question_hash=question_hash,
        gold_answer="A",
        answers=answers,
        valid_vector=valid,
        tie_break="abstain",
        seed=42,
    )


def inputs(states):
    contexts = {}
    opportunities = {}
    for row in states:
        contexts[row.question_hash] = {}
        opportunities[row.question_hash] = []
        for agent_id in range(5):
            peer = build_peer_vote_context(row, agent_id)
            contexts[row.question_hash][agent_id] = peer
            opportunities[row.question_hash].append(
                compute_member_aware_repair_opportunity(
                    team_state=row,
                    peer_context=peer,
                )
            )
    examples = tuple(
        ProbeExample(
            question=f"Representative question text for {row.question_hash}",
            question_hash=row.question_hash,
            gold_answer="A",
        )
        for row in states
    )
    return examples, contexts, opportunities


def test_answer_roles_use_gold_wrong_cluster_size_stable_hash_and_invalid():
    row = state(
        "q",
        ("B", "A", "B", "C", ""),
        (True, True, True, True, False),
    )
    assert answer_role_signature(row) == ("W1", "G", "W1", "W2", "I")

    tied = state("tie", ("B", "C", "A", "B", "C"))
    permuted = state("tie2", ("C", "B", "C", "A", "B"))
    roles_by_answer = {
        answer: role
        for answer, role in zip(tied.team_answers, answer_role_signature(tied), strict=True)
    }
    permuted_roles = answer_role_signature(permuted)
    assert all(
        role == roles_by_answer[answer]
        for answer, role in zip(permuted.team_answers, permuted_roles, strict=True)
    )


def test_full_probe_groups_structural_equivalents_and_splits_different_vote_state():
    states = (
        state("q1", ("B", "B", "B", "A", "A")),
        state("q2", ("B", "B", "B", "A", "A")),
        state("q3", ("B", "B", "A", "A", "C")),
        state("q4", ("C", "C", "C", "C", "B")),
    )
    examples, contexts, opportunities = inputs(states)
    result = aggregate_probe_diagnosis(
        target_agent_id=0,
        examples=examples,
        states=states,
        peer_contexts=contexts,
        opportunities=opportunities,
        assigned_hashes={"q1", "q4"},
        context_policy="member_aware_responsibility_conditioned",
        max_patterns=3,
        max_cases=3,
    )
    assert (
        DIAGNOSIS_AGGREGATION_VERSION
        == "single_lane_pattern_aggregation_v1"
    )
    assert result.full_probe_case_count == 4
    assert all(
        set(pattern.represented_question_hashes) <= {"q1", "q4"}
        for pattern in result.available_patterns
    )
    conversion_vote_keys = {
        (
            row.key.gold_vote_count,
            row.key.largest_wrong_vote_count,
            row.key.plurality_margin,
        )
        for row in result.available_patterns
        if row.key.case_family == "conversion_failure"
    }
    assert len(conversion_vote_keys) >= 1
    assert len(result.selected_patterns) <= 3
    assert len(result.evidence_cases) <= 3
    assert len({row.pattern_id for row in result.evidence_cases}) == len(
        result.evidence_cases
    )


def test_pattern_and_representative_selection_are_deterministic():
    states = (
        state("q3", ("B", "B", "B", "A", "A")),
        state("q1", ("B", "B", "B", "A", "A")),
        state("q2", ("C", "C", "C", "A", "A")),
    )
    examples, contexts, opportunities = inputs(states)
    kwargs = dict(
        target_agent_id=0,
        examples=examples,
        states=states,
        peer_contexts=contexts,
        opportunities=opportunities,
        assigned_hashes={"q2"},
        context_policy="generic_peer_state",
    )
    first = aggregate_probe_diagnosis(**kwargs)
    second = aggregate_probe_diagnosis(**kwargs)
    assert first == second


def test_single_lane_preservation_can_use_stable_current_correct_case():
    states = (
        state("repair", ("B", "A", "A", "B", "B")),
        state("stable", ("A", "A", "A", "B", "C")),
    )
    examples, _, opportunities = inputs(states)
    result = aggregate_single_lane_diagnosis(
        target_agent_id=0,
        examples=examples,
        states=states,
        opportunities=opportunities,
        active_hashes={"repair"},
        active_lane=RepairLane.DIRECT_FLIP,
    )
    assert [row.question_hash for row in result.repair_cases] == ["repair"]
    assert result.preservation_case is not None
    assert result.preservation_case.question_hash == "stable"
    assert not result.preservation_case.unique_correct
    assert not result.preservation_case.pivotal_correct
