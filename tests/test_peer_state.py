import pytest

from multi_dataset_diverse_rl.peer_state import (
    build_peer_vote_context,
    build_team_vote_state,
    soft_vote_utility,
)


def state(answers, gold="A", valid=None, question_hash="q"):
    return build_team_vote_state(
        question_hash=question_hash,
        gold_answer=gold,
        answers=answers,
        valid_vector=valid,
        normalize_answer=str.upper,
        match_answer=lambda left, right: left == right,
        tie_break="abstain",
        seed=42,
    )


def test_team_is_five_agents_and_peer_context_is_four_peers():
    team = state(["A", "B", "B", "C", "D"])
    peer = build_peer_vote_context(team, 4)
    assert len(team.team_answers) == 5
    assert len(peer.peer_answers) == 4
    assert peer.target_agent_id == 4
    assert peer.peer_answers == ("A", "B", "B", "C")
    assert "D" not in dict(peer.peer_wrong_vote_histogram)


def test_same_gold_count_distinguishes_wrong_histogram_and_margin():
    concentrated = state(["A", "A", "B", "B", "B"])
    dispersed = state(["A", "A", "B", "C", "D"])
    assert concentrated.gold_vote_count == dispersed.gold_vote_count == 2
    assert concentrated.largest_wrong_vote_count == 3
    assert dispersed.largest_wrong_vote_count == 1
    assert concentrated.plurality_margin == -1
    assert dispersed.plurality_margin == 1


def test_tie_as_abstain_and_margin_equivalence():
    result = state(["A", "A", "B", "B", "C"])
    assert result.top_tie is True
    assert result.vote_answer == ""
    assert result.vote_correct is False
    assert result.plurality_margin == 0


def test_invalid_answer_does_not_vote_or_count_correct():
    result = state(["A", "A", "B", "C", "D"], valid=[False, True, True, True, True])
    assert result.team_answers[0] == "A"
    assert result.team_correctness[0] is False
    assert result.gold_vote_count == 1
    assert result.wrong_vote_histogram == (("B", 1), ("C", 1), ("D", 1))


def test_team_and_target_invariants_fail_explicitly():
    with pytest.raises(ValueError, match="exactly 5"):
        state(["A", "B", "C"])
    team = state(["A", "B", "B", "C", "D"])
    with pytest.raises(IndexError):
        build_peer_vote_context(team, 5)


def test_soft_utility_has_dense_but_not_c0_semantics():
    assert soft_vote_utility(0, -5) == 0.0
    assert soft_vote_utility(1, -3) > 0.0
    assert soft_vote_utility(2, 1) > soft_vote_utility(2, -1)
    with pytest.raises(ValueError):
        soft_vote_utility(1, 0, 0.0)
