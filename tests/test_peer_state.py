import pytest

from multi_dataset_diverse_rl.peer_state import build_leave_one_out_peer_state, build_peer_vote_state, soft_vote_utility


def state(answers, gold="A", valid=None, question_hash="q"):
    return build_peer_vote_state(
        question_hash=question_hash, gold_answer=gold, answers=answers, valid_vector=valid,
        normalize_answer=lambda value: value.upper(), match_answer=lambda left, right: left == right,
        tie_break="random", seed=42,
    )


def test_same_gold_count_distinguishes_wrong_histogram_and_margin():
    concentrated = state(["A", "A", "B", "B", "B"])
    dispersed = state(["A", "A", "B", "C", "D"])
    assert concentrated.gold_vote_count == dispersed.gold_vote_count == 2
    assert concentrated.largest_wrong_vote_count == 3
    assert dispersed.largest_wrong_vote_count == 1
    assert concentrated.plurality_margin == -1
    assert dispersed.plurality_margin == 1


def test_dominant_wrong_tie_and_deterministic_vote_tie_break():
    first = state(["A", "B", "B", "C", "C"])
    second = state(["A", "B", "B", "C", "C"])
    assert first.dominant_wrong_answers == ("B", "C")
    assert first.top_tie is True
    assert first.vote_answer == second.vote_answer


def test_invalid_answer_does_not_vote_or_count_correct():
    result = state(["A", "A", "B"], valid=[False, True, True])
    assert result.normalized_answers == ("A", "A", "B")
    assert result.correctness_vector == (False, True, False)
    assert result.gold_vote_count == 1


def test_leave_one_out_uses_same_plurality_policy():
    result = build_leave_one_out_peer_state(
        agent_id=0, question_hash="q", gold_answer="A", answers=["A", "B", "B", "C", "C"],
        normalize_answer=str.upper, match_answer=lambda left, right: left == right,
        tie_break="random", seed=42,
    )
    assert result.normalized_answers == ("B", "B", "C", "C")
    assert result.gold_vote_count == 0
    assert result.top_tie is True


def test_soft_utility_has_required_dense_semantics():
    assert soft_vote_utility(0, -5) == 0.0
    assert soft_vote_utility(0, -2) == 0.0
    assert soft_vote_utility(1, -3) > 0.0
    assert soft_vote_utility(2, 1) > soft_vote_utility(2, -1)
    assert soft_vote_utility(5, 5) - soft_vote_utility(5, 4) < 0.02
    with pytest.raises(ValueError):
        soft_vote_utility(1, 0, 0.0)
