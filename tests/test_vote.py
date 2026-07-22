import pytest

from multi_dataset_diverse_rl.utils import plurality_vote_with_diagnostics


def test_plurality_no_tie():
    vote = plurality_vote_with_diagnostics(["A", "A", "B"])
    assert vote["vote_answer"] == "A"
    assert vote["vote_tie"] is False
    assert vote["vote_counts"] == {"A": 2, "B": 1}


def test_plurality_first_and_abstain_tie_breaks():
    first = plurality_vote_with_diagnostics(["B", "A"], tie_break_method="first")
    abstain = plurality_vote_with_diagnostics(["A", "B"], tie_break_method="abstain")
    assert first["vote_answer"] == "B"
    assert abstain["vote_answer"] == ""
    assert first["vote_tie"] and abstain["vote_tie"]


def test_plurality_default_tie_policy_is_abstain():
    vote = plurality_vote_with_diagnostics(["A", "B"])
    assert vote["vote_answer"] == ""
    assert vote["tie_break_method"] == "abstain"


def test_plurality_random_is_deterministic():
    left = plurality_vote_with_diagnostics(["A", "B"], tie_break_method="random", seed=42, question_hash="q1")
    right = plurality_vote_with_diagnostics(["A", "B"], tie_break_method="random", seed=42, question_hash="q1")
    assert left == right


def test_plurality_invalid_tie_break_fails():
    with pytest.raises(ValueError, match="Unknown plurality"):
        plurality_vote_with_diagnostics(["A", "B"], tie_break_method="legacy")
