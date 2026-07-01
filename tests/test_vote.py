from multi_dataset_diverse_rl.utils import majority_vote_with_diagnostics


def test_majority_vote_no_tie():
    vote = majority_vote_with_diagnostics(["A", "A", "B"])
    assert vote["vote_answer"] == "A"
    assert vote["vote_tie"] is False
    assert vote["vote_counts"] == {"A": 2, "B": 1}


def test_majority_vote_first_tie_break():
    vote = majority_vote_with_diagnostics(["A", "B"], tie_break_method="first")
    assert vote["vote_answer"] == "A"
    assert vote["vote_tie"] is True
    assert set(vote["tie_candidates"]) == {"A", "B"}


def test_majority_vote_abstain_tie_break():
    vote = majority_vote_with_diagnostics(["A", "B"], tie_break_method="abstain")
    assert vote["vote_answer"] == ""
    assert vote["vote_tie"] is True


def test_majority_vote_random_is_deterministic():
    left = majority_vote_with_diagnostics(["A", "B"], tie_break_method="random", seed=42, question_hash="q1")
    right = majority_vote_with_diagnostics(["A", "B"], tie_break_method="random", seed=42, question_hash="q1")
    assert left["vote_answer"] == right["vote_answer"]
    assert left["vote_tie"] is True


def test_majority_vote_counts_are_correct():
    vote = majority_vote_with_diagnostics(["A", "B", "B", "", "C", "C"])
    assert vote["vote_counts"] == {"A": 1, "B": 2, "C": 2}
