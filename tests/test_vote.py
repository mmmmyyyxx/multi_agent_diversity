from multi_dataset_diverse_rl.utils import compute_gold_vote_diagnostics, majority_vote_with_diagnostics


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


def test_gold_vote_diagnostics_respects_matcher_and_boundary_condition():
    matcher = lambda prediction, gold: prediction.lower() == gold.lower()
    concentrated = compute_gold_vote_diagnostics(["A", "A", "B", "B", "C"], "A", matcher, 5)
    dispersed = compute_gold_vote_diagnostics(["A", "A", "B", "C", "D"], "A", matcher, 5)
    assert concentrated["gold_vote_count"] == 2
    assert concentrated["largest_wrong_vote_count"] == 2
    assert concentrated["normalized_vote_margin"] == 0.0
    assert concentrated["boundary_useful_diversity"] < dispersed["boundary_useful_diversity"]


def test_gold_vote_diagnostics_empty_answers_have_negative_margin():
    metrics = compute_gold_vote_diagnostics(["", ""], "A", lambda prediction, gold: prediction == gold, 2)
    assert metrics == {
        "gold_vote_count": 0,
        "largest_wrong_vote_count": 0,
        "normalized_vote_margin": -1.0,
        "boundary_useful_diversity": 0.0,
    }
