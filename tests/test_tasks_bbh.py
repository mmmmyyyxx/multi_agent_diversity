from multi_dataset_diverse_rl.tasks import (
    extract_pred_answer_bbh,
    match_bbh_answer,
    normalize_bbh_answer,
    parse_gold_bbh,
)


def test_bbh_extract_final_answer_yes():
    assert extract_pred_answer_bbh("Some reasoning\nFINAL_ANSWER: yes") == "yes"


def test_bbh_true_yes_equivalence():
    assert extract_pred_answer_bbh("FINAL_ANSWER: True") == "yes"
    assert match_bbh_answer("True", "yes")
    assert match_bbh_answer("yes", "true")


def test_bbh_option_normalization():
    assert normalize_bbh_answer("FINAL_ANSWER: (A)") == "a"
    assert normalize_bbh_answer("A.") == "a"
    assert normalize_bbh_answer("option A") == "a"


def test_bbh_numeric_normalization():
    assert extract_pred_answer_bbh("FINAL_ANSWER: 3.0") == "3"
    assert match_bbh_answer("3.0", "3")


def test_bbh_without_final_answer_uses_last_non_empty_line():
    text = "First line\n\nmaybe yes\nNo"
    assert extract_pred_answer_bbh(text) == "no"


def test_bbh_does_not_blindly_take_last_number():
    text = "There are 12 tokens and 47 distractors.\nThe valid expression is false."
    assert extract_pred_answer_bbh(text) == "the valid expression is false"


def test_bbh_uses_final_answer_number_when_marked():
    text = "There are 12 tokens and 47 distractors.\nFINAL_ANSWER: 3"
    assert extract_pred_answer_bbh(text) == "3"


def test_bbh_gold_aliases_list_can_match():
    gold = parse_gold_bbh('["yes", "true"]')
    assert match_bbh_answer("FINAL_ANSWER: True", gold)
    assert match_bbh_answer("yes", gold)
