from multi_dataset_diverse_rl.answer_formats import canonical_answer, extract_prediction, match_answer


def test_answer_format_boolean_is_not_yes_no():
    assert canonical_answer("Final answer: true", "boolean") == "true"
    assert canonical_answer("Final answer: false", "boolean") == "false"
    assert canonical_answer("Final answer: yes", "yes_no") == "yes"
    assert not match_answer("true", "yes", "boolean")


def test_answer_format_valid_invalid():
    assert canonical_answer("Final answer: invalid", "valid_invalid") == "invalid"
    assert canonical_answer("Answer: valid", "valid_invalid") == "valid"


def test_answer_format_option_letter():
    assert canonical_answer("Final answer: (C)", "option_letter") == "C"
    assert canonical_answer("option A", "option_letter") == "A"
    assert extract_prediction("reasoning\nFINAL_ANSWER: B", "option_letter") == "B"


def test_answer_format_numeric_matches_commas():
    assert canonical_answer("Final answer: 1,234.0", "numeric") == "1234"
    assert match_answer("1,234", "1234", "numeric")


def test_answer_format_alias_list_matches():
    assert match_answer("FINAL_ANSWER: invalid", '["invalid", "not valid"]', "valid_invalid")
