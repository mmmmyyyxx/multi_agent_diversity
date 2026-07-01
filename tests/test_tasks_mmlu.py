from multi_dataset_diverse_rl.tasks import (
    extract_pred_answer_mmlu,
    get_task_spec,
    parse_mmlu_gold,
)


def test_mmlu_gold_parsing_keeps_old_behavior():
    assert parse_mmlu_gold("A") == "A"
    assert parse_mmlu_gold("0") == "A"
    assert parse_mmlu_gold("The answer is C.") == "C"


def test_mmlu_prediction_extraction():
    assert extract_pred_answer_mmlu("Reasoning...\nFINAL_ANSWER: B") == "B"
    assert extract_pred_answer_mmlu("I choose option d.") == "D"


def test_mmlu_task_spec_match():
    spec = get_task_spec("mmlu")
    assert spec.match_answer("b", "B")
    assert spec.match_answer("FINAL_ANSWER: C", "C")
