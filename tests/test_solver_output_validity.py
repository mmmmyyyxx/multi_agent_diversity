import pytest

from multi_dataset_diverse_rl.evaluation.solver_output import parse_solver_output
from multi_dataset_diverse_rl.tasks import get_task_spec


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("Reasoning without a marker", "missing_final_answer"),
        ("FINAL_ANSWER: A\nFINAL_ANSWER: B", "multiple_final_answers"),
        ("FINAL_ANSWER:", "unparseable_final_answer"),
        ("FINAL_ANSWER: Z", "out_of_domain_answer"),
        ("FINAL_ANSWER: A because it is the best option", "out_of_domain_answer"),
        ("Mention FINAL_ANSWER: A inside prose", "missing_final_answer"),
    ],
)
def test_solver_output_requires_exactly_one_valid_final_answer_line(text, status):
    result = parse_solver_output(
        text,
        question="Question\n(A) left\n(B) right\n(C) up\n(D) down",
        task_spec=get_task_spec("mmlu"),
        answer_format="option_letter",
    )
    assert result.valid is False
    assert result.validity_status == status


def test_solver_output_accepts_one_domain_valid_line():
    result = parse_solver_output(
        "Reason carefully.\nFINAL_ANSWER: B",
        question="Question\n(A) left\n(B) right\n(C) up\n(D) down",
        task_spec=get_task_spec("mmlu"),
        answer_format="option_letter",
    )
    assert result.valid is True
    assert result.answer == "B"
    assert result.validity_status == "valid"
