import asyncio

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.solver_output import parse_solver_output
from multi_dataset_diverse_rl.evaluation.output_contract import (
    SOLVER_OUTPUT_CONTRACT_VERSION,
    solver_output_contract,
)
from multi_dataset_diverse_rl.tasks import get_task_spec
from multi_dataset_diverse_rl.llm_client import LLMCallResult
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


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


def test_option_letter_contract_matches_strict_parser():
    contract = solver_output_contract("option_letter")
    assert SOLVER_OUTPUT_CONTRACT_VERSION in contract
    assert "FINAL_ANSWER: X" in contract
    assert "one uppercase option letter" in contract
    assert "Do not add parentheses, punctuation, explanation" in contract


def test_system_appends_non_optimizable_task_contract_to_solver_prompt(tmp_path):
    system = PromptEnsembleOptimizationSystem(
        Config.from_flat(
            out_dir=str(tmp_path),
            task_type="mmlu",
            answer_format="option_letter",
        )
    )
    captured = {}

    async def chat_result(model, system_prompt, user_prompt, temperature, max_tokens, role):
        captured.update({
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "role": role,
        })
        return LLMCallResult(
            "Reason\nFINAL_ANSWER: A", 2, 3, 5, 0.01,
            "stop", max_tokens, False,
        )

    system.llm.chat_result = chat_result
    answer = asyncio.run(system.solve(
        "Question\n(A) first\n(B) second",
        0,
        "This is the optimizable decision procedure.",
    ))
    assert answer.valid is True
    assert "Solver output contract (task_output_contract_v1)" in captured["system_prompt"]
    assert "FINAL_ANSWER: X" in captured["system_prompt"]
    assert captured["system_prompt"].endswith("This is the optimizable decision procedure.")
    assert captured["role"] == "solver"
