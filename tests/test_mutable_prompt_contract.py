import asyncio
import json
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import (
    mutable_prompt_violation_reasons,
    validate_mutable_decision_procedure,
)
from multi_dataset_diverse_rl.evaluation.output_contract import solver_system_prompt
from multi_dataset_diverse_rl.tcs import parse_student_candidates
from multi_dataset_diverse_rl.system import (
    CandidateFunnel,
    PromptEnsembleOptimizationSystem,
)
from scripts.audit_mutable_prompt_contamination import scan_run_directory


@pytest.mark.parametrize(
    "marker",
    [
        "FINAL_ANSWER: A",
        "FINAL_ANSWER:A",
        "final_answer: b",
        "FINAL ANSWER: C",
        "FINAL-ANSWER: D",
        "FINAL_ANSWER ： A",
        "FINAL_ANSWER: X",
        "FINAL_ANSWER: <answer>",
        "Mandatory output interface",
        "Solver output contract",
        "There must be exactly one FINAL_ANSWER line",
        "Answer: A",
        '"answer": "B"',
        "Output the answer as one option letter.",
        "End your response with the selected label.",
        "The final output must be exactly in the format: [Option Letter] [Confidence Indicator].",
        "Append a confidence indicator to the final output.",
        "If the evidence is insufficient, return 'None'.",
        "Add a confidence label after deciding.",
    ],
)
@pytest.mark.parametrize(
    "template",
    [
        "{}\nThen reason carefully.",
        "Reason carefully.\n{}\nThen verify.",
        "Reason carefully.\n{}",
        "```text\n{}\n```",
        "> Invalid example: {}",
    ],
)
def test_mutable_prompt_rejects_contract_markers_in_any_position(marker, template):
    prompt = template.format(marker)
    assert mutable_prompt_violation_reasons(prompt)
    with pytest.raises(
        ValueError,
        match="mutable_prompt_contract_violation: output_contract_contamination",
    ):
        validate_mutable_decision_procedure(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "Finalize the answer after checking all antecedents.",
        "Choose the final interpretation only after verification.",
        "Do not rely on the first candidate.",
    ],
)
def test_mutable_prompt_allows_nearby_legal_reasoning_language(prompt):
    assert mutable_prompt_violation_reasons(prompt) == ()
    validate_mutable_decision_procedure(prompt)


def test_student_parser_keeps_clean_candidate_and_rejects_only_contaminated_one():
    context = SimpleNamespace(evidence_cases=())
    result = parse_student_candidates(
        {
            "candidate_prompts": [
                "A valid reasoning procedure.",
                "A procedure.\nFINAL_ANSWER: A",
            ]
        },
        parent_prompt="parent",
        context=context,
        expected_count=2,
    )
    assert result.raw_count == 2
    assert [row.candidate_prompt for row in result.candidates] == [
        "A valid reasoning procedure."
    ]
    assert result.rejection_reasons == (
        (),
        ("output_contract_contamination",),
    )


def test_solver_boundary_rejects_contaminated_mutable_section():
    with pytest.raises(ValueError, match="output_contract_contamination"):
        solver_system_prompt("Reason.\nFINAL_ANSWER: A", "option_letter")
    request = solver_system_prompt(
        "Compare every compatible antecedent before selecting one.",
        "option_letter",
    )
    mutable_section, immutable_section = request.split(
        "Mandatory output interface:", maxsplit=1
    )
    assert "FINAL_ANSWER" not in mutable_section
    assert "FINAL_ANSWER: X" in immutable_section
    assert "FINAL_ANSWER: A" not in request
    assert "FINAL_ANSWER: B" not in request


def test_initial_prompts_reject_contract_contamination(tmp_path):
    calls = 0

    async def solver(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("Solver must not run")

    with pytest.raises(ValueError, match="output_contract_contamination"):
        PromptEnsembleOptimizationSystem(
            Config.from_flat(
                out_dir=str(tmp_path),
                shared_prompt="Reason carefully. FINAL_ANSWER: A",
            ),
            solver=solver,
        )
    assert calls == 0


def test_contaminated_active_parent_stops_before_tcs_or_solver_calls(tmp_path):
    calls = {"optimizer": 0, "solver": 0}

    async def optimizer(*_args):
        calls["optimizer"] += 1
        raise AssertionError("optimizer must not run")

    async def solver(*_args):
        calls["solver"] += 1
        raise AssertionError("Solver must not run")

    system = PromptEnsembleOptimizationSystem(
        Config.from_flat(
            out_dir=str(tmp_path),
            experiment_setting="shared_vote_state_diagnosis",
        ),
        solver=solver,
        optimizer_chat=optimizer,
    )
    contaminated = "Reason carefully.\nFINAL_ANSWER: A"
    system.agents[0].current_prompt = contaminated
    with pytest.raises(ValueError, match="output_contract_contamination"):
        asyncio.run(
            system.propose_candidates(0, set(), CandidateFunnel(), 0)
        )
    with pytest.raises(ValueError, match="output_contract_contamination"):
        asyncio.run(system.solve("q", 0, contaminated))
    assert calls == {"optimizer": 0, "solver": 0}


def test_historical_scanner_returns_hashes_and_categories_without_prompt_text(
    tmp_path,
):
    secret_prompt = "Sensitive reasoning text.\nFINAL_ANSWER: A"
    (tmp_path / "best_prompts.json").write_text(
        json.dumps([secret_prompt, "Clean reasoning procedure."]),
        encoding="utf-8",
    )
    (tmp_path / "run_meta.json").write_text(
        json.dumps({
            "run_identity": {"experiment_setting": "setting", "id": "run"},
            "config": {"seed": 45},
        }),
        encoding="utf-8",
    )
    result = scan_run_directory(tmp_path)
    rendered = json.dumps(result, ensure_ascii=False)
    assert result["contaminated_prompt_count"] == 1
    assert result["prompt_records"][0]["contaminated"] is True
    assert result["prompt_records"][0]["marker_category"] == [
        "forbidden_final_answer_marker"
    ]
    assert secret_prompt not in rendered
    assert "Sensitive reasoning text" not in rendered
