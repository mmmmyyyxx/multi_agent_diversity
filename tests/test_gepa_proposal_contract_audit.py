"""Zero-provider replay and edge cases for the component proposal boundary."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import (
    compact_prompt_failed_checks,
    primary_prompt_rejection_category,
    validate_complete_compact_prompt,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_proposer_contract import (
    DECISION_PROCEDURE_REFLECTION_TEMPLATE,
)
from scripts.audit_gepa_proposal_contract import audit


def _checks(prompt: str) -> tuple[str, ...]:
    return compact_prompt_failed_checks(
        prompt, parent_prompt="Compare plausible antecedents.", examples=(), max_chars=3000,
    )


def test_stored_four_extracted_proposals_replay_when_private_artifacts_exist() -> None:
    root = os.environ.get("GEPA_PROPOSAL_AUDIT_RUN_ROOT")
    optimize = os.environ.get("GEPA_PROPOSAL_AUDIT_OPTIMIZE_CSV")
    if not root or not optimize:
        pytest.skip("private completed-run evidence not configured")
    result = audit(Path(root), Path(optimize))
    assert result["raw_reflection_response_replay"] == "NOT_RECOVERABLE_ZERO_API"
    rows = result["proposals"]
    assert len(rows) == 4
    assert all(row["persisted_hash_match"] for row in rows)
    assert all(row["failed_checks_match"] for row in rows)
    assert all(row["validator_rejected"] for row in rows)
    assert [row["recorded_primary_category"] for row in rows] == [
        "over_length", "over_length", "over_length", "output_contract_contamination",
    ]
    assert [row["normalized_proposal_characters"] for row in rows] == [
        3130, 3388, 3483, 2996,
    ]


@pytest.mark.parametrize(
    ("proposal", "expected"),
    [
        ("Compare each candidate's grammatical and causal fit, then verify the inference.", ()),
        ("Reason carefully. " + "x" * (3000 - len("Reason carefully. ")), ()),
        ("Reason carefully. " + "x" * (3001 - len("Reason carefully. ")), ("over_length",)),
        (
            "You are the full solver. Mandatory output interface: FINAL_ANSWER: A",
            ("forbidden_final_answer_marker", "copied_solver_interface"),
        ),
        ("Answer: A", ("fixed_answer_payload",)),
        ("There must be exactly one FINAL_ANSWER line.",
         ("forbidden_final_answer_marker", "copied_solver_interface")),
        ("Output the final answer as one option.",
         ("forbidden_final_answer_marker", "copied_solver_interface")),
        # Current frozen validator does not have a role-boilerplate or
        # duplication check. This is observation, not a relaxed new rule.
        ("You are a solver. Compare antecedents. You are a solver. Compare antecedents.", ()),
    ],
)
def test_current_contract_classification_is_deterministic(
    proposal: str, expected: tuple[str, ...],
) -> None:
    assert _checks(proposal) == expected
    assert _checks(proposal) == expected
    if expected:
        with pytest.raises(ValueError):
            validate_complete_compact_prompt(
                proposal, parent_prompt="Compare plausible antecedents.",
                examples=(), max_chars=3000,
            )
        assert primary_prompt_rejection_category(expected) is not None
    else:
        validate_complete_compact_prompt(
            proposal, parent_prompt="Compare plausible antecedents.",
            examples=(), max_chars=3000,
        )


def test_component_template_explicitly_excludes_solver_shell_and_output_contract() -> None:
    template = DECISION_PROCEDURE_REFLECTION_TEMPLATE.casefold()
    assert "<curr_param>" in template and "<side_info>" in template
    assert "decision procedure" in template
    assert "system prompt boilerplate" in template
    assert "answer-format instructions" in template
    assert "immutable solver" in template
