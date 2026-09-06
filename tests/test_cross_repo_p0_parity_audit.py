from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cross_repo_p0_parity", ROOT / "scripts/audit_cross_repo_p0_parity.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_render_task_input_normalizes_only_transport_newlines() -> None:
    raw = "Stem\r\nOptions:\r\n(A) First\r\n(B) Second\r\n(C) Ambiguous"
    assert MODULE.render_task_input(raw) == (
        "Stem\nOptions:\n(A) First\n(B) Second\n(C) Ambiguous"
    )


def test_parity_matrix_freezes_identified_contract_differences() -> None:
    rows = {row["dimension"]: row for row in MODULE.parity_matrix()}
    assert rows["initial_prompt_exact_bytes"]["classification"] == "NONE"
    assert rows["question_payload_exact_bytes"]["classification"] == "DATA_DIFFERENCE"
    assert rows["provider_seed"]["classification"] == "DECODING_DIFFERENCE"
    assert rows["invalid_output_attempt_cap"]["classification"] == "RETRY_DIFFERENCE"


def test_report_file_contract_is_complete() -> None:
    expected = {
        "README.md",
        "parity_matrix.csv",
        "split_identity.json",
        "prompt_identity.json",
        "request_contract_diff.json",
        "parser_diff.json",
        "prediction_disagreement.csv",
        "classifier.json",
        "fact_assertions.json",
        "provenance.json",
        "sanitization_manifest.json",
    }
    assert set(MODULE.REPORT_FILES) == expected
