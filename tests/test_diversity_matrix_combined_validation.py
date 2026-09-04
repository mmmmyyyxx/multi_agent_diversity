from __future__ import annotations

import pytest

from scripts.run_diversity_matrix_combined_validation import (
    AMENDMENT_VERSION,
    ORIGINAL_TRAINING_COMMIT,
    SEEDS,
    _read_json_array,
    _write_former_test_evidence,
    combine_counts,
)


def test_combined_validation_uses_counts_not_equal_accuracy_average() -> None:
    validation = {
        "validation_row_count": 50,
        "vote_correct_count": 25,
        "oracle_correct_count": 35,
        "per_agent_correct_counts": [20, 21, 22, 23, 24],
    }
    former_test = {
        "row_count": 125,
        "vote_correct_count": 100,
        "oracle_correct_count": 110,
        "per_agent_correct_counts": [90, 91, 92, 93, 94],
    }
    result = combine_counts(validation, former_test)
    assert result["row_count"] == 175
    assert result["vote_correct_count"] == 125
    assert result["vote_accuracy"] == pytest.approx(125 / 175)
    assert result["vote_accuracy"] != pytest.approx((0.5 + 0.8) / 2)
    assert result["oracle_correct_count"] == 145
    assert result["per_agent_correct_counts"] == [110, 112, 114, 116, 118]


def test_combined_validation_rejects_wrong_total_size() -> None:
    with pytest.raises(ValueError, match="175"):
        combine_counts(
            {"validation_row_count": 49, "vote_correct_count": 0, "oracle_correct_count": 0, "per_agent_correct_counts": [0] * 5},
            {"row_count": 125, "vote_correct_count": 0, "oracle_correct_count": 0, "per_agent_correct_counts": [0] * 5},
        )


def test_posthoc_amendment_identity_is_explicit() -> None:
    assert AMENDMENT_VERSION == "combined_development_validation175_posthoc_v1"
    assert SEEDS == (72, 73, 74)
    assert ORIGINAL_TRAINING_COMMIT == "ca2c4b2e7e78d5594b702298c7a392ed3ca5ee28"


def test_former_test_persistence_accepts_runtime_created_cell_dir(tmp_path) -> None:
    cell = tmp_path / "seed72" / "D0"
    cell.mkdir(parents=True)
    result = {"logical_evaluation_count": 1, "row_count": 1}
    rows = [{"example_id_hash": "abc", "vote_correct": True}]

    _write_former_test_evidence(cell, result, rows)

    assert (cell / "evaluation_summary_private.json").is_file()
    assert (cell / "former_test_rows_sanitized.jsonl").read_text(encoding="utf-8") == (
        '{"example_id_hash":"abc","vote_correct":true}\n'
    )


def test_read_json_array_accepts_history_shape_and_rejects_object(tmp_path) -> None:
    history = tmp_path / "history.json"
    history.write_text('[{"active_probe":{"vote_acc":0.5}}]', encoding="utf-8")
    assert _read_json_array(history)[0]["active_probe"]["vote_acc"] == 0.5

    history.write_text('{"active_probe":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON array"):
        _read_json_array(history)
