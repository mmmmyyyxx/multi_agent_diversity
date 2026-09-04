from __future__ import annotations

import pytest

from scripts.audit_diversity_matrix_split_balance import (
    core_sentence,
    js_divergence,
    label_standardization,
    label,
    plurality_outcome,
    standardized_difference,
    text_features,
    total_variation,
)


def test_label_and_plurality_contract() -> None:
    assert label("(B)") == "B"
    assert label("B") == "B"
    assert label("unknown") == "INVALID"
    assert plurality_outcome(["A", "A", "A", "B", "C"], "A") == (True, True)
    assert plurality_outcome(["A", "A", "B", "B", "C"], "A") == (False, True)
    assert plurality_outcome(["B", "B", "B", "C", "C"], "A") == (False, False)


def test_text_features_use_core_sentence() -> None:
    question = "Instruction.\nSentence: Alice told Bob that she was not ready, but he waited.\nOptions:\n(A) Alice\n(B) Bob\n(C) Ambiguous"
    assert core_sentence(question) == "Alice told Bob that she was not ready, but he waited."
    features = text_features(question)
    assert features["pronoun_count"] == 2
    assert features["negation_count"] == 1
    assert features["conjunction_count"] == 1
    assert features["option_count"] == 3


def test_distribution_effect_sizes() -> None:
    left = {"A": 0.5, "B": 0.5}
    right = {"A": 0.25, "B": 0.75}
    assert total_variation(left, right) == pytest.approx(0.25)
    assert js_divergence(left, left) == pytest.approx(0.0)
    assert js_divergence(left, right) > 0
    assert standardized_difference([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_label_standardization_telescopes() -> None:
    items = {
        "validation50": [
            {"gold_label": "A"}, {"gold_label": "B"},
        ],
        "former_test125": [
            {"gold_label": "A"}, {"gold_label": "A"}, {"gold_label": "A"}, {"gold_label": "B"},
        ],
    }
    rows = []
    for arm in ("D0", "D1", "D2", "D3", "D4", "D5"):
        rows.extend([
            {"split": "validation50", "arm": arm, "label": "A", "mean_vote_accuracy": 0.5},
            {"split": "validation50", "arm": arm, "label": "B", "mean_vote_accuracy": 1.0},
            {"split": "validation50", "arm": arm, "label": "C", "mean_vote_accuracy": 0.0},
            {"split": "former_test125", "arm": arm, "label": "A", "mean_vote_accuracy": 0.75},
            {"split": "former_test125", "arm": arm, "label": "B", "mean_vote_accuracy": 0.5},
            {"split": "former_test125", "arm": arm, "label": "C", "mean_vote_accuracy": 0.0},
        ])
    result = label_standardization(items, rows)[0]
    assert result["raw_gap"] == pytest.approx(0.6875 - 0.75)
    assert result["label_composition_component"] == pytest.approx(0.625 - 0.75)
    assert result["within_label_residual_component"] == pytest.approx(0.6875 - 0.625)
