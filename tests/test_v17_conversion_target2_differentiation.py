from __future__ import annotations

from scripts.analyze_v17_conversion_target2_differentiation import (
    PROSPECTIVE_PARENTS,
    SIGNALS,
    analyze,
    is_conversion_residual,
    select_signal_target,
)


def test_conversion_residual_definition_is_strict() -> None:
    assert not is_conversion_residual({"gold_vote_count": 0, "largest_wrong_vote_count": 3})
    assert is_conversion_residual({"gold_vote_count": 1, "largest_wrong_vote_count": 3})
    assert is_conversion_residual({"gold_vote_count": 2, "largest_wrong_vote_count": 2})
    assert not is_conversion_residual({"gold_vote_count": 3, "largest_wrong_vote_count": 2})


def test_signal_selector_uses_rr_only_as_tie_break() -> None:
    scores = {
        4: {signal: 2 for signal in SIGNALS},
        1: {signal: 3 for signal in SIGNALS},
        3: {signal: 3 for signal in SIGNALS},
    }
    assert select_signal_target([4, 1, 3], scores, SIGNALS[0]) == 1


def test_all_historical_and_prospective_inventories_reconstruct() -> None:
    result = analyze()
    assert result["historical_state_count"] == 24
    assert result["conversion_active_state_count"] == 14
    assert result["prospective_five_parent_count"] == len(PROSPECTIVE_PARENTS) == 5
    assert not result["candidate_outcomes_used"]
    assert not result["validation_used"]
    assert not result["test_used"]


def test_simple_signals_create_nonzero_prospective_interventions() -> None:
    result = analyze()
    expected = {
        "conversion_responsibility_count": 3,
        "singleton_conversion_count": 2,
        "direct_vote_flip_count": 3,
        "dominant_wrong_weakening_count": 3,
    }
    assert {
        signal: result["signals"][signal]["prospective_five_parent"][
            "target2_different_count"
        ]
        for signal in SIGNALS
    } == expected


def test_dominant_wrong_count_adds_no_new_discrimination() -> None:
    result = analyze()
    assert result["dominant_wrong_signal_identical_to_conversion_count"]
    assert (
        result["signals"]["dominant_wrong_weakening_count"]
        == result["signals"]["conversion_responsibility_count"]
    )
