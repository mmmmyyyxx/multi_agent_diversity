from __future__ import annotations

import importlib.util
import csv
import json
from pathlib import Path

from multi_dataset_diverse_rl.peer_state import build_team_vote_state


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_v17_hybrid_recovered_update_conversion.py"


def load():
    spec = importlib.util.spec_from_file_location("hybrid_conversion", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state(answers: list[str], gold: str = "A"):
    return build_team_vote_state(
        question_hash="q",
        gold_answer=gold,
        answers=answers,
        valid_vector=[True] * 5,
        normalize_answer=lambda value: value.strip().upper(),
        match_answer=lambda left, right: left.strip().upper() == right.strip().upper(),
        tie_break="abstain",
        seed=1,
    )


def test_singleton_oracle_gain_remains_vote_wrong() -> None:
    module = load()
    before = state(["B", "B", "C", "D", "E"])
    after = state(["A", "B", "C", "D", "E"])
    row = module.transition_row(
        transition_id="t",
        parent_id="p",
        target_member=0,
        candidate_id="c",
        conceptual_arms="HYBRID_EXPLOIT_EXPLORE",
        question_hash="q",
        before=before,
        after=after,
    )
    assert row["g_transition"] == "G0_to_G1"
    assert row["oracle_gain"] == 1
    assert row["vote_gain"] == 0
    assert row["coverage_role_after"] == "singleton_correct"
    assert row["additional_gold_votes_needed_without_wrong_reduction"] == 1
    assert row["minimum_dominant_wrong_to_gold_flips_needed"] == 1


def test_wrong_coalition_reduction_and_vote_gain_are_separate() -> None:
    module = load()
    before = state(["A", "B", "B", "C", "D"])
    after = state(["A", "A", "B", "C", "D"])
    row = module.transition_row(
        transition_id="t",
        parent_id="p",
        target_member=1,
        candidate_id="c",
        conceptual_arms="RR_TOP2",
        question_hash="q",
        before=before,
        after=after,
    )
    assert row["oracle_gain"] == 0
    assert row["vote_gain"] == 1
    assert row["wrong_coalition_direction"] == "reduced"
    assert row["G_before"] == 1 and row["G_after"] == 2


def test_summary_counts_gain_loss_independently() -> None:
    module = load()
    rows = [
        {"target_gain": 1, "target_loss": 0, "oracle_gain": 1, "oracle_loss": 0,
         "vote_gain": 0, "vote_loss": 0, "wrong_coalition_direction": "unchanged",
         "g_transition": "G0_to_G1", "coverage_role_after": "singleton_correct",
         "additional_gold_votes_needed_without_wrong_reduction": 2,
         "minimum_dominant_wrong_to_gold_flips_needed": 1},
        {"target_gain": 0, "target_loss": 1, "oracle_gain": 0, "oracle_loss": 0,
         "vote_gain": 0, "vote_loss": 0, "wrong_coalition_direction": "increased",
         "g_transition": "G2_to_G1", "coverage_role_after": "singleton_correct",
         "additional_gold_votes_needed_without_wrong_reduction": 2,
         "minimum_dominant_wrong_to_gold_flips_needed": 1},
    ]
    summary = module.summarize_rows(rows)
    assert summary["target_gain_count"] == summary["target_loss_count"] == 1
    assert summary["oracle_gain_count"] == 1
    assert summary["vote_gain_count"] == summary["vote_loss_count"] == 0
    assert summary["oracle_gain_g_transition_counts"] == {"G0_to_G1": 1}


def test_persisted_cache_identity_reconciliation_is_exact() -> None:
    module = load()
    rows = [
        ("historical", "q1", '{"answer":"A"}'),
        ("historical", "q2", '{"answer":"B"}'),
        ("current-incomplete", "q1", '{"answer":"C"}'),
    ]
    selected = module.select_persisted_identity_rows(rows, {"q1", "q2"})
    assert selected == [("q1", '{"answer":"A"}'), ("q2", '{"answer":"B"}')]


def test_ambiguous_complete_cache_identities_hard_fail() -> None:
    module = load()
    rows = [
        ("one", "q", "{}"),
        ("two", "q", "{}"),
    ]
    try:
        module.select_persisted_identity_rows(rows, {"q"})
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("ambiguous persisted identities must fail")


def test_csv_nested_values_use_canonical_json(tmp_path: Path) -> None:
    module = load()
    path = tmp_path / "rows.csv"
    module.write_csv(path, [{"name": "x", "counts": {"b": 2, "a": 1}}])
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["counts"] == '{"a":1,"b":2}'
    assert json.loads(row["counts"]) == {"a": 1, "b": 2}
