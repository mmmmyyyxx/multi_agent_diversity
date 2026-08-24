from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_v18_trajectory_gain_loss_decomposition.py"
SPEC = importlib.util.spec_from_file_location("v18_trajectory_decomposition", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_gain_persistence_classes_are_exhaustive() -> None:
    cases = {
        (True, True, True): "retained_to_final",
        (True, False, True): "overwritten_then_recovered_to_final",
        (True, False, True, False): "overwritten_then_recovered_but_not_final",
        (True, False, False): "overwritten_not_recovered",
    }
    for sequence, expected in cases.items():
        assert MODULE.classify_gain_persistence(list(sequence))["persistence_class"] == expected


def test_loss_origin_tracks_current_correct_spell() -> None:
    assert MODULE.loss_origin([True, True]) == "new_collateral_regression"
    assert MODULE.loss_origin([False, True, True]) == "prior_conversion_overwritten"
    assert MODULE.loss_origin([True, False, True]) == "prior_conversion_overwritten"


def _state(index: int, after: int, votes: tuple[bool, bool]) -> dict:
    examples = [
        {"example_id_hash": f"h{item}", "vote_correct": value}
        for item, value in enumerate(votes)
    ]
    return {
        "state_index": index,
        "after_update_index": after,
        "team_state_hash": f"team{index}",
        "metrics": {"vote_correct_count": sum(votes)},
        "examples": examples,
    }


def test_trajectory_telescope_and_loss_provenance() -> None:
    states = [
        _state(0, -1, (True, False)),
        _state(1, 0, (False, True)),
        _state(2, 1, (True, False)),
    ]
    updates = [
        {
            "update_index": 0, "committed": True, "validation_evaluated": True,
            "validation_state_index_before": 0, "validation_state_index_after": 1,
            "validation_vote_delta": 0, "train_vote_delta": 1, "train_target_delta": 2,
            "parent_team_hash": "team0", "successor_team_hash": "team1", "committed_target": 2,
        },
        {
            "update_index": 1, "committed": True, "validation_evaluated": True,
            "validation_state_index_before": 1, "validation_state_index_after": 2,
            "validation_vote_delta": 0, "train_vote_delta": 0, "train_target_delta": 1,
            "parent_team_hash": "team1", "successor_team_hash": "team2", "committed_target": 3,
        },
    ]
    result = MODULE.decompose_trajectory(
        seed=1, arm="HYBRID_BASE", states=states, updates=updates,
        summary={"new_test_calls": 0, "infrastructure_failure_count": 0, "accepted_commit_count": 2},
    )
    assert result["trajectory_row"]["transition_net_sum"] == 0
    assert result["trajectory_row"]["telescoping_identity_pass"] is True
    origins = [row["loss_origin"] for row in result["loss_rows"]]
    assert origins == ["new_collateral_regression", "prior_conversion_overwritten"]
    assert all(row["simultaneous_gain_and_loss"] for row in result["commit_rows"])


def test_classifier_uses_arm_level_quality_without_commit_matching() -> None:
    commits = [
        {"arm": "W1_TOP2", "validation_net_delta": 1, "validation_gain_count": 1,
         "validation_loss_count": 0, "simultaneous_gain_and_loss": False,
         "train_vote_progress_not_transferred": False},
        {"arm": "HYBRID_BASE", "validation_net_delta": 0, "validation_gain_count": 1,
         "validation_loss_count": 1, "simultaneous_gain_and_loss": True,
         "train_vote_progress_not_transferred": True},
        {"arm": "HYBRID_BASE", "validation_net_delta": -1, "validation_gain_count": 0,
         "validation_loss_count": 1, "simultaneous_gain_and_loss": False,
         "train_vote_progress_not_transferred": False},
    ]
    gains = [{"arm": "HYBRID_BASE", "overwritten_later": True}]
    losses = [{"arm": "HYBRID_BASE", "loss_origin": "new_collateral_regression"}]
    result = MODULE.classify_bottlenecks(commits, gains, losses)
    assert all(result["flags"].values())
    assert result["rules_frozen_before_result_readout"] is True


def test_analysis_input_allowlist_excludes_test_artifacts() -> None:
    assert MODULE.ALLOWED_INPUT_FILES == (
        "validation_states.jsonl",
        "update_lineage.jsonl",
        "online_run_summary.json",
    )
    assert all("test" not in name.lower() for name in MODULE.ALLOWED_INPUT_FILES)


def test_published_decomposition_facts_if_report_exists() -> None:
    report = ROOT / "reports" / "v18_trajectory_gain_loss_decomposition_20260824"
    if not report.is_dir():
        return
    import json

    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    facts = json.loads((report / "fact_assertions.json").read_text(encoding="utf-8"))
    assert summary["telescoping_identity_pass_count"] == 6
    assert summary["scope"]["new_api_calls"] == 0
    assert summary["scope"]["new_test_calls"] == 0
    assert summary["scope"]["cross_arm_commit_matching"] is False
    assert facts["hybrid_gain_overwritten_later_count"] == 0
    assert facts["hybrid_new_collateral_loss_count"] == 7
