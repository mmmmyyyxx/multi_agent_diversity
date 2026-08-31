from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gepa_candidate_breadth_support",
    ROOT / "scripts" / "gepa_candidate_breadth_support.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(name: str, slot: int, feasible: bool, loss: int, net: int) -> dict:
    return {
        "candidate_hash": name,
        "candidate_stage": "source",
        "source_candidate_hash": name,
        "source_slot": slot,
        "valid": True,
        "feasible": feasible,
        "train_target_gain": 1,
        "train_vote_gain": net + loss,
        "train_vote_loss": loss,
        "train_vote_net": net,
        "quality_key": [net, 1, 0.0, -loss, name],
    }


def test_nested_n2_n4_pool_and_novel_safer_count() -> None:
    rows = [
        row("a", 1, True, 2, 2), row("b", 2, True, 3, 1),
        row("c", 3, True, 0, 1), row("d", 4, False, 0, 0),
    ]
    n2, n4 = MODULE.choose_pool(rows, 2), MODULE.choose_pool(rows, 4)
    MODULE.finalize_pool_comparison(n2, n4, rows)
    assert n4["n2_pool_is_subset"] is True
    assert n4["n4_only_zero_loss_feasible_count"] == 1
    assert n4["n4_only_lower_than_n2_best_loss_feasible_count"] == 1


def test_supported_label_requires_selected_loss_reduction_and_nonworse_validation() -> None:
    cases = [{
        "n2": {"feasible_candidate_count": 1, "winner_train_vote_loss": 2, "validation_vote_delta": -1},
        "n4": {
            "feasible_candidate_count": 2, "winner_train_vote_loss": 0,
            "validation_vote_delta": 0, "n4_only_zero_loss_feasible_count": 1,
            "n4_only_lower_than_n2_best_loss_feasible_count": 1,
        },
    }]
    assert MODULE.classify(cases)["final_label"] == "PROPOSAL_BREADTH_SUPPORTED"


def test_throughput_only_label_does_not_claim_quality() -> None:
    cases = [{
        "n2": {"feasible_candidate_count": 1, "winner_train_vote_loss": 1, "validation_vote_delta": 0},
        "n4": {
            "feasible_candidate_count": 3, "winner_train_vote_loss": 1,
            "validation_vote_delta": 0, "n4_only_zero_loss_feasible_count": 0,
            "n4_only_lower_than_n2_best_loss_feasible_count": 0,
        },
    }]
    assert MODULE.classify(cases)["final_label"] == "PROPOSAL_BREADTH_THROUGHPUT_ONLY"


def test_runner_requires_explicit_authorization_and_forbids_test() -> None:
    source = (ROOT / "scripts" / "run_gepa_candidate_breadth_pilot.py").read_text(encoding="utf-8")
    assert "GEPA_PROPOSAL_BREADTH_API_AUTHORIZED" not in source
    assert "AUTHORIZATION_ENV" in source
    assert '"test_calls": int(system.test_evaluation_count)' in source
    assert "team_prompt_commit_count\": 0" in source
