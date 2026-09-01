from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from scripts.model_headroom_screening_support import ARMS, MODELS, SEEDS
from scripts.model_headroom_screening_support import selection_rule


ROOT = Path(__file__).resolve().parents[1]


def row(static: float, uplift: float, gap: float, wins: int, unstable: bool = False):
    return {
        "static_mean_vote_acc": static,
        "mean_vote_uplift": uplift,
        "generic_mean_oracle_vote_gap": gap,
        "generic_vote_win_count": wins,
        "serious_output_instability": unstable,
    }


def test_only_passing_model_is_selected() -> None:
    result = selection_rule({
        "A": row(0.60, 0.05, 0.10, 2),
        "B": row(0.70, 0.05, 0.10, 3),
    })
    assert result["decision"] == "SELECT"
    assert result["selected_model_key"] == "A"


def test_both_fail_yields_hold() -> None:
    result = selection_rule({
        "A": row(0.60, 0.01, 0.10, 1),
        "B": row(0.62, 0.03, 0.07, 2),
    })
    assert result["decision"] == "HOLD"
    assert result["selected_task_model"] == ""


def test_frozen_tie_break_order() -> None:
    result = selection_rule({
        "A": row(0.60, 0.05, 0.10, 2),
        "B": row(0.61, 0.07, 0.09, 2),
    })
    assert result["selected_model_key"] == "B"
    assert result["reason"] == "larger_mean_vote_uplift"


def test_serious_instability_fails_model() -> None:
    result = selection_rule({
        "A": row(0.60, 0.05, 0.10, 2, unstable=True),
        "B": row(0.70, 0.01, 0.01, 0),
    })
    assert result["decision"] == "HOLD"
    assert result["model_evaluations"]["A"]["pass"] is False


def test_inventory_excludes_full_and_module_settings() -> None:
    assert MODELS == {"A": "qwen2.5-7b-instruct", "B": "qwen3-8b"}
    assert SEEDS == (62, 63, 64)
    assert ARMS == {
        "STATIC": "shared_static_reference",
        "GENERIC": "shared_generic_evolution",
    }


def test_validation_evaluator_does_not_load_test_split() -> None:
    source = Path("scripts/evaluate_model_headroom_validation.py").read_text(
        encoding="utf-8"
    )
    assert "_load(cfg.data.test_path" not in source
    assert '"test_calls": 0' in source


def test_runner_freezes_task_and_optimizer_roles() -> None:
    source = Path("scripts/run_model_headroom_screening.ps1").read_text(
        encoding="utf-8"
    )
    assert '"--settings", "shared_static_reference,shared_generic_evolution"' in source
    assert '"--optimizer_model", "qwen3-14b"' in source
    assert '"--evaluator_model", "qwen3-14b"' in source
    assert "Full" not in source


def test_project_scripts_are_directly_invocable() -> None:
    for script in (
        "prepare_model_headroom_screening.py",
        "audit_model_headroom_screening.py",
        "analyze_model_headroom_screening.py",
        "evaluate_model_headroom_validation.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (script, completed.stderr)
