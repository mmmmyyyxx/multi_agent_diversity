from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.solver_headroom_screening_support import (
    ARMS, CANDIDATES, ROLE_MODEL, SEEDS, select_solver,
)


ROOT = Path(__file__).resolve().parents[1]


def row(static: float, uplift: float, gap: float, wins: int, unstable: bool = False):
    return {
        "solver_model": "model",
        "static_mean_vote_acc": static,
        "mean_vote_uplift": uplift,
        "generic_mean_oracle_vote_gap": gap,
        "generic_vote_win_count": wins,
        "serious_output_instability": unstable,
    }


def test_frozen_inventory() -> None:
    assert CANDIDATES == (
        ("A", "qwen3-8b"),
        ("B", "qwen3-4b-instruct-2507"),
        ("C", "qwen3-1.7b"),
    )
    assert SEEDS == (65, 66, 67)
    assert ARMS == {
        "STATIC": "shared_static_reference",
        "GENERIC": "shared_generic_evolution",
    }
    assert ROLE_MODEL == "qwen3.7-flash"


def test_only_passing_solver_is_selected() -> None:
    result = select_solver({
        "A": row(0.64, 0.05, 0.09, 2),
        "B": row(0.66, 0.08, 0.12, 3),
    })
    assert result["decision"] == "SELECT"
    assert result["selected_solver_key"] == "A"


def test_no_solver_passes_yields_hold() -> None:
    result = select_solver({"A": row(0.64, 0.03, 0.09, 2)})
    assert result["decision"] == "HOLD"


def test_tie_break_and_exact_tie() -> None:
    result = select_solver({
        "A": row(0.62, 0.05, 0.10, 2),
        "B": row(0.60, 0.05, 0.10, 2),
    })
    assert result["selected_solver_key"] == "B"
    tied = select_solver({
        "A": row(0.60, 0.05, 0.10, 2),
        "B": row(0.60, 0.05, 0.10, 2),
    })
    assert tied["decision"] == "HOLD"


def test_validation_evaluator_never_loads_test() -> None:
    source = (ROOT / "scripts" / "evaluate_solver_headroom_validation.py").read_text(encoding="utf-8")
    assert "cfg.data.test_path" not in source
    assert '"test_calls": 0' in source


def test_runner_has_only_static_generic_and_flash_roles() -> None:
    source = (ROOT / "scripts" / "run_solver_headroom_screening.ps1").read_text(encoding="utf-8")
    assert '"--settings", "shared_static_reference,shared_generic_evolution"' in source
    assert '"--optimizer_model", "qwen3.7-flash"' in source
    assert '"--evaluator_model", "qwen3.7-flash"' in source
    for forbidden in ("shared_member_aware", "M20", "M2F", "Hybrid"):
        assert forbidden not in source


def test_smoke_does_not_request_denied_qwen25() -> None:
    source = (ROOT / "scripts" / "run_solver_headroom_availability_smoke.py").read_text(encoding="utf-8")
    assert "qwen2.5-7b-instruct" not in source


def test_scripts_are_directly_invocable() -> None:
    for script in (
        "prepare_solver_headroom_screening.py",
        "run_solver_headroom_availability_smoke.py",
        "audit_solver_headroom_screening.py",
        "evaluate_solver_headroom_validation.py",
        "analyze_solver_headroom_screening.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, (script, completed.stderr)
