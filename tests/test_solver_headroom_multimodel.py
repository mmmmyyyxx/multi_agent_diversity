from __future__ import annotations

from pathlib import Path

from scripts.solver_headroom_multimodel_support import CANDIDATES, ROLE_MODEL, SEED


ROOT=Path(__file__).resolve().parents[1]


def test_frozen_candidates_and_seed() -> None:
    assert SEED == 65
    assert [m for _,m in CANDIDATES] == ["qwen-turbo","qwen3-4b","qwen-flash","qwen3-8b","deepseek-r1-distill-qwen-7b","deepseek-r1-distill-llama-8b","deepseek-r1-distill-qwen-1.5b","glm-4.5-air"]
    assert ROLE_MODEL == "qwen3.7-flash"


def test_runner_has_only_static_generic() -> None:
    source=(ROOT/"scripts/run_solver_headroom_multimodel.ps1").read_text(encoding="utf-8")
    assert "shared_static_reference" in source and "shared_generic_evolution" in source
    for forbidden in ("shared_member_aware","M20","M2F","Hybrid"):
        assert forbidden not in source


def test_evaluator_never_loads_test() -> None:
    source=(ROOT/"scripts/evaluate_solver_headroom_multimodel.py").read_text(encoding="utf-8")
    assert "cfg.data.test_path" not in source
    assert '"test_calls": 0' in source


def test_preregistration_freezes_static_gate() -> None:
    source=(ROOT/"experiments/solver_headroom_multimodel_seed65_20260901/PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "0.50 <= Static Validation VoteAcc <= 0.64" in source
    assert "OracleAcc - VoteAcc >= 0.08" in source
    assert "At most three" in source
