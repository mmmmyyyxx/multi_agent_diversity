from __future__ import annotations

from pathlib import Path

from scripts.solver_headroom_multimodel_support import CANDIDATES, ROLE_MODEL, SEED


ROOT=Path(__file__).resolve().parents[1]


def test_frozen_candidates_and_seed() -> None:
    assert SEED == 65
    assert [m for _,m in CANDIDATES] == ["qwen-turbo","qwen3-4b","qwen-flash","qwen3-8b","deepseek-r1-distill-qwen-7b","deepseek-r1-distill-llama-8b","deepseek-r1-distill-qwen-1.5b","glm-4.5-air"]
    assert ROLE_MODEL == "qwen3.7-flash"


def test_retry_runner_has_only_generic_and_never_resumes_static() -> None:
    source=(ROOT/"scripts/run_solver_headroom_multimodel.ps1").read_text(encoding="utf-8")
    assert '$Settings="shared_generic_evolution"' in source
    assert '--resume_completed 0' in source
    assert '--optimized_only 1' in source
    assert 'static_selection_retry2_private.json' in source
    for forbidden in ("shared_member_aware","M20","M2F","Hybrid"):
        assert forbidden not in source


def test_evaluator_never_loads_test() -> None:
    source=(ROOT/"scripts/evaluate_solver_headroom_multimodel.py").read_text(encoding="utf-8")
    assert "cfg.data.test_path" not in source
    assert '"test_calls": 0' in source
    assert 'validation_retry1_freeze' not in source  # symbolic constant only


def test_retry_paths_are_fresh_and_old_failure_is_not_reused() -> None:
    source=(ROOT/"scripts/solver_headroom_multimodel_support.py").read_text(encoding="utf-8")
    assert 'VALIDATION_ROOT = RUN_ROOT / "validation_retry1"' in source
    assert 'GENERIC_ROOT = RUN_ROOT / "generic_retry4"' in source
    assert 'RUN_ROOT / "validation"' not in source


def test_retry_prepare_script_is_directly_executable() -> None:
    source=(ROOT/"scripts/prepare_solver_headroom_multimodel_retry.py").read_text(encoding="utf-8")
    assert 'Path(__file__).resolve().parents[1]' in source
    assert 'sys.path.insert(0, str(ROOT))' in source
    assert 'checkpoint.get("training_completed")' in source
    assert 'checkpoint.get("training_complete")' not in source


def test_preregistration_freezes_static_gate() -> None:
    source=(ROOT/"experiments/solver_headroom_multimodel_seed65_20260901/PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "0.50 <= Static Validation VoteAcc <= 0.64" in source
    assert "OracleAcc - VoteAcc >= 0.08" in source
    assert "At most three" in source
    assert "Static gate v2 structural amendment" in source
    assert "structurally impossible prefilter is removed" in source


def test_static_gate_v2_does_not_use_static_oracle_gap_for_qualification() -> None:
    source=(ROOT/"scripts/select_solver_headroom_multimodel.py").read_text(encoding="utf-8")
    qualification=next(line for line in source.splitlines() if "qualified=" in line)
    assert '0.50 <= val["vote_accuracy"] <= 0.64' in qualification
    assert "gap" not in qualification
    assert "invalid_rate <= 0.01" in qualification


def test_preflight_does_not_receive_runner_only_shared_cache_argument() -> None:
    source=(ROOT/"scripts/run_solver_headroom_multimodel.ps1").read_text(encoding="utf-8")
    preflight_line=next(line for line in source.splitlines() if "preflight_member_aware.py" in line)
    runner_line=next(line for line in source.splitlines() if "run_task_level_accuracy.py" in line)
    assert "shared_solver_cache_path" not in preflight_line
    assert "shared_solver_cache_path" not in runner_line
    assert "Copy-Item -LiteralPath $StaticFreeze" in source
    assert "Split-Path (Split-Path $StaticRun -Parent) -Parent" in source
    assert '"_frozen_initialization\\disambiguation_qa\\seed65"' in source
