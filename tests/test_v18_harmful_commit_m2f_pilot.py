from __future__ import annotations

import importlib.util
from pathlib import Path

from multi_dataset_diverse_rl.compatibility_repair import repair_eligible


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_v18_harmful_commit_m2f_repair_pilot.py"
SPEC = importlib.util.spec_from_file_location("v18_harmful_m2f", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_m2f_does_not_repair_common_safe_source() -> None:
    assert repair_eligible(
        responsibility_gain_count=3,
        rejection_reasons=(),
        loss_evidence_count=2,
    ) is False


def test_frozen_m2f_requires_collateral_rejection_reason() -> None:
    assert repair_eligible(
        responsibility_gain_count=1,
        rejection_reasons=("team_vote_regression",),
        loss_evidence_count=1,
    ) is True


def test_fixed_case_inventory_and_no_test_inputs() -> None:
    assert MODULE.CASES == ((59, 3), (61, 5))
    assert MODULE.ARM == "HYBRID_BASE"
    assert all("test" not in path.lower() for path in MODULE.SOURCE_FILES)


def test_phase_a_script_imports_from_external_working_directory() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
