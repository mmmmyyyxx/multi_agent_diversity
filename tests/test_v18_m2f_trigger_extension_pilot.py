from __future__ import annotations

import importlib.util
from pathlib import Path

from multi_dataset_diverse_rl.compatibility_repair import (
    EXTENDED_TRAIN_VOTE_LOSS_TRIGGER_VERSION,
    REPAIR_INSTRUCTION,
    REPAIR_SYSTEM_PROMPT,
    extended_repair_eligible,
    repair_eligible,
)


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extended_trigger_adds_only_common_safe_train_vote_loss() -> None:
    common = {
        "responsibility_gain_count": 1,
        "rejection_reasons": (),
        "loss_evidence_count": 1,
    }
    assert repair_eligible(**common) is False
    assert extended_repair_eligible(
        **common, source_common_safe=True, source_vote_loss_count=1
    ) is True
    assert extended_repair_eligible(
        **common, source_common_safe=True, source_vote_loss_count=0
    ) is False
    assert extended_repair_eligible(
        **common, source_common_safe=False, source_vote_loss_count=1
    ) is False


def test_original_trigger_remains_available_without_extended_condition() -> None:
    values = {
        "responsibility_gain_count": 1,
        "rejection_reasons": ("team_vote_regression",),
        "loss_evidence_count": 1,
    }
    assert repair_eligible(**values) is True
    assert extended_repair_eligible(
        **values, source_common_safe=False, source_vote_loss_count=0
    ) is True


def test_trigger_and_repair_semantics_are_separately_versioned() -> None:
    assert EXTENDED_TRAIN_VOTE_LOSS_TRIGGER_VERSION == (
        "common_safe_train_vote_loss_extended_trigger_v1"
    )
    assert "SOURCE CANDIDATE" in REPAIR_INSTRUCTION
    assert "strict JSON" in REPAIR_SYSTEM_PROMPT


def test_phase_a_inventory_and_classifier_are_frozen() -> None:
    module = load_script("prepare_v18_m2f_trigger_extension_pilot.py")
    assert module.CASES == ((59, 3), (61, 5))
    assert module.ARM == "HYBRID_BASE"
    assert module.CLASSIFIER_LABELS == (
        "EXTENDED_M2F_WRITEBACK_RISK_REDUCTION_SUPPORTED",
        "EXTENDED_M2F_TRAIN_COLLATERAL_REDUCTION_ONLY",
        "EXTENDED_M2F_TRIGGER_NOT_SUPPORTED",
        "EXTENDED_M2F_HARMFUL",
    )


def test_runner_requires_explicit_authorization_and_has_no_test_path() -> None:
    module = load_script("run_v18_m2f_trigger_extension_pilot.py")
    assert module.AUTH_ENV == "V18_M2F_TRIGGER_EXTENSION_AUTHORIZED"
    source = (ROOT / "scripts" / "run_v18_m2f_trigger_extension_pilot.py").read_text(
        encoding="utf-8"
    )
    assert '"new_test_calls": 0' in source
    assert "evaluate_final_test" not in source
    assert "commit_prompt" not in source
