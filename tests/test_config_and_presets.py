import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.experiment_config import DEFAULT_EXPERIMENT_SETTING_NAMES, select_settings


def test_config_has_section_only_access_and_frozen_field_count():
    cfg = Config()
    assert len(cfg.to_flat_dict()) == 74
    assert cfg.training.method_version == "peer_state_counterfactual_v1"
    with pytest.raises(AttributeError):
        _ = cfg.method_version


def test_only_five_settings_exist_and_old_setting_fails():
    assert DEFAULT_EXPERIMENT_SETTING_NAMES == [
        "shared_baseline", "shared_independent_accuracy_tcs",
        "shared_peer_state_credit_round_robin", "shared_peer_state_responsibility",
        "shared_peer_state_full",
    ]
    with pytest.raises(ValueError, match="Unknown experiment setting"):
        select_settings("shared_" + "v" + "9_sequential_accuracy")


def test_run_metadata_declares_method_boundaries(tmp_path):
    system = PromptEnsembleOptimizationSystem(Config.from_flat(out_dir=str(tmp_path)))
    metadata = system.run_meta()
    assert metadata["method_version"] == "peer_state_counterfactual_v1"
    metadata_key = "joint_" + "team_enumeration_enabled"
    assert metadata[metadata_key] is False
    assert metadata["generic_diversity_reward_used"] is False
    assert metadata["trace_diversity_used_for_selection"] is False
    assert metadata["legacy_compatibility_enabled"] is False
    system.flush_artifacts()
    persisted = __import__("json").loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert persisted[metadata_key] is False
