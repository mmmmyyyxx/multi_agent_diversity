import asyncio

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.persistence.checkpoint import build_checkpoint, restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.versions import CHECKPOINT_VERSION


async def solver(_question, agent_id, _prompt):
    answer = "A" if agent_id == 0 else "B"
    return PromptAnswer(answer, f"FINAL_ANSWER: {answer}", True)


def identity():
    return RunIdentity(
        method_version="member_aware_peer_state_v5", experiment_setting="shared_member_aware_full",
        git_commit="commit", git_dirty=False, config_fingerprint="config", manifest_sha256="manifest",
        train_file_sha256="train", val_file_sha256="val", test_file_sha256="test",
        train_question_set_hash="train-q", val_question_set_hash="val-q", test_question_set_hash="test-q",
    )


def build_system(tmp_path, run_identity=None):
    system = PromptEnsembleOptimizationSystem(
        Config.from_flat(out_dir=str(tmp_path), answer_format="option_letter"), solver=solver
    )
    system.set_run_identity(run_identity or identity())
    asyncio.run(system.initialize_fixed_probe([{"question": "q", "answer": "A"}]))
    return system


def test_v13_checkpoint_persists_final_state_lifecycle(tmp_path):
    source = build_system(tmp_path / "source")
    source.planned_update_count = 24
    source.completed_update_count = 3
    source.training_dynamics = [{"update_index": -1}, {"update_index": 0}]
    source.team_differentiation_trajectory = [{"update_index": -1}]
    source.update_transition_decomposition = [{"update_index": 0}]
    source.final_state_selection = {"selected_checkpoint_source": "final_active_state"}
    payload = build_checkpoint(source, epoch_index=1, update_index=0, training_state={"planned_update_count": 24})
    assert payload["checkpoint_version"] == CHECKPOINT_VERSION == 13
    assert "validation_state_cache" not in payload
    assert "validation_probe" not in payload
    target = build_system(tmp_path / "target")
    epoch, update, state = restore_checkpoint(target, payload)
    assert (epoch, update, state) == (1, 0, {"planned_update_count": 24})
    assert target.planned_update_count == 24
    assert target.completed_update_count == 3
    assert target.training_dynamics == source.training_dynamics
    assert target.team_differentiation_trajectory == source.team_differentiation_trajectory
    assert target.update_transition_decomposition == source.update_transition_decomposition


def test_v12_checkpoint_is_explicitly_incompatible(tmp_path):
    system = build_system(tmp_path)
    payload = build_checkpoint(system, epoch_index=0, update_index=0, training_state={})
    payload["checkpoint_version"] = 12
    with pytest.raises(ValueError, match="incompatible"):
        restore_checkpoint(system, payload)
