import asyncio
from dataclasses import replace

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.persistence.checkpoint import build_checkpoint, restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


async def solver(_question, agent_id, _prompt):
    answer = "A" if agent_id == 0 else "B"
    return PromptAnswer(answer=answer, trace=f"reason FINAL_ANSWER: {answer}", valid=True)


def identity():
    return RunIdentity(
        method_version="peer_state_counterfactual_v2",
        experiment_setting="shared_peer_state_full",
        git_commit="commit",
        git_dirty=False,
        config_fingerprint="config",
        manifest_sha256="manifest",
        train_file_sha256="train",
        val_file_sha256="val",
        test_file_sha256="test",
        train_question_set_hash="train-q",
        val_question_set_hash="val-q",
        test_question_set_hash="test-q",
    )


def build_system(tmp_path, run_identity=None):
    selected_identity = run_identity or identity()
    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        answer_format="option_letter",
        experiment_setting=selected_identity.experiment_setting,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=solver)
    system.set_run_identity(selected_identity)
    data = [{"question": "q", "answer": "A"}]
    system.validation_probe = system.build_validation_probe(data)
    asyncio.run(system.initialize_fixed_probe(data))
    return system


def test_current_checkpoint_exact_resume_and_owner_state(tmp_path):
    source = build_system(tmp_path / "source")
    source.responsibility_state.primary_owner_by_question = {"q": 3}
    source.responsibility_state.owner_age_by_question = {"q": 2}
    payload = build_checkpoint(source, epoch_index=1, update_index=2, best_state={"epoch": 0})
    assert "shared_solver_cache_audit" in payload
    assert payload["shared_solver_cache_audit"]["ready_entries"] == 1
    target = build_system(tmp_path / "target")
    epoch, update, best = restore_checkpoint(target, payload)
    assert (epoch, update, best) == (1, 2, {"epoch": 0})
    assert target.responsibility_state.primary_owner_by_question == {"q": 3}
    assert target.responsibility_state.owner_age_by_question == {"q": 2}
    assert target.fixed_probe.to_dict() == source.fixed_probe.to_dict()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("config_fingerprint", "different-seed-model-tie-or-constraint"),
        ("train_file_sha256", "different-split"),
        ("git_commit", "different-commit"),
        ("experiment_setting", "shared_peer_state_responsibility"),
    ],
)
def test_any_run_identity_mismatch_rejects_resume(tmp_path, field, value):
    source = build_system(tmp_path / "source")
    payload = build_checkpoint(source, epoch_index=0, update_index=0, best_state={})
    target = build_system(tmp_path / "target", replace(identity(), **{field: value}))
    with pytest.raises(ValueError, match="Run identity mismatch"):
        restore_checkpoint(target, payload)


def test_old_checkpoint_and_probe_mismatch_fail_explicitly(tmp_path):
    system = build_system(tmp_path)
    with pytest.raises(ValueError, match="lacks exact run identity"):
        restore_checkpoint(system, {"checkpoint_version": 1, "method_version": "old"})
    payload = build_checkpoint(system, epoch_index=0, update_index=0, best_state={})
    payload["probe_version"] = "stale"
    with pytest.raises(ValueError, match="Fixed probe"):
        restore_checkpoint(system, payload)
