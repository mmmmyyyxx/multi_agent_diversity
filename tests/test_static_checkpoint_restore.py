from __future__ import annotations

import asyncio

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.persistence.checkpoint import build_checkpoint, restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.versions import METHOD_VERSION


async def solver(_question, agent_id, _prompt):
    answer = "A" if agent_id == 0 else "B"
    return PromptAnswer(answer, f"FINAL_ANSWER: {answer}", True)


def build_system(tmp_path, setting: str) -> PromptEnsembleOptimizationSystem:
    system = PromptEnsembleOptimizationSystem(
        Config.from_flat(
            out_dir=str(tmp_path),
            experiment_setting=setting,
            answer_format="option_letter",
            initialization_mode="provided_prompt_set",
            provided_prompts_json='["p0", "p1", "p2", "p3", "p4"]',
        ),
        solver=solver,
    )
    system.set_run_identity(RunIdentity(
        method_version=METHOD_VERSION,
        experiment_setting=setting,
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
    ))
    asyncio.run(system.initialize_fixed_probe([{"question": "q", "answer": "A"}]))
    return system


@pytest.mark.parametrize(
    "setting",
    ("shared_static_reference", "shared_generic_evolution"),
)
def test_disabled_responsibility_checkpoint_restores_empty_cache(tmp_path, setting):
    source = build_system(tmp_path / "source", setting)
    assert source.protocol.service_routing_enabled is False
    assert source.cached_responsibility_eligibility == {}
    payload = build_checkpoint(
        source, epoch_index=0, update_index=0, training_state={}
    )
    target = build_system(tmp_path / "target", setting)
    target.proposal_memory_run_id = str(payload["proposal_memory_run_id"])

    restore_checkpoint(target, payload)

    assert target.team_prompt_state_hash() == source.team_prompt_state_hash()
    assert target.active_profiles == source.active_profiles
    assert target.cached_responsibility_eligibility == {}
    assert target.cached_responsibility_assignments == {
        agent_id: [] for agent_id in range(5)
    }


def test_disabled_responsibility_checkpoint_rejects_cached_eligibility(tmp_path):
    source = build_system(tmp_path / "source", "shared_static_reference")
    payload = build_checkpoint(
        source, epoch_index=0, update_index=0, training_state={}
    )
    payload["cached_responsibility_eligibility"] = {"contaminated": [0]}
    target = build_system(tmp_path / "target", "shared_static_reference")
    target.proposal_memory_run_id = str(payload["proposal_memory_run_id"])

    with pytest.raises(ValueError, match="disabled responsibility protocol"):
        restore_checkpoint(target, payload)
