import asyncio

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.persistence.checkpoint import build_checkpoint, restore_checkpoint
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


async def solver(_question, agent_id, _prompt):
    answer = "A" if agent_id == 0 else "B"
    return PromptAnswer(answer=answer, trace=f"reason FINAL_ANSWER: {answer}", valid=True)


def build_system(tmp_path):
    cfg = Config.from_flat(out_dir=str(tmp_path), answer_format="option_letter")
    system = PromptEnsembleOptimizationSystem(cfg, solver=solver)
    asyncio.run(system.initialize_fixed_probe([{"question": "q", "answer": "A"}]))
    return system


def test_current_checkpoint_exact_resume_and_owner_state(tmp_path):
    source = build_system(tmp_path / "source")
    source.responsibility_state.previous_primary_owner_by_question = {"q": 3}
    payload = build_checkpoint(source, epoch_index=1, update_index=2, best_state={"epoch": 0})
    target = build_system(tmp_path / "target")
    epoch, update, best = restore_checkpoint(target, payload)
    assert (epoch, update, best) == (1, 2, {"epoch": 0})
    assert target.responsibility_state.previous_primary_owner_by_question == {"q": 3}
    assert target.fixed_probe.to_dict() == source.fixed_probe.to_dict()


@pytest.mark.parametrize("legacy_version", ["residual_v7", "quality_lineage_v8", "state_accuracy_v9"])
def test_legacy_and_probe_mismatch_fail_explicitly(tmp_path, legacy_version):
    system = build_system(tmp_path)
    with pytest.raises(ValueError, match="Legacy checkpoint is incompatible"):
        restore_checkpoint(system, {"checkpoint_version": 6, "method_version": legacy_version})
    payload = build_checkpoint(system, epoch_index=0, update_index=0, best_state={})
    payload["probe_version"] = "stale"
    with pytest.raises(ValueError, match="Fixed probe"):
        restore_checkpoint(system, payload)
