# Historical filename retained for compatibility with prior test invocation;
# assertions below describe the version-agnostic active final-state lifecycle.
import asyncio
import json

import pytest

import multi_dataset_diverse_rl.cli as cli
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.persistence.checkpoint import build_checkpoint, restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


def _split(tmp_path, name):
    path = tmp_path / f"{name}.jsonl"
    path.write_text(json.dumps({"question": f"{name}-q", "answer": "A"}) + "\n", encoding="utf-8")
    return str(path)


def test_cli_uses_final_active_state_without_validation(monkeypatch, tmp_path):
    events = []

    async def solver(question, _agent_id, prompt):
        events.append(("solver", question.split("-", 1)[0], prompt))
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    class FinalStateSystem(PromptEnsembleOptimizationSystem):
        def __init__(self, cfg):
            super().__init__(cfg, solver=solver)

        async def update_once(self, update_index):
            events.append(("update", update_index))
            self.agents[0].current_prompt = f"changed-{update_index}"
            return update_index == 0

    monkeypatch.setattr(cli, "PromptEnsembleOptimizationSystem", FinalStateSystem)
    cfg = Config.from_flat(
        train_path=_split(tmp_path, "train"),
        val_path=_split(tmp_path, "val"),
        test_path=_split(tmp_path, "test"),
        train_size=1,
        val_size=1,
        test_size=1,
        answer_format="option_letter",
        epochs=3,
        update_every=1,
        out_dir=str(tmp_path / "run"),
    )
    result = asyncio.run(cli.run(cfg))
    selection = result["selection_summary"]
    assert not [row for row in events if row[:2] == ("solver", "val")]
    assert [row[1] for row in events if row[0] == "update"] == [0, 1, 2]
    assert selection["validation_used"] is False
    assert selection["validation_evaluation_count"] == 0
    assert selection["selected_checkpoint_source"] == "final_active_state"
    assert selection["selected_checkpoint_update_index"] == 3
    assert selection["selected_epoch"] == 3
    assert selection["test_evaluation_count"] == 1
    assert selection["test_called_before_training_complete"] is False
    assert selection["test_used_for_training"] is False


def test_short_mechanism_run_can_skip_test_without_relaxing_final_state_selection(monkeypatch, tmp_path):
    events = []

    async def solver(_question, _agent_id, _prompt):
        events.append("solver")
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    class ShortSystem(PromptEnsembleOptimizationSystem):
        def __init__(self, cfg):
            super().__init__(cfg, solver=solver)

        async def update_once(self, update_index):
            events.append(("update", update_index))
            return False

    monkeypatch.setattr(cli, "PromptEnsembleOptimizationSystem", ShortSystem)
    cfg = Config.from_flat(
        train_path=_split(tmp_path, "train"), val_path=_split(tmp_path, "val"),
        test_path=_split(tmp_path, "test"), train_size=1, val_size=1, test_size=1,
        answer_format="option_letter", epochs=2, update_every=1,
        final_test_enabled=False, out_dir=str(tmp_path / "run"),
    )
    result = asyncio.run(cli.run(cfg))
    selection = result["selection_summary"]
    assert result["selected_test"] is None
    assert selection["selected_checkpoint_source"] == "final_active_state"
    assert selection["selected_checkpoint_update_index"] == 2
    assert selection["test_evaluation_count"] == 0
    assert selection["final_test_enabled"] is False


def test_final_test_is_forbidden_before_training_complete_and_cached(tmp_path):
    calls = 0

    async def solver(_question, _agent_id, _prompt):
        nonlocal calls
        calls += 1
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    system = PromptEnsembleOptimizationSystem(Config.from_flat(out_dir=str(tmp_path)), solver=solver)
    data = [{"question": "q", "answer": "A"}]

    async def run():
        with pytest.raises(RuntimeError, match="before training completes"):
            await system.evaluate_final_test(data)
        system.completed_update_count = 0
        system.mark_training_complete(0)
        first = await system.evaluate_final_test(data)
        second = await system.evaluate_final_test(data)
        return first, second

    first, second = asyncio.run(run())
    assert first == second
    assert calls == 1
    assert system.test_evaluation_count == 1


def test_frozen_initialization_snapshot_is_hash_only_and_deterministic(tmp_path):
    async def solver(_question, _agent_id, _prompt):
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    identity = RunIdentity(
        method_version="member_aware_peer_state_v8", experiment_setting="shared_member_aware_full",
        git_commit="commit", git_dirty=False, config_fingerprint="config", manifest_sha256="manifest",
        train_file_sha256="train", val_file_sha256="val", test_file_sha256="test",
        train_question_set_hash="train-q", val_question_set_hash="val-q", test_question_set_hash="test-q",
    )

    async def run_snapshot():
        system = PromptEnsembleOptimizationSystem(
            Config.from_flat(out_dir=str(tmp_path)), solver=solver,
        )
        system.set_run_identity(identity)
        await system.initialize_fixed_probe([{"question": "q", "answer": "A"}])
        return system.frozen_initialization_snapshot()

    first = asyncio.run(run_snapshot())
    second = asyncio.run(run_snapshot())
    assert first == second
    assert first["initial_member_correct_counts"] == [1, 1, 1, 1, 1]
    assert len(first["initial_train_state_hash"]) == 64
    assert "careful reasoning solver" not in json.dumps(first).lower()


def test_checkpoint_reuses_completed_final_test(tmp_path):
    identity = RunIdentity(
            method_version="member_aware_peer_state_v8", experiment_setting="shared_member_aware_full",
        git_commit="commit", git_dirty=False, config_fingerprint="config", manifest_sha256="manifest",
        train_file_sha256="train", val_file_sha256="val", test_file_sha256="test",
        train_question_set_hash="train-q", val_question_set_hash="val-q", test_question_set_hash="test-q",
    )
    calls = {"source": 0, "target": 0}

    def make_solver(kind):
        async def solver(_question, _agent_id, _prompt):
            calls[kind] += 1
            return PromptAnswer("A", "FINAL_ANSWER: A", True)
        return solver

    async def run():
        source = PromptEnsembleOptimizationSystem(Config.from_flat(out_dir=str(tmp_path / "source")), solver=make_solver("source"))
        source.set_run_identity(identity)
        await source.initialize_fixed_probe([{"question": "train", "answer": "A"}])
        source.completed_update_count = 0
        source.mark_training_complete(0)
        expected = await source.evaluate_final_test([{"question": "test", "answer": "A"}])
        payload = build_checkpoint(source, epoch_index=0, update_index=0, training_state={"planned_update_count": 0})
        target = PromptEnsembleOptimizationSystem(Config.from_flat(out_dir=str(tmp_path / "source")), solver=make_solver("target"))
        target.set_run_identity(identity)
        target.fixed_probe = target.build_probe([{"question": "train", "answer": "A"}])
        restore_checkpoint(target, payload)
        actual = await target.evaluate_final_test([{"question": "test", "answer": "A"}])
        return expected, actual

    expected, actual = asyncio.run(run())
    assert expected == actual
    assert calls["target"] == 0
