import asyncio
import json

import pytest

import multi_dataset_diverse_rl.cli as cli
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.persistence.checkpoint import (
    build_checkpoint,
    restore_checkpoint,
)
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


def test_validation_evaluates_each_unique_team_state_once(tmp_path):
    calls = []

    async def solver(question, _agent_id, prompt):
        calls.append((question, prompt))
        answer = "A" if prompt == "changed" else "B"
        return PromptAnswer(answer, f"FINAL_ANSWER: {answer}", True)

    system = PromptEnsembleOptimizationSystem(
        Config.from_flat(out_dir=str(tmp_path)), solver=solver
    )
    data = [{"question": "val-q", "answer": "A"}]
    system.validation_probe = system.build_validation_probe(data)

    async def run():
        first, first_audit = await system.evaluate_validation_state(data)
        reused, reused_audit = await system.evaluate_validation_state(data)
        system.agents[0].current_prompt = "changed"
        changed, changed_audit = await system.evaluate_validation_state(data)
        return first, reused, changed, first_audit, reused_audit, changed_audit

    first, reused, changed, first_audit, reused_audit, changed_audit = (
        asyncio.run(run())
    )
    assert first == reused
    assert changed.per_agent_correct_counts[0] == 1
    assert first_audit["validation_cache_hit"] is False
    assert reused_audit["validation_cache_hit"] is True
    assert changed_audit["validation_cache_hit"] is False
    assert system.validation_evaluation_count == 2
    assert system.validation_reuse_count == 1
    assert len(system.validation_state_cache) == 2
    assert len(calls) == 2


def test_cli_validates_initial_and_accepted_states_and_tests_selected_once(
    monkeypatch, tmp_path
):
    events = []

    async def solver(question, _agent_id, prompt):
        split = question.split("-", 1)[0]
        events.append(("solver", split, prompt))
        if split == "val":
            answer = "A" if prompt == "validation-best" else "B"
        elif split == "test":
            answer = "B" if prompt == "validation-best" else "A"
        else:
            answer = "B"
        return PromptAnswer(answer, f"FINAL_ANSWER: {answer}", True)

    class StateAwareSystem(PromptEnsembleOptimizationSystem):
        def __init__(self, cfg):
            super().__init__(cfg, solver=solver)

        async def update_once(self, update_index):
            events.append(("update", update_index))
            if update_index == 0:
                for agent in self.agents:
                    agent.current_prompt = "validation-best"
                return True
            if update_index == 1:
                return False
            for agent in self.agents:
                agent.current_prompt = "test-would-prefer"
            return True

        async def evaluate_dataset(self, data, *, validation=False):
            split = str(data[0]["question"]).split("-", 1)[0]
            events.append((
                "evaluate",
                split,
                validation,
                tuple(agent.current_prompt for agent in self.agents),
            ))
            return await super().evaluate_dataset(data, validation=validation)

    def split(name):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            json.dumps({"question": f"{name}-q", "answer": "A"}) + "\n",
            encoding="utf-8",
        )
        return str(path)

    monkeypatch.setattr(cli, "PromptEnsembleOptimizationSystem", StateAwareSystem)
    cfg = Config.from_flat(
        train_path=split("train"),
        val_path=split("val"),
        test_path=split("test"),
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
    assert selection["validation_unique_state_count"] == 3
    assert selection["validation_evaluation_count"] == 3
    assert selection["validation_reused_state_count"] == 1
    assert selection["selected_checkpoint_source"] == "validation"
    assert selection["selected_by_validation"] is True
    assert selection["selected_checkpoint_update_index"] == 1
    assert selection["selected_epoch"] == 1
    assert selection["test_evaluation_count"] == 1
    assert selection["test_used_for_selection"] is False
    assert selection["test_called_before_selection"] is False
    test_events = [row for row in events if row[:2] == ("evaluate", "test")]
    assert len(test_events) == 1
    assert set(test_events[0][3]) == {"validation-best"}
    assert events.index(test_events[0]) > max(
        index for index, row in enumerate(events) if row[0] == "update"
    )
    history = json.loads(
        (tmp_path / "run" / "history.json").read_text(encoding="utf-8")
    )
    assert history[1]["validation_cache_hit"] is True


def test_test_is_forbidden_before_selection_and_persistently_single(tmp_path):
    calls = 0

    async def solver(_question, _agent_id, _prompt):
        nonlocal calls
        calls += 1
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    system = PromptEnsembleOptimizationSystem(
        Config.from_flat(out_dir=str(tmp_path)), solver=solver
    )
    data = [{"question": "test-q", "answer": "A"}]

    async def run():
        with pytest.raises(RuntimeError, match="before validation selection"):
            await system.evaluate_selected_test(data)
        system.complete_validation_selection({"source": "validation"})
        first = await system.evaluate_selected_test(data)
        second = await system.evaluate_selected_test(data)
        return first, second

    first, second = asyncio.run(run())
    assert first == second
    assert system.test_evaluation_count == 1
    assert calls == 1


def test_checkpoint_resume_reuses_completed_final_test(tmp_path):
    identity = RunIdentity(
        method_version="member_aware_peer_state_v4",
        experiment_setting="shared_member_aware_full",
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
    source_calls = 0
    target_calls = 0

    async def source_solver(_question, _agent_id, _prompt):
        nonlocal source_calls
        source_calls += 1
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    async def target_solver(_question, _agent_id, _prompt):
        nonlocal target_calls
        target_calls += 1
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    probe_rows = [{"question": "train-q", "answer": "A"}]
    validation_rows = [{"question": "val-q", "answer": "A"}]
    test_rows = [{"question": "test-q", "answer": "A"}]

    async def build(path, solver):
        system = PromptEnsembleOptimizationSystem(
            Config.from_flat(out_dir=str(path)), solver=solver
        )
        system.set_run_identity(identity)
        system.validation_probe = system.build_validation_probe(validation_rows)
        await system.initialize_fixed_probe(probe_rows)
        return system

    async def run():
        source = await build(tmp_path / "source", source_solver)
        source.complete_validation_selection({"source": "validation"})
        expected = await source.evaluate_selected_test(test_rows)
        payload = build_checkpoint(
            source,
            epoch_index=1,
            update_index=0,
            best_state={"initial_validation": expected.to_dict()},
        )
        target = await build(tmp_path / "target", target_solver)
        restore_checkpoint(target, payload)
        actual = await target.evaluate_selected_test(test_rows)
        return expected, actual, target

    expected, actual, target = asyncio.run(run())
    assert actual == expected
    assert source_calls == 2  # one fixed-probe request and one test request
    assert target_calls == 1  # fixed-probe initialization only; no repeated test
    assert target.test_evaluation_count == 1
