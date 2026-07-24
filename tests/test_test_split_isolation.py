import asyncio
import json

import multi_dataset_diverse_rl.cli as cli
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


def test_cli_never_uses_test_split_for_prompt_selection(monkeypatch, tmp_path):
    events = []

    async def solver(_question, _agent_id, prompt):
        value = "A" if prompt == "selected-prompt" else "B"
        return PromptAnswer(value, f"FINAL_ANSWER: {value}", True)

    class IsolatedSystem(PromptEnsembleOptimizationSystem):
        def __init__(self, cfg):
            super().__init__(cfg, solver=solver)

        async def update_once(self, update_index):
            events.append(("update", update_index))
            for agent in self.agents:
                agent.current_prompt = "selected-prompt"
            return True

        async def evaluate_dataset(self, data, *, validation=False):
            split = str(data[0]["question"]).split("-", 1)[0]
            events.append(
                (
                    "evaluate",
                    split,
                    validation,
                    tuple(agent.current_prompt for agent in self.agents),
                )
            )
            return await super().evaluate_dataset(data, validation=validation)

    def write_split(name):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            json.dumps({"question": f"{name}-q", "answer": "A"}) + "\n",
            encoding="utf-8",
        )
        return str(path)

    monkeypatch.setattr(cli, "PromptEnsembleOptimizationSystem", IsolatedSystem)
    cfg = Config.from_flat(
        train_path=write_split("train"),
        val_path=write_split("val"),
        test_path=write_split("test"),
        train_size=1,
        val_size=1,
        test_size=1,
        answer_format="option_letter",
        epochs=1,
        update_every=1,
        out_dir=str(tmp_path / "run"),
    )
    result = asyncio.run(cli.run(cfg))

    test_events = [row for row in events if row[:2] == ("evaluate", "test")]
    assert len(test_events) == 2
    assert events.index(test_events[0]) > max(
        index for index, row in enumerate(events) if row[0] == "update"
    )
    assert all(prompt != "selected-prompt" for prompt in test_events[0][3])
    assert all(prompt == "selected-prompt" for prompt in test_events[1][3])
    assert result["selection_summary"]["selected_epoch"] == 1
