from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


async def run() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        calls: list[str] = []

        async def solver(question, _agent_id, _prompt):
            calls.append(question)
            return PromptAnswer("A", "FINAL_ANSWER: A", True)

        system = PromptEnsembleOptimizationSystem(
            Config.from_flat(out_dir=str(Path(directory) / "run")), solver=solver
        )
        train = [{"question": "train", "answer": "A"}]
        test = [{"question": "test", "answer": "A"}]
        await system.initialize_fixed_probe(train)
        system.planned_update_count = 1
        system.completed_update_count = 1
        system.record_training_dynamics(update_index=-1)
        system.mark_training_complete(1)
        metrics = await system.evaluate_final_test(test)
        assert system.validation_probe is None
        assert system.validation_evaluation_count == 0
        assert system.validation_state_cache == {}
        assert system.test_evaluation_count == 1
        assert system.test_called_before_training_complete is False
        return {
            "validation_solver_calls": sum(value == "validation" for value in calls),
            "validation_evaluation_count": system.validation_evaluation_count,
            "test_evaluation_count": system.test_evaluation_count,
            "selected_vote_correct_count": metrics.vote_correct_count,
        }


if __name__ == "__main__":
    print(json.dumps({"final_state_protocol_smoke": asyncio.run(run())}, indent=2))
