from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


async def run() -> dict:
    calls = {"validation": 0, "test": 0}

    async def solver(question, _agent_id, prompt):
        split = question.split("-", 1)[0]
        calls[split] += 1
        answer = "A" if prompt == "selected" else "B"
        return PromptAnswer(answer, f"FINAL_ANSWER: {answer}", True)

    with tempfile.TemporaryDirectory() as directory:
        system = PromptEnsembleOptimizationSystem(
            Config.from_flat(out_dir=directory), solver=solver
        )
        validation = [{"question": "validation-q", "answer": "A"}]
        test = [{"question": "test-q", "answer": "A"}]
        system.validation_probe = system.build_validation_probe(validation)
        initial, first = await system.evaluate_validation_state(validation)
        reused, second = await system.evaluate_validation_state(validation)
        system.agents[0].current_prompt = "selected"
        changed, third = await system.evaluate_validation_state(validation)
        assert initial == reused
        assert first["validation_cache_hit"] is False
        assert second["validation_cache_hit"] is True
        assert third["validation_cache_hit"] is False
        assert system.validation_evaluation_count == 2
        assert system.validation_reuse_count == 1
        system.complete_validation_selection({
            "team_prompt_state_hash": system.team_prompt_state_hash()
        })
        selected_test = await system.evaluate_selected_test(test)
        repeated = await system.evaluate_selected_test(test)
        assert selected_test == repeated
        assert system.test_evaluation_count == 1
        return {
            "validation_unique_state_count": len(system.validation_state_cache),
            "validation_evaluation_count": system.validation_evaluation_count,
            "validation_reuse_count": system.validation_reuse_count,
            "test_evaluation_count": system.test_evaluation_count,
            "test_used_for_selection": system.test_used_for_selection,
            "test_called_before_selection": system.test_called_before_selection,
            "selected_member_correct_count": changed.per_agent_correct_counts[0],
        }


def main() -> None:
    print(json.dumps({"validation_selection_smoke": asyncio.run(run())}, indent=2))


if __name__ == "__main__":
    main()
