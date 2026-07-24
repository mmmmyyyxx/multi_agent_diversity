from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.config import Config  # noqa: E402
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer  # noqa: E402
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem  # noqa: E402


async def main() -> int:
    calls = 0

    async def solver(_question: str, _agent_id: int, _prompt: str) -> PromptAnswer:
        nonlocal calls
        calls += 1
        if calls < 3:
            return PromptAnswer("", "missing", False, "missing_final_answer")
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    system = PromptEnsembleOptimizationSystem(
        Config.from_flat(
            answer_format="option_letter",
            solver_invalid_max_retries=3,
        ),
        solver=solver,
    )
    answer = await system.solve("Question\n(A) first\n(B) second", 0, "procedure")
    assert answer.valid
    assert answer.solver_attempt_count == 3
    assert answer.recovered_from_invalid
    assert not answer.terminal_invalid
    assert answer.raw_invalid_attempt_count == 2
    assert answer.attempt_validity_statuses == (
        "missing_final_answer",
        "missing_final_answer",
        "valid",
    )
    print("deterministic solver invalid recovery smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
