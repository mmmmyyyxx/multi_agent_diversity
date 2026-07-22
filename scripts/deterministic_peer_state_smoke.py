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


async def solver(question: str, agent_id: int, prompt: str) -> PromptAnswer:
    if "repair-uncovered" in prompt and question == "q0":
        answer = "A"
    elif question == "q1" and agent_id < 3:
        answer = "A"
    elif question == "q2" and agent_id < 2:
        answer = "A"
    else:
        answer = "B"
    return PromptAnswer(answer, f"verify FINAL_ANSWER: {answer}", True)


async def optimizer(system_prompt: str, _user_prompt: str, _temperature: float, _max_tokens: int) -> str:
    if "Audit whether" in system_prompt:
        return json.dumps({"approved": True, "score": 1.0, "feedback": "approved"})
    if "strict JSON" in system_prompt:
        return json.dumps({"candidates": [{
            "candidate_prompt": "repair-uncovered",
            "target_failure_mechanism": "uncovered ambiguity",
            "repair_procedure": "compare interpretations and verify the referent",
            "preservation_rule": "retain established correct decisions",
            "expected_responsibility_effect": "create the first correct vote",
        }]})
    return "When ambiguity remains, compare interpretations and verify the selected referent."


async def run(out_dir: Path) -> dict:
    cfg = Config.from_flat(
        out_dir=str(out_dir), answer_format="option_letter", num_candidates_per_parent=1,
        stage_a_channel_top_k=1, stage_b_candidate_budget=2,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=solver, optimizer_chat=optimizer)
    data = [{"question": f"q{index}", "answer": "A"} for index in range(3)]
    await system.initialize_fixed_probe(data)
    before, _ = system.current_states_and_credits()
    changed = await system.update_once(0)
    after, _ = system.current_states_and_credits()
    system.history.append({"epoch": 1, "accepted": changed})
    system.flush_artifacts()
    return {
        "accepted": changed,
        "before_g": before[0].gold_vote_count,
        "after_g": after[0].gold_vote_count,
        "candidate_decisions": len(system.candidate_decisions),
        "responsibility_assignments": len(system.responsibility_assignments),
        "probe_cache_hits": system.fixed_probe.cache_hits,
        "probe_cache_misses": system.fixed_probe.cache_misses,
        "outputs": sorted(path.name for path in out_dir.iterdir()),
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="peer_state_smoke_") as temporary:
        report = asyncio.run(run(Path(temporary)))
    if not report["accepted"] or report["before_g"] != 0 or report["after_g"] != 1:
        raise SystemExit(f"deterministic smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
