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
from multi_dataset_diverse_rl.system import (
    CandidateFunnel,
    PromptEnsembleOptimizationSystem,
)


TEACHER = {
    "failure_pattern": "premature commitment",
    "repair_rule": "Check every explicit condition before committing.",
    "preservation_rule": "Keep conclusions that still pass the checks.",
}
REGENERATED = {
    **TEACHER,
    "repair_rule": "Compare every option in order before committing.",
}
APPROVED = {"failed_checks": [], "risk_case_ids": [], "feedback": ""}


async def solver(_question, _agent_id, _prompt):
    return PromptAnswer("A", "FINAL_ANSWER: A", True)


async def run() -> dict:
    counts = {"teacher": 0, "critic": 0, "student": 0}
    student_requests: list[str] = []

    async def chat(system_prompt, user_prompt, _temperature, _max_tokens):
        if system_prompt.startswith("Return strict JSON only."):
            role = "student"
        elif "Check only explicit hard blockers" in system_prompt:
            role = "critic"
        else:
            role = "teacher"
        counts[role] += 1
        if role == "critic":
            return json.dumps(APPROVED)
        if role == "teacher":
            return json.dumps(
                REGENERATED
                if "student_upstream_regeneration" in user_prompt
                else TEACHER
            )
        student_requests.append(user_prompt)
        if counts["student"] <= 4:
            return json.dumps({"candidate_prompts": [
                "Check every condition.\nFINAL_ANSWER: A",
                "Compare all options.\nFINAL ANSWER: B",
            ]})
        return json.dumps({"candidate_prompts": ["valid recovered repair"]})

    with tempfile.TemporaryDirectory() as directory:
        system = PromptEnsembleOptimizationSystem(
            Config.from_flat(
                out_dir=directory,
                answer_format="option_letter",
                experiment_setting="shared_generic_evolution",
                num_candidates_per_parent=2,
                stage_a_channel_top_k=1,
                stage_b_candidate_budget=2,
            ),
            solver=solver,
            optimizer_chat=chat,
        )
        rows = [{"question": "q", "answer": "A"}]
        system.validation_probe = system.build_validation_probe(rows)
        await system.initialize_fixed_probe(rows)
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel, 0)
        assert len(candidates) == 1
        assert counts == {"teacher": 2, "critic": 2, "student": 5}
        assert funnel.student_cycle_exhausted
        assert funnel.upstream_regeneration_count == 1
        assert funnel.student_recovered
        assert funnel.output_contract_contamination_count == 8
        assert "A candidate included the immutable solver output interface." in (
            student_requests[1]
        )
        assert all(
            row["valid_candidate_count"] == 0
            for row in system.student_recovery_observations[:4]
        )
        return {
            "student_calls": counts["student"],
            "upstream_regeneration_count": funnel.upstream_regeneration_count,
            "student_recovered": funnel.student_recovered,
            "output_contract_contamination_count": (
                funnel.output_contract_contamination_count
            ),
            "invalid_candidates_entered_stage_a": False,
        }


def main() -> None:
    print(json.dumps({"student_recovery_smoke": asyncio.run(run())}, indent=2))


if __name__ == "__main__":
    main()
