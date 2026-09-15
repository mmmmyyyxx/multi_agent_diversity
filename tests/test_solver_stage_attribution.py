from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.evaluation.solver_stage import (
    SOLVER_PHASES,
    team_solver_stage_attribution,
    validate_solver_stage_attribution,
)
from multi_dataset_diverse_rl.local_optimizers.schemas import (
    LocalEvidenceExample,
    LocalPromptCandidate,
)
from multi_dataset_diverse_rl.team_search.schemas import TeamSearchAssignment
from multi_dataset_diverse_rl.team_search.system_runtime import (
    SystemLocalSolverEvaluator,
    SystemTeamCandidateEvaluator,
)
from scripts.run_seed78_primary_responsibility_ab import Seed78System


class StageHarness:
    def __init__(self) -> None:
        self.current = None
        self.produced = []
        self.solver_ledger = []
        self.successful_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def set_stage(self, stage):
        if stage is None:
            self.current = None
            return
        self.current = validate_solver_stage_attribution(stage)
        self.produced.append(dict(self.current))

    def accounting(self):
        return {
            "successful_provider_calls": self.successful_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    async def solve(self, _question, _agent, _prompt):
        assert self.current is not None
        self.successful_calls += 1
        self.prompt_tokens += 2
        self.completion_tokens += 1
        self.solver_ledger.append(
            {"phase": self.current["phase"], "input_tokens": 2, "output_tokens": 1}
        )
        return SimpleNamespace(
            answer="A", valid=True, trace="reasoning", validity_status="valid"
        )


class FakeProbe:
    def __init__(self, harness: StageHarness, size: int, *, cache_hit: bool = False) -> None:
        self.harness = harness
        self.cache_hit = cache_hit
        self.examples = tuple(
            SimpleNamespace(question_hash=f"q-{index}", gold_answer="A")
            for index in range(size)
        )

    async def evaluate_prompt_indices(self, agent, prompt, _prompt_hash, indices, solve):
        result = {}
        for index in indices:
            if self.cache_hit:
                result[index] = SimpleNamespace(
                    answer="A", valid=True, trace="cached", validity_status="valid"
                )
            else:
                result[index] = await solve(f"question-{index}", agent, prompt)
        return result

    async def evaluate_prompt(self, agent, prompt, prompt_hash, solve):
        rows = await self.evaluate_prompt_indices(
            agent, prompt, prompt_hash, range(len(self.examples)), solve
        )
        return tuple(rows[index] for index in range(len(self.examples)))


class FakeSystem:
    def __init__(self, harness: StageHarness, probe: FakeProbe) -> None:
        self.cfg = SimpleNamespace(training=SimpleNamespace(seed=78))
        self.fixed_probe = probe
        self.solve = harness.solve
        self.agents = [SimpleNamespace(current_prompt=f"parent-{index}") for index in range(5)]

    @staticmethod
    def prompt_hash(prompt):
        return hashlib.sha256(prompt.encode()).hexdigest()

    @staticmethod
    def match_answer(answer, gold_answer):
        return answer == gold_answer

    @staticmethod
    def team_prompt_state_hash():
        return "team-hash"

    @staticmethod
    def _dataset_metrics_from_profiles(_examples, _profiles):
        return SimpleNamespace(
            vote_correct_count=1,
            per_agent_correct_counts=(1, 1, 1, 1, 1),
        )


def assignment() -> TeamSearchAssignment:
    return TeamSearchAssignment(
        target_member=0,
        parent_prompt="parent-0",
        evidence=(),
        optimization_context="",
        responsibility_identity="stage-test",
    )


def candidate() -> LocalPromptCandidate:
    return LocalPromptCandidate("candidate", "replacement", 1.0, {}, (), 1)


def test_stage_boundary_fails_before_solver_for_missing_empty_or_divergent_phase() -> None:
    system = object.__new__(Seed78System)
    system._solver_stage = None
    for invalid in (
        {},
        {"phase": ""},
        {"phase": "team_full_eval", "evaluation_stage": "team_shadow_eval"},
    ):
        with pytest.raises(ValueError):
            system.set_stage(invalid)
        assert system._solver_stage is None
    system.set_stage({"phase": "initialization", "candidate_id": "P0"})
    assert system._solver_stage["phase"] == "initialization"
    system.set_stage(None)


@pytest.mark.parametrize("cache_hit", [False, True])
@pytest.mark.parametrize("phase", ["team_minibatch_eval", "team_full_eval"])
def test_actual_team_profile_producer_has_phase_for_cache_hit_and_miss(
    cache_hit: bool, phase: str
) -> None:
    harness = StageHarness()
    probe = FakeProbe(harness, 2, cache_hit=cache_hit)
    system = FakeSystem(harness, probe)

    async def run():
        evaluator = SystemTeamCandidateEvaluator(
            system=system,
            shadow_probe=probe,
            loop=asyncio.get_running_loop(),
            stage=harness.set_stage,
            accounting=harness.accounting,
            update_index_reader=lambda: 0,
        )
        return await asyncio.to_thread(
            evaluator._profile,
            assignment(),
            candidate(),
            indices=(0,) if phase == "team_minibatch_eval" else None,
            evaluation_stage=phase,
        )

    asyncio.run(run())
    produced = harness.produced[-1]
    assert produced["phase"] == produced["evaluation_stage"] == phase
    assert set(
        ("seed", "parent_id", "update_index", "target_member", "candidate_id", "proposal_engine")
    ).issubset(produced)
    assert len(harness.solver_ledger) == (0 if cache_hit else (1 if phase == "team_minibatch_eval" else 2))


def test_actual_local_gepa_producer_preserves_phase() -> None:
    harness = StageHarness()
    probe = FakeProbe(harness, 1)
    system = FakeSystem(harness, probe)
    row = LocalEvidenceExample("q-0", "question", "A")

    async def run():
        evaluator = SystemLocalSolverEvaluator(
            system=system,
            loop=asyncio.get_running_loop(),
            stage=harness.set_stage,
            accounting=harness.accounting,
            solver_contract_id="solver-contract",
            output_contract_id="output-contract",
        )
        evaluator.task_context = {
            "phase": "local_optimizer_solver_eval",
            "update_index": 0,
            "target_member": 0,
            "parent_id": "parent",
        }
        return await asyncio.to_thread(evaluator.evaluate, "replacement", row)

    result = asyncio.run(run())
    assert result.valid is True
    assert harness.produced[-1]["phase"] == "local_optimizer_solver_eval"
    assert harness.produced[-1]["evaluation_stage"] == "local_optimizer_solver_eval"


def test_actual_shadow_producer_reaches_solver_with_phase() -> None:
    harness = StageHarness()
    fixed = FakeProbe(harness, 2, cache_hit=True)
    shadow = FakeProbe(harness, 50, cache_hit=False)
    system = FakeSystem(harness, fixed)

    async def run():
        evaluator = SystemTeamCandidateEvaluator(
            system=system,
            shadow_probe=shadow,
            loop=asyncio.get_running_loop(),
            stage=harness.set_stage,
            accounting=harness.accounting,
            update_index_reader=lambda: 0,
        )
        return await asyncio.to_thread(evaluator.evaluate_shadow, assignment(), candidate())

    decision, cost = asyncio.run(run())
    assert decision.passed is True
    assert cost.solver_calls == 300
    assert harness.produced[-1]["phase"] == "team_shadow_eval"
    assert harness.produced[-1]["evaluation_stage"] == "team_shadow_eval"


def test_solver_ledger_phase_and_token_arithmetic() -> None:
    harness = StageHarness()

    async def run():
        for phase in sorted(SOLVER_PHASES - {"final_validation"}):
            harness.set_stage(team_solver_stage_attribution(
                phase=phase,
                seed=78,
                parent_id="parent",
                update_index=0,
                target_member=0,
                candidate_id="candidate",
                proposal_engine="fake",
            ))
            await harness.solve("question", 0, "prompt")
            harness.set_stage(None)

    asyncio.run(run())
    assert len(harness.solver_ledger) == len(SOLVER_PHASES) - 1
    assert sum(row["input_tokens"] + row["output_tokens"] for row in harness.solver_ledger) == 3 * len(harness.solver_ledger)
    assert {row["phase"] for row in harness.solver_ledger} == SOLVER_PHASES - {"final_validation"}
