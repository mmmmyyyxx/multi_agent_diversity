"""Zero-provider checks of the production GEPA batch evaluation boundary."""

from __future__ import annotations

import asyncio
import hashlib
import pytest

from multi_dataset_diverse_rl.evaluation.fixed_probe import FixedProbeEvaluator, ProbeExample
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer, PromptQuestionEvaluator
from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import DiversityGEPAAdapter
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample
from multi_dataset_diverse_rl.team_search.execution_runtime import LOCAL_SOLVER_BATCH_OBSERVER
from multi_dataset_diverse_rl.team_search.system_runtime import SystemLocalSolverEvaluator
from multi_dataset_diverse_rl.versions import COMMON_SOLVER_CONTRACT_V1_ID


class FakeSystem:
    def __init__(self, count: int, concurrency: int, *, delay: float = 0.01,
                 vary_delay: bool = True) -> None:
        self.rows = tuple(
            ProbeExample(f"question {i}", hashlib.sha256(f"question {i}".encode()).hexdigest(), "A")
            for i in range(count)
        )
        self.cache_hits: list[str] = []
        self.prompt_question_evaluator = PromptQuestionEvaluator(
            model_request_identity="fake-solver", parser_version="strict",
            temperature=0, decoding_seed=80,
            cache_hit_callback=lambda _p, q, _a: self.cache_hits.append(q),
        )
        self.fixed_probe = FixedProbeEvaluator(self.rows, "frozen", self.prompt_question_evaluator)
        self.solver_semaphore = asyncio.Semaphore(concurrency)
        self.delay = delay
        self.vary_delay = vary_delay
        self.inflight = self.max_inflight = 0
        self.physical_calls = 0
        self.completed: list[int] = []
        self.cancelled = 0
        self.fail_index: int | None = None

    @staticmethod
    def prompt_hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    @staticmethod
    def match_answer(left: str, right: str) -> bool:
        return left == right

    async def solve(self, question: str, agent_id: int, prompt: str) -> PromptAnswer:
        del agent_id, prompt
        index = int(question.split()[-1])
        async with self.solver_semaphore:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            try:
                if index == self.fail_index:
                    await asyncio.sleep(0.001)
                    raise RuntimeError("fake provider failure")
                await asyncio.sleep(
                    self.delay * (1 + (len(self.rows) - index) % 3)
                    if self.vary_delay else self.delay
                )
                self.physical_calls += 1
                self.completed.append(index)
                observer = LOCAL_SOLVER_BATCH_OBSERVER.get()
                if observer is not None:
                    observer(self.rows[index].question_hash, f"request-{index}", True, 11, 5)
                answer = "A" if index % 2 == 0 else "B"
                return PromptAnswer(answer, f"reasoning {index}", True, prompt_tokens=11,
                                    completion_tokens=5, total_tokens=16)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            finally:
                self.inflight -= 1


def _batch(system: FakeSystem, indices: list[int]) -> list[LocalEvidenceExample]:
    return [LocalEvidenceExample(system.rows[i].question_hash, system.rows[i].question, "A")
            for i in indices]


def _adapter(system: FakeSystem, loop: asyncio.AbstractEventLoop):
    stages = []
    evaluator = SystemLocalSolverEvaluator(
        system=system, loop=loop, stage=stages.append, accounting=lambda: {},
        solver_contract_id=COMMON_SOLVER_CONTRACT_V1_ID,
        output_contract_id="fake-output-contract",
    )
    evaluator.task_context = {"target_member": 2, "update_index": 7,
                              "phase": "local_optimizer_solver_eval"}
    adapter = DiversityGEPAAdapter(
        evaluator, parent_prompt="Use context clues.",
        all_examples=_batch(system, list(range(len(system.rows)))),
        optimization_context="general", output_contract_id="fake-output-contract",
    )
    return adapter, stages


def test_production_gepa_batch_uses_configured_concurrency_and_preserves_order():
    async def run():
        system = FakeSystem(8, 4)
        adapter, stages = _adapter(system, asyncio.get_running_loop())
        indices = [7, 5, 3, 1, 6, 4, 2, 0]
        result = await asyncio.to_thread(
            adapter.evaluate, _batch(system, indices),
            {"decision_procedure": "Use context clues."}, True,
        )
        assert system.max_inflight == 4
        assert system.completed != indices
        assert [row.example.example_id for row in result.trajectories] == [
            system.rows[i].question_hash for i in indices
        ]
        assert result.scores == [float(i % 2 == 0) for i in indices]
        assert adapter.solver_calls == system.physical_calls == 8
        assert stages[0]["phase"] == "local_optimizer_solver_eval"
        assert stages[0]["update_index"] == 7
        assert stages[0]["target_member"] == 2
        assert stages[0]["candidate_id"] == "local_gepa"
        assert stages == [stages[0], None]
    asyncio.run(run())


def test_role_qualified_duplicate_has_one_physical_call_and_two_logical_rows():
    async def run():
        system = FakeSystem(1, 4)
        adapter, stages = _adapter(system, asyncio.get_running_loop())
        row = _batch(system, [0])[0]
        batch = [row, LocalEvidenceExample(f"focus:{row.example_id}", row.input_payload, row.gold),
                 LocalEvidenceExample(f"anchor:{row.example_id}", row.input_payload, row.gold)]
        result = await asyncio.to_thread(
            adapter.evaluate, batch, {"decision_procedure": "Use context clues."}, True,
        )
        assert result.scores == [1.0] * 3
        assert system.physical_calls == adapter.solver_calls == 1
        assert len(system.cache_hits) == 2
        assert [t.observation.provider_called for t in result.trajectories] == [True, False, False]
        assert [t.observation.input_tokens for t in result.trajectories] == [11, 0, 0]
        assert stages[-1] is None
    asyncio.run(run())


def test_minibatch_to_full_reuses_run_local_prompt_question_cache():
    async def run():
        system = FakeSystem(100, 16, delay=0)
        prompt = "Use context clues."
        prompt_hash = system.prompt_hash(prompt)
        probe = system.fixed_probe
        first = await probe.evaluate_prompt_indices(2, prompt, prompt_hash, range(12), system.solve)
        assert len(first) == 12 and system.physical_calls == 12
        full = await probe.evaluate_prompt_indices(2, prompt, prompt_hash, range(100), system.solve)
        assert len(full) == 100
        assert system.physical_calls == 100
        assert len(system.cache_hits) == 12
    asyncio.run(run())


def test_failed_batch_drains_sibling_cancellations_before_clearing_stage():
    async def run():
        system = FakeSystem(8, 8, delay=0.1)
        system.fail_index = 0
        adapter, stages = _adapter(system, asyncio.get_running_loop())
        with pytest.raises(RuntimeError, match="fake provider failure"):
            await asyncio.to_thread(
                adapter.evaluate, _batch(system, list(range(8))),
                {"decision_procedure": "Use context clues."}, True,
            )
        assert system.inflight == 0
        assert system.cancelled == 7
        assert stages[-1] is None
    asyncio.run(run())


def test_cancelled_cache_owner_releases_inflight_waiter():
    async def run():
        cache = PromptQuestionEvaluator(
            model_request_identity="fake", parser_version="strict", temperature=0,
            decoding_seed=80,
        )
        started = asyncio.Event()

        async def slow(_question, _agent_id, _prompt):
            started.set()
            await asyncio.sleep(10)
            raise AssertionError("cancelled fake request unexpectedly completed")

        kwargs = dict(question="question 0", question_hash="hash", prompt="prompt",
                      prompt_hash="prompt-hash", agent_id=0, solve=slow)
        owner = asyncio.create_task(cache.evaluate(**kwargs))
        await started.wait()
        waiter = asyncio.create_task(cache.evaluate(**kwargs))
        await asyncio.sleep(0)
        owner.cancel()
        results = await asyncio.wait_for(
            asyncio.gather(owner, waiter, return_exceptions=True), timeout=1,
        )
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        assert cache.inflight == {}
        assert cache.cache == {}
    asyncio.run(run())
