from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from infrastructure.common_solver_contract_v1.contract import (
    COMMON_SOLVER_CONTRACT_ID,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import (
    LocalSolverObservation,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import (
    GEPALocalPromptOptimizer,
)
from multi_dataset_diverse_rl.local_optimizers.schemas import (
    LocalEvidenceExample,
    LocalOptimizationResult,
    LocalOptimizationTask,
    LocalOptimizerBudget,
)
from multi_dataset_diverse_rl.team_search.execution_runtime import (
    ContextualLocalPromptOptimizer,
    LOCAL_OPTIMIZER_INVOCATION,
    LocalOptimizerConfigurationError,
    LocalOptimizerExecutionContext,
)


def _task() -> LocalOptimizationTask:
    examples = tuple(
        LocalEvidenceExample(
            example_id=f"example-{index}",
            input_payload=f"question {index}",
            gold="A",
            parent_output="B",
            textual_feedback="distinguish compatible referents",
        )
        for index in range(12)
    )
    return LocalOptimizationTask(
        task_id="seed80_update0_member2",
        parent_prompt="Use a general reasoning procedure.",
        search_examples=examples,
        local_validation_examples=examples,
        optimization_context="Repair the assigned responsibility evidence.",
        solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
        output_contract_id="task_output_contract_v1",
        seed=8_000_002,
        budget=LocalOptimizerBudget(max_metric_calls=36),
        run_seed=80,
        update_index=0,
        target_member=2,
    )


def _runtime(**changes) -> LocalOptimizerExecutionContext:
    values = {
        "run_seed": 80,
        "provider_profile": "lwj",
        "solver_model": "qwen3-8b",
        "optimizer_model": "qwen3.7-flash",
        "evaluator_model": "qwen3.7-flash",
        "run_identity_sha256": "a" * 64,
        "local_no_update_patience": 3,
        "team_no_update_patience": 2,
        "saturation_mode": "single_opportunity_engineering_canary",
        "arm": "GEPA_LAYER2_REAL_CANARY",
    }
    values.update(changes)
    return LocalOptimizerExecutionContext(**values)


class _LocalSolver:
    def __init__(self) -> None:
        self.task_context = None
        self.calls = 0
        self.solver_contract_id = COMMON_SOLVER_CONTRACT_ID
        self.output_contract_id = "task_output_contract_v1"

    def evaluate(self, decision_procedure, example):
        assert self.task_context is not None
        assert self.task_context["run_seed"] == 80
        assert self.task_context["target_member"] == 2
        assert self.task_context["provider_profile"] == "lwj"
        self.calls += 1
        correct = "distinguish" in decision_procedure.casefold()
        return LocalSolverObservation(
            parsed_answer="A" if correct else "B",
            raw_output="fake output",
            correct=correct,
            valid=True,
            input_tokens=2,
            output_tokens=1,
        )


class _Reflection:
    def __init__(self) -> None:
        self.accounting = {
            "successful_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def __call__(self, prompt):
        del prompt
        assert LOCAL_OPTIMIZER_INVOCATION.get() is not None
        self.accounting["successful_calls"] += 1
        self.accounting["input_tokens"] += 10
        self.accounting["output_tokens"] += 5
        return "Use semantic compatibility to distinguish candidate referents carefully."


class _BoundaryOnlyOptimizer:
    async def optimize(self, task):
        context = LOCAL_OPTIMIZER_INVOCATION.get()
        assert context is not None
        assert context["parent_id"] == "seed80_update0"
        assert context["optimizer_model"] == "qwen3.7-flash"
        return LocalOptimizationResult(
            candidates=(),
            backend_name="boundary-only",
            backend_version="v1",
            optimizer_state=None,
            solver_calls=0,
            optimizer_calls=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            termination_reason="GEPA_LOCAL_OPTIMIZER_READY",
        )


def test_original_config_seed_failure_is_closed_by_explicit_runtime_contract() -> None:
    cfg = Config.from_flat(seed=80)
    assert not hasattr(cfg, "seed")
    assert cfg.training.seed == 80
    local_solver = _LocalSolver()
    result = asyncio.run(
        ContextualLocalPromptOptimizer(
            _BoundaryOnlyOptimizer(), local_solver, _runtime()
        ).optimize(_task())
    )
    assert result.termination_reason == "GEPA_LOCAL_OPTIMIZER_READY"
    assert local_solver.task_context is None


def test_full_fake_gepa_path_uses_production_contextual_wrapper(tmp_path: Path) -> None:
    local_solver = _LocalSolver()
    reflection = _Reflection()
    official = GEPALocalPromptOptimizer(
        evaluator=local_solver,
        reflection_lm=reflection,
        accounting_reader=lambda: reflection.accounting,
        run_root=tmp_path,
    )
    result = asyncio.run(
        ContextualLocalPromptOptimizer(
            official, local_solver, _runtime()
        ).optimize(_task())
    )
    assert result.candidates
    assert local_solver.calls > 0
    assert reflection.accounting["successful_calls"] > 0
    assert result.optimizer_state is not None
    telemetry = result.optimizer_state.payload["telemetry"]
    assert telemetry["proposal_attempts"] >= 1
    assert telemetry["proposer_diagnostics"]["solver_reached"] >= 1


@pytest.mark.parametrize(
    "task",
    [
        None,
        replace(_task(), run_seed=None),
        replace(_task(), update_index=None),
        replace(_task(), target_member=None),
    ],
)
def test_missing_task_runtime_fields_fail_typed_before_backend(task) -> None:
    local_solver = _LocalSolver()
    with pytest.raises(LocalOptimizerConfigurationError):
        asyncio.run(
            ContextualLocalPromptOptimizer(
                _BoundaryOnlyOptimizer(), local_solver, _runtime()
            ).optimize(task)
        )
    assert local_solver.calls == 0


def test_missing_parent_fails_typed_before_backend() -> None:
    task = _task()
    object.__setattr__(task, "parent_prompt", "")
    with pytest.raises(LocalOptimizerConfigurationError, match="parent"):
        asyncio.run(
            ContextualLocalPromptOptimizer(
                _BoundaryOnlyOptimizer(), _LocalSolver(), _runtime()
            ).optimize(task)
        )


@pytest.mark.parametrize(
    "field",
    [
        "provider_profile",
        "solver_model",
        "optimizer_model",
        "evaluator_model",
        "run_identity_sha256",
    ],
)
def test_missing_execution_identity_fields_fail_typed(field: str) -> None:
    with pytest.raises(LocalOptimizerConfigurationError, match=field):
        _runtime(**{field: ""})


def test_run_seed_mismatch_fails_typed_before_backend() -> None:
    with pytest.raises(LocalOptimizerConfigurationError, match="run seed"):
        asyncio.run(
            ContextualLocalPromptOptimizer(
                _BoundaryOnlyOptimizer(), _LocalSolver(), _runtime(run_seed=81)
            ).optimize(_task())
        )
