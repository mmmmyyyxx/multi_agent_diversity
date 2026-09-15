from __future__ import annotations

import importlib.util
import asyncio
from pathlib import Path

from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import LocalSolverObservation
from multi_dataset_diverse_rl.local_optimizers.gepa_callbacks import GEPALineageCallback
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPALocalPromptOptimizer
from multi_dataset_diverse_rl.local_optimizers.schemas import (
    LocalEvidenceExample,
    LocalOptimizationTask,
    LocalOptimizerBudget,
)
from infrastructure.common_solver_contract_v1.contract import COMMON_SOLVER_CONTRACT_ID
from multi_dataset_diverse_rl.evaluation.output_contract import SOLVER_OUTPUT_CONTRACT_VERSION


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "scripts/run_level_b_gepa_local_acceptance_rate_pilot.py"
    spec = importlib.util.spec_from_file_location("local_acceptance_rate_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_protocol_and_preflight() -> None:
    runner = load_runner()
    protocol = runner.protocol_document()
    assert protocol["parent_task_count"] == 3
    assert protocol["proposal_attempts_per_parent"] == 4
    assert protocol["total_proposal_attempts"] == 12
    assert protocol["local_metric_call_ceiling_per_parent"] == 120
    assert protocol["layer1_only"] is True
    assert all(protocol[key] == 0 for key in (
        "team_minibatch_calls", "full_team_calls", "shadow50_calls",
        "validation50_calls", "test50_calls", "team_commits",
    ))
    assert runner.preflight()["gate"] == "PASS"


def test_exact_proposal_stopper_uses_observed_proposals() -> None:
    runner = load_runner()

    class Callback:
        proposal_count = 0

    callback = Callback()
    stopper = runner.ExactProposalCountStopper(callback, 4)
    assert stopper(None) is False
    callback.proposal_count = 3
    assert stopper(None) is False
    callback.proposal_count = 4
    assert stopper(None) is True


def test_callback_exports_sanitized_local_effect_decomposition(tmp_path: Path) -> None:
    examples = (
        LocalEvidenceExample("r", "problem r", "a", tags=("responsibility", "coverage")),
        LocalEvidenceExample("p", "problem p", "a", tags=("preservation",)),
        LocalEvidenceExample("c", "problem c", "a", tags=("coalition",)),
    )
    callback = GEPALineageCallback(
        tmp_path / "lineage.jsonl",
        parent_prompt="Use careful semantic reasoning.",
        examples=examples,
    )
    callback.on_minibatch_sampled({"iteration": 1, "minibatch_ids": ["r", "p", "c"]})
    callback.on_evaluation_end({"iteration": 1, "candidate_idx": 0, "scores": [0, 1, 1]})
    callback.on_proposal_end({
        "iteration": 1,
        "new_instructions": {"decision_procedure": "Use explicit semantic compatibility reasoning."},
    })
    callback.on_evaluation_end({"iteration": 1, "candidate_idx": None, "scores": [1, 0, 1]})
    callback.on_candidate_accepted({
        "iteration": 1, "new_candidate_idx": 1, "new_score": 2,
        "parent_ids": [0],
    })
    diagnostics = callback.proposal_diagnostics()
    outcome = diagnostics["proposal_outcomes"][0]
    assert outcome["delta_local"] == 0
    assert outcome["newly_fixed"] == 1
    assert outcome["newly_broken"] == 1
    assert outcome["preservation_loss"] == 1
    assert outcome["official_strict_improvement"] is True
    assert diagnostics["reflection_failure_pattern_concentration"] == 1 / 3
    assert "problem r" not in str(diagnostics)


def test_real_frozen_gepa_stops_after_exactly_four_observed_proposals(tmp_path: Path) -> None:
    runner = load_runner()

    class Evaluator:
        solver_contract_id = COMMON_SOLVER_CONTRACT_ID
        output_contract_id = SOLVER_OUTPUT_CONTRACT_VERSION

        def evaluate(self, _prompt, _example):
            return LocalSolverObservation("a", "reasoning", False, True)

    class Reflection:
        def __init__(self):
            self.calls = 0
            self.accounting = {"successful_calls": 0, "input_tokens": 0, "output_tokens": 0}

        def __call__(self, _prompt):
            self.calls += 1
            self.accounting["successful_calls"] += 1
            return f"Use careful semantic compatibility reasoning variant {self.calls}."

    rows = tuple(
        LocalEvidenceExample(f"e{index}", f"problem {index}", "a", tags=("coverage",))
        for index in range(3)
    )
    task = LocalOptimizationTask(
        task_id="exact_four",
        parent_prompt="Use careful semantic reasoning.",
        search_examples=rows,
        local_validation_examples=rows,
        optimization_context="primary_responsibility_lane=coverage",
        solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
        output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
        seed=79,
        budget=LocalOptimizerBudget(120, 3, 4),
    )
    reflection = Reflection()
    optimizer = GEPALocalPromptOptimizer(
        evaluator=Evaluator(),
        reflection_lm=reflection,
        accounting_reader=lambda: reflection.accounting,
        run_root=tmp_path,
        optimize_fn=runner.fixed_proposal_optimize,
    )
    result = asyncio.run(optimizer.optimize(task))
    diagnostics = result.optimizer_state.payload["telemetry"]["proposer_diagnostics"]
    assert reflection.calls == 4
    assert diagnostics["proposal_attempts"] == 4
    assert len(diagnostics["proposal_outcomes"]) == 4
