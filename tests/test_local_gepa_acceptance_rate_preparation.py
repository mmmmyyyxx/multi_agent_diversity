from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from infrastructure.common_solver_contract_v1.contract import COMMON_SOLVER_CONTRACT_ID
from multi_dataset_diverse_rl.evaluation.output_contract import SOLVER_OUTPUT_CONTRACT_VERSION
from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import LocalSolverObservation
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import import_frozen_gepa, verify_frozen_gepa
from multi_dataset_diverse_rl.local_optimizers.proposal_telemetry import (
    EVALUATION_FIELDS, ExactProposalStopper, ProposalTelemetry, cost_envelope, edit_similarity,
    text_hash, wilson, evaluation_schema,
)
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample, LocalOptimizationTask, LocalOptimizerBudget
from scripts.run_local_gepa_acceptance_rate_pilot import pooled_summary, require_execution_ready, run_parent

ROOT_TEXT = "Examine semantic evidence carefully."
CHILD_TEXT = "Compare referents using explicit semantic constraints."


def task(metric=205, n=3):
    examples = tuple(LocalEvidenceExample(f"e{i}", f"synthetic problem {i}", "a",
                    tags=(("responsibility", "coalition", "preservation")[i % 3], "coverage")) for i in range(n))
    return LocalOptimizationTask("synthetic", ROOT_TEXT, examples, examples,
                                 "primary_responsibility_lane=coverage", COMMON_SOLVER_CONTRACT_ID,
                                 SOLVER_OUTPUT_CONTRACT_VERSION, 79, LocalOptimizerBudget(metric))


def callback(tmp_path, rows=None):
    rows = rows or task().search_examples
    return ProposalTelemetry(tmp_path / "lineage.jsonl", parent_prompt=ROOT_TEXT,
                             examples=rows, search_examples=rows, metric_ceiling=1000)


def paired(cb, before, after, *, iteration=1, parent_index=0, parent=ROOT_TEXT,
           child=CHILD_TEXT, groups=None):
    cb.on_candidate_selected({"iteration": iteration, "candidate_idx": parent_index,
                              "candidate": {"decision_procedure": parent}, "score": 0.0})
    cb.on_minibatch_sampled({"iteration": iteration, "minibatch_ids": list(range(len(before)))})
    ids = [row.example_id for row in cb.search_examples[:len(before)]]
    groups = groups or [row.tags[0] for row in cb.search_examples[:len(before)]]
    for role, scores, prompt in (("parent", before, parent), ("child", after, child)):
        if role == "child":
            cb.on_proposal_end({"iteration": iteration, "new_instructions": {"decision_procedure": child}})
        cb.observe_evaluation({"candidate_hash": text_hash(prompt), "example_ids": ids,
                               "binary_scores": scores, "provider_called": [False] * len(ids),
                               "capture_traces": role == "parent", "evidence_group": groups,
                               "reasoning_lane": ["coverage"] * len(ids)})
        cb.on_evaluation_end({"iteration": iteration, "candidate_idx": parent_index if role == "parent" else None,
                              "scores": scores})
    if sum(after) > sum(before):
        cb.on_candidate_accepted({"iteration": iteration, "new_candidate_idx": iteration,
                                  "new_score": sum(after), "parent_ids": [parent_index]})
    else:
        cb.on_candidate_rejected({"iteration": iteration, "old_score": sum(before), "new_score": sum(after), "reason": "ignored"})


@pytest.mark.parametrize("before,after,fix,broken", [
    ([0, 0, 0], [1, 0, 0], 1, 0), ([1, 0, 0], [0, 0, 0], 0, 1),
    ([0, 1, 0], [1, 0, 0], 1, 1), ([0, 1, 0], [0, 1, 0], 0, 0),
    ([0, 0, 1], [0, 0, 0], 0, 1),
])
def test_exact_paired_metrics(tmp_path, before, after, fix, broken):
    cb = callback(tmp_path)
    paired(cb, before, after)
    result = cb.scientific_summary(1)
    row = result["proposals"][0]
    assert (row["newly_fixed"], row["newly_broken"]) == (fix, broken)
    assert row["delta_local_count"] == fix - broken
    assert row["delta_local_rate"] == (fix - broken) / 3
    assert row["sampled_local_score_parent"] == sum(before)
    assert row["sampled_local_score_proposal"] == sum(after)
    assert row["outcome_class"] == ("STRICT_POSITIVE" if fix > broken else
                                     "STRICT_NEGATIVE" if fix < broken else "EXACT_EQUAL")
    if fix == broken == 0:
        assert row["zero_delta_decomposition"] == "BEHAVIORAL_NO_OP"
    elif fix == broken:
        assert row["zero_delta_decomposition"] == "REPAIR_PRESERVATION_CANCELLATION"
    assert row["preservation_loss"] == int(before[2] == 1 and after[2] == 0)
    assert result["primary"]["denominator"] == 1  # cached empirical evidence counts
    assert result["primary"]["numerator"] == int(fix > broken)
    assert row["parent_failure_count"] == before.count(0)
    assert all(set(row) == EVALUATION_FIELDS for row in cb.evaluations)


def test_no_preservation_is_na_and_failure_patterns_exclude_correct_rows(tmp_path):
    cb = callback(tmp_path)
    paired(cb, [0, 1, 1], [0, 1, 1], groups=["responsibility", "coalition", "coalition"])
    row = cb.scientific_summary(1)["proposals"][0]
    assert row["preservation_loss"] is None
    assert row["dominant_failure_pattern"] == ["responsibility", "coverage"]
    assert row["dominant_failure_pattern_share"] == 1


def test_evolved_parent_similarity_lineage_and_duplicates(tmp_path):
    cb = callback(tmp_path)
    paired(cb, [0, 0, 0], [1, 0, 0])
    paired(cb, [1, 0, 0], [1, 0, 0], iteration=2, parent_index=1, parent=CHILD_TEXT, child=CHILD_TEXT)
    summary = cb.scientific_summary(2)
    row = summary["proposals"][1]
    assert row["parent_hash"] == text_hash(CHILD_TEXT)
    assert row["similarity"] == 1 and not row["changed"] and row["duplicate"]
    assert row["parent_lineage_depth"] == 1
    assert summary["primary"]["denominator"] == 1
    assert summary["pairwise_minibatch_jaccard"] == [1.0]
    assert summary["fraction_proposals_reusing_previous_failure"] == 0.5


def test_budget_wilson_and_similarity():
    envelope = cost_envelope(8, 12, 3, 16)
    assert envelope["normal_max_metric_calls"] == 156
    assert envelope["metric_ceiling"] == 205
    assert wilson(0, 0)["rate"] is None
    assert wilson(0, 0)["interpretation"] == "NAIVE_IID_REFERENCE_ONLY"
    assert wilson(1, 2)["wilson_95"] == pytest.approx([0.09453120573423074, 0.9054687942657693])
    assert edit_similarity("A a b", "a b a")["similarity"] < 1
    assert edit_similarity("X!", "x")["similarity"] == 1


class FakeEvaluator:
    solver_contract_id = COMMON_SOLVER_CONTRACT_ID
    output_contract_id = SOLVER_OUTPUT_CONTRACT_VERSION

    def __init__(self, mode="reject", n=3):
        self.mode, self.calls, self.n = mode, 0, n

    def evaluate(self, prompt, example):
        self.calls += 1
        correct = False
        if self.mode == "skip_once":
            correct = self.n < self.calls <= self.n + 3
        elif self.mode == "perfect":
            correct = True
        elif self.mode == "accept":
            correct = prompt != ROOT_TEXT and ("variant 2" in prompt or example.example_id != "e5")
        return LocalSolverObservation("a", "synthetic reasoning", correct, True)


class FakeReflection:
    def __init__(self, mode="normal"):
        self.mode, self.calls = mode, 0

    def __call__(self, _request):
        self.calls += 1
        if self.mode == "invalid":
            return "FINAL_ANSWER: a"
        if self.mode == "duplicate":
            return CHILD_TEXT
        return f"Compare semantic constraints using reasoning variant {self.calls}."


def run_fake(tmp_path, mode="reject", quota=4, budget=120, n=3, reflection_mode="normal", skip=16):
    evaluator, reflection = FakeEvaluator(mode, n), FakeReflection(reflection_mode)
    result = asyncio.run(run_parent(task=task(budget, n), evaluator=evaluator, reflection_lm=reflection,
                      accounting_reader=lambda: {"successful_calls": reflection.calls}, run_root=tmp_path,
                      proposal_quota=quota, skip_allowance=skip))
    return result


@pytest.mark.parametrize("mode", ["reject", "skip_once"])
def test_official_engine_exact_proposal_events_including_skips(tmp_path, mode):
    result = run_fake(tmp_path, mode=mode)
    assert result["funnel"]["proposal_attempts_observed"] == 4
    assert result["quota_status"] == "COMPLETE"
    assert result["skipped_iterations"] == (mode == "skip_once")
    assert result["primary"]["denominator"] == 4


def test_official_engine_accepted_mutation_becomes_parent(tmp_path):
    result = run_fake(tmp_path, mode="accept", quota=2, n=6, budget=1000, skip=100)
    assert result["quota_status"] == "COMPLETE"
    assert result["funnel"]["gepa_accepted"] == 2
    assert result["proposals"][1]["parent_candidate_index"] == 1
    assert result["proposals"][1]["parent_hash"] == result["proposals"][0]["proposal_hash"]
    assert result["accepted_generations"] == [1, 2]


@pytest.mark.parametrize("reflection_mode,invalid,duplicate", [("invalid", 4, 3), ("duplicate", 0, 3)])
def test_official_invalid_and_duplicate_funnel(tmp_path, reflection_mode, invalid, duplicate):
    result = run_fake(tmp_path, reflection_mode=reflection_mode)
    assert result["funnel"]["contract_invalid"] == invalid
    assert result["funnel"]["duplicate"] == duplicate
    assert result["primary"]["denominator"] == (0 if invalid else 4)


def test_perfect_forever_cannot_guarantee_quota(tmp_path):
    result = run_fake(tmp_path, mode="perfect", skip=2)
    assert result["funnel"]["proposal_attempts_observed"] == 0
    assert result["quota_status"] == "PARENT_PROPOSAL_QUOTA_INCOMPLETE"
    assert result["proposal_shortfall"] == 4


def test_metric_budget_shortfall_and_no_overshoot(tmp_path):
    result = run_fake(tmp_path, budget=7)
    assert result["quota_status"] == "PARENT_PROPOSAL_QUOTA_INCOMPLETE"
    assert result["metric_calls"] <= 7
    assert result["termination"] == "metric_ceiling_before_batch"


def test_budget_interrupts_accepted_candidate_full_validation(tmp_path):
    result = run_fake(tmp_path, mode="accept", quota=1, budget=10)
    assert result["metric_calls"] == 9
    assert result["quota_status"] == "PARENT_PROPOSAL_QUOTA_INCOMPLETE"
    assert result["unfinished_comparison_count"] == 1
    assert result["funnel"]["strict_positive_sampled_delta"] == 1
    assert result["funnel"]["gepa_accepted"] == 0


def test_builtin_stopper_counts_skipped_iterations(tmp_path):
    import_frozen_gepa()
    from gepa.utils.stop_condition import MaxCandidateProposalsStopper
    cb = callback(tmp_path)
    state = SimpleNamespace(i=3)
    assert MaxCandidateProposalsStopper(4)(state)
    assert not ExactProposalStopper(cb, 4, 16)(state)


def test_four_parent_stoppers_enforce_independent_eight_event_ceilings(tmp_path):
    callbacks = [callback(tmp_path / str(member)) for member in range(1, 5)]
    stoppers = [ExactProposalStopper(cb, 8, 16) for cb in callbacks]
    for cb, stopper in zip(callbacks, stoppers, strict=True):
        cb.proposal_count = 7
        assert stopper(None) is False
        cb.proposal_count = 8
        assert stopper(None) is True
    assert [cb.proposal_count for cb in callbacks] == [8, 8, 8, 8]


def test_fidelity_sanitization_authorization_and_pooled_rate(tmp_path):
    verify_frozen_gepa()
    result = run_fake(tmp_path)
    public = json.dumps(result)
    assert ROOT_TEXT not in public and CHILD_TEXT not in public and "synthetic problem" not in public
    assert "synthetic reasoning" not in public
    assert all(row["parent_task_id"] == "synthetic" for row in result["proposals"])
    assert pooled_summary([result, result])["primary"]["denominator"] == 8
    assert pooled_summary([result, result])["primary"]["estimand"] == "DESCRIPTIVE_NON_IID"
    assert pooled_summary([result, result])["iid_inference_permitted"] is False
    with pytest.raises(PermissionError, match="API_AUTHORIZATION_REQUIRED"):
        require_execution_ready({"api_authorization": {"authorized": False}})


def test_misaligned_rows_fail_closed(tmp_path):
    cb = callback(tmp_path)
    paired(cb, [0, 0, 0], [0, 0, 0])
    cb.pairs[1]["child"]["example_ids"] = ["wrong"] * 3
    with pytest.raises(ValueError, match="paired identity"):
        cb.scientific_summary(1)


def test_real_builtin_stopper_shortchanges_perfect_skip(tmp_path):
    from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPALocalPromptOptimizer
    engine = import_frozen_gepa()
    from gepa.utils.stop_condition import MaxCandidateProposalsStopper
    local = task()
    captured = []

    def factory(*args, **kwargs):
        cb = ProposalTelemetry(*args, **kwargs, search_examples=local.search_examples, metric_ceiling=205)
        captured.append(cb)
        return cb

    def optimize(**kwargs):
        return engine.optimize(**kwargs, stop_callbacks=MaxCandidateProposalsStopper(4))

    reflection = FakeReflection()
    optimizer = GEPALocalPromptOptimizer(evaluator=FakeEvaluator("skip_once"), reflection_lm=reflection,
        accounting_reader=lambda: {}, run_root=tmp_path, optimize_fn=optimize, callback_factory=factory)
    asyncio.run(optimizer.optimize(local))
    result = captured[0].scientific_summary(4)
    assert result["skipped_iterations"] == 1
    assert result["funnel"]["proposal_attempts_observed"] == 3
    assert result["proposal_shortfall"] == 1


def test_deterministic_replay_and_schema(tmp_path):
    import jsonschema
    a = run_fake(tmp_path / "a", mode="accept", quota=2, n=6, budget=1000, skip=100)
    b = run_fake(tmp_path / "b", mode="accept", quota=2, n=6, budget=1000, skip=100)
    assert a == b
    for row in a["evaluation_batches"]:
        jsonschema.validate(row, evaluation_schema())


def test_callback_persistence_failure_cannot_be_swallowed(tmp_path):
    cb = callback(tmp_path)
    cb.event_path.mkdir()
    with pytest.raises(OSError):
        cb._append("synthetic")
    with pytest.raises(ValueError, match="integrity"):
        cb.before_evaluation(3)
    with pytest.raises(ValueError, match="integrity"):
        ExactProposalStopper(cb, 8, 16)(None)


def test_parent_catalog_and_prep_identity_are_review_only():
    from scripts.prepare_local_gepa_acceptance_rate_pilot import parent_catalog
    from scripts.run_local_gepa_acceptance_rate_pilot import preflight
    catalog = parent_catalog()
    assert catalog["status"] == "PARENT_CATALOG_INSUFFICIENT"
    assert catalog["verified_eligible_parent_count"] == 0
    assert catalog["selection_uses_proposal_outcomes"] is False
    assert preflight()["preparation_gate"] == "PASS"


def test_layer1_runner_has_no_team_search_or_provider_construction():
    import ast
    import inspect
    import scripts.run_local_gepa_acceptance_rate_pilot as runner
    tree = ast.parse(inspect.getsource(runner))
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("team_search" in (module or "") for module in modules)
    calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert not {"AsyncOpenAI", "OpenAI", "Seed78System", "TeamSearchController"}.intersection(calls)
