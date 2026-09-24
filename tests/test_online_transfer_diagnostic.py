"""Zero-provider invariants for the prospective online transfer diagnostic."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import local_gepa_budget_capacity
from multi_dataset_diverse_rl.experiment import (
    ExperimentEarlyStop, ExperimentInputs, ExperimentServices, ExperimentSpec, Layer2Opportunity,
    OptimizerBackend, OptimizationScope, RuntimeContext, StoppingRegime,
    run_experiment,
)
from multi_dataset_diverse_rl.local_optimizers.schemas import (
    LocalOptimizationResult, LocalPromptCandidate, OpaqueOptimizerState,
)
from multi_dataset_diverse_rl.team_search.candidate_evaluator import EvaluationCost
from multi_dataset_diverse_rl.team_search.candidate_selector import CommonSafeTeamCandidateSelector
from multi_dataset_diverse_rl.team_search.controller import (
    DiagnosticSamplingIntegrityError, TeamSearchController,
)
from multi_dataset_diverse_rl.team_search.execution_runtime import CappedDurableLedger
from multi_dataset_diverse_rl.team_search.schemas import (
    TeamEvidenceCase, TeamMiniBatchMetrics, TeamSearchAssignment, TeamSearchRequest,
    TeamCostAccounting, TeamSearchOutcome,
)
from multi_dataset_diverse_rl.team_search.task_builder import LocalTaskBuilder
from multi_dataset_diverse_rl.production_transfer_diagnostic import (
    _candidate_stage_costs, _diagnostic_stop_reason, _mark_duplicate_accepted_event,
    _opportunity_costs, _record_ordinary_scheduler_outcome, _usage,
)


class _Assignment:
    def __init__(self):
        evidence = tuple(
            TeamEvidenceCase(f"{group}-{i}", f"question {group} {i}", "A", None,
                             None, group, ("direct_flip",) if group == "responsibility" else ())
            for group in ("responsibility", "coalition", "preservation")
            for i in range(4)
        )
        self.value = TeamSearchAssignment(
            0, "parent reasoning", evidence, "context", "responsibility",
            primary_responsibility_lane="direct_flip",
        )

    def assign(self, request):
        assert request.team_state_hash == "parent-team"
        return self.value


class _Local:
    def __init__(self, *, accepted=1, returned=1, delta=1.0):
        self.accepted = accepted
        self.returned = returned
        self.delta = delta
        self.calls = 0

    async def optimize(self, task):
        self.calls += 1
        assert task.parent_prompt == "parent reasoning"
        candidate = LocalPromptCandidate(
            "candidate-hash", "changed reasoning", 0.75, {}, (), 1,
            backend_metadata={
                "local_parent_score": 0.5, "local_full_validation_delta": 0.25,
                "local_acceptance_delta": self.delta,
                "local_newly_fixed": 2, "local_newly_broken": 1,
                "local_preservation_loss": 1,
            },
        )
        return LocalOptimizationResult(
            (candidate,) if self.returned else (), "gepa", "frozen", OpaqueOptimizerState(
                "gepa", "frozen", {"telemetry": {
                    "accepted_mutations": self.accepted, "proposal_attempts": 1,
                }},
            ), 18, 1, 18, 2, 20, "complete",
        )


def _evaluation(vote: int, member: int, oracle_delta: int):
    return SimpleNamespace(
        team_outcome=SimpleNamespace(vote_correct_count=vote, mean_soft_vote_utility=0.5),
        competence=SimpleNamespace(correct_count=member, invalid_count=0),
        prompt_hash="fake-candidate-hash",
        marginal=SimpleNamespace(
            coverage_gain_count=max(0, oracle_delta),
            coverage_loss_count=max(0, -oracle_delta),
            vote_gain_count=max(0, vote - 50),
            vote_loss_count=max(0, 50 - vote),
        ),
        protection=SimpleNamespace(
            pivotal_correct_gain_count=1,
            pivotal_correct_loss_count=0,
        ),
    )


class _Evaluator:
    def __init__(self, *, promote: bool):
        self.promote = promote
        self.ordinary_full_calls = 0
        self.diagnostic_full_calls = 0
        self.shadow_calls = 0
        self.minibatch_calls = 0

    def active_evaluation(self, assignment):
        return _evaluation(50, 60, 0)

    def parent_vote_responsiveness(self, assignment):
        return {
            "target_pivotal_count": 0,
            "total_pivotality": 0,
            "single_member_vote_changeable_cases": 0,
        }

    def evaluate_minibatch(self, assignment, candidate, minibatch):
        self.minibatch_calls += 1
        assert len(minibatch) == 12
        return TeamMiniBatchMetrics(target_delta=int(self.promote)), EvaluationCost(12, 12, 0)

    def evaluate_full(self, assignment, candidate):
        self.ordinary_full_calls += 1
        return _evaluation(51, 61, 1), EvaluationCost(88, 88, 0)

    def evaluate_diagnostic_full(self, assignment, candidate):
        self.diagnostic_full_calls += 1
        return _evaluation(51, 61, 1), EvaluationCost(88, 88, 0)

    def evaluate_shadow(self, assignment, candidate):
        self.shadow_calls += 1
        return SimpleNamespace(passed=False), EvaluationCost(0, 0, 0)


class _NoCommit:
    def commit(self, **kwargs):
        raise AssertionError("diagnostic Full must not create an online commit")


@pytest.mark.parametrize("promote", [False, True])
def test_mandatory_full_isolated_from_ordinary_selection(monkeypatch, promote):
    import multi_dataset_diverse_rl.team_search.controller as controller_module
    import multi_dataset_diverse_rl.team_search.candidate_selector as selector_module

    safe = lambda candidate, active: SimpleNamespace(passed=True)
    monkeypatch.setattr(controller_module, "evaluate_constraints", safe)
    monkeypatch.setattr(selector_module, "evaluate_constraints", safe)
    evaluator = _Evaluator(promote=promote)
    controller = TeamSearchController(
        responsibility=_Assignment(), task_builder=LocalTaskBuilder(),
        local_optimizer=_Local(), evaluator=evaluator,
        selector=CommonSafeTeamCandidateSelector(), committer=_NoCommit(),
        diagnostic_full_for_local_accepts=True,
    )
    outcome = asyncio.run(controller.run_opportunity(
        TeamSearchRequest(81, 0, "parent-team", 36, "COMMON_SOLVER_CONTRACT_V1", "task_output_contract_v1")
    ))
    assert outcome.committed_candidate_id is None
    assert outcome.funnel["diagnostic_full_evaluated_candidates"] == 1
    assert evaluator.ordinary_full_calls == int(promote)
    assert evaluator.diagnostic_full_calls == int(not promote)
    assert evaluator.shadow_calls == int(promote)
    if not promote:
        assert outcome.candidates[0].full_evaluation is None
        assert outcome.candidates[0].constraint is None
        assert outcome.funnel["feasible_candidates"] == 0
    row = outcome.audit_metadata["transfer_diagnostic"][0]
    assert row["full"]["vote_delta"] == 1
    assert row["full"]["diagnostic_only"] is (not promote)
    assert row["committed"] is False
    assert row["parent_responsiveness"]["single_member_vote_changeable_cases"] == 0


@pytest.mark.parametrize("accepted,returned,delta", [
    (1, 0, 1.0), (0, 1, 1.0), (2, 1, 1.0),
    (1, 1, 0.0), (1, 1, -0.5), (1, 1, None),
])
def test_actual_diagnostic_boundary_aborts_before_any_team_provider(
    accepted, returned, delta,
):
    local = _Local(accepted=accepted, returned=returned, delta=delta)
    evaluator = _Evaluator(promote=True)
    controller = TeamSearchController(
        responsibility=_Assignment(), task_builder=LocalTaskBuilder(),
        local_optimizer=local, evaluator=evaluator,
        selector=CommonSafeTeamCandidateSelector(), committer=_NoCommit(),
        diagnostic_full_for_local_accepts=True,
    )
    with pytest.raises(DiagnosticSamplingIntegrityError):
        asyncio.run(controller.run_opportunity(
            TeamSearchRequest(81, 0, "parent-team", 36,
                              "COMMON_SOLVER_CONTRACT_V1", "task_output_contract_v1")
        ))
    assert local.calls == 1
    assert evaluator.minibatch_calls == 0
    assert evaluator.ordinary_full_calls == evaluator.diagnostic_full_calls == 0
    assert evaluator.shadow_calls == 0


def test_mismatch_aborts_dynamic_engine_before_next_opportunity():
    local = _Local(accepted=1, returned=0)
    evaluator = _Evaluator(promote=True)
    controller = TeamSearchController(
        responsibility=_Assignment(), task_builder=LocalTaskBuilder(),
        local_optimizer=local, evaluator=evaluator,
        selector=CommonSafeTeamCandidateSelector(), committer=_NoCommit(),
        diagnostic_full_for_local_accepts=True,
    )
    seen = []

    class Backend:
        name = "gepa"
        fidelity = "fake"

    def next_opportunity(index, parent):
        seen.append(index)
        return Layer2Opportunity(TeamSearchRequest(
            81, index - 1, parent, 36,
            "COMMON_SOLVER_CONTRACT_V1", "task_output_contract_v1",
        ))

    spec = ExperimentSpec(
        OptimizerBackend.GEPA, OptimizationScope.LAYER2,
        StoppingRegime.FIXED_BUDGET, "BBH", "fold_ab", fixed_budget_units=10,
    )
    runtime = RuntimeContext(81, "fake", "solver", "reflection", "evaluator",
                             "a" * 64, "attempt", "cache", "ledger")
    with pytest.raises(DiagnosticSamplingIntegrityError):
        asyncio.run(run_experiment(
            spec, runtime, ExperimentInputs(initial_state_hash="parent-team"),
            ExperimentServices(
                backend=Backend(), layer2_controller_factory=lambda bridge: controller,
                team_state_hash_reader=lambda: "parent-team",
                layer2_opportunity_factory=next_opportunity,
            ),
        ))
    assert seen == [1]
    assert evaluator.minibatch_calls == evaluator.ordinary_full_calls == 0
    assert evaluator.diagnostic_full_calls == evaluator.shadow_calls == 0


def test_stage_costs_partition_ledger_without_double_counting_shared_local():
    def record(record_id, phase, candidate, *, success=1, cache=False, tokens=3):
        return {
            "record_id": record_id, "record_kind": "solver_logical_completion",
            "phase": phase, "logical_role": "reflection" if "reflection" in phase else "solver",
            "update_index": 0, "candidate_id": candidate,
            "provider_attempts": int(not cache), "successful_provider_calls": success,
            "cache_hit": cache, "input_tokens": tokens - 1,
            "output_tokens": 1, "total_tokens": tokens,
        }
    rows = [
        record("local", "local_optimizer_solver_eval", "root"),
        record("reflection", "local_optimizer_reflection", "reflection"),
        record("mini", "team_minibatch_eval", "child", cache=True, success=0),
        record("full", "diagnostic_full_eval", "child"),
    ]
    opportunity = _opportunity_costs(rows, 0)
    candidate = _candidate_stage_costs(
        rows, update_index=0, candidate_id="child", diagnostic_only=True,
    )
    assert opportunity["total"]["provider_successes"] == 3
    assert opportunity["total"]["cache_hits"] == 1
    assert opportunity["total"]["total_tokens"] == 12
    assert candidate["local_optimizer"]["scope"] == "opportunity_shared_not_candidate_additive"
    assert candidate["local_optimizer"]["reflection_calls"] is None
    assert candidate["local_optimizer"]["reflection_provider_records"] == 1
    assert candidate["team_minibatch"]["cache_hits"] == 1
    assert candidate["full"]["ordinary_or_diagnostic"] == "diagnostic"
    assert candidate["full"]["provider_successes"] == 1
    assert candidate["shadow"]["reached"] is False
    cached_shadow = _candidate_stage_costs(
        rows, update_index=0, candidate_id="child", diagnostic_only=True,
        shadow_reached=True,
    )
    assert cached_shadow["shadow"]["reached"] is True
    assert cached_shadow["shadow"]["provider_successes"] == 0
    assert _usage(rows, scope="global")["provider_success_tokens"] == 9


def test_duplicate_accepted_events_are_marked_not_deduplicated():
    seen = {}
    first = {"parent_team_hash": "S0", "target_member": 2}
    second = dict(first)
    _mark_duplicate_accepted_event(first, "a" * 64, seen)
    _mark_duplicate_accepted_event(second, "a" * 64, seen)
    assert len(seen) == 1
    assert first["duplicate_accepted_candidate_group"] == second[
        "duplicate_accepted_candidate_group"
    ]
    assert len(first["duplicate_accepted_candidate_group"]) == 64


def test_scheduler_feedback_ignores_diagnostic_full_vote_sign():
    class Scheduler:
        def __init__(self):
            self.state = []

        def record_outcome(self, *, decision, update_index, committed_member_id, valid_outcome):
            self.state.append((decision, update_index, committed_member_id, valid_outcome))

    states = []
    for diagnostic_vote_delta in (-2, 3):
        scheduler = Scheduler()
        outcome = SimpleNamespace(audit_metadata={
            "committed_member_id": None,
            "transfer_diagnostic": [{"full": {"vote_delta": diagnostic_vote_delta}}],
        })
        _record_ordinary_scheduler_outcome(
            scheduler, decision="frozen", update_index=0, outcome=outcome,
        )
        states.append(scheduler.state)
    assert states == [[("frozen", 0, None, True)]] * 2


def test_parent_changes_iff_ordinary_commit_not_diagnostic_full():
    state = {"hash": "S0"}
    parents = []

    class Backend:
        name = "gepa"
        fidelity = "fake"

    class Controller:
        async def run_opportunity(self, request):
            parents.append(request.team_state_hash)
            commit = request.update_index == 1
            if commit:
                state["hash"] = "S1"
            return TeamSearchOutcome(
                _Assignment().value, (), "ordinary" if commit else None,
                "completed", TeamCostAccounting(),
                funnel={"local_candidates": 1},
                audit_metadata={"transfer_diagnostic": [{"full": {
                    "vote_delta": 3}, "committed": False}]},
            )

    def next_opportunity(index, parent):
        assert parent == state["hash"]
        return Layer2Opportunity(TeamSearchRequest(
            81, index - 1, parent, 36,
            "COMMON_SOLVER_CONTRACT_V1", "task_output_contract_v1",
        ))

    spec = ExperimentSpec(
        OptimizerBackend.GEPA, OptimizationScope.LAYER2,
        StoppingRegime.FIXED_BUDGET, "BBH", "fold_ab", fixed_budget_units=3,
    )
    runtime = RuntimeContext(81, "fake", "solver", "reflection", "evaluator",
                             "a" * 64, "attempt", "cache", "ledger")
    result = asyncio.run(run_experiment(
        spec, runtime, ExperimentInputs(initial_state_hash="S0"),
        ExperimentServices(
            backend=Backend(), layer2_controller_factory=lambda bridge: Controller(),
            team_state_hash_reader=lambda: state["hash"],
            layer2_opportunity_factory=next_opportunity,
        ),
    ))
    assert parents == ["S0", "S0", "S1"]
    assert [row.committed_candidate_id is not None for row in result.team_outcomes] == [
        False, True, False,
    ]


@pytest.mark.parametrize("phase", ["diagnostic_full_eval", "team_shadow_eval"])
def test_emergency_ceiling_marks_incomplete_attempt_aborted(tmp_path, monkeypatch, phase):
    from scripts import run_experiment as entry

    run_root = tmp_path / "formal"
    run_root.mkdir()
    (run_root / "ledger.jsonl").write_text(json.dumps({
        "record_id": "attempt-1", "record_kind": "solver_logical_completion",
        "phase": phase, "logical_role": "solver", "provider_attempts": 1,
        "successful_provider_calls": 1, "cache_hit": False,
        "input_tokens": 2, "output_tokens": 1, "total_tokens": 3,
    }) + "\n", encoding="utf-8")
    permit = SimpleNamespace(
        experiment_id="gepa_layer2_local_to_team_transfer_diagnostic_v2",
        run_root=run_root,
    )
    monkeypatch.setattr(entry, "validate_execution", lambda **kwargs: permit)
    monkeypatch.setattr(entry, "admit_execution", lambda existing, root: permit)
    statuses = []
    monkeypatch.setattr(entry, "terminal_lifecycle", lambda *args, **kwargs: statuses.append(kwargs))

    async def incomplete(*args, **kwargs):
        raise RuntimeError("successful_provider_emergency_ceiling")

    monkeypatch.setattr(entry, "execute_online_transfer_diagnostic", incomplete)
    with pytest.raises(RuntimeError, match="successful_provider_emergency_ceiling"):
        asyncio.run(entry.execute_frozen(tmp_path / "prep", run_root))
    assert statuses[0]["status"] == "ABORTED"
    assert not (run_root / "execution_summary.json").exists()


def test_local_capacity_and_unconstrained_cost_envelope():
    capacity = local_gepa_budget_capacity(
        metric_budget=36, validation_size=12, reflection_minibatch_size=3,
    )
    assert (capacity.seed_evaluation_calls, capacity.proposal_attempt_calls,
            capacity.accepted_full_evaluation_calls) == (12, 6, 12)
    assert capacity.max_accepted_children == 1
    assert capacity.max_rejected_proposals == 4
    assert 500 + 10 * 36 + 20 + 5 * 100 + 5 * 6 * 50 == 2880


def test_frozen_target_and_proposal_stop_rules():
    assert _diagnostic_stop_reason(4, 16) is None
    assert _diagnostic_stop_reason(5, 16) == "ACCEPTED_MUTATION_TARGET_REACHED"
    assert _diagnostic_stop_reason(4, 17) == "REFLECTION_PROPOSAL_PREOPPORTUNITY_GUARD"
    assert _diagnostic_stop_reason(4, 20) == "REFLECTION_PROPOSAL_CEILING_REACHED"
    with pytest.raises(RuntimeError, match="ceiling overshoot"):
        _diagnostic_stop_reason(6, 1)


def test_global_provider_attempt_reservations_fail_closed(tmp_path):
    ledger = CappedDurableLedger(
        tmp_path / "ledger.jsonl", successful_ceiling=1, attempt_ceiling=2,
    )
    row = {
        "record_id": "r1", "phase": "diagnostic_full_eval", "logical_role": "solver",
        "client_role": "solver", "provider_attempts": 1,
        "successful_provider_calls": 1, "cache_hit": False,
        "input_tokens": 2, "output_tokens": 1, "total_tokens": 3,
        "seed": 81, "arm": "diagnostic", "update_index": 0,
        "target_member": 0,
    }
    ledger.reserve_provider_attempt("solver")
    with pytest.raises(RuntimeError, match="successful_provider_emergency_ceiling"):
        ledger.reserve_provider_attempt("reflection")
    ledger.append(row)
    with pytest.raises(RuntimeError, match="successful_provider_emergency_ceiling"):
        ledger.reserve_provider_attempt("reflection")


def test_dynamic_engine_uses_only_actual_committed_parent():
    state = {"hash": "S0"}
    seen = []
    assignment = _Assignment().value

    class Backend:
        name = "gepa"
        fidelity = "fake"

    class Controller:
        async def run_opportunity(self, request):
            seen.append(request.team_state_hash)
            if request.update_index == 0:
                state["hash"] = "S1"  # fake ordinary commit only
                committed = "child0"
            else:
                committed = None
            return TeamSearchOutcome(
                assignment, (), committed, "completed", TeamCostAccounting(),
                funnel={"local_candidates": 0},
            )

    def next_opportunity(index, parent):
        assert parent == state["hash"]
        return Layer2Opportunity(TeamSearchRequest(
            81, index - 1, parent, 36,
            "COMMON_SOLVER_CONTRACT_V1", "task_output_contract_v1",
        ))

    spec = ExperimentSpec(
        OptimizerBackend.GEPA, OptimizationScope.LAYER2,
        StoppingRegime.FIXED_BUDGET, "BBH", "fold_ab", fixed_budget_units=3,
    )
    runtime = RuntimeContext(81, "fake", "solver", "reflection", "evaluator",
                             "a" * 64, "attempt", "cache", "ledger")
    result = asyncio.run(run_experiment(
        spec, runtime, ExperimentInputs(initial_state_hash="S0"),
        ExperimentServices(
            backend=Backend(), layer2_controller_factory=lambda bridge: Controller(),
            team_state_hash_reader=lambda: state["hash"],
            layer2_opportunity_factory=next_opportunity,
            layer2_outcome_observer=lambda index, outcome: (
                "DIAGNOSTIC_STOP" if index == 2 else None
            ),
        ),
    ))
    assert seen == ["S0", "S1"]
    assert result.final_state_hash == "S1"
    assert result.stop_reason == "DIAGNOSTIC_STOP"


def test_frozen_preparation_has_no_authorization_or_validation(monkeypatch):
    from scripts.prepare_online_transfer_diagnostic import frozen_payload
    from multi_dataset_diverse_rl.provider_credentials import LWJ_DASHSCOPE_BASE_URL_ENV

    monkeypatch.setenv(LWJ_DASHSCOPE_BASE_URL_ENV, "https://example.invalid/compatible-mode/v1")
    manifest, protocol = frozen_payload(execution_source_sha="a" * 40)
    assert manifest["runtime"]["seed"] == protocol["seed"] == 81
    assert manifest["api_authorization"]["authorized"] is False
    assert manifest["api_authorization"]["allowed_phases"] == ["diagnostic"]
    assert manifest["access"] == {"validation50_calls": 0, "test50_calls": 0}
    assert manifest["diagnostic_contract"]["mandatory_full"] is True
    assert protocol["accepted_mutation_target"] == 5
    assert protocol["successful_provider_ceiling"] == 1200


def test_v2_integrity_amendment_preserves_scientific_freeze(monkeypatch):
    from scripts.prepare_online_transfer_diagnostic import frozen_payload as old
    from scripts.prepare_online_transfer_diagnostic_v2 import frozen_payload as amended
    from multi_dataset_diverse_rl.provider_credentials import LWJ_DASHSCOPE_BASE_URL_ENV

    monkeypatch.setenv(LWJ_DASHSCOPE_BASE_URL_ENV, "https://example.invalid/compatible-mode/v1")
    v1, old_protocol = old(execution_source_sha="a" * 40)
    v2, new_protocol = amended(execution_source_sha="a" * 40)
    for key in ("scientific", "diagnostic_contract", "runtime", "models", "dependency", "access"):
        assert v1[key] == v2[key]
    for key in (
        "seed", "optimize_rows", "shadow_rows", "accepted_mutation_target",
        "max_opportunities", "reflection_proposal_ceiling",
        "successful_provider_ceiling", "transport_attempt_ceiling",
        "team_minibatch_rows", "full_rows_per_accepted_mutation",
        "diagnostic_full_policy", "ordinary_shadow_policy",
    ):
        assert old_protocol[key] == new_protocol[key]
    assert v2["integrity_amendment"]["scientific_method_changed"] is False
    assert v2["api_authorization"]["authorized"] is False
    assert new_protocol["accepted_frontier_enforcement"] == "immediate_pre_team_typed_abort"


def test_dynamic_preopportunity_stop_uses_no_extra_parent_or_provider():
    class Backend:
        name = "gepa"
        fidelity = "fake"

    class Controller:
        async def run_opportunity(self, request):
            raise AssertionError("no actionable task should enter the optimizer")

    spec = ExperimentSpec(
        OptimizerBackend.GEPA, OptimizationScope.LAYER2,
        StoppingRegime.FIXED_BUDGET, "BBH", "fold_ab", fixed_budget_units=10,
    )
    runtime = RuntimeContext(81, "fake", "solver", "reflection", "evaluator",
                             "a" * 64, "attempt", "cache", "ledger")

    def no_opportunity(index, parent):
        assert (index, parent) == (1, "S0")
        raise ExperimentEarlyStop("NO_ALIGNED_RESPONSIBILITY_TARGET_NOT_REACHED")

    result = asyncio.run(run_experiment(
        spec, runtime, ExperimentInputs(initial_state_hash="S0"),
        ExperimentServices(
            backend=Backend(), layer2_controller_factory=lambda bridge: Controller(),
            team_state_hash_reader=lambda: "S0",
            layer2_opportunity_factory=no_opportunity,
        ),
    ))
    assert result.events == ()
    assert result.stop_reason == "NO_ALIGNED_RESPONSIBILITY_TARGET_NOT_REACHED"


def test_offline_auditor_detects_false_negative_without_online_override(tmp_path):
    from scripts.audit_online_transfer_diagnostic import audit
    from multi_dataset_diverse_rl.team_search.execution_runtime import ledger_summary

    (tmp_path / "ledger.jsonl").write_text(json.dumps({
        "record_id": "r1", "phase": "diagnostic_full_eval", "logical_role": "solver",
        "provider_attempts": 1, "successful_provider_calls": 1,
        "cache_hit": False, "input_tokens": 2, "output_tokens": 1,
        "total_tokens": 3,
    }) + "\n", encoding="utf-8")
    usage = ledger_summary(tmp_path / "ledger.jsonl")
    (tmp_path / "run_lifecycle.json").write_text(
        json.dumps({"status": "EXECUTION_COMPLETE"}), encoding="utf-8",
    )
    summary = {
        "experiment_id": "gepa_layer2_local_to_team_transfer_diagnostic_v1",
        "seed": 81, "validation50_calls": 0, "test50_calls": 0,
        "ledger": usage, "opportunities": 1, "accepted_mutations": 1,
        "reflection_proposals": 1, "target_status": "TARGET_NOT_REACHED",
        "parent_sequence": ["S0"],
        "candidate_diagnostics": [{
            "update_index": 0, "parent_team_hash": "S0",
            "local_acceptance_delta": 1.0,
            "team_minibatch": {"passed": False},
            "full": {"vote_delta": 1, "diagnostic_only": True},
            "ordinary_common_safe": "NOT_REACHED",
            "ordinary_shadow": "NOT_REACHED", "committed": False,
        }],
    }
    (tmp_path / "execution_summary.json").write_text(
        json.dumps(summary), encoding="utf-8",
    )
    result = audit(tmp_path)
    assert result["gate"] == "PASS"
    assert result["minibatch_false_negative_full_vote_positive"] == 1
    summary["candidate_diagnostics"][0]["committed"] = True
    (tmp_path / "execution_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert audit(tmp_path)["gate"] == "HOLD"


def test_v2_completed_fixture_reconciles_candidate_and_global_stage_costs(tmp_path):
    from scripts.audit_online_transfer_diagnostic import audit
    from multi_dataset_diverse_rl.team_search.execution_runtime import ledger_summary

    ledger_rows = [{
        "record_id": "full-1", "record_kind": "solver_logical_completion",
        "phase": "diagnostic_full_eval", "logical_role": "solver",
        "provider_attempts": 1, "successful_provider_calls": 1,
        "cache_hit": False, "input_tokens": 2, "output_tokens": 1,
        "total_tokens": 3, "update_index": 0, "candidate_id": "child",
    }]
    (tmp_path / "ledger.jsonl").write_text(
        "\n".join(json.dumps(row) for row in ledger_rows) + "\n", encoding="utf-8",
    )
    (tmp_path / "run_lifecycle.json").write_text(
        json.dumps({"status": "EXECUTION_COMPLETE"}), encoding="utf-8",
    )
    candidate = {
        "candidate_id": "child", "candidate_hash": "a" * 64,
        "update_index": 0, "parent_team_hash": "S0",
        "local_acceptance_delta": 1.0,
        "team_minibatch": {"passed": False},
        "full": {"vote_delta": 1, "diagnostic_only": True},
        "ordinary_common_safe": "NOT_REACHED", "ordinary_shadow": "NOT_REACHED",
        "committed": False,
        "stage_costs": _candidate_stage_costs(
            ledger_rows, update_index=0, candidate_id="child", diagnostic_only=True,
        ),
    }
    summary = {
        "experiment_id": "gepa_layer2_local_to_team_transfer_diagnostic_v2",
        "seed": 81, "validation50_calls": 0, "test50_calls": 0,
        "ledger": ledger_summary(tmp_path / "ledger.jsonl"),
        "opportunities": 1, "accepted_mutations": 1,
        "reflection_proposals": 1, "target_status": "TARGET_NOT_REACHED",
        "parent_sequence": ["S0"], "candidate_diagnostics": [candidate],
        "stage_accounting": {
            "global": _usage(ledger_rows, scope="global"),
            "initialization": _usage([], scope="global_initialization"),
            "opportunities": [_opportunity_costs(ledger_rows, 0)],
        },
    }
    path = tmp_path / "execution_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert audit(tmp_path)["gate"] == "PASS"
    summary["candidate_diagnostics"][0]["stage_costs"]["full"]["provider_successes"] = 0
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert audit(tmp_path)["failed_checks"] == ["candidate_stage_cost_reconciliation"]
