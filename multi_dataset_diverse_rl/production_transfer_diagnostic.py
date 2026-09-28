"""One prospectively bounded, online GEPA local-to-team diagnostic.

This is execution composition only.  The Layer-1 search, Layer-2 scheduler,
MiniBatch, Common-Safe, Shadow and committer are the ordinary implementations.
Diagnostic Full results are isolated in the controller's observational sidecar.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import Config
from .evaluation.output_contract import SOLVER_OUTPUT_CONTRACT_VERSION
from .experiment import (
    ExperimentEarlyStop, ExperimentInputs, ExperimentServices, Layer2Opportunity, RuntimeContext,
    experiment_spec_from_mapping, run_experiment,
)
from .governance.freeze_hash import source_freeze_sha256
from .governance.production_execution import ValidatedExecutionContext, mark_provider_client_constructed
from .local_optimizers.gepa_native import GEPALayer2EvidenceOptimizer
from .local_optimizers.gepa_optimizer import GEPALocalPromptOptimizer, local_gepa_budget_capacity
from .governance import v4_attempt3_contract as attempt3_contract
from .local_optimizers.production_backends import GEPABackend
from .persistence.identity import build_run_identity
from .persistence.durable_io import io_path
from .production_canary import ContextBoundGEPALayer2, _UnusedNative
from .team_search.candidate_selector import CommonSafeTeamCandidateSelector
from .team_search.controller import TeamSearchController
from .team_search.execution_runtime import (
    CappedDurableLedger, CommonContractExecutionSystem, ReflectionLM,
    ledger_summary, read_csv_rows,
)
from .team_search.primary_responsibility_scheduler import PrimaryResponsibilityPersistentRealizabilityScheduler
from .team_search.feasibility import Layer2EvidenceInfeasible, Layer2FeasibilityReason
from .team_search.schemas import TeamSearchAssignment, TeamSearchRequest
from .team_search.system_runtime import (
    LatestTransitionStore, SystemLocalSolverEvaluator,
    SystemResponsibilityAssignmentFactory, SystemTeamCandidateEvaluator,
    SystemTeamCommitter, freeze_current_responsibility,
)
from .team_search.task_builder import Layer2EvidenceRequestBuilder
from .team_search.v4_opportunity import (
    V4NoFeasibleOpportunity, select_v4_feasible_opportunity,
)
from .versions import (
    PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
    PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION,
    LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION,
)


_OPPORTUNITY_PHASES = (
    "local_optimizer_solver_eval", "local_optimizer_reflection",
    "team_minibatch_eval", "team_full_eval", "diagnostic_full_eval",
    "team_shadow_eval",
)


def _ledger_rows(path: Path) -> list[dict[str, Any]]:
    with open(io_path(path), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _usage(rows: list[dict[str, Any]], *, scope: str) -> dict[str, Any]:
    """Durable-ledger accounting, including cache hits and failed attempts.

    Ledger tokens include reported token counts on cached logical completions;
    provider-success tokens exclude those cache echoes. The two are not mixed.
    """
    completed = [row for row in rows if row.get("record_kind") not in {
        "solver_provider_attempt_failure", "solver_provider_attempt_success",
    }]
    provider = [row for row in rows if int(row.get("successful_provider_calls", 0))]
    reflection_records = sum(row.get("logical_role") == "reflection" for row in completed)
    return {
        "scope": scope,
        "logical_solver_rows": sum(row.get("logical_role") == "solver" for row in completed),
        "reflection_calls": None if reflection_records else 0,
        "reflection_provider_records": reflection_records,
        "logical_calls": None if reflection_records else len(completed),
        "provider_attempts": sum(int(row["provider_attempts"]) for row in rows),
        "provider_successes": sum(int(row["successful_provider_calls"]) for row in rows),
        "failed_provider_attempts": sum(
            int(row["provider_attempts"]) - int(row["successful_provider_calls"])
            for row in rows
        ),
        "cache_hits": sum(bool(row["cache_hit"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "provider_success_tokens": sum(int(row["total_tokens"]) for row in provider),
    }


def _opportunity_costs(rows: list[dict[str, Any]], update_index: int) -> dict[str, Any]:
    selected = [row for row in rows if int(row.get("update_index", -2)) == update_index]
    if {str(row["phase"]) for row in selected} - set(_OPPORTUNITY_PHASES):
        raise RuntimeError("diagnostic opportunity has unexpected ledger phase")
    return {
        "update_index": update_index,
        "total": _usage(selected, scope="opportunity_shared"),
        "by_phase": {
            phase: _usage([row for row in selected if row["phase"] == phase],
                          scope="opportunity_shared")
            for phase in _OPPORTUNITY_PHASES
        },
    }


def _candidate_stage_costs(
    rows: list[dict[str, Any]], *, update_index: int, candidate_id: str,
    diagnostic_only: bool, shadow_reached: bool | None = None,
) -> dict[str, Any]:
    selected = [row for row in rows if int(row.get("update_index", -2)) == update_index]
    attributable = lambda phase: [
        row for row in selected
        if row["phase"] == phase and row.get("candidate_id") == candidate_id
    ]
    local = [row for row in selected if row["phase"] in {
        "local_optimizer_solver_eval", "local_optimizer_reflection",
    }]
    full_phase = "diagnostic_full_eval" if diagnostic_only else "team_full_eval"
    shadow = attributable("team_shadow_eval")
    return {
        "local_optimizer": _usage(local, scope="opportunity_shared_not_candidate_additive"),
        "team_minibatch": _usage(attributable("team_minibatch_eval"),
                                 scope="candidate_attributable"),
        "full": {
            "ordinary_or_diagnostic": "diagnostic" if diagnostic_only else "ordinary",
            **_usage(attributable(full_phase), scope="candidate_attributable"),
        },
        "shadow": {
            "reached": bool(shadow) if shadow_reached is None else shadow_reached,
            **_usage(shadow, scope="candidate_attributable"),
        },
    }


def _mark_duplicate_accepted_event(
    row: dict[str, Any], candidate_hash: str,
    seen: dict[tuple[str, int, str], dict[str, Any]],
) -> None:
    """Mark repeated accepted events without changing their count or order."""
    group = (row["parent_team_hash"], int(row["target_member"]), candidate_hash)
    prior = seen.get(group)
    row["duplicate_accepted_candidate_group"] = None
    if prior is not None:
        group_id = hashlib.sha256(json.dumps(group).encode("utf-8")).hexdigest()
        prior["duplicate_accepted_candidate_group"] = group_id
        row["duplicate_accepted_candidate_group"] = group_id
    else:
        seen[group] = row


def _record_ordinary_scheduler_outcome(
    scheduler: PrimaryResponsibilityPersistentRealizabilityScheduler,
    *, decision: Any, update_index: int, outcome: Any,
) -> None:
    """The diagnostic Full sidecar has no scheduler feedback channel."""
    scheduler.record_outcome(
        decision=decision, update_index=update_index,
        committed_member_id=outcome.audit_metadata.get("committed_member_id"),
        valid_outcome=True,
    )


def _diagnostic_stop_reason(accepted_total: int, proposal_total: int, *, attempt3: bool = False) -> str | None:
    if attempt3:
        if proposal_total > attempt3_contract.PROPOSAL_CEILING:
            raise RuntimeError("diagnostic proposal ceiling overshoot")
        if accepted_total >= attempt3_contract.SAMPLE_TARGET:
            return "RETURNED_CANDIDATE_TARGET_REACHED"
        if proposal_total == attempt3_contract.PROPOSAL_CEILING:
            return "REFLECTION_PROPOSAL_CEILING_REACHED"
        if proposal_total + attempt3_contract.MAX_PROPOSALS_PER_OPPORTUNITY > attempt3_contract.PROPOSAL_CEILING:
            return "REFLECTION_PROPOSAL_PREOPPORTUNITY_GUARD"
        return None
    if accepted_total > 5 or proposal_total > 20:
        raise RuntimeError("diagnostic accepted/proposal ceiling overshoot")
    if accepted_total == 5:
        return "ACCEPTED_MUTATION_TARGET_REACHED"
    if proposal_total == 20:
        return "REFLECTION_PROPOSAL_CEILING_REACHED"
    capacity = local_gepa_budget_capacity(
        metric_budget=36, validation_size=12, reflection_minibatch_size=3,
    )
    if proposal_total + capacity.max_rejected_proposals > 20:
        return "REFLECTION_PROPOSAL_PREOPPORTUNITY_GUARD"
    return None


class _CurrentAssignment:
    def __init__(self) -> None:
        self.assignment: TeamSearchAssignment | None = None
        self.parent_hash: str | None = None

    def assign(self, request: TeamSearchRequest) -> TeamSearchAssignment:
        if self.assignment is None or request.team_state_hash != self.parent_hash:
            raise RuntimeError("online assignment/parent mismatch")
        return self.assignment


async def execute_online_transfer_diagnostic(
    permit: ValidatedExecutionContext, *, root: Path,
) -> dict[str, Any]:
    if not permit.admitted or permit.run_root is None:
        raise PermissionError("diagnostic execution requires an admitted permit")
    prep, run_root = permit.prep_root, permit.run_root
    manifest = json.loads((prep / "manifest.json").read_text(encoding="utf-8"))
    spec = experiment_spec_from_mapping(manifest["scientific"])
    protocol = json.loads((prep / "protocol.json").read_text(encoding="utf-8"))
    attempt3 = permit.attempt_id == "gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt3"
    if (
        permit.experiment_id not in {
            "gepa_layer2_local_to_team_transfer_diagnostic_v2",
            "gepa_layer2_local_to_team_transfer_diagnostic_v3",
            "gepa_layer2_local_to_team_transfer_diagnostic_v4",
        }
        or permit.allowed_phase != "diagnostic"
        or spec.mode_id != "GEPA_LAYER2"
        or spec.fixed_budget_units != 10
        or (protocol.get("returned_candidate_target") if attempt3 else protocol.get("accepted_mutation_target")) != 5
        or protocol["reflection_proposal_ceiling"] != 20
    ):
        raise ValueError("online transfer diagnostic freeze mismatch")
    runtime = RuntimeContext(
        seed=int(manifest["runtime"]["seed"]),
        provider_profile=permit.provider_profile,
        solver_model=str(manifest["runtime"]["solver_model"]),
        optimizer_model=str(manifest["runtime"]["optimizer_model"]),
        evaluator_model=str(manifest["runtime"]["evaluator_model"]),
        run_identity_sha256=permit.run_identity_sha256,
        authorization_identity=permit.attempt_id,
        cache_identity=permit.run_identity_sha256,
        ledger_identity=permit.run_identity_sha256,
    )
    if runtime.seed != 81:
        raise ValueError("diagnostic seed identity mismatch")
    optimize_path = prep / "splits_private/optimize100.csv"
    shadow_path = prep / "splits_private/shadow50.csv"
    optimize_rows, shadow_rows = read_csv_rows(optimize_path), read_csv_rows(shadow_path)
    cfg = Config.from_flat(
        task_type="bbh", dataset_format="mars", comparison_task_id="disambiguation_qa",
        benchmark="BBH", answer_format="option_letter",
        train_path=str(optimize_path), val_path=str(shadow_path),
        test_path="TEST50_BLOCKED", manifest_sha256=source_freeze_sha256(prep / "manifest.json"),
        train_size=100, val_size=50, test_size=0,
        provider_profile=permit.provider_profile,
        agent_model=runtime.solver_model, optimizer_model=runtime.optimizer_model,
        evaluator_model=runtime.evaluator_model, temperature=0.0,
        solver_max_tokens=1800, solver_invalid_max_retries=0,
        solver_contract_id=spec.solver_contract_id,
        experiment_setting="experimental_diversity_d2_rr_generic",
        target_scheduler="round_robin", agents=5, epochs=1, update_every=1,
        seed=runtime.seed, proposal_memory_mode="off", num_candidates_per_parent=2,
        candidate_eval_pool_size=100, eval_solver_call_concurrency=8,
        stage_b_candidate_budget=2, out_dir=str(run_root / "system"),
        shared_solver_cache_path="", provider_call_budget=(
            attempt3_contract.SUCCESSFUL_PROVIDER_CEILING if attempt3 else 1200
        ),
        total_token_budget=100_000_000, final_test_enabled=False,
        preserve_final_checkpoint=True,
    )
    ledger = CappedDurableLedger(
        run_root / "ledger.jsonl", successful_ceiling=(
            attempt3_contract.SUCCESSFUL_PROVIDER_CEILING if attempt3 else 1200
        ), attempt_ceiling=(
            attempt3_contract.TRANSPORT_ATTEMPT_CEILING if attempt3 else 4800
        ),
    )
    system = CommonContractExecutionSystem(
        cfg, arm="GEPA_LAYER2_ONLINE_TRANSFER_DIAGNOSTIC", ledger=ledger, raw_cache={},
    )
    mark_provider_client_constructed(permit)
    system.set_run_identity(build_run_identity(
        cfg, train_rows=optimize_rows, val_rows=shadow_rows,
        test_rows=[], workspace=root,
    ))
    system.set_stage({
        "phase": "initialization", "update_index": -1,
        "target_member": -1, "candidate_id": "P0",
    })
    try:
        await system.initialize_fixed_probe(optimize_rows)
    finally:
        system.set_stage(None)

    initial_hash = system.team_prompt_state_hash()
    current = _CurrentAssignment()
    v4 = permit.experiment_id == "gepa_layer2_local_to_team_transfer_diagnostic_v4"
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler(
        version=(PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION if v4
                 else PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION),
    )
    transition_store = LatestTransitionStore()
    task_builder = Layer2EvidenceRequestBuilder(bounded_search_view=v4)
    loop = asyncio.get_running_loop()
    update_index = 0
    local_solver = SystemLocalSolverEvaluator(
        system=system, loop=loop, stage=system.set_stage,
        accounting=system.common.accounting,
        solver_contract_id=spec.solver_contract_id,
        output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
    )
    official = GEPALocalPromptOptimizer(
        evaluator=local_solver, reflection_lm=ReflectionLM(system),
        accounting_reader=system.optimizer_accounting,
        run_root=run_root / "local_gepa",
    )
    layer2 = ContextBoundGEPALayer2(
        GEPALayer2EvidenceOptimizer(engine=official),
        local_solver, runtime, system, update_index_reader=lambda: update_index,
        saturation_mode="online_local_to_team_transfer_diagnostic_v1",
    )
    backend = GEPABackend(native=_UnusedNative(), layer2=layer2)
    evaluator = SystemTeamCandidateEvaluator(
        system=system, shadow_probe=system.build_probe(shadow_rows), loop=loop,
        stage=system.set_stage, accounting=system.common.accounting,
        update_index_reader=lambda: update_index,
    )
    decision = None
    accepted_total = proposal_total = 0
    internal_accepted_total = 0
    local_boundary_trace: list[dict[str, Any]] = []
    parent_sequence: list[str] = []
    opportunity_costs: list[dict[str, Any]] = []
    accepted_by_group: dict[tuple[str, int, str], dict[str, Any]] = {}
    feasibility_trace: list[dict[str, Any]] = []
    evidence_trace: list[dict[str, Any]] = []

    def next_opportunity(index: int, parent_hash: str) -> Layer2Opportunity:
        nonlocal update_index, decision
        update_index = index - 1
        if system.team_prompt_state_hash() != parent_hash:
            raise RuntimeError("online parent must equal actual committed team state")
        snapshot = freeze_current_responsibility(system, update_index=update_index)
        raw_decision = scheduler.select(
            assigned=snapshot.assigned,
            current_margin_by_question=snapshot.current_margin_by_question,
            seed=runtime.seed, update_index=update_index, target_count=1,
        )
        if not v4 and not any(row.primary_score > 0 for row in raw_decision.summaries):
            raise ExperimentEarlyStop("NO_ALIGNED_RESPONSIBILITY_TARGET_NOT_REACHED")
        request = TeamSearchRequest(
            seed=runtime.seed, update_index=update_index,
            team_state_hash=parent_hash, local_metric_budget=36,
            solver_contract_id=spec.solver_contract_id,
            output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
        )
        factory = SystemResponsibilityAssignmentFactory(
            system=system, snapshot_reader=lambda: snapshot,
            task_builder=task_builder, transition_store=transition_store,
        )
        if v4:
            selected = select_v4_feasible_opportunity(
                snapshot=snapshot, scheduler=scheduler, factory=factory,
                task_builder=task_builder, request=request,
                raw_decision=raw_decision,
            )
            feasibility_reasons = dict(selected.feasibility_reasons)
            raw_order = sorted(raw_decision.summaries, key=lambda row: (-row.target_score, row.member_id))
            eligible_order = [
                row for row in raw_order
                if not isinstance(selected, V4NoFeasibleOpportunity)
                and row.member_id in selected.eligible_member_ids
            ]
            feasibility_trace.append({
                "update_index": update_index,
                "parent_team_hash": parent_hash,
                "policy_version": LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION,
                "members": [{
                    "member_id": row.member_id,
                    "direct_count": row.direct_count,
                    "near_margin_count": row.near_margin_count,
                    "coverage_count": row.coverage_count,
                    "raw_V": row.primary_score,
                    "failure_count": row.failure_count,
                    "raw_target_score": row.target_score,
                    "primary_lane": row.primary_lane,
                    "feasibility_reason": feasibility_reasons[row.member_id],
                    "raw_rank": raw_order.index(row) + 1,
                    "feasible_rank": (eligible_order.index(row) + 1 if row in eligible_order else None),
                } for row in raw_decision.summaries],
                "selected_member": eligible_order[0].member_id if eligible_order else None,
                "latest_transition_effect_hash_by_member": {
                    str(member_id): (
                        transition_store.get(member_id).transition_effect_hash
                        if transition_store.get(member_id) is not None else None
                    )
                    for member_id in range(5)
                },
            })
            if isinstance(selected, V4NoFeasibleOpportunity):
                raise ExperimentEarlyStop("NO_FEASIBLE_LAYER2_OPPORTUNITY")
            decision = selected.decision
        else:
            decision = raw_decision
        if len(decision.selected_member_ids) != 1:
            raise RuntimeError("online diagnostic requires one production target")
        target = decision.selected_member_ids[0]
        summary = next(row for row in decision.summaries if row.member_id == target)
        if v4:
            packet = selected.packet
            evidence_trace.append({
                "update_index": update_index,
                "parent_team_hash": parent_hash,
                "target_member": target,
                "raw_V": summary.primary_score,
                "assignment_V": selected.assignment.responsibility_value,
                "packet_V": packet.responsibility_value,
                "responsibility_universe": {
                    key: value for key, value in packet.provenance
                    if key.startswith("responsibility_universe_")
                },
                "responsibility_scheduled": {
                    "ids": [row.example_id for row in packet.responsibility_examples],
                    "count": len(packet.responsibility_examples),
                },
                "focus_ids": [row.example_id for row in packet.focus_examples],
                "anchor_ids": [row.example_id for row in packet.anchor_examples],
                "nominal_schedule": [list(batch) for batch in packet.ordered_batch_schedule],
                "team_minibatch_ids": list(selected.assignment.local_validation_example_ids),
                "local_eval_ids": [row.example_id for row in packet.local_eval_examples],
                "packet_hash": packet.packet_hash,
                "latest_transition_effect_hash": (
                    packet.latest_transition.transition_effect_hash
                    if packet.latest_transition is not None else None
                ),
                "latest_transition_parent_candidate_hash": (
                    packet.latest_transition.parent_candidate_hash
                    if packet.latest_transition is not None else None
                ),
                "latest_transition_child_candidate_hash": (
                    packet.latest_transition.child_candidate_hash
                    if packet.latest_transition is not None else None
                ),
            })
        current.assignment = (
            selected.assignment if v4 else factory.build_from_member(
                request=request, member_id=target,
                primary_lane=summary.primary_lane,
                responsibility_identity=PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
                responsibility_value=summary.primary_score,
            )
        )
        current.parent_hash = parent_hash
        parent_sequence.append(parent_hash)
        return Layer2Opportunity(
            request,
            eligible_member_ids=selected.eligible_member_ids if v4 else None,
        )

    def observe(index: int, outcome) -> str | None:
        nonlocal accepted_total, proposal_total, internal_accepted_total
        if decision is None:
            raise RuntimeError("scheduler decision missing for online outcome")
        ledger_rows = _ledger_rows(run_root / "ledger.jsonl")
        opportunity_costs.append(_opportunity_costs(ledger_rows, index - 1))
        for row in outcome.audit_metadata.get("transfer_diagnostic", ()):
            candidate = next(
                item.local_candidate for item in outcome.candidates
                if item.local_candidate.candidate_id == row["candidate_id"]
            )
            candidate_hash = hashlib.sha256(candidate.prompt.encode("utf-8")).hexdigest()
            row["candidate_hash"] = candidate_hash
            row["stage_costs"] = _candidate_stage_costs(
                ledger_rows, update_index=index - 1,
                candidate_id=row["candidate_id"],
                diagnostic_only=bool(row["full"]["diagnostic_only"]),
                shadow_reached=row["ordinary_shadow"] != "NOT_REACHED",
            )
            _mark_duplicate_accepted_event(row, candidate_hash, accepted_by_group)
        _record_ordinary_scheduler_outcome(
            scheduler, decision=decision, update_index=index - 1, outcome=outcome,
        )
        telemetry = outcome.audit_metadata["local_optimizer_telemetry"]
        if v4:
            evidence_trace[-1]["evidence_delivered"] = {
                "batch_ids": telemetry["delivered_batch_ids"],
                "role_item_ids": telemetry["evidence_delivered_role_item_ids"],
                "source_ids": telemetry["evidence_delivered_source_ids"],
                "scheduled_but_not_delivered_count": telemetry[
                    "scheduled_but_not_delivered_role_item_count"
                ],
            }
            evidence_trace[-1]["committed_candidate_id"] = outcome.committed_candidate_id
            evidence_trace[-1]["successor_team_hash"] = system.team_prompt_state_hash()
        internal_accepted_total += int(telemetry["accepted_mutations"])
        if attempt3:
            returned_ids = [item.local_candidate.candidate_id for item in outcome.candidates]
            returned_indices = telemetry.get("returned_candidate_indices")
            accepted_indices = telemetry.get("accepted_event_indices")
            if (not isinstance(returned_indices, list) or not isinstance(accepted_indices, list)
                    or len(returned_indices) != len(returned_ids)
                    or len(accepted_indices) != int(telemetry["accepted_mutations"])
                    or not set(returned_indices).issubset(set(accepted_indices))):
                raise RuntimeError("diagnostic Layer1 return/acceptance telemetry mismatch")
            local_boundary_trace.append({
                "update_index": index - 1,
                "accepted_event_indices": accepted_indices,
                "accepted_event_iterations": telemetry.get("accepted_event_iterations", []),
                "frontier_candidate_indices": telemetry.get("frontier_candidate_indices", []),
                "changed_frontier_indices": telemetry.get("changed_frontier_indices", []),
                "valid_unique_frontier_indices": telemetry.get("valid_unique_frontier_indices", []),
                "returned_candidate_indices": returned_indices,
                "returned_candidate_ids": returned_ids,
                "internal_accepted_mutations": len(accepted_indices),
                "returned_candidate_count": len(returned_ids),
                "accepted_to_returned_retention": (
                    len(returned_ids) / len(accepted_indices) if accepted_indices else None
                ),
            })
            accepted_total += len(returned_ids)
        else:
            accepted_total += int(telemetry["accepted_mutations"])
        proposal_total += int(telemetry["proposal_attempts"])
        return _diagnostic_stop_reason(accepted_total, proposal_total, attempt3=attempt3)

    def controller_factory(bound_backend):
        return TeamSearchController(
            responsibility=current, task_builder=task_builder,
            local_optimizer=bound_backend, evaluator=evaluator,
            selector=CommonSafeTeamCandidateSelector(),
            committer=SystemTeamCommitter(
                system=system, evaluator=evaluator,
                update_index_reader=lambda: update_index,
                transition_store=transition_store,
            ),
            diagnostic_full_for_local_accepts=True,
            diagnostic_allow_multi_accepted=attempt3,
        )

    result = await run_experiment(
        spec, runtime, ExperimentInputs(initial_state_hash=initial_hash),
        ExperimentServices(
            backend=backend, layer2_controller_factory=controller_factory,
            team_state_hash_reader=system.team_prompt_state_hash,
            layer2_opportunity_factory=next_opportunity,
            layer2_outcome_observer=observe,
        ),
    )
    rows = [
        dict(row)
        for outcome in result.team_outcomes
        for row in outcome.audit_metadata.get("transfer_diagnostic", ())
    ]
    if len(rows) != accepted_total or any(
        outcome.funnel.get("diagnostic_full_evaluated_candidates", 0)
        != outcome.funnel["local_candidates"]
        for outcome in result.team_outcomes
    ):
        raise RuntimeError("sampled candidate lacks mandatory diagnostic Full")
    final_ledger_rows = _ledger_rows(run_root / "ledger.jsonl")
    if any(int(row.get("update_index", -2)) < -1 for row in final_ledger_rows):
        raise RuntimeError("diagnostic ledger has unassigned update index")
    partition_rows = [
        row for row in final_ledger_rows if int(row.get("update_index", -2)) == -1
    ]
    for index in range(len(result.team_outcomes)):
        partition_rows.extend(
            row for row in final_ledger_rows
            if int(row.get("update_index", -2)) == index
        )
    if len(partition_rows) != len(final_ledger_rows):
        raise RuntimeError("diagnostic ledger partition is incomplete")
    return {
        "experiment_id": permit.experiment_id,
        "attempt_id": permit.attempt_id,
        "run_identity_sha256": permit.run_identity_sha256,
        "seed": runtime.seed,
        "initial_team_hash": initial_hash,
        "final_team_hash": result.final_state_hash,
        "parent_sequence": parent_sequence,
        "feasibility_trace": feasibility_trace,
        "evidence_view_trace": evidence_trace,
        "opportunities": len(result.team_outcomes),
        "accepted_mutations": internal_accepted_total if attempt3 else accepted_total,
        **({
            "sample_unit": attempt3_contract.SAMPLE_UNIT,
            "returned_candidate_count": accepted_total,
            "internal_accepted_mutations": internal_accepted_total,
            "accepted_to_returned_retention": (
                accepted_total / internal_accepted_total if internal_accepted_total else None
            ),
            "local_boundary_trace": local_boundary_trace,
        } if attempt3 else {}),
        "reflection_proposals": proposal_total,
        "target_status": "TARGET_REACHED" if accepted_total >= 5 else "TARGET_NOT_REACHED",
        "stop_reason": result.stop_reason,
        "candidate_diagnostics": rows,
        "stage_accounting": {
            "global": _usage(final_ledger_rows, scope="global"),
            "initialization": _usage(
                [row for row in final_ledger_rows if int(row.get("update_index", -2)) == -1],
                scope="global_initialization",
            ),
            "opportunities": opportunity_costs,
        },
        "commits": sum(outcome.committed_candidate_id is not None for outcome in result.team_outcomes),
        "ledger": ledger_summary(run_root / "ledger.jsonl"),
        "validation50_calls": 0,
        "test50_calls": 0,
    }
