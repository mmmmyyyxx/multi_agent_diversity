"""One prospectively bounded, online GEPA local-to-team diagnostic.

This is execution composition only.  The Layer-1 search, Layer-2 scheduler,
MiniBatch, Common-Safe, Shadow and committer are the ordinary implementations.
Diagnostic Full results are isolated in the controller's observational sidecar.
"""

from __future__ import annotations

import asyncio
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
from .local_optimizers.production_backends import GEPABackend
from .persistence.identity import build_run_identity
from .production_canary import ContextBoundGEPALayer2, _UnusedNative
from .team_search.candidate_selector import CommonSafeTeamCandidateSelector
from .team_search.controller import TeamSearchController
from .team_search.execution_runtime import (
    CappedDurableLedger, CommonContractExecutionSystem, ReflectionLM,
    ledger_summary, read_csv_rows,
)
from .team_search.primary_responsibility_scheduler import PrimaryResponsibilityPersistentRealizabilityScheduler
from .team_search.schemas import TeamSearchAssignment, TeamSearchRequest
from .team_search.system_runtime import (
    LatestTransitionStore, SystemLocalSolverEvaluator,
    SystemResponsibilityAssignmentFactory, SystemTeamCandidateEvaluator,
    SystemTeamCommitter, freeze_current_responsibility,
)
from .team_search.task_builder import Layer2EvidenceRequestBuilder


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
    if (
        permit.experiment_id != "gepa_layer2_local_to_team_transfer_diagnostic_v1"
        or permit.allowed_phase != "diagnostic"
        or spec.mode_id != "GEPA_LAYER2"
        or spec.fixed_budget_units != 10
        or protocol["accepted_mutation_target"] != 5
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
        shared_solver_cache_path="", provider_call_budget=1200,
        total_token_budget=100_000_000, final_test_enabled=False,
        preserve_final_checkpoint=True,
    )
    ledger = CappedDurableLedger(
        run_root / "ledger.jsonl", successful_ceiling=1200, attempt_ceiling=4800,
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
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler()
    transition_store = LatestTransitionStore()
    task_builder = Layer2EvidenceRequestBuilder()
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
    parent_sequence: list[str] = []

    def next_opportunity(index: int, parent_hash: str) -> Layer2Opportunity:
        nonlocal update_index, decision
        update_index = index - 1
        if system.team_prompt_state_hash() != parent_hash:
            raise RuntimeError("online parent must equal actual committed team state")
        snapshot = freeze_current_responsibility(system, update_index=update_index)
        decision = scheduler.select(
            assigned=snapshot.assigned,
            current_margin_by_question=snapshot.current_margin_by_question,
            seed=runtime.seed, update_index=update_index, target_count=1,
        )
        if len(decision.selected_member_ids) != 1:
            raise RuntimeError("online diagnostic requires one production target")
        target = decision.selected_member_ids[0]
        summary = next(row for row in decision.summaries if row.member_id == target)
        if summary.primary_lane == "fallback":
            raise ExperimentEarlyStop("NO_ALIGNED_RESPONSIBILITY_TARGET_NOT_REACHED")
        request = TeamSearchRequest(
            seed=runtime.seed, update_index=update_index,
            team_state_hash=parent_hash, local_metric_budget=36,
            solver_contract_id=spec.solver_contract_id,
            output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
        )
        current.assignment = SystemResponsibilityAssignmentFactory(
            system=system, snapshot_reader=lambda: snapshot,
            task_builder=task_builder, transition_store=transition_store,
        ).build_from_member(
            request=request, member_id=target,
            primary_lane=summary.primary_lane,
            responsibility_identity="primary_responsibility_persistent_realizability_v1",
        )
        current.parent_hash = parent_hash
        parent_sequence.append(parent_hash)
        return Layer2Opportunity(request)

    def observe(index: int, outcome) -> str | None:
        nonlocal accepted_total, proposal_total
        if decision is None:
            raise RuntimeError("scheduler decision missing for online outcome")
        scheduler.record_outcome(
            decision=decision, update_index=index - 1,
            committed_member_id=outcome.audit_metadata.get("committed_member_id"),
            valid_outcome=True,
        )
        telemetry = outcome.audit_metadata["local_optimizer_telemetry"]
        accepted_total += int(telemetry["accepted_mutations"])
        proposal_total += int(telemetry["proposal_attempts"])
        if accepted_total > 5 or proposal_total > 20:
            raise RuntimeError("diagnostic accepted/proposal ceiling overshoot")
        if accepted_total == 5:
            return "ACCEPTED_MUTATION_TARGET_REACHED"
        if proposal_total == 20:
            return "REFLECTION_PROPOSAL_CEILING_REACHED"
        if proposal_total + local_gepa_budget_capacity(
            metric_budget=36, validation_size=12, reflection_minibatch_size=3,
        ).max_rejected_proposals > 20:
            return "REFLECTION_PROPOSAL_PREOPPORTUNITY_GUARD"
        return None

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
        raise RuntimeError("accepted mutation lacks mandatory diagnostic Full")
    return {
        "experiment_id": permit.experiment_id,
        "run_identity_sha256": permit.run_identity_sha256,
        "seed": runtime.seed,
        "initial_team_hash": initial_hash,
        "final_team_hash": result.final_state_hash,
        "parent_sequence": parent_sequence,
        "opportunities": len(result.team_outcomes),
        "accepted_mutations": accepted_total,
        "reflection_proposals": proposal_total,
        "target_status": "TARGET_REACHED" if accepted_total == 5 else "TARGET_NOT_REACHED",
        "stop_reason": result.stop_reason,
        "candidate_diagnostics": rows,
        "commits": sum(outcome.committed_candidate_id is not None for outcome in result.team_outcomes),
        "ledger": ledger_summary(run_root / "ledger.jsonl"),
        "validation50_calls": 0,
        "test50_calls": 0,
    }
