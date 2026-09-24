"""Post-refactor GEPA/Layer-2 canary composition; no historical runner imports.

This module creates real clients only from an admitted execution context. It
reuses the existing scientific engine, scheduler, packet builder, official GEPA
backend, and team controller without changing their search rules.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .experiment import (
    ExperimentInputs, ExperimentServices, ExperimentSpec, Layer2Opportunity,
    RuntimeContext, experiment_spec_from_mapping, run_experiment,
)
from .governance.freeze_hash import source_freeze_sha256
from .governance.production_execution import (
    ValidatedExecutionContext, mark_provider_client_constructed,
)
from .local_optimizers.gepa_native import GEPALayer2EvidenceOptimizer
from .local_optimizers.gepa_optimizer import GEPALocalPromptOptimizer
from .local_optimizers.production_backends import GEPABackend
from .native_feed import Layer2OptimizationRequest
from .persistence.identity import build_run_identity
from .team_search.candidate_selector import CommonSafeTeamCandidateSelector
from .team_search.controller import TeamSearchController
from .team_search.execution_runtime import (
    CommonContractExecutionSystem, DurableLedger, LOCAL_OPTIMIZER_INVOCATION,
    ReflectionLM, ledger_summary, read_csv_rows,
)
from .team_search.primary_responsibility_scheduler import (
    PrimaryResponsibilityPersistentRealizabilityScheduler,
)
from .team_search.schemas import TeamSearchAssignment, TeamSearchRequest
from .team_search.system_runtime import (
    SystemLocalSolverEvaluator, SystemResponsibilityAssignmentFactory,
    SystemTeamCandidateEvaluator,
    SystemTeamCommitter, freeze_current_responsibility,
)
from .team_search.task_builder import Layer2EvidenceRequestBuilder
from .evaluation.output_contract import SOLVER_OUTPUT_CONTRACT_VERSION


@dataclass(frozen=True)
class FrozenAssignmentProvider:
    assignment: TeamSearchAssignment
    team_state_hash: str

    def assign(self, request: TeamSearchRequest) -> TeamSearchAssignment:
        if request.team_state_hash != self.team_state_hash:
            raise ValueError("canary assignment parent team changed")
        return self.assignment


class _UnusedNative:
    async def optimize_native(self, request: Any) -> Any:
        del request
        raise RuntimeError("native backend is outside GEPA Layer2 canary scope")


class ContextBoundGEPALayer2:
    """Execution attribution around the unmodified official GEPA search core."""

    def __init__(
        self, inner: GEPALayer2EvidenceOptimizer,
        local_solver: SystemLocalSolverEvaluator,
        runtime: RuntimeContext,
        system: CommonContractExecutionSystem,
        update_index_reader: Callable[[], int] | None = None,
        saturation_mode: str = "single_opportunity_engineering_canary",
    ) -> None:
        self.inner = inner
        self.local_solver = local_solver
        self.runtime = runtime
        self.system = system
        self.update_index_reader = update_index_reader or (lambda: 0)
        self.saturation_mode = saturation_mode

    async def optimize_layer2(self, request: Layer2OptimizationRequest):
        if request.packet.target_member not in range(5):
            raise ValueError("invalid local target")
        update_index = self.update_index_reader()
        if f"_update{update_index}_" not in request.request_id:
            raise ValueError("local optimizer update identity mismatch")
        context = {
            "loop": asyncio.get_running_loop(),
            "run_seed": self.runtime.seed,
            "update_index": update_index,
            "target_member": request.packet.target_member,
            "phase": "local_optimizer_solver_eval",
            "parent_id": f"seed{self.runtime.seed}_update{update_index}",
            "parent_prompt_sha256": hashlib.sha256(
                request.parent_decision_procedure.encode("utf-8")
            ).hexdigest(),
            "provider_profile": self.runtime.provider_profile,
            "solver_model": self.runtime.solver_model,
            "optimizer_model": self.runtime.optimizer_model,
            "evaluator_model": self.runtime.evaluator_model,
            "run_identity_sha256": self.runtime.run_identity_sha256,
            "local_no_update_patience": 3,
            "team_no_update_patience": 2,
            "saturation_mode": self.saturation_mode,
            "arm": self.system.arm,
        }
        token = LOCAL_OPTIMIZER_INVOCATION.set(context)
        self.local_solver.task_context = context
        try:
            return await self.inner.optimize_layer2(request)
        finally:
            self.local_solver.task_context = None
            LOCAL_OPTIMIZER_INVOCATION.reset(token)


async def execute_post_refactor_canary(
    permit: ValidatedExecutionContext, *, root: Path,
) -> dict[str, Any]:
    """Exactly one deterministic target and one frozen local search attempt."""

    if not permit.admitted or permit.run_root is None:
        raise PermissionError("real services require an admitted execution permit")
    prep = permit.prep_root
    run_root = permit.run_root
    manifest = json.loads((prep / "manifest.json").read_text(encoding="utf-8"))
    spec: ExperimentSpec = experiment_spec_from_mapping(manifest["scientific"])
    if spec.mode_id != "GEPA_LAYER2" or spec.fixed_budget_units != 1:
        raise ValueError("canary requires exactly one GEPA Layer2 opportunity")
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
    optimize_path = prep / "splits_private/optimize100.csv"
    shadow_path = prep / "splits_private/shadow50.csv"
    optimize_rows = read_csv_rows(optimize_path)
    shadow_rows = read_csv_rows(shadow_path)
    cfg = Config.from_flat(
        task_type="bbh", dataset_format="mars", comparison_task_id="disambiguation_qa",
        benchmark="BBH", answer_format="option_letter",
        train_path=str(optimize_path), val_path=str(shadow_path),
        test_path="TEST50_BLOCKED", manifest_sha256=source_freeze_sha256(prep / "manifest.json"),
        train_size=100, val_size=50, test_size=0,
        provider_profile=permit.provider_profile,
        agent_model=runtime.solver_model, optimizer_model=runtime.optimizer_model,
        evaluator_model=runtime.evaluator_model,
        temperature=0.0, solver_max_tokens=1800,
        solver_invalid_max_retries=0, solver_contract_id=spec.solver_contract_id,
        experiment_setting="experimental_diversity_d2_rr_generic",
        target_scheduler="round_robin", agents=5, epochs=1, update_every=1,
        seed=runtime.seed, proposal_memory_mode="off", num_candidates_per_parent=2,
        candidate_eval_pool_size=100, eval_solver_call_concurrency=8,
        stage_b_candidate_budget=2, out_dir=str(run_root / "system"),
        shared_solver_cache_path="", provider_call_budget=256,
        total_token_budget=100_000_000, final_test_enabled=False,
        preserve_final_checkpoint=True,
    )
    ledger = DurableLedger(run_root / "ledger.jsonl")
    # This constructor is the first provider-client construction on this path.
    system = CommonContractExecutionSystem(
        cfg, arm="GEPA_LAYER2_POST_REFACTOR_CANARY", ledger=ledger, raw_cache={},
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

    parent_hash = system.team_prompt_state_hash()
    snapshot = freeze_current_responsibility(system, update_index=0)
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler()
    decision = scheduler.select(
        assigned=snapshot.assigned,
        current_margin_by_question=snapshot.current_margin_by_question,
        seed=runtime.seed, update_index=0, target_count=1,
    )
    if len(decision.selected_member_ids) != 1:
        raise RuntimeError("canary requires one deterministic target")
    target = decision.selected_member_ids[0]
    summary = next(row for row in decision.summaries if row.member_id == target)
    task_builder = Layer2EvidenceRequestBuilder()
    assignment_factory = SystemResponsibilityAssignmentFactory(
        system=system, snapshot_reader=lambda: snapshot, task_builder=task_builder,
    )
    request = TeamSearchRequest(
        seed=runtime.seed, update_index=0, team_state_hash=parent_hash,
        local_metric_budget=36, solver_contract_id=spec.solver_contract_id,
        output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
    )
    assignment = assignment_factory.build_from_member(
        request=request, member_id=target,
        primary_lane=summary.primary_lane,
        responsibility_identity="primary_responsibility_persistent_realizability_v1",
    )
    loop = asyncio.get_running_loop()
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
        local_solver, runtime, system,
    )
    backend = GEPABackend(native=_UnusedNative(), layer2=layer2)
    shadow_probe = system.build_probe(shadow_rows)
    evaluator = SystemTeamCandidateEvaluator(
        system=system, shadow_probe=shadow_probe, loop=loop,
        stage=system.set_stage, accounting=system.common.accounting,
        update_index_reader=lambda: 0,
    )

    def controller_factory(bound_backend):
        return TeamSearchController(
            responsibility=FrozenAssignmentProvider(assignment, parent_hash),
            task_builder=task_builder, local_optimizer=bound_backend,
            evaluator=evaluator, selector=CommonSafeTeamCandidateSelector(),
            committer=SystemTeamCommitter(
                system=system, evaluator=evaluator, update_index_reader=lambda: 0,
            ),
        )

    result = await run_experiment(
        spec, runtime,
        ExperimentInputs(
            initial_state_hash=parent_hash,
            layer2_opportunities=(Layer2Opportunity(request),),
        ),
        ExperimentServices(
            backend,
            layer2_controller_factory=controller_factory,
            team_state_hash_reader=system.team_prompt_state_hash,
            technical_local_canary_only=True,
        ),
    )
    outcome = result.team_outcomes[0]
    telemetry = dict(outcome.audit_metadata.get("local_optimizer_telemetry", {}))
    proposer = dict(telemetry.get("proposer_diagnostics", {}))
    return {
        "method_identity": spec.method_identity,
        "spec_identity": spec.identity(),
        "run_identity_sha256": permit.run_identity_sha256,
        "target_member": target,
        "packet_hash": outcome.audit_metadata.get("responsibility_packet_hash"),
        "funnel": dict(outcome.funnel),
        "local_empirical": {
            "proposal_attempts": int(telemetry.get("proposal_attempts", 0)),
            "changed": int(proposer.get("proposal_changed", 0)),
            "contract_invalid": int(proposer.get("proposal_contract_invalid", 0)),
            "solver_reached": int(proposer.get("solver_reached", 0)),
            "positive_minibatch_deltas": int(telemetry.get("positive_minibatch_deltas", 0)),
            "accepted_mutations": int(telemetry.get("accepted_mutations", 0)),
            "full_local_evaluations": int(telemetry.get("full_local_evaluations", 0)),
            "native_example_selection_calls": int(telemetry.get("backend_example_selection_calls", -1)),
        },
        "stop_reason": result.stop_reason,
        "ledger": ledger_summary(run_root / "ledger.jsonl"),
        "validation50_calls": 0,
        "test50_calls": 0,
    }
