"""Governed GEPA native versus V4 Layer-2 saturation composition.

This module selects production components; it owns no GEPA search, Layer-2
responsibility, admission, or stopping rules. Real execution requires a
separately authorized, admitted formal V3 permit.
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
    ExperimentEarlyStop, ExperimentInputs, ExperimentServices, Layer2Opportunity,
    OptimizationScope, OptimizerBackend, RuntimeContext, StoppingRegime,
    experiment_spec_from_mapping, run_experiment,
)
from .formal_final_team import persist_final_team
from .governance.freeze_hash import source_freeze_sha256
from .governance.production_execution import ValidatedExecutionContext, mark_provider_client_constructed
from .local_optimizers.gepa_native import (
    GEPALayer2EvidenceOptimizer, GEPANativeDataBuilder, GEPANativeFeedOptimizer,
    GEPANativeSplitConfig,
)
from .local_optimizers.gepa_optimizer import GEPALocalPromptOptimizer
from .local_optimizers.production_backends import GEPABackend
from .local_optimizers.schemas import LocalEvidenceExample
from .native_feed import NativeOptimizationRequest, NativeResourceBudget, ResponsibilityContext
from .persistence.identity import build_run_identity
from .production_canary import ContextBoundGEPALayer2, _UnusedNative
from .saturation import SaturationConfig
from .team_search.candidate_selector import CommonSafeTeamCandidateSelector
from .team_search.controller import TeamSearchController
from .team_search.execution_runtime import (
    CappedDurableLedger, CommonContractExecutionSystem, LOCAL_OPTIMIZER_INVOCATION,
    ReflectionLM, ledger_summary, read_csv_rows,
)
from .team_search.primary_responsibility_scheduler import PrimaryResponsibilityPersistentRealizabilityScheduler
from .team_search.schemas import TeamSearchAssignment, TeamSearchRequest
from .team_search.system_runtime import (
    LatestTransitionStore, SystemLocalSolverEvaluator,
    SystemResponsibilityAssignmentFactory, SystemTeamCandidateEvaluator,
    SystemTeamCommitter, freeze_current_responsibility,
)
from .team_search.task_builder import Layer2EvidenceRequestBuilder
from .team_search.v4_opportunity import V4NoFeasibleOpportunity, select_v4_feasible_opportunity
from .versions import (
    LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
    LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION,
    PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION,
)


class _UnusedLayer2:
    async def optimize_layer2(self, request: Any) -> Any:
        del request
        raise RuntimeError("Layer-2 backend is outside native formal arm")


class _ContextBoundGEPANative:
    """Give the existing native backend the same durable stage attribution."""

    def __init__(self, inner, local_solver, runtime, system):
        self.inner, self.local_solver = inner, local_solver
        self.runtime, self.system = runtime, system

    async def optimize_native(self, request):
        context = {
            "loop": asyncio.get_running_loop(),
            "run_seed": self.runtime.seed, "update_index": 0,
            "target_member": request.target_member,
            "phase": "local_optimizer_solver_eval",
            "parent_id": f"seed{self.runtime.seed}_update0",
            "parent_prompt_sha256": hashlib.sha256(
                request.parent_decision_procedure.encode("utf-8")
            ).hexdigest(),
            "provider_profile": self.runtime.provider_profile,
            "solver_model": self.runtime.solver_model,
            "optimizer_model": self.runtime.optimizer_model,
            "evaluator_model": self.runtime.evaluator_model,
            "run_identity_sha256": self.runtime.run_identity_sha256,
            "local_no_update_patience": 3, "team_no_update_patience": 2,
            "saturation_mode": "formal_gepa_native_v3", "arm": self.system.arm,
        }
        token = LOCAL_OPTIMIZER_INVOCATION.set(context)
        self.local_solver.task_context = context
        try:
            return await self.inner.optimize_native(request)
        finally:
            self.local_solver.task_context = None
            LOCAL_OPTIMIZER_INVOCATION.reset(token)


def _native_examples(system: CommonContractExecutionSystem) -> tuple[LocalEvidenceExample, ...]:
    if system.fixed_probe is None or len(system.fixed_probe.examples) != 100:
        raise RuntimeError("formal native requires frozen Optimize100 initialization")
    rows = []
    for index, row in enumerate(system.fixed_probe.examples):
        parent = system.active_profiles[0][index]
        rows.append(LocalEvidenceExample(
            example_id=row.question_hash, input_payload=row.question,
            gold=row.gold_answer,
            parent_output=parent.answer if parent.valid else None,
            textual_feedback="Improve the general decision procedure without copying examples.",
        ))
    return tuple(rows)


async def execute_formal_gepa_saturation(
    permit: ValidatedExecutionContext, *, root: Path,
) -> dict[str, Any]:
    """Execute one separately authorized Seed×arm logical cell, never a campaign."""

    if not permit.admitted or permit.run_root is None or permit.allowed_phase != "formal":
        raise PermissionError("formal execution requires an admitted formal permit")
    prep, run_root = permit.prep_root, permit.run_root
    manifest = json.loads((prep / "manifest.json").read_text(encoding="utf-8"))
    spec = experiment_spec_from_mapping(manifest["scientific"])
    if (not permit.experiment_id.startswith("gepa_saturation_comparison_v3_")
            or spec.backend is not OptimizerBackend.GEPA
            or spec.stopping_regime is not StoppingRegime.SATURATION
            or (spec.optimization_scope is OptimizationScope.LAYER2
                and spec.layer2_protocol_version != LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION)
            or (spec.optimization_scope is OptimizationScope.NATIVE
                and spec.layer2_protocol_version is not None)):
        raise ValueError("formal V3 method identity mismatch")
    runtime = RuntimeContext(
        seed=int(manifest["runtime"]["seed"]), provider_profile=permit.provider_profile,
        solver_model=str(manifest["runtime"]["solver_model"]),
        optimizer_model=str(manifest["runtime"]["optimizer_model"]),
        evaluator_model=str(manifest["runtime"]["evaluator_model"]),
        run_identity_sha256=permit.run_identity_sha256,
        authorization_identity=permit.attempt_id,
        cache_identity=permit.run_identity_sha256,
        ledger_identity=permit.run_identity_sha256,
    )
    if runtime.seed not in {80, 81, 82}:
        raise ValueError("formal seed identity mismatch")
    optimize_path = prep / "splits_private/optimize100.csv"
    shadow_path = prep / "splits_private/shadow50.csv"
    optimize_rows, shadow_rows = read_csv_rows(optimize_path), read_csv_rows(shadow_path)
    if len(optimize_rows) != 100 or len(shadow_rows) != 50:
        raise ValueError("formal split cardinality mismatch")
    cfg = Config.from_flat(
        task_type="bbh", dataset_format="mars", comparison_task_id="disambiguation_qa",
        benchmark="BBH", answer_format="option_letter",
        train_path=str(optimize_path), val_path=str(shadow_path),
        test_path="VALIDATION50_AND_TEST50_BLOCKED",
        manifest_sha256=source_freeze_sha256(prep / "manifest.json"),
        train_size=100, val_size=50, test_size=0,
        provider_profile=permit.provider_profile,
        agent_model=runtime.solver_model, optimizer_model=runtime.optimizer_model,
        evaluator_model=runtime.evaluator_model, temperature=0.0,
        solver_max_tokens=1800, solver_invalid_max_retries=0,
        solver_contract_id=spec.solver_contract_id,
        experiment_setting="experimental_diversity_d2_rr_generic",
        target_scheduler="round_robin", agents=5, epochs=1, update_every=1,
        seed=runtime.seed, proposal_memory_mode="off", num_candidates_per_parent=2,
        candidate_eval_pool_size=100,
        eval_solver_call_concurrency=int(manifest["runtime"].get("eval_solver_call_concurrency", 8)),
        stage_b_candidate_budget=2, out_dir=str(run_root / "system"),
        shared_solver_cache_path="",
        provider_call_budget=spec.emergency_max_provider_calls,
        total_token_budget=100_000_000, final_test_enabled=False,
        preserve_final_checkpoint=True,
    )
    ledger = CappedDurableLedger(
        run_root / "ledger.jsonl",
        successful_ceiling=spec.emergency_max_provider_calls,
        attempt_ceiling=spec.emergency_max_provider_calls,
    )
    arm = "GEPA_NATIVE" if spec.optimization_scope is OptimizationScope.NATIVE else "GEPA_LAYER2_V4"
    system = CommonContractExecutionSystem(cfg, arm=arm, ledger=ledger, raw_cache={})
    mark_provider_client_constructed(permit)
    system.set_run_identity(build_run_identity(
        cfg, train_rows=optimize_rows, val_rows=shadow_rows,
        test_rows=[], workspace=root,
    ))
    system.set_stage({
        "phase": "initialization", "update_index": -1,
        "target_member": -1, "candidate_id": "P0",
    })
    initialization_parent_tasks = asyncio.all_tasks()
    try:
        await system.initialize_fixed_probe(optimize_rows)
    except BaseException:
        # Nested gather raises on its first failed row while sibling Solver
        # requests can still be in flight. Stop new work, then let each
        # cancelled transport persist its reserved physical-attempt record
        # before the formal attempt is classified as aborted.
        spawned = [task for task in asyncio.all_tasks() - initialization_parent_tasks
                   if not task.done()]
        for task in spawned:
            task.cancel()
        if spawned:
            await asyncio.gather(*spawned, return_exceptions=True)
        raise
    finally:
        system.set_stage(None)
    initial_hash = system.team_prompt_state_hash()
    initial_prompts = tuple(agent.current_prompt for agent in system.agents)
    loop = asyncio.get_running_loop()
    local_solver = SystemLocalSolverEvaluator(
        system=system, loop=loop, stage=system.set_stage,
        accounting=system.common.accounting,
        solver_contract_id=spec.solver_contract_id,
        output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
    )
    saturation = SaturationConfig(
        enabled=True, scientific_budget_enabled=False,
        local_no_update_patience=spec.local_no_update_patience,
        team_no_update_patience=spec.team_no_update_patience,
        emergency_max_provider_calls=spec.emergency_max_provider_calls,
        emergency_max_optimizer_steps=spec.emergency_max_optimizer_steps,
        emergency_max_team_epochs=spec.emergency_max_team_epochs,
        emergency_max_wall_seconds=spec.emergency_max_wall_seconds,
    )
    official = GEPALocalPromptOptimizer(
        evaluator=local_solver, reflection_lm=ReflectionLM(system),
        accounting_reader=system.optimizer_accounting,
        durable_usage_reader=lambda: ledger_summary(run_root / "ledger.jsonl"),
        run_root=run_root / "local_gepa",
    )
    if spec.optimization_scope is OptimizationScope.NATIVE:
        examples = _native_examples(system)
        builder = GEPANativeDataBuilder(
            optimize_examples=examples,
            optimize_universe_id=system.fixed_probe.probe_hash,
            config=GEPANativeSplitConfig(75, 25),
        )
        request = NativeOptimizationRequest(
            request_id=f"seed{runtime.seed}_update0_member0_native",
            parent_decision_procedure=system.agents[0].current_prompt,
            target_member=0,
            responsibility=ResponsibilityContext(
                primary_lane="generic",
                responsibility_identity="native_control_no_layer2_responsibility_v1",
                responsibility_value=0.0,
            ),
            team_state_identity=initial_hash,
            optimize_universe_id=system.fixed_probe.probe_hash,
            solver_contract_id=spec.solver_contract_id,
            output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
            seed=runtime.seed * 100_000,
            budget=NativeResourceBudget(1, 36, 36, 4),
            provenance={"source_split": "optimize_only", "responsibility_semantics": "absent_native_control"},
        )
        backend = GEPABackend(
            native=_ContextBoundGEPANative(
                GEPANativeFeedOptimizer(
                    engine=official, data_builder=builder,
                    layer2_overlay_enabled=False, saturation_config=saturation,
                ), local_solver, runtime, system,
            ),
            layer2=_UnusedLayer2(),
        )
        result = await run_experiment(
            spec, runtime, ExperimentInputs(initial_hash, native_problem=request),
            ExperimentServices(
                backend=backend,
                durable_usage_reader=lambda: ledger_summary(run_root / "ledger.jsonl"),
            ),
        )
        candidate = result.local_result.candidates[0] if (
            result.local_result is not None and result.local_result.candidates
        ) else None
        native_prompts = ((candidate.prompt,) * 5 if candidate is not None else initial_prompts)
        final_team = persist_final_team(
            run_root / "final_team_materialization.json", mode="GEPA_NATIVE",
            initial_prompts=initial_prompts, final_prompts=native_prompts,
            candidate_id=candidate.candidate_id if candidate is not None else None,
            initial_team_hash=initial_hash, search_final_identity=result.final_state_hash,
        )
        return {
            "experiment_id": permit.experiment_id, "mode_id": "GEPA_NATIVE",
            "seed": runtime.seed, "initial_team_hash": initial_hash,
            "final_native_candidate_hash": result.final_state_hash,
            "final_team_materialization": final_team,
            "stop_reason": result.stop_reason,
            "events": [event.__dict__ for event in result.events],
            "ledger": ledger_summary(run_root / "ledger.jsonl"),
            "validation50_calls": 0, "test50_calls": 0,
        }

    # The formal treatment reuses the active V4 selector and task builder;
    # its GEPA search core is the same official instance as the native arm.
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler(
        version=PRIMARY_RESPONSIBILITY_FEASIBILITY_VERSION,
    )
    transition_store = LatestTransitionStore()
    task_builder = Layer2EvidenceRequestBuilder(bounded_search_view=True)
    assignment = {"current": None, "parent_hash": None, "decision": None, "update": 0}
    feasibility_trace: list[dict[str, Any]] = []
    evidence_trace: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    transition_trace: list[dict[str, Any]] = []

    class CurrentAssignment:
        def assign(self, request):
            if assignment["current"] is None or assignment["parent_hash"] != request.team_state_hash:
                raise RuntimeError("formal assignment/parent mismatch")
            return assignment["current"]

    backend = GEPABackend(
        native=_UnusedNative(),
        layer2=ContextBoundGEPALayer2(
            GEPALayer2EvidenceOptimizer(engine=official, saturation_config=saturation),
            local_solver, runtime, system,
            update_index_reader=lambda: assignment["update"],
            saturation_mode="formal_gepa_layer2_v4",
        ),
    )
    evaluator = SystemTeamCandidateEvaluator(
        system=system, shadow_probe=system.build_probe(shadow_rows),
        loop=loop, stage=system.set_stage, accounting=system.common.accounting,
        update_index_reader=lambda: assignment["update"],
    )

    def next_opportunity(index: int, parent_hash: str) -> Layer2Opportunity:
        update = index - 1
        assignment["update"] = update
        if system.team_prompt_state_hash() != parent_hash:
            raise RuntimeError("formal parent must equal actual committed team")
        snapshot = freeze_current_responsibility(system, update_index=update)
        request = TeamSearchRequest(
            seed=runtime.seed, update_index=update, team_state_hash=parent_hash,
            local_metric_budget=36, solver_contract_id=spec.solver_contract_id,
            output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
        )
        factory = SystemResponsibilityAssignmentFactory(
            system=system, snapshot_reader=lambda: snapshot,
            task_builder=task_builder, transition_store=transition_store,
        )
        selected = select_v4_feasible_opportunity(
            snapshot=snapshot, scheduler=scheduler, factory=factory,
            task_builder=task_builder, request=request,
        )
        raw_order = sorted(selected.raw_decision.summaries,
                           key=lambda row: (-row.target_score, row.member_id))
        eligible_order = [
            row for row in raw_order
            if not isinstance(selected, V4NoFeasibleOpportunity)
            and row.member_id in selected.eligible_member_ids
        ]
        feasibility_reasons = dict(selected.feasibility_reasons)
        feasibility_trace.append({
            "update_index": update,
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
            } for row in selected.raw_decision.summaries],
            "selected_member": eligible_order[0].member_id if eligible_order else None,
            "eligible_member_ids": [row.member_id for row in eligible_order],
            "latest_transition_effect_hash_by_member": {
                str(member_id): (
                    transition_store.get(member_id).transition_effect_hash
                    if transition_store.get(member_id) is not None else None
                ) for member_id in range(5)
            },
        })
        if isinstance(selected, V4NoFeasibleOpportunity):
            raise ExperimentEarlyStop("NO_FEASIBLE_LAYER2_OPPORTUNITY")
        assignment.update(current=selected.assignment, parent_hash=parent_hash,
                          decision=selected.decision)
        packet = selected.packet
        target = selected.decision.selected_member_ids[0]
        raw_summary = next(row for row in selected.raw_decision.summaries
                           if row.member_id == target)
        evidence_trace.append({
            "update_index": update, "parent_team_hash": parent_hash,
            "target_member": target,
            "raw_V": raw_summary.primary_score,
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
            "responsibility_scheduled_ids": [
                row.example_id for row in packet.responsibility_examples
            ],
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
        })
        return Layer2Opportunity(request, eligible_member_ids=selected.eligible_member_ids)

    def observe(index: int, outcome):
        scheduler.record_outcome(
            decision=assignment["decision"], update_index=index - 1,
            committed_member_id=outcome.audit_metadata.get("committed_member_id"),
            valid_outcome=True,
        )
        local_telemetry = outcome.audit_metadata.get("local_optimizer_telemetry", {})
        evidence_trace[-1]["evidence_delivered"] = {
            "batch_ids": local_telemetry.get("delivered_batch_ids"),
            "role_item_ids": local_telemetry.get("evidence_delivered_role_item_ids"),
            "source_ids": local_telemetry.get("evidence_delivered_source_ids"),
            "scheduled_but_not_delivered_count": local_telemetry.get(
                "scheduled_but_not_delivered_role_item_count"
            ),
        }
        candidates = {
            row.local_candidate.candidate_id: row.local_candidate
            for row in outcome.candidates
        }
        for row in outcome.audit_metadata.get("transfer_diagnostic", ()):
            candidate = candidates[row["candidate_id"]]
            diagnostic = dict(row)
            diagnostic["candidate_hash"] = hashlib.sha256(
                candidate.prompt.encode("utf-8")
            ).hexdigest()
            diagnostic["generation"] = candidate.generation
            diagnostic["shadow_gate"] = next((
                dict(event) for event in evaluator.shadow_events
                if event["update_index"] == index - 1
                and event["candidate_id"] == candidate.candidate_id
            ), None)
            candidate_diagnostics.append(diagnostic)
        parent_hash = evidence_trace[-1]["parent_team_hash"]
        successor_hash = system.team_prompt_state_hash()
        committed_id = outcome.committed_candidate_id
        committed_member = outcome.audit_metadata.get("committed_member_id")
        transition = (
            transition_store.get(committed_member)
            if committed_member is not None else None
        )
        if committed_id is None:
            if successor_hash != parent_hash or committed_member is not None:
                raise RuntimeError("formal state changed without commit")
        elif (successor_hash == parent_hash or transition is None
              or committed_id not in candidates):
            raise RuntimeError("formal commit lacks successor transition")
        committed_hash = (
            hashlib.sha256(candidates[committed_id].prompt.encode("utf-8")).hexdigest()
            if committed_id is not None else None
        )
        if (transition is not None and transition.child_candidate_hash
                != system.prompt_hash(candidates[committed_id].prompt)):
            raise RuntimeError("formal committed candidate transition mismatch")
        transition_trace.append({
            "update_index": index - 1,
            "parent_team_hash": parent_hash,
            "selected_member": outcome.assignment.target_member,
            "committed_candidate_id": committed_id,
            "committed_candidate_hash": committed_hash,
            "committed_member_id": committed_member,
            "successor_team_hash": successor_hash,
            "candidate_transition": (
                dict(transition.sanitized_payload()) if transition is not None else None
            ),
        })
        evidence_trace[-1]["committed_candidate_id"] = committed_id
        evidence_trace[-1]["successor_team_hash"] = successor_hash
        return None

    def controller_factory(bound_backend):
        return TeamSearchController(
            responsibility=CurrentAssignment(), task_builder=task_builder,
            local_optimizer=bound_backend, evaluator=evaluator,
            selector=CommonSafeTeamCandidateSelector(),
            committer=SystemTeamCommitter(
                system=system, evaluator=evaluator,
                update_index_reader=lambda: assignment["update"],
                transition_store=transition_store,
            ),
            diagnostic_full_for_local_accepts=True,
            diagnostic_allow_multi_accepted=True,
        )

    result = await run_experiment(
        spec, runtime, ExperimentInputs(initial_hash),
        ExperimentServices(
            backend=backend, layer2_controller_factory=controller_factory,
            team_state_hash_reader=system.team_prompt_state_hash,
            layer2_opportunity_factory=next_opportunity,
            layer2_outcome_observer=observe,
            durable_usage_reader=lambda: ledger_summary(run_root / "ledger.jsonl"),
        ),
    )
    final_team = persist_final_team(
        run_root / "final_team_materialization.json", mode="GEPA_LAYER2_V4",
        initial_prompts=initial_prompts,
        final_prompts=tuple(agent.current_prompt for agent in system.agents),
        candidate_id=None, initial_team_hash=initial_hash,
        search_final_identity=result.final_state_hash,
    )
    return {
        "experiment_id": permit.experiment_id, "mode_id": "GEPA_LAYER2_V4",
        "seed": runtime.seed, "initial_team_hash": initial_hash,
        "final_team_hash": result.final_state_hash,
        "final_team_materialization": final_team,
        "stop_reason": result.stop_reason,
        "events": [event.__dict__ for event in result.events],
        "feasibility_trace": feasibility_trace,
        "evidence_view_trace": evidence_trace,
        "candidate_diagnostics": candidate_diagnostics,
        "transition_trace": transition_trace,
        "commits": sum(outcome.committed_candidate_id is not None for outcome in result.team_outcomes),
        "ledger": ledger_summary(run_root / "ledger.jsonl"),
        "validation50_calls": 0, "test50_calls": 0,
    }
