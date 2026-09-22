"""Preregistered Layer-1-only GEPA local acceptance-rate pilot."""

from __future__ import annotations

import argparse
import asyncio
import csv
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_level_b_gepa_real_canary_v2 as canary_v2  # noqa: E402
from multi_dataset_diverse_rl.evaluation.output_contract import (  # noqa: E402
    SOLVER_OUTPUT_CONTRACT_VERSION,
)
from multi_dataset_diverse_rl.governance.authorization import (  # noqa: E402
    require_api_authorization,
)
from multi_dataset_diverse_rl.governance.manifest import (  # noqa: E402
    preregistration_hash,
    validate_manifest,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import (  # noqa: E402
    GEPAOptimizerConfig,
    GEPALocalPromptOptimizer,
    verify_frozen_gepa_engine_contract,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import (  # noqa: E402
    import_frozen_gepa,
)
from multi_dataset_diverse_rl.team_search.execution_runtime import (  # noqa: E402
    ContextualLocalPromptOptimizer,
    DurableLedger,
    ReflectionLM,
    execution_context_from_system,
    ledger_summary as _ledger_summary,
    profile_identity as _profile_identity,
    read_csv_rows as _rows,
)
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (  # noqa: E402
    PrimaryResponsibilityPersistentRealizabilityScheduler,
)
from multi_dataset_diverse_rl.team_search.schemas import TeamSearchRequest  # noqa: E402
from multi_dataset_diverse_rl.team_search.system_runtime import (  # noqa: E402
    SystemLocalSolverEvaluator,
    SystemResponsibilityAssignmentFactory,
    freeze_current_responsibility,
)
from multi_dataset_diverse_rl.team_search.task_builder import LocalTaskBuilder  # noqa: E402
from infrastructure.common_solver_contract_v1.contract import (  # noqa: E402
    COMMON_SOLVER_CONTRACT_ID,
    CONTRACT_SPEC,
)


base = canary_v2.base

EXPERIMENT_ID = "level_b_gepa_local_acceptance_rate_pilot_v1"
ATTEMPT_ID = "level_b_gepa_local_acceptance_rate_pilot_v1_authorized1"
SEED = 79
PARENT_TASKS = 3
PROPOSALS_PER_PARENT = 4
LOCAL_METRIC_CALL_CEILING = 120
LOCAL_VALIDATION_SIZE = 12
AUTH_ENV = "LEVEL_B_GEPA_LOCAL_ACCEPTANCE_RATE_AUTHORIZED"
MANIFEST = ROOT / "experiments/manifests/level_b_gepa_local_acceptance_rate_pilot_v1.yaml"
PROTOCOL = ROOT / "experiments/level_b_gepa_local_acceptance_rate_pilot_v1/PROTOCOL.md"
DEFAULT_PREP = ROOT / "runs/level_b_gepa_local_acceptance_rate_pilot_v1_prep_authorized1"
DEFAULT_RUN = ROOT / "runs/level_b_gepa_local_acceptance_rate_pilot_v1_authorized1"
DEFAULT_REPORT = ROOT / "reports/level_b_gepa_local_acceptance_rate_pilot_v1_authorized1"
CLASSIFIER_VERSION = "level_b_local_acceptance_rate_classifier_v1"


class ExactProposalCountStopper:
    """Stop only after the callback observed the frozen number of proposals."""

    def __init__(self, callback: Any, proposal_count: int) -> None:
        self.callback = callback
        self.proposal_count = proposal_count

    def __call__(self, _state: Any) -> bool:
        return int(self.callback.proposal_count) >= self.proposal_count


def fixed_proposal_optimize(**kwargs: Any) -> Any:
    callbacks = kwargs.get("callbacks")
    if not isinstance(callbacks, list) or len(callbacks) != 1:
        raise RuntimeError("acceptance-rate pilot requires one authoritative GEPA callback")
    kwargs["stop_callbacks"] = ExactProposalCountStopper(
        callbacks[0], PROPOSALS_PER_PARENT
    )
    return import_frozen_gepa().optimize(**kwargs)


def _logical_hash(values: list[str]) -> str:
    payload = json.dumps(sorted(values), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_document() -> dict[str, Any]:
    config = GEPAOptimizerConfig()
    return {
        "schema_version": "level_b_local_acceptance_rate_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "evidence_type": "prospective_layer1_mechanism_pilot",
        "seed": SEED,
        "data_identity": "anti_overfitting_split_v1_fold_a_plus_b_optimize100",
        "parent_task_definition": (
            "top_three_distinct_update0_scheduler_members_from_one_frozen_P0_team"
        ),
        "parent_task_count": PARENT_TASKS,
        "proposal_attempts_per_parent": PROPOSALS_PER_PARENT,
        "total_proposal_attempts": PARENT_TASKS * PROPOSALS_PER_PARENT,
        "local_metric_call_ceiling_per_parent": LOCAL_METRIC_CALL_CEILING,
        "local_validation_size": LOCAL_VALIDATION_SIZE,
        "reflection_minibatch_size": config.reflection_minibatch_size,
        "proposal_stopper": "callback_observed_exact_proposal_count_v1",
        "official_gepa": asdict(config),
        "strict_improvement": config.engine_acceptance_semantics,
        "layer1_only": True,
        "team_minibatch_calls": 0,
        "full_team_calls": 0,
        "shadow50_calls": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
        "team_commits": 0,
        "no_resume": True,
        "experiment_retry_count": 0,
        "transport_attempt_cap": CONTRACT_SPEC.transport_attempt_cap,
        "classifier_version": CLASSIFIER_VERSION,
        "primary_metrics": [
            "proposal_level_delta_local",
            "strict_improvement_acceptance_rate",
            "parent_with_at_least_one_accepted_rate",
        ],
        "diagnostics": [
            "newly_fixed", "newly_broken", "preservation_loss",
            "proposal_edit_similarity", "reflection_failure_pattern_concentration",
        ],
        "failure_rule": (
            "HOLD if parent_task_count or exact proposal denominator is not realized"
        ),
    }


def source_paths() -> list[Path]:
    return sorted(
        set(canary_v2.source_paths())
        | {
            Path("scripts/run_level_b_gepa_local_acceptance_rate_pilot.py"),
            MANIFEST.relative_to(ROOT),
            PROTOCOL.relative_to(ROOT),
        },
        key=lambda path: path.as_posix(),
    )


_BASE_OVERRIDES = {
    "EXPERIMENT_ID": EXPERIMENT_ID,
    "ATTEMPT_ID": ATTEMPT_ID,
    "SEED": SEED,
    "ARM": "LEVEL_B_LOCAL_ACCEPTANCE_RATE",
    "LOCAL_METRIC_BUDGET": LOCAL_METRIC_CALL_CEILING,
    "AUTH_ENV": AUTH_ENV,
    "MANIFEST": MANIFEST,
    "PROTOCOL": PROTOCOL,
    "DEFAULT_PREP": DEFAULT_PREP,
    "DEFAULT_RUN": DEFAULT_RUN,
    "DEFAULT_REPORT": DEFAULT_REPORT,
    "CLASSIFIER_VERSION": CLASSIFIER_VERSION,
    "protocol_document": protocol_document,
    "source_paths": source_paths,
}


@contextmanager
def configured_base():
    overrides = {**_BASE_OVERRIDES, "preflight": preflight}
    prior = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in prior.items():
            setattr(base, name, value)


def prepare(prep: Path) -> dict[str, Any]:
    with configured_base():
        return canary_v2.prepare(prep)


def verify_freeze(prep: Path) -> None:
    with configured_base():
        canary_v2.verify_freeze(prep)


def authorize(run_root: Path) -> None:
    if os.environ.get(AUTH_ENV) != "1":
        raise PermissionError(f"{AUTH_ENV}=1 is required")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PREFLIGHT_PASS":
        raise RuntimeError("tracked manifest must remain at PREFLIGHT_PASS")
    lifecycle = base.read_json(run_root / base.RUN_LIFECYCLE_FILE)
    if lifecycle.get("status") != "RUNNING":
        raise RuntimeError("run-local lifecycle must be RUNNING before authorization")
    runtime_manifest = json.loads(json.dumps(manifest))
    runtime_manifest["status"] = "RUNNING"
    runtime_manifest["lifecycle_history"].append(
        {"status": "RUNNING", "timestamp": lifecycle["events"][-1]["timestamp"]}
    )
    for role in ("solver", "reflection"):
        require_api_authorization(
            runtime_manifest, phase="local_acceptance", role=role
        )


def _metric_calls(payload: dict[str, Any]) -> int:
    events = payload.get("callback_events", [])
    rows = [row for row in events if row.get("event_type") == "optimization_end"]
    if len(rows) != 1:
        raise RuntimeError("missing unique GEPA optimization_end accounting event")
    return int(rows[0]["total_metric_calls"])


async def execute(prep: Path, run_root: Path) -> dict[str, Any]:
    manager = configured_base()
    manager.__enter__()
    try:
        verify_freeze(prep)
        base.start_run_attempt(prep, run_root)
        try:
            authorize(run_root)
            optimize_rows = _rows(prep / "splits_private/optimize100.csv")
            validation_rows = _rows(prep / "splits_private/validation.csv")
            ledger = DurableLedger(run_root / "ledger.jsonl")
            base.transition_run_attempt(run_root, provider_boundary_reached=True)
            system = await base.initialize_system(
                root=run_root / "system",
                optimize_rows=optimize_rows,
                validation_rows=validation_rows,
                optimize_path=prep / "splits_private/optimize100.csv",
                validation_path=prep / "splits_private/validation.csv",
                ledger=ledger,
            )
            frozen_parent = _profile_identity(system)
            snapshot = freeze_current_responsibility(system, update_index=0)
            scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler()
            decision = scheduler.select(
                assigned=snapshot.assigned,
                current_margin_by_question=snapshot.current_margin_by_question,
                seed=SEED,
                update_index=0,
                target_count=PARENT_TASKS,
            )
            if len(decision.selected_member_ids) != PARENT_TASKS:
                raise RuntimeError("frozen pilot requires exactly three actionable parent-tasks")
            summary_by_member = {row.member_id: row for row in decision.summaries}
            task_builder = LocalTaskBuilder()
            factory = SystemResponsibilityAssignmentFactory(
                system=system,
                snapshot_reader=lambda: snapshot,
                task_builder=task_builder,
            )
            loop = asyncio.get_running_loop()
            parent_rows: list[dict[str, Any]] = []
            proposal_rows: list[dict[str, Any]] = []
            for rank, target in enumerate(decision.selected_member_ids, start=1):
                request = TeamSearchRequest(
                    seed=SEED,
                    update_index=0,
                    team_state_hash=system.team_prompt_state_hash(),
                    local_metric_budget=LOCAL_METRIC_CALL_CEILING,
                    solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
                    output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
                )
                assignment = factory.build_from_member(
                    request=request,
                    member_id=target,
                    primary_lane=summary_by_member[target].primary_lane,
                    responsibility_identity=f"local_acceptance_parent_rank_{rank}",
                )
                task = task_builder.build(request, assignment)
                if len(task.local_validation_examples) != LOCAL_VALIDATION_SIZE:
                    raise RuntimeError("local validation size drift")
                local_solver = SystemLocalSolverEvaluator(
                    system=system,
                    loop=loop,
                    stage=system.set_stage,
                    accounting=system.common.accounting,
                    solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
                    output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
                )
                optimizer = GEPALocalPromptOptimizer(
                    evaluator=local_solver,
                    reflection_lm=ReflectionLM(system),
                    accounting_reader=system.optimizer_accounting,
                    run_root=run_root / "local_gepa" / f"parent_{rank}",
                    optimize_fn=fixed_proposal_optimize,
                )
                result = await ContextualLocalPromptOptimizer(
                    optimizer,
                    local_solver,
                    execution_context_from_system(
                        system,
                        local_no_update_patience=3,
                        team_no_update_patience=2,
                        saturation_mode="layer1_acceptance_rate_pilot",
                    ),
                ).optimize(task)
                if _profile_identity(system) != frozen_parent:
                    raise RuntimeError("Layer-1 pilot mutated the frozen P0 team")
                if result.optimizer_state is None:
                    raise RuntimeError("GEPA result lacks optimizer state")
                payload = dict(result.optimizer_state.payload)
                telemetry = dict(payload.get("telemetry", {}))
                diagnostics = dict(telemetry.get("proposer_diagnostics", {}))
                outcomes = list(diagnostics.get("proposal_outcomes", []))
                if int(diagnostics.get("proposal_attempts", -1)) != PROPOSALS_PER_PARENT:
                    raise RuntimeError("exact proposal denominator was not realized")
                if len(outcomes) != PROPOSALS_PER_PARENT:
                    raise RuntimeError("proposal outcome telemetry denominator mismatch")
                accepted = sum(bool(row["official_strict_improvement"]) for row in outcomes)
                parent_id = f"seed{SEED}_rank{rank}_member{target}"
                for row in outcomes:
                    proposal_rows.append({"parent_id": parent_id, **row})
                parent_rows.append({
                    "parent_id": parent_id,
                    "target_rank": rank,
                    "target_member": int(target),
                    "primary_lane": assignment.primary_responsibility_lane,
                    "proposal_attempts": PROPOSALS_PER_PARENT,
                    "evaluable_proposals": sum(bool(row["evaluable"]) for row in outcomes),
                    "contract_valid_proposals": sum(not bool(row["contract_invalid"]) for row in outcomes),
                    "solver_reached": int(diagnostics.get("solver_reached", 0)),
                    "accepted_mutations": accepted,
                    "at_least_one_accepted": accepted > 0,
                    "metric_calls": _metric_calls(payload),
                    "solver_calls": int(result.solver_calls),
                    "optimizer_calls": int(result.optimizer_calls),
                    "mean_pairwise_proposal_token_jaccard": diagnostics.get(
                        "mean_pairwise_proposal_token_jaccard"
                    ),
                    "reflection_failure_pattern_concentration": diagnostics.get(
                        "reflection_failure_pattern_concentration"
                    ),
                    "reflection_pattern_counts": diagnostics.get("reflection_pattern_counts", {}),
                })
                base.write_json(run_root / f"parent_{rank}_summary.json", parent_rows[-1])
            total_accepted = sum(int(row["accepted_mutations"]) for row in parent_rows)
            total_evaluable = sum(int(row["evaluable_proposals"]) for row in parent_rows)
            summary = {
                "execution_gate": "PASS",
                "experiment_id": EXPERIMENT_ID,
                "attempt_id": ATTEMPT_ID,
                "seed": SEED,
                "parent_team_hash": frozen_parent["team_hash"],
                "parent_task_count": len(parent_rows),
                "proposal_attempts_per_parent": PROPOSALS_PER_PARENT,
                "total_proposal_attempts": len(proposal_rows),
                "evaluable_proposals": total_evaluable,
                "accepted_mutations": total_accepted,
                "strict_improvement_acceptance_rate": (
                    total_accepted / total_evaluable if total_evaluable else None
                ),
                "parents_with_at_least_one_accepted": sum(
                    bool(row["at_least_one_accepted"]) for row in parent_rows
                ),
                "parent_acceptance_rate": sum(
                    bool(row["at_least_one_accepted"]) for row in parent_rows
                ) / PARENT_TASKS,
                "newly_fixed": sum(int(row["newly_fixed"] or 0) for row in proposal_rows),
                "newly_broken": sum(int(row["newly_broken"] or 0) for row in proposal_rows),
                "preservation_loss": sum(int(row["preservation_loss"] or 0) for row in proposal_rows),
                "classifier": (
                    "LOCAL_STRICT_IMPROVEMENT_OBSERVED"
                    if total_accepted else "NO_LOCAL_STRICT_IMPROVEMENT_OBSERVED_IN_PILOT"
                ),
                "classifier_version": CLASSIFIER_VERSION,
                "parents": parent_rows,
                "proposals": proposal_rows,
                "ledger": _ledger_summary(run_root / "ledger.jsonl"),
                "team_minibatch_calls": 0,
                "full_team_calls": 0,
                "shadow50_calls": 0,
                "validation50_calls": 0,
                "test50_calls": 0,
                "team_commits": 0,
            }
            base.write_json(run_root / "execution_summary.json", summary)
            base.transition_run_attempt(
                run_root,
                status="COMPLETE",
                provider_calls_observed=base._provider_calls_observed(run_root),
            )
            return summary
        except BaseException as exc:
            calls = base._provider_calls_observed(run_root)
            base.transition_run_attempt(
                run_root,
                status="FAILED_START" if calls == 0 else "ABORTED",
                provider_calls_observed=calls,
                failure_category=type(exc).__name__,
            )
            raise
    finally:
        manager.__exit__(*sys.exc_info())


def audit(prep: Path, run_root: Path) -> dict[str, Any]:
    verify_freeze(prep)
    summary = base.read_json(run_root / "execution_summary.json")
    lifecycle = base.read_json(run_root / base.RUN_LIFECYCLE_FILE)
    errors: list[str] = []
    expected_attempts = PARENT_TASKS * PROPOSALS_PER_PARENT
    if lifecycle.get("status") != "COMPLETE":
        errors.append("lifecycle")
    if summary.get("parent_task_count") != PARENT_TASKS:
        errors.append("parent_task_count")
    if summary.get("total_proposal_attempts") != expected_attempts:
        errors.append("proposal_attempt_denominator")
    if len(summary.get("proposals", [])) != expected_attempts:
        errors.append("proposal_rows")
    if any(row.get("proposal_attempts") != PROPOSALS_PER_PARENT for row in summary.get("parents", [])):
        errors.append("per_parent_proposal_denominator")
    if any(int(row.get("metric_calls", 0)) > LOCAL_METRIC_CALL_CEILING for row in summary.get("parents", [])):
        errors.append("metric_call_ceiling")
    for key in (
        "team_minibatch_calls", "full_team_calls", "shadow50_calls",
        "validation50_calls", "test50_calls", "team_commits",
    ):
        if summary.get(key) != 0:
            errors.append(key)
    ledger = summary.get("ledger", {})
    if ledger.get("input_tokens", 0) + ledger.get("output_tokens", 0) != ledger.get("total_tokens", -1):
        errors.append("ledger_token_arithmetic")
    if any(_profile_key in json.dumps(summary) for _profile_key in ("FINAL_ANSWER:", "api_key", "base_url")):
        errors.append("sanitization")
    result = {
        "gate": "PASS" if not errors else "HOLD",
        "errors": errors,
        "parent_task_count": summary.get("parent_task_count"),
        "total_proposal_attempts": summary.get("total_proposal_attempts"),
        "team_minibatch_calls": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    }
    base.write_json(run_root / "audit.json", result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(prep: Path, run_root: Path, report: Path) -> dict[str, Any]:
    gate = audit(prep, run_root)
    if gate["gate"] != "PASS":
        raise RuntimeError("local acceptance-rate audit HOLD")
    if report.exists():
        raise FileExistsError("fresh report root required")
    report.mkdir(parents=True)
    summary = base.read_json(run_root / "execution_summary.json")
    public = dict(summary)
    public["proposals"] = [
        {key: value for key, value in row.items() if key != "proposal_hash"}
        for row in summary["proposals"]
    ]
    base.write_json(report / "summary.json", public)
    base.write_json(report / "audit.json", gate)
    base.write_json(report / "protocol_freeze.json", base.read_json(prep / "protocol_freeze.json"))
    _write_csv(
        report / "proposal_level_effects.csv",
        public["proposals"],
        [
            "parent_id", "iteration", "evaluable", "official_strict_improvement",
            "before_correct", "after_correct", "delta_local", "newly_fixed",
            "newly_broken", "preservation_loss", "parent_token_jaccard",
            "added_token_count", "removed_token_count", "contract_invalid", "duplicate",
        ],
    )
    _write_csv(
        report / "parent_level_acceptance.csv",
        public["parents"],
        [
            "parent_id", "target_rank", "target_member", "primary_lane",
            "proposal_attempts", "evaluable_proposals", "contract_valid_proposals",
            "solver_reached", "accepted_mutations", "at_least_one_accepted",
            "metric_calls", "solver_calls", "optimizer_calls",
            "mean_pairwise_proposal_token_jaccard",
            "reflection_failure_pattern_concentration",
        ],
    )
    base.write_json(report / "provenance.json", {
        "evidence_type": "prospective_layer1_mechanism_pilot",
        "execution_commit": base.read_json(prep / "source_freeze.json")["execution_commit"],
        "raw_prompts_published": False,
        "raw_responses_published": False,
        "team_minibatch_calls": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    })
    base.write_json(report / "fact_assertions.json", {
        "gate_pass": True,
        "three_parent_tasks": summary["parent_task_count"] == PARENT_TASKS,
        "four_proposals_each": all(
            row["proposal_attempts"] == PROPOSALS_PER_PARENT for row in summary["parents"]
        ),
        "twelve_total_proposals": summary["total_proposal_attempts"] == 12,
        "layer1_only": all(summary[key] == 0 for key in (
            "team_minibatch_calls", "full_team_calls", "shadow50_calls",
            "validation50_calls", "test50_calls", "team_commits",
        )),
    })
    (report / "README.md").write_text(
        "# Level-B GEPA local acceptance-rate pilot\n\n"
        f"Gate: **{gate['gate']}**  \n"
        f"Classifier: **{summary['classifier']}**  \n"
        f"Strict-improvement acceptance: **{summary['accepted_mutations']}/"
        f"{summary['evaluable_proposals']}**  \n"
        f"Parents with at least one accepted mutation: **"
        f"{summary['parents_with_at_least_one_accepted']}/{PARENT_TASKS}**\n\n"
        "This independent prospective pilot estimates official GEPA Layer-1 local "
        "acceptance on three frozen responsibility-conditioned parent-tasks with "
        "four actual proposal attempts per parent. `delta_local`, newly-fixed, and "
        "newly-broken are defined on each official three-example reflection minibatch; "
        "they are not full-team or held-out outcomes. TeamMiniBatch, Full team, Shadow50, "
        "Validation50, Test50, and write-back were never invoked.\n",
        encoding="utf-8",
    )
    return public


def preflight() -> dict[str, Any]:
    verify_frozen_gepa_engine_contract()
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads(
        (ROOT / "infrastructure/experiment_manifest.schema.json").read_text(encoding="utf-8")
    )
    protocol = protocol_document()
    checks = {
        "three_parent_tasks": protocol["parent_task_count"] == 3,
        "four_actual_proposals_each": protocol["proposal_attempts_per_parent"] == 4,
        "fixed_total_denominator": protocol["total_proposal_attempts"] == 12,
        "fixed_metric_ceiling": protocol["local_metric_call_ceiling_per_parent"] == 120,
        "official_gepa_level_b": protocol["official_gepa"]["optimizer_fidelity_level"] == "LEVEL_B_API_COMPATIBLE_ADAPTATION",
        "layer1_only": protocol["layer1_only"],
        "downstream_zero": all(protocol[key] == 0 for key in (
            "team_minibatch_calls", "full_team_calls", "shadow50_calls",
            "validation50_calls", "test50_calls", "team_commits",
        )),
        "manifest_preflight_pass": manifest.get("status") == "PREFLIGHT_PASS",
        "manifest_schema": not validate_manifest(manifest, schema),
        "attempt_identity": protocol["attempt_id"] == ATTEMPT_ID,
        "authorization": manifest.get("api_authorization", {}).get("authorized") is True,
        "authorization_scope": manifest.get("api_authorization", {}).get("authorization_scope") == "explicit_user_authorization_2026_09_15_layer1_acceptance_rate_single_fresh_attempt",
        "preregistration_hash": manifest.get("artifacts", {}).get("preregistration", {}).get("sha256") == preregistration_hash(manifest),
        "no_resume_retry": protocol["no_resume"] and protocol["experiment_retry_count"] == 0,
    }
    return {
        "gate": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.prep)
    elif args.preflight:
        result = preflight()
    elif args.execute:
        result = asyncio.run(execute(args.prep, args.run))
    elif args.audit:
        result = audit(args.prep, args.run)
    else:
        result = analyze(args.prep, args.run, args.report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
