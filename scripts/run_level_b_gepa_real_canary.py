"""One-parent real-provider canary for the Level-B GEPA adapter."""

from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from multi_dataset_diverse_rl.evaluation.output_contract import SOLVER_OUTPUT_CONTRACT_VERSION
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.governance.authorization import require_api_authorization
from multi_dataset_diverse_rl.governance.freeze_hash import (
    SOURCE_FREEZE_HASH_SEMANTICS,
    source_freeze_sha256,
)
from multi_dataset_diverse_rl.governance.manifest import (
    preregistration_hash,
    validate_manifest,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import (
    GEPAOptimizerConfig,
    GEPALocalPromptOptimizer,
    local_gepa_budget_capacity,
    verify_frozen_gepa_engine_contract,
)
from multi_dataset_diverse_rl.team_search.candidate_selector import CommonSafeTeamCandidateSelector
from multi_dataset_diverse_rl.team_search.controller import TeamSearchController
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (
    PrimaryResponsibilityPersistentRealizabilityScheduler,
)
from multi_dataset_diverse_rl.team_search.schemas import TeamSearchRequest
from multi_dataset_diverse_rl.team_search.system_runtime import (
    SystemLocalSolverEvaluator,
    SystemResponsibilityAssignmentFactory,
    SystemTeamCandidateEvaluator,
    SystemTeamCommitter,
    freeze_current_responsibility,
)
from multi_dataset_diverse_rl.team_search.task_builder import LocalTaskBuilder
from multi_dataset_diverse_rl.persistence.identity import build_run_identity
from infrastructure.common_solver_contract_v1.contract import (
    COMMON_SOLVER_CONTRACT_ID,
    CONTRACT_SPEC,
)
from scripts.anti_overfitting_shadow_support import (
    construct_assignment,
    export_private_splits,
    metadata,
    sha256_file,
    sha256_json,
    write_json,
)
from scripts.run_seed78_primary_responsibility_ab import (
    ContextualOptimizer,
    DurableLedger,
    ReflectionLM,
    Seed78System,
    _ledger_summary,
    _profile_identity,
    _rows,
)


EXPERIMENT_ID = "level_b_gepa_real_canary_v1"
ATTEMPT_ID = "level_b_gepa_real_canary_v1_authorized3"
SEED = 78
ARM = "LEVEL_B_REAL_CANARY"
LOCAL_METRIC_BUDGET = 36
AUTH_ENV = "LEVEL_B_GEPA_REAL_CANARY_AUTHORIZED"
MANIFEST = ROOT / "experiments/manifests/level_b_gepa_real_canary_v1.yaml"
PROTOCOL = ROOT / "experiments/level_b_gepa_real_canary_v1/PROTOCOL.md"
DEFAULT_PREP = ROOT / "runs/level_b_gepa_real_canary_v1_prep_authorized3"
DEFAULT_RUN = ROOT / "runs/level_b_gepa_real_canary_v1_authorized3"
DEFAULT_REPORT = ROOT / "reports/level_b_gepa_real_canary_v1_authorized3"
RUN_LIFECYCLE_FILE = "run_lifecycle.json"
LAUNCH_TRANSACTION_VERSION = "atomic_run_local_lifecycle_v1"
PROPOSER_DIAGNOSTICS_VERSION = "sanitized_proposer_diagnostics_v1"
CLASSIFIER_VERSION = "level_b_local_empirical_path_classifier_v1"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def source_paths() -> list[Path]:
    paths = [
        path.relative_to(ROOT)
        for path in (ROOT / "multi_dataset_diverse_rl").rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    paths.extend([
        Path("infrastructure/common_solver_contract_v1/contract.py"),
        Path("infrastructure/common_solver_contract_v1/evaluator.py"),
        Path("infrastructure/experiment_manifest.schema.json"),
        Path("scripts/anti_overfitting_shadow_support.py"),
        Path("scripts/run_seed78_primary_responsibility_ab.py"),
        Path("scripts/run_level_b_gepa_real_canary.py"),
        MANIFEST.relative_to(ROOT),
        PROTOCOL.relative_to(ROOT),
        Path("experiments/anti_overfitting_split_v1/split_manifest.json"),
        Path("experiments/anti_overfitting_split_v1/fold_assignment.json"),
    ])
    return sorted(set(paths), key=lambda path: path.as_posix())


def protocol_document() -> dict[str, Any]:
    return {
        "schema_version": "level_b_gepa_real_canary_protocol_v2",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "seed": SEED,
        "evidence_type": "engineering_canary_not_efficacy_evidence",
        "source_parent": "Seed78 fold-a-plus-fold-b Optimize100 initialization",
        "target": "first frozen primary-responsibility target at update zero",
        "opportunities": 1,
        "target_branches": 1,
        "official_gepa": asdict(GEPAOptimizerConfig()),
        "local_metric_budget": LOCAL_METRIC_BUDGET,
        "budget_capacity": asdict(local_gepa_budget_capacity(
            metric_budget=36, validation_size=12, reflection_minibatch_size=3
        )),
        "team_minibatch": "strict_primary_lane_4_coalition_4_preservation_4",
        "common_safe": True,
        "winner_only_shadow": True,
        "validation50_calls": 0,
        "test50_calls": 0,
        "no_resume": True,
        "experiment_retry_count": 0,
        "transport_attempt_cap": CONTRACT_SPEC.transport_attempt_cap,
        "launch_transaction_version": LAUNCH_TRANSACTION_VERSION,
        "proposer_diagnostics_version": PROPOSER_DIAGNOSTICS_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "source_freeze_hash_semantics": SOURCE_FREEZE_HASH_SEMANTICS,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def start_run_attempt(prep: Path, run_root: Path) -> dict[str, Any]:
    """Atomically publish a fresh run root containing its RUNNING fact."""

    if run_root.exists():
        raise FileExistsError("fresh canary run root required; retry/resume forbidden")
    staging = run_root.with_name(f".{run_root.name}.{ATTEMPT_ID}.starting")
    if staging.exists():
        raise FileExistsError("fresh canary launch staging root required")
    freeze = read_json(prep / "source_freeze.json")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    lifecycle = {
        "schema_version": LAUNCH_TRANSACTION_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "status": "RUNNING",
        "source_commit": freeze["execution_commit"],
        "protocol_sha256": freeze["protocol_sha256"],
        "preregistration_sha256": preregistration_hash(manifest),
        "provider_call_boundary_reached": False,
        "provider_calls_observed": 0,
        "events": [{"status": "RUNNING", "timestamp": _utc_now()}],
    }
    run_root.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    _atomic_write_json(staging / RUN_LIFECYCLE_FILE, lifecycle)
    os.replace(staging, run_root)
    return lifecycle


def transition_run_attempt(
    run_root: Path,
    *,
    status: str | None = None,
    provider_boundary_reached: bool | None = None,
    provider_calls_observed: int | None = None,
    failure_category: str | None = None,
) -> dict[str, Any]:
    path = run_root / RUN_LIFECYCLE_FILE
    lifecycle = read_json(path)
    if lifecycle.get("attempt_id") != ATTEMPT_ID:
        raise RuntimeError("run lifecycle attempt identity mismatch")
    if provider_boundary_reached is not None:
        lifecycle["provider_call_boundary_reached"] = bool(provider_boundary_reached)
    if provider_calls_observed is not None:
        lifecycle["provider_calls_observed"] = int(provider_calls_observed)
    if status is not None:
        if status not in {"COMPLETE", "FAILED_START", "ABORTED"}:
            raise ValueError("invalid terminal run lifecycle status")
        lifecycle["status"] = status
        event: dict[str, Any] = {"status": status, "timestamp": _utc_now()}
        if failure_category is not None:
            event["failure_category"] = failure_category
        lifecycle["events"].append(event)
    _atomic_write_json(path, lifecycle)
    return lifecycle


def _provider_calls_observed(run_root: Path) -> int:
    ledger = run_root / "ledger.jsonl"
    if not ledger.is_file():
        return 0
    return int(_ledger_summary(ledger)["provider_attempts"])


def prepare(prep: Path) -> dict[str, Any]:
    if prep.exists():
        raise FileExistsError("fresh canary prep root required")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before canary freeze")
    if preflight()["gate"] != "PASS":
        raise RuntimeError("canary preflight must pass before freeze")
    verify_frozen_gepa_engine_contract()
    items, raw = metadata()
    assignment = construct_assignment(items)
    prep.mkdir(parents=True)
    export_private_splits(raw, assignment, prep / "splits_private")
    optimize = prep / "splits_private/optimize100.csv"
    rows = _rows(prep / "splits_private/fold_a.csv") + _rows(
        prep / "splits_private/fold_b.csv"
    )
    import csv
    with optimize.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(rows)
    protocol = protocol_document()
    write_json(prep / "protocol_freeze.json", protocol)
    write_json(prep / "test_access_registry.json", {
        "events": [], "validation50_calls": 0, "test50_calls": 0,
    })
    freeze = {
        "execution_commit": git("rev-parse", "HEAD"),
        "hash_semantics": SOURCE_FREEZE_HASH_SEMANTICS,
        "protocol_sha256": sha256_json(protocol),
        "preregistration_sha256": preregistration_hash(
            yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
        ),
        "attempt_id": ATTEMPT_ID,
        "files": [
            {"path": path.as_posix(), "sha256": source_freeze_sha256(ROOT / path)}
            for path in source_paths()
        ],
        "private_split_sha256": {
            path.name: source_freeze_sha256(path)
            for path in sorted((prep / "splits_private").glob("*.csv"))
        },
    }
    write_json(prep / "source_freeze.json", freeze)
    result = {
        "gate": "PASS", "execution_commit": freeze["execution_commit"],
        "protocol_sha256": freeze["protocol_sha256"], "api_calls": 0,
        "preregistration_sha256": freeze["preregistration_sha256"],
        "attempt_id": ATTEMPT_ID, "validation_calls": 0, "test_calls": 0,
    }
    write_json(prep / "phase_a_gate.json", result)
    return result


def verify_freeze(prep: Path) -> None:
    freeze = read_json(prep / "source_freeze.json")
    if freeze.get("hash_semantics") != SOURCE_FREEZE_HASH_SEMANTICS:
        raise RuntimeError("canary source-freeze hash semantics mismatch")
    if git("rev-parse", "HEAD") != freeze["execution_commit"]:
        raise RuntimeError("canary execution commit mismatch")
    if sha256_json(read_json(prep / "protocol_freeze.json")) != freeze["protocol_sha256"]:
        raise RuntimeError("canary protocol hash mismatch")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    if preregistration_hash(manifest) != freeze["preregistration_sha256"]:
        raise RuntimeError("canary preregistration hash mismatch")
    if freeze.get("attempt_id") != ATTEMPT_ID:
        raise RuntimeError("canary attempt identity mismatch")
    for row in freeze["files"]:
        if source_freeze_sha256(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"canary source freeze mismatch: {row['path']}")
    for name, digest in freeze["private_split_sha256"].items():
        if source_freeze_sha256(prep / "splits_private" / name) != digest:
            raise RuntimeError(f"canary split freeze mismatch: {name}")


def authorize(run_root: Path) -> None:
    if os.environ.get(AUTH_ENV) != "1":
        raise PermissionError(f"{AUTH_ENV}=1 is required")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PREFLIGHT_PASS":
        raise RuntimeError("tracked canary manifest must remain at PREFLIGHT_PASS")
    lifecycle = read_json(run_root / RUN_LIFECYCLE_FILE)
    if lifecycle.get("status") != "RUNNING":
        raise RuntimeError("run-local lifecycle must be RUNNING before authorization")
    runtime_manifest = copy.deepcopy(manifest)
    runtime_manifest["status"] = "RUNNING"
    runtime_manifest["lifecycle_history"].append(
        {"status": "RUNNING", "timestamp": lifecycle["events"][-1]["timestamp"]}
    )
    for role in ("solver", "reflection"):
        require_api_authorization(
            runtime_manifest, phase="canary", role=role, explicit_user_authorized=True
        )


def classify(telemetry: dict[str, Any], outcome: Any) -> str:
    del outcome
    diagnostics = telemetry.get("proposer_diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    proposal_attempts = int(
        diagnostics.get("proposal_attempts", telemetry.get("proposal_attempts", 0))
    )
    solver_reached = int(diagnostics.get("solver_reached", 0))
    if proposal_attempts == 0:
        return "NO_REAL_PROPOSAL_ATTEMPT"
    if solver_reached == 0:
        return "PROPOSAL_CONTRACT_STILL_BLOCKS_EMPIRICAL_SEARCH"
    return "LOCAL_EMPIRICAL_PATH_CONFIRMED"


def config(out: Path, *, optimize_path: Path, validation_path: Path) -> Config:
    return Config.from_flat(
        task_type="bbh", dataset_format="mars", comparison_task_id="disambiguation_qa",
        benchmark="BBH", answer_format="option_letter",
        train_path=str(optimize_path.resolve()), val_path=str(validation_path.resolve()),
        test_path="TEST50_BLOCKED", manifest_sha256=source_freeze_sha256(MANIFEST),
        train_size=100, val_size=50, test_size=0,
        agent_model="qwen3-8b", optimizer_model="qwen3.7-flash",
        evaluator_model="qwen3.7-flash", temperature=0.0, solver_max_tokens=1800,
        solver_invalid_max_retries=0, solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
        experiment_setting="experimental_diversity_d2_rr_generic",
        target_scheduler="round_robin", agents=5, epochs=1, update_every=1, seed=SEED,
        proposal_memory_mode="off", num_candidates_per_parent=2,
        candidate_eval_pool_size=100, eval_solver_call_concurrency=8,
        stage_b_candidate_budget=2, out_dir=str(out), shared_solver_cache_path="",
        provider_call_budget=100000, total_token_budget=100_000_000,
        final_test_enabled=False, preserve_final_checkpoint=True,
    )


async def initialize_system(
    *, root: Path, optimize_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]], optimize_path: Path,
    validation_path: Path, ledger: DurableLedger,
) -> Seed78System:
    cfg = config(root, optimize_path=optimize_path, validation_path=validation_path)
    system = Seed78System(cfg, arm=ARM, ledger=ledger, raw_cache={})
    system.set_run_identity(build_run_identity(
        cfg, train_rows=optimize_rows, val_rows=validation_rows,
        test_rows=[], workspace=ROOT,
    ))
    system.set_stage({
        "phase": "initialization", "update_index": -1,
        "target_member": -1, "candidate_id": "P0",
    })
    try:
        await system.initialize_fixed_probe(optimize_rows)
    finally:
        system.set_stage(None)
    return system


async def execute(prep: Path, run_root: Path) -> dict[str, Any]:
    if os.environ.get(AUTH_ENV) != "1":
        raise PermissionError(f"{AUTH_ENV}=1 is required")
    verify_freeze(prep)
    start_run_attempt(prep, run_root)
    try:
        authorize(run_root)
        optimize_rows = _rows(prep / "splits_private/optimize100.csv")
        shadow_rows = _rows(prep / "splits_private/fold_c.csv")
        validation_rows = _rows(prep / "splits_private/validation.csv")
        ledger = DurableLedger(run_root / "ledger.jsonl")
        transition_run_attempt(run_root, provider_boundary_reached=True)
        system = await initialize_system(
            root=run_root / "system",
            optimize_rows=optimize_rows,
            validation_rows=validation_rows,
            optimize_path=prep / "splits_private/optimize100.csv",
            validation_path=prep / "splits_private/validation.csv",
            ledger=ledger,
        )
        parent_identity = _profile_identity(system)
        snapshot = freeze_current_responsibility(system, update_index=0)
        scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler()
        decision = scheduler.select(
            assigned=snapshot.assigned,
            current_margin_by_question=snapshot.current_margin_by_question,
            seed=SEED,
            update_index=0,
            target_count=1,
        )
        if len(decision.selected_member_ids) != 1:
            raise RuntimeError("canary must freeze exactly one target")
        target = decision.selected_member_ids[0]
        summary_by_member = {row.member_id: row for row in decision.summaries}
        task_builder = LocalTaskBuilder()
        factory = SystemResponsibilityAssignmentFactory(
            system=system,
            snapshot_reader=lambda: snapshot,
            task_builder=task_builder,
        )
        assignment = factory.build_from_member(
            request=TeamSearchRequest(
                seed=SEED,
                update_index=0,
                team_state_hash=system.team_prompt_state_hash(),
                local_metric_budget=LOCAL_METRIC_BUDGET,
                solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
                output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
            ),
            member_id=target,
            primary_lane=summary_by_member[target].primary_lane,
            responsibility_identity="level_b_real_canary_v1",
        )
        loop = asyncio.get_running_loop()
        local_solver = SystemLocalSolverEvaluator(
            system=system,
            loop=loop,
            stage=system.set_stage,
            accounting=system.common.accounting,
            solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
            output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
        )
        official = GEPALocalPromptOptimizer(
            evaluator=local_solver,
            reflection_lm=ReflectionLM(system),
            accounting_reader=system.optimizer_accounting,
            run_root=run_root / "local_gepa",
        )
        contextual = ContextualOptimizer(official, local_solver)
        shadow_probe = system.build_probe(shadow_rows)
        evaluator = SystemTeamCandidateEvaluator(
            system=system,
            shadow_probe=shadow_probe,
            loop=loop,
            stage=system.set_stage,
            accounting=system.common.accounting,
            update_index_reader=lambda: 0,
        )
        controller = TeamSearchController(
            responsibility=factory,
            task_builder=task_builder,
            local_optimizer=contextual,
            evaluator=evaluator,
            selector=CommonSafeTeamCandidateSelector(),
            committer=SystemTeamCommitter(
                system=system, evaluator=evaluator, update_index_reader=lambda: 0
            ),
        )
        request = TeamSearchRequest(
            seed=SEED,
            update_index=0,
            team_state_hash=system.team_prompt_state_hash(),
            local_metric_budget=LOCAL_METRIC_BUDGET,
            solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
            output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
        )
        outcome = await controller.run_frozen_opportunity(request, (assignment,))
        telemetry = dict(outcome.audit_metadata.get("local_optimizer_telemetry", {}))
        result = {
            "execution_gate": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "seed": SEED,
            "parent_team_hash": parent_identity["team_hash"],
            "target_member": target,
            "primary_responsibility_lane": assignment.primary_responsibility_lane,
            "proposal_attempts": int(telemetry.get("proposal_attempts", 0)),
            "proposer_diagnostics_version": PROPOSER_DIAGNOSTICS_VERSION,
            "proposer_diagnostics": telemetry.get("proposer_diagnostics", {}),
            "classifier_version": CLASSIFIER_VERSION,
            "positive_minibatch_deltas": int(
                telemetry.get("positive_minibatch_deltas", 0)
            ),
            "accepted_mutations": int(telemetry.get("accepted_mutations", 0)),
            "full_local_evaluations": int(
                telemetry.get("full_local_evaluations", 0)
            ),
            "local_optimizer_solver_calls": outcome.cost.local_optimizer_solver_calls,
            "candidate_solver_calls_beyond_seed": max(
                0, outcome.cost.local_optimizer_solver_calls - 12
            ),
            "changed_valid_candidates": outcome.funnel["local_candidates"],
            "team_minibatch_solver_calls": outcome.cost.team_minibatch_solver_calls,
            "team_minibatch_survivors": outcome.funnel["team_minibatch_survivors"],
            "full_team_evaluated_candidates": outcome.funnel[
                "full_team_evaluated_candidates"
            ],
            "shadow_solver_calls": outcome.cost.team_shadow_solver_calls,
            "commits": outcome.funnel["committed_candidates"],
            "team_minibatch": outcome.audit_metadata["team_minibatch"],
            "local_termination_reason": outcome.audit_metadata[
                "local_termination_reason"
            ],
            "classifier": classify(telemetry, outcome),
            "ledger": _ledger_summary(run_root / "ledger.jsonl"),
            "validation50_calls": 0,
            "test50_calls": 0,
        }
        write_json(run_root / "execution_summary.json", result)
        transition_run_attempt(
            run_root,
            status="COMPLETE",
            provider_calls_observed=_provider_calls_observed(run_root),
        )
        return result
    except BaseException as exc:
        calls = _provider_calls_observed(run_root)
        transition_run_attempt(
            run_root,
            status="FAILED_START" if calls == 0 else "ABORTED",
            provider_calls_observed=calls,
            failure_category=type(exc).__name__,
        )
        raise


def audit(prep: Path, run_root: Path) -> dict[str, Any]:
    verify_freeze(prep)
    summary = read_json(run_root / "execution_summary.json")
    errors: list[str] = []
    lifecycle = read_json(run_root / RUN_LIFECYCLE_FILE)
    if lifecycle.get("status") != "COMPLETE":
        errors.append("run_lifecycle")
    if lifecycle.get("attempt_id") != ATTEMPT_ID:
        errors.append("attempt_identity")
    if lifecycle.get("provider_call_boundary_reached") is not True:
        errors.append("provider_call_boundary")
    for key in ("validation50_calls", "test50_calls"):
        if summary.get(key) != 0:
            errors.append(key)
    minibatch = summary.get("team_minibatch", {})
    if minibatch != {
        "contract_version": "primary_lane_strict_4_4_4_v1",
        "total_count": 12, "responsibility_count": 4,
        "coalition_count": 4, "preservation_count": 4, "backfill_count": 0,
    }:
        errors.append("team_minibatch_contract")
    ledger = summary["ledger"]
    if ledger["input_tokens"] + ledger["output_tokens"] != ledger["total_tokens"]:
        errors.append("ledger_token_arithmetic")
    diagnostics = summary.get("proposer_diagnostics", {})
    required_diagnostics = {
        "proposal_attempts", "materialized_candidates", "unmaterialized_proposals",
        "proposal_changed", "proposal_unchanged", "proposal_duplicate",
        "proposal_contract_invalid", "accepted_candidates",
        "local_frontier_candidates", "returned_frontier_candidates",
        "solver_reached", "positive_minibatch_delta", "accepted_mutation",
        "primary_rejection_category_counts", "failed_check_counts",
    }
    if required_diagnostics - set(diagnostics):
        errors.append("proposer_diagnostics")
    else:
        attempts = int(diagnostics["proposal_attempts"])
        materialized = int(diagnostics.get("materialized_candidates", 0))
        changed = int(diagnostics["proposal_changed"])
        unchanged = int(diagnostics["proposal_unchanged"])
        invalid = int(diagnostics["proposal_contract_invalid"])
        solver_reached = int(diagnostics["solver_reached"])
        categories = diagnostics["primary_rejection_category_counts"]
        if changed + unchanged != attempts:
            errors.append("proposer_attempt_arithmetic")
        if materialized > attempts:
            errors.append("proposer_materialization_arithmetic")
        if sum(int(value) for value in categories.values()) != invalid:
            errors.append("proposer_rejection_arithmetic")
        if not 0 <= solver_reached <= changed - invalid:
            errors.append("proposer_solver_reach_arithmetic")
    result = {
        "gate": "PASS" if not errors else "HOLD", "errors": errors,
        "protocol_sha256": read_json(prep / "source_freeze.json")["protocol_sha256"],
        "validation50_calls": 0, "test50_calls": 0,
    }
    write_json(run_root / "audit.json", result)
    return result


def analyze(prep: Path, run_root: Path, report: Path) -> dict[str, Any]:
    gate = audit(prep, run_root)
    if gate["gate"] != "PASS":
        raise RuntimeError("canary audit HOLD")
    if report.exists():
        raise FileExistsError("fresh canary report root required")
    report.mkdir(parents=True)
    summary = read_json(run_root / "execution_summary.json")
    write_json(report / "summary.json", summary)
    write_json(report / "audit.json", gate)
    write_json(report / "protocol_freeze.json", read_json(prep / "protocol_freeze.json"))
    write_json(report / "provenance.json", {
        "evidence_type": "real_provider_engineering_canary",
        "execution_commit": read_json(prep / "source_freeze.json")["execution_commit"],
        "validation50_calls": 0, "test50_calls": 0,
        "raw_prompts_published": False, "raw_responses_published": False,
    })
    (report / "README.md").write_text(
        "# Level-B GEPA real-provider canary\n\n"
        f"Gate: **{gate['gate']}**  \n"
        f"Classifier: **{summary['classifier']}**\n\n"
        "This single-parent engineering canary tests empirical path activation, not "
        "scheduler or accuracy efficacy. Validation50 and Test50 calls are zero.\n",
        encoding="utf-8",
    )
    return summary


def preflight() -> dict[str, Any]:
    verify_frozen_gepa_engine_contract()
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    manifest_schema = json.loads(
        (ROOT / "infrastructure/experiment_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    protocol = protocol_document()
    checks = {
        "one_opportunity": protocol["opportunities"] == 1,
        "one_target": protocol["target_branches"] == 1,
        "unchanged_gepa_budget": protocol["local_metric_budget"] == 36,
        "decision_procedure_component": protocol["official_gepa"]["candidate_component_name"] == "decision_procedure",
        "level_b": protocol["official_gepa"]["optimizer_fidelity_level"] == "LEVEL_B_API_COMPATIBLE_ADAPTATION",
        "validation_zero": protocol["validation50_calls"] == 0,
        "test_zero": protocol["test50_calls"] == 0,
        "manifest_preflight_pass": manifest.get("status") == "PREFLIGHT_PASS",
        "manifest_schema": not validate_manifest(manifest, manifest_schema),
        "attempt_identity": protocol["attempt_id"] == ATTEMPT_ID,
        "launch_transaction": protocol["launch_transaction_version"] == LAUNCH_TRANSACTION_VERSION,
        "proposer_diagnostics": protocol["proposer_diagnostics_version"] == PROPOSER_DIAGNOSTICS_VERSION,
        "classifier_version": protocol["classifier_version"] == CLASSIFIER_VERSION,
        "preregistration_hash": (
            manifest.get("artifacts", {}).get("preregistration", {}).get("sha256")
            == preregistration_hash(manifest)
        ),
        "no_retry_resume": protocol["no_resume"] and protocol["experiment_retry_count"] == 0,
    }
    return {"gate": "PASS" if all(checks.values()) else "HOLD", "checks": checks,
            "api_calls": 0, "validation_calls": 0, "test_calls": 0}


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
