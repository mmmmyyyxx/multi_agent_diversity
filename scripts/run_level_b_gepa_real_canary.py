"""One-parent real-provider canary for the Level-B GEPA adapter."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
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
from infrastructure.common_solver_contract_v1.contract import COMMON_SOLVER_CONTRACT_ID
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
SEED = 78
ARM = "LEVEL_B_REAL_CANARY"
LOCAL_METRIC_BUDGET = 36
AUTH_ENV = "LEVEL_B_GEPA_REAL_CANARY_AUTHORIZED"
MANIFEST = ROOT / "experiments/manifests/level_b_gepa_real_canary_v1.yaml"
PROTOCOL = ROOT / "experiments/level_b_gepa_real_canary_v1/PROTOCOL.md"
DEFAULT_PREP = ROOT / "runs/level_b_gepa_real_canary_v1_prep"
DEFAULT_RUN = ROOT / "runs/level_b_gepa_real_canary_v1"
DEFAULT_REPORT = ROOT / "reports/level_b_gepa_real_canary_v1"


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
        "schema_version": "level_b_gepa_real_canary_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
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
        "no_retry": True,
    }


def prepare(prep: Path) -> dict[str, Any]:
    if prep.exists():
        raise FileExistsError("fresh canary prep root required")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before canary freeze")
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
        "protocol_sha256": sha256_json(protocol),
        "files": [
            {"path": path.as_posix(), "sha256": sha256_file(ROOT / path)}
            for path in source_paths()
        ],
        "private_split_sha256": {
            path.name: sha256_file(path)
            for path in sorted((prep / "splits_private").glob("*.csv"))
        },
    }
    write_json(prep / "source_freeze.json", freeze)
    result = {
        "gate": "PASS", "execution_commit": freeze["execution_commit"],
        "protocol_sha256": freeze["protocol_sha256"], "api_calls": 0,
        "validation_calls": 0, "test_calls": 0,
    }
    write_json(prep / "phase_a_gate.json", result)
    return result


def verify_freeze(prep: Path) -> None:
    freeze = read_json(prep / "source_freeze.json")
    if git("rev-parse", "HEAD") != freeze["execution_commit"]:
        raise RuntimeError("canary execution commit mismatch")
    if sha256_json(read_json(prep / "protocol_freeze.json")) != freeze["protocol_sha256"]:
        raise RuntimeError("canary protocol hash mismatch")
    for row in freeze["files"]:
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"canary source freeze mismatch: {row['path']}")
    for name, digest in freeze["private_split_sha256"].items():
        if sha256_file(prep / "splits_private" / name) != digest:
            raise RuntimeError(f"canary split freeze mismatch: {name}")


def authorize() -> None:
    if os.environ.get(AUTH_ENV) != "1":
        raise PermissionError(f"{AUTH_ENV}=1 is required")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for role in ("solver", "reflection"):
        require_api_authorization(
            manifest, phase="canary", role=role, explicit_user_authorized=True
        )


def classify(telemetry: dict[str, Any], outcome: Any) -> str:
    proposal_attempts = int(telemetry.get("proposal_attempts", 0))
    post_seed_solver_calls = max(0, outcome.cost.local_optimizer_solver_calls - 12)
    changed = int(outcome.funnel["local_candidates"])
    entered = int(outcome.cost.team_minibatch_solver_calls) > 0
    if proposal_attempts <= 0:
        return "NO_REAL_PROPOSAL_ATTEMPT"
    if post_seed_solver_calls <= 0:
        return "PROPOSAL_CONTRACT_STILL_BLOCKS_EMPIRICAL_SEARCH"
    if changed <= 0 or not entered:
        return "LOCAL_SEARCH_EMPIRICAL_BUT_NO_CHANGED_FRONTIER"
    return "BACKEND_EMPIRICAL_PATH_CONFIRMED"


def config(out: Path, *, optimize_path: Path, validation_path: Path) -> Config:
    return Config.from_flat(
        task_type="bbh", dataset_format="mars", comparison_task_id="disambiguation_qa",
        benchmark="BBH", answer_format="option_letter",
        train_path=str(optimize_path.resolve()), val_path=str(validation_path.resolve()),
        test_path="TEST50_BLOCKED", manifest_sha256=sha256_file(MANIFEST),
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
    authorize()
    verify_freeze(prep)
    if run_root.exists():
        raise FileExistsError("fresh canary run root required; retry/resume forbidden")
    run_root.mkdir(parents=True)
    optimize_rows = _rows(prep / "splits_private/optimize100.csv")
    shadow_rows = _rows(prep / "splits_private/fold_c.csv")
    validation_rows = _rows(prep / "splits_private/validation.csv")
    ledger = DurableLedger(run_root / "ledger.jsonl")
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
        system=system, snapshot_reader=lambda: snapshot, task_builder=task_builder
    )
    assignment = factory.build_from_member(
        request=TeamSearchRequest(
            seed=SEED, update_index=0, team_state_hash=system.team_prompt_state_hash(),
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
        system=system, loop=loop, stage=system.set_stage,
        accounting=system.common.accounting,
        solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
        output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
    )
    official = GEPALocalPromptOptimizer(
        evaluator=local_solver, reflection_lm=ReflectionLM(system),
        accounting_reader=system.optimizer_accounting,
        run_root=run_root / "local_gepa",
    )
    contextual = ContextualOptimizer(official, local_solver)
    shadow_probe = system.build_probe(shadow_rows)
    evaluator = SystemTeamCandidateEvaluator(
        system=system, shadow_probe=shadow_probe, loop=loop,
        stage=system.set_stage, accounting=system.common.accounting,
        update_index_reader=lambda: 0,
    )
    controller = TeamSearchController(
        responsibility=factory, task_builder=task_builder,
        local_optimizer=contextual, evaluator=evaluator,
        selector=CommonSafeTeamCandidateSelector(),
        committer=SystemTeamCommitter(
            system=system, evaluator=evaluator, update_index_reader=lambda: 0
        ),
    )
    request = TeamSearchRequest(
        seed=SEED, update_index=0, team_state_hash=system.team_prompt_state_hash(),
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
        "positive_minibatch_deltas": int(telemetry.get("positive_minibatch_deltas", 0)),
        "accepted_mutations": int(telemetry.get("accepted_mutations", 0)),
        "full_local_evaluations": int(telemetry.get("full_local_evaluations", 0)),
        "local_optimizer_solver_calls": outcome.cost.local_optimizer_solver_calls,
        "candidate_solver_calls_beyond_seed": max(
            0, outcome.cost.local_optimizer_solver_calls - 12
        ),
        "changed_valid_candidates": outcome.funnel["local_candidates"],
        "team_minibatch_solver_calls": outcome.cost.team_minibatch_solver_calls,
        "team_minibatch_survivors": outcome.funnel["team_minibatch_survivors"],
        "full_team_evaluated_candidates": outcome.funnel["full_team_evaluated_candidates"],
        "shadow_solver_calls": outcome.cost.team_shadow_solver_calls,
        "commits": outcome.funnel["committed_candidates"],
        "team_minibatch": outcome.audit_metadata["team_minibatch"],
        "local_termination_reason": outcome.audit_metadata["local_termination_reason"],
        "classifier": classify(telemetry, outcome),
        "ledger": _ledger_summary(run_root / "ledger.jsonl"),
        "validation50_calls": 0,
        "test50_calls": 0,
    }
    write_json(run_root / "execution_summary.json", result)
    return result


def audit(prep: Path, run_root: Path) -> dict[str, Any]:
    verify_freeze(prep)
    summary = read_json(run_root / "execution_summary.json")
    errors: list[str] = []
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
    for role in ("solver", "reflection"):
        require_api_authorization(
            manifest, phase="canary", role=role, explicit_user_authorized=True
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
        "no_retry_resume": protocol["no_resume"] and protocol["no_retry"],
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
