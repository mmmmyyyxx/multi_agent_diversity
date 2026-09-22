"""Fresh-initialized successor to the sequential symmetry-breaking pilot.

Only initialization/provider/freeze mechanics differ from v1.  The online
controller, GEPA backend, evaluation stages, safety rules, and write-back path
are reused verbatim from the audited implementation.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
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

import run_sequential_symmetry_breaking_online_pilot_v1 as base  # noqa: E402
from scripts.anti_overfitting_shadow_support import source_items  # noqa: E402
from multi_dataset_diverse_rl.governance.execution_harness_v2 import (  # noqa: E402
    INITIALIZATION_POLICY,
    LOCAL_NO_UPDATE_PATIENCE,
    PROVIDER_PROFILE,
    ROLE_MODEL,
    SOLVER_MODEL,
    TEAM_NO_UPDATE_PATIENCE,
    preflight_provider_binding,
)
from multi_dataset_diverse_rl.governance.startup_identity import (  # noqa: E402
    build_startup_bundle,
    read_bundle,
    validate_startup_bundle,
    write_bundle,
)


EXPERIMENT_ID = "sequential_symmetry_breaking_online_pilot_v2"
ATTEMPT_ID = "sequential_symmetry_breaking_online_pilot_v2_pending_authorization"
SEED = 80
MANIFEST = ROOT / "experiments/manifests/sequential_symmetry_breaking_online_pilot_v2.yaml"
DESIGN = ROOT / "experiments/sequential_symmetry_breaking_online_pilot_v2"
DEFAULT_PREP = ROOT / "runs/sequential_symmetry_breaking_online_pilot_v2_prep"
DEFAULT_RUN = ROOT / "runs/sequential_symmetry_breaking_online_pilot_v2_attempt1"
DEFAULT_REPORT = ROOT / "reports/sequential_symmetry_breaking_online_pilot_v2_attempt1"
AUTH_ENV = "SEQUENTIAL_SYMMETRY_BREAKING_ONLINE_PILOT_V2_AUTHORIZED"

base.EXPERIMENT_ID = EXPERIMENT_ID
base.SEED = SEED
base.MANIFEST = MANIFEST
base.DESIGN = DESIGN
base.DEFAULT_PREP = DEFAULT_PREP
base.DEFAULT_RUN = DEFAULT_RUN
base.DEFAULT_REPORT = DEFAULT_REPORT
base.AUTH_ENV = AUTH_ENV

_base_config = base._config
_base_protocol_document = base.protocol_document


def _config(out: Path, *, optimize_path: Path, shadow_path: Path):
    cfg = _base_config(out, optimize_path=optimize_path, shadow_path=shadow_path)
    return replace(
        cfg,
        models=replace(
            cfg.models,
            provider_profile=PROVIDER_PROFILE,
            agent_model=SOLVER_MODEL,
            optimizer_model=ROLE_MODEL,
            evaluator_model=ROLE_MODEL,
        ),
        training=replace(cfg.training, seed=SEED),
    )


def protocol_document() -> dict[str, Any]:
    protocol = _base_protocol_document()
    prompt = _config(
        ROOT / "runs/_preflight_only",
        optimize_path=ROOT / "PREPARED_OPTIMIZE100.csv",
        shadow_path=ROOT / "PREPARED_SHADOW50.csv",
    ).training.shared_prompt
    protocol.update(
        {
            "schema_version": "sequential_symmetry_breaking_online_protocol_v2",
            "experiment_id": EXPERIMENT_ID,
            "seed": SEED,
            "provider_profile": PROVIDER_PROFILE,
            "initialization": {
                "policy": INITIALIZATION_POLICY,
                "canonical_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "member_prompt_policy": "shared_identical",
                "private_parent_dependency": False,
                "focus": [],
                "anchor": [],
                "cost_partition": "INITIALIZATION_SOLVER_CALLS",
            },
            "saturation_patience_reference": {
                "local_no_update_patience": LOCAL_NO_UPDATE_PATIENCE,
                "team_no_update_patience": TEAM_NO_UPDATE_PATIENCE,
                "pilot_ceiling_remains_distinct": True,
            },
        }
    )
    protocol["baseline"] = {
        "policy": INITIALIZATION_POLICY,
        "members": 5,
        "canonical_prompt_sha256": protocol["initialization"]["canonical_prompt_sha256"],
        "fresh_empirical_materialization": True,
        "historical_private_parent_used": False,
    }
    return protocol


def _source_paths() -> list[Path]:
    paths = [
        path.relative_to(ROOT)
        for path in (ROOT / "multi_dataset_diverse_rl").rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    paths.extend(
        [
            Path("infrastructure/common_solver_contract_v1/contract.py"),
            Path("infrastructure/common_solver_contract_v1/evaluator.py"),
            Path("scripts/anti_overfitting_shadow_support.py"),
            Path("scripts/run_seed78_primary_responsibility_ab.py"),
            Path("scripts/run_sequential_symmetry_breaking_online_pilot_v1.py"),
            Path("scripts/run_sequential_symmetry_breaking_online_pilot_v2.py"),
            Path("multi_dataset_diverse_rl/governance/startup_identity.py"),
            MANIFEST.relative_to(ROOT),
            (DESIGN / "PROTOCOL.md").relative_to(ROOT),
            Path("experiments/sequential_symmetry_breaking_online_pilot_v1/classifier_definition.json"),
            Path("experiments/anti_overfitting_split_v1/split_manifest.json"),
            Path("experiments/anti_overfitting_split_v1/fold_assignment.json"),
        ]
    )
    return sorted(set(paths), key=lambda value: value.as_posix())


def _canonical_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    _items, raw = source_items()
    assignment = json.loads(
        (ROOT / "experiments/anti_overfitting_split_v1/fold_assignment.json").read_text(
            encoding="utf-8"
        )
    )["folds"]
    optimize = [raw[digest] for digest in assignment["fold_a"] + assignment["fold_b"]]
    shadow = [raw[digest] for digest in assignment["fold_c"]]
    return optimize, shadow


def _startup_bundle(prep: Path, *, execution_source_sha: str | None = None) -> dict[str, Any]:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    source_files = [
        {"path": path.as_posix(), "sha256": base.sha256_file(ROOT / path)}
        for path in _source_paths()
    ]
    data_hashes = {
        path.name: base.sha256_file(path)
        for path in sorted((prep / "splits_private").glob("*.csv"))
    }
    return build_startup_bundle(
        manifest=manifest,
        protocol=protocol_document(),
        experiment_id=EXPERIMENT_ID,
        attempt_id=ATTEMPT_ID,
        scientific_method_anchor_sha=str(
            manifest["execution_freeze"]["scientific_method_anchor_sha"]
        ),
        execution_source_sha=execution_source_sha or base._git("rev-parse", "HEAD"),
        provider_profile=PROVIDER_PROFILE,
        endpoint_fingerprint=str(
            manifest["execution_freeze"]["provider"]["endpoint_fingerprint"]
        ),
        models=manifest["execution_freeze"]["provider"]["models"],
        data_hashes=data_hashes,
        initialization=protocol_document()["initialization"],
        seeds=[SEED],
        local_patience=LOCAL_NO_UPDATE_PATIENCE,
        team_patience=TEAM_NO_UPDATE_PATIENCE,
        saturation_mode="sequential_mechanism_pilot",
        source_files=source_files,
    )


def prepare(prep: Path) -> dict[str, Any]:
    if prep.exists():
        raise FileExistsError("fresh prep root required")
    if base._git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before source freeze")
    gate = preflight()
    if gate["gate"] != "PASS":
        raise RuntimeError("sequential v2 preflight must pass before freeze")
    optimize_rows, shadow_rows = _canonical_rows()
    prep.mkdir(parents=True)
    base._write_csv(
        prep / "splits_private/optimize100.csv", optimize_rows, ("question", "answer")
    )
    base._write_csv(
        prep / "splits_private/shadow50.csv", shadow_rows, ("question", "answer")
    )
    protocol = protocol_document()
    base.write_json(prep / "protocol_freeze.json", protocol)
    base.write_json(
        prep / "evaluation_access_registry.json",
        {
            "events": [],
            "initialization_logical_rows": 500,
            "initialization_cache_hits": None,
            "initialization_provider_successes": None,
            "validation50_calls": 0,
            "test50_calls": 0,
        },
    )
    startup = _startup_bundle(prep)
    write_bundle(prep / "startup_identity", startup)
    freeze = {
        "execution_commit": base._git("rev-parse", "HEAD"),
        "protocol_sha256": base.sha256_json(protocol),
        "preregistration_sha256": startup["scientific_identity"]["preregistration_sha256"],
        "run_identity_sha256": startup["run_identity"]["run_identity_sha256"],
        "source_files": startup["scientific_identity"]["payload"]["source_files"],
        "private_inputs": startup["scientific_identity"]["payload"]["data_hashes"],
        "private_parent_dependencies": [],
        "initialization_policy": INITIALIZATION_POLICY,
    }
    base.write_json(prep / "source_freeze.json", freeze)
    result = {
        "gate": "PASS",
        "ready_to_run": True,
        "authorization_state": "AUTHORIZATION_REQUIRED",
        "execution_commit": freeze["execution_commit"],
        "protocol_sha256": freeze["protocol_sha256"],
        "provider_attempts": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    }
    base.write_json(prep / "preflight_gate.json", result)
    return result


def _verify_freeze(prep: Path) -> None:
    freeze = base._read_json(prep / "source_freeze.json")
    if base.sha256_json(base._read_json(prep / "protocol_freeze.json")) != freeze["protocol_sha256"]:
        raise RuntimeError("protocol freeze mismatch")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze["execution_commit"], "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise RuntimeError("execution commit is not an ancestor of HEAD")
    for row in freeze["source_files"]:
        if base.sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {row['path']}")
    for name, digest in freeze["private_inputs"].items():
        if base.sha256_file(prep / "splits_private" / name) != digest:
            raise RuntimeError(f"private split freeze mismatch: {name}")
    if freeze.get("private_parent_dependencies") != []:
        raise RuntimeError("private parent dependency reintroduced")
    expected = _startup_bundle(prep, execution_source_sha=freeze["execution_commit"])
    result = validate_startup_bundle(
        stored=read_bundle(prep / "startup_identity"),
        expected=expected,
        require_authorized=False,
    )
    if result["preregistration_sha256"] != freeze["preregistration_sha256"]:
        raise RuntimeError("startup preregistration mismatch")


async def execute(prep: Path, run_root: Path) -> dict[str, Any]:
    _verify_freeze(prep)
    validate_startup_bundle(
        stored=read_bundle(prep / "startup_identity"),
        expected=_startup_bundle(
            prep,
            execution_source_sha=base._read_json(prep / "source_freeze.json")[
                "execution_commit"
            ],
        ),
        require_authorized=True,
        phase="online_trajectory",
        roles=("solver", "reflection"),
    )
    if run_root.exists():
        raise FileExistsError("fresh run root required; resume and automatic retry are forbidden")
    run_root.mkdir(parents=True)
    optimize_rows = base._rows(prep / "splits_private/optimize100.csv")
    shadow_rows = base._rows(prep / "splits_private/shadow50.csv")
    ledger = base.BoundedLedger(run_root / "ledger.jsonl")
    cfg = _config(
        run_root,
        optimize_path=prep / "splits_private/optimize100.csv",
        shadow_path=prep / "splits_private/shadow50.csv",
    )
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    preflight_provider_binding(cfg, manifest["execution_freeze"]["provider"])
    system = base.Seed78System(cfg, arm=base.ARM, ledger=ledger, raw_cache={})
    system.set_run_identity(
        base.build_run_identity(
            cfg,
            train_rows=optimize_rows,
            val_rows=shadow_rows,
            test_rows=[],
            workspace=ROOT,
        )
    )
    initialization_successes_before = ledger.successes
    system.set_stage(
        {"phase": "initialization", "update_index": -1, "target_member": -1, "candidate_id": "P0"}
    )
    try:
        await system.initialize_fixed_probe(optimize_rows)
    finally:
        system.set_stage(None)
    initialization_successes = ledger.successes - initialization_successes_before
    baseline_snapshot = system.persist_endpoint_identifiability_state(
        update_index=-1, trigger="fresh_deterministic_initialization_v1"
    )
    baseline_metrics = system.active_probe_metrics()
    initial_prompt_hashes = [system.prompt_hash(agent.current_prompt) for agent in system.agents]
    if len(set(initial_prompt_hashes)) != 1:
        raise RuntimeError("canonical initialization is not shared-identical")
    tracker = base.SymmetryBreakingTrajectory(
        baseline_vote_correct=baseline_metrics.vote_correct_count,
        baseline_p_i=baseline_snapshot["p_i"],
    )

    loop = asyncio.get_running_loop()
    task_builder = base.LocalTaskBuilder()
    holder: dict[str, Any] = {}
    assignment_factory = base.SystemResponsibilityAssignmentFactory(
        system=system,
        snapshot_reader=lambda: holder["value"],
        task_builder=task_builder,
    )
    local_solver = base.SystemLocalSolverEvaluator(
        system=system,
        loop=loop,
        stage=system.set_stage,
        accounting=system.common.accounting,
        solver_contract_id=base.COMMON_SOLVER_CONTRACT_ID,
        output_contract_id=base.SOLVER_OUTPUT_CONTRACT_VERSION,
    )
    official = base.GEPALocalPromptOptimizer(
        evaluator=local_solver,
        reflection_lm=base.ReflectionLM(system),
        accounting_reader=system.optimizer_accounting,
        run_root=run_root / "local_gepa",
    )
    freeze = json.loads((prep / "source_freeze.json").read_text(encoding="utf-8"))
    optimizer = base.ContextualOptimizer(
        official,
        local_solver,
        base.execution_context_from_system(
            system,
            run_identity_sha256=freeze.get("run_identity_sha256"),
            local_no_update_patience=LOCAL_NO_UPDATE_PATIENCE,
            team_no_update_patience=TEAM_NO_UPDATE_PATIENCE,
            saturation_mode="sequential_symmetry_breaking_online_saturation",
        ),
    )
    update = {"value": -1}
    evaluator = base.SystemTeamCandidateEvaluator(
        system=system,
        shadow_probe=system.build_probe(shadow_rows),
        loop=loop,
        stage=system.set_stage,
        accounting=system.common.accounting,
        update_index_reader=lambda: update["value"],
    )
    controller = base.TeamSearchController(
        responsibility=assignment_factory,
        task_builder=task_builder,
        local_optimizer=optimizer,
        evaluator=evaluator,
        selector=base.CommonSafeTeamCandidateSelector(),
        committer=base.SystemTeamCommitter(
            system=system, evaluator=evaluator, update_index_reader=lambda: update["value"]
        ),
    )
    scheduler = base.PrimaryResponsibilityPersistentRealizabilityScheduler()
    binding = base.PrimaryResponsibilityOnlineBinding(
        scheduler=scheduler,
        assignment_factory=assignment_factory,
        controller=controller,
    )

    events: list[dict[str, Any]] = []
    stop_reason = "max_opportunities"
    for update_index in range(base.MAX_OPPORTUNITIES):
        update["value"] = update_index
        frozen = base.freeze_current_responsibility(system, update_index=update_index)
        holder["value"] = frozen
        outcome = await binding.run_opportunity(
            base.TeamSearchRequest(
                seed=SEED,
                update_index=update_index,
                team_state_hash=system.team_prompt_state_hash(),
                local_metric_budget=base.LOCAL_GEPA_METRIC_BUDGET,
                solver_contract_id=base.COMMON_SOLVER_CONTRACT_ID,
                output_contract_id=base.SOLVER_OUTPUT_CONTRACT_VERSION,
            ),
            assigned=frozen.assigned,
            current_margin_by_question=frozen.current_margin_by_question,
        )
        state = system.persist_endpoint_identifiability_state(
            update_index=update_index,
            trigger="post_online_opportunity",
            committed_target_member=outcome.team_outcome.audit_metadata["committed_member_id"],
        )
        metrics = system.active_probe_metrics()
        tracker.observe(
            committed=outcome.team_outcome.committed_candidate_id is not None,
            vote_correct=metrics.vote_correct_count,
            p_i=state["p_i"],
        )
        event = base._trajectory_event(outcome, state, metrics)
        events.append(event)
        base.write_json(run_root / f"opportunity_{update_index:02d}.json", event)
        base.write_json(
            run_root / "checkpoint_private.json",
            {
                "completed_opportunities": len(events),
                "team_identity": base._profile_identity(system),
                "prompts": [agent.current_prompt for agent in system.agents],
                "scheduler_state": scheduler.state.checkpoint_payload(),
                "trajectory": tracker.payload(),
                "provider_attempts": ledger.attempts,
                "successful_provider_calls": ledger.successes,
            },
        )
        reason = tracker.stop_reason(
            max_opportunities=base.MAX_OPPORTUNITIES,
            max_commits=base.MAX_SAFE_COMMITS,
            no_commit_patience=base.NO_COMMIT_PATIENCE,
        )
        if reason is not None:
            stop_reason = reason
            break

    trajectory = tracker.payload()
    reflection_attempt_rows = [
        row for row in system.llm.calls if row.get("client_role") == "optimizer"
    ]
    reflection_successes = sum(bool(row.get("success")) for row in reflection_attempt_rows)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "execution_gate": "PASS",
        "initialization": {
            "policy": INITIALIZATION_POLICY,
            "logical_member_rows": len(optimize_rows) * 5,
            "provider_successes": initialization_successes,
            "cache_hits": len(optimize_rows) * 5 - initialization_successes,
            "initial_prompt_hash": initial_prompt_hashes[0],
            "member_prompt_hashes": initial_prompt_hashes,
            "team_prompt_state_hash": baseline_snapshot["team_prompt_state_hash"],
            "profile_state_sha256": baseline_snapshot["profile_state_sha256"],
            "vote_correct": baseline_metrics.vote_correct_count,
            "oracle_correct": sum(
                int(row.gold_vote_count > 0) for row in baseline_metrics.rows
            ),
            "member_correct": list(baseline_metrics.per_agent_correct_counts),
            "p_i": baseline_snapshot["p_i"],
            "focus": [],
            "anchor": [],
        },
        "stop_reason": stop_reason,
        "trajectory": trajectory,
        "events": events,
        "scheduler_telemetry": scheduler.telemetry_summary(),
        "final_team_identity": base._profile_identity(system),
        "provider_attempts": ledger.attempts,
        "successful_provider_calls": ledger.successes,
        "initialization_solver_calls": initialization_successes,
        "optimization_provider_successes": ledger.successes - initialization_successes,
        "successful_reflection_calls": reflection_successes,
        "validation50_calls": 0,
        "test50_calls": 0,
    }
    base.write_json(run_root / "execution_summary_private.json", summary)
    return summary


def preflight() -> dict[str, Any]:
    base.verify_frozen_gepa_engine_contract()
    optimize_rows, shadow_rows = _canonical_rows()
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    cfg = _config(
        ROOT / "runs/_preflight_only",
        optimize_path=ROOT / "PREPARED_OPTIMIZE100.csv",
        shadow_path=ROOT / "PREPARED_SHADOW50.csv",
    )
    provider = preflight_provider_binding(
        cfg, manifest["execution_freeze"]["provider"], construct_client=False
    )
    protocol = protocol_document()
    checks = {
        "provider_profile_explicit": cfg.models.provider_profile == PROVIDER_PROFILE,
        "models_exact": cfg.models.agent_model == SOLVER_MODEL
        and cfg.models.optimizer_model == ROLE_MODEL
        and cfg.models.evaluator_model == ROLE_MODEL,
        "endpoint_fingerprint": bool(provider.endpoint_fingerprint),
        "optimize100_rows": len(optimize_rows) == 100,
        "shadow50_rows": len(shadow_rows) == 50,
        "fresh_initialization": protocol["initialization"]["policy"] == INITIALIZATION_POLICY,
        "private_parent_independent": protocol["initialization"]["private_parent_dependency"] is False,
        "root_transition_empty": protocol["initialization"]["focus"] == []
        and protocol["initialization"]["anchor"] == [],
        "validation_zero": protocol["access"]["validation50"] == "zero access",
        "test_zero": protocol["access"]["test50"] == "zero access",
        "authorization_required": manifest.get("api_authorization", {}).get("authorized") is False,
        "ready_for_authorization": manifest.get("status") == "PREFLIGHT_PASS",
    }
    return {
        "gate": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "ready_to_run": all(checks.values()),
        "authorization_state": "AUTHORIZATION_REQUIRED",
        "provider_attempts": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    }


base._config = _config
base.protocol_document = protocol_document
base._source_paths = _source_paths
base.prepare = prepare
base._verify_freeze = _verify_freeze
base.execute = execute
base.preflight = preflight


if __name__ == "__main__":
    base.main()
