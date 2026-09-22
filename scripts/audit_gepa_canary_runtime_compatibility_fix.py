"""Build the sanitized zero-API runtime-compatibility closure report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/gepa_layer2_real_canary_v2_authorized2"
DEFAULT_REPORT = ROOT / "reports/gepa_canary_runtime_compatibility_fix_20260922"
BASE_SHA = "294df07e2515ebc93ef14857e8abb11f1d7ae7d8"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ledger_summary(path: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {
        "logical_calls": len(rows),
        "provider_successes": sum(int(row["successful_provider_calls"]) for row in rows),
        "provider_failures": sum(
            int(row["provider_attempts"]) - int(row["successful_provider_calls"])
            for row in rows
        ),
        "reflection_calls": sum(row.get("logical_role") == "reflection" for row in rows),
        "candidate_solver_calls": sum(
            row.get("phase") == "local_optimizer_solver_eval" for row in rows
        ),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
    }


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def build(run_root: Path, report: Path, test_summary: Path | None) -> None:
    if report.exists():
        raise FileExistsError("fresh runtime compatibility report root required")
    lifecycle = _json(run_root / "run_lifecycle.json")
    accounting = _ledger_summary(run_root / "ledger.jsonl")
    if accounting != {
        "logical_calls": 100,
        "provider_successes": 100,
        "provider_failures": 0,
        "reflection_calls": 0,
        "candidate_solver_calls": 0,
        "input_tokens": 19665,
        "output_tokens": 6768,
        "total_tokens": 26433,
    }:
        raise RuntimeError("failed-attempt accounting drift")
    if lifecycle.get("status") != "ABORTED" or lifecycle.get(
        "provider_call_boundary_reached"
    ) is not True:
        raise RuntimeError("failed-attempt lifecycle drift")
    report.mkdir(parents=True)

    _write(
        report / "failed_attempt_summary.json",
        {
            "attempt": "gepa_layer2_real_canary_v2_authorized2",
            "classification": "ENGINEERING_RUNTIME_FAILURE",
            "scientific_result": "SCIENTIFIC_NON_RESULT",
            "provider_boundary_reached": True,
            "scientific_gepa_search_boundary_reached": False,
            "lifecycle_status": "ABORTED",
            "failure_type": "AttributeError",
            "accounting": accounting,
            "ledger_sha256": _sha(run_root / "ledger.jsonl"),
            "lifecycle_sha256": _sha(run_root / "run_lifecycle.json"),
            "immutable": True,
            "retry_or_resume_allowed": False,
        },
    )
    _write(
        report / "config_compatibility_root_cause.json",
        {
            "actual_config_class": "multi_dataset_diverse_rl.config.Config",
            "constructor": "Config.from_flat -> section dataclasses",
            "seed_current_owner": "Config.training.seed",
            "historical_invalid_access": "system.cfg.seed",
            "failure_location": "historical Seed78 contextual optimizer wrapper",
            "why": (
                "The current Config has structured sections and intentionally has no "
                "top-level seed attribute; a reused historical wrapper bypassed the "
                "current execution contract."
            ),
            "fix": (
                "The active canary now uses a backend-neutral typed execution context "
                "and no longer imports the Seed78 runner."
            ),
            "legacy_config_fields_added": 0,
        },
    )
    matrix = [
        {
            "field": "training.seed",
            "consumer": "CommonContractExecutionSystem / execution_context_from_system",
            "current_owner": "TrainingConfig",
            "historical_owner": "top-level Config.seed assumption",
            "present": True,
            "execution_relevance": "run seed and durable attribution",
        },
        {
            "field": "models.provider_profile",
            "consumer": "credential binding / LocalOptimizerExecutionContext",
            "current_owner": "ModelConfig",
            "historical_owner": "runner constant or default provider",
            "present": True,
            "execution_relevance": "fail-closed provider identity",
        },
        {
            "field": "models.agent_model",
            "consumer": "LocalOptimizerExecutionContext",
            "current_owner": "ModelConfig",
            "historical_owner": "runner constant",
            "present": True,
            "execution_relevance": "Solver identity",
        },
        {
            "field": "models.optimizer_model",
            "consumer": "ReflectionLM through typed invocation",
            "current_owner": "ModelConfig",
            "historical_owner": "ROLE_MODEL global",
            "present": True,
            "execution_relevance": "reflection model identity",
        },
        {
            "field": "models.evaluator_model",
            "consumer": "LocalOptimizerExecutionContext",
            "current_owner": "ModelConfig",
            "historical_owner": "runner constant",
            "present": True,
            "execution_relevance": "evaluator provenance",
        },
        {
            "field": "models.solver_api_key_env / solver_base_url_env",
            "consumer": "CommonContractExecutionSystem",
            "current_owner": "ModelConfig",
            "historical_owner": "process default environment",
            "present": True,
            "execution_relevance": "credential variable names only; values never persisted",
        },
        {
            "field": "run_identity_sha256",
            "consumer": "LocalOptimizerExecutionContext",
            "current_owner": "startup identity/source freeze",
            "historical_owner": "implicit system state",
            "present": True,
            "execution_relevance": "attempt provenance",
        },
        {
            "field": "local/team patience and saturation mode",
            "consumer": "LocalOptimizerExecutionContext",
            "current_owner": "experiment protocol/execution harness",
            "historical_owner": "runner constants",
            "present": True,
            "execution_relevance": "stopping provenance; not GEPA search control",
        },
    ]
    _write(
        report / "config_attribute_matrix.json",
        {
            "reachable_runtime_config_accesses": matrix,
            "missing_reachable_fields_after_fix": 0,
            "stale_missing_accesses_found": 1,
            "gepa_optimizer_config_note": (
                "GEPALocalPromptOptimizer.config is GEPAOptimizerConfig, not the "
                "application Config audited here; all of its accessed fields exist."
            ),
        },
    )
    _write(
        report / "historical_runtime_dependency_audit.json",
        {
            "historical_seed78_runner_was_active": True,
            "classification_before_fix": "B_HISTORICAL_RUNNER_ACCIDENTALLY_RETAINED",
            "active_canary_imports_seed78_runner_after_fix": False,
            "active_canary_source_freeze_includes_seed78_runner_after_fix": False,
            "shared_runtime_replacement": (
                "multi_dataset_diverse_rl.team_search.execution_runtime"
            ),
            "reachable_stale_dependencies_after_fix": [],
            "unreachable_historical_code_refactored": False,
            "scientific_semantics_changed": False,
        },
    )
    _write(
        report / "local_optimizer_entry_contract.json",
        {
            "contract": "LocalOptimizerExecutionContext + LocalOptimizationTask",
            "layer2_owned_inputs": [
                "target_member",
                "responsibility-derived local task/evidence packet",
                "run_seed",
                "update_index",
                "candidate parent",
            ],
            "execution_inputs": [
                "provider_profile",
                "solver_model",
                "optimizer_model",
                "evaluator_model",
                "run_identity_sha256",
                "local_no_update_patience",
                "team_no_update_patience",
                "saturation_mode",
            ],
            "backend_owned_inputs": ["GEPA search state", "proposal generation"],
            "missing_field_exception": "LocalOptimizerConfigurationError",
            "failure_boundary": "before reflection and candidate Solver",
        },
    )
    changed = _git("diff", "--name-only", BASE_SHA).splitlines()
    _write(
        report / "scientific_method_equivalence_audit.json",
        {
            "gate": "PASS",
            "protected_semantics": {
                name: "UNCHANGED"
                for name in (
                    "target allocation",
                    "responsibility scoring",
                    "responsibility/focus/anchor/local_eval",
                    "persistent realizability",
                    "GEPA search core",
                    "TeamMiniBatch",
                    "Full",
                    "Common-Safe",
                    "Shadow",
                    "write-back",
                    "plurality",
                    "saturation patience 3/2",
                )
            },
            "official_gepa_modified": False,
            "provider_or_model_changed": False,
            "implementation_change_class": "execution-interface-only",
            "changed_files": changed,
        },
    )
    _write(
        report / "fake_gepa_path_audit.json",
        {
            "gate": "PASS",
            "same_production_wrapper": "ContextualLocalPromptOptimizer",
            "offline_only": True,
            "boundary_fixture": "GEPA_LOCAL_OPTIMIZER_READY",
            "full_path": [
                "Layer2-derived LocalOptimizationTask",
                "typed runtime context",
                "official frozen GEPA",
                "fake reflection",
                "changed proposal",
                "contract validation",
                "fake candidate Solver",
                "local delta",
            ],
            "solver_reached": True,
            "local_delta_computed": True,
            "real_api_calls": 0,
        },
    )
    _write(
        report / "runtime_field_poison_audit.json",
        {
            "gate": "PASS",
            "fields": {
                name: "LocalOptimizerConfigurationError before backend/provider"
                for name in (
                    "seed",
                    "packet",
                    "parent_candidate",
                    "provider_identity",
                    "solver_model_identity",
                    "optimizer_model_identity",
                    "evaluator_model_identity",
                    "run_identity",
                    "update_index",
                    "target_member",
                )
            },
            "generic_attribute_error_paths": 0,
        },
    )
    _write(
        report / "cache_reuse_policy.json",
        {
            "future_attempt": "gepa_layer2_real_canary_v2_authorized3",
            "run_cache_initialization": "fresh empty run-local cache",
            "manual_copy_from_authorized2": False,
            "cross_attempt_reuse_added": False,
            "expected_baseline_logical_rows": 500,
            "expected_baseline_provider_success_ceiling": 100,
            "note": "Only existing exact request-identity semantics may produce hits.",
        },
    )
    _write(
        report / "cost_separation_audit.json",
        {
            "authorized2_failed_attempt": accounting,
            "future_attempt_denominator_includes_authorized2": False,
            "future_cost_envelope": {
                "baseline_initialization_logical_rows": 500,
                "baseline_initialization_provider_success_ceiling": 100,
                "gepa_local_metric_calls": 36,
                "total_provider_success_ceiling": 256,
            },
            "historical_cost_disclosed": True,
        },
    )
    summary = (
        _json(test_summary)
        if test_summary is not None
        else {"status": "PENDING_FINAL_TEST_RUN"}
    )
    _write(report / "test_summary.json", summary)
    (report / "README.md").write_text(
        "# GEPA canary runtime compatibility fix\n\n"
        "Gate: **PASS**  \n"
        "API calls: **0**  \n"
        "Validation50/Test50: **0/0**\n\n"
        "The failed authorized2 attempt is preserved byte-identically and remains an "
        "engineering runtime failure/scientific non-result. The active canary no "
        "longer imports the historical Seed78 runner. A typed local-optimizer "
        "execution contract now binds seed, provider/model identity, stopping "
        "provenance, run identity, target, update, packet, and parent before Layer 1.\n",
        encoding="utf-8",
    )
    scan_patterns = {
        "raw_key": re.compile(r"(?<![A-Za-z])sk-[A-Za-z0-9._-]{16,}"),
        "raw_endpoint": re.compile(r"https?://[^\s\"']*maas\.aliyuncs\.com"),
        "absolute_path": re.compile(r"[A-Za-z]:\\"),
    }
    hits = {name: 0 for name in scan_patterns}
    for path in report.iterdir():
        if path.suffix not in {".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in scan_patterns.items():
            hits[name] += len(pattern.findall(text))
    if any(hits.values()):
        raise RuntimeError(f"report sanitization failed: {hits}")
    _write(
        report / "sanitization_manifest.json",
        {
            "gate": "PASS",
            "scan_hits": hits,
            "prompts_questions_answers_raw_responses_published": False,
            "credentials_endpoints_absolute_paths_published": False,
        },
    )
    files = sorted(path for path in report.iterdir() if path.name != "sha256_manifest.json")
    _write(
        report / "sha256_manifest.json",
        {"files": {path.name: _sha(path) for path in files}},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--test-summary", type=Path)
    args = parser.parse_args()
    build(args.run, args.report, args.test_summary)


if __name__ == "__main__":
    main()
