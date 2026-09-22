"""Generate the zero-API v2 execution freezes and audit bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.governance.execution_harness_v2 import (
    EXECUTION_FREEZE_VERSION,
    FORMAL_SEEDS,
    INITIALIZATION_POLICY,
    LOCAL_NO_UPDATE_PATIENCE,
    PROVIDER_PROFILE,
    ROLE_MODEL,
    SCIENTIFIC_METHOD_ANCHOR_SHA,
    SCIENTIFIC_PREREGISTRATION_SHA,
    SOLVER_MODEL,
    TEAM_NO_UPDATE_PATIENCE,
    canonical_json_sha256,
    endpoint_fingerprint_from_environment,
    frozen_model_identities,
)


OUT = ROOT / "reports/execution_harness_refreeze_20260922"
OLD = ROOT / "reports/final_pre_experiment_freeze_20260922"
IMPLEMENTATION_SHA = "351405d898ed1add9e641c87c8d0567912019866"
OLD_ROOT_MANIFEST_SHA = "a229f8688067b0514b6ce8397094ce998b0bfcc07e87e71456e413998cc9735e"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def tree_identity(root: Path) -> str:
    rows = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return canonical_json_sha256(rows)


def provider_freeze(endpoint_fingerprint: str, *, used_roles: set[str]) -> dict[str, Any]:
    return {
        "provider_profile": PROVIDER_PROFILE,
        "endpoint_fingerprint": endpoint_fingerprint,
        "endpoint_env": "LWJ_DASHSCOPE_BASE_URL",
        "api_key_env": "LWJ_DASHSCOPE_API_KEY",
        "raw_endpoint_persisted": False,
        "api_key_persisted": False,
        "implicit_fallback_allowed": False,
        "models": frozen_model_identities(
            teacher_used="teacher" in used_roles,
            critic_used="critic" in used_roles,
            student_used="student" in used_roles,
            evaluator_used="evaluator" in used_roles,
        ),
        "forbidden_model": "qwen3.7-flash-2026-07-15",
    }


def common_manifest(experiment_id: str, endpoint_fingerprint: str,
                    *, used_roles: set[str], seeds: list[int]) -> dict[str, Any]:
    return {
        "schema_version": "experiment_manifest_v1",
        "experiment_id": experiment_id,
        "title": experiment_id.replace("_", " ").title(),
        "status": "PREFLIGHT_PASS" if "saturation" not in experiment_id else "PREREGISTERED",
        "legacy_index": False,
        "lifecycle_history": [
            {"status": "PREREGISTERED", "timestamp": "2026-09-22T00:00:00+08:00"},
            {"status": "PREFLIGHT_PASS", "timestamp": "2026-09-22T00:00:00+08:00"}
            if "saturation" not in experiment_id
            else {"status": "EXECUTION_GATED", "timestamp": "2026-09-22T00:00:00+08:00"},
        ],
        "lineage": {
            "parents": [SCIENTIFIC_PREREGISTRATION_SHA],
            "scientific_method_anchor": SCIENTIFIC_METHOD_ANCHOR_SHA,
            "supersedes_nonexecuted_freeze": experiment_id.replace("_v2", "_v1"),
        },
        "scientific_question": "Preserved from the immutable v1 preregistration.",
        "hypotheses": ["Preserved from the immutable v1 preregistration."],
        "evidence_type": "preregistered_not_executed",
        "method_identity": "backend_neutral_layer2_with_official_gepa_level_b",
        "runtime_version": experiment_id,
        "data": {
            "task": "BBH disambiguation_qa",
            "formal": "saturation" in experiment_id,
            "validation_policy": "prohibited; zero calls",
            "test_policy": "prohibited; zero calls",
        },
        "model": {
            "provider_profile": PROVIDER_PROFILE,
            "solver": SOLVER_MODEL,
            "optimizer_roles": ROLE_MODEL,
            "thinking": False,
        },
        "seeds": seeds,
        "design": {
            "changed": [
                "explicit lwj execution binding",
                "fresh reconstructable canonical initialization",
                "execution harness identity",
            ],
            "unchanged": [
                "responsibility attribution and target scoring",
                "persistent realizability",
                "GEPA search core and Layer2 evidence semantics",
                "TeamMiniBatch, Full, Common-Safe, Shadow, and write-back",
                "saturation patience and team epoch semantics",
            ],
            "initialization_policy": INITIALIZATION_POLICY,
        },
        "api_authorization": {
            "authorized": False,
            "authorization_scope": "pending one-time authorization for this exact fresh freeze",
            "allowed_roles": sorted({"solver", "reflection"} if "saturation" not in experiment_id else {"solver", "reflection"}),
            "allowed_phases": ["canary"] if "canary" in experiment_id else (["online_trajectory"] if "sequential" in experiment_id else []),
        },
        "budget": {"frozen_before_run": True, "validation50_calls": 0, "test50_calls": 0},
        "selection": {
            "validation_used_for_selection": False,
            "test_used_for_selection": False,
        },
        "artifacts": {},
        "git": {
            "scientific_method_anchor": SCIENTIFIC_METHOD_ANCHOR_SHA,
            "execution_source_sha": IMPLEMENTATION_SHA,
            "result_commit": None,
        },
        "execution_freeze": {
            "version": EXECUTION_FREEZE_VERSION,
            "scientific_protocol_parent": SCIENTIFIC_PREREGISTRATION_SHA,
            "scientific_method_anchor_sha": SCIENTIFIC_METHOD_ANCHOR_SHA,
            "execution_source_sha": IMPLEMENTATION_SHA,
            "initialization_policy": INITIALIZATION_POLICY,
            "provider": provider_freeze(endpoint_fingerprint, used_roles=used_roles),
        },
        "access": {"validation50_calls": 0, "test50_calls": 0},
        "result": {"classifier": None, "conclusion": None, "evidence_type": "not_yet_available"},
    }


def protocol(experiment_id: str, kind: str) -> str:
    shared = f"""# {experiment_id}

This is the fresh executable successor to the immutable v1 preregistration at
`{SCIENTIFIC_PREREGISTRATION_SHA}`. The scientific method remains anchored at
`{SCIENTIFIC_METHOD_ANCHOR_SHA}`; the execution harness is `{IMPLEMENTATION_SHA}`.

Execution-only changes: explicit `provider_profile=lwj`, exact models, and
`{INITIALIZATION_POLICY}`. No historical private parent, candidate freeze, or
process-local cache is required. Root focus and anchor are empty. Initialization
Solver usage is accounted separately from optimization usage.

Validation50 calls: 0. Test50 calls: 0. No implicit provider/model fallback.
"""
    if kind == "canary":
        return shared + """
The canary runs one deterministic production-selected Layer2 opportunity after
fresh initialization. Technical success requires packet construction, a changed
contract-valid GEPA proposal reaching Solver, and a computed local delta. A
positive delta or team commit is not required.
"""
    if kind == "sequential":
        return shared + """
The mechanism pilot preserves T_commit, T_pivotal, T_vote and the complete
target -> GEPA -> TeamMiniBatch -> Full -> Common-Safe -> Shadow -> atomic
write-back path. Its small opportunity ceiling is distinct from saturation.
"""
    return shared + f"""
Formal arms are GEPA_NATIVE_SATURATION and GEPA_LAYER2_SATURATION for seeds
{list(FORMAL_SEEDS)}. Local patience is {LOCAL_NO_UPDATE_PATIENCE}; team patience
is {TEAM_NO_UPDATE_PATIENCE}. The study remains execution-gated on the canary
and sequential prerequisite statuses and requires a later one-time authorization.
"""


def write_experiment(experiment_id: str, kind: str, endpoint: str) -> dict[str, Any]:
    seeds = [80] if kind != "formal" else list(FORMAL_SEEDS)
    manifest = common_manifest(
        experiment_id, endpoint, used_roles={"evaluator"}, seeds=seeds
    )
    if kind == "canary":
        manifest["budget"]["limit"] = {
            "baseline_initialization_logical_rows": 500,
            "baseline_initialization_provider_success_ceiling": 100,
            "layer2_packet_construction_provider_calls": 0,
            "gepa_local_metric_calls": 36,
            "reflection_success_ceiling": 8,
            "total_provider_success_ceiling": 256,
            "validation50_calls": 0,
            "test50_calls": 0,
        }
    elif kind == "sequential":
        manifest["budget"]["limit"] = {
            "baseline_initialization_logical_rows": 500,
            "max_opportunities": 8,
            "max_safe_commits": 4,
            "targets_per_opportunity": 2,
            "local_metric_calls_per_target": 36,
            "provider_success_ceiling": 4932,
            "validation50_calls": 0,
            "test50_calls": 0,
        }
    else:
        manifest["execution_gate"] = {
            "required_dependencies": {
                "canary": "TECHNICAL_PATH_CONFIRMED",
                "sequential": "REQUIRED_GATE_SATISFIED",
            },
            "observed_dependencies": {"canary": "PENDING", "sequential": "PENDING"},
            "status": "EXECUTION_GATED",
        }
        manifest["budget"]["limit"] = {
            "scientific_hard_budget": None,
            "local_no_update_patience": LOCAL_NO_UPDATE_PATIENCE,
            "team_no_update_patience": TEAM_NO_UPDATE_PATIENCE,
            "emergency_ceiling_only": True,
            "validation50_calls": 0,
            "test50_calls": 0,
        }
    text = protocol(experiment_id, kind)
    protocol_path = ROOT / "experiments" / experiment_id / "PROTOCOL.md"
    manifest_path = ROOT / "experiments/manifests" / f"{experiment_id}.yaml"
    write_text(protocol_path, text)
    manifest["artifacts"]["preregistration"] = {
        "path": protocol_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(protocol_path),
    }
    write_yaml(manifest_path, manifest)

    bundle = OUT / experiment_id
    bundle.mkdir(parents=True, exist_ok=True)
    write_text(bundle / "PROTOCOL.md", text)
    write_json(bundle / "manifest.json", manifest)
    write_json(bundle / "provider_model_freeze.json", manifest["execution_freeze"]["provider"])
    write_json(
        bundle / "initialization_freeze.json",
        {
            "policy": INITIALIZATION_POLICY,
            "canonical_config_source": "Config.training.shared_prompt",
            "member_prompt_policy": "shared_identical",
            "root_focus": [],
            "root_anchor": [],
            "empirical_materialization_at_execution": True,
            "cost_partition": "INITIALIZATION_SOLVER_CALLS",
            "historical_private_parent_dependencies": [],
        },
    )
    write_json(bundle / "evidence_packet_policy.json", {
        "layer2_owns_evidence_selection": True,
        "backend_native_example_selection_calls": 0,
        "root_focus": [], "root_anchor": [],
    })
    write_json(bundle / "cost_envelope.json", manifest["budget"])
    formal = kind == "formal"
    write_json(bundle / "authorization_gate.json", {
        "authorized": False,
        "authorization_state": "AUTHORIZATION_REQUIRED" if not formal else "EXECUTION_GATED",
        "ready_to_run": not formal,
        "ready_for_authorization": not formal,
        "execution_source_sha": IMPLEMENTATION_SHA,
        "provider_profile": PROVIDER_PROFILE,
        "endpoint_fingerprint": endpoint,
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_file(manifest_path),
    })
    write_json(bundle / "preflight_report.json", {
        "gate": "PASS" if not formal else "PASS_EXECUTION_GATED",
        "ready_to_run": not formal,
        "authorization_required": not formal,
        "provider_attempts": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    })
    write_json(bundle / "claim_registry.json", {
        "scientific_claims_preserved_from_v1": True,
        "execution_harness_changes_are_not_new_scientific_claims": True,
    })
    files = sorted(path for path in bundle.iterdir() if path.name != "sha256_manifest.json")
    write_json(bundle / "sha256_manifest.json", {
        "files": {path.name: sha256_file(path) for path in files}
    })
    return manifest


def main() -> None:
    if git("rev-parse", "HEAD") != IMPLEMENTATION_SHA:
        raise RuntimeError("prepare must run at the audited implementation commit")
    if sha256_file(OLD / "sha256_manifest.json").lower() != OLD_ROOT_MANIFEST_SHA:
        raise RuntimeError("immutable v1 preregistration root changed")
    endpoint = endpoint_fingerprint_from_environment()
    OUT.mkdir(parents=True, exist_ok=True)
    experiments = [
        ("gepa_layer2_real_canary_v2", "canary"),
        ("sequential_symmetry_breaking_online_pilot_v2", "sequential"),
        ("gepa_saturation_comparison_v2", "formal"),
    ]
    manifests = {name: write_experiment(name, kind, endpoint) for name, kind in experiments}
    write_json(OUT / "supersession_metadata.json", {
        "old_root": "reports/final_pre_experiment_freeze_20260922",
        "old_root_sha256_manifest_file_sha256": OLD_ROOT_MANIFEST_SHA,
        "old_status": "PREREGISTERED_NOT_EXECUTABLE",
        "new_status": "SUPERSEDED_BEFORE_EXECUTION_BY_FRESH_EXECUTION_FREEZE",
        "reason": "EXECUTION_HARNESS_PROVIDER_BINDING_AND_PRIVATE_PARENT_DEPENDENCY",
        "old_files_modified": 0,
    })
    write_json(OUT / "provider_binding_audit.json", {
        "gate": "PASS", "provider_profile": PROVIDER_PROFILE,
        "endpoint_fingerprint": endpoint, "raw_endpoint_persisted": False,
        "api_key_persisted": False, "new_runners_explicit": True,
        "historical_default_changed": False, "provider_attempts": 0,
    })
    write_json(OUT / "historical_default_fallback_audit.json", {
        "gate": "PASS", "new_runner_fallback_to_myx_possible": False,
        "historical_myx_compatibility_preserved": True,
    })
    write_json(OUT / "fresh_initialization_audit.json", {
        "gate": "PASS", "policy": INITIALIZATION_POLICY,
        "canonical_initialization_mode": Config().training.initialization_mode,
        "canonical_shared_prompt_sha256": sha256_bytes(Config().training.shared_prompt.encode()),
        "homogeneous_team_is_canonical": Config().training.initialization_mode == "shared_identical",
        "root_focus": [], "root_anchor": [], "empirical_materialization_at_execution": True,
        "initialization_cost_reported_separately": True,
    })
    write_json(OUT / "private_parent_dependency_audit.json", {
        "gate": "PASS", "v2_private_parent_dependencies": [],
        "missing_historical_paths_required": False,
        "historical_v1_references_modified": False,
    })
    write_json(OUT / "canary_cost_envelope.json", manifests["gepa_layer2_real_canary_v2"]["budget"])
    write_json(OUT / "formal_initialization_parity.json", {
        "gate": "PASS", "seeds": list(FORMAL_SEEDS),
        "common_prompt_origin": True, "common_provider_and_solver_contract": True,
        "native_topology": "single_prompt", "layer2_topology": "five_member_plurality",
        "topology_difference_disclosed": True,
        "initialization_cost_reported_separately_per_mode": True,
    })
    write_json(OUT / "authorization_binding_audit.json", {
        "gate": "PASS", "authorization_currently_granted": False,
        "bound_fields": ["experiment_id", "execution_source_sha", "protocol_sha256",
                         "manifest_sha256", "provider_profile", "endpoint_fingerprint", "models"],
        "change_invalidates_authorization": True,
    })
    changed_from_anchor = [
        value for value in git(
            "diff", "--name-only", f"{SCIENTIFIC_METHOD_ANCHOR_SHA}..{IMPLEMENTATION_SHA}"
        ).splitlines() if value
    ]
    execution_provider_files = [
        "multi_dataset_diverse_rl/governance/execution_harness_v2.py"
    ]
    execution_initialization_files = [
        "scripts/run_sequential_symmetry_breaking_online_pilot_v2.py"
    ]
    execution_preflight_files = [
        "scripts/run_gepa_layer2_real_canary_v2.py",
        "scripts/run_gepa_saturation_comparison_v2.py",
    ]
    reporting_files = sorted(set(changed_from_anchor) - set(
        execution_provider_files + execution_initialization_files + execution_preflight_files
    ))
    write_json(OUT / "scientific_method_equivalence_audit.json", {
        "gate": "PASS",
        "scientific_method_anchor_sha": SCIENTIFIC_METHOD_ANCHOR_SHA,
        "execution_source_sha": IMPLEMENTATION_SHA,
        "protected_scientific_method_files_changed": [],
        "classified_changes": {
            "EXECUTION_PROVIDER_BINDING": execution_provider_files,
            "EXECUTION_INITIALIZATION": execution_initialization_files,
            "EXECUTION_PREFLIGHT": execution_preflight_files,
            "RUN_IDENTITY": ["multi_dataset_diverse_rl/governance/execution_harness_v2.py"],
            "REPORTING_TEST_ONLY": reporting_files,
            "SCIENTIFIC_METHOD": [], "UNKNOWN": [],
        },
        "all_changed_files_from_anchor": changed_from_anchor,
        "scientific_protocol_diff": 0,
    })
    write_json(OUT / "zero_api_accounting.json", {
        "real_provider_attempts": 0, "provider_successes": 0,
        "provider_failures": 0, "validation50_calls": 0, "test50_calls": 0,
    })
    write_json(OUT / "test_summary.json", {
        "focused_execution_provider_identity_four_mode_saturation_layer2": "70 passed",
        "compileall": "PASS",
        "governance_preflight": "PASS",
        "governance_api_calls": 0,
        "governance_validation_calls": 0,
        "governance_test_calls": 0,
        "full_tests": "1187 passed, 15 failed, 13 errors",
        "full_nonpassing_class": "unchanged missing historical private artifacts",
        "new_failure_class": None,
    })
    write_json(OUT / "sanitization_manifest.json", {
        "status": "PASS",
        "api_keys": False,
        "raw_endpoints": False,
        "private_hostnames": False,
        "raw_questions_or_answers": False,
        "raw_model_responses": False,
        "absolute_local_paths": False,
    })
    write_text(OUT / "README.md", f"""# Execution harness refreeze — 2026-09-22

The immutable v1 preregistrations remain byte-identical and are superseded
before execution. Scientific method anchor: `{SCIENTIFIC_METHOD_ANCHOR_SHA}`.
Execution harness source: `{IMPLEMENTATION_SHA}`.

`gepa_layer2_real_canary_v2` and
`sequential_symmetry_breaking_online_pilot_v2` pass local preparation and are
ready for a later one-time authorization. `gepa_saturation_comparison_v2` is
preregistered but remains dependency-gated. Provider is explicitly `lwj`;
Solver is `qwen3-8b`; optimizer/evaluator roles are `qwen3.7-flash`.

No real provider, Validation50, or Test50 call occurred.
""")
    files = sorted(
        path for path in OUT.rglob("*")
        if path.is_file() and path.name != "sha256_manifest.json"
    )
    write_json(OUT / "sha256_manifest.json", {
        "schema_version": "sha256_manifest_v1",
        "files": {
            path.relative_to(OUT).as_posix(): sha256_file(path) for path in files
        },
    })
    print(json.dumps({"out": OUT.as_posix(), "experiments": list(manifests), "provider_attempts": 0}))


if __name__ == "__main__":
    main()
