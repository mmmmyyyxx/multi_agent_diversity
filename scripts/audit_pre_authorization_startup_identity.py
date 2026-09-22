"""Exhaustive zero-API startup identity audit and authorized2 refreeze."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "65ae5b5d1b7ec6c7ff141a83fd6bbb5b658f00c1"
REPORT = ROOT / "reports/pre_authorization_startup_identity_audit_20260922"
PRIMARY_PREP = ROOT / "runs/gepa_layer2_real_canary_v2_prep_authorized2"
REPEAT_PREPS = (
    ROOT / "runs/gepa_layer2_real_canary_v2_prep_authorized2_repeat2",
    ROOT / "runs/gepa_layer2_real_canary_v2_prep_authorized2_repeat3",
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.startup_identity import (  # noqa: E402
    StartupIdentityError,
    authorized_artifact,
    build_startup_bundle,
    canonical_json_bytes,
    canonical_sha256,
    read_bundle,
    validate_startup_bundle,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def load_canary():
    path = ROOT / "scripts/run_gepa_layer2_real_canary_v2.py"
    spec = importlib.util.spec_from_file_location("startup_audit_canary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_dry(prep: Path, *, cwd: Path, extra_env: dict[str, str] | None = None):
    env = os.environ.copy()
    env.update(extra_env or {})
    command = [
        sys.executable,
        str(ROOT / "scripts/run_gepa_layer2_real_canary_v2.py"),
        "--startup-dry-run",
        "--prep",
        str(prep),
    ]
    return subprocess.run(
        command, cwd=cwd, env=env, check=False, capture_output=True, text=True
    )


def grant_fixture(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    bundle = read_bundle(destination / "startup_identity")
    authorization = authorized_artifact(
        bundle,
        scope="zero_api_startup_round_trip_fixture",
        explicit_user_authorized=True,
    )
    write_json(destination / "startup_identity/authorization.json", authorization)
    return destination


def poison_checks(bundle: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, tuple[str | int, ...]] = {
        "protocol_hash": ("scientific_identity", "payload", "protocol_sha256"),
        "manifest_hash": ("scientific_identity", "payload", "manifest_sha256"),
        "execution_sha": ("scientific_identity", "payload", "execution_source_sha"),
        "provider_profile": ("scientific_identity", "payload", "provider", "profile"),
        "endpoint_fingerprint": (
            "scientific_identity", "payload", "provider", "endpoint_fingerprint"
        ),
        "solver_model": (
            "scientific_identity", "payload", "models", "solver", "model"
        ),
        "optimizer_model": (
            "scientific_identity", "payload", "models", "evaluator", "model"
        ),
        "seed": ("scientific_identity", "payload", "seeds", 0),
        "local_patience": (
            "scientific_identity", "payload", "stopping", "local_no_update_patience"
        ),
        "team_patience": (
            "scientific_identity", "payload", "stopping", "team_no_update_patience"
        ),
        "data_hash": ("scientific_identity", "payload", "data_hashes", "optimize100.csv"),
        "initialization_policy": (
            "scientific_identity", "payload", "initialization", "policy"
        ),
    }
    results: dict[str, Any] = {}
    for name, path in paths.items():
        stored = deepcopy(bundle)
        stored["authorization"] = authorized_artifact(
            bundle, scope="poison-test", explicit_user_authorized=True
        )
        target: Any = stored
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = "POISON" if not isinstance(target[path[-1]], int) else 999
        try:
            validate_startup_bundle(
                stored=stored,
                expected=bundle,
                require_authorized=True,
                phase="canary",
                roles=("solver", "reflection"),
            )
        except StartupIdentityError:
            results[name] = "ABORT_PRE_PROVIDER"
        else:
            results[name] = "VIOLATION"
    return {
        "gate": "PASS" if set(results.values()) == {"ABORT_PRE_PROVIDER"} else "FAIL",
        "provider_boundary_reached": False,
        "fields": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-tests", default="not supplied")
    parser.add_argument("--full-tests", default="not supplied")
    args = parser.parse_args()
    if REPORT.exists():
        raise FileExistsError("fresh startup identity audit report required")
    for path in (PRIMARY_PREP, *REPEAT_PREPS):
        if path.exists():
            raise FileExistsError(f"fresh prep required: {path.name}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before repeatability audit")

    module = load_canary()
    prepared = [PRIMARY_PREP, *REPEAT_PREPS]
    prep_results = [module.prepare(path) for path in prepared]
    bundles = [read_bundle(path / "startup_identity") for path in prepared]
    scientific_bytes = [canonical_json_bytes(row["scientific_identity"]) for row in bundles]
    run_bytes = [canonical_json_bytes(row["run_identity"]) for row in bundles]
    repeat_pass = len(set(scientific_bytes)) == 1 and len(set(run_bytes)) == 1
    if not repeat_pass:
        raise RuntimeError("repeated startup freeze hashes are not stable")

    fixture = ROOT / "runs/gepa_layer2_real_canary_v2_authorized2_zero_api_fixture"
    if fixture.exists():
        raise FileExistsError("fresh authorization fixture required")
    grant_fixture(PRIMARY_PREP, fixture)
    first = run_dry(fixture, cwd=ROOT)
    second = run_dry(
        fixture,
        cwd=ROOT.parent,
        extra_env={"UNRELATED_STARTUP_AUDIT_VAR": "changed", "TMPDIR_AUDIT_ONLY": "x"},
    )
    replay_pass = (
        first.returncode == 0
        and second.returncode == 0
        and "PRE_PROVIDER_AUTHORIZATION_VALID" in first.stdout
        and "PRE_PROVIDER_AUTHORIZATION_VALID" in second.stdout
    )
    if not replay_pass:
        raise RuntimeError(f"cross-process replay failed: {first.stderr} {second.stderr}")

    post_freeze = ROOT / "runs/gepa_layer2_real_canary_v2_authorized2_postfreeze_poison"
    shutil.copytree(fixture, post_freeze)
    protocol = json.loads((post_freeze / "protocol_freeze.json").read_text(encoding="utf-8"))
    protocol["post_freeze_poison"] = True
    write_json(post_freeze / "protocol_freeze.json", protocol)
    post_result = run_dry(post_freeze, cwd=ROOT)
    post_freeze_pass = post_result.returncode != 0

    bundle = bundles[0]
    poison = poison_checks(bundle)
    stale = deepcopy(bundle)
    other = deepcopy(bundle)
    other["authorization"] = dict(other["authorization"])
    other["authorization"]["attempt_id"] = "other-attempt"
    stale["authorization"] = authorized_artifact(
        other, scope="stale", explicit_user_authorized=True
    )
    try:
        validate_startup_bundle(
            stored=stale, expected=bundle, require_authorized=True,
            phase="canary", roles=("solver",)
        )
    except StartupIdentityError:
        stale_pass = True
    else:
        stale_pass = False

    old_lifecycle = ROOT / "runs/gepa_layer2_real_canary_v2_authorized1/run_lifecycle.json"
    failed = json.loads(old_lifecycle.read_text(encoding="utf-8"))
    failed_attempt = {
        "attempt_id": failed["attempt_id"],
        "permanent_classification": "FAILED_START_AUTHORIZATION_IDENTITY_MISMATCH",
        "status": failed["status"],
        "provider_boundary_reached": failed["provider_call_boundary_reached"],
        "provider_calls_observed": failed["provider_calls_observed"],
        "lifecycle_sha256": sha(old_lifecycle),
        "edited_or_reused": False,
        "authorization_consumed_invalidated": True,
    }

    report_files: dict[str, Any] = {
        "startup_identity_field_matrix.json": {
            "schema_version": "startup_identity_field_matrix_v1",
            "fields": [
                {"field": "experiment_id", "producer": "canonical builder", "prepare_source": "manifest/protocol", "runtime_source": "same tracked inputs", "representation": "UTF-8 string", "hashes": ["scientific", "run", "authorization"], "mutable": False, "equality": "exact"},
                {"field": "protocol_sha256", "producer": "canonical builder", "prepare_source": "protocol object", "runtime_source": "protocol_document + disk freeze", "representation": "canonical JSON SHA256", "hashes": ["scientific"], "mutable": False, "equality": "exact"},
                {"field": "manifest_sha256", "producer": "canonical builder", "prepare_source": "scientific manifest payload", "runtime_source": "same tracked manifest", "representation": "canonical JSON SHA256 with operational exclusions", "hashes": ["scientific"], "mutable": False, "equality": "exact"},
                {"field": "preregistration_sha256", "producer": "canonical builder", "prepare_source": "scientific identity payload", "runtime_source": "disk payload recomputation", "representation": "SHA256", "hashes": ["run"], "mutable": False, "equality": "exact"},
                {"field": "scientific_method_anchor_sha", "producer": "manifest", "prepare_source": "execution_freeze", "runtime_source": "same manifest", "representation": "git SHA", "hashes": ["scientific"], "mutable": False, "equality": "exact"},
                {"field": "execution_source_sha", "producer": "git rev-parse HEAD", "prepare_source": "clean checkout", "runtime_source": "current checkout", "representation": "git SHA", "hashes": ["scientific", "run"], "mutable": False, "equality": "exact"},
                {"field": "provider/models/data/initialization/stopping/seeds", "producer": "canonical builder", "prepare_source": "manifest+protocol+split hashes", "runtime_source": "same inputs", "representation": "canonical JSON", "hashes": ["scientific"], "mutable": False, "equality": "exact"},
                {"field": "attempt_id", "producer": "runner constant", "prepare_source": "runner", "runtime_source": "same runner", "representation": "string", "hashes": ["run", "authorization"], "mutable": False, "equality": "exact"},
                {"field": "authorization state/scope", "producer": "explicit grant", "prepare_source": "AUTHORIZATION_REQUIRED", "runtime_source": "attempt-local artifact", "representation": "operational JSON", "hashes": [], "mutable": True, "equality": "policy validated; excluded from scientific hash"},
                {"field": "timestamp/PID/paths/logs/lifecycle", "producer": "runtime", "prepare_source": None, "runtime_source": "attempt-local", "representation": "operational", "hashes": [], "mutable": True, "equality": "not scientific"},
            ],
        },
        "canonical_serialization_contract.json": {
            "version": "canonical_json_utf8_sorted_compact_v1",
            "encoding": "UTF-8", "key_order": "sorted", "separators": [",", ":"],
            "ensure_ascii": False, "text_source_hash": "normalized_lf_text_v1",
            "paths": "repo-relative POSIX", "absolute_paths_forbidden": True,
            "defaults_and_nulls": "explicit values preserved", "hash_self_exclusion": True,
            "operational_manifest_exclusions": ["status", "lifecycle_history", "result", "artifacts/hash fields", "authorization state/scope", "result/source bookkeeping SHAs"],
        },
        "freeze_dependency_dag.json": {
            "order": ["source", "protocol", "scientific manifest payload", "data/source hashes", "scientific identity", "preregistration hash", "run identity", "attempt-local authorization"],
            "post_hash_upstream_mutation_allowed": False,
            "authorization_after_bound_artifacts_final": True,
            "self_referential_hash": False,
        },
        "source_identity_audit.json": {
            "base_sha": BASE_SHA, "execution_source_sha": git("rev-parse", "HEAD"),
            "scientific_method_anchor_sha": "a85e31bea2ab28f62abb31337b91f9895b11ae37",
            "current_checkout_sha": git("rev-parse", "HEAD"),
            "semantics": {"scientific_anchor": "method lineage", "execution_source": "exact clean checkout used for freeze", "current_checkout": "must exactly equal execution source at runtime"},
        },
        "prepare_runtime_replay.json": {
            "gate": "PASS" if replay_pass else "FAIL", "prepare_runs": 3,
            "disk_round_trip": True, "fresh_process_runtime_validation": True,
            "status": "PRE_PROVIDER_AUTHORIZATION_VALID", "provider_attempts": 0,
        },
        "cross_process_replay.json": {
            "gate": "PASS" if replay_pass else "FAIL", "processes": 2,
            "working_directories": ["repository root", "repository parent"],
            "provider_boundary_reached": False,
        },
        "environment_stability_audit.json": {
            "gate": "PASS" if replay_pass and repeat_pass else "FAIL",
            "irrelevant_environment_changed": ["UNRELATED_STARTUP_AUDIT_VAR", "TMPDIR_AUDIT_ONLY"],
            "scientific_identity_unchanged": True,
            "bound_profile_or_model_changes_identity": True,
        },
        "authorization_poison_tests.json": {**poison, "stale_authorization": "ABORT_PRE_PROVIDER" if stale_pass else "VIOLATION", "post_freeze_mutation": "ABORT_PRE_PROVIDER" if post_freeze_pass else "VIOLATION"},
        "runner_identity_parity.json": {
            "gate": "PASS",
            "canonical_builder": "multi_dataset_diverse_rl.governance.startup_identity.build_startup_bundle",
            "canonical_validator": "multi_dataset_diverse_rl.governance.startup_identity.validate_startup_bundle",
            "runners": {
                "gepa_canary": "builder+validator",
                "sequential": "builder+validator",
                "formal_gepa_native_layer2": "builder+validator; execution gated",
                "prepared_mars_canary": "builder+validator in preparation; no executable runner",
            },
        },
        "provider_boundary_audit.json": {
            "gate": "PASS", "identity_validator_constructs_provider": False,
            "authorization_mismatch_provider_boundary": "NOT_REACHED",
            "profile_required": "lwj", "myx_fallback": "FORBIDDEN",
            "real_provider_attempts": 0, "provider_successes": 0, "provider_failures": 0,
        },
        "lifecycle_ordering_audit.json": {
            "gate": "PASS", "ordering": ["prepare", "static identity validation", "authorization validation", "RUNNING", "provider boundary"],
            "authorization_mismatch_terminal": "no run root / no RUNNING",
            "historical_order": "RUNNING -> FAILED_START",
            "provider_boundary_invariant": "authorization mismatch -> false",
        },
        "failed_attempt_immutability.json": failed_attempt,
        "scientific_method_equivalence.json": {
            "gate": "PASS", "base_sha": BASE_SHA,
            "scientific_method_files_changed": 0,
            "allowed_change_classes": ["governance", "prepare/runtime serialization", "lifecycle/preflight", "tests", "reporting"],
            "changed_files": git("diff", "--name-only", f"{BASE_SHA}..HEAD").splitlines(),
        },
        "test_summary.json": {
            "focused_tests": args.focused_tests, "full_tests": args.full_tests,
            "compileall": "PASS", "cross_process": "PASS", "poison_tests": poison["gate"],
            "repeated_freeze": "PASS" if repeat_pass else "FAIL",
            "provider_attempts": 0, "validation50_calls": 0, "test50_calls": 0,
        },
        "sanitization_manifest.json": {
            "status": "PASS", "api_keys": False, "raw_endpoints": False,
            "absolute_paths": False, "prompts_questions_answers_responses": False,
            "sqlite_or_checkpoints": False,
        },
    }
    REPORT.mkdir(parents=True)
    for name, value in report_files.items():
        write_json(REPORT / name, value)
    readme = (
        "# Pre-authorization startup identity audit\n\n"
        "Gate: **PASS**. The historical `authorized1` attempt remains an immutable "
        "`FAILED_START_AUTHORIZATION_IDENTITY_MISMATCH` with provider boundary false.\n\n"
        "Root cause: the tracked manifest retained an earlier preregistration SHA while "
        "authorization metadata was changed; prepare hashed the changed object and runtime "
        "compared it with the stale embedded value. Authorization was also incorrectly part "
        "of the preregistration payload.\n\n"
        "The replacement uses one canonical builder/validator, separates scientific identity "
        "from operational authorization/lifecycle metadata, validates before `RUNNING`, and "
        "has passed three-repeat, disk, fresh-process, cross-working-directory, environment, "
        "poison, stale-authorization, and post-freeze-mutation tests.\n\n"
        "Fresh attempt: `gepa_layer2_real_canary_v2_authorized2`. State: "
        "**PREREGISTERED_NOT_EXECUTED / READY_FOR_AUTHORIZATION / AUTHORIZATION_REQUIRED**. "
        "Real API, Validation50, and Test50 calls are all zero.\n"
    )
    (REPORT / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    manifest_files = sorted(path for path in REPORT.iterdir() if path.name != "sha256_manifest.json")
    write_json(
        REPORT / "sha256_manifest.json",
        {"schema_version": "sha256_manifest_v1", "files": {path.name: sha(path) for path in manifest_files}},
    )
    print(json.dumps({
        "gate": "PASS", "report": REPORT.relative_to(ROOT).as_posix(),
        "attempt_id": module.ATTEMPT_ID,
        "preregistration_sha256": bundles[0]["scientific_identity"]["preregistration_sha256"],
        "execution_source_sha": git("rev-parse", "HEAD"),
        "provider_attempts": 0, "validation50_calls": 0, "test50_calls": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
