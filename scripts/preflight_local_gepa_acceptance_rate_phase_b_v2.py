"""Zero-API audit for the frozen Phase-B preregistration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash, validate_manifest
from multi_dataset_diverse_rl.parent_acquisition import digest, restore_task

IDENTITY = "local_gepa_acceptance_rate_pilot_phase_b_v2"
REPORT = ROOT / "reports" / f"{IDENTITY}_prep_20260918"
MANIFEST = ROOT / "experiments/manifests" / f"{IDENTITY}.yaml"
PRIVATE_BUNDLE = ROOT / "runs" / f"{IDENTITY}_freeze/selected_parent_tasks_private.json"
FORMAL_RUN = ROOT / "runs" / f"{IDENTITY}_attempt1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    errors = []
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "infrastructure/experiment_manifest.schema.json").read_text())
    errors.extend(validate_manifest(manifest, schema))
    if manifest["artifacts"]["preregistration"]["sha256"] != preregistration_hash(manifest):
        errors.append("preregistration hash mismatch")
    freeze = json.loads((REPORT / "PARENT_FREEZE.json").read_text(encoding="utf-8"))
    tasks = json.loads(PRIVATE_BUNDLE.read_text(encoding="utf-8"))
    if sha(PRIVATE_BUNDLE) != freeze["private_selected_task_bundle_sha256"]:
        errors.append("private selected task bundle mismatch")
    selected = freeze["selected_parents"]
    if [row["target_member"] for row in selected] != [1, 2, 3, 4]:
        errors.append("selected members mismatch")
    if len({row["source_state_hash"] for row in selected}) != 1:
        errors.append("parents do not share exactly one source state")
    for row in selected:
        payload = tasks.get(row["parent_task_id"])
        if payload is None or digest(payload) != row["task_payload_sha256"]:
            errors.append(f"task payload mismatch:{row['parent_task_id']}")
            continue
        task = restore_task(payload)
        if (task.seed != row["task_seed"]
                or digest([item.example_id for item in task.search_examples]) != row["search_example_identity_hash"]
                or digest([item.example_id for item in task.local_validation_examples]) != row["local_validation_identity_hash"]):
            errors.append(f"task identity mismatch:{row['parent_task_id']}")
    authorization = manifest["api_authorization"]
    if (authorization["authorized"] is not True
            or authorization["allowed_roles"] != ["solver", "reflection"]
            or authorization["allowed_phases"] != ["phase_b_local_optimizer"]):
        errors.append("Phase-B authorization metadata mismatch")
    if FORMAL_RUN.exists():
        errors.append("formal run root exists")
    if scan_sanitized_artifacts(REPORT):
        errors.append("sanitization findings")
    if json.loads((REPORT / "sha256_manifest.json").read_text()) != build_sha256_manifest(REPORT):
        errors.append("report deterministic hash replay mismatch")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "exact_parent_replay": not any("task" in error or "parent" in error for error in errors),
            "selected_parent_count": len(selected), "shared_state_count": len({r["source_state_hash"] for r in selected}),
            "formal_run_root_absent": not FORMAL_RUN.exists(), "phase_b_api_calls": 0,
            "Validation50_calls": 0, "Test50_calls": 0, "READY_TO_RUN": False,
            "execution_gate": "EXECUTABLE_HANDOFF_REQUIRED"}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "PASS")
