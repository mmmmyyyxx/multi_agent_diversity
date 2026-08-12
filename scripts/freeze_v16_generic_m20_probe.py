from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generic_m20_probe_support import (
    FROZEN_DEFINITION_SHA256,
    source_manifest,
    tracked_source_dirty,
)
from multi_dataset_diverse_rl.versions import CHECKPOINT_VERSION, METHOD_VERSION
from run_v16_generic_m20_fixed_parent_probe import (
    canonical_registry_hash,
    validate_registry_contract,
)


DEFINITION_FILES = (
    "DESIGN_SPEC.md",
    "analysis_spec.json",
    "probe_preregistration.json",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_freeze(
    *,
    registry_path: Path,
    prep_root: Path,
    preflight_path: Path,
    offline_verification_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    offline = json.loads(offline_verification_path.read_text(encoding="utf-8"))
    execution_commit = git_head()
    errors = validate_registry_contract(registry)
    if registry.get("execution_commit") != execution_commit:
        errors.append("registry_execution_commit")
    dirty = tracked_source_dirty()
    if dirty:
        errors.append("tracked_source_dirty")
    if preflight.get("status") != "PASS":
        errors.append("semantic_preflight")
    if int(preflight.get("api_calls", -1)) != 0 or int(
        preflight.get("model_calls", -1)
    ) != 0:
        errors.append("preflight_provider_calls")
    if offline.get("status") != "PASS":
        errors.append("offline_verification")
    if any(
        int(offline.get(key, -1)) != 0
        for key in ("api_calls", "model_calls", "validation_calls", "test_calls")
    ):
        errors.append("offline_verification_calls")
    definition_hashes = {}
    for name in DEFINITION_FILES:
        path = prep_root / name
        if not path.is_file():
            errors.append(f"definition_missing:{name}")
        else:
            definition_hashes[name] = file_sha256(path)
            if definition_hashes[name] != FROZEN_DEFINITION_SHA256[name]:
                errors.append(f"definition_hash:{name}")
    manifest = source_manifest()
    freeze = {
        "freeze_version": "v16_generic_m20_source_freeze_v1",
        "source_freeze_status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "execution_commit": execution_commit,
        "canonical_method_version": METHOD_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "repo_dirty": bool(dirty),
        "tracked_source_dirty_entries": dirty,
        "registry_file_sha256": file_sha256(registry_path),
        "registry_content_hash": registry["registry_content_hash"],
        "registry_content_hash_recomputed": canonical_registry_hash(registry),
        "case_ids": [str(row["case_id"]) for row in registry["cases"]],
        "cell_order": [
            {
                "case_id": str(row["case_id"]),
                "variants": list(row["cell_order"]),
            }
            for row in registry["cases"]
        ],
        "case_count": 8,
        "cell_count": 16,
        "candidate_count": 32,
        "frozen_definition_sha256": definition_hashes,
        "offline_test_count": int(offline.get("test_count", 0)),
        "offline_verification_status": offline.get("status"),
        "semantic_preflight_status": preflight.get("status"),
        **manifest,
    }
    semantics = {
        "semantics_diff_version": "v16_generic_m20_semantics_diff_v1",
        "intentional_differences": [
            "G0 uses current generic AccuracyDiagnosisContext and generic rendering",
            "M20 uses byte-current v15 SingleLaneDiagnosisContext and rendering",
        ],
        "shared_probe_controls": [
            "fixed parent prompts and profiles",
            "fixed target",
            "qwen3-14b with thinking disabled",
            "Teacher/Critic/Student ceilings and two-candidate budget",
            "M20 member-aware Stage A/B pool",
            "fixed-peer common-safe acceptance and common ranking",
        ],
        "g0_generation_responsibility_hash_count": 0,
        "m20_context_semantics_changed": False,
        "module1_semantics_changed": False,
        "common_safe_semantics_changed": False,
        "canonical_method_promoted": False,
        "checkpoint_bumped": False,
    }
    verification = {
        "verification_version": "v16_generic_m20_pre_probe_verification_v1",
        "status": freeze["source_freeze_status"],
        "execution_commit": execution_commit,
        "source_freeze_status": freeze["source_freeze_status"],
        "registry_content_hash": registry["registry_content_hash"],
        "registry_file_sha256": freeze["registry_file_sha256"],
        "case_count": 8,
        "cell_count": 16,
        "candidate_count_per_cell": 2,
        "maximum_planned_candidates": 32,
        "semantic_preflight_status": preflight.get("status"),
        "g0_responsibility_leakage_count": preflight.get(
            "g0_responsibility_leakage_count"
        ),
        "m20_context_semantics_changed": False,
        "commit_enabled": False,
        "parent_mutation_enabled": False,
        "validation_enabled": False,
        "final_test_enabled": False,
        "api_calls": 0,
        "model_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "offline_test_count": int(offline.get("test_count", 0)),
    }
    return freeze, verification, semantics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--prep_root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--offline_verification", type=Path, required=True)
    args = parser.parse_args()
    prep = args.prep_root.resolve()
    if ROOT.resolve() not in prep.parents:
        raise SystemExit("prep root must remain under the repository root")
    freeze, verification, semantics = build_freeze(
        registry_path=args.registry.resolve(),
        prep_root=prep,
        preflight_path=args.preflight.resolve(),
        offline_verification_path=args.offline_verification.resolve(),
    )
    atomic_write(prep / "source_freeze_manifest.json", freeze)
    atomic_write(prep / "pre_probe_verification.json", verification)
    atomic_write(prep / "runtime_semantics_diff.json", semantics)
    print(json.dumps({
        "status": verification["status"],
        "execution_commit": verification["execution_commit"],
        "source_file_count": freeze["source_file_count"],
        "test_count": verification["offline_test_count"],
        "api_calls": 0,
    }, indent=2))
    if verification["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
