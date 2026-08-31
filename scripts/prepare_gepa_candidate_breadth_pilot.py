from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from gepa_candidate_breadth_support import (
    CASES, PROBE_VERSION, ROOT, canonical_hash, read_json, sha256_file, write_json,
)


SOURCE_FILES = (
    "scripts/gepa_candidate_breadth_support.py",
    "scripts/prepare_gepa_candidate_breadth_pilot.py",
    "scripts/run_gepa_candidate_breadth_pilot.py",
    "scripts/audit_gepa_candidate_breadth_pilot.py",
    "scripts/analyze_gepa_candidate_breadth_pilot.py",
    "tests/test_gepa_candidate_breadth_pilot.py",
    "experiments/gepa_candidate_breadth_pilot_20260831/DESIGN_SPEC.md",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def prepare(source_registry: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh preparation root required")
    if git("status", "--porcelain"):
        raise RuntimeError("tracked worktree must be clean before source freeze")
    registry = read_json(source_registry)
    selected = []
    for seed, update in CASES:
        rows = [
            row for row in registry["cases"]
            if int(row["source_seed"]) == seed
            and int(row["source_update_index"]) == update
        ]
        committed = [row for row in rows if bool(row["historically_committed_source"])]
        if len(committed) != 1:
            raise ValueError("each frozen parent requires one committed witness")
        case = dict(committed[0])
        case["case_id"] = f"seed{seed}_update{update}"
        case["breadth_target_counts"] = [2, 4]
        case["requested_source_candidate_count"] = 4
        case["revision_per_valid_source"] = 1
        selected.append(case)
    head = git("rev-parse", "HEAD")
    payload = {
        "registry_version": PROBE_VERSION,
        "execution_commit": head,
        "source_evidence_registry_sha256": sha256_file(source_registry),
        "case_selection_rule": "fixed_v18_harmful_parents_seed59_u3_seed61_u5",
        "case_count": 2,
        "breadths": [2, 4],
        "nested_pool_semantics": "N2_is_first_two_sources_and_their_revisions_within_N4",
        "validation_after_train_decisions_frozen": True,
        "test_enabled": False,
        "team_prompt_commit_enabled": False,
        "trajectory_mutation_enabled": False,
        "cases": selected,
    }
    payload["registry_content_hash"] = canonical_hash(payload)
    out.mkdir(parents=True)
    registry_path = out / "private_registry.json"
    write_json(registry_path, payload)
    files = []
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        files.append({"path": relative, "sha256": sha256_file(path)})
    freeze = {
        "freeze_version": "gepa_candidate_breadth_source_freeze_v1",
        "execution_commit": head,
        "registry_file_sha256": sha256_file(registry_path),
        "registry_content_hash": payload["registry_content_hash"],
        "files": files,
        "source_freeze_status": "PASS",
    }
    write_json(out / "source_freeze.json", freeze)
    gate = {
        "phase_b_preflight": "PASS",
        "case_count": 2,
        "breadths": [2, 4],
        "same_parent_target_peers_prompt_common_safe_ranking": True,
        "only_source_mutation_breadth_changes": True,
        "new_test_calls": 0,
    }
    write_json(out / "phase_b_preflight.json", gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if ROOT.resolve() not in args.out.resolve().parents:
        raise SystemExit("output must be project-local")
    print(prepare(args.source_registry.resolve(), args.out.resolve()))


if __name__ == "__main__":
    main()
