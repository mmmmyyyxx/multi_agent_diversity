from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("multi_dataset_diverse_rl", "scripts", "tests")
DEFINITIONS = (
    "experiments/v16_m2f_online_mechanism_pilot_seed52_20260813/DESIGN_SPEC.md",
    "experiments/v16_m2f_online_mechanism_pilot_seed52_20260813/pilot_preregistration.json",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full_tests", required=True)
    args = parser.parse_args()
    if args.out.exists() or ROOT.resolve() not in args.out.resolve().parents:
        raise SystemExit("fresh repo-local prep output required")
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    files = git("ls-files", *SOURCE_ROOTS).splitlines()
    combined = hashlib.sha256()
    rows = []
    for relative in sorted(files):
        raw = (ROOT / relative).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        normalized = relative.replace("\\", "/")
        combined.update(normalized.encode() + b"\0" + digest.encode() + b"\n")
        rows.append({"path": normalized, "sha256": digest})
    definitions = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in DEFINITIONS
    }
    errors = []
    if dirty:
        errors.append("tracked_or_untracked_worktree_dirty")
    prereg = json.loads((ROOT / DEFINITIONS[1]).read_text(encoding="utf-8"))
    expected = {
        "seed": 52, "updates": 8,
        "setting": "experimental_v16_m2f_online_compatibility_repair",
        "validation_enabled": False, "final_test_enabled": False,
    }
    if any(prereg.get(key) != value for key, value in expected.items()):
        errors.append("preregistration_contract")
    payload = {
        "source_freeze_status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "git_head": head,
        "git_dirty": bool(dirty),
        "working_tree_source_hash": combined.hexdigest(),
        "source_file_count": len(rows),
        "files": rows,
        "definition_sha256": definitions,
        "full_test_result": args.full_tests,
        "seed": 52,
        "planned_updates": 8,
        "validation_calls": 0,
        "test_calls": 0,
        "api_calls": 0,
        "model_calls": 0,
    }
    args.out.mkdir(parents=True)
    (args.out / "source_freeze_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in (
        "source_freeze_status", "errors", "git_head", "source_file_count",
        "seed", "planned_updates", "api_calls", "model_calls",
    )}, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
