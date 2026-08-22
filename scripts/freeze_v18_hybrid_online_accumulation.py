from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from scripts.build_v18_hybrid_online_accumulation_registry import ROOT, build_registry
from scripts.preflight_v18_hybrid_online_accumulation import preflight


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_root.resolve()
    if ROOT.resolve() not in out.parents:
        raise SystemExit("freeze output must be project-local")
    if out.exists():
        raise SystemExit("fresh freeze output required")
    if git("status", "--porcelain"):
        raise SystemExit("tracked worktree must be clean before source freeze")
    registry = build_registry()
    gate = preflight(registry)
    if gate["gate"] != "PASS":
        raise SystemExit(json.dumps(gate, indent=2))
    out.mkdir(parents=True)
    registry_path = out / "private_registry.json"
    write_json(registry_path, registry)
    tracked = git(
        "ls-files", "multi_dataset_diverse_rl", "scripts", "tests",
        "experiments/v18_hybrid_online_accumulation_pilot_20260822",
    ).splitlines()
    files = []
    for relative in sorted(tracked):
        path = ROOT / relative
        files.append({
            "path": relative.replace("\\", "/"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    freeze = {
        "freeze_version": "v18_hybrid_online_accumulation_source_freeze_v1",
        "execution_commit": git("rev-parse", "HEAD"),
        "registry_file_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "registry_content_hash": registry["registry_content_hash"],
        "file_count": len(files),
        "files": files,
        "phase_a_gate": gate,
        "source_freeze_status": "PASS",
        "api_calls": 0,
        "model_calls": 0,
    }
    write_json(out / "source_freeze_manifest.json", freeze)
    write_json(out / "phase_a_gate.json", gate)
    print(json.dumps({
        "source_freeze_status": "PASS",
        "execution_commit": freeze["execution_commit"],
        "file_count": len(files),
        "phase_a_gate": gate["gate"],
        "api_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
