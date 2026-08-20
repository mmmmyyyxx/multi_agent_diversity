from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_v17_module1_2x2_registry import build_registry
from preflight_v17_module1_2x2 import preflight


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_root.resolve()
    if ROOT.resolve() not in out.parents or out.exists():
        raise SystemExit("freeze root must be fresh and project-local")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("tracked worktree must be clean")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    out.mkdir(parents=True)
    registry = build_registry(head)
    registry_path = out / "private_registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    result = preflight(registry, out / "preflight_scratch")
    (out / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "PASS":
        raise SystemExit("Phase A preflight failed")
    definitions = ROOT / "experiments/v17_module1_2x2_causal_isolation_20260820"
    tracked = subprocess.check_output(
        ["git", "ls-files", "multi_dataset_diverse_rl", "scripts", "tests"],
        cwd=ROOT, text=True,
    ).splitlines()
    files = [{"path": name.replace("\\", "/"), "sha256": sha(ROOT / name)} for name in sorted(tracked)]
    manifest = {
        "freeze_version": "v17_module1_2x2_source_freeze_v1",
        "execution_commit": head,
        "registry_content_hash": registry["registry_content_hash"],
        "registry_file_sha256": sha(registry_path),
        "definition_sha256": {
            path.name: sha(path) for path in sorted(definitions.iterdir()) if path.is_file()
        },
        "source_file_count": len(files), "files": files,
        "phase_a_api_calls": 0, "validation_calls": 0, "test_calls": 0,
        "source_freeze_status": "PASS",
    }
    (out / "source_freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "execution_commit": head,
        "registry_content_hash": registry["registry_content_hash"],
        "case_count": 6, "cell_count": 24,
        "api_calls": 0, "validation_calls": 0, "test_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
