from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_v18_safety_only_critic_pilot import prepare as prepare_base
from scripts.v18_safety_only_critic_pilot_support import canonical_hash, read_json, sha256_file, write_json


SOURCE_FILES = (
    "scripts/v18_shadow_raw_critic_support.py",
    "scripts/prepare_v18_shadow_raw_critic_pilot.py",
    "scripts/run_v18_shadow_raw_critic_pilot.py",
    "scripts/audit_v18_shadow_raw_critic_pilot.py",
    "scripts/analyze_v18_shadow_raw_critic_pilot.py",
    "scripts/prepare_v18_safety_only_critic_pilot.py",
    "scripts/run_v18_safety_only_critic_pilot.py",
    "scripts/v18_safety_only_critic_pilot_support.py",
    "tests/test_v18_shadow_raw_critic_pilot.py",
    "experiments/v18_shadow_raw_critic_pilot_20260902/DESIGN_SPEC.md",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def prepare(out: Path) -> dict:
    if out.exists():
        raise FileExistsError("fresh preparation root required")
    if git("status", "--porcelain"):
        raise RuntimeError("tracked worktree must be clean")
    out.mkdir(parents=True)
    base = out / "base_reconstruction"
    base_gate = prepare_base(base)
    if base_gate["gate"] != "PASS":
        raise RuntimeError("base reconstruction failed")
    registry = read_json(base / "private_registry.json")
    registry.update({
        "registry_version": "v18_shadow_raw_critic_pilot_v1",
        "execution_commit": git("rev-parse", "HEAD"),
        "arms": ["canonical_control", "shadow_raw"],
        "sole_intervention": "continue_after_valid_canonical_semantic_rejection",
        "shadow_critic_api_calls": 0,
        "test_enabled": False,
    })
    registry["registry_content_hash"] = canonical_hash({key: value for key, value in registry.items() if key != "registry_content_hash"})
    write_json(out / "private_registry.json", registry)
    freeze = {
        "execution_commit": registry["execution_commit"],
        "registry_sha256": sha256_file(out / "private_registry.json"),
        "files": [{"path": path, "sha256": sha256_file(ROOT / path)} for path in SOURCE_FILES],
    }
    write_json(out / "source_freeze.json", freeze)
    gate = {
        "gate": "PASS",
        "case_count": 6,
        "context_reconstruction_match_count": base_gate["context_reconstruction_match_count"],
        "arms": registry["arms"],
        "api_calls": 0,
        "test_calls": 0,
    }
    write_json(out / "preflight.json", gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if ROOT.resolve() not in args.out.resolve().parents:
        raise SystemExit("project-local output required")
    print(json.dumps(prepare(args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
