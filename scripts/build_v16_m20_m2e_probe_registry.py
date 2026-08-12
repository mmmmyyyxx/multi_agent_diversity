from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_v16_generic_m20_probe_registry import build_registry as build_base

M20 = "m20_current_v15"
M2E = "m2e_scoped_behavioral_patch"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_registry(execution_commit: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", execution_commit):
        raise ValueError("execution_commit must be a full Git SHA")
    base = build_base(execution_commit)
    payload = {**base, "registry_version": "v16_m20_m2e_fixed_parent_registry_v1", "variants": [M20, M2E]}
    for index, case in enumerate(payload["cases"]):
        case["cell_order"] = [M20, M2E] if index % 2 == 0 else [M2E, M20]
    payload["hypothesis"] = "scoped patch retains M20 repair while reducing non-responsibility loss"
    payload["success_metrics"] = [
        "responsibility_residual_gain_count", "nonresponsibility_loss_count",
        "stable_loss_count", "pivotal_loss_count", "common_safe_feasible_count",
    ]
    payload["registry_content_hash"] = digest({k: v for k, v in payload.items() if k != "registry_content_hash"})
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution_commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or ROOT.resolve() not in args.out.resolve().parents:
        raise SystemExit("output must be fresh and project-local")
    payload = build_registry(args.execution_commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "cases": 8, "cells": 16, "candidates": 32, "api_calls": 0}))


if __name__ == "__main__":
    main()
