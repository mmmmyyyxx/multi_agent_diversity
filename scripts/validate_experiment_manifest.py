"""Validate experiment manifests and the compact experiment registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.registries import (
    load_yaml,
    validate_experiment_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve()
    schema = load_yaml(root / "infrastructure" / "experiment_manifest.schema.json")
    registry = load_yaml(root / "experiments" / "registry.yaml")
    errors, manifests = validate_experiment_registry(root, registry, schema)
    if args.manifest:
        from multi_dataset_diverse_rl.governance.manifest import load_manifest, validate_manifest

        manifest = load_manifest(args.manifest)
        errors.extend(validate_manifest(manifest, schema))
    result = {
        "ok": not errors,
        "manifest_count": len(manifests),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
