"""Check machine-readable CURRENT_SPEC invariants and their references."""

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
    validate_failure_registry,
    validate_invariants,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.workspace.resolve()
    failures = load_yaml(root / "docs" / "failures" / "registry.yaml")
    failure_errors = validate_failure_registry(root, failures)
    ids = [row["failure_id"] for row in failures.get("failures", [])]
    invariants = load_yaml(root / "docs" / "design" / "invariants.yaml")
    invariant_errors = validate_invariants(root, invariants, ids)
    errors = failure_errors + invariant_errors
    active = sum(row.get("status") == "ACTIVE" for row in invariants.get("invariants", []))
    print(json.dumps({"ok": not errors, "active_invariant_count": active, "failure_count": len(ids), "errors": errors}, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
