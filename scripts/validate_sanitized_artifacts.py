"""Validate that publishable artifacts contain no raw or host-local content."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.artifacts import scan_sanitized_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    findings = scan_sanitized_artifacts(args.path)
    result = {"status": "PASS" if not findings else "FAIL", "findings": findings}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not findings else 1)


if __name__ == "__main__":
    main()
