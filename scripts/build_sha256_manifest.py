"""Build a deterministic relative-path SHA256 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = args.out or args.path / "sha256_manifest.json"
    rendered = json.dumps(build_sha256_manifest(args.path), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != rendered:
            raise SystemExit("SHA256 manifest is stale")
    else:
        target.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
