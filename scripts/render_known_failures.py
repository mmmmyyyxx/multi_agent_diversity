"""Render the human-readable failure index from the YAML authority."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.registries import load_yaml, validate_failure_registry


def render(workspace: Path) -> str:
    registry = load_yaml(workspace / "docs" / "failures" / "registry.yaml")
    errors = validate_failure_registry(workspace, registry)
    if errors:
        raise ValueError("; ".join(errors))
    lines = [
        "# Known Failures",
        "",
        "Generated from `docs/failures/registry.yaml`; edit the YAML authority, not this file.",
        "",
    ]
    for row in registry["failures"]:
        lines.extend(
            [
                f"## {row['failure_id']}: {row['title']}",
                "",
                f"- Status: `{row['status']}`",
                f"- Evidence level: `{row['evidence_level']}`",
                f"- First observed: `{row['first_observed']}`",
                f"- Root-cause status: {row['root_cause_status']}",
                f"- Symptom: {row['symptom']}",
                f"- Forbidden inference: {row['forbidden_inference']}",
                f"- Mitigation: {row['mitigation']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.workspace.resolve()
    target = root / "docs" / "failures" / "KNOWN_FAILURES.md"
    rendered = render(root)
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != rendered:
            raise SystemExit("KNOWN_FAILURES.md is stale")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
