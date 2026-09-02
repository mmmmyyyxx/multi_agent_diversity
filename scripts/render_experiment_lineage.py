"""Validate the experiment DAG and deterministically render its Markdown view."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.registries import (
    load_yaml,
    render_lineage_mermaid,
    validate_lineage,
)


def render(workspace: Path) -> str:
    registry = load_yaml(workspace / "experiments" / "registry.yaml")
    lineage = load_yaml(workspace / "experiments" / "lineage.yaml")
    known = [row["experiment_id"] for row in registry["experiments"]]
    errors, order = validate_lineage(lineage, known)
    if errors:
        raise ValueError("; ".join(errors))
    return (
        "# Experiment Lineage\n\n"
        "Generated from `experiments/lineage.yaml`; edit the YAML authority, not this file.\n\n"
        + render_lineage_mermaid(lineage)
        + "\n\n## Topological order\n\n"
        + "\n".join(f"{index}. `{item}`" for index, item in enumerate(order, 1))
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.workspace.resolve()
    target = root / "docs" / "experiments" / "LINEAGE.md"
    rendered = render(root)
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != rendered:
            raise SystemExit("LINEAGE.md is stale")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
