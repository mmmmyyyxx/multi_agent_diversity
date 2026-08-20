from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v17_module1_2x2_support import CELLS, classify


def analyze(root: Path) -> dict:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/*/cell_result.json"))
    ]
    if len(rows) != 24:
        raise ValueError("analysis requires exactly 24 complete cells")
    aggregate = {
        cell: {
            "vote": sum(
                int(row["realized_validation_vote_delta"])
                for row in rows if row["cell"] == cell
            ),
            "oracle": sum(
                int(row["realized_validation_oracle_delta"])
                for row in rows if row["cell"] == cell
            ),
        }
        for cell in CELLS
    }
    return {
        "analysis_version": "v17_module1_2x2_realized_validation_v1",
        "case_count": 6, "cell_count": 24,
        "aggregate_realized_deltas": aggregate,
        **classify(aggregate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
