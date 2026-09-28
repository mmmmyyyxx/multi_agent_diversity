"""Write a deterministic, private diagnostic projection after Formal V3 freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.formal_trajectory import derive_formal_trajectory_records  # noqa: E402
from multi_dataset_diverse_rl.persistence.durable_io import io_path, read_json  # noqa: E402


def derive(run_root: Path) -> dict[str, object]:
    if read_json(run_root / "run_lifecycle.json").get("status") != "EXECUTION_COMPLETE":
        raise ValueError("Formal V3 trajectory requires a completed frozen execution")
    inventory = read_json(run_root / "execution_evidence_freeze.json")
    files = {row["path"]: row for row in inventory.get("files", ())}
    for name in ("execution_summary.json", "run_lifecycle.json", "final_team_materialization.json"):
        if name not in files:
            raise ValueError("Formal V3 trajectory requires frozen execution artifacts")
        path = run_root / name
        raw = path.read_bytes()
        if len(raw) != files[name]["size_bytes"] or hashlib.sha256(raw).hexdigest() != files[name]["sha256"]:
            raise ValueError("Formal V3 frozen execution artifact hash mismatch")
    summary = read_json(run_root / "execution_summary.json")
    rows = derive_formal_trajectory_records(summary)
    destination = run_root / "formal_trajectory_trace.jsonl"
    with open(io_path(destination), "x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False) + "\n")
    return {"mode_id": summary["mode_id"], "record_count": len(rows),
            "derived_after_execution": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    print(json.dumps(derive(parser.parse_args().run_root), sort_keys=True))
