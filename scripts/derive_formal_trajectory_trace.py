"""Atomically publish a read-only trajectory from verified frozen Formal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.formal_trajectory import derive_formal_trajectory_records  # noqa: E402
from multi_dataset_diverse_rl.persistence.durable_io import (  # noqa: E402
    atomic_replace, canonical_json_payload, io_path, read_json,
)
from scripts.freeze_formal_v3_execution import verify_freeze  # noqa: E402


def derive(run_root: Path) -> dict[str, object]:
    if read_json(run_root / "run_lifecycle.json").get("status") != "EXECUTION_COMPLETE":
        raise ValueError("Formal V3 trajectory requires a completed frozen execution")
    verify_freeze(run_root, allow_derived=True)
    summary = read_json(run_root / "execution_summary.json")
    rows = derive_formal_trajectory_records(summary)
    destination = run_root / "formal_trajectory_trace.jsonl"
    if os.path.lexists(io_path(destination)):
        raise FileExistsError("Formal trajectory already exists")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    intended_hash = hashlib.sha256()
    published = False
    try:
        with open(io_path(temporary), "x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                line = json.dumps(row, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":"), allow_nan=False) + "\n"
                handle.write(line)
                intended_hash.update(line.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(io_path(destination)):
            raise FileExistsError("Formal trajectory already exists")
        atomic_replace(temporary, destination)
        published = True
        observed_hash = hashlib.sha256()
        observed_rows = []
        with open(io_path(destination), "rb") as handle:
            for line in handle:
                observed_hash.update(line)
                observed_rows.append(json.loads(line))
        if (len(observed_rows) != len(rows) or observed_rows != canonical_json_payload(rows)
                or observed_hash.hexdigest() != intended_hash.hexdigest()):
            raise ValueError("Formal trajectory read-back mismatch")
        return {"mode_id": summary["mode_id"], "record_count": len(rows),
                "derived_after_execution": True,
                "trajectory_sha256": observed_hash.hexdigest()}
    except BaseException:
        if published:
            os.unlink(io_path(destination))
        raise
    finally:
        if os.path.lexists(io_path(temporary)):
            os.unlink(io_path(temporary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    print(json.dumps(derive(parser.parse_args().run_root), sort_keys=True))
