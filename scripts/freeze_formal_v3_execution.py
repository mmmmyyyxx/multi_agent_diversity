"""Freeze a completed Formal search cell before deriving read-only trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.persistence.durable_io import read_json  # noqa: E402


def freeze(run_root: Path) -> dict[str, object]:
    if read_json(run_root / "run_lifecycle.json").get("status") != "EXECUTION_COMPLETE":
        raise ValueError("Formal execution must complete before raw freeze")
    summary = read_json(run_root / "execution_summary.json")
    if (summary.get("validation50_calls") != 0 or summary.get("test50_calls") != 0
            or summary.get("mode_id") not in {"GEPA_NATIVE", "GEPA_LAYER2_V4"}):
        raise ValueError("Formal execution summary is not search-only")
    final_team = read_json(run_root / "final_team_materialization.json")
    if (final_team.get("mode_id") != summary["mode_id"]
            or final_team.get("initial_team_hash") != summary["initial_team_hash"]
            or final_team.get("final_team_hash")
            != summary["final_team_materialization"]["final_team_hash"]):
        raise ValueError("Formal final-team materialization mismatch")
    destination = run_root / "execution_evidence_freeze.json"
    if destination.exists():
        raise FileExistsError("Formal execution is already frozen")
    files = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path == destination:
            continue
        if path.is_symlink():
            raise ValueError("Formal raw freeze cannot contain symlinks")
        raw = path.read_bytes()
        files.append({"path": path.relative_to(run_root).as_posix(),
                      "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    payload = {"schema_version": "formal_v3_execution_evidence_freeze_v1",
               "attempt_id": summary["experiment_id"], "files": files,
               "artifact_count": len(files)}
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return {"attempt_id": payload["attempt_id"], "artifact_count": len(files),
            "freeze_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    print(json.dumps(freeze(parser.parse_args().run_root), sort_keys=True))
