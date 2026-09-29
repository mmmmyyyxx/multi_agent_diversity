"""Zero-provider replay of the frozen aborted Formal Native JSON mismatch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.experiment import EngineEvent  # noqa: E402
from multi_dataset_diverse_rl.persistence.durable_io import canonical_json_payload  # noqa: E402


def _compare(before, after, path: str, changes: list[str]) -> int:
    if isinstance(before, dict) and isinstance(after, dict):
        if before.keys() != after.keys():
            raise AssertionError(f"JSON keys changed at {path}")
        return sum(_compare(before[key], after[key], f"{path}.{key}", changes)
                   for key in before)
    if isinstance(before, (list, tuple)) and isinstance(after, list):
        if len(before) != len(after):
            raise AssertionError(f"JSON array length changed at {path}")
        if isinstance(before, tuple):
            changes.append(path)
        return sum(_compare(left, right, f"{path}[{index}]", changes)
                   for index, (left, right) in enumerate(zip(before, after)))
    if type(before) is not type(after) or before != after:
        raise AssertionError(f"JSON scalar changed at {path}")
    return 1


def replay(summary_path: Path, freeze_path: Path) -> dict[str, object]:
    inventory = json.loads(freeze_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (inventory.get("schema_version") != "formal_v3_aborted_execution_evidence_freeze_v1"
            or inventory.get("attempt_id") != summary.get("experiment_id")
            or summary.get("experiment_id") != "gepa_saturation_comparison_v3_seed80_native_attempt2"):
        raise AssertionError("aborted Formal evidence identity mismatch")
    frozen_summary = next((row for row in inventory["files"]
                           if row["path"] == "execution_summary.json"), None)
    if (frozen_summary is None
            or frozen_summary["sha256"] != hashlib.sha256(summary_path.read_bytes()).hexdigest()
            or frozen_summary["size_bytes"] != summary_path.stat().st_size):
        raise AssertionError("aborted Formal summary is not the frozen raw artifact")
    if summary.get("stop_reason") != "SATURATION_REACHED" or len(summary.get("events", ())) != 1:
        raise AssertionError("aborted Native summary shape mismatch")

    # Rehydrate the typed engine field that was present in the lost in-memory
    # result. Other fields are reconstructed from the exact frozen JSON bytes.
    reconstructed = copy.deepcopy(summary)
    reconstructed["events"] = [EngineEvent(
        **{**row, "candidate_ids": tuple(row["candidate_ids"])}
    ).__dict__ for row in summary["events"]]
    if reconstructed == summary:
        raise AssertionError("historical tuple/list mismatch was not reproduced")
    normalized = canonical_json_payload(reconstructed)
    if normalized != summary:
        raise AssertionError("canonical JSON payload differs from frozen summary")
    changes: list[str] = []
    scalar_count = _compare(reconstructed, normalized, "$", changes)
    if changes != ["$.events[0].candidate_ids"]:
        raise AssertionError("unexpected JSON container normalization")
    return {
        "schema_version": "formal_v3_attempt2_json_roundtrip_replay_v1",
        "attempt_id": summary["experiment_id"],
        "frozen_summary_sha256": frozen_summary["sha256"],
        "raw_python_equal_to_json_readback": False,
        "canonical_json_equal_to_readback": True,
        "container_type_changes": [{"path": path, "before": "tuple", "after": "list"}
                                   for path in changes],
        "unchanged_scalar_count": scalar_count,
        "numeric_string_boolean_and_hash_values_unchanged": True,
        "provider_constructions": 0,
        "provider_calls": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--raw-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.summary, args.raw_freeze)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))
