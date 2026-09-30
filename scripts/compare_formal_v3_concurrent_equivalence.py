"""Compare private old/new Formal fake captures without publishing examples."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


OLD_SOURCE = "9e498496abff9228bd10697d8d97d36f56fca3f9"
GEPA_FILES = ("candidates.json", "run_log.json", "gepa_state.bin")


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_summary(value: dict) -> tuple[dict, int]:
    result = deepcopy(value)
    if result["stop_reason"] not in {"SATURATION_REACHED", "NO_FEASIBLE_LAYER2_OPPORTUNITY"}:
        raise ValueError("equivalence capture did not reach a scientific stop")
    removed = 0

    def strip_timing(node):
        nonlocal removed
        if isinstance(node, dict):
            if "wall_seconds" in node:
                del node["wall_seconds"]
                removed += 1
            for child in node.values():
                strip_timing(child)
        elif isinstance(node, list):
            for child in node:
                strip_timing(child)

    for event in result["events"]:
        strip_timing(event["telemetry"])
    return result, removed


def _gepa_artifacts(root: Path, case: str) -> dict[str, str]:
    local = root / case / "run/local_gepa"
    result = {}
    for name in GEPA_FILES:
        for path in sorted(local.rglob(name)):
            relative = path.relative_to(local).as_posix()
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(local.rglob("iter_*.json")):
        relative = path.relative_to(local).as_posix()
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not result:
        raise ValueError("missing official GEPA fake artifacts")
    return result


def compare(old_root: Path, new_root: Path) -> dict:
    old = json.loads((old_root / "private_semantic_capture.json").read_text(encoding="utf-8"))
    new = json.loads((new_root / "private_semantic_capture.json").read_text(encoding="utf-8"))
    if old["source_sha"] != OLD_SOURCE or old["source_sha"] == new["source_sha"]:
        raise ValueError("old/new source identity mismatch")
    if set(old["cases"]) != set(new["cases"]):
        raise ValueError("old/new fake case inventory mismatch")
    cases = {}
    for name in sorted(old["cases"]):
        before, after = old["cases"][name], new["cases"][name]
        old_summary, old_timing = _normalize_summary(before["summary"])
        new_summary, new_timing = _normalize_summary(after["summary"])
        old_artifacts = _gepa_artifacts(old_root, name)
        new_artifacts = _gepa_artifacts(new_root, name)
        checks = {
            "complete_scientific_summary": old_summary == new_summary,
            "ledger_semantic_multiset": before["ledger_semantic_multiset"] == after["ledger_semantic_multiset"],
            "lineage": before["lineage"] == after["lineage"],
            "official_gepa_artifacts": old_artifacts == new_artifacts,
            "fake_physical_calls": before["fake_physical_calls"] == after["fake_physical_calls"],
            "optimize_question_ids": before["optimize_question_ids"] == after["optimize_question_ids"],
            "shadow_question_ids": before["shadow_question_ids"] == after["shadow_question_ids"],
            "reflection_input_identities": before["reflection_request_sha256"] == after["reflection_request_sha256"],
            "reflection_outputs": before["reflection_response_sha256"] == after["reflection_response_sha256"],
            "heldout_zero": all(c["summary"]["validation50_calls"] == c["summary"]["test50_calls"] == 0
                                for c in (before, after)),
        }
        cases[name] = {
            "checks": checks,
            "normalized_summary_sha256": _digest(old_summary),
            "normalized_ledger_sha256": _digest(before["ledger_semantic_multiset"]),
            "lineage_sha256": _digest(before["lineage"]),
            "official_gepa_artifact_count": len(old_artifacts),
            "official_gepa_artifact_hashes_sha256": _digest(old_artifacts),
            "optimize_question_ids_sha256": _digest(before["optimize_question_ids"]),
            "shadow_question_ids_sha256": _digest(before["shadow_question_ids"]),
            "reflection_input_sequence_sha256": _digest(before["reflection_request_sha256"]),
            "reflection_output_sequence_sha256": _digest(before["reflection_response_sha256"]),
            "reflection_call_count": len(before["reflection_request_sha256"]),
            "old_wall_seconds_fields": old_timing,
            "new_wall_seconds_fields": new_timing,
            "old_successful_provider_calls": before["summary"]["ledger"]["successful_provider_calls"],
            "new_successful_provider_calls": after["summary"]["ledger"]["successful_provider_calls"],
            "stop_reason": before["summary"]["stop_reason"],
        }
    passed = all(all(row["checks"].values()) for row in cases.values())
    return {
        "schema_version": "formal_v3_attempt4_deterministic_semantic_equivalence_v1",
        "old_execution_source_sha": old["source_sha"],
        "new_execution_source_sha": new["source_sha"],
        "case_count": len(cases),
        "allowed_normalization": "event_telemetry.wall_seconds_only",
        "cases": cases,
        "SEMANTIC_EQUIVALENCE": "PASS" if passed else "FAIL",
        "real_provider_calls": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("semantic equivalence report already exists")
    result = compare(args.old, args.new)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"SEMANTIC_EQUIVALENCE": result["SEMANTIC_EQUIVALENCE"],
                      "case_count": result["case_count"]}, sort_keys=True))
    if result["SEMANTIC_EQUIVALENCE"] != "PASS":
        raise SystemExit(1)
