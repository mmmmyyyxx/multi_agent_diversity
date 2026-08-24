from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ORIGINAL_BLOCKERS = {
    "revision_parity:59:HYBRID_BASE",
    "revision_parity:61:HYBRID_BASE",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def artifact_tree_identity(root: Path) -> dict[str, Any]:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return {
        "artifact_count": len(rows),
        "tree_sha256": hashlib.sha256(
            canonical_json(rows).encode("utf-8")
        ).hexdigest(),
    }


def build_admission(
    *,
    run_root: Path,
    original_audit: dict[str, Any],
    semantics_audit: dict[str, Any],
    corrected_gate: dict[str, Any],
    source_freeze: dict[str, Any],
    canonical_report_dir: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    execution = read_json(run_root / "execution_summary.json")
    raw_identity = artifact_tree_identity(run_root)
    recorded_raw = semantics_audit.get("raw_artifact_identity")

    if execution.get("trajectory_count") != 6:
        blockers.append("execution_complete")
    if execution.get("new_test_calls") != 0:
        blockers.append("execution_test_calls")
    if execution.get("infrastructure_failure_count") != 0:
        blockers.append("execution_infrastructure_failure")
    if original_audit.get("gate") != "FAIL":
        blockers.append("original_frozen_gate_status")
    if set(map(str, original_audit.get("blockers", ()))) != EXPECTED_ORIGINAL_BLOCKERS:
        blockers.append("original_frozen_blockers")
    if semantics_audit.get("semantics_audit_gate") != "PASS":
        blockers.append("semantics_audit")
    if semantics_audit.get("semantics_audit_blockers"):
        blockers.append("semantics_integrity")
    if semantics_audit.get("scientific_interpretation_status") != "not_performed":
        blockers.append("gate_tuned_after_scientific_analysis")
    if corrected_gate.get("version") != "post_hoc_corrected_gate_v1":
        blockers.append("corrected_gate_version")
    if corrected_gate.get("gate") != "PASS" or corrected_gate.get("blockers"):
        blockers.append("corrected_gate")
    if corrected_gate.get("replaces_original_gate") is not False:
        blockers.append("original_gate_replacement")
    if corrected_gate.get("original_gate_remains") != "FAIL/HOLD":
        blockers.append("original_gate_history")
    if corrected_gate.get("correction_scope") != "revision_parity_representation_only":
        blockers.append("correction_scope")
    if raw_identity != recorded_raw:
        blockers.append("raw_hash_match")
    if canonical_report_dir.exists():
        blockers.append("scientific_analyzer_previously_run")

    frozen_files = {
        str(row["path"]): str(row["sha256"])
        for row in source_freeze.get("files", ())
    }
    for rel in (
        "scripts/analyze_v18_hybrid_online_accumulation.py",
        "scripts/v18_hybrid_online_accumulation_support.py",
        "experiments/v18_hybrid_online_accumulation_pilot_20260822/classifier_spec.json",
    ):
        if rel not in frozen_files or not (ROOT / rel).is_file():
            blockers.append(f"frozen_scientific_source:{rel}")
        elif sha256_file(ROOT / rel) != frozen_files[rel]:
            blockers.append(f"frozen_scientific_source_hash:{rel}")

    admission_passed = not blockers
    # Preserve the original execution counters expected by the frozen analyzer,
    # but make the provenance of this PASS explicit and non-revisionist.
    result = dict(original_audit)
    result.update({
        "gate": "PASS" if admission_passed else "FAIL",
        "audit_version": "v18_scientific_analysis_admission_v1",
        "scientific_analysis_admitted": admission_passed,
        "admission_blockers": sorted(set(blockers)),
        "original_frozen_audit_status": "FAIL/HOLD",
        "original_frozen_audit_version": original_audit.get("audit_version"),
        "original_frozen_blockers": sorted(EXPECTED_ORIGINAL_BLOCKERS),
        "independent_semantics_audit_status": semantics_audit.get(
            "semantics_audit_gate"
        ),
        "post_hoc_corrected_gate_status": corrected_gate.get("gate"),
        "corrected_gate_version": corrected_gate.get("version"),
        "raw_artifact_identity": raw_identity,
        "raw_hash_match": raw_identity == recorded_raw,
        "scientific_analyzer_not_previously_used_to_tune_gate": (
            semantics_audit.get("scientific_interpretation_status")
            == "not_performed"
            and not canonical_report_dir.exists()
        ),
        "new_api_calls": 0,
        "new_model_calls": 0,
        "new_test_calls": 0,
        "admission_order": [
            "original_frozen_hold_recorded",
            "independent_semantics_audit_passed",
            "post_hoc_corrected_gate_verified",
            "scientific_analysis_admitted",
        ],
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original_audit", type=Path, required=True)
    parser.add_argument("--semantics_audit", type=Path, required=True)
    parser.add_argument("--corrected_gate", type=Path, required=True)
    parser.add_argument("--source_freeze", type=Path, required=True)
    parser.add_argument("--canonical_report_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"fresh admission output required: {args.out}")
    result = build_admission(
        run_root=args.root.resolve(),
        original_audit=read_json(args.original_audit),
        semantics_audit=read_json(args.semantics_audit),
        corrected_gate=read_json(args.corrected_gate),
        source_freeze=read_json(args.source_freeze),
        canonical_report_dir=args.canonical_report_dir.resolve(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "SCIENTIFIC_ANALYSIS_ADMITTED": result["scientific_analysis_admitted"],
        "ORIGINAL_FROZEN_GATE_RECORDED": result["original_frozen_audit_status"],
        "SEMANTICS_AUDIT": result["independent_semantics_audit_status"],
        "POST_HOC_CORRECTED_GATE_V1": result["post_hoc_corrected_gate_status"],
        "RAW_HASH_MATCH": result["raw_hash_match"],
        "NEW_API_CALLS": result["new_api_calls"],
    }, indent=2))
    raise SystemExit(0 if result["scientific_analysis_admitted"] else 1)


if __name__ == "__main__":
    main()
