"""Reflection-evidence-fixed successor to the Level-B GEPA path canary."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_level_b_gepa_real_canary as base  # noqa: E402


base.EXPERIMENT_ID = "level_b_gepa_real_canary_v2"
base.ATTEMPT_ID = "level_b_gepa_real_canary_v2_stagefix2_governance_refreeze1"
base.MANIFEST = ROOT / "experiments/manifests/level_b_gepa_real_canary_v2.yaml"
base.PROTOCOL = ROOT / "experiments/level_b_gepa_real_canary_v2/PROTOCOL.md"
base.DEFAULT_PREP = ROOT / "runs/level_b_gepa_real_canary_v2_prep_stagefix2_governance_refreeze1"
base.DEFAULT_RUN = ROOT / "runs/level_b_gepa_real_canary_v2_stagefix2_governance_refreeze1"
base.DEFAULT_REPORT = ROOT / "reports/level_b_gepa_real_canary_v2_stagefix2_governance_refreeze1"
base.AUTH_ENV = "LEVEL_B_GEPA_REAL_CANARY_V2_AUTHORIZED"

CLASSIFIER_VERSION = "level_b_local_empirical_path_classifier_v1"
SPLIT_MANIFEST = ROOT / "experiments/anti_overfitting_split_v1/split_manifest.json"
FOLD_ASSIGNMENT = ROOT / "experiments/anti_overfitting_split_v1/fold_assignment.json"

_base_source_paths = base.source_paths
_base_protocol_document = base.protocol_document
_base_prepare = base.prepare
_base_verify_freeze = base.verify_freeze
_base_preflight = base.preflight


def _logical_hash(values: list[str]) -> str:
    payload = json.dumps(sorted(values), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def logical_split_identities() -> dict[str, dict[str, object]]:
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    folds = json.loads(FOLD_ASSIGNMENT.read_text(encoding="utf-8"))["folds"]
    values = {
        "optimize100": list(folds["fold_a"]) + list(folds["fold_b"]),
        "shadow50": list(folds["fold_c"]),
        "validation50": list(manifest["question_hashes"]["validation"]),
        "test50": list(manifest["question_hashes"]["test"]),
    }
    return {
        name: {"count": len(hashes), "question_hashes_sha256": _logical_hash(hashes)}
        for name, hashes in values.items()
    }


def classify(telemetry: dict[str, object], _outcome: object) -> str:
    diagnostics = telemetry.get("proposer_diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    attempts = int(diagnostics.get("proposal_attempts", telemetry.get("proposal_attempts", 0)))
    solver_reached = int(diagnostics.get("solver_reached", 0))
    if attempts == 0:
        return "NO_REAL_PROPOSAL_ATTEMPT"
    if solver_reached == 0:
        return "PROPOSAL_CONTRACT_STILL_BLOCKS_EMPIRICAL_SEARCH"
    return "LOCAL_EMPIRICAL_PATH_CONFIRMED"


def source_paths() -> list[Path]:
    return sorted(
        set(_base_source_paths())
        | {Path("scripts/run_level_b_gepa_real_canary_v2.py")},
        key=lambda path: path.as_posix(),
    )


def protocol_document():
    protocol = _base_protocol_document()
    protocol.update(
        {
            "schema_version": "level_b_gepa_real_canary_protocol_v5",
            "reflection_input_repair": (
                "component_specific_reasoning_evidence_v1"
            ),
            "minimum_technical_success": (
                "contract_valid_changed_proposal_reaches_solver_gt_zero"
            ),
            "classifier_version": CLASSIFIER_VERSION,
            "solver_stage_attribution_contract": "explicit_solver_phase_v1",
            "logical_split_identities": logical_split_identities(),
        }
    )
    return protocol


def _prepared_split_identities(prep: Path) -> dict[str, dict[str, object]]:
    mapping = {
        "optimize100": "optimize100.csv",
        "shadow50": "fold_c.csv",
        "validation50": "validation.csv",
        "test50": "test.csv",
    }
    result: dict[str, dict[str, object]] = {}
    for name, filename in mapping.items():
        rows = base._rows(prep / "splits_private" / filename)
        hashes = [hashlib.sha256(row["question"].encode("utf-8")).hexdigest() for row in rows]
        result[name] = {"count": len(hashes), "question_hashes_sha256": _logical_hash(hashes)}
    return result


def prepare(prep: Path):
    result = _base_prepare(prep)
    observed = _prepared_split_identities(prep)
    expected = logical_split_identities()
    if observed != expected:
        raise RuntimeError("prepared logical split identity mismatch")
    freeze = base.read_json(prep / "source_freeze.json")
    freeze["logical_split_identities"] = observed
    base.write_json(prep / "source_freeze.json", freeze)
    result["logical_split_identities"] = observed
    base.write_json(prep / "phase_a_gate.json", result)
    return result


def verify_freeze(prep: Path) -> None:
    _base_verify_freeze(prep)
    freeze = base.read_json(prep / "source_freeze.json")
    expected = logical_split_identities()
    if freeze.get("logical_split_identities") != expected:
        raise RuntimeError("frozen logical split identity mismatch")
    if _prepared_split_identities(prep) != expected:
        raise RuntimeError("prepared logical split identity mismatch")


def preflight():
    result = _base_preflight()
    manifest = base.yaml.safe_load(base.MANIFEST.read_text(encoding="utf-8"))
    expected = logical_split_identities()
    result["checks"]["logical_split_protocol"] = (
        protocol_document()["logical_split_identities"] == expected
    )
    result["checks"]["logical_split_manifest"] = (
        manifest.get("data", {}).get("split_hashes") == expected
    )
    result["gate"] = "PASS" if all(result["checks"].values()) else "HOLD"
    return result


base.source_paths = source_paths
base.protocol_document = protocol_document
base.classify = classify
base.prepare = prepare
base.verify_freeze = verify_freeze
base.CLASSIFIER_VERSION = CLASSIFIER_VERSION
base.preflight = preflight


if __name__ == "__main__":
    base.main()
