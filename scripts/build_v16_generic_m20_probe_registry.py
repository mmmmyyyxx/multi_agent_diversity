from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_v16_m2d_probe_registry import build_registry as build_m2d_registry


PREREGISTRATION = (
    ROOT
    / "runs"
    / "v16_responsibility_coherence_generic_m20_prep"
    / "probe_preregistration.json"
)
REGISTRY_VERSION = "v16_generic_m20_fixed_parent_registry_v1"
VARIANTS = ("g0_fixed_target_generic", "m20_current_v15")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def responsibility_evidence_hash(hashes: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(map(str, hashes)), separators=(",", ":")).encode()
    ).hexdigest()


def case_identity(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(case["case_id"]),
        "seed": int(case.get("seed", case.get("source_seed"))),
        "update_index": int(
            case.get("update_index", case.get("source_update_index"))
        ),
        "parent_team_hash": str(case["parent_team_hash"]),
        "target_agent_id": int(case["target_agent_id"]),
        "responsibility_evidence_hash": str(
            case.get("responsibility_evidence_hash")
            or responsibility_evidence_hash(case["assigned_question_hashes"])
        ),
    }


def validate_frozen_preregistration(payload: dict[str, Any]) -> None:
    if payload.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("frozen generic/M20 preregistration version mismatch")
    if tuple(payload.get("variants", ())) != VARIANTS:
        raise ValueError("frozen generic/M20 variant inventory mismatch")
    if int(payload.get("case_count", -1)) != 8:
        raise ValueError("frozen generic/M20 case count must be eight")
    if int(payload.get("cell_count", -1)) != 16:
        raise ValueError("frozen generic/M20 cell count must be sixteen")
    if int(payload.get("candidate_count_per_cell", -1)) != 2:
        raise ValueError("frozen generic/M20 candidate budget must be two")
    for key in (
        "commit_enabled",
        "parent_mutation_enabled",
        "optimizer_state_update_enabled",
        "validation_enabled",
        "final_test_enabled",
    ):
        if bool(payload.get(key)):
            raise ValueError(f"frozen probe isolation flag must be false: {key}")
    cases = payload.get("cases", [])
    if len(cases) != 8 or len({row["case_id"] for row in cases}) != 8:
        raise ValueError("frozen generic/M20 cases must be eight distinct identities")
    for index, case in enumerate(cases):
        expected = (
            list(VARIANTS)
            if index % 2 == 0
            else [VARIANTS[1], VARIANTS[0]]
        )
        if list(case.get("cell_order", ())) != expected:
            raise ValueError("frozen generic/M20 order is not the balanced A/B order")


def build_registry(
    execution_commit: str,
    *,
    preregistration_path: Path = PREREGISTRATION,
) -> dict[str, Any]:
    execution_commit = str(execution_commit).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", execution_commit):
        raise ValueError("execution_commit must be a full 40-character Git SHA")
    frozen = json.loads(preregistration_path.read_text(encoding="utf-8"))
    validate_frozen_preregistration(frozen)

    source = build_m2d_registry(execution_commit)
    source_cases = {str(row["case_id"]): row for row in source["cases"]}
    cases: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for frozen_case in frozen["cases"]:
        case_id = str(frozen_case["case_id"])
        if case_id not in source_cases:
            mismatches.append({"case_id": case_id, "reason": "missing"})
            continue
        reconstructed = source_cases[case_id]
        expected_identity = case_identity(frozen_case)
        actual_identity = case_identity(reconstructed)
        if expected_identity != actual_identity:
            mismatches.append({
                "case_id": case_id,
                "expected": expected_identity,
                "actual": actual_identity,
            })
            continue
        copied = dict(reconstructed)
        copied["cell_order"] = list(frozen_case["cell_order"])
        copied["frozen_responsibility_evidence_hash"] = expected_identity[
            "responsibility_evidence_hash"
        ]
        cases.append(copied)
    if mismatches or len(cases) != 8:
        raise ValueError(
            "frozen eight-case reconstruction mismatch: "
            + canonical_json(mismatches)
        )

    payload: dict[str, Any] = {
        "registry_version": REGISTRY_VERSION,
        "execution_commit": execution_commit,
        "frozen_preregistration_sha256": file_sha256(preregistration_path),
        "source_case_registry_version": source["registry_version"],
        "case_selection_uses_candidate_outcomes": False,
        "candidate_count_per_cell": 2,
        "case_count": 8,
        "cell_count": 16,
        "maximum_planned_candidates": 32,
        "commit_enabled": False,
        "parent_mutation_enabled": False,
        "optimizer_state_update_enabled": False,
        "validation_enabled": False,
        "final_test_enabled": False,
        "fresh_generation_both_arms": True,
        "same_evaluation_pool": True,
        "same_common_safe_geometry": True,
        "model": "qwen3-14b",
        "thinking": False,
        "variants": list(VARIANTS),
        "cases": cases,
    }
    payload["registry_content_hash"] = digest(payload)
    return payload


def public_reconstruction(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "registry_version": payload["registry_version"],
        "execution_commit": payload["execution_commit"],
        "registry_content_hash": payload["registry_content_hash"],
        "frozen_preregistration_sha256": payload[
            "frozen_preregistration_sha256"
        ],
        "case_count": 8,
        "cell_count": 16,
        "candidate_count_per_cell": 2,
        "variants": list(VARIANTS),
        "cases": [
            {
                **case_identity(case),
                "cell_order": list(case["cell_order"]),
            }
            for case in payload["cases"]
        ],
        "case_identity_mismatch_count": 0,
        "api_calls": 0,
        "model_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }


def _project_local_fresh(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise SystemExit("output must remain under the repository root")
    if resolved.exists():
        raise SystemExit("registry output must be fresh")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution_commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reconstruction_out", type=Path, required=True)
    parser.add_argument(
        "--preregistration", type=Path, default=PREREGISTRATION
    )
    args = parser.parse_args()
    out = _project_local_fresh(args.out)
    reconstruction_out = _project_local_fresh(args.reconstruction_out)
    payload = build_registry(
        args.execution_commit,
        preregistration_path=args.preregistration.resolve(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    reconstruction_out.parent.mkdir(parents=True, exist_ok=True)
    reconstruction_out.write_text(
        json.dumps(public_reconstruction(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "case_count": 8,
        "cell_count": 16,
        "candidate_budget": 32,
        "api_calls": 0,
        "registry_content_hash": payload["registry_content_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()
