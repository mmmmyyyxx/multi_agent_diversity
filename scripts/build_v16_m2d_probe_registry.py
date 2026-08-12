from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_v16_residual_diag_probe_registry import ROOT, SOURCE_RUNS, build_case


VARIANTS = (
    "m20_current_v15",
    "m2d_raw_responsibility_minimal_edit",
    "m2b_diagnosis_minimal_edit",
)


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_registry(execution_commit: str) -> dict[str, Any]:
    cases = []
    for seed in (48, 49, 50, 51):
        for ordinal in (0, 1):
            case = build_case(
                seed, SOURCE_RUNS[seed], len(cases),
                eligible_parent_ordinal=ordinal,
            )
            case["cell_order"] = [
                VARIANTS[(len(cases) + offset) % 3] for offset in range(3)
            ]
            cases.append(case)
    for seed in (48, 49, 50, 51):
        hashes = {
            case["parent_team_hash"] for case in cases
            if case["source_seed"] == seed
        }
        if len(hashes) != 2:
            raise ValueError(f"Seed{seed}: selected parents are not distinct")
    payload = {
        "registry_version": "v16_m2d_fixed_parent_registry_v1",
        "execution_commit": execution_commit,
        "case_selection_uses_candidate_outcomes": False,
        "candidate_count_per_cell": 2,
        "commit_enabled": False,
        "validation_enabled": False,
        "final_test_enabled": False,
        "model": "qwen3-14b",
        "thinking": False,
        "variants": list(VARIANTS),
        "cases": cases,
    }
    payload["registry_content_hash"] = digest(payload)
    return payload


def public_registry(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "registry_version": payload["registry_version"],
        "execution_commit": payload["execution_commit"],
        "registry_content_hash": payload["registry_content_hash"],
        "case_selection_uses_candidate_outcomes": False,
        "candidate_count_per_cell": 2,
        "commit_enabled": False,
        "validation_enabled": False,
        "final_test_enabled": False,
        "model": "qwen3-14b",
        "thinking": False,
        "variants": payload["variants"],
        "cases": [{
            "case_id": case["case_id"],
            "seed": case["source_seed"],
            "update_index": case["source_update_index"],
            "parent_team_hash": case["parent_team_hash"],
            "target_agent_id": case["target_agent_id"],
            "responsibility_evidence_hash": hashlib.sha256(
                json.dumps(
                    sorted(case["assigned_question_hashes"]),
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "variant_order": case["cell_order"],
        } for case in payload["cases"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public_out", type=Path, required=True)
    parser.add_argument("--execution_commit", default="PHASE_A_PENDING")
    args = parser.parse_args()
    for path in (args.out.resolve(), args.public_out.resolve()):
        if ROOT.resolve() not in path.parents or path.exists():
            raise SystemExit("outputs must be fresh and project-local")
    payload = build_registry(args.execution_commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.public_out.parent.mkdir(parents=True, exist_ok=True)
    args.public_out.write_text(
        json.dumps(public_registry(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS", "case_count": 8, "cell_count": 24,
        "candidate_budget": 2, "api_calls": 0,
        "registry_content_hash": payload["registry_content_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()
