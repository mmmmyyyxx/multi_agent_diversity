"""Zero-API refreeze for the six-parent contract-adapted RG-GEPA V2 pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.experimental_contract_adapted_rg_gepa import (  # noqa: E402
    AVOID_IDS,
    ContractAdaptedProtocolV2,
    EDIT_IDS,
    FAILURE_IDS,
    HypothesisSelectionV2,
    PRESERVE_IDS,
    build_hypothesis_request_v2,
    hypothesis_interface_v2_identity,
    render_contract_adapted_prompt_v2,
    renderer_vocabulary_identity,
)
from scripts.run_contract_adapted_rg_gepa_fixed_parent_pilot_v2 import (  # noqa: E402
    _protocol_payload,
    _request,
)


SOURCE_PREP = ROOT / "runs" / "responsibility_guided_gepa_fixed_parent_pilot_v1_prep_20260909_retry4"
QUALIFICATION_REPORT = ROOT / "reports" / "rg_gepa_hypothesis_interface_v2_qualification_execution"
QUALIFICATION_PREP = ROOT / "runs" / "rg_gepa_hypothesis_interface_v2_qualification_prep_20260910"
DEFAULT_PREP = ROOT / "runs" / "contract_adapted_rg_gepa_fixed_parent_pilot_v2_prep_20260910"
DEFAULT_REPORT = ROOT / "reports" / "contract_adapted_rg_gepa_fixed_parent_pilot_v2"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def _sources() -> list[Path]:
    return [
        Path("multi_dataset_diverse_rl/experimental_contract_adapted_rg_gepa.py"),
        Path("multi_dataset_diverse_rl/experimental_rg_gepa.py"),
        Path("scripts/prepare_contract_adapted_rg_gepa_fixed_parent_pilot_v2.py"),
        Path("scripts/run_contract_adapted_rg_gepa_fixed_parent_pilot_v2.py"),
        Path("scripts/run_responsibility_guided_gepa_fixed_parent_pilot.py"),
        Path("scripts/run_rg_gepa_hypothesis_interface_v2_qualification.py"),
        Path("infrastructure/common_solver_contract_v1/contract.py"),
        Path("infrastructure/common_solver_contract_v1/evaluator.py"),
        Path("infrastructure/common_solver_contract_v1/system_adapter.py"),
        Path("tests/test_experimental_contract_adapted_rg_gepa.py"),
        Path("tests/test_contract_adapted_rg_gepa_fixed_parent_pilot_v2.py"),
    ]


def prepare(prep: Path, report: Path) -> dict[str, Any]:
    if prep.exists() or report.exists():
        raise FileExistsError("V2 prep/report roots must be fresh")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before V2 refreeze")
    qualification = _read(QUALIFICATION_REPORT / "fact_assertions.json")
    if qualification["status"] != "PASS" or qualification["schema_valid"] != 12:
        raise RuntimeError("V2 12/12 schema qualification is not PASS")
    qualification_registry = _read(QUALIFICATION_PREP / "private_registry.json")
    from scripts.run_rg_gepa_hypothesis_interface_v2_qualification import (
        build_request as build_qualification_request,
    )
    current_qualification_hashes = []
    for case in qualification_registry["cases"]:
        system, user = build_qualification_request(case)
        current_qualification_hashes.append({
            "qualification_id": case["qualification_id"],
            "system_hash": _sha(system.encode("utf-8")),
            "user_hash": _sha(user.encode("utf-8")),
        })
    if current_qualification_hashes != qualification_registry["request_hashes"]:
        raise RuntimeError("qualified V2 request builder bytes have drifted")
    source_registry = _read(SOURCE_PREP / "private_registry.json")
    if len(source_registry["cases"]) != 6:
        raise RuntimeError("source registry does not contain six frozen parents")
    cases = source_registry["cases"]
    minibatches = source_registry["minibatches"]
    for case in cases:
        case["minibatch_private"] = [
            next(row for row in case["questions"] if row["example_id"] == item["example_id"])
            | {"source_type": item["source_type"]}
            for item in minibatches[case["case_id"]]
        ]
        # Freeze production-shaped request bytes for both independent mutations.
        for index in range(2):
            _request(case, index)
        case.pop("minibatch_private")
    protocol = ContractAdaptedProtocolV2()
    structural = 0
    for case in cases:
        parent = case["parent_prompts"][int(case["target_member"])]
        for failure in FAILURE_IDS:
            for edit in EDIT_IDS:
                for preserve in PRESERVE_IDS:
                    for avoid in AVOID_IDS:
                        render_contract_adapted_prompt_v2(
                            parent, HypothesisSelectionV2(failure, edit, preserve, avoid)
                        )
                        structural += 1
    if structural != 6 * 576:
        raise AssertionError("V2 structural renderer coverage mismatch")
    request_hashes = []
    for case in cases:
        private_rows = [
            next(row for row in case["questions"] if row["example_id"] == item["example_id"])
            | {"source_type": item["source_type"]}
            for item in minibatches[case["case_id"]]
        ]
        case["minibatch_private"] = private_rows
        for index in range(2):
            system, user = _request(case, index)
            request_hashes.append({
                "case_id": case["case_id"],
                "candidate_index": index,
                "system_hash": _sha(system.encode("utf-8")),
                "user_hash": _sha(user.encode("utf-8")),
            })
        case.pop("minibatch_private")
    registry = {
        "registry_version": "contract_adapted_rg_gepa_fixed_parent_registry_v2",
        "execution_commit": _git("rev-parse", "HEAD"),
        "protocol": _protocol_payload(),
        "protocol_hash": protocol.identity(),
        "hypothesis_interface_hash": hypothesis_interface_v2_identity(),
        "renderer_vocabulary_hash": renderer_vocabulary_identity(),
        "source_retry4_registry_hash": source_registry["registry_hash"],
        "qualification_fact_hash": _sha_json(qualification),
        "qualification_request_hashes_verified": True,
        "same_parent_and_minibatch_bytes_as_retry4": True,
        "case_selection_uses_outcomes": False,
        "all_candidates_full_evaluated": True,
        "nonpromoted_full_results_audit_only": True,
        "selection_uses_promoted_only": True,
        "api_authorized": False,
        "commit_enabled": False,
        "validation_enabled": False,
        "test_enabled": False,
        "memory_enabled": False,
        "cases": cases,
        "minibatches": minibatches,
        "production_request_hashes": request_hashes,
    }
    registry["registry_hash"] = _sha_json(registry)
    source_files = [
        {"path": path.as_posix(), "sha256": _sha((ROOT / path).read_bytes())}
        for path in _sources()
    ]
    prep.mkdir(parents=True)
    _write(prep / "private_registry.json", registry)
    freeze = {
        "status": "PASS",
        "freeze_version": "contract_adapted_rg_gepa_fixed_parent_v2",
        "execution_commit": registry["execution_commit"],
        "registry_hash": registry["registry_hash"],
        "protocol_hash": protocol.identity(),
        "hypothesis_interface_hash": hypothesis_interface_v2_identity(),
        "renderer_vocabulary_hash": renderer_vocabulary_identity(),
        "optimizer_model": protocol.optimizer_model,
        "reflection_temperature": protocol.reflection_temperature,
        "reflection_max_tokens": protocol.reflection_max_tokens,
        "reflection_role": protocol.reflection_role,
        "source_retry4_registry_hash": source_registry["registry_hash"],
        "qualification_status": "PASS_12_OF_12",
        "qualification_request_hashes_verified": True,
        "case_count": 6,
        "candidate_count": 12,
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "source_files": source_files,
    }
    _write(prep / "PRE_API_FREEZE.json", freeze)

    report.mkdir(parents=True)
    _write(report / "protocol.json", {
        **_protocol_payload(),
        "protocol_hash": protocol.identity(),
        "case_count": 6,
        "candidate_count": 12,
        "all_candidates_full_evaluated": True,
        "nonpromoted_full_results_audit_only": True,
        "promotion_frozen_before_full_results": True,
        "selection_uses_promoted_candidates_only": True,
        "reflection_context": [
            "parent_prompt", "question", "gold", "target_prediction", "target_correct", "responsibility_source"
        ],
        "reasoning_trace_generated": False,
    })
    _write(report / "fixed_parent_manifest.json", [
        {
            "case_id": case["case_id"],
            "seed": case["source_seed"],
            "update_index": case["source_update_index"],
            "target_member": case["target_member"],
            "responsibility_type": case["responsibility_type"],
            "parent_team_hash": case["parent_team_hash"],
            "minibatch_hash": _sha_json(minibatches[case["case_id"]]),
        }
        for case in cases
    ])
    _write(report / "production_request_manifest.json", request_hashes)
    _write(report / "frozen_classifier.json", {
        "proposal_quality_denominator": "all_12_full_evaluated_candidates",
        "progressive_recall_denominator": "all_full_evaluated_feasible_candidates",
        "selection_pool": "promoted_and_feasible_only",
        "rules": [
            "true feasible yield <=1 => ADAPTED_PROPOSAL_QUALITY_POOR",
            "true feasible yield >=2 and any B0-prime/B1-prime difference => PARETO_SELECTION_DIFFERENTIATES",
            "true feasible yield >=2 and no selector difference => PROPOSAL_SIGNAL_PARETO_NOT_NEEDED",
        ],
        "validation_superiority_claim_allowed": False,
        "coalition_aware_pareto_claim_allowed": False,
    })
    _write(report / "fact_assertions.json", {
        "status": "PASS",
        "phase": "ZERO_API_PRE_API_REFREEZE",
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "qualification": "PASS_12_OF_12",
        "case_count": 6,
        "candidate_count": 12,
        "structural_renderings_checked": structural,
        "renderer_contract_violations": 0,
        "historical_artifacts_modified": 0,
        "source_hash_equality_required_at_execution": True,
    })
    _write(report / "provenance.json", {
        "source_retry4_registry_hash": source_registry["registry_hash"],
        "qualification_fact_hash": registry["qualification_fact_hash"],
        "private_material": "runs-only",
    })
    (report / "README.md").write_text(
        "# Contract-adapted RG-GEPA fixed-parent pilot V2\n\n"
        "This zero-API refreeze wires the qualified V2 enum-array interface into a new scientific runner; "
        "the V1 runner remains unchanged for historical replay. Qualification and production share one "
        "authoritative request builder and the same qwen3.7-flash decoding settings. Protocol identity binds "
        "the interface and renderer hashes, and execution recomputes every relevant source hash.\n\n"
        "The six retry4 parents and minibatches are unchanged. Every one of the 12 candidates will receive a "
        "full fixed-probe rollout, but non-promoted results are audit-only and cannot alter promotion or the "
        "B0-prime/B1-prime winners. This separates true proposal yield from progressive-screen recall. No API, "
        "commit, Validation, Test, or memory operation occurred during this refreeze.\n",
        encoding="utf-8",
    )
    prohibited = ("FINAL_ANSWER:", "api_key", "dashscope", "https://", "D:\\\\")
    hashes = []
    for path in sorted(report.iterdir()):
        data = path.read_bytes()
        text = data.decode("utf-8", errors="ignore").lower()
        if any(item.lower() in text for item in prohibited):
            raise RuntimeError(f"V2 refreeze sanitization failure: {path.name}")
        hashes.append({"path": path.name, "sha256": _sha(data), "bytes": len(data)})
    _write(report / "sanitization_manifest.json", {"status": "PASS", "files_checked": len(hashes)})
    _write(report / "sha256_manifest.json", {"files": hashes})
    return {"status": "PASS", "api_calls": 0, "registry_hash": registry["registry_hash"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prep.resolve(), args.report.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
