"""Freeze the zero-dataset RG-GEPA Hypothesis Interface V2 qualification."""
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
    EDIT_IDS,
    FAILURE_IDS,
    HypothesisSelectionV2,
    PRESERVE_IDS,
    SchemaQualificationV2Protocol,
    hypothesis_interface_v2_identity,
    render_contract_adapted_prompt_v2,
)
from scripts.run_rg_gepa_hypothesis_interface_v2_qualification import (  # noqa: E402
    build_request,
    qualification_cases,
)


DEFAULT_PREP = ROOT / "runs" / "rg_gepa_hypothesis_interface_v2_qualification_prep_20260910"
DEFAULT_REPORT = ROOT / "reports" / "rg_gepa_hypothesis_interface_v2_qualification"
PARENT_FIXTURE = "Use a careful, evidence-grounded decision procedure."


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def prepare(prep: Path, report: Path) -> dict[str, Any]:
    if prep.exists() or report.exists():
        raise FileExistsError("qualification prep/report roots must be fresh")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before qualification freeze")
    protocol = SchemaQualificationV2Protocol()
    cases = qualification_cases()
    request_hashes = []
    for case in cases:
        system, user = build_request(case)
        request_hashes.append({
            "qualification_id": case["qualification_id"],
            "system_hash": _sha(system.encode("utf-8")),
            "user_hash": _sha(user.encode("utf-8")),
        })
    render_count = 0
    for failure in FAILURE_IDS:
        for edit in EDIT_IDS:
            for preserve in PRESERVE_IDS:
                for avoid in AVOID_IDS:
                    render_contract_adapted_prompt_v2(
                        PARENT_FIXTURE,
                        HypothesisSelectionV2(failure, edit, preserve, avoid),
                    )
                    render_count += 1
    if render_count != 576:
        raise AssertionError("V2 representable-space preflight mismatch")

    registry = {
        "registry_version": "rg_gepa_hypothesis_interface_v2_qualification_registry_v1",
        "execution_commit": _git("rev-parse", "HEAD"),
        "protocol": asdict(protocol),
        "protocol_hash": protocol.identity(),
        "interface_hash": hypothesis_interface_v2_identity(),
        "cases": cases,
        "request_hashes": request_hashes,
        "dataset_evidence_used": False,
        "scientific_evidence_eligible": False,
        "api_authorized": False,
    }
    registry["registry_hash"] = _sha_json(registry)
    sources = [
        Path("multi_dataset_diverse_rl/experimental_contract_adapted_rg_gepa.py"),
        Path("scripts/prepare_rg_gepa_hypothesis_interface_v2_qualification.py"),
        Path("scripts/run_rg_gepa_hypothesis_interface_v2_qualification.py"),
        Path("tests/test_experimental_contract_adapted_rg_gepa.py"),
        Path("tests/test_rg_gepa_hypothesis_interface_v2_qualification.py"),
    ]
    prep.mkdir(parents=True)
    _write(prep / "private_registry.json", registry)
    _write(prep / "PRE_API_FREEZE.json", {
        "status": "PASS",
        "freeze_version": "rg_gepa_hypothesis_interface_v2_qualification_v1",
        "execution_commit": registry["execution_commit"],
        "registry_hash": registry["registry_hash"],
        "protocol_hash": protocol.identity(),
        "interface_hash": hypothesis_interface_v2_identity(),
        "request_count": 12,
        "pass_threshold": 12,
        "api_calls": 0,
        "solver_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "source_files": [
            {"path": path.as_posix(), "sha256": _sha((ROOT / path).read_bytes())}
            for path in sources
        ],
    })

    report.mkdir(parents=True)
    _write(report / "protocol.json", {
        **asdict(protocol),
        "protocol_hash": protocol.identity(),
        "interface_hash": hypothesis_interface_v2_identity(),
        "wire_format": ["failure_id", "edit_id", "preserve_id", "avoid_id"],
        "wire_representation": "exact JSON array; positions are program-owned fields",
        "transport_retries": "provider-only; schema output is never retried",
        "qualification_pass_rule": "12_OF_12_STRICT_SCHEMA_VALID",
        "fixed_parent_execution_after_pass": "requires separate freeze and authorization",
    })
    _write(report / "qualification_manifest.json", {
        "cases": cases,
        "request_hashes": request_hashes,
        "real_questions_or_answers": 0,
        "experiment_parent_states": 0,
    })
    _write(report / "action_language.json", {
        "interface_hash": hypothesis_interface_v2_identity(),
        "failure_ids": list(FAILURE_IDS),
        "edit_ids": list(EDIT_IDS),
        "preserve_ids": list(PRESERVE_IDS),
        "avoid_ids": list(AVOID_IDS),
        "model_authored_keys": 0,
        "free_text_values": 0,
        "representable_combinations": render_count,
    })
    _write(report / "fact_assertions.json", {
        "status": "PASS",
        "phase": "ZERO_API_PRE_API_FREEZE",
        "api_calls": 0,
        "solver_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "scientific_evidence_eligible": False,
        "request_count": 12,
        "strict_pass_threshold": 12,
        "representable_combinations_checked": render_count,
        "renderer_contract_violations": 0,
        "historical_artifacts_modified": 0,
    })
    _write(report / "provenance.json", {
        "source_hold": "contract-adapted retry1 schema mismatch",
        "source_hold_mechanism_evidence_eligible": False,
        "qualification_context": "synthetic symbolic only",
        "private_registry": "runs-only",
    })
    (report / "README.md").write_text(
        "# RG-GEPA Hypothesis Interface V2 qualification\n\n"
        "This zero-API freeze replaces model-authored JSON keys with one exact four-position array of "
        "opaque enum IDs. Array position supplies the program-owned field meaning; the existing deterministic "
        "renderer expands only program-owned clauses. No cleanup, alias mapping, fallback, or schema retry is "
        "allowed.\n\n"
        "The qualification consists of 12 qwen3.7-flash requests over synthetic symbolic contexts. It calls no "
        "Solver and uses no experiment parent, dataset row, Validation, or Test evidence. PASS requires 12/12 "
        "strictly valid outputs. This engineering qualification is not scientific evidence and does not authorize "
        "a fixed-parent retry.\n",
        encoding="utf-8",
    )
    prohibited = ("FINAL_ANSWER:", "api_key", "dashscope", "https://", "D:\\\\")
    hashes = []
    for path in sorted(report.iterdir()):
        data = path.read_bytes()
        text = data.decode("utf-8", errors="ignore").lower()
        if any(item.lower() in text for item in prohibited):
            raise RuntimeError(f"qualification freeze sanitization failure: {path.name}")
        hashes.append({"path": path.name, "sha256": _sha(data), "bytes": len(data)})
    _write(report / "sanitization_manifest.json", {"status": "PASS", "files_checked": len(hashes)})
    _write(report / "sha256_manifest.json", {"files": hashes})
    return {
        "status": "PASS",
        "api_calls": 0,
        "request_count": 12,
        "registry_hash": registry["registry_hash"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prep.resolve(), args.report.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
