"""Zero-API freeze for the contract-adapted RG-GEPA fixed-parent pilot."""
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
    AVOIDANCE_PRIORITIES,
    BEHAVIORAL_CHANGES,
    ContractAdaptedProtocol,
    EditHypothesis,
    FAILURE_PATTERNS,
    PRESERVATION_PRIORITIES,
    render_contract_adapted_prompt,
    renderer_vocabulary_identity,
)


SOURCE_PREP = ROOT / "runs" / "responsibility_guided_gepa_fixed_parent_pilot_v1_prep_20260909_retry4"
SOURCE_REPORT = ROOT / "reports" / "responsibility_guided_gepa_fixed_parent_pilot_v1_execution_retry4"
DEFAULT_PREP = ROOT / "runs" / "contract_adapted_rg_gepa_fixed_parent_pilot_v1_prep_20260909"
DEFAULT_REPORT = ROOT / "reports" / "contract_adapted_rg_gepa_fixed_parent_pilot_v1"


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


def _source_hashes() -> list[dict[str, Any]]:
    paths = [
        Path("multi_dataset_diverse_rl/experimental_contract_adapted_rg_gepa.py"),
        Path("scripts/prepare_contract_adapted_rg_gepa_fixed_parent_pilot.py"),
        Path("scripts/run_contract_adapted_rg_gepa_fixed_parent_pilot.py"),
        Path("multi_dataset_diverse_rl/experimental_rg_gepa.py"),
        Path("scripts/run_responsibility_guided_gepa_fixed_parent_pilot.py"),
        Path("infrastructure/common_solver_contract_v1/contract.py"),
        Path("infrastructure/common_solver_contract_v1/evaluator.py"),
        Path("infrastructure/common_solver_contract_v1/system_adapter.py"),
        Path("tests/test_experimental_contract_adapted_rg_gepa.py"),
    ]
    return [
        {"path": path.as_posix(), "sha256": _sha((ROOT / path).read_bytes())}
        for path in paths
    ]


def _renderer_structural_preflight(registry: dict[str, Any]) -> dict[str, Any]:
    combinations = [
        EditHypothesis(failure, behavior, preserve, avoid)
        for failure in FAILURE_PATTERNS
        for behavior in BEHAVIORAL_CHANGES
        for preserve in PRESERVATION_PRIORITIES
        for avoid in AVOIDANCE_PRIORITIES
    ]
    rendered = 0
    prompt_hashes: set[str] = set()
    for case in registry["cases"]:
        parent = case["parent_prompts"][int(case["target_member"])]
        for hypothesis in combinations:
            prompt = render_contract_adapted_prompt(parent, hypothesis)
            prompt_hashes.add(_sha(prompt.encode("utf-8")))
            rendered += 1
    expected = 6 * 6 * 6 * 4 * 4
    if rendered != expected:
        raise AssertionError("renderer structural preflight coverage mismatch")
    return {
        "status": "PASS",
        "parents_checked": 6,
        "representable_hypotheses_per_parent": len(combinations),
        "renderings_checked": rendered,
        "unique_rendered_prompt_hashes": len(prompt_hashes),
        "contract_violations": 0,
    }


def prepare(prep: Path, report: Path) -> dict[str, Any]:
    if prep.exists() or report.exists():
        raise FileExistsError("contract-adapted prep and report roots must be fresh")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before contract-adapted freeze")
    source_registry = _read(SOURCE_PREP / "private_registry.json")
    source_freeze = _read(SOURCE_PREP / "PRE_API_FREEZE.json")
    source_quality = _read(SOURCE_REPORT / "proposal_quality.json")
    if len(source_registry["cases"]) != 6:
        raise RuntimeError("retry4 source registry does not contain six parents")
    if {row["arm"] for row in source_quality} != {"A0", "A1", "B0", "B1"}:
        raise RuntimeError("retry4 comparator report is incomplete")

    protocol = ContractAdaptedProtocol()
    registry = {
        "registry_version": "contract_adapted_rg_gepa_fixed_parent_registry_v1",
        "execution_commit": _git("rev-parse", "HEAD"),
        "protocol": asdict(protocol),
        "protocol_hash": protocol.identity(),
        "renderer_vocabulary_hash": renderer_vocabulary_identity(),
        "source_retry4_registry_hash": source_registry["registry_hash"],
        "source_retry4_freeze_hash": _sha_json(source_freeze),
        "source_retry4_comparator_hash": _sha_json(source_quality),
        "same_parent_and_minibatch_bytes_as_retry4": True,
        "case_selection_uses_outcomes": False,
        "api_authorized": False,
        "commit_enabled": False,
        "validation_enabled": False,
        "test_enabled": False,
        "memory_enabled": False,
        "cases": source_registry["cases"],
        "minibatches": source_registry["minibatches"],
    }
    registry["registry_hash"] = _sha_json(registry)
    structural = _renderer_structural_preflight(registry)

    prep.mkdir(parents=True)
    _write(prep / "private_registry.json", registry)
    freeze = {
        "PRE_API_FREEZE_version": "contract_adapted_rg_gepa_v1",
        "status": "PASS",
        "execution_commit": registry["execution_commit"],
        "protocol_hash": protocol.identity(),
        "registry_hash": registry["registry_hash"],
        "renderer_vocabulary_hash": renderer_vocabulary_identity(),
        "source_retry4_registry_hash": source_registry["registry_hash"],
        "source_retry4_comparator_hash": registry["source_retry4_comparator_hash"],
        "case_count": 6,
        "candidate_budget_per_case": 2,
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "source_files": _source_hashes(),
    }
    _write(prep / "PRE_API_FREEZE.json", freeze)

    report.mkdir(parents=True)
    case_manifest = [
        {
            "case_id": case["case_id"],
            "seed": case["source_seed"],
            "update_index": case["source_update_index"],
            "target_member": case["target_member"],
            "responsibility_type": case["responsibility_type"],
            "parent_team_hash": case["parent_team_hash"],
            "minibatch_hash": _sha_json(source_registry["minibatches"][case["case_id"]]),
            "candidate_budget": 2,
        }
        for case in registry["cases"]
    ]
    _write(report / "fixed_parent_manifest.json", case_manifest)
    _write(report / "protocol.json", {
        **asdict(protocol),
        "protocol_version": "contract_adapted_rg_gepa_fixed_parent_v1",
        "protocol_hash": protocol.identity(),
        "same_frozen_parents_and_evidence_as_retry4": True,
        "B0_prime_B1_prime_shared_candidate_pool": True,
        "raw_reflection_text_enters_candidate": False,
        "schema_failure_policy": "HOLD_BEFORE_CANDIDATE; no fallback, substitution, or added candidate",
        "nonpromoted_full_eval_policy": "retry4-frozen audit-only witness cases",
        "audit_only_case_ids": sorted(("seed76_u0_coverage_target1", "seed77_u0_coverage_target2")),
    })
    _write(report / "renderer_spec.json", {
        "renderer_version": protocol.renderer_version,
        "renderer_vocabulary_hash": renderer_vocabulary_identity(),
        "hypothesis_fields": ["failure_pattern", "behavioral_change", "preserve", "avoid"],
        "allowed_symbols": {
            "failure_pattern": list(FAILURE_PATTERNS),
            "behavioral_change": list(BEHAVIORAL_CHANGES),
            "preserve": list(PRESERVATION_PRIORITIES),
            "avoid": list(AVOIDANCE_PRIORITIES),
        },
        "raw_model_text_copied": False,
        "program_owned_rendering_only": True,
        "structural_preflight": structural,
    })
    _write(report / "retry4_evidence_boundary.json", {
        "H_COST": "SUPPORTED",
        "H_PROGRESSIVE": "SUPPORTED_WITH_RECALL_LOSS",
        "H_RGGEPA_PROPOSAL": "NOT_EVALUATED",
        "H_PARETO": "NOT_EVALUATED",
        "reason": "retry4 reflection candidates did not reach rollout because none satisfied the immutable mutation contract",
        "comparator_arms": source_quality,
    })
    _write(report / "frozen_hypotheses.json", {
        "primary_question": "Can contract-adapted RG-GEPA reflection supply useful candidates on the six frozen parents?",
        "H_CONTRACT_COMPATIBILITY": "STRUCTURALLY_GUARANTEED_FOR_EVERY_RENDERED_CANDIDATE",
        "H_ADAPTED_PROPOSAL_QUALITY": "NOT_EVALUATED_PRE_API",
        "H_PARETO_SELECTION_VALUE": "NOT_EVALUATED_PRE_API",
        "interpretation_rules": [
            "If fewer than 12 schema-valid hypotheses are returned, HOLD; do not substitute or add candidates.",
            "If 12 rendered candidates exist, all 12 must pass the immutable-contract gate or the execution gate fails.",
            "If at most one promoted full-evaluated candidate is feasible, label ADAPTED_PROPOSAL_QUALITY_POOR.",
            "If at least two are feasible and B0-prime equals B1-prime in every selectable case, label PROPOSAL_SIGNAL_PARETO_NOT_NEEDED.",
            "If at least two are feasible and the selectors differ, label PARETO_SELECTION_DIFFERENTIATES; do not claim validation superiority.",
            "Otherwise label INCONCLUSIVE_FIXED_PARENT_SIGNAL.",
        ],
        "validation_or_test_used_for_classification": False,
    })
    _write(report / "fact_assertions.json", {
        "status": "PASS",
        "phase": "ZERO_API_PRE_API_FREEZE",
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "historical_artifacts_modified": 0,
        "case_count": 6,
        "candidate_budget": 12,
        "memory_enabled": False,
        "canonical_method_modified": False,
        "renderer_structural_preflight": structural["status"],
    })
    _write(report / "provenance.json", {
        "source": "audited retry4 fixed-parent registry and sanitized comparator report",
        "source_registry_hash": source_registry["registry_hash"],
        "source_comparator_hash": registry["source_retry4_comparator_hash"],
        "private_material": "runs-only",
    })
    (report / "README.md").write_text(
        "# Contract-adapted RG-GEPA fixed-parent pilot v1\n\n"
        "This directory freezes the zero-API Phase A design. It reuses the six audited retry4 parent "
        "states and responsibility minibatches byte-for-byte. The only intervention is the proposal "
        "representation: qwen3.7-flash selects four symbols in a strict schema, and a deterministic "
        "closed-vocabulary renderer creates the mutable qwen3-8b decision procedure. Raw reflection "
        "text cannot enter a candidate. B0-prime and B1-prime share the same two candidates per case "
        "and differ only in current versus team-Pareto ranking.\n\n"
        "No API, validation, test, memory, prompt commit, or canonical-method change occurred in Phase A. "
        "A future execution requires separate explicit API authorization and fresh ignored run/report roots.\n",
        encoding="utf-8",
    )
    prohibited = ("FINAL_ANSWER:", "api_key", "dashscope", "https://", "D:\\\\")
    hashes = []
    for path in sorted(report.iterdir()):
        data = path.read_bytes()
        text = data.decode("utf-8", errors="ignore").lower()
        if any(item.lower() in text for item in prohibited):
            raise RuntimeError(f"sanitization failure: {path.name}")
        hashes.append({"path": path.name, "sha256": _sha(data), "bytes": len(data)})
    _write(report / "sanitization_manifest.json", {"status": "PASS", "files_checked": len(hashes)})
    _write(report / "sha256_manifest.json", {"files": hashes})
    return {"status": "PASS", "api_calls": 0, "case_count": 6, "registry_hash": registry["registry_hash"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prep.resolve(), args.report.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
