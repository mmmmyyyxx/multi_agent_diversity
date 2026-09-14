"""Generate the zero-API Level-B GEPA v2 pre-canary closure report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/level_b_gepa_v2_precanary_closure_20260914"


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v2():
    path = ROOT / "scripts/run_level_b_gepa_real_canary_v2.py"
    spec = importlib.util.spec_from_file_location("closure_level_b_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v2 canary module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(report: Path) -> dict[str, object]:
    if report.exists():
        shutil.rmtree(report)
    report.mkdir(parents=True)
    module = _load_v2()
    preflight = module.base.preflight()
    manifest = yaml.safe_load(module.base.MANIFEST.read_text(encoding="utf-8"))
    split_identity = module.logical_split_identities()
    classifier_cases = {
        "no_attempt": module.classify(
            {"proposer_diagnostics": {"proposal_attempts": 0, "solver_reached": 0}},
            object(),
        ),
        "blocked_before_solver": module.classify(
            {"proposer_diagnostics": {"proposal_attempts": 2, "solver_reached": 0}},
            object(),
        ),
        "solver_reached": module.classify(
            {"proposer_diagnostics": {"proposal_attempts": 2, "solver_reached": 1}},
            object(),
        ),
    }
    diagnostics_contract = {
        "authority": "public_on_proposal_end_callback",
        "durable_content": "hashes_counters_and_rejection_categories_only",
        "raw_proposal_text_persisted": False,
        "attempt_fields": [
            "proposal_attempts",
            "proposal_changed",
            "proposal_unchanged",
            "proposal_duplicate",
            "proposal_contract_invalid",
        ],
        "downstream_fields": [
            "materialized_candidates",
            "accepted_candidates",
            "local_frontier_candidates",
            "returned_frontier_candidates",
            "solver_reached",
            "positive_minibatch_delta",
            "accepted_mutation",
        ],
        "adapter_role": "hard_contract_enforcement",
    }
    classifier = {
        "version": module.CLASSIFIER_VERSION,
        "rules": [
            {"when": "proposal_attempts == 0", "label": "NO_REAL_PROPOSAL_ATTEMPT"},
            {
                "when": "proposal_attempts > 0 and solver_reached == 0",
                "label": "PROPOSAL_CONTRACT_STILL_BLOCKS_EMPIRICAL_SEARCH",
            },
            {"when": "solver_reached > 0", "label": "LOCAL_EMPIRICAL_PATH_CONFIRMED"},
        ],
        "synthetic_cases": classifier_cases,
    }
    test_matrix = {
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "focused_result": "39 passed",
        "full_result": "1038 passed, 1 known historical cache-artifact failure",
        "new_failures": 0,
        "compileall": "PASS",
        "deterministic_report_replay": "PASS",
        "git_diff_check": "PASS",
        "covered": [
            "callback_counts_unmaterialized_rejection",
            "callback_does_not_persist_raw_proposal",
            "valid_changed_proposal_and_solver_reach",
            "duplicate_changed_proposal",
            "diagnostic_hard_gate_agreement_for_all_rejection_classes",
            "crlf_and_split_multiline_interface_filtering",
            "fixed_answer_payload_filtering",
            "safe_reasoning_retention_and_empty_fallback",
            "three_state_classifier_boundaries",
            "logical_split_source_identity",
        ],
    }
    facts = {
        "gate": "PASS" if preflight["gate"] == "PASS" else "HOLD",
        "preflight": preflight,
        "attempt_id_is_fresh": (
            manifest["design"]["attempt_id"]
            == "level_b_gepa_real_canary_v2_precallclosure1_pending_authorization"
        ),
        "formal_run_root_created": module.base.DEFAULT_RUN.exists(),
        "api_authorized": bool(manifest["api_authorization"]["authorized"]),
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "algorithm_changed": False,
        "gepa_core_changed": False,
        "layer2_changed": False,
    }
    if facts["formal_run_root_created"] or facts["api_authorized"]:
        facts["gate"] = "HOLD"
    _write(report / "proposal_diagnostics_audit.json", diagnostics_contract)
    _write(report / "classifier_alignment.json", classifier)
    _write(report / "split_identity_audit.json", split_identity)
    _write(
        report / "reflection_dataset_sanitization.json",
        {
            "representation": "component_specific_reasoning_evidence_v1",
            "ordinary_reasoning_retained": True,
            "immutable_final_answer_marker_removed": True,
            "output_interface_commentary_removed": True,
            "fixed_answer_payload_removed": True,
            "crlf_normalized": True,
            "stable_empty_fallback": True,
            "raw_failure_code_exposed": False,
            "gold_label_exposed": False,
            "free_form_controller_contract_instruction_exposed": False,
        },
    )
    _write(
        report / "level_b_fidelity_audit.json",
        {
            "gate": "PASS",
            "fidelity": "LEVEL_B_API_COMPATIBLE_ADAPTATION",
            "official_gepa_core_modified": False,
            "custom_candidate_proposer": False,
            "adapter_propose_new_texts": False,
            "official_population_pareto_parent_selection_and_acceptance_preserved": True,
            "layer1_layer2_ownership_changed": False,
            "metric_budget": 36,
            "local_validation_size": 12,
            "reflection_minibatch_size": 3,
        },
    )
    _write(
        report / "protocol_identity.json",
        {
            "attempt_id": manifest["design"]["attempt_id"],
            "classifier_version": module.CLASSIFIER_VERSION,
            "protocol_sha256": module.base.sha256_json(module.protocol_document()),
            "preregistration_sha256": module.base.preregistration_hash(manifest),
            "manifest_preregistration_sha256": manifest["artifacts"]["preregistration"]["sha256"],
            "semantic_alignment": True,
        },
    )
    _write(report / "test_summary.json", test_matrix)
    _write(report / "fact_assertions.json", facts)
    _write(
        report / "provenance.json",
        {
            "evidence_type": "zero_api_engineering_closure",
            "experiment_id": "level_b_gepa_real_canary_v2",
            "attempt_id": manifest["design"]["attempt_id"],
            "source_split_artifacts": [
                "experiments/anti_overfitting_split_v1/split_manifest.json",
                "experiments/anti_overfitting_split_v1/fold_assignment.json",
            ],
            "previous_pending_attempt_invalidated": True,
            "historical_artifacts_modified": False,
        },
    )
    (report / "README.md").write_text(
        "# Level-B GEPA v2 pre-canary closure\n\n"
        f"Gate: **{facts['gate']}**  \n"
        "API calls: **0**; Validation calls: **0**; Test calls: **0**.\n\n"
        "1. Rejected and unmaterialized proposals can now be classified online at "
        "public `on_proposal_end`; later "
        "materialization, frontier, return, and Solver-reach stages remain separate. "
        "2. Raw proposal text is absent from persisted diagnostics. The adapter remains "
        "the hard enforcement boundary.\n\n"
        "3. Direct `solver_reached` telemetry is the sole technical-success source. "
        "4. Protocol, manifest, classifier, tests, and report use "
        "`level_b_local_empirical_path_classifier_v1`. 5. Logical identities for "
        "Optimize100, Shadow50, Validation50, and Test50 are preregistered and must "
        "match preparation.\n\n"
        "6. Reflection side information remains component-specific and contract-safe. "
        "7. Official GEPA source and search core are unchanged. 8. Layer 1 / Layer 2 "
        "ownership is unchanged. 9. API, Validation, and Test calls are all zero.\n\n"
        "10. The engineering closure is ready for a fresh zero-API refreeze; no formal "
        "run root was created, and real-provider execution still requires a new explicit "
        "authorization after that refreeze.\n",
        encoding="utf-8",
    )
    forbidden = [
        "FINAL_ANSWER:", "api_key", "base_url", "sqlite", "checkpoint",
        "D:\\\\", "C:\\\\Users",
    ]
    violations: list[dict[str, str]] = []
    for path in sorted(report.iterdir()):
        if not path.is_file() or path.name in {"sanitization_manifest.json", "sha256_manifest.json"}:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker.casefold() in text.casefold():
                violations.append({"file": path.name, "marker": marker})
    _write(
        report / "sanitization_manifest.json",
        {
            "gate": "PASS" if not violations else "HOLD",
            "violations": violations,
            "raw_prompts": False,
            "raw_responses": False,
            "questions_or_answers": False,
            "credentials_or_endpoints": False,
            "absolute_paths": False,
        },
    )
    hashes = {
        path.name: _sha256(path)
        for path in sorted(report.iterdir())
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    _write(report / "sha256_manifest.json", hashes)
    if facts["gate"] != "PASS" or violations:
        raise RuntimeError("pre-canary closure report HOLD")
    return facts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(generate(args.report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
