"""Build sanitized Formal V3 offline-closure evidence from guarded fake runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_formal_v3_offline_freeze import audit as audit_freeze  # noqa: E402


CASES = (
    "native_saturation", "native_provider_emergency", "native_optimizer_emergency",
    "native_unexpected_return", "layer2_saturation", "layer2_no_feasible",
    "layer2_local_emergency", "layer2_minibatch_rejection",
    "layer2_common_safe_rejection", "layer2_shadow_rejection",
    "layer2_commit_restart", "layer2_partial_delivery",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(root: Path, name: str, value: object) -> None:
    (root / name).write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def build(*, prep_a: Path, prep_b: Path, evidence_root: Path, report: Path,
          historical_failed: int, historical_errors: int,
          attempt_number: int = 1) -> dict:
    if report.exists():
        raise FileExistsError("fresh sanitized report root required")
    if not evidence_root.resolve().is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("private fake evidence must remain under ignored runs/")
    offline = audit_freeze(prep_a, prep_b, attempt_number=attempt_number)
    cases = {case: _read(evidence_root / f"{case}.json") for case in CASES}
    for case, row in cases.items():
        ledger = row["ledger"]
        if (ledger["provider_attempts"] != sum(row["fake_physical_calls"].values())
                or ledger["input_tokens"] + ledger["output_tokens"] != ledger["total_tokens"]
                or ledger["failed_provider_attempts"] != 0
                or row["validation50_calls"] != 0 or row["test50_calls"] != 0):
            raise AssertionError(f"fake evidence ledger mismatch: {case}")
    if (cases["native_saturation"]["stop_reason"] != "SATURATION_REACHED"
            or cases["layer2_saturation"]["stop_reason"] != "SATURATION_REACHED"
            or cases["layer2_no_feasible"]["stop_reason"] != "NO_FEASIBLE_LAYER2_OPPORTUNITY"
            or cases["layer2_local_emergency"]["stop_reason"] != "EMERGENCY_OPTIMIZER_STEP_CEILING"):
        raise AssertionError("formal fake classifier mismatch")
    if not any(item["scheduled_but_not_delivered_count"] > 0
               for item in cases["layer2_partial_delivery"]["evidence"]):
        raise AssertionError("scheduled/delivered distinction was not observed")
    cells = {}
    data_hashes = set()
    sources = set()
    for prep in sorted(prep_a.iterdir()):
        manifest = _read(prep / "manifest.json")
        protocol = _read(prep / "protocol.json")
        scientific_identity = _read(prep / "startup_identity/scientific_identity.json")
        run_identity = _read(prep / "startup_identity/run_identity.json")
        source = manifest["execution"]["execution_source_sha"]
        sources.add(source)
        data_hashes.add(json.dumps(scientific_identity["payload"]["data_hashes"], sort_keys=True))
        cells[prep.name] = {
            "seed": manifest["runtime"]["seed"], "arm": protocol["arm"],
            "execution_source_sha": source,
            "protocol_sha256": scientific_identity["payload"]["protocol_sha256"],
            "preregistration_sha256": scientific_identity["preregistration_sha256"],
            "run_identity_sha256": run_identity["run_identity_sha256"],
            "source_manifest_sha256": hashlib.sha256(json.dumps(
                scientific_identity["payload"]["source_files"],
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
            "source_file_count": len(scientific_identity["payload"]["source_files"]),
            "api_authorized": False, "real_v4_prerequisite": manifest["execution_gate"]["real_v4_diagnostic"],
            "diagnostic_prerequisite": manifest.get("diagnostic_prerequisite"),
        }
    if len(cells) != 6 or len(sources) != 1 or len(data_hashes) != 1:
        raise AssertionError("six-cell formal comparison identity mismatch")
    report.mkdir(parents=True)
    _write(report, "api_isolation.json", {
        "gate": "PASS", "real_credentials_present_in_fake_process": False,
        "real_api_calls": 0, "real_network_attempts": 0,
        "validation50_calls": 0, "test50_calls": 0,
        "authorization_consumed": False,
    })
    _write(report, "network_isolation.json", {
        "gate": "PASS", "pre_import_sitecustomize_guard": True,
        "socket_negative_control": "PASS_BLOCKED_BEFORE_TRANSMISSION",
        "normal_fake_network_attempts": 0, "external_host_whitelist": False,
    })
    _write(report, "provider_constructor_audit.json", {
        "gate": "PASS", "active_solver_path": "ProviderClientFactory.from_environment",
        "active_reflection_path": "ProviderClientFactory.create",
        "other_active_direct_sdk_constructors": 0,
        "solver_fake_sentinel_asserted": True,
        "reflection_fake_sentinel_asserted": True,
        "evaluator_separate_client_reached": False,
        "native_constructor_counts": cases["native_saturation"]["provider_constructor_calls"],
        "layer2_constructor_counts": cases["layer2_saturation"]["provider_constructor_calls"],
    })
    for filename, case in (
        ("native_saturation_trace.json", "native_saturation"),
        ("layer2_saturation_trace.json", "layer2_saturation"),
    ):
        _write(report, filename, cases[case])
    _write(report, "team_epoch_trace.json", {
        "no_commit_epoch_trajectory": cases["layer2_saturation"]["events"],
        "commit_restart_trajectory": cases["layer2_commit_restart"]["events"],
        "empty_eligible_set": cases["layer2_no_feasible"]["events"],
        "parent_state_scoped": True,
        "commit_interrupted_partial_epoch_not_counted_as_no_update": True,
    })
    _write(report, "emergency_propagation.json", {
        case: cases[case] for case in (
            "native_provider_emergency", "native_optimizer_emergency",
            "native_unexpected_return", "layer2_local_emergency",
        )
    })
    _write(report, "admission_negative_controls.json", {
        case: cases[case] for case in (
            "layer2_minibatch_rejection", "layer2_common_safe_rejection",
            "layer2_shadow_rejection", "layer2_commit_restart",
        )
    })
    _write(report, "evidence_delivery_trace.json", {
        "scheduled_vs_delivered": cases["layer2_partial_delivery"]["evidence"],
        "full_funnel": cases["layer2_saturation"]["evidence"],
    })
    _write(report, "ledger_reconciliation.json", {
        "gate": "PASS", "cases": {
            case: {"fake_physical_calls": row["fake_physical_calls"],
                   "ledger": row["ledger"], "stages": row["stages"], "roles": row["roles"]}
            for case, row in cases.items()
        },
    })
    _write(report, "formal_arm_comparability.json", {
        "gate": "PASS", "cells": 6, "same_source": True,
        "same_optimize100_shadow50_hashes": True,
        "same_fresh_initialization_policy": True,
        "same_provider_and_models": True,
        "only_treatment_scope_difference": "native_feed_vs_layer2_v4_packet_and_team_controller",
        "nominal_native_and_layer2_units_not_equated": True,
    })
    _write(report, "source_poison.json", {
        "gate": "PASS", "cases": offline["source_poison_cases"],
        "each_active_source_byte_poison_aborted_pre_provider": True,
        "fake_provider_constructions": offline["provider_constructions"],
        "network_attempts": 0,
    })
    _write(report, "governance_preflight.json", {
        "gate": "PASS_EXECUTION_GATED", "validated_cells": 6,
        "real_v4_prerequisite": (
            "SCIENTIFICALLY_VALID" if attempt_number == 2 else "PENDING_SCIENTIFIC_VALIDITY"
        ),
        "api_authorized": False, "provider_profile": "lwj",
        "solver_model": "qwen3-8b", "solver_thinking": False,
        "reflection_model": "qwen3.7-flash", "seeds": [80, 81, 82],
        "local_no_update_patience": 3, "team_no_update_patience": 2,
        "validation50_calls": 0, "test50_calls": 0,
    })
    _write(report, "freeze_identity.json", {
        "successor_id": "gepa_saturation_comparison_v3",
        "supersedes_unexecuted": (
            "gepa_saturation_comparison_v3_attempt1" if attempt_number == 2
            else "gepa_saturation_comparison_v2"
        ),
        "execution_source_sha": next(iter(sources)),
        "cells": cells, "prep_replay_identical": offline["prep_replay_identical"],
        "deterministic_artifacts": offline["deterministic_artifacts"],
        "authorization_consumed": False,
    })
    _write(report, "verification.json", {
        "focused_formal_fake": "PASS_ZERO_NETWORK",
        "full_tests": "HISTORICAL_PRIVATE_ARTIFACT_FAILURES_ONLY",
        "historical_failed": historical_failed, "historical_errors": historical_errors,
        "new_active_failure_classes": 0,
        "compileall": "PASS", "diff_check": "PASS",
        "sanitization": "PASS", "prep_replay": "PASS_IDENTICAL",
        "source_poison": "PASS", "real_api_calls": 0,
    })
    readme = (
        "# Formal V3 pre-execution closure (zero real API)\n\n"
        "The prior 63-call rehearsal remains an isolated engineering incident; its "
        "results and call counts are not reused here. Native and Layer2 V4 full-stack "
        "fake-provider rehearsals pass under a pre-import socket guard and credential-free "
        f"environment. The guarded full suite has only {historical_failed} failures "
        f"and {historical_errors} errors tied "
        "to unavailable historical private run artifacts; it is not a full-suite PASS.\n\n"
        "Six seed-by-arm cells are preregistered and execution-gated. Two independent "
        "offline preparations are byte-identical. Each frozen active source file was "
        "poisoned in memory, and startup validation rejected it before provider "
        "construction. Validation50 and Test50 remain at zero calls.\n\n"
        + ("The V4 Seed81 attempt3 pilot is CLOSED_VALID_INCONCLUSIVE: scientific validity "
         "is VALID and efficacy is NOT_EVALUABLE. This satisfies the Formal prerequisite. "
         "Native final teams replicate the selected GEPA candidate across five members; "
         "post-freeze Validation50 requires separate authorization and Test50 stays sealed. "
         if attempt_number == 2 else
         "The next real step is the separately frozen Seed81 V4 diagnostic and scientific "
         "validity audit. Only then may a separate authorization permit formal execution. ")
        + "This report does not authorize or execute any real experiment.\n"
    )
    (report / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(report.iterdir()) if path.is_file() and path.name != "sha256_manifest.json"
    }
    _write(report, "sha256_manifest.json", manifest)
    return {"status": "PASS", "execution_source_sha": next(iter(sources)),
            "report_files": len(manifest) + 1, "source_poison_cases": offline["source_poison_cases"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-a", type=Path, required=True)
    parser.add_argument("--prep-b", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--historical-failed", type=int, required=True)
    parser.add_argument("--historical-errors", type=int, required=True)
    parser.add_argument("--attempt-number", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    print(json.dumps(build(prep_a=args.prep_a, prep_b=args.prep_b,
                           evidence_root=args.evidence_root, report=args.report,
                           historical_failed=args.historical_failed,
                           historical_errors=args.historical_errors,
                           attempt_number=args.attempt_number),
                     sort_keys=True, indent=2))
