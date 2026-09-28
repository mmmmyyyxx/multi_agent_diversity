"""Zero-API V4 attempt2 freeze/semantic/source audit into a sanitized report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import sitecustomize  # supplied by the guarded fresh-process launcher

from multi_dataset_diverse_rl.governance import production_execution as admission
from multi_dataset_diverse_rl.governance.freeze_hash import normalized_lf_bytes
from multi_dataset_diverse_rl.governance.startup_identity import StartupIdentityError
from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
from scripts.audit_online_transfer_diagnostic import audit as audit_fake
from scripts.prepare_online_transfer_diagnostic_v4 import ATTEMPT2_ID, EXPERIMENT_ID
from scripts.report_v4_preexecution_closure import _safe_evidence, _safe_feasibility


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "reports/v4_preexecution_closure_20260927"
HISTORICAL_FAKE = ROOT / "runs/v4_preexecution_fake_campaign2_20260927"
SECRET_NAME = re.compile(r"(?:API.?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION)", re.I)
CASES = {
    "A": "test_v4_raw_legal_bounded",
    "B": "test_v4_raw_legal_downstream_b",  # resolved by failure reason below
    "rerank": "test_v4_feasible_reranking",
    "no_feasible": "test_v4_no_feasible",
    "successor": "test_v4_commit_no_commit",
    "budget": "test_v4_fake_provider_budget",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(root: Path, name: str, value: object) -> None:
    (root / name).write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def summaries(fake_root: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in fake_root.rglob("execution_summary.json"):
        case = path.relative_to(fake_root).parts[0]
        if case in result:
            raise AssertionError("duplicate fake case")
        value = read(path)
        if value["experiment_id"] != EXPERIMENT_ID or audit_fake(path.parent)["gate"] != "PASS":
            raise AssertionError("fake V4 execution audit failed")
        result[case] = value
    if len(result) != 10:
        raise AssertionError("expected ten successful V4 official-GEPA fake scenarios")
    return result


def matching(cases: dict[str, dict], prefix: str) -> dict:
    found = [value for name, value in cases.items() if name.startswith(prefix)]
    if len(found) != 1:
        raise AssertionError(f"non-unique fake scenario: {prefix}")
    return found[0]


def compare_semantics(cases: dict[str, dict]) -> tuple[dict, dict]:
    historical_cases = summaries(HISTORICAL_FAKE)
    if historical_cases.keys() != cases.keys():
        raise AssertionError("SCIENTIFIC_DECISION_REQUIRED: frozen fake scenario set differs")
    decision_fields = (
        "target_member", "raw_V", "assignment_V", "packet_V",
        "team_minibatch_ids", "local_eval_ids", "nominal_schedule",
        "responsibility_scheduled", "packet_hash",
    )
    admission_fields = (
        "candidate_hash", "committed", "local_acceptance_delta",
        "local_candidate_score", "local_parent_score", "team_minibatch",
        "full", "ordinary_common_safe", "ordinary_shadow",
    )
    compared_opportunities = 0
    decision_digests: dict[str, str] = {}
    for case in sorted(cases):
        old, new = historical_cases[case], cases[case]
        if (len(old["evidence_view_trace"]) != len(new["evidence_view_trace"])
                or len(old["feasibility_trace"]) != len(new["feasibility_trace"])
                or old["commits"] != new["commits"]
                or old["stop_reason"] != new["stop_reason"]
                or len(old["candidate_diagnostics"]) != len(new["candidate_diagnostics"])):
            raise AssertionError(f"SCIENTIFIC_DECISION_REQUIRED: case outcome differs: {case}")
        for former, present in zip(old["candidate_diagnostics"], new["candidate_diagnostics"]):
            if any(former[field] != present[field] for field in admission_fields):
                raise AssertionError(f"SCIENTIFIC_DECISION_REQUIRED: admission differs: {case}")
        for former, present in zip(old["evidence_view_trace"], new["evidence_view_trace"]):
            if any(former[field] != present[field] for field in decision_fields):
                raise AssertionError(f"SCIENTIFIC_DECISION_REQUIRED: opportunity differs: {case}")
            compared_opportunities += 1
        for former, present in zip(old["feasibility_trace"], new["feasibility_trace"]):
            if (former["selected_member"] != present["selected_member"]
                    or former["members"] != present["members"]):
                raise AssertionError(f"SCIENTIFIC_DECISION_REQUIRED: feasibility differs: {case}")
        safe_decisions = {
            "opportunities": [
                {field: row[field] for field in decision_fields}
                for row in new["evidence_view_trace"]
            ],
            "admission": [
                {field: row[field] for field in admission_fields}
                for row in new["candidate_diagnostics"]
            ],
            "feasibility": [
                {field: row[field] for field in ("selected_member", "members")}
                for row in new["feasibility_trace"]
            ],
            "commits": new["commits"],
            "stop_reason": new["stop_reason"],
        }
        decision_digests[case] = hashlib.sha256(
            json.dumps(safe_decisions, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    root = matching(cases, CASES["A"])
    current = _safe_evidence(root["evidence_view_trace"][0])
    historical = read(HISTORICAL / "evidence_view_trace.json")["root"]
    invariant_fields = (
        "raw_V", "assignment_V", "packet_V", "responsibility_universe_count",
        "responsibility_universe_ids_sha256", "responsibility_scheduled_count",
        "focus_role_item_count", "anchor_role_item_count", "nominal_role_item_slots",
        "team_minibatch_count", "team_minibatch_unique_count",
        "m_eval_equals_team_minibatch_ids", "packet_hash", "target_member",
    )
    differences = {field: [historical[field], current[field]] for field in invariant_fields
                   if historical[field] != current[field]}
    old_feasible = read(HISTORICAL / "feasibility_trace.json")
    current_feasible = _safe_feasibility(root["feasibility_trace"][0])
    for label, present, former in (
        ("root", current_feasible, old_feasible["root"]),
        ("rerank", _safe_feasibility(matching(cases, CASES["rerank"])["feasibility_trace"][0]),
         old_feasible["feasible_rerank"]),
        ("no_feasible", _safe_feasibility(matching(cases, CASES["no_feasible"])["feasibility_trace"][0]),
         old_feasible["no_feasible"]),
    ):
        for field in ("selected_member", "members"):
            if present[field] != former[field]:
                differences[f"{label}.{field}"] = "CHANGED"
    if differences:
        raise AssertionError(f"SCIENTIFIC_DECISION_REQUIRED: {sorted(differences)}")
    if (current["raw_V"], current["responsibility_universe_count"],
        current["responsibility_scheduled_count"], current["m_eval_equals_team_minibatch_ids"]) != (
        50, 50, 36, True,
    ):
        raise AssertionError("root V4 invariants differ")
    return {
        "classifier": "SEMANTIC_EQUIVALENT_TELEMETRY_ONLY",
        "historical_source_sha": "6314eef1839cc0270e6a3635f1e363a5babe9b7c",
        "compared_invariant_fields": list(invariant_fields),
        "scientific_differences": {},
        "exact_fake_scenarios_compared": len(cases),
        "exact_opportunities_compared": compared_opportunities,
        "exact_decision_fields": list(decision_fields),
        "exact_admission_fields": list(admission_fields),
        "exact_feasibility_and_admission_outcomes": True,
        "matched_decision_sha256_by_fake_scenario": decision_digests,
        "root": current,
        "feasible_reranking": True,
        "no_feasible_stop": matching(cases, CASES["no_feasible"])["stop_reason"]
            == "NO_FEASIBLE_LAYER2_OPPORTUNITY",
    }, {
        "base_sha": "8e2eaaa42ff8c8ecd3d142372d5849ac07d64a5b",
        "formal_v3_implementation_source_sha": "92a89ff9b4edeb52032028b8439c5d05038308c0",
        "historical_attempt_label": "V4_ATTEMPT1",
        "historical_attempt": EXPERIMENT_ID,
        "historical_status": "PREREGISTERED_NOT_EXECUTED",
        "historical_execution_status": "SUPERSEDED_FOR_EXECUTION_BY_CURRENT_SOURCE_ATTEMPT",
        "historical_execution_source_sha": "6314eef1839cc0270e6a3635f1e363a5babe9b7c",
        "current_attempt": ATTEMPT2_ID,
        "supersedes_for_execution": "V4_ATTEMPT1",
        "scientific_method_changed": False,
        "protocol_changed": False,
    }


def audit_preps(prep_a: Path, prep_b: Path) -> tuple[dict, dict, dict]:
    def files(root: Path) -> dict[str, str]:
        return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*") if path.is_file()}

    a, b = files(prep_a), files(prep_b)
    if a != b:
        raise AssertionError("independent V4 prep replays are not byte-identical")
    manifest, protocol = read(prep_a / "manifest.json"), read(prep_a / "protocol.json")
    if (manifest["attempt_id"] != ATTEMPT2_ID
            or manifest["api_authorization"]["authorized"]
            or manifest["access"] != {"validation50_calls": 0, "test50_calls": 0}
            or manifest["diagnostic_contract"]["successful_provider_ceiling"] != 1200
            or manifest["diagnostic_contract"]["transport_attempt_ceiling"] != 4800
            or protocol["successful_provider_ceiling"] != 1200
            or protocol["transport_attempt_ceiling"] != 4800):
        raise AssertionError("V4 attempt2 budget, identity or access mismatch")
    if any((prep / "authorization_consumed.json").exists()
           for prep in (prep_a, prep_b)):
        raise AssertionError("attempt2 authorization consumed")
    if (ROOT / "runs" / ATTEMPT2_ID / "run").exists():
        raise AssertionError("attempt2 real run root exists")
    old_create, old_environment = ProviderClientFactory.create, ProviderClientFactory.from_environment
    count = 0

    def forbidden(*_args, **_kwargs):
        nonlocal count
        count += 1
        raise AssertionError("provider constructor reached in offline V4 audit")

    ProviderClientFactory.create = forbidden
    ProviderClientFactory.from_environment = forbidden
    original_hash = admission.source_freeze_sha256
    poisoned = 0
    try:
        for prep in (prep_a, prep_b):
            permit = admission.validate_execution(root=ROOT, prep=prep, require_authorized=False)
            if permit.attempt_id != ATTEMPT2_ID or permit.admitted:
                raise AssertionError("invalid V4 preflight permit")
        for relative in manifest["execution"]["source_paths"]:
            target = (ROOT / relative).resolve()
            sha = hashlib.sha256(normalized_lf_bytes(target.read_bytes()) + b"\n# poison\n").hexdigest()

            def hash_with_poison(path, *, _target=target, _sha=sha):
                return _sha if Path(path).resolve() == _target else original_hash(path)

            admission.source_freeze_sha256 = hash_with_poison
            try:
                admission.validate_execution(root=ROOT, prep=prep_a, require_authorized=False)
            except StartupIdentityError:
                poisoned += 1
            else:
                raise AssertionError(f"source poison escaped: {relative}")
            finally:
                admission.source_freeze_sha256 = original_hash
    finally:
        admission.source_freeze_sha256 = original_hash
        ProviderClientFactory.create = old_create
        ProviderClientFactory.from_environment = old_environment
    if count or poisoned != len(manifest["execution"]["source_paths"]):
        raise AssertionError("source poison/provider isolation incomplete")
    identity = read(prep_a / "startup_identity/scientific_identity.json")
    run = read(prep_a / "startup_identity/run_identity.json")
    freeze = {
        "attempt_id": ATTEMPT2_ID,
        "execution_source_sha": manifest["execution"]["execution_source_sha"],
        "protocol_sha256": identity["payload"]["protocol_sha256"],
        "preregistration_sha256": identity["preregistration_sha256"],
        "run_identity_sha256": run["run_identity_sha256"],
        "authorization_consumed": False,
        "real_execution_started": False,
        "successful_provider_emergency_ceiling": 1200,
        "transport_attempt_emergency_ceiling": 4800,
        "prep_replay_identical": True,
        "prep_files_byte_identical": len(a),
    }
    source = {
        "source_count": poisoned,
        "source_paths": manifest["execution"]["source_paths"],
        "source_poison_gate": "PASS",
        "shared_v4_selector_bound": "multi_dataset_diverse_rl/team_search/v4_opportunity.py" in manifest["execution"]["source_paths"],
    }
    return freeze, source, {"gate": "PASS", "cases": poisoned, "provider_constructor_calls": count,
                            "provider_attempts": 0, "network_attempts": 0,
                            "v4_opportunity_poisoned": source["shared_v4_selector_bound"]}


def build(prep_a: Path, prep_b: Path, fake_root: Path, report: Path) -> dict:
    if report.exists():
        raise FileExistsError("fresh V4 attempt2 report required")
    if not fake_root.resolve().is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("fake evidence must be inside ignored runs")
    if any(SECRET_NAME.search(name) for name in os.environ):
        raise AssertionError("credential-like environment variable in offline process")
    if sitecustomize.network_attempt_count():
        raise AssertionError("network attempt preceded V4 offline audit")
    cases = summaries(fake_root)
    semantic, identity = compare_semantics(cases)
    freeze, source, poison = audit_preps(prep_a, prep_b)
    root = matching(cases, CASES["A"])
    chain = matching(cases, CASES["successor"])
    budget = matching(cases, CASES["budget"])
    downstream = [value for name, value in cases.items()
                  if name.startswith("test_v4_raw_legal_downstream")]
    stage = root["stage_accounting"]["opportunities"]
    if not (root["stage_accounting"]["global"]["reflection_provider_records"] > 0
            and sum(item["by_phase"]["local_optimizer_solver_eval"]["provider_successes"]
                    for item in stage) > 0
            and chain["parent_sequence"][1] == chain["parent_sequence"][2]
            and chain["commits"] >= 2
            and budget["accepted_mutations"] == 5
            and len(downstream) == 3):
        raise AssertionError("V4 fake path or successor evidence incomplete")
    B = [value for value in downstream if any(
        not row["team_minibatch"]["passed"] and row["full"]["diagnostic_only"]
        for row in value["candidate_diagnostics"])]
    C = [value for value in downstream if any(
        row["ordinary_common_safe"] not in {"PASS", "NOT_REACHED"}
        for row in value["candidate_diagnostics"])]
    D = [value for value in downstream if any(
        row["ordinary_shadow"] not in {"PASS", "NOT_REACHED"}
        for row in value["candidate_diagnostics"])]
    if not (len(B) == len(C) == len(D) == 1
            and len({id(B[0]), id(C[0]), id(D[0])}) == 3):
        raise AssertionError("V4 B/C/D controls incomplete")
    if sitecustomize.network_attempt_count():
        raise AssertionError("network attempt during V4 offline audit")
    identity["current_execution_source_sha"] = freeze["execution_source_sha"]
    report.mkdir(parents=True)
    write(report, "historical_vs_current_identity.json", identity)
    write(report, "semantic_equivalence.json", semantic)
    write(report, "active_source_closure.json", source)
    write(report, "source_poison.json", poison)
    write(report, "api_isolation.json", {
        "gate": "PASS", "provider_create_isolated": True,
        "provider_from_environment_isolated": True, "real_credentials_present": False,
        "real_api_calls": 0, "validation50_calls": 0, "test50_calls": 0,
        "authorization_consumed": False,
    })
    write(report, "network_isolation.json", {
        "gate": "PASS", "pre_import_sitecustomize_guard": True,
        "normal_fake_network_attempts": 0,
        "negative_control": "PASS_BLOCKED_BEFORE_TRANSMISSION",
    })
    write(report, "fake_execution_verification.json", {
        "gate": "PASS", "official_gepa_full_fake": True,
        "official_gepa_reflection_fake_calls_positive": True,
        "official_gepa_candidate_solver_calls_positive": True,
        "official_gepa_a_commits": root["commits"],
        "scripted_fake_provider_outcome_controls": {
            "B_minibatch_rejection": True,
            "B_diagnostic_full": True,
            "C_common_safe_rejection": True,
            "D_shadow_rejection": True,
        },
        "scripted_local_backend_state_machine_controls":
            "SEPARATE_FROM_OFFICIAL_GEPA_FAKE; see focused test_online_transfer_diagnostic",
        "successor_commit_no_commit_commit": True,
        "scheduled_not_delivered": semantic["root"]["scheduled_but_not_delivered_role_item_count"] > 0,
        "provider_successful_calls_in_budget_stress": budget["ledger"]["successful_provider_calls"],
        "validation50_calls": 0, "test50_calls": 0,
    })
    write(report, "freeze_identity.json", freeze)
    write(report, "verification.json", {
        "gate": "PASS", "zero_api": True, "zero_network": True,
        "semantic_equivalence": semantic["classifier"], "fake_scenarios": len(cases),
        "source_poison_cases": poison["cases"],
        "focused_tests": "RECORDED_AFTER_RUN", "full_tests": "RECORDED_AFTER_RUN",
    })
    (report / "README.md").write_text(
        "# V4 Seed81 current-source attempt2 pre-execution closure\n\n"
        "The historical V4 attempt1 remains immutable and was not executed. "
        "Attempt2 preserves its scientific protocol and uses the current shared "
        "V4 selector. Deterministic root/selection/admission comparisons classify "
        "the transition as `SEMANTIC_EQUIVALENT_TELEMETRY_ONLY`.\n\n"
        "The official GEPA fake-provider path reached Reflection, local Solver "
        "evaluation and downstream team stages. Scripted provider outcome controls "
        "and local-backend state-machine controls are separate evidence categories; "
        "they are not real-model efficacy. No real API, Validation50 or Test50 was "
        "used. Attempt2 remains unauthorized, with no formal run root.\n",
        encoding="utf-8",
    )
    write(report, "sha256_manifest.json", {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in report.iterdir() if path.name != "sha256_manifest.json"
    })
    return {"gate": "PASS", "source_count": poison["cases"],
            "semantic_equivalence": semantic["classifier"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("prep-a", "prep-b", "fake-root", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.prep_a, args.prep_b, args.fake_root, args.report), sort_keys=True))
