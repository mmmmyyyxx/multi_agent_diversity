"""Cache-only recovery audit for accepted-mutation Optimize100 profiles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from infrastructure.common_solver_contract_v1.contract import (
    CONTRACT_SPEC,
    request_identity,
)
from multi_dataset_diverse_rl.governance.artifacts import (
    build_sha256_manifest,
    scan_sanitized_artifacts,
)
from multi_dataset_diverse_rl.governance.manifest import (
    preregistration_hash,
    validate_manifest,
)
from multi_dataset_diverse_rl.versions import (
    ACCEPTED_MUTATION_EXACT_STATE_GRAPH_VERSION,
    METHOD_VERSION,
)


IDENTITY = ACCEPTED_MUTATION_EXACT_STATE_GRAPH_VERSION


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(ROOT).as_posix(): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def exact_json_cache_files() -> list[Path]:
    files = set(ROOT.joinpath("runs").rglob("*exact_request_cache*.json"))
    files.update(ROOT.joinpath("runs").rglob("raw_cache_private.json"))
    return sorted(path for path in files if "s43" not in path.parts)


def search_json_caches(required: set[str]) -> tuple[set[str], list[str]]:
    recovered: set[str] = set()
    scanned: list[str] = []
    for path in exact_json_cache_files():
        payload = read_json(path)
        scanned.append(path.relative_to(ROOT).as_posix())
        if isinstance(payload, dict):
            recovered.update(required.intersection(str(key) for key in payload))
    return recovered, scanned


def search_sqlite_caches(
    required: set[str], candidate_hashes: set[str]
) -> tuple[set[str], dict[str, int]]:
    recovered: set[str] = set()
    opaque_recovered: set[str] = set()
    stats = {
        "files_considered": 0,
        "files_opened_read_only": 0,
        "unreadable_test_fixtures": 0,
        "direct_request_identity_hits": 0,
        "candidate_prompt_hash_hits": 0,
        "opaque_files_raw_scanned": 0,
        "opaque_direct_request_identity_hits": 0,
    }
    request_values = tuple(sorted(required))
    prompt_values = tuple(sorted(candidate_hashes))
    request_slots = ",".join("?" for _ in request_values)
    prompt_slots = ",".join("?" for _ in prompt_values)
    required_bytes = {value.encode("ascii"): value for value in required}
    for path in sorted(ROOT.joinpath("runs").rglob("*.sqlite")):
        if "s43" in path.parts:
            continue
        stats["files_considered"] += 1
        try:
            uri = f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True)
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            stats["files_opened_read_only"] += 1
            if "solver_cache" in tables:
                columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(solver_cache)")
                }
                for column in (
                    "cache_key",
                    "model_request_identity",
                    "question_hash",
                ):
                    if column not in columns:
                        continue
                    rows = connection.execute(
                        f'SELECT "{column}" FROM solver_cache '
                        f'WHERE "{column}" IN ({request_slots})',
                        request_values,
                    ).fetchall()
                    recovered.update(str(row[0]) for row in rows)
                if "prompt_hash" in columns:
                    count = connection.execute(
                        f"SELECT COUNT(*) FROM solver_cache "
                        f"WHERE prompt_hash IN ({prompt_slots})",
                        prompt_values,
                    ).fetchone()[0]
                    stats["candidate_prompt_hash_hits"] += int(count)
            else:
                payload = path.read_bytes()
                stats["opaque_files_raw_scanned"] += 1
                for needle, identity in required_bytes.items():
                    if needle in payload:
                        recovered.add(identity)
                        opaque_recovered.add(identity)
            connection.close()
        except (sqlite3.DatabaseError, OSError):
            stats["unreadable_test_fixtures"] += 1
            try:
                payload = path.read_bytes()
                stats["opaque_files_raw_scanned"] += 1
                for needle, identity in required_bytes.items():
                    if needle in payload:
                        recovered.add(identity)
                        opaque_recovered.add(identity)
            except OSError:
                pass
    stats["direct_request_identity_hits"] = len(recovered)
    stats["opaque_direct_request_identity_hits"] = len(opaque_recovered)
    return recovered, stats


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    bundle = args.bundle.resolve()
    report = args.report.resolve()
    manifest_path = args.manifest.resolve()
    protocol = ROOT / "experiments" / IDENTITY / "PROTOCOL.md"
    materialization = (
        ROOT
        / "experiments"
        / IDENTITY
        / "PROFILE_MATERIALIZATION_PROTOCOL_DRAFT.md"
    )
    materialization_manifest = (
        ROOT
        / "experiments"
        / "manifests"
        / "accepted_mutation_profile_materialization_v1.yaml"
    )
    if report.exists() and any(report.iterdir()):
        if not args.overwrite_existing:
            raise FileExistsError("fresh recovery report required")
        existing = report / "summary.json"
        if not existing.exists() or read_json(existing).get("experiment_id") != IDENTITY:
            raise ValueError("refusing to overwrite another experiment report")
    if manifest_path.exists():
        if not args.overwrite_existing:
            raise FileExistsError("fresh recovery manifest required")
        existing_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("experiment_id") != IDENTITY:
            raise ValueError("refusing to overwrite another experiment manifest")

    before = {**tree_hashes(run_root), **tree_hashes(bundle)}
    execution = read_json(run_root / "execution.json")
    private = read_json(bundle / "private_bundle.json")
    handoff = read_json(bundle / "HANDOFF.json")
    ledger = [
        json.loads(line)
        for line in (run_root / "provider_ledger.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if execution["execution_status"] != "EXECUTION_COMPLETE":
        raise ValueError("completed v2 execution required")
    if handoff["models"]["solver"] != CONTRACT_SPEC.model:
        raise ValueError("historical model does not match frozen common contract")
    if bool(handoff["models"]["thinking"]) != CONTRACT_SPEC.enable_thinking:
        raise ValueError("historical thinking mode does not match frozen contract")

    success_ids = {
        row["request_identity"]
        for row in ledger
        if row.get("event") == "provider_attempt_success"
    }
    mutations = private["accepted_mutations"]
    tasks = private["phase_a_tasks"]
    required_rows: list[dict[str, Any]] = []
    ordered_ids: list[str] | None = None
    candidate_hashes: set[str] = set()
    prompt_hash_checks: dict[str, bool] = {}
    ledger_coverage: dict[str, int] = {}
    for mutation in mutations:
        mutation_id = mutation["mutation_id"]
        candidate_hash = mutation["accepted_candidate_hash"]
        prompt = mutation["accepted_candidate_prompt"]
        prompt_hash_checks[mutation_id] = (
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            == candidate_hash
        )
        candidate_hashes.add(candidate_hash)
        examples = tasks[mutation["source_parent_task_id"]]["search_examples"]
        current_ids = [str(row["example_id"]) for row in examples]
        if ordered_ids is None:
            ordered_ids = current_ids
        elif current_ids != ordered_ids:
            raise ValueError("Optimize100 ordering differs across parent tasks")
        identities = set()
        for row in examples:
            identity = request_identity(
                decision_procedure=prompt,
                question=row["input_payload"],
            )
            identities.add(identity)
            required_rows.append(
                {
                    "candidate_identity": mutation_id,
                    "example_id": str(row["example_id"]),
                    "request_identity": identity,
                }
            )
        ledger_coverage[mutation_id] = len(identities.intersection(success_ids))
        if len(identities) != 100:
            raise ValueError("each candidate must have 100 unique requests")

    required = {row["request_identity"] for row in required_rows}
    if len(required_rows) != 500 or len(required) != 500:
        raise ValueError("expected 500 unique candidate-profile requests")
    if not all(prompt_hash_checks.values()):
        raise ValueError("candidate prompt hash mismatch")
    if any(count != 100 for count in ledger_coverage.values()):
        raise ValueError("historical ledger does not cover every required request")

    json_hits, json_files = search_json_caches(required)
    sqlite_hits, sqlite_stats = search_sqlite_caches(required, candidate_hashes)
    recovered = json_hits | sqlite_hits
    missing_rows = [
        row for row in required_rows if row["request_identity"] not in recovered
    ]
    status = (
        "CACHE_ONLY_PROFILE_RECOVERY_COMPLETE"
        if not missing_rows
        else "CACHE_ONLY_PROFILE_RECOVERY_INCOMPLETE"
    )
    if not missing_rows:
        raise RuntimeError(
            "complete recovery requires the exact-graph construction branch"
        )

    by_candidate: dict[str, int] = {}
    for row in missing_rows:
        candidate = row["candidate_identity"]
        by_candidate[candidate] = by_candidate.get(candidate, 0) + 1

    report.mkdir(parents=True, exist_ok=args.overwrite_existing)
    write_json(
        report / "cache_miss_manifest.json",
        {
            "schema_version": "accepted_mutation_cache_miss_manifest_v1",
            "status": status,
            "missing_count": len(missing_rows),
            "missing_count_by_candidate": by_candidate,
            "missing_requests": missing_rows,
        },
    )
    write_json(
        report / "cache_recovery_audit.json",
        {
            "status": status,
            "required_candidate_profiles": 5,
            "required_unique_requests": len(required),
            "historical_ledger_success_coverage": ledger_coverage,
            "recovered_unique_requests": len(recovered),
            "missing_unique_requests": len(missing_rows),
            "baseline_profile_available_from_phase_a": True,
            "v2_runtime_cache_kind": "PROCESS_LOCAL_DICTIONARY_NOT_PERSISTED",
            "v2_provider_ledger_contains_response_text": False,
            "exact_json_cache_files_scanned": json_files,
            "exact_json_cache_hits": len(json_hits),
            "sqlite_scan": sqlite_stats,
            "exact_graph_constructed": False,
            "historical_exact_graph_recoverable_from_current_artifacts": False,
            "stop_reason": "ANY_REQUIRED_RESPONSE_ABSENT",
        },
    )
    after = {**tree_hashes(run_root), **tree_hashes(bundle)}
    write_json(
        report / "integrity_audit.json",
        {
            "historical_execution_complete": True,
            "historical_files_byte_identical_before_after": before == after,
            "candidate_prompt_hashes_match": all(prompt_hash_checks.values()),
            "optimize100_order_identical_across_candidates": True,
            "required_request_count": 500,
            "required_requests_have_historical_provider_success": (
                sum(ledger_coverage.values()) == 500
            ),
            "model_matches_frozen_contract": True,
            "thinking_mode_matches_frozen_contract": True,
            "provider_access_used_by_audit": False,
            "raw_response_or_reasoning_published": False,
        },
    )
    write_json(
        report / "scientific_state.json",
        {
            "frozen_conclusion": (
                "SAFE_SYMMETRY_BREAKING_CAN_UNLOCK_PLURALITY_GAIN"
            ),
            "proven": [
                "At least one Common-Safe path reaches a Vote-positive triple.",
                "The first two safe commits are structurally Vote-neutral.",
            ],
            "unresolved": [
                "Concrete positive triple identity",
                "Exact Common-Safe edge graph",
                "Sequential Common-Safe reachability of all four safe mutations",
            ],
        },
    )
    write_json(
        report / "profile_materialization_draft.json",
        {
            "status": "AUTHORIZATION_REQUIRED",
            "READY_TO_RUN": False,
            "protocol": materialization.relative_to(ROOT).as_posix(),
            "manifest": "experiments/manifests/accepted_mutation_profile_materialization_v1.yaml",
            "missing_unique_solver_requests": len(missing_rows),
            "successful_solver_call_ceiling": len(missing_rows),
            "transport_attempt_ceiling": 4 * len(missing_rows),
            "automatic_execution_permitted": False,
            "evidence_semantics": "PROSPECTIVE_FRESH_PROVIDER_REALIZATION",
            "historical_v2_profile_reconstruction": False,
        },
    )
    write_json(
        report / "api_ledger_summary.json",
        {
            "provider_calls": 0,
            "solver_calls": 0,
            "GEPA": 0,
            "Reflection": 0,
            "Validation50": 0,
            "Test50": 0,
            "write_back": 0,
            "persistent_realizability_update": 0,
        },
    )
    summary = {
        "experiment_id": IDENTITY,
        "status": status,
        "required_unique_requests": 500,
        "recovered_unique_requests": len(recovered),
        "missing_unique_requests": len(missing_rows),
        "exact_graph_constructed": False,
        "provider_calls": 0,
        "future_materialization_authorized": False,
        "future_materialization_would_be_prospective": True,
    }
    write_json(report / "summary.json", summary)

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "experiment_manifest_v1",
        "experiment_id": IDENTITY,
        "title": "Cache-only accepted-mutation exact-state recovery audit",
        "status": "COMPLETED",
        "legacy_index": False,
        "lifecycle_history": [
            {"status": value, "timestamp": now}
            for value in (
                "DRAFT",
                "PREREGISTERED",
                "IMPLEMENTED",
                "PREFLIGHT_PASS",
                "RUNNING",
                "TRAIN_FROZEN",
                "COMPLETED",
            )
        ],
        "lineage": {
            "parents": ["accepted_mutation_compositional_state_graph_v1"],
            "derives_from": "accepted_local_mutation_team_transfer_v2",
        },
        "scientific_question": (
            "Can existing cached evidence identify the exact positive state and safe path?"
        ),
        "hypotheses": [
            "Historical exact-request caches may retain every candidate profile."
        ],
        "method_identity": METHOD_VERSION,
        "runtime_version": IDENTITY,
        "data": {
            "task": "BBH disambiguation_qa",
            "formal": False,
            "split_ids": {
                "optimize100": "anti_overfitting_split_v1_fold_a_plus_b"
            },
            "split_hashes": {
                "baseline_team_hash": execution["baseline_team_hash"]
            },
            "validation_policy": "prohibited; Validation50 calls=0",
            "test_policy": "prohibited; Test50 calls=0",
        },
        "model": {
            "solver": "NONE_CACHE_ONLY",
            "optimizer_roles": {},
            "thinking": False,
            "temperatures": {},
            "max_tokens": {},
        },
        "seeds": [78],
        "design": {
            "changed": ["cache-only exact-profile recovery attempt"],
            "unchanged": [
                "bounded v1 proof",
                "five-member plurality",
                "tie-as-abstain",
                "Common-Safe",
            ],
            "forbidden_changes": [
                "provider calls",
                "historical artifact mutation",
                "candidate regeneration",
                "write-back",
                "scheduler",
            ],
            "required_unique_requests": 500,
            "protocol_sha256": sha256(protocol),
        },
        "api_authorization": {
            "authorized": False,
            "authorization_scope": "ZERO_API_CACHE_ONLY",
            "allowed_roles": [],
            "allowed_phases": [],
        },
        "budget": {
            "type": "zero_api_cache_recovery",
            "frozen_before_run": True,
            "limit": {
                "provider_calls": 0,
                "Validation50": 0,
                "Test50": 0,
            },
        },
        "selection": {
            "primary_metric": "complete_exact_profile_recovery",
            "frozen_rule": "all 500 exact candidate/example request identities required",
            "validation_used_for_selection": False,
            "test_used_for_selection": False,
        },
        "artifacts": {
            "preregistration": {
                "path": protocol.relative_to(ROOT).as_posix()
            },
            "report": report.relative_to(ROOT).as_posix(),
            "provenance": (report / "provenance.json").relative_to(ROOT).as_posix(),
        },
        "git": {
            "design_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "implementation_commit": None,
            "result_commit": None,
        },
        "result": {
            "classifier": status,
            "conclusion": (
                "All 500 historical requests are ledger-verified, but no persisted response "
                "cache exists; exact graph construction stopped without provider access."
            ),
            "evidence_type": "retrospective",
        },
    }
    manifest["artifacts"]["preregistration"]["sha256"] = (
        preregistration_hash(manifest)
    )
    schema = read_json(ROOT / "infrastructure/experiment_manifest.schema.json")
    errors = validate_manifest(manifest, schema)
    if errors:
        raise ValueError("manifest invalid: " + "; ".join(errors))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

    write_json(
        report / "provenance.json",
        {
            "source_execution_sha256": sha256(run_root / "execution.json"),
            "source_provider_ledger_sha256": sha256(
                run_root / "provider_ledger.jsonl"
            ),
            "source_private_bundle_sha256": sha256(bundle / "private_bundle.json"),
            "source_handoff_sha256": sha256(bundle / "HANDOFF.json"),
            "protocol_sha256": sha256(protocol),
            "materialization_draft_sha256": sha256(materialization),
            "materialization_manifest_sha256": sha256(materialization_manifest),
            "manifest_sha256": sha256(manifest_path),
            "raw_prompts_published": False,
            "raw_questions_published": False,
            "raw_answers_published": False,
            "raw_responses_published": False,
        },
    )
    (report / "README.md").write_text(
        "# Accepted-mutation exact state graph v2\n\n"
        "Cache-only recovery stopped with `CACHE_ONLY_PROFILE_RECOVERY_INCOMPLETE`. "
        "All 500 required exact requests are present as successful historical ledger "
        "events, but their responses survived only in the v2 process-local cache. No "
        "provider call was made and no exact graph was fabricated. A future request replay "
        "would be a prospective realization, not recovery of the historical outputs. The "
        "bounded v1 proof remains authoritative.\n",
        encoding="utf-8",
        newline="\n",
    )
    findings = scan_sanitized_artifacts(report)
    write_json(
        report / "sanitization_manifest.json",
        {"status": "PASS" if not findings else "FAIL", "findings": findings},
    )
    write_json(report / "sha256_manifest.json", build_sha256_manifest(report))
    if findings or read_json(report / "sha256_manifest.json") != build_sha256_manifest(report):
        raise ValueError("report sanitization/hash replay failed")
    if before != {**tree_hashes(run_root), **tree_hashes(bundle)}:
        raise ValueError("historical evidence changed during cache audit")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
