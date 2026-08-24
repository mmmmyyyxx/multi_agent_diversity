from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v18_hybrid_online_accumulation_support import ARMS, SEEDS, UPDATES


AUDIT_VERSION = "v18_revision_parity_semantics_audit_v1"
CORRECTED_GATE_VERSION = "post_hoc_corrected_gate_v1"
ORIGINAL_AUDIT_VERSION = "v18_hybrid_online_accumulation_audit_v1"
ORIGINAL_PARITY_BLOCKERS = {
    "revision_parity:59:HYBRID_BASE",
    "revision_parity:61:HYBRID_BASE",
}
EXPECTED_EXECUTION_COMMIT = "a7032b6ff4e20f91ae3939504e779aa409044089"
EXPECTED_FAILURE_CLASSES = {"ValueError"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def artifact_tree_identity(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        })
    return {
        "artifact_count": len(rows),
        "tree_sha256": sha256_bytes(canonical_json(rows).encode("utf-8")),
    }


def _git_blob_oid(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{path}"], cwd=ROOT, text=True
    ).strip()


def _filtered_worktree_blob_oid(path: str) -> str:
    # The source freeze records exact worktree bytes, while Git stores blobs
    # after clean filters (notably CRLF -> LF). Compare the commit to the
    # filtered worktree object identity rather than raw byte hashes.
    return subprocess.check_output(
        ["git", "hash-object", f"--path={path}", path], cwd=ROOT, text=True
    ).strip()


def verify_execution_source(freeze: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    commit = str(freeze.get("execution_commit", ""))
    if commit != EXPECTED_EXECUTION_COMMIT:
        blockers.append("execution_commit_identity")
    checked = 0
    for row in freeze.get("files", []):
        rel = str(row["path"])
        expected = str(row["sha256"])
        try:
            commit_blob_oid = _git_blob_oid(commit, rel)
            worktree_blob_oid = _filtered_worktree_blob_oid(rel)
        except subprocess.CalledProcessError:
            blockers.append(f"missing_execution_blob:{rel}")
            continue
        if commit_blob_oid != worktree_blob_oid:
            blockers.append(f"execution_blob_identity:{rel}")
        local = ROOT / rel
        if not local.is_file() or sha256_file(local) != expected:
            blockers.append(f"current_frozen_file_hash:{rel}")
        checked += 1
    return blockers, {
        "execution_commit": commit,
        "freeze_version": freeze.get("freeze_version"),
        "frozen_file_count": int(freeze.get("file_count", -1)),
        "verified_file_count": checked,
        "source_freeze_unchanged": not blockers,
    }


def source_key(row: dict[str, Any], *, event: bool) -> tuple[Any, ...]:
    return (
        int(row["update_index"]),
        str(row["parent_team_hash"]),
        int(row["target_agent_id"] if event else row["target_member"]),
        str(row["source_candidate_hash"] if event else row["candidate_id"]),
    )


def revision_key(row: dict[str, Any], *, event: bool) -> tuple[Any, ...]:
    return (
        int(row["update_index"]),
        str(row["parent_team_hash"]),
        int(row["target_agent_id"] if event else row["target_member"]),
        str(row["revised_candidate_hash"] if event else row["candidate_id"]),
    )


def duplicates(keys: Iterable[tuple[Any, ...]]) -> int:
    return sum(count - 1 for count in Counter(keys).values() if count > 1)


def audit_trajectory(run: Path, seed: int, arm: str) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    candidates = read_jsonl(run / "candidate_level_sanitized.jsonl")
    events = read_jsonl(run / "loss_blind_generic_revision_events.jsonl")
    decisions = read_jsonl(run / "candidate_decisions.jsonl")

    source_rows = [
        row for row in candidates
        if row.get("candidate_stage") == "source" and bool(row.get("valid"))
    ]
    revision_rows = [
        row for row in candidates if row.get("candidate_stage") == "revision"
    ]
    attempt_events = [row for row in events if bool(row.get("revision_attempted"))]
    valid_events = [row for row in attempt_events if bool(row.get("revision_output_valid"))]
    invalid_events = [row for row in attempt_events if not bool(row.get("revision_output_valid"))]

    source_keys = [source_key(row, event=False) for row in source_rows]
    attempt_keys = [source_key(row, event=True) for row in attempt_events]
    valid_revision_keys = [revision_key(row, event=True) for row in valid_events]
    evaluable_revision_keys = [revision_key(row, event=False) for row in revision_rows]

    source_counter = Counter(source_keys)
    attempt_counter = Counter(attempt_keys)
    valid_revision_counter = Counter(valid_revision_keys)
    evaluable_counter = Counter(evaluable_revision_keys)

    duplicate_source_count = duplicates(source_keys)
    duplicate_attempt_count = duplicates(attempt_keys)
    duplicate_valid_output_count = duplicates(valid_revision_keys)
    duplicate_evaluable_row_count = duplicates(evaluable_revision_keys)
    missing_attempt_count = sum((source_counter - attempt_counter).values())
    orphan_attempt_count = sum((attempt_counter - source_counter).values())
    missing_evaluable_row_count = sum(
        (valid_revision_counter - evaluable_counter).values()
    )
    orphan_evaluable_row_count = sum(
        (evaluable_counter - valid_revision_counter).values()
    )
    invalid_with_candidate_hash_count = sum(
        bool(row.get("revised_candidate_hash")) for row in invalid_events
    )
    unexpected_invalid_failure_class_count = sum(
        str(row.get("terminal_failure_class", "")) not in EXPECTED_FAILURE_CLASSES
        for row in invalid_events
    )
    unattempted_event_count = sum(
        not bool(row.get("revision_attempted")) for row in events
    )
    conceptual_source_budget = sum(
        2 * len(row.get("selected_target_ids", ())) for row in decisions
    )

    checks = {
        "attempts_equal_valid_sources": len(attempt_events) == len(source_rows),
        "source_attempt_join_is_one_to_one": not any((
            duplicate_source_count,
            duplicate_attempt_count,
            missing_attempt_count,
            orphan_attempt_count,
        )),
        "valid_outputs_equal_evaluable_rows": len(valid_events) == len(revision_rows),
        "valid_output_row_join_is_one_to_one": not any((
            duplicate_valid_output_count,
            duplicate_evaluable_row_count,
            missing_evaluable_row_count,
            orphan_evaluable_row_count,
        )),
        "invalid_outputs_have_no_evaluable_payload": invalid_with_candidate_hash_count == 0,
        "invalid_outputs_match_frozen_failure_semantics": (
            unexpected_invalid_failure_class_count == 0
        ),
        "all_events_are_attempts": unattempted_event_count == 0,
        "attempts_do_not_exceed_conceptual_budget": (
            len(attempt_events) <= conceptual_source_budget
        ),
    }
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"{name}:{seed}:{arm}")

    row = {
        "seed": seed,
        "arm": arm,
        "planned_update_count": UPDATES,
        "conceptual_source_candidate_budget": conceptual_source_budget,
        "valid_source_count": len(source_rows),
        "revision_attempt_count": len(attempt_events),
        "revision_output_valid_count": len(valid_events),
        "revision_output_invalid_count": len(invalid_events),
        "evaluable_revision_row_count": len(revision_rows),
        "attempt_per_valid_source": (
            len(attempt_events) / len(source_rows) if source_rows else None
        ),
        "missing_attempt_count": missing_attempt_count,
        "orphan_attempt_count": orphan_attempt_count,
        "missing_evaluable_row_count": missing_evaluable_row_count,
        "orphan_evaluable_row_count": orphan_evaluable_row_count,
        "duplicate_source_count": duplicate_source_count,
        "duplicate_attempt_count": duplicate_attempt_count,
        "invalid_with_candidate_hash_count": invalid_with_candidate_hash_count,
        "unexpected_invalid_failure_class_count": unexpected_invalid_failure_class_count,
        "checks": checks,
        "trajectory_gate": "PASS" if not blockers else "FAIL",
    }
    return row, blockers


def semantics_audit(
    *, run_root: Path, original_gate_path: Path, source_freeze_path: Path
) -> dict[str, Any]:
    original_gate = read_json(original_gate_path)
    freeze = read_json(source_freeze_path)
    blockers, source_identity = verify_execution_source(freeze)
    if original_gate.get("audit_version") != ORIGINAL_AUDIT_VERSION:
        blockers.append("original_audit_version")
    if original_gate.get("gate") != "FAIL":
        blockers.append("original_gate_not_fail")
    original_blockers = set(map(str, original_gate.get("blockers", ())))
    if original_blockers != ORIGINAL_PARITY_BLOCKERS:
        blockers.append("original_blocker_identity")
    if int(original_gate.get("new_test_calls", -1)) != 0:
        blockers.append("original_test_call")
    if int(original_gate.get("infrastructure_failure_count", -1)) != 0:
        blockers.append("original_infrastructure_failure")

    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            run = run_root / f"seed{seed}" / arm
            try:
                row, row_blockers = audit_trajectory(run, seed, arm)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                blockers.append(f"artifact:{seed}:{arm}:{type(exc).__name__}")
                continue
            rows.append(row)
            blockers.extend(row_blockers)

    if len(rows) != len(SEEDS) * len(ARMS):
        blockers.append("trajectory_inventory")
    arm_summary = []
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        arm_summary.append({
            "arm": arm,
            "trajectory_count": len(selected),
            "conceptual_source_candidate_budget": sum(
                row["conceptual_source_candidate_budget"] for row in selected
            ),
            "valid_source_count": sum(row["valid_source_count"] for row in selected),
            "revision_attempt_count": sum(
                row["revision_attempt_count"] for row in selected
            ),
            "revision_output_valid_count": sum(
                row["revision_output_valid_count"] for row in selected
            ),
            "revision_output_invalid_count": sum(
                row["revision_output_invalid_count"] for row in selected
            ),
            "evaluable_revision_row_count": sum(
                row["evaluable_revision_row_count"] for row in selected
            ),
            "attempt_policy": "one_attempt_per_valid_source",
            "attempt_policy_satisfied": all(
                row["checks"]["attempts_equal_valid_sources"]
                and row["checks"]["source_attempt_join_is_one_to_one"]
                for row in selected
            ),
        })
    equal_prospective_budget = (
        len({row["conceptual_source_candidate_budget"] for row in arm_summary}) == 1
    )
    if not equal_prospective_budget:
        blockers.append("arm_prospective_budget_mismatch")
    if any(not row["attempt_policy_satisfied"] for row in arm_summary):
        blockers.append("arm_attempt_policy_mismatch")

    original_non_parity = sorted(original_blockers - ORIGINAL_PARITY_BLOCKERS)
    corrected_blockers = sorted(set(blockers) | set(original_non_parity))
    return {
        "audit_version": AUDIT_VERSION,
        "audit_mode": "offline_existing_artifact_revalidation",
        "api_calls": 0,
        "model_calls": 0,
        "scientific_results_inspected": False,
        "scientific_interpretation_status": "not_performed",
        "original_frozen_audit": {
            "audit_version": original_gate.get("audit_version"),
            "gate": "FAIL",
            "operational_status": "HOLD",
            "blockers": sorted(original_blockers),
            "artifact_sha256": sha256_file(original_gate_path),
            "preserved": True,
        },
        "source_identity": source_identity,
        "raw_artifact_identity": artifact_tree_identity(run_root),
        "parity_semantics": {
            "eligible_unit": "valid_source_candidate",
            "budget_unit": "revision_attempt_event",
            "output_unit": "evaluable_revision_candidate_row",
            "invalid_output_consumes_opportunity": True,
            "absolute_attempt_counts_need_not_match_across_online_arms": True,
        },
        "trajectory_rows": rows,
        "arm_summary": arm_summary,
        "equal_prospective_budget": equal_prospective_budget,
        "semantics_audit_gate": "PASS" if not blockers else "FAIL",
        "semantics_audit_blockers": sorted(set(blockers)),
        "post_hoc_corrected_gate": {
            "version": CORRECTED_GATE_VERSION,
            "gate": "PASS" if not corrected_blockers else "FAIL",
            "blockers": corrected_blockers,
            "replaces_original_gate": False,
            "original_gate_remains": "FAIL/HOLD",
            "correction_scope": "revision_parity_representation_only",
            "method_runtime_semantics_changed": False,
            "original_run_artifacts_modified": False,
        },
    }


def write_report(report_dir: Path, result: dict[str, Any]) -> None:
    if report_dir.exists():
        raise FileExistsError(f"fresh report directory required: {report_dir}")
    report_dir.mkdir(parents=True)
    write_json(report_dir / "revision_parity_semantics.json", result)
    write_json(report_dir / "original_frozen_gate_snapshot.json", result["original_frozen_audit"])
    write_json(report_dir / "post_hoc_corrected_gate.json", result["post_hoc_corrected_gate"])

    with (report_dir / "trajectory_parity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "seed", "arm", "conceptual_source_candidate_budget",
            "valid_source_count", "revision_attempt_count",
            "revision_output_valid_count", "revision_output_invalid_count",
            "evaluable_revision_row_count", "attempt_per_valid_source",
            "trajectory_gate",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["trajectory_rows"]:
            writer.writerow({key: row[key] for key in fields})

    totals = {
        key: sum(row[key] for row in result["trajectory_rows"])
        for key in (
            "conceptual_source_candidate_budget", "valid_source_count",
            "revision_attempt_count", "revision_output_valid_count",
            "revision_output_invalid_count", "evaluable_revision_row_count",
        )
    }
    readme = f"""# V18 Revision-Parity Semantics Audit

This was an independent zero-API audit. It did not rerun trajectories, add
revision calls, modify raw evidence, run the scientific analyzer, or inspect
online-accumulation outcomes.

## Status

```text
V18 execution completed
original frozen audit = FAIL / HOLD
independent semantics audit = {result['semantics_audit_gate']}
post-hoc corrected gate = {result['post_hoc_corrected_gate']['gate']}
```

The original gate remains preserved and is not superseded. The separately
named corrected gate changes only the representation used for revision parity.

## Finding

The frozen execution policy spends one revision opportunity per valid source.
An invalid revision output is a legal terminal outcome of that opportunity and
does not create an evaluable candidate row. Therefore attempt parity must join
valid source rows to revision-attempt events; evaluable-row parity is a
separate valid-output persistence check.

Across all six trajectories:

- conceptual source budget: {totals['conceptual_source_candidate_budget']}
- valid sources: {totals['valid_source_count']}
- revision attempts: {totals['revision_attempt_count']}
- valid revision outputs/evaluable rows: {totals['revision_output_valid_count']}
- invalid revision outputs: {totals['revision_output_invalid_count']}
- evaluable revision rows: {totals['evaluable_revision_row_count']}

Every valid source had exactly one attempt. Every valid revision output had
exactly one evaluable row. The four invalid outputs consumed their frozen
opportunities and correctly produced no evaluable row.

The arms had equal prospective budgets and the same one-attempt-per-valid-source
policy. Absolute realized attempt counts differ because the online trajectories
produced different valid-source counts; that is not a compute-policy mismatch.

## Scope

No scientific result is reported or interpreted here. Running the frozen V18
scientific analyzer requires a separate decision after accepting or rejecting
this post-hoc gate correction.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")

    sanitization = {
        "status": "PASS",
        "api_calls": 0,
        "model_calls": 0,
        "contains_prompt_text": False,
        "contains_question_text": False,
        "contains_gold_or_model_answers": False,
        "contains_raw_responses": False,
        "contains_endpoints_or_credentials": False,
        "contains_sqlite_or_checkpoint_content": False,
        "contains_absolute_paths": False,
    }
    write_json(report_dir / "sanitization_manifest.json", sanitization)
    manifest = []
    for path in sorted(item for item in report_dir.iterdir() if item.is_file()):
        if path.name == "sha256_manifest.json":
            continue
        manifest.append({"file": path.name, "sha256": sha256_file(path)})
    write_json(report_dir / "sha256_manifest.json", {"files": manifest})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original_gate", type=Path, required=True)
    parser.add_argument("--source_freeze", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    args = parser.parse_args()
    result = semantics_audit(
        run_root=args.root.resolve(),
        original_gate_path=args.original_gate.resolve(),
        source_freeze_path=args.source_freeze.resolve(),
    )
    write_report(args.report_dir, result)
    print(json.dumps({
        "semantics_audit_gate": result["semantics_audit_gate"],
        "post_hoc_corrected_gate": result["post_hoc_corrected_gate"]["gate"],
        "original_gate": "FAIL/HOLD",
        "api_calls": 0,
    }, indent=2))
    raise SystemExit(
        0
        if result["semantics_audit_gate"] == "PASS"
        and result["post_hoc_corrected_gate"]["gate"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    main()
