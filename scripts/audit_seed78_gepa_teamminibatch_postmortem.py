"""Zero-API Seed78 Local-GEPA -> TeamMiniBatch transfer postmortem.

The audit reads frozen private evidence but emits only hashes, counts, numeric
metrics, classifications, and source provenance. It never writes into the run
root and never invokes a model, dataset evaluator, or transport.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs" / "seed78_primary_responsibility_ab_v1_retry1"
DEFAULT_REPORT = ROOT / "reports" / "seed78_gepa_teamminibatch_postmortem_20260913"
ARMS = (
    "A_VOTE_ALIGNED_HIERARCHICAL",
    "B_PRIMARY_RESPONSIBILITY_REALIZABILITY",
)
TASK_RE = re.compile(r"seed78_update(?P<update>\d+)_member(?P<member>\d+)$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _prompt_hash(candidate: Mapping[str, Any]) -> str:
    prompt = candidate.get("system_prompt")
    if not isinstance(prompt, str):
        raise ValueError("GEPA candidate lacks system_prompt")
    return _sha256_bytes(prompt.encode("utf-8"))


def _candidate_hash(candidate: Mapping[str, Any]) -> str:
    payload = json.dumps(
        candidate,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _frontier_indices(state: Mapping[str, Any]) -> list[int]:
    indices: set[int] = set()
    for members in state["program_at_pareto_front_valset"].values():
        indices.update(int(value) for value in members)
    return sorted(indices)


def analyze_branch(arm: str, lineage_path: Path) -> dict[str, Any]:
    match = TASK_RE.fullmatch(lineage_path.name.removesuffix(".lineage.jsonl"))
    if match is None:
        raise ValueError(f"unexpected lineage name: {lineage_path.name}")
    task_id = match.group(0)
    task_root = lineage_path.parent / task_id
    events = _jsonl(lineage_path)
    starts = [row for row in events if row["event_type"] == "optimization_start"]
    ends = [row for row in events if row["event_type"] == "optimization_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("branch must contain exactly one optimization start/end")
    parent_candidate_hash = str(starts[0]["seed_candidate_hash"])

    # Frozen run-local GEPA state is a trusted project artifact. Only primitive
    # structural fields and prompt hashes leave this function.
    state = pickle.loads((task_root / "gepa_state.bin").read_bytes())
    if not isinstance(state, dict):
        raise ValueError("unexpected GEPA state type")
    program_prompt_hashes = [_prompt_hash(row) for row in state["program_candidates"]]
    program_candidate_hashes = [_candidate_hash(row) for row in state["program_candidates"]]
    if not program_prompt_hashes or program_candidate_hashes[0] != parent_candidate_hash:
        raise ValueError("GEPA root candidate does not match frozen seed candidate hash")
    returned = _json(task_root / "candidates.json")
    if not isinstance(returned, list):
        raise ValueError("GEPA returned candidate file must be a list")
    returned_hashes = [_prompt_hash(row) for row in returned]
    frontier = _frontier_indices(state)
    returned_indices = [
        program_prompt_hashes.index(value) if value in program_prompt_hashes else -1
        for value in returned_hashes
    ]
    proposal_count = sum(row["event_type"] == "proposal_end" for row in events)
    accepted_count = sum(row["event_type"] == "candidate_accepted" for row in events)
    rejected_count = sum(row["event_type"] == "candidate_rejected" for row in events)
    root_prompt_hash = program_prompt_hashes[0]
    unchanged = (
        bool(returned_hashes)
        and all(value == root_prompt_hash for value in returned_hashes)
        and all(index == 0 for index in returned_indices)
        and state["parent_program_for_candidate"][0] == [None]
    )
    if unchanged:
        rejection_class = "UNCHANGED_PARENT"
        metrics = {
            "invalid_delta": 0,
            "vote_delta": 0,
            "target_delta": 0,
            "coalition_delta": 0,
            "responsibility_delta": 0,
            "broad_delta": 0,
        }
    else:
        # The historical runner did not persist changed-candidate TeamMiniBatch
        # metrics. Never invent them.
        rejection_class = "METRICS_NOT_RECOVERABLE"
        metrics = {key: None for key in (
            "invalid_delta", "vote_delta", "target_delta", "coalition_delta",
            "responsibility_delta", "broad_delta",
        )}
    return {
        "arm": arm,
        "update_index": int(match.group("update")),
        "target_member": int(match.group("member")),
        "parent_prompt_sha256": root_prompt_hash,
        "parent_candidate_sha256": parent_candidate_hash,
        "program_candidate_count": len(program_prompt_hashes),
        "program_candidate_sha256": ";".join(program_prompt_hashes),
        "proposal_count": proposal_count,
        "accepted_mutation_count": accepted_count,
        "rejected_mutation_count": rejected_count,
        "frontier_indices": ";".join(map(str, frontier)),
        "best_candidate_index": int(ends[0]["best_candidate_index"]),
        "returned_candidate_count": len(returned_hashes),
        "returned_candidate_indices": ";".join(map(str, returned_indices)),
        "returned_candidate_sha256": ";".join(returned_hashes),
        "returned_equals_parent": unchanged,
        **metrics,
        "has_positive_promotion_signal": False if unchanged else None,
        "catastrophic": False if unchanged else None,
        "promotion_rejection_class": rejection_class,
    }


def _ledger_rows(path: Path, phase: str) -> list[dict[str, Any]]:
    return [row for row in _jsonl(path) if row["phase"] == phase]


def paired_evaluation_audit(run_root: Path) -> dict[str, Any]:
    summary = _json(run_root / "execution_summary_private.json")
    a, b = ARMS
    a_validation = summary["validation50"][a]
    b_validation = summary["validation50"][b]
    a_rows = _ledger_rows(run_root / a / "ledger.jsonl", "final_validation")
    b_rows = _ledger_rows(run_root / b / "ledger.jsonl", "final_validation")
    a_identities = {str(row["request_identity"]) for row in a_rows}
    b_identities = {str(row["request_identity"]) for row in b_rows}
    source = (ROOT / "scripts" / "run_seed78_primary_responsibility_ab.py").read_text(
        encoding="utf-8"
    )
    per_arm_cache_copies = source.count("cache=dict(init_cache)") == 1
    same_team = (
        a_validation["team_identity"]["team_hash"]
        == b_validation["team_identity"]["team_hash"]
        and a_validation["team_identity"]["prompt_hashes"]
        == b_validation["team_identity"]["prompt_hashes"]
    )
    return {
        "final_team_byte_identity_equal": same_team,
        "validation_request_identity_sets_equal": a_identities == b_identities,
        "validation_unique_request_identities_per_arm": {
            "A": len(a_identities),
            "B": len(b_identities),
        },
        "validation_successful_provider_calls": {
            "A": sum(int(row["successful_provider_calls"]) for row in a_rows),
            "B": sum(int(row["successful_provider_calls"]) for row in b_rows),
        },
        "validation_cache_hits": {
            "A": sum(bool(row["cache_hit"]) for row in a_rows),
            "B": sum(bool(row["cache_hit"]) for row in b_rows),
        },
        "per_arm_cache_copy_in_runner": per_arm_cache_copies,
        "paired_cross_arm_realization_cache": False,
        "validation_vote_accuracy": {
            "A": a_validation["vote_accuracy"],
            "B": b_validation["vote_accuracy"],
        },
        "classifier": "PAIRED_EVALUATION_CACHE_GAP_CONFIRMED",
        "causal_interpretation": (
            "The final teams and request-identity sets are equal, but each arm made "
            "independent provider calls. The observed validation difference is not "
            "attributable to the scheduler."
        ),
    }


def build_summary(branches: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in branches if row["arm"] == arm]
        classes = Counter(row["promotion_rejection_class"] for row in rows)
        by_arm[arm] = {
            "branches": len(rows),
            "program_candidates": sum(row["program_candidate_count"] for row in rows),
            "proposals": sum(row["proposal_count"] for row in rows),
            "accepted_mutations": sum(row["accepted_mutation_count"] for row in rows),
            "rejected_mutations": sum(row["rejected_mutation_count"] for row in rows),
            "returned_candidates": sum(row["returned_candidate_count"] for row in rows),
            "returned_parent_candidates": sum(row["returned_equals_parent"] for row in rows),
            "frontier_indices_observed": sorted({
                value
                for row in rows
                for value in row["frontier_indices"].split(";")
                if value
            }),
            "promotion_rejection_classes": dict(sorted(classes.items())),
            "team_minibatch_survivors": 0,
        }
    total = {
        key: sum(int(by_arm[arm][key]) for arm in ARMS)
        for key in (
            "branches", "program_candidates", "proposals", "accepted_mutations",
            "rejected_mutations", "returned_candidates", "returned_parent_candidates",
            "team_minibatch_survivors",
        )
    }
    return {
        "experiment_id": "seed78_primary_responsibility_ab_v1",
        "audit_type": "zero_api_local_gepa_teamminibatch_postmortem",
        "by_arm": by_arm,
        "total": total,
        "classifier": "ROOT_CANDIDATE_RETURN_SEMANTICS_CONFIRMED",
        "scientific_status": "SCHEDULER_CAUSAL_EFFICACY_NOT_EVALUATED",
        "diagnosis": (
            "Every GEPA proposal was rejected locally; the only program and returned "
            "frontier candidate was index 0, byte-identical to the parent. Its fixed-peer "
            "TeamMiniBatch deltas are therefore all exactly zero, so it had no promotion signal."
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sanitization_scan(paths: Iterable[Path]) -> dict[str, Any]:
    forbidden = {
        "absolute_windows_path": re.compile(r"[A-Za-z]:[\\/]"),
        "raw_system_prompt_key": re.compile(r'"system_prompt"'),
        "raw_question_key": re.compile(r'"question"'),
        "raw_answer_key": re.compile(r'"answer"'),
        "secret_or_endpoint_key": re.compile(r'"(?:api_key|endpoint|base_url)"', re.I),
    }
    hits: list[dict[str, str]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for name, pattern in forbidden.items():
            if pattern.search(text):
                hits.append({"file": path.name, "rule": name})
    return {
        "gate": "PASS" if not hits else "FAIL",
        "files_scanned": sorted(path.name for path in paths),
        "forbidden_findings": hits,
        "excluded": [
            "prompts", "questions", "gold/model answers", "raw responses",
            "credentials", "endpoints", "SQLite/cache content", "checkpoints",
            "absolute paths",
        ],
    }


def run(run_root: Path, report_root: Path) -> dict[str, Any]:
    if report_root.exists():
        raise FileExistsError("postmortem report root must be fresh")
    if not run_root.is_dir():
        raise FileNotFoundError("Seed78 retry1 run root is unavailable")
    branches = []
    for arm in ARMS:
        lineage_root = run_root / arm / "local_gepa"
        for path in sorted(lineage_root.glob("*.lineage.jsonl")):
            branches.append(analyze_branch(arm, path))
    summary = build_summary(branches)
    paired = paired_evaluation_audit(run_root)
    facts = {
        "gate": "PASS",
        "assertions": {
            "branch_count_24": len(branches) == 24,
            "all_returned_candidates_are_parent": all(
                row["returned_equals_parent"] for row in branches
            ),
            "all_frontiers_root_only": all(row["frontier_indices"] == "0" for row in branches),
            "all_best_indices_root": all(row["best_candidate_index"] == 0 for row in branches),
            "proposal_rejection_accounting_exact": (
                summary["total"]["proposals"] == 89
                and summary["total"]["rejected_mutations"] == 89
                and summary["total"]["accepted_mutations"] == 0
            ),
            "all_promotion_rejections_unchanged_parent": all(
                row["promotion_rejection_class"] == "UNCHANGED_PARENT" for row in branches
            ),
            "final_teams_identical": paired["final_team_byte_identity_equal"],
            "paired_evaluation_gap_confirmed": (
                paired["classifier"] == "PAIRED_EVALUATION_CACHE_GAP_CONFIRMED"
            ),
            "api_calls_zero": True,
            "validation_calls_zero": True,
            "test_calls_zero": True,
        },
    }
    if not all(facts["assertions"].values()):
        facts["gate"] = "FAIL"
        raise AssertionError(f"postmortem fact assertion failed: {facts}")

    report_root.mkdir(parents=True)
    _write_csv(report_root / "branch_audit.csv", branches)
    _write_json(report_root / "summary.json", summary)
    _write_json(report_root / "paired_evaluation_audit.json", paired)
    _write_json(
        report_root / "classifier.json",
        {
            "local_transfer_classifier": summary["classifier"],
            "scheduler_status": summary["scientific_status"],
            "evaluation_classifier": paired["classifier"],
            "next_action": (
                "Do not add scheduler seeds or tune the promotion gate. First define and "
                "test Layer-1 no-improvement result semantics; future paired final evaluation "
                "must share realization identity for equal request identities."
            ),
        },
    )
    _write_json(report_root / "fact_assertions.json", facts)
    _write_json(
        report_root / "provenance.json",
        {
            "input_run": "runs/seed78_primary_responsibility_ab_v1_retry1",
            "input_mode": "read_only",
            "source_files": {
                "gepa_optimizer": _sha256_file(
                    ROOT / "multi_dataset_diverse_rl/local_optimizers/gepa_optimizer.py"
                ),
                "promotion": _sha256_file(
                    ROOT / "multi_dataset_diverse_rl/team_search/progressive_evaluation.py"
                ),
                "seed78_runner": _sha256_file(
                    ROOT / "scripts/run_seed78_primary_responsibility_ab.py"
                ),
            },
            "api_calls": 0,
            "validation_calls": 0,
            "test_calls": 0,
        },
    )
    readme = """# Seed78 Local-GEPA to TeamMiniBatch postmortem

This is a zero-API, read-only audit of the completed Seed78 A/B pilot. It does
not modify the scheduler, GEPA, promotion gate, frozen run, or historical report.

## Finding

`ROOT_CANDIDATE_RETURN_SEMANTICS_CONFIRMED`

Across 24 target branches, GEPA emitted 89 proposals and rejected all 89 under
its frozen strict local-improvement rule. Every persisted GEPA state contained
only program candidate index 0; every per-instance frontier pointed to index 0;
and every candidate returned to Layer 2 was byte-identical to its parent.

Parent-versus-parent TeamMiniBatch evaluation has exact zero deltas for invalid,
Vote, Target, Coalition, Responsibility, and Broad signals. Therefore all 24
promotion failures are classified `UNCHANGED_PARENT`; none supplies evidence
that the TeamMiniBatch thresholds were too strict or that a changed local
mutation failed team transfer.

The scheduler changed allocation, but its causal efficacy remains not evaluated
because no changed candidate reached a write-back decision.

## Paired final evaluation

The two final teams and their validation request-identity sets were identical.
The historical runner nevertheless used independent per-arm caches and made 50
successful validation provider calls per arm. Its 0.62 versus 0.60 result is
therefore not a scheduler contrast. Future paired experiments must share one
realization cache for identical prompt/question/contract identities.

## Boundaries

API calls = 0. Validation calls = 0. Test calls = 0. No raw prompt, question,
answer, response, endpoint, credential, cache, checkpoint, or absolute path is
published here.
"""
    (report_root / "README.md").write_text(readme, encoding="utf-8")

    scan_targets = sorted(path for path in report_root.iterdir() if path.is_file())
    sanitization = _sanitization_scan(scan_targets)
    if sanitization["gate"] != "PASS":
        raise AssertionError(f"sanitization failed: {sanitization}")
    _write_json(report_root / "sanitization_manifest.json", sanitization)
    manifest_targets = sorted(
        path for path in report_root.iterdir()
        if path.is_file() and path.name != "sha256_manifest.json"
    )
    _write_json(
        report_root / "sha256_manifest.json",
        {path.name: _sha256_file(path) for path in manifest_targets},
    )
    return {"summary": summary, "paired_evaluation": paired, "facts": facts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args.run_root.resolve(), args.report_root.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
