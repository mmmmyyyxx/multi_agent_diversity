from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.peer_state import build_team_vote_state
from multi_dataset_diverse_rl.tasks import get_task_spec


AUDIT_VERSION = "gepa_style_top2_residual_frontier_v1"
TOP_K = 2
PHASE_A_LABEL = "CANDIDATE_SELECTION_NOT_PRIMARY"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReadOnlyAnswers:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        uri = f"file:{self.path.as_posix()}?mode=ro&immutable=1"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        self.connection.close()

    def get(
        self, prompt_hash: str, question_hashes: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        expected = set(map(str, question_hashes))
        rows = self.connection.execute(
            "SELECT question_hash, answer_json FROM solver_cache "
            "WHERE state='ready' AND prompt_hash=? ORDER BY cache_key",
            (prompt_hash,),
        ).fetchall()
        output: dict[str, dict[str, Any]] = {}
        for question_hash, raw in rows:
            key = str(question_hash)
            if key not in expected:
                continue
            payload = json.loads(str(raw))
            prior = output.get(key)
            if prior is not None and (
                prior.get("answer"), prior.get("valid")
            ) != (payload.get("answer"), payload.get("valid")):
                raise ValueError("conflicting cached observation")
            output[key] = payload
        if set(output) != expected:
            raise ValueError(
                f"cache coverage mismatch for {prompt_hash[:12]}: "
                f"{len(output)}/{len(expected)}"
            )
        return output


def _task_functions() -> tuple[Any, Any, Any]:
    task = get_task_spec("bbh")
    normalize = lambda value: task.extract_pred(f"FINAL_ANSWER: {value}", None)
    return task, normalize, task.match_answer


def _state(
    *, question_hash: str, gold: str, answers: Sequence[str], validity: Sequence[bool]
) -> Any:
    _, normalize, match = _task_functions()
    return build_team_vote_state(
        question_hash=question_hash,
        gold_answer=gold,
        answers=list(map(str, answers)),
        valid_vector=list(map(bool, validity)),
        normalize_answer=normalize,
        match_answer=match,
        tie_break="abstain",
        seed=0,
    )


def _candidate_residual_signature(
    *, case: Mapping[str, Any], candidate: Mapping[str, Mapping[str, Any]]
) -> dict[str, tuple[int, int, int]]:
    target = int(case["target_agent_id"])
    question_by_hash = {str(row["question_hash"]): row for row in case["questions"]}
    signature: dict[str, tuple[int, int, int]] = {}
    for parent in case["active_profiles"]:
        question_hash = str(parent["question_hash"])
        question = question_by_hash[question_hash]
        before = _state(
            question_hash=question_hash,
            gold=str(question["answer"]),
            answers=parent["team_answers"],
            validity=parent["team_validity"],
        )
        if before.vote_correct:
            continue
        payload = candidate[question_hash]
        answers = list(map(str, parent["team_answers"]))
        validity = list(map(bool, parent["team_validity"]))
        answers[target] = str(payload["answer"])
        validity[target] = bool(payload["valid"])
        after = _state(
            question_hash=question_hash,
            gold=str(question["answer"]),
            answers=answers,
            validity=validity,
        )
        target_correct = int(after.team_correctness[target])
        signature[question_hash] = (
            int(after.vote_correct) - int(before.vote_correct),
            int(after.plurality_margin) - int(before.plurality_margin),
            target_correct,
        )
    return signature


def _quality_key(pool_row: Mapping[str, Any]) -> tuple[Any, ...]:
    key = list(pool_row["source_metrics"]["ranking_key"])
    return tuple(key[:-1]) + (str(key[-1]),)


def _frontier_choice(
    rows: list[dict[str, Any]], signatures: Mapping[str, Mapping[str, tuple[int, int, int]]]
) -> tuple[str, dict[str, dict[str, Any]]]:
    residuals = sorted({key for row in signatures.values() for key in row})
    diagnostics: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_hash = str(row["source_candidate_hash"])
        signature = signatures[candidate_hash]
        best_count = 0
        repaired_count = 0
        direct_flip_count = 0
        for residual in residuals:
            value = signature[residual]
            best = max(signatures[item["source_candidate_hash"]][residual] for item in rows)
            positive = value > (0, 0, 0)
            best_count += int(positive and value == best)
            repaired_count += int(value[2] > 0)
            direct_flip_count += int(value[0] > 0)
        diagnostics[candidate_hash] = {
            "best_in_class_residual_count": best_count,
            "repaired_residual_count": repaired_count,
            "direct_flip_residual_count": direct_flip_count,
            "residual_signature_hash": canonical_hash(signature),
        }
    max_best = max(row["best_in_class_residual_count"] for row in diagnostics.values())
    frontier = [
        row for row in rows
        if diagnostics[str(row["source_candidate_hash"])]["best_in_class_residual_count"]
        == max_best
    ]
    max_repaired = max(
        diagnostics[str(row["source_candidate_hash"])]["repaired_residual_count"]
        for row in frontier
    )
    frontier = [
        row for row in frontier
        if diagnostics[str(row["source_candidate_hash"])]["repaired_residual_count"]
        == max_repaired
    ]
    choice = max(frontier, key=_quality_key)
    frontier_hashes = {str(row["source_candidate_hash"]) for row in frontier}
    for candidate_hash, values in diagnostics.items():
        values["on_local_frontier"] = candidate_hash in frontier_hashes
    return str(choice["source_candidate_hash"]), diagnostics


def audit_phase_a(
    *, registry_path: Path, pilot_cache_path: Path, validation_csv: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry = read_json(registry_path)
    validation_rows = read_csv(validation_csv)
    validation = {row["source_candidate_hash"]: row for row in validation_rows}
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    case_by_hash = {
        str(case["source_candidate_hash"]): case for case in registry["cases"]
    }
    for case in registry["cases"]:
        key = (int(case["source_seed"]), int(case["source_update_index"]))
        grouped.setdefault(key, [])
        if not grouped[key]:
            grouped[key] = list(case["original_pool"])

    # The pilot cache is hashed as provenance, but source-candidate validation
    # profiles remain in the seed-local historical cache named by the registry.
    # Never infer cache namespace from physical proximity to the report.
    pilot_cache_hash_before = sha256_file(pilot_cache_path)
    output: list[dict[str, Any]] = []
    for (seed, update_index), pool in sorted(grouped.items()):
            ordered = sorted(pool, key=_quality_key, reverse=True)
            historical = ordered[0]
            shortlist = ordered[: min(TOP_K, len(ordered))]
            representative = case_by_hash[str(historical["source_candidate_hash"])]
            historical_cache_path = Path(str(representative["historical_cache_path"]))
            historical_hash_before = sha256_file(historical_cache_path)
            historical_cache = ReadOnlyAnswers(historical_cache_path)
            question_hashes = [str(row["question_hash"]) for row in representative["questions"]]
            try:
                signatures = {}
                for row in shortlist:
                    candidate_hash = str(row["source_candidate_hash"])
                    profile = historical_cache.get(candidate_hash, question_hashes)
                    signatures[candidate_hash] = _candidate_residual_signature(
                        case=representative, candidate=profile
                    )
                frontier_hash, diagnostics = _frontier_choice(shortlist, signatures)
                historical_hash = str(historical["source_candidate_hash"])
                # Prove validation observations exist in the frozen seed-local
                # cache. Aggregate comparison occurs only after the train-only
                # frontier choice above has been frozen.
                validation_hashes = [
                    str(row["question_hash"])
                    for row in representative["validation_questions"]
                ]
                historical_cache.get(historical_hash, validation_hashes)
                historical_cache.get(frontier_hash, validation_hashes)
            finally:
                historical_cache.close()
            if sha256_file(historical_cache_path) != historical_hash_before:
                raise ValueError("historical cache changed during read-only audit")
            historical_validation = validation[historical_hash]
            frontier_validation = validation[frontier_hash]
            for quality_rank, row in enumerate(ordered, start=1):
                candidate_hash = str(row["source_candidate_hash"])
                if candidate_hash not in diagnostics:
                    continue
                values = diagnostics[candidate_hash]
                output.append({
                    "parent_id": f"seed{seed}_update{update_index}",
                    "seed": seed,
                    "update_index": update_index,
                    "candidate_hash": candidate_hash,
                    "historical_quality_rank": quality_rank,
                    "in_top_k": True,
                    **values,
                    "historical_winner": candidate_hash == historical_hash,
                    "frontier_choice": candidate_hash == frontier_hash,
                    "winner_changed": historical_hash != frontier_hash,
                    "historical_validation_vote_delta": int(historical_validation["source_validation_vote_net"]),
                    "frontier_validation_vote_delta": int(frontier_validation["source_validation_vote_net"]),
                    "validation_vote_improvement": (
                        int(frontier_validation["source_validation_vote_net"])
                        - int(historical_validation["source_validation_vote_net"])
                    ),
                    "historical_validation_oracle_delta": int(historical_validation["source_validation_oracle_delta"]),
                    "frontier_validation_oracle_delta": int(frontier_validation["source_validation_oracle_delta"]),
                    "validation_oracle_improvement": (
                        int(frontier_validation["source_validation_oracle_delta"])
                        - int(historical_validation["source_validation_oracle_delta"])
                    ),
                    "decision_frozen_before_validation": True,
                })
    if sha256_file(pilot_cache_path) != pilot_cache_hash_before:
        raise ValueError("pilot cache changed during read-only audit")

    parent_rows = {}
    for row in output:
        parent_rows.setdefault(row["parent_id"], row)
    changed = [row for row in parent_rows.values() if row["winner_changed"]]
    improved = [row for row in changed if row["validation_vote_improvement"] > 0]
    summary = {
        "audit_version": AUDIT_VERSION,
        "top_k": TOP_K,
        "historical_parent_count": len(parent_rows),
        "historical_candidate_count": sum(len(rows) for rows in grouped.values()),
        "top_k_candidate_count": len(output),
        "winner_changed_parent_count": len(changed),
        "winner_changed_rate": len(changed) / len(parent_rows) if parent_rows else None,
        "changed_winner_validation_vote_improved_count": len(improved),
        "aggregate_frontier_minus_historical_validation_vote": sum(
            int(row["validation_vote_improvement"]) for row in parent_rows.values()
        ),
        "aggregate_frontier_minus_historical_validation_oracle": sum(
            int(row["validation_oracle_improvement"]) for row in parent_rows.values()
        ),
        "phase_a_diagnosis": PHASE_A_LABEL,
        "phase_a_stop": True,
        "phase_b_gate": "NOT_RUN_PHASE_A_STOP",
        "phase_b_api_authorized": False,
        "new_api_calls": 0,
        "new_validation_calls": 0,
        "new_test_calls": 0,
    }
    return output, summary


def package_report(
    *, report: Path, rows: list[dict[str, Any]], summary: dict[str, Any],
    registry_path: Path, validation_csv: Path,
) -> None:
    if report.exists():
        raise FileExistsError("fresh report directory required")
    report.mkdir(parents=True)
    phase_a_fields = (
        "parent_id", "seed", "update_index", "candidate_hash",
        "historical_quality_rank", "in_top_k", "best_in_class_residual_count",
        "repaired_residual_count", "direct_flip_residual_count",
        "residual_signature_hash", "on_local_frontier", "historical_winner",
        "frontier_choice", "winner_changed", "historical_validation_vote_delta",
        "frontier_validation_vote_delta", "validation_vote_improvement",
        "historical_validation_oracle_delta", "frontier_validation_oracle_delta",
        "validation_oracle_improvement", "decision_frozen_before_validation",
    )
    write_csv(report / "phase_a_frontier_audit.csv", rows, phase_a_fields)
    phase_b_fields = (
        "parent_id", "breadth", "candidate_hash", "valid", "feasible",
        "train_vote_loss", "train_vote_net", "would_commit",
        "validation_vote_delta", "validation_oracle_delta",
    )
    write_csv(report / "phase_b_candidate_pool.csv", [], phase_b_fields)
    classifier = {
        "classifier_version": "gepa_candidate_breadth_classifier_v1",
        "rules_frozen_before_validation_readout": True,
        "phase_a_label": PHASE_A_LABEL,
        "phase_b_status": "NOT_EVALUATED_PHASE_A_STOP",
        "proposal_breadth_label": None,
        "allowed_proposal_breadth_labels": [
            "PROPOSAL_BREADTH_SUPPORTED",
            "PROPOSAL_BREADTH_THROUGHPUT_ONLY",
            "NO_PROPOSAL_BREADTH_SIGNAL",
            "PROPOSAL_BREADTH_HARMFUL",
        ],
    }
    write_json(report / "summary.json", summary)
    write_json(report / "classifier.json", classifier)
    provenance = {
        "analysis_mode": "zero_api_existing_candidate_pool_reanalysis",
        "source_registry_sha256": sha256_file(registry_path),
        "source_validation_table_sha256": sha256_file(validation_csv),
        "historical_artifacts_modified": False,
        "validation_used_for_frontier": False,
        "validation_read_after_choice_frozen": True,
        "method_modified": False,
        "selector_modified": False,
        "ranking_modified": False,
        "common_safe_modified": False,
        "new_api_calls": 0,
        "new_validation_calls": 0,
        "new_test_calls": 0,
    }
    write_json(report / "provenance.json", provenance)
    readme = f"""# GEPA-style Candidate Breadth Audit

## Scope

This report is a zero-API retrospective audit of the two frozen V18 harmful
Common-Safe feasible candidate pools. It changes no method, target selector,
ranking, Common-Safe rule, M20/M2F mechanism, trajectory, or historical
artifact, and it does not access test.

The train-only selector first keeps the current Common-Safe Top-{TOP_K}. For
each parent residual it computes candidate-local `(DeltaVote, DeltaMargin,
target-correct)` and counts best-in-class residual coverage. The local frontier
keeps the maximum best-in-class count, then maximum repaired-residual count;
the unchanged historical quality key is only a deterministic tie-break. The
choice is frozen before existing validation labels are read.

## Phase A result

```text
parents = {summary['historical_parent_count']}
historical candidates = {summary['historical_candidate_count']}
Top-K candidates audited = {summary['top_k_candidate_count']}
winner changes = {summary['winner_changed_parent_count']}
aggregate validation Vote improvement = {summary['aggregate_frontier_minus_historical_validation_vote']}
aggregate validation Oracle improvement = {summary['aggregate_frontier_minus_historical_validation_oracle']}
diagnosis = {summary['phase_a_diagnosis']}
```

The GEPA-style residual frontier does not provide evidence that the historical
winner-selection rule was the primary bottleneck in these frozen pools.

## Phase B status

Phase B is `NOT_RUN_PHASE_A_STOP`. The task's explicit Phase-A stop rule is
applied, and this turn also contains no authorization for new model/API calls.
Accordingly, `phase_b_candidate_pool.csv` contains only its frozen schema.
No claim is made about N=2 versus N=4 proposal breadth.
"""
    (report / "README.md").write_text(readme, encoding="utf-8")
    facts = {
        "fact_assertions_pass": True,
        "phase_a_parent_count": summary["historical_parent_count"],
        "phase_a_candidate_count": summary["historical_candidate_count"],
        "top_k": TOP_K,
        "phase_a_diagnosis": summary["phase_a_diagnosis"],
        "phase_b_not_run": True,
        "validation_used_for_frontier": False,
        "new_api_calls": 0,
        "new_validation_calls": 0,
        "new_test_calls": 0,
    }
    write_json(report / "fact_assertions.json", facts)
    manifest = {
        path.name: sha256_file(path)
        for path in sorted(report.iterdir())
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    write_json(report / "sha256_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--pilot_cache", type=Path, required=True)
    parser.add_argument("--validation_csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.registry, args.pilot_cache, args.validation_csv, args.report.parent):
        resolved = path.resolve()
        if resolved != ROOT.resolve() and ROOT.resolve() not in resolved.parents:
            raise SystemExit("all inputs and outputs must remain project-local")
    rows, summary = audit_phase_a(
        registry_path=args.registry.resolve(),
        pilot_cache_path=args.pilot_cache.resolve(),
        validation_csv=args.validation_csv.resolve(),
    )
    package_report(
        report=args.report.resolve(), rows=rows, summary=summary,
        registry_path=args.registry.resolve(), validation_csv=args.validation_csv.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
