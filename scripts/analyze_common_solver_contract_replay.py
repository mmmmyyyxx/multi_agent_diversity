"""Audit and sanitize the completed COMMON_SOLVER_CONTRACT_V1 replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.common_solver_contract_v1.contract import (
    COMMON_SOLVER_CONTRACT_ID,
    contract_identity,
    request_identity,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, encoding="utf-8"
    ).strip()


def plurality(labels: Sequence[str | None]) -> str | None:
    valid = [label for label in labels if label]
    if not valid:
        return None
    counts = Counter(valid)
    maximum = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == maximum)
    return winners[0] if len(winners) == 1 else None


def coalition_decomposition(
    predictions: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    diversity_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize P0-to-Diversity redistribution without exposing row payloads."""
    gold_by_position = {int(row["position"]): str(row["gold"]) for row in cases}
    indexed: dict[str, dict[int, Mapping[str, Any]]] = {
        "P0_COMMON": {},
        diversity_id: {},
    }
    for row in predictions:
        state_id = str(row["state_id"])
        if state_id in indexed:
            indexed[state_id][int(row["case_position"])] = row
    if set(indexed["P0_COMMON"]) != set(range(50)) or set(indexed[diversity_id]) != set(range(50)):
        raise AssertionError("coalition decomposition requires 50 aligned rows")

    matrix = [[0 for _ in range(6)] for _ in range(6)]
    initial_hist = [0] * 6
    final_hist = [0] * 6
    new_coverage_cases = 0
    lost_coverage_cases = 0
    new_correct_member_votes = 0
    lost_correct_member_votes = 0
    p0_wrong = 0
    p0_wrong_to_final_majority = 0
    p0_correct = 0
    p0_correct_preserved = 0
    oracle_gains = 0
    oracle_gains_converted_to_vote = 0
    oracle_losses = 0

    for position in range(50):
        initial = indexed["P0_COMMON"][position]
        final = indexed[diversity_id][position]
        gold = gold_by_position[position]
        initial_g = sum(
            bool(valid) and label == gold
            for label, valid in zip(initial["member_labels"], initial["member_valid"], strict=True)
        )
        final_g = sum(
            bool(valid) and label == gold
            for label, valid in zip(final["member_labels"], final["member_valid"], strict=True)
        )
        if not (0 <= initial_g <= 5 and 0 <= final_g <= 5):
            raise AssertionError("G must remain within the five-member team")
        matrix[initial_g][final_g] += 1
        initial_hist[initial_g] += 1
        final_hist[final_g] += 1
        new_coverage_cases += int(initial_g == 0 and final_g > 0)
        lost_coverage_cases += int(initial_g > 0 and final_g == 0)
        new_correct_member_votes += max(0, final_g - initial_g)
        lost_correct_member_votes += max(0, initial_g - final_g)

        if bool(initial["vote_correct"]):
            p0_correct += 1
            p0_correct_preserved += int(bool(final["vote_correct"]))
        else:
            p0_wrong += 1
            p0_wrong_to_final_majority += int(bool(final["vote_correct"]))
        if not bool(initial["oracle_correct"]) and bool(final["oracle_correct"]):
            oracle_gains += 1
            oracle_gains_converted_to_vote += int(bool(final["vote_correct"]))
        if bool(initial["oracle_correct"]) and not bool(final["oracle_correct"]):
            oracle_losses += 1

    matrix_rows = [
        {"p0_g": initial_g, "final_g": final_g, "case_count": matrix[initial_g][final_g]}
        for initial_g in range(6)
        for final_g in range(6)
    ]
    summary = {
        "state_id": diversity_id,
        "aligned_case_count": 50,
        "p0_g_distribution": {str(index): count for index, count in enumerate(initial_hist)},
        "final_g_distribution": {str(index): count for index, count in enumerate(final_hist)},
        "p0_gte1": sum(initial_hist[1:]),
        "final_gte1": sum(final_hist[1:]),
        "p0_gte3": sum(initial_hist[3:]),
        "final_gte3": sum(final_hist[3:]),
        "new_coverage_cases": new_coverage_cases,
        "lost_coverage_cases": lost_coverage_cases,
        "new_correct_member_votes": new_correct_member_votes,
        "lost_correct_member_votes": lost_correct_member_votes,
        "p0_wrong_case_count": p0_wrong,
        "p0_wrong_to_final_majority_count": p0_wrong_to_final_majority,
        "p0_wrong_to_final_majority_rate": p0_wrong_to_final_majority / p0_wrong if p0_wrong else None,
        "p0_correct_case_count": p0_correct,
        "p0_correct_to_final_majority_preserved_count": p0_correct_preserved,
        "p0_correct_to_final_majority_preservation_rate": p0_correct_preserved / p0_correct if p0_correct else None,
        "oracle_gain_case_count": oracle_gains,
        "oracle_gain_to_vote_gain_count": oracle_gains_converted_to_vote,
        "oracle_gain_to_vote_gain_rate": oracle_gains_converted_to_vote / oracle_gains if oracle_gains else None,
        "oracle_loss_case_count": oracle_losses,
    }
    return summary, matrix_rows


def audit(
    root: Path,
    run_root: Path,
    prep: Path | None = None,
    manifest_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prep = prep or root / "runs/common_solver_contract_v1_prep_20260906/private_replay_registry.json"
    manifest_path = manifest_path or root / "reports/common_solver_contract_v1_prep_20260906/contract_manifest.json"
    manifest = read_json(manifest_path)
    registry = read_json(prep)
    execution = read_json(run_root / "execution_manifest.json")
    summary = read_json(run_root / "summary_private.json")
    predictions = read_json(run_root / "predictions_private.json")
    cache = read_json(run_root / "exact_request_cache_private.json")
    errors: list[str] = []
    identity = contract_identity()
    if any(
        payload.get("contract_id") != COMMON_SOLVER_CONTRACT_ID
        or payload.get("contract_identity") != identity
        for payload in (registry, execution, summary)
    ):
        errors.append("contract_identity_mismatch")
    if execution.get("status") != "COMPLETE" or summary.get("status") != "PASS":
        errors.append("execution_not_complete")
    if execution.get("test50_accessed") or summary.get("test50_accessed"):
        errors.append("test50_accessed")
    if summary.get("selection_or_optimization") or registry.get("selection_or_optimization"):
        errors.append("selection_or_optimization_detected")
    if len(registry.get("cases", [])) != 50 or len(registry.get("states", [])) != 4:
        errors.append("registry_inventory_mismatch")
    if len(predictions) != 200:
        errors.append("prediction_inventory_mismatch")
    accounting = summary.get("accounting", {})
    request_identities = [
        str(request_id)
        for row in predictions
        for request_id in row.get("request_identities", [])
    ]
    unique_request_count = len(set(request_identities))
    logical_call_count = len(request_identities)
    expected_accounting = {
        "logical_calls": logical_call_count,
        "provider_attempts": unique_request_count,
        "successful_provider_calls": unique_request_count,
        "failed_provider_attempts": 0,
        "cache_hits": logical_call_count - unique_request_count,
        "cache_entries": unique_request_count,
    }
    for key, expected in expected_accounting.items():
        if int(accounting.get(key, -1)) != expected:
            errors.append(f"accounting_{key}_mismatch")
    if len(cache) != unique_request_count:
        errors.append("cache_inventory_mismatch")

    states = {row["state_id"]: row for row in registry["states"]}
    cases = {int(row["position"]): row for row in registry["cases"]}
    by_state: dict[str, list[dict[str, Any]]] = {state_id: [] for state_id in states}
    for row in predictions:
        state_id = str(row["state_id"])
        position = int(row["case_position"])
        if state_id not in states or position not in cases:
            errors.append("prediction_key_outside_registry")
            continue
        state = states[state_id]
        case = cases[position]
        prompts = state["ordered_prompts"]
        expected_ids = [
            request_identity(decision_procedure=prompt, question=case["question"])
            for prompt in prompts
        ]
        if row["request_identities"] != expected_ids:
            errors.append("request_identity_mismatch")
        labels = row["member_labels"]
        vote = plurality(labels)
        if vote != row["vote_label"]:
            errors.append("plurality_mismatch")
        if bool(vote == case["gold"]) != bool(row["vote_correct"]):
            errors.append("vote_scoring_mismatch")
        if bool(any(label == case["gold"] for label in labels)) != bool(row["oracle_correct"]):
            errors.append("oracle_scoring_mismatch")
        by_state[state_id].append(row)

    recomputed: dict[str, dict[str, Any]] = {}
    for state_id, rows in by_state.items():
        rows.sort(key=lambda row: int(row["case_position"]))
        member_count = len(states[state_id]["ordered_prompts"])
        member_correct = [0] * member_count
        invalid_member_outputs = 0
        for row in rows:
            gold = cases[int(row["case_position"])]["gold"]
            for index, (label, valid) in enumerate(zip(row["member_labels"], row["member_valid"], strict=True)):
                member_correct[index] += int(bool(valid) and label == gold)
                invalid_member_outputs += int(not bool(valid))
        recomputed[state_id] = {
            "vote_correct": sum(bool(row["vote_correct"]) for row in rows),
            "oracle_correct": sum(bool(row["oracle_correct"]) for row in rows),
            "valid_vote_count": sum(row["vote_label"] is not None for row in rows),
            "member_correct": member_correct,
            "invalid_member_outputs": invalid_member_outputs,
        }
    for row in summary["states"]:
        actual = recomputed[str(row["state_id"])]
        for key in ("vote_correct", "oracle_correct", "valid_vote_count"):
            if int(row[key]) != int(actual[key]):
                errors.append(f"summary_{key}_mismatch")
        if list(row["individual_member_correct"]) != actual["member_correct"]:
            errors.append("summary_member_correct_mismatch")

    source_map = {
        "__init__.py": root / "infrastructure/common_solver_contract_v1/__init__.py",
        "contract.py": root / "infrastructure/common_solver_contract_v1/contract.py",
        "entrypoints.py": root / "infrastructure/common_solver_contract_v1/entrypoints.py",
        "evaluator.py": root / "infrastructure/common_solver_contract_v1/evaluator.py",
        "run_common_solver_contract_replay.py": root / "scripts/run_common_solver_contract_replay.py",
    }
    for name, path in source_map.items():
        if sha256_file(path) != manifest["source_sha256"][name]:
            errors.append(f"source_freeze_mismatch_{name}")
    return {
        "gate": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "contract_id": COMMON_SOLVER_CONTRACT_ID,
        "contract_identity": identity,
        "state_count": 4,
        "case_count_per_state": 50,
        "prediction_row_count": len(predictions),
        "test50_accessed": False,
        "optimization_rerun": False,
        "source_freeze": "PASS" if not any("source_freeze" in error for error in errors) else "FAIL",
        "accounting": accounting,
        "recomputed": recomputed,
    }, predictions


def sanitize(report: Path) -> dict[str, Any]:
    forbidden = re.compile(
        r"(?:[A-Za-z]:\\|DASHSCOPE|api[_-]?key|authorization\s*:|bearer\s+|"
        r"raw_response|question_text|gold_answer|model_answer|\.sqlite|checkpoint)",
        re.IGNORECASE,
    )
    findings: list[dict[str, Any]] = []
    for path in sorted(report.iterdir()):
        if not path.is_file() or path.name == "sanitization_manifest.json":
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if forbidden.search(line):
                findings.append({"file": path.name, "line": line_no})
    return {
        "status": "PASS" if not findings else "FAIL",
        "finding_count": len(findings),
        "findings": findings,
    }


def run(
    root: Path,
    run_root: Path,
    report: Path,
    prep: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    root, run_root, report = root.resolve(), run_root.resolve(), report.resolve()
    prep = prep or root / "runs/common_solver_contract_v1_prep_20260906/private_replay_registry.json"
    manifest_path = manifest_path or root / "reports/common_solver_contract_v1_prep_20260906/contract_manifest.json"
    if root not in run_root.parents or root not in report.parents:
        raise ValueError("run and report roots must remain project-local")
    if report.exists() and any(report.iterdir()):
        raise FileExistsError(f"report must be fresh: {report}")
    report.mkdir(parents=True, exist_ok=True)
    audit_result, predictions = audit(root, run_root, prep=prep, manifest_path=manifest_path)
    if audit_result["gate"] != "PASS":
        raise AssertionError(json.dumps(audit_result, sort_keys=True))
    summary = read_json(run_root / "summary_private.json")
    state_rows: list[dict[str, Any]] = []
    summary_by_id = {row["state_id"]: row for row in summary["states"]}
    p0 = summary_by_id["P0_COMMON"]
    diversity_ids = [state_id for state_id in summary_by_id if state_id.startswith("DIVERSITY_SEED")]
    if len(diversity_ids) != 1:
        raise AssertionError("exactly one Diversity final state is required")
    diversity_id = diversity_ids[0]
    source_seed = int(summary_by_id[diversity_id]["source_seed"])
    mars_id = f"MARS_SEED{source_seed}_FINAL"
    gepa_id = f"GEPA_SEED{source_seed}_FINAL"
    final_ids = (mars_id, gepa_id, diversity_id)
    vote_correct_deltas = {
        state_id: int(summary_by_id[state_id]["vote_correct"]) - int(p0["vote_correct"])
        for state_id in final_ids
    }
    improved_final_count = sum(delta > 0 for delta in vote_correct_deltas.values())
    equal_final_count = sum(delta == 0 for delta in vote_correct_deltas.values())
    worse_final_count = sum(delta < 0 for delta in vote_correct_deltas.values())
    registry = read_json(prep)
    coalition, transition_rows = coalition_decomposition(
        predictions,
        registry["cases"],
        diversity_id,
    )
    write_json(report / "diversity_coalition_decomposition.json", coalition)
    write_csv(
        report / "diversity_g_transition_matrix.csv",
        transition_rows,
        ("p0_g", "final_g", "case_count"),
    )
    for state_id in ("P0_COMMON", mars_id, gepa_id, diversity_id):
        row = summary_by_id[state_id]
        mean_member = sum(row["individual_member_accuracy"]) / len(row["individual_member_accuracy"])
        state_rows.append(
            {
                "state_id": state_id,
                "source_repo": row["source_repo"],
                "source_seed": "" if row["source_seed"] is None else row["source_seed"],
                "member_count": row["member_count"],
                "vote_correct": row["vote_correct"],
                "vote_accuracy": f"{row['vote_accuracy']:.6f}",
                "oracle_correct": row["oracle_correct"],
                "oracle_accuracy": f"{row['oracle_accuracy']:.6f}",
                "mean_member_accuracy": f"{mean_member:.6f}",
                "valid_vote_count": row["valid_vote_count"],
                "invalid_member_outputs": audit_result["recomputed"][state_id]["invalid_member_outputs"],
            }
        )
    write_csv(
        report / "per_state_results.csv",
        state_rows,
        (
            "state_id",
            "source_repo",
            "source_seed",
            "member_count",
            "vote_correct",
            "vote_accuracy",
            "oracle_correct",
            "oracle_accuracy",
            "mean_member_accuracy",
            "valid_vote_count",
            "invalid_member_outputs",
        ),
    )
    contrast_rows: list[dict[str, Any]] = []
    for state_id in final_ids:
        row = summary_by_id[state_id]
        contrast_rows.append(
            {
                "contrast": f"{state_id}_minus_P0_COMMON",
                "vote_correct_delta": int(row["vote_correct"]) - int(p0["vote_correct"]),
                "vote_accuracy_delta": f"{row['vote_accuracy'] - p0['vote_accuracy']:+.6f}",
                "oracle_correct_delta": int(row["oracle_correct"]) - int(p0["oracle_correct"]),
                "oracle_accuracy_delta": f"{row['oracle_accuracy'] - p0['oracle_accuracy']:+.6f}",
            }
        )
    write_csv(
        report / "contrasts.csv",
        contrast_rows,
        ("contrast", "vote_correct_delta", "vote_accuracy_delta", "oracle_correct_delta", "oracle_accuracy_delta"),
    )
    by_state = {
        state_id: {int(row["case_position"]): bool(row["vote_correct"]) for row in predictions if row["state_id"] == state_id}
        for state_id in summary_by_id
    }
    disagreement_rows: list[dict[str, Any]] = []
    comparisons = ((mars_id, gepa_id), (mars_id, diversity_id), (gepa_id, diversity_id))
    for left, right in comparisons:
        disagreement_rows.append(
            {
                "left": left,
                "right": right,
                "aligned_cases": 50,
                "correctness_disagreements": sum(
                    by_state[left][index] != by_state[right][index] for index in range(50)
                ),
                "left_only_correct": sum(
                    by_state[left][index] and not by_state[right][index] for index in range(50)
                ),
                "right_only_correct": sum(
                    by_state[right][index] and not by_state[left][index] for index in range(50)
                ),
            }
        )
    write_csv(
        report / "prediction_disagreement.csv",
        disagreement_rows,
        ("left", "right", "aligned_cases", "correctness_disagreements", "left_only_correct", "right_only_correct"),
    )
    classifier = {
        "common_contract_replay": "PASS",
        "evaluation_contract_parity": "CONFIRMED_FOR_THIS_REPLAY",
        "end_to_end_optimization_parity": "NOT_ESTABLISHED",
        "formal_cross_method_raw_ranking": "NOT_YET_ELIGIBLE",
        "full_parity_rerun_decision": "JUSTIFIED_IF_FORMAL_CROSS_METHOD_RANKING_IS_REQUIRED",
        "reason": (
            "The common replay removes evaluation-adapter drift and compares all three frozen finals with the "
            f"shared P0, but only Seed{source_seed} was replayed and the frozen finals were produced under different "
            "optimization-time Solver contracts. Small cross-method gaps on 50 cases are not "
            "sufficient for a formal ranking without an end-to-end parity rerun."
        ),
        "internal_gain_and_saturation_claims": "REMAIN_USABLE",
    }
    write_json(report / "classifier.json", classifier)
    public_audit = {key: value for key, value in audit_result.items() if key != "recomputed"}
    write_json(report / "audit.json", public_audit)
    facts = {
        "gate": "PASS",
        "api_execution_complete": True,
        "test50_accessed": False,
        "optimization_rerun": False,
        "state_count": 4,
        "external_validation_cases_per_state": 50,
        "logical_calls": summary["accounting"]["logical_calls"],
        "provider_calls": summary["accounting"]["successful_provider_calls"],
        "failed_provider_attempts": summary["accounting"]["failed_provider_attempts"],
        "cache_hits": summary["accounting"]["cache_hits"],
        "source_freeze": audit_result["source_freeze"],
        "coalition_transition_case_count": sum(row["case_count"] for row in transition_rows),
        "coalition_decomposition_state_id": diversity_id,
    }
    write_json(report / "fact_assertions.json", facts)
    write_json(
        report / "provenance.json",
        {
            "report_version": f"common_solver_contract_v1_seed{source_seed}_replay_20260907",
            "contract_identity": contract_identity(),
            "execution_commit": git(root, "rev-parse", "HEAD"),
            "source_repositories": ["Diversity", "GEPA", "MARS"],
            "source_seed": source_seed,
            "split": "ExternalValidation50",
            "raw_evidence_tracked": False,
        },
    )
    readme = f"""# COMMON_SOLVER_CONTRACT_V1 Seed{source_seed} frozen-artifact replay

The single shared evaluator completed P0 plus the frozen Seed{source_seed} MARS, GEPA, and
Diversity final artifacts on ExternalValidation50. No optimization was rerun
and Test50 was not accessed.

| State | Vote | Oracle | Mean member |
|---|---:|---:|---:|
| P0 common | {p0['vote_accuracy']:.2f} | {p0['oracle_accuracy']:.2f} | {sum(p0['individual_member_accuracy']) / len(p0['individual_member_accuracy']):.3f} |
| MARS final | {summary_by_id[mars_id]['vote_accuracy']:.2f} | {summary_by_id[mars_id]['oracle_accuracy']:.2f} | {summary_by_id[mars_id]['individual_member_accuracy'][0]:.3f} |
| GEPA final | {summary_by_id[gepa_id]['vote_accuracy']:.2f} | {summary_by_id[gepa_id]['oracle_accuracy']:.2f} | {summary_by_id[gepa_id]['individual_member_accuracy'][0]:.3f} |
| Diversity P1 final | {summary_by_id[diversity_id]['vote_accuracy']:.2f} | {summary_by_id[diversity_id]['oracle_accuracy']:.2f} | {sum(summary_by_id[diversity_id]['individual_member_accuracy']) / 5:.3f} |

The Diversity P0-to-final coalition-depth decomposition is recorded separately:
new/lost coverage = {coalition['new_coverage_cases']}/{coalition['lost_coverage_cases']},
new/lost correct member-votes = {coalition['new_correct_member_votes']}/{coalition['lost_correct_member_votes']},
and G>=3 cases = {coalition['p0_gte3']} -> {coalition['final_gte3']}.

Relative to shared P0, {improved_final_count} frozen finals improve Vote,
{equal_final_count} tie it, and {worse_final_count} are lower in this replay. The
evaluation contract is aligned, but end-to-end optimization parity is not: the
artifacts were generated under different historical Solver adapters. A full
parity rerun is justified only if a formal cross-method ranking is required;
the existing within-method gain and saturation evidence remains usable.
"""
    (report / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    sanitization = sanitize(report)
    write_json(report / "sanitization_manifest.json", sanitization)
    if sanitization["status"] != "PASS":
        raise AssertionError("sanitization failed")
    write_json(
        report / "sha256_manifest.json",
        {
            path.name: sha256_file(path)
            for path in sorted(report.iterdir())
            if path.is_file() and path.name != "sha256_manifest.json"
        },
    )
    return classifier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT / "runs/common_solver_contract_v1_replay_20260906",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/common_solver_contract_v1_replay_20260906",
    )
    parser.add_argument(
        "--prep",
        type=Path,
        default=None,
        help="Private replay registry used to freeze states and cases.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Tracked source-freeze manifest for this replay.",
    )
    args = parser.parse_args()
    result = run(
        args.root,
        args.run_root,
        args.report,
        prep=args.prep,
        manifest_path=args.manifest,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
