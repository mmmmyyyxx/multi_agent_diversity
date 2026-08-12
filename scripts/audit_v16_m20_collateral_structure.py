from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.peer_state import build_peer_vote_context

sys.path.insert(0, str(ROOT / "scripts"))
from generic_m20_probe_support import evaluation_system


AUDIT_VERSION = "v16_m20_collateral_structure_audit_v1"
M20 = "m20_current_v15"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def competence_role(system: Any, target: int, state: Any, stable: set[str]) -> str:
    if not state.team_correctness[target]:
        raise ValueError("competence role requires a parent-correct target")
    peer = build_peer_vote_context(state, target)
    if state.gold_vote_count == 1:
        return "unique"
    if state.vote_correct and peer.peer_margin <= 0:
        return "pivotal"
    if state.question_hash in stable:
        return "stable"
    return "fragile"


def collateral_class(row: dict[str, Any]) -> str:
    if row["loss_count"] == 0:
        return "NONE"
    if row["nonresponsibility_unique_loss_count"] + row["nonresponsibility_pivotal_loss_count"]:
        return "SPECIALIZATION_OVERWRITE"
    if row["nonresponsibility_loss_count"] >= 3 and row["nonresponsibility_loss_count"] >= 2 * max(1, row["responsibility_gain_count"]):
        return "BROAD_DEGRADATION"
    if row["nonresponsibility_loss_count"] == 1:
        return "LOCAL_ACCIDENTAL"
    if row["responsibility_gain_count"]:
        return "SKILL_TRADEOFF"
    return "UNEXPLAINED_REGRESSION"


def cached_answers(cache_path: Path, prompt_hash: str, expected_questions: set[str], system: Any) -> dict[str, dict[str, Any]]:
    uri = f"file:{cache_path.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            "SELECT question_hash, answer_json FROM solver_cache "
            "WHERE state='ready' AND prompt_hash=? "
            "AND model_request_identity=? AND parser_version=? "
            "AND temperature=? AND evaluation_replica_seed=? "
            "AND solver_model=? AND max_tokens=? AND output_contract_version=? "
            "ORDER BY cache_key",
            (
                prompt_hash,
                system.prompt_question_evaluator.model_request_identity,
                system.prompt_question_evaluator.parser_version,
                system.prompt_question_evaluator.temperature,
                system.prompt_question_evaluator.decoding_seed,
                system.cfg.models.agent_model,
                system.cfg.models.solver_max_tokens,
                system.cfg.peer_state.solver_output_contract_version,
            ),
        ).fetchall()
    finally:
        connection.close()
    result: dict[str, dict[str, Any]] = {}
    for question_hash, raw in rows:
        key = str(question_hash)
        if key not in expected_questions:
            continue
        if key in result:
            raise ValueError("duplicate candidate observation")
        payload = json.loads(str(raw))
        if not isinstance(payload, dict):
            raise ValueError("invalid cached candidate observation")
        result[key] = payload
    if set(result) != expected_questions:
        raise ValueError("candidate cache does not cover the exact fixed probe")
    return result


def audit(registry: dict[str, Any], summary: dict[str, Any], cache_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cases = {str(row["case_id"]): row for row in registry["cases"]}
    cells = [row for row in summary["cells"] if row["variant"] == M20]
    if len(cells) != 8 or set(cases) != {str(row["case_id"]) for row in cells}:
        raise ValueError("M20 cell identity mismatch")
    candidate_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for cell in cells:
        case = cases[str(cell["case_id"])]
        target = int(case["target_agent_id"])
        system = evaluation_system(case, out_dir=ROOT / "runs" / "collateral_audit_read_only", cache_path=cache_path)
        states, _, _ = system.current_states_and_opportunities()
        state_by_hash = {row.question_hash: row for row in states}
        expected = set(state_by_hash)
        responsibility = set(map(str, case["assigned_question_hashes"]))
        stable = set(map(str, case["stable_correct_question_hashes_by_agent"][str(target)]))
        for candidate in cell["candidates"]:
            prompt_hash = str(candidate["prompt_hash"])
            answers = cached_answers(cache_path, prompt_hash, expected, system)
            counts: Counter[str] = Counter()
            candidate_examples: list[dict[str, Any]] = []
            for question_hash in sorted(expected):
                state = state_by_hash[question_hash]
                observation = answers[question_hash]
                before = bool(state.team_correctness[target])
                after = bool(observation.get("valid")) and system.match_answer(str(observation.get("answer", "")), state.gold_answer)
                transition = "gain" if not before and after else "loss" if before and not after else "unchanged"
                if transition == "unchanged":
                    continue
                scope = "responsibility" if question_hash in responsibility else "nonresponsibility"
                role = competence_role(system, target, state, stable) if transition == "loss" else "not_parent_correct"
                counts[f"{transition}_count"] += 1
                counts[f"{scope}_{transition}_count"] += 1
                if transition == "loss":
                    counts[f"{scope}_{role}_loss_count"] += 1
                candidate_examples.append({
                    "case_id": cell["case_id"], "seed": int(cell["seed"]),
                    "candidate_hash": prompt_hash, "question_hash": question_hash,
                    "transition": transition, "scope": scope, "parent_competence_role": role,
                    "parent_gold_vote_count": int(state.gold_vote_count),
                    "parent_largest_wrong_vote_count": int(state.largest_wrong_vote_count),
                    "parent_plurality_margin": int(state.plurality_margin),
                    "parent_vote_correct": bool(state.vote_correct),
                })
            row = {
                "case_id": cell["case_id"], "seed": int(cell["seed"]),
                "candidate_hash": prompt_hash, "target_gain": int(candidate["target_gain"]),
                "responsibility_gain_count": counts["responsibility_gain_count"],
                "responsibility_loss_count": counts["responsibility_loss_count"],
                "nonresponsibility_gain_count": counts["nonresponsibility_gain_count"],
                "nonresponsibility_loss_count": counts["nonresponsibility_loss_count"],
                "gain_count": counts["gain_count"], "loss_count": counts["loss_count"],
                "nonresponsibility_unique_loss_count": counts["nonresponsibility_unique_loss_count"],
                "nonresponsibility_pivotal_loss_count": counts["nonresponsibility_pivotal_loss_count"],
                "nonresponsibility_stable_loss_count": counts["nonresponsibility_stable_loss_count"],
                "nonresponsibility_fragile_loss_count": counts["nonresponsibility_fragile_loss_count"],
            }
            if row["gain_count"] - row["loss_count"] != row["target_gain"]:
                raise AssertionError("reconstructed target gain mismatch")
            if row["responsibility_gain_count"] != int(candidate["responsibility_residual_gain_count"]):
                raise AssertionError("reconstructed responsibility gain mismatch")
            row["collateral_class"] = collateral_class(row)
            candidate_rows.append(row)
            example_rows.extend(candidate_examples)
    regressions = [row for row in candidate_rows if row["target_gain"] < 0]
    classes = Counter(row["collateral_class"] for row in regressions)
    report = {
        "audit_version": AUDIT_VERSION, "status": "PASS",
        "m20_cell_count": len(cells), "candidate_count": len(candidate_rows),
        "target_regression_candidate_count": len(regressions),
        "collateral_class_counts": dict(sorted(classes.items())),
        "responsibility_gain_count": sum(row["responsibility_gain_count"] for row in candidate_rows),
        "regression_responsibility_gain_count": sum(row["responsibility_gain_count"] for row in regressions),
        "nonresponsibility_loss_count": sum(row["nonresponsibility_loss_count"] for row in candidate_rows),
        "regression_nonresponsibility_loss_count": sum(row["nonresponsibility_loss_count"] for row in regressions),
        "regression_stable_loss_count": sum(row["nonresponsibility_stable_loss_count"] for row in regressions),
        "regression_fragile_loss_count": sum(row["nonresponsibility_fragile_loss_count"] for row in regressions),
        "regression_unique_loss_count": sum(row["nonresponsibility_unique_loss_count"] for row in regressions),
        "regression_pivotal_loss_count": sum(row["nonresponsibility_pivotal_loss_count"] for row in regressions),
        "regression_unique_or_pivotal_loss_count": sum(row["nonresponsibility_unique_loss_count"] + row["nonresponsibility_pivotal_loss_count"] for row in regressions),
        "specialization_overwrite_also_broad_count": sum(
            row["collateral_class"] == "SPECIALIZATION_OVERWRITE"
            and row["nonresponsibility_loss_count"] >= 3
            and row["nonresponsibility_loss_count"] >= 2 * max(1, row["responsibility_gain_count"])
            for row in regressions
        ),
        "api_calls": 0, "model_calls": 0, "validation_files_read": 0,
        "test_files_read": 0, "cache_open_mode": "read_only_immutable",
    }
    return candidate_rows, example_rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--protocol_gate", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("audit output must be fresh")
    gate = json.loads(args.protocol_gate.read_text(encoding="utf-8"))
    if gate.get("gate") != "PASS":
        raise SystemExit("Study B protocol gate must PASS")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    summary = json.loads((args.run_root / "probe_summary.json").read_text(encoding="utf-8"))
    candidate_rows, example_rows, report = audit(registry, summary, args.run_root / "_shared_solver_cache.sqlite")
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "candidate_collateral.csv", candidate_rows)
    write_csv(args.out / "gain_loss_examples.csv", example_rows)
    report["audit_spec_sha256"] = sha256(args.spec)
    report["registry_content_hash"] = registry["registry_content_hash"]
    report["execution_commit"] = summary["execution_commit"]
    write_json(args.out / "audit_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
