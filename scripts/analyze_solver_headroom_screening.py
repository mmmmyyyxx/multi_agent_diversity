from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.solver_headroom_screening_support import (
    ARMS, CANDIDATES, REPORT_ROOT, ROLE_MODEL, RUN_ROOT, SEEDS,
    accepted_update_count, entrant_rows, infrastructure_failure_count,
    read_json, run_dir, sanitization_problems, select_solver, validation_dir,
    write_json,
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    phase_a = read_json(RUN_ROOT / "phase_a" / "availability_smoke_private.json")
    train_gate = read_json(RUN_ROOT / "train_gate.json")
    val_gate = read_json(RUN_ROOT / "validation_gate.json")
    if phase_a["gate"] != "PASS" or train_gate["gate"] != "PASS" or val_gate["gate"] != "PASS":
        raise RuntimeError("all canonical gates must pass")
    entrants = entrant_rows()
    per_seed: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, Any]] = {}
    for entry in entrants:
        key, model = entry["model_key"], entry["solver_model"]
        model_rows = []
        for seed in SEEDS:
            static = read_json(validation_dir(key, seed, "STATIC") / "validation_summary_private.json")
            generic = read_json(validation_dir(key, seed, "GENERIC") / "validation_summary_private.json")
            train_rows = {
                arm: next(
                    row for row in train_gate["rows"]
                    if row["model_key"] == key and row["seed"] == seed and row["arm"] == arm
                ) for arm in ARMS
            }
            terminal = sum(
                train_rows[arm]["terminal_invalid_count"] for arm in ARMS
            ) + static["terminal_invalid_count"] + generic["terminal_invalid_count"]
            resolved = sum(
                train_rows[arm]["resolved_request_count"] for arm in ARMS
            ) + static["resolved_request_count"] + generic["resolved_request_count"]
            first_invalid = sum(
                train_rows[arm]["first_attempt_invalid_count"] for arm in ARMS
            ) + static["first_attempt_invalid_count"] + generic["first_attempt_invalid_count"]
            infra = sum(
                infrastructure_failure_count(run_dir(key, seed, arm)) for arm in ARMS
            )
            row = {
                "model_key": key, "solver_model": model, "seed": seed,
                "static_validation_vote_acc": static["vote_accuracy"],
                "static_validation_oracle_acc": static["oracle_accuracy"],
                "generic_validation_vote_acc": generic["vote_accuracy"],
                "generic_validation_oracle_acc": generic["oracle_accuracy"],
                "generic_minus_static_vote_delta": generic["vote_accuracy"] - static["vote_accuracy"],
                "static_oracle_vote_gap": static["oracle_accuracy"] - static["vote_accuracy"],
                "generic_oracle_vote_gap": generic["oracle_accuracy"] - generic["vote_accuracy"],
                "static_member_accuracies": static["per_agent_accuracies"],
                "generic_member_accuracies": generic["per_agent_accuracies"],
                "accepted_commits": accepted_update_count(run_dir(key, seed, "GENERIC")),
                "terminal_invalid_count": terminal,
                "resolved_request_count": resolved,
                "terminal_invalid_rate": terminal / resolved if resolved else 0.0,
                "first_attempt_invalid_count": first_invalid,
                "infrastructure_failure_count": infra,
            }
            per_seed.append(row)
            model_rows.append(row)
        aggregates[key] = {
            "model_key": key, "solver_model": model,
            "static_vote_accs": [r["static_validation_vote_acc"] for r in model_rows],
            "generic_vote_accs": [r["generic_validation_vote_acc"] for r in model_rows],
            "static_mean_vote_acc": mean([r["static_validation_vote_acc"] for r in model_rows]),
            "generic_mean_vote_acc": mean([r["generic_validation_vote_acc"] for r in model_rows]),
            "mean_vote_uplift": mean([r["generic_minus_static_vote_delta"] for r in model_rows]),
            "generic_vote_win_count": sum(r["generic_minus_static_vote_delta"] > 0 for r in model_rows),
            "static_mean_oracle_vote_gap": mean([r["static_oracle_vote_gap"] for r in model_rows]),
            "generic_mean_oracle_vote_gap": mean([r["generic_oracle_vote_gap"] for r in model_rows]),
            "accepted_commits": sum(r["accepted_commits"] for r in model_rows),
            "terminal_invalid_count": sum(r["terminal_invalid_count"] for r in model_rows),
            "resolved_request_count": sum(r["resolved_request_count"] for r in model_rows),
            "terminal_invalid_rate": (
                sum(r["terminal_invalid_count"] for r in model_rows)
                / sum(r["resolved_request_count"] for r in model_rows)
            ),
            "infrastructure_failure_count": sum(r["infrastructure_failure_count"] for r in model_rows),
            "serious_output_instability": (
                any(r["infrastructure_failure_count"] for r in model_rows)
                or sum(r["terminal_invalid_count"] for r in model_rows)
                / sum(r["resolved_request_count"] for r in model_rows) > 0.01
            ),
        }
    selection = select_solver(aggregates)
    summary = {
        "screening_version": "solver_headroom_screening_v1",
        "role_model": ROLE_MODEL,
        "seeds": list(SEEDS),
        "eligible_solver_count": len(entrants),
        "completed_run_count": len(entrants) * 6,
        "aggregates": aggregates,
        "selection": selection,
        "validation_evaluation_count": len(entrants) * 6,
        "test_evaluation_count": 0,
        "full_method_run": False,
        "test_accessed": False,
    }
    sanitized_phase_a = {
        "phase": "availability_smoke", "gate": phase_a["gate"],
        "candidates": [
            {
                "model_key": row["model_key"], "solver_model": row["solver_model"],
                "priority": row["priority"], "listed": row["listed"],
                "smoke_attempted": row["smoke"]["attempted"],
                "smoke_success": row["smoke"]["success"],
                "status_code": row["smoke"]["status_code"],
                "error_type": row["smoke"]["error_type"],
                "finish_reason": row["smoke"]["finish_reason"],
                "prompt_tokens": row["smoke"]["prompt_tokens"],
                "completion_tokens": row["smoke"]["completion_tokens"],
                "screening_eligible": row["screening_eligible"],
            } for row in phase_a["candidates"]
        ],
        "role_model": {
            "model": ROLE_MODEL, "listed": phase_a["role_model"]["listed"],
            "smoke_success": phase_a["role_model"]["smoke"]["success"],
            "status_code": phase_a["role_model"]["smoke"]["status_code"],
        },
        "excluded_model_requested": phase_a["excluded_model_requested"],
        "successful_smoke_call_count": phase_a["successful_smoke_call_count"],
        "test_calls": 0,
    }
    return per_seed, summary, sanitized_phase_a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("fresh report root required")
    per_seed, summary, phase_a = build()
    selection = summary["selection"]
    args.out.mkdir(parents=True)
    write_json(args.out / "availability_smoke.json", phase_a)
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "solver_selection.json", selection)
    columns = list(per_seed[0].keys())
    with (args.out / "per_seed_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in per_seed:
            encoded = dict(row)
            for field in ("static_member_accuracies", "generic_member_accuracies"):
                encoded[field] = json.dumps(encoded[field], separators=(",", ":"))
            writer.writerow(encoded)
    lines = [
        "# Solver Model Headroom Screening",
        "", "```text", "FULL_METHOD_NOT_RUN=true", "TEST_ACCESSED=false", "```", "",
        f"All shared optimizer roles used `{ROLE_MODEL}`. Phase A admitted "
        f"{summary['eligible_solver_count']} Solver model(s).", "",
        f"Frozen decision: **{selection['decision']}**",
        f"Selected Solver: `{selection['selected_solver_model'] or 'none'}`", "",
        "The screening used only Static and canonical Generic. Validation was a",
        "post-training read-only evaluation and never selected or changed a state.",
        "No test split was loaded by the screening evaluator.",
    ]
    (args.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "validation_report.txt").write_text(
        "SOLVER_HEADROOM_SCREENING\n"
        "phase_a_gate=PASS\ntrain_gate=PASS\nvalidation_gate=PASS\n"
        f"eligible_solver_count={summary['eligible_solver_count']}\n"
        f"completed_run_count={summary['completed_run_count']}\n"
        f"validation_evaluation_count={summary['validation_evaluation_count']}\n"
        "test_evaluation_count=0\nfull_method_run=false\n"
        "fact_assertions=PENDING\ntests=PENDING\nsanitation=PENDING\n",
        encoding="utf-8",
    )
    for path in args.out.iterdir():
        if path.suffix == ".json":
            problems = sanitization_problems(read_json(path))
            if problems:
                raise RuntimeError(f"sanitization failed: {path.name}: {problems}")
    print({
        "decision": selection["decision"],
        "selected_solver": selection["selected_solver_model"],
        "report": args.out.as_posix(),
    })


if __name__ == "__main__":
    main()
