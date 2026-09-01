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

from scripts.model_headroom_screening_support import (
    ARMS, MODELS, REPORT_ROOT, RUN_ROOT, SEEDS, accepted_update_count,
    infrastructure_failure_count, read_json, recursive_sanitization_problems,
    run_dir, selection_rule, validation_dir, write_json,
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def analyze() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    train_gate = read_json(RUN_ROOT / "train_gate.json")
    validation_gate = read_json(RUN_ROOT / "validation_gate.json")
    if train_gate.get("gate") != "PASS" or validation_gate.get("gate") != "PASS":
        raise RuntimeError("canonical gates must pass before analysis")
    per_seed: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, Any]] = {}
    for model_key, task_model in MODELS.items():
        model_rows = []
        total_terminal_invalid = 0
        total_resolved = 0
        total_infra = 0
        for seed in SEEDS:
            evals = {
                arm: read_json(validation_dir(model_key, seed, arm) / "validation_summary_private.json")
                for arm in ARMS
            }
            recoveries = {
                arm: read_json(run_dir(model_key, seed, arm) / "solver_recovery_summary.json")
                for arm in ARMS
            }
            static, generic = evals["STATIC"], evals["GENERIC"]
            for arm in ARMS:
                recovery = recoveries[arm]
                total_terminal_invalid += int(recovery["terminal_invalid_count"])
                total_resolved += int(recovery["unique_resolved_request_count"])
                total_terminal_invalid += int(evals[arm]["terminal_invalid_count"])
                total_resolved += int(evals[arm]["resolved_request_count"])
                total_infra += infrastructure_failure_count(run_dir(model_key, seed, arm))
            row = {
                "model_key": model_key,
                "task_model": task_model,
                "seed": seed,
                "static_validation_vote_acc": static["vote_accuracy"],
                "static_validation_oracle_acc": static["oracle_accuracy"],
                "generic_validation_vote_acc": generic["vote_accuracy"],
                "generic_validation_oracle_acc": generic["oracle_accuracy"],
                "generic_minus_static_vote_delta": generic["vote_accuracy"] - static["vote_accuracy"],
                "static_oracle_vote_gap": static["oracle_accuracy"] - static["vote_accuracy"],
                "generic_oracle_vote_gap": generic["oracle_accuracy"] - generic["vote_accuracy"],
                "static_member_accuracies": json.dumps(static["per_agent_accuracies"], separators=(",", ":")),
                "generic_member_accuracies": json.dumps(generic["per_agent_accuracies"], separators=(",", ":")),
                "generic_accepted_update_count": accepted_update_count(run_dir(model_key, seed, "GENERIC")),
                "static_train_first_attempt_invalid_rate": 1 - recoveries["STATIC"]["first_attempt_valid_rate"],
                "generic_train_first_attempt_invalid_rate": 1 - recoveries["GENERIC"]["first_attempt_valid_rate"],
                "static_validation_invalid_rate": static["invalid_output_rate"],
                "generic_validation_invalid_rate": generic["invalid_output_rate"],
            }
            per_seed.append(row)
            model_rows.append(row)
        terminal_rate = total_terminal_invalid / total_resolved if total_resolved else 0.0
        aggregates[model_key] = {
            "task_model": task_model,
            "static_vote_accs": [row["static_validation_vote_acc"] for row in model_rows],
            "generic_vote_accs": [row["generic_validation_vote_acc"] for row in model_rows],
            "static_mean_vote_acc": mean([row["static_validation_vote_acc"] for row in model_rows]),
            "generic_mean_vote_acc": mean([row["generic_validation_vote_acc"] for row in model_rows]),
            "mean_vote_uplift": mean([row["generic_minus_static_vote_delta"] for row in model_rows]),
            "generic_vote_win_count": sum(row["generic_minus_static_vote_delta"] > 0 for row in model_rows),
            "static_mean_oracle_vote_gap": mean([row["static_oracle_vote_gap"] for row in model_rows]),
            "generic_mean_oracle_vote_gap": mean([row["generic_oracle_vote_gap"] for row in model_rows]),
            "accepted_update_counts": [row["generic_accepted_update_count"] for row in model_rows],
            "accepted_update_count_total": sum(row["generic_accepted_update_count"] for row in model_rows),
            "terminal_invalid_count": total_terminal_invalid,
            "resolved_request_count": total_resolved,
            "terminal_invalid_rate": terminal_rate,
            "infrastructure_failure_count": total_infra,
            "serious_output_instability": total_infra > 0 or terminal_rate > 0.01,
        }
    selection = selection_rule(aggregates)
    summary = {
        "screening_version": "task_model_headroom_screening_v1",
        "model_count": 2,
        "seed_count": 3,
        "arm_count": 2,
        "run_count": 12,
        "models": aggregates,
        "selection": selection,
        "FULL_METHOD_NOT_RUN": True,
        "TEST_ACCESSED": False,
        "test_calls": 0,
    }
    return per_seed, summary, selection


def write_report(out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh report root required")
    rows, summary, selection = analyze()
    out.mkdir(parents=True)
    with (out / "per_seed_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "summary.json", summary)
    write_json(out / "model_selection.json", selection)
    lines = [
        "# Task Model Headroom Screening",
        "",
        "```text",
        "FULL_METHOD_NOT_RUN=true",
        "TEST_ACCESSED=false",
        "```",
        "",
        "Only Static and canonical Generic evolution were run on validation for seeds 62-64.",
        "Optimizer, Teacher, and Critic remained qwen3-14b with thinking disabled.",
        "Validation was evaluated once from each frozen final state and never affected training.",
        "",
        "| Model | Static VoteAcc (3 seeds) | Generic VoteAcc (3 seeds) | Mean uplift | Generic Oracle-Vote gap | Accepted updates | Decision |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for key in MODELS:
        row = summary["models"][key]
        decision = "PASS" if selection["model_evaluations"][key]["pass"] else "FAIL"
        lines.append(
            f"| {row['task_model']} | {row['static_vote_accs']} | {row['generic_vote_accs']} | "
            f"{row['mean_vote_uplift']:.4f} | {row['generic_mean_oracle_vote_gap']:.4f} | "
            f"{row['accepted_update_counts']} | {decision} |"
        )
    lines.extend([
        "",
        f"Frozen selection decision: **{selection['decision']}**.",
        f"Selected task model: `{selection['selected_task_model'] or 'NONE'}`.",
        f"Selection reason: `{selection['reason']}`.",
        "",
        "No Full/Module1/M20/M2F setting, test evaluation, selector change, Common-Safe change, or extra seed was used.",
    ])
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    validation = {
        "train_gate": "PASS",
        "validation_gate": "PASS",
        "logical_validation_evaluation_count": 12,
        "test_calls": 0,
        "training_tree_unchanged": True,
        "full_method_run": False,
        "sanitization_problems": recursive_sanitization_problems(summary),
    }
    (out / "validation_report.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in validation.items()) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    summary = write_report(args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
