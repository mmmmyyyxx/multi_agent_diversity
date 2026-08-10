from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REPORT_VERSION = "final_method_complete_evaluation_v15_reduced_v1"
REQUIRED_STAGE_AUDIT_VERSION = "final_method_stage_gate_v15_reduced_v1"
ABLATION_COMPARISONS = (
    (
        "shared_static_reference",
        "shared_generic_evolution",
        "generic_evolution_effect",
    ),
    (
        "shared_generic_evolution",
        "shared_member_aware_dual_target",
        "member_aware_dual_target_effect",
    ),
    (
        "shared_member_aware_dual_target",
        "shared_responsibility_conditioned_dual_target",
        "responsibility_conditioned_proposal_effect",
    ),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _member_outcome(baseline: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    baseline_counts = [int(value) for value in baseline["per_agent_correct_counts"]]
    selected_counts = [int(value) for value in selected["per_agent_correct_counts"]]
    gains = [current - initial for current, initial in zip(selected_counts, baseline_counts, strict=True)]
    vote_delta = int(selected["vote_correct_count"]) - int(baseline["vote_correct_count"])
    g_min, g_sum = min(gains), sum(gains)
    return {
        "baseline_vote_correct_count": int(baseline["vote_correct_count"]),
        "final_vote_correct_count": int(selected["vote_correct_count"]),
        "vote_delta": vote_delta,
        "baseline_vote_accuracy": float(baseline["plurality_vote_acc"]),
        "final_vote_accuracy": float(selected["plurality_vote_acc"]),
        "per_member_correct_count_gains": gains,
        "g_min": g_min,
        "g_sum": g_sum,
        "N_positive": sum(value > 0 for value in gains),
        "strict_pareto_success": (
            vote_delta >= 0 and g_min >= 0 and g_sum >= 0
            and (vote_delta > 0 or g_min > 0 or g_sum > 0)
        ),
    }


def _aggregate(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values) if values else "unavailable",
        "median": statistics.median(values) if values else "unavailable",
        "positive_count": sum(value > 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
        "direction_consistent": (
            all(value >= 0 for value in values) or all(value <= 0 for value in values)
            if values else "unavailable"
        ),
    }


def _run_map(gate: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {
        (row["task"], int(row["seed"]), row["setting"]): row
        for row in gate["runs"]
        if row.get("complete")
    }


def _candidate_summary(run_dir: Path) -> dict[str, Any]:
    decisions = _read_jsonl(run_dir / "candidate_decisions.jsonl")
    candidates = [
        candidate
        for decision in decisions
        for candidate in decision.get("candidates", [])
        if isinstance(candidate.get("constraint"), dict)
    ]
    target_positive = sum(int(row["constraint"].get("target_gain", 0)) > 0 for row in candidates)
    vote_positive = sum(int(row["constraint"].get("vote_net_gain", 0)) > 0 for row in candidates)
    acceptable = sum(bool(row["constraint"].get("passed")) for row in candidates)
    assigned_repairs = sum(
        int((row.get("evaluation") or {}).get("marginal", {}).get(
            "assigned_residual_repair_count", 0
        ))
        for row in candidates
    )
    accepted_types = {"target-only": 0, "vote-only": 0, "target-and-vote": 0}
    for decision in decisions:
        accepted_hash = decision.get("accepted_prompt_hash")
        if not accepted_hash:
            continue
        accepted = next(
            (row for row in decision.get("candidates", []) if row.get("prompt_hash") == accepted_hash),
            None,
        )
        constraint = (accepted or {}).get("constraint") or {}
        target = int(constraint.get("target_gain", 0)) > 0
        vote = int(constraint.get("vote_net_gain", 0)) > 0
        if target and vote:
            accepted_types["target-and-vote"] += 1
        elif target:
            accepted_types["target-only"] += 1
        elif vote:
            accepted_types["vote-only"] += 1
    return {
        "update_count": len(decisions),
        "evaluated_candidate_count": len(candidates),
        "target_positive_candidate_rate": target_positive / len(candidates) if candidates else "unavailable",
        "vote_positive_candidate_rate": vote_positive / len(candidates) if candidates else "unavailable",
        "acceptable_candidate_rate": acceptable / len(candidates) if candidates else "unavailable",
        "assigned_residual_repair_count": assigned_repairs,
        "acceptance_types": accepted_types,
    }


def _training_summary(run_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(run_dir / "training_dynamics.jsonl")
    accepted = [row for row in rows if row.get("accepted")]
    target_sequence = [
        row.get("target_agent_id") for row in rows if int(row.get("update_index", -1)) >= 0
    ]
    accepted_by_member = {
        str(agent): sum(int(row.get("target_agent_id", -1)) == agent for row in accepted)
        for agent in range(5)
    }
    selected_by_member = {
        str(agent): sum(target == agent for target in target_sequence)
        for agent in range(5)
    }
    rejection_streak = longest = 0
    for row in rows:
        if int(row.get("update_index", -1)) < 0:
            continue
        if row.get("accepted"):
            rejection_streak = 0
        else:
            rejection_streak += 1
            longest = max(longest, rejection_streak)
    accepted_values = list(accepted_by_member.values())
    return {
        "trajectory_length": len(rows),
        "selected_updates_by_member": selected_by_member,
        "accepted_updates_by_member": accepted_by_member,
        "never_selected_members": [agent for agent in range(5) if selected_by_member[str(agent)] == 0],
        "never_accepted_members": [agent for agent in range(5) if accepted_by_member[str(agent)] == 0],
        "top_2_accepted_share": (
            sum(sorted(accepted_values, reverse=True)[:2]) / sum(accepted_values)
            if sum(accepted_values) else "unavailable"
        ),
        "max_min_accepted_gap": max(accepted_values) - min(accepted_values),
        "last_accepted_update": max(
            (int(row["update_index"]) for row in accepted),
            default="unavailable",
        ),
        "longest_rejection_streak": longest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_identity", type=Path, required=True)
    parser.add_argument("--code_audit_dir", type=Path, required=True)
    parser.add_argument("--pilot_gate", type=Path, required=True)
    parser.add_argument("--disambiguation_gate", type=Path, required=True)
    parser.add_argument("--cross_task_gate", type=Path, required=True)
    parser.add_argument("--disambiguation_root", type=Path, required=True)
    parser.add_argument("--cross_task_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    source = _read_json(args.source_identity)
    pilot = _read_json(args.pilot_gate)
    disambiguation = _read_json(args.disambiguation_gate)
    cross = _read_json(args.cross_task_gate)
    if any(gate.get("gate") != "PASS" for gate in (pilot, disambiguation, cross)):
        raise ValueError("Every stage gate must pass before the final report is built")
    if any(
        gate.get("audit_version") != REQUIRED_STAGE_AUDIT_VERSION
        for gate in (pilot, disambiguation, cross)
    ):
        raise ValueError(
            "Every stage must use the matched-observation-aware v2 audit"
        )
    if any(
        not all(
            row.get("passed") is True
            for row in gate.get("matched_observation_consistency", [])
        )
        for gate in (pilot, disambiguation, cross)
    ):
        raise ValueError("Matched observation consistency failed")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    disamb_runs = _run_map(disambiguation)
    cross_runs = _run_map(cross)

    disamb_rows = []
    for seed in (44, 45, 46):
        baseline = disamb_runs[
            ("disambiguation_qa", seed, "shared_static_reference")
        ]["selected_test"]
        for setting in (
            "shared_static_reference",
            "shared_generic_evolution",
            "shared_member_aware_dual_target",
            "shared_responsibility_conditioned_dual_target",
        ):
            run = disamb_runs[("disambiguation_qa", seed, setting)]
            outcome = _member_outcome(baseline, run["selected_test"])
            disamb_rows.append({
                "seed": seed,
                "setting": setting,
                **outcome,
                "accepted_update_count": run["accepted_update_count"],
                "tokens_per_accepted_update": run["tokens_per_accepted_update"],
            })

    ablation_rows = []
    for left, right, comparison_axis in ABLATION_COMPARISONS:
        raw = []
        for seed in (44, 45, 46):
            lhs = next(row for row in disamb_rows if row["seed"] == seed and row["setting"] == left)
            rhs = next(row for row in disamb_rows if row["seed"] == seed and row["setting"] == right)
            raw.append({
                "seed": seed,
                "vote_accuracy_delta": rhs["final_vote_accuracy"] - lhs["final_vote_accuracy"],
                "g_min_delta": rhs["g_min"] - lhs["g_min"],
                "g_sum_delta": rhs["g_sum"] - lhs["g_sum"],
                "N_positive_delta": rhs["N_positive"] - lhs["N_positive"],
            })
        ablation_rows.append({
            "comparison_axis": comparison_axis,
            "comparison": f"{left} -> {right}",
            "per_seed": raw,
            "vote_accuracy_delta": _aggregate([row["vote_accuracy_delta"] for row in raw]),
            "g_min_delta": _aggregate([row["g_min_delta"] for row in raw]),
            "g_sum_delta": _aggregate([row["g_sum_delta"] for row in raw]),
        })

    cross_rows = []
    for task in ("geometric_shapes", "ruin_names"):
        for seed in (44, 45, 46):
            baseline = cross_runs[
                (task, seed, "shared_static_reference")
            ]["selected_test"]
            full = cross_runs[
                (task, seed, "shared_responsibility_conditioned_dual_target")
            ]
            cross_rows.append({
                "task": task,
                "seed": seed,
                **_member_outcome(baseline, full["selected_test"]),
                "accepted_update_count": full["accepted_update_count"],
            })

    training, proposals, votes = [], [], []
    for run in (*disambiguation["runs"], *cross["runs"]):
        if (
            not run.get("complete")
            or run["setting"] == "shared_static_reference"
        ):
            continue
        root = args.disambiguation_root if run["task"] == "disambiguation_qa" else args.cross_task_root
        run_dir = root / run["task"] / f"{run['setting']}_seed{run['seed']}"
        label = {"task": run["task"], "seed": run["seed"], "setting": run["setting"]}
        training.append({**label, **_training_summary(run_dir)})
        proposals.append({**label, **_candidate_summary(run_dir)})
        differentiation = _read_json(run_dir / "final_test_differentiation.json")
        votes.append({
            **label,
            **{
                key: differentiation.get(key, "unavailable")
                for key in (
                    "oracle_correct_count", "mean_G", "mean_H", "mean_M",
                    "oracle_covered_but_vote_wrong_rate", "all_wrong_rate",
                    "mean_off_diagonal_double_fault", "mean_off_diagonal_same_wrong_excess",
                )
            },
        })

    cost = {
        "pilot": pilot["cost"],
        "disambiguation": disambiguation["cost"],
        "cross_task": cross["cost"],
        "total_tokens": sum(
            int(gate["cost"]["total_tokens"]) for gate in (pilot, disambiguation, cross)
        ),
        "role_breakdown": "unavailable",
        "wall_clock_time": "unavailable",
    }
    invalid = {
        "failed_or_incomplete_runs": [
            row for gate in (pilot, disambiguation, cross)
            for row in gate["runs"] if not row.get("complete")
        ],
        "infrastructure_failure_count": sum(
            int(row.get("infrastructure_failure_count", 0))
            for gate in (pilot, disambiguation, cross)
            for row in gate["runs"]
        ),
        "retried_runs": "unavailable",
    }

    _write(out / "source_identity.json", source)
    _write(out / "experiment_manifest.json", {
        "report_version": REPORT_VERSION,
        "pilot_run_count": pilot["complete_run_count"],
        "disambiguation_run_count": disambiguation["complete_run_count"],
        "cross_task_run_count": cross["complete_run_count"],
        "stage_gates": {"pilot": "PASS", "disambiguation": "PASS", "cross_task": "PASS"},
    })
    _write(out / "code_audit_summary.json", _read_json(args.code_audit_dir / "semantic_alignment_matrix.json"))
    _write(out / "disambiguation_main_table.json", disamb_rows)
    _write(out / "ablation_comparisons.json", ablation_rows)
    _write(out / "cross_task_baseline_full.json", cross_rows)
    _write(out / "training_dynamics_summary.json", training)
    _write(out / "member_gain_summary.json", {
        "disambiguation": disamb_rows,
        "cross_task": cross_rows,
    })
    _write(out / "proposal_realization_summary.json", proposals)
    _write(out / "vote_structure_summary.json", votes)
    _write(out / "cost_summary.json", cost)
    _write(out / "invalid_and_failed_runs.json", invalid)
    (out / "README.md").write_text(
        "\n".join((
            "# Final Method Complete Evaluation",
            "",
            "- Method: Member-Aware Prompt-Team Optimization",
            "- Stage A/B gate: PASS",
            "- Seed46 short pilot gate: PASS",
            "- disambiguation_qa full comparison gate: PASS",
            "- cross-task baseline/full gate: PASS",
            "",
            "All files are sanitized aggregates containing no questions, labels, prompts, raw model responses, credentials, caches, or checkpoints.",
            "",
        )),
        encoding="utf-8",
    )
    print(json.dumps({
        "report_version": REPORT_VERSION,
        "out_dir": str(out),
        "disambiguation_rows": len(disamb_rows),
        "cross_task_rows": len(cross_rows),
        "total_tokens": cost["total_tokens"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
