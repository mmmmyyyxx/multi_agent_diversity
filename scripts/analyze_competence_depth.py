"""Offline audit for competence-depth experiment roots. Never calls model APIs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json(path: Path, default: Any):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


METRICS = [
    "vote_acc", "plurality_vote_acc", "strict_plurality_win_rate", "plurality_top_tie_rate",
    "oracle_acc", "aggregation_gap", "oracle_minus_plurality_vote", "c2_minus_plurality_vote",
    "c3_minus_plurality_vote", "mean_plurality_margin_votes", "mean_normalized_plurality_margin",
    "plurality_pivotal_fix_opportunity_rate", "plurality_pivotal_fix_rate",
    "plurality_pivotal_hold_rate", "mean_individual_acc", "bottom2_mean_acc",
    "coverage_depth_c1", "coverage_depth_c2", "coverage_depth_c3", "coverage_depth_c4",
    "coverage_depth_c5", "c1_minus_c2", "c2_minus_c3", "max_minority_rescue_share",
    "minority_rescue_hhi", "specialization_strength_final", "mean_specialization_strength",
    "prompt_overlength_rejection_count", "truncated_prompt_count",
    "all_wrong_rate", "initial_competence_probe_bottom2", "final_competence_probe_bottom2",
    "competence_probe_bottom2_gain", "initial_competence_probe_mean_acc",
    "final_competence_probe_mean_acc", "competence_probe_mean_acc_gain",
    "initial_competence_probe_c1", "final_competence_probe_c1", "competence_probe_c1_gain",
    "initial_competence_probe_c2", "final_competence_probe_c2", "competence_probe_c2_gain",
    "first_nonzero_specialization_epoch", "effective_specialization_epoch_count",
    "max_specialization_strength", "progressive_stage_exercised",
    "progressive_stage_not_exercised_reason", "depth1_guard_rejection_count",
    "distinct_top_capability_family_count", "dominant_capability_family_share",
    "capability_family_hhi", "mean_pairwise_capability_profile_cosine",
    "catastrophic_accuracy_guard_rejection_count", "soft_error_dependence_penalty_count",
    "soft_cycle_penalty_count", "soft_mechanism_shift_penalty_count",
    "exploration_candidate_count", "exploration_slot_occupancy_rate",
    "exploration_to_active_conversion_count", "distinct_final_mechanism_signature_count",
    "dominant_final_mechanism_signature_share", "mean_pairwise_mechanism_signature_distance",
] + [f"correct_agent_count_{index}" for index in range(6)]


PLURALITY_CANDIDATE_METRICS = (
    "plurality_vote_gain_rate", "plurality_vote_loss_rate", "plurality_vote_net_delta",
    "plurality_pivotal_fix_opportunity_rate", "plurality_pivotal_fix_rate",
    "plurality_pivotal_loss_rate", "baseline_plurality_margin_votes",
    "candidate_plurality_margin_votes", "plurality_margin_vote_delta",
)
DEPTH2_CANDIDATE_METRICS = ("depth2_gain_rate", "depth2_loss_rate", "depth2_net_delta")
REWARD_COMPONENT_METRICS = ("competence_reward_component", "stage_aux_objective", "reward")


def present_value(metrics: Dict[str, Any], key: str) -> Any:
    return metrics[key] if key in metrics else ""


def analyze(root: Path) -> Dict[str, int]:
    run_rows, agent_rows, epoch_rows, candidate_rows = [], [], [], []
    schedule_rows, guard_rows, mechanism_rows = [], [], []
    for history_path in sorted(root.rglob("history.json")):
        run_dir = history_path.parent
        history = read_json(history_path, [])
        if not isinstance(history, list):
            continue
        setting_seed = run_dir.name
        task = run_dir.parent.name
        final = next((record.get("test", {}) for record in reversed(history) if isinstance(record, dict) and isinstance(record.get("test"), dict)), {})
        run_row = {"task": task, "setting_seed": setting_seed, "run_dir": str(run_dir), **{key: present_value(final, key) for key in METRICS}}
        run_row["metric_missing"] = any(key not in final for key in METRICS)
        run_rows.append(run_row)
        mechanism_row = {
            "task": task,
            "setting_seed": setting_seed,
            "top_capability_family_per_agent": present_value(final, "top_capability_family_per_agent"),
            "distinct_top_capability_family_count": present_value(final, "distinct_top_capability_family_count"),
            "dominant_capability_family_share": present_value(final, "dominant_capability_family_share"),
            "capability_family_hhi": present_value(final, "capability_family_hhi"),
            "mean_pairwise_capability_profile_cosine": present_value(final, "mean_pairwise_capability_profile_cosine"),
            "metric_missing": "top_capability_family_per_agent" not in final,
            "declared_mechanism_per_agent": {},
        }
        mechanism_rows.append(mechanism_row)
        for agent_id, accuracy in enumerate(final.get("per_agent_acc", [])):
            agent_rows.append({
                "task": task, "setting_seed": setting_seed, "agent_id": agent_id, "accuracy": accuracy,
                "minority_rescue_count": (final.get("minority_rescue_count_per_agent", []) + [0] * 5)[agent_id],
                "unique_correct_count": (final.get("unique_correct_count_per_agent", []) + [0] * 5)[agent_id],
                "minority_rescue_share": (final.get("minority_rescue_share_per_agent", []) + [0] * 5)[agent_id],
            })
        for record in history:
            if not isinstance(record, dict):
                continue
            for split in ("train", "val", "test"):
                metrics = record.get(split)
                if isinstance(metrics, dict):
                    epoch_rows.append({"task": task, "setting_seed": setting_seed, "epoch": record.get("epoch"), "split": split, **{key: present_value(metrics, key) for key in METRICS}})
            schedule = record.get("competence_schedule")
            if isinstance(schedule, dict):
                schedule_rows.append({"task": task, "setting_seed": setting_seed, **schedule, "metric_missing": False})
        run_candidate_rows = []
        for row in read_jsonl(run_dir / "update_logs.jsonl"):
            metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else row
            if row.get("event") in {"candidate_evaluation", "beam_candidate"} or "candidate_id" in row:
                required = PLURALITY_CANDIDATE_METRICS + DEPTH2_CANDIDATE_METRICS + REWARD_COMPONENT_METRICS
                candidate_row = {
                    "task": task, "setting_seed": setting_seed, "epoch": row.get("epoch"), "step": row.get("step"),
                    "agent_id": row.get("agent_id"), "candidate_id": row.get("candidate_id"),
                    **{key: metrics[key] if key in metrics else row[key] if key in row else "" for key in (
                        "candidate_target_accuracy", "depth1_gain_rate", "depth1_loss_rate", "depth1_net_delta",
                        "depth2_gain_rate", "depth2_loss_rate", "depth2_net_delta", "depth3_gain_rate",
                        "depth3_loss_rate", "depth3_net_delta", "vote_gain_rate", "vote_loss_rate",
                        *PLURALITY_CANDIDATE_METRICS,
                        "competence_reward_component", "stage_aux_objective", "specialization_strength", "reward",
                        "candidate_prompt_char_count", "candidate_prompt_over_soft_limit",
                        "candidate_prompt_overlength_rejected", "candidate_prompt_ends_with_sentence_boundary",
                        "declared_mechanism", "accepted", "is_top1", "active_prompt_changed",
                        "candidate_type", "mechanism_signature", "mechanism_signature_distance",
                        "hard_guard_passed", "hard_rejection_reason", "soft_guard_penalty",
                        "raw_reward", "penalized_reward", "beam_slot",
                    )},
                    "metric_missing": any(key not in metrics and key not in row for key in required),
                    "plurality_metric_missing": any(key not in metrics and key not in row for key in PLURALITY_CANDIDATE_METRICS),
                    "depth2_metric_missing": any(key not in metrics and key not in row for key in DEPTH2_CANDIDATE_METRICS),
                    "candidate_reward_component_missing": any(key not in metrics and key not in row for key in REWARD_COMPONENT_METRICS),
                }
                candidate_rows.append(candidate_row)
                run_candidate_rows.append(candidate_row)
                declared = str(candidate_row.get("declared_mechanism", "") or "").strip()
                if declared and bool(candidate_row.get("accepted")) and bool(candidate_row.get("is_top1")):
                    mechanism_row["declared_mechanism_per_agent"][str(row.get("agent_id"))] = declared
            if row.get("event") == "beam_update_summary":
                guard_rows.append({
                    "task": task, "setting_seed": setting_seed, "epoch": row.get("epoch"),
                    "step": row.get("step"), "agent_id": row.get("agent_id"),
                    **{key: present_value(row, key) for key in (
                        "candidate_count", "depth1_guard_rejection_count",
                        "accuracy_guard_rejection_count", "invalid_guard_rejection_count",
                        "dependence_guard_rejection_count", "pareto_not_retained_count",
                        "retained_candidate_count", "active_prompt_changed_count",
                    )},
                    "metric_missing": "depth1_guard_rejection_count" not in row,
                })
        denominator = len(run_candidate_rows)
        run_row["plurality_metric_coverage"] = (
            sum(not row["plurality_metric_missing"] for row in run_candidate_rows) / denominator if denominator else ""
        )
        run_row["depth2_metric_coverage"] = (
            sum(not row["depth2_metric_missing"] for row in run_candidate_rows) / denominator if denominator else ""
        )
        run_row["candidate_reward_component_coverage"] = (
            sum(not row["candidate_reward_component_missing"] for row in run_candidate_rows) / denominator if denominator else ""
        )
    paired = []
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in run_rows:
        seed = row["setting_seed"].rsplit("_seed", 1)[-1] if "_seed" in row["setting_seed"] else ""
        grouped.setdefault((row["task"], seed), []).append(row)
    for (task, seed), rows in grouped.items():
        baseline = next((row for row in rows if "residual_cycle_guard" in row["setting_seed"]), None)
        if baseline:
            for row in rows:
                if row is baseline:
                    continue
                paired.append({"task": task, "seed": seed, "setting": row["setting_seed"], **{
                    f"delta_{key}": float(row[key]) - float(baseline[key])
                    for key in METRICS
                    if isinstance(row.get(key), (int, float)) and isinstance(baseline.get(key), (int, float))
                }})
    outputs = {
        "competence_depth_run_summary.csv": run_rows,
        "competence_depth_agent_summary.csv": agent_rows,
        "competence_depth_epoch_summary.csv": epoch_rows,
        "competence_depth_candidate_summary.csv": candidate_rows,
        "competence_depth_paired_comparison.csv": paired,
        "competence_depth_schedule_summary.csv": schedule_rows,
        "competence_depth_mechanism_summary.csv": mechanism_rows,
        "competence_depth_guard_summary.csv": guard_rows,
    }
    for name, rows in outputs.items():
        write_csv(root / name, rows)
    not_exercised = sum(row.get("progressive_stage_exercised") is False for row in run_rows)
    report = [
        "# Competence Depth Audit", "", f"Runs: {len(run_rows)}", f"Epoch/split rows: {len(epoch_rows)}",
        f"Candidate rows: {len(candidate_rows)}", f"Runs without an exercised progressive stage: {not_exercised}", "",
        "Schedule rows audit the fixed optimization probe, online/static competence separation, gates, and applied specialization strength.",
        "Guard rows report candidate retention and C1-preservation rejection counts.",
        "Mechanism rows report zero-API capability-profile differentiation.",
        "", "A lower aggregation gap is only favorable when C1/oracle coverage is preserved.",
    ]
    if not_exercised:
        report.extend(["", "Warning: This run did not exercise progressive specialization; it only evaluated the competence phase."])
    (root / "COMPETENCE_DEPTH_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"runs": len(run_rows), "epochs": len(epoch_rows), "candidates": len(candidate_rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_root), indent=2))


if __name__ == "__main__":
    main()
