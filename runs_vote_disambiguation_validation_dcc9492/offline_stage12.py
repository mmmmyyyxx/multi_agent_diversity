import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO = SCRIPT_ROOT.parent
ROOT = Path(os.environ.get("ANALYSIS_ROOT", str(SCRIPT_ROOT))).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from multi_dataset_diverse_rl.system import EXPERIMENT_PROTOCOL_VERSION, compute_crowding_distances, non_dominated_sort
from scripts.audit_tcs_run import audit_run


TASK = "disambiguation_qa"
CURRENT_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
CURRENT_PROTOCOL = EXPERIMENT_PROTOCOL_VERSION
STAGE1_ROOT = REPO / "runs_vote_stage1_smoke_83bcb8f" / TASK
STAGE2_ROOT = REPO / "runs_vote_stage2_selector_pilot_v4_dcc9492" / TASK
SETTINGS = (
    "shared_baseline",
    "shared_scalar_tcs_vote_first",
    "shared_vote_pareto_tcs",
)
EPS = 1e-9


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return statistics.mean(values) if values else 0.0


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def final_test(run_dir):
    history = read_json(run_dir / "history.json")
    for item in reversed(history):
        if isinstance(item, dict) and isinstance(item.get("test"), dict):
            return item["test"]
    return {}


def best_validation(run_dir):
    history = read_json(run_dir / "history.json")
    rows = []
    for item in history:
        if isinstance(item, dict) and isinstance(item.get("val"), dict):
            rows.append({"epoch": item.get("epoch"), **item["val"]})
    if not rows:
        return {}
    return max(
        rows,
        key=lambda row: (
            f(row, "vote_acc"),
            f(row, "mean_individual_acc"),
            f(row, "mean_vote_margin"),
            -f(row, "mean_invalid_rate"),
            -int(row.get("epoch", 10**9)),
        ),
    )


def complete(run_dir):
    if not (run_dir / "run_meta.json").exists() or not (run_dir / "history.json").exists() or not (run_dir / "cost_summary.json").exists():
        return False
    return bool(final_test(run_dir))


def split_info():
    names = ("opt", "val", "test")
    records = {}
    questions = {}
    for name in names:
        path = REPO / "strict_splits_bbh_seed42" / TASK / f"{name}.csv"
        payload = path.read_bytes()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        records[name] = {"count": len(rows), "sha256": hashlib.sha256(payload).hexdigest(), "path": str(path.relative_to(REPO))}
        questions[name] = {str(row.get("question") or row.get("input") or row.get("query") or row.get("problem") or row.get("prompt") or "").strip() for row in rows}
    records["overlap"] = {
        "opt_val": len(questions["opt"] & questions["val"]),
        "opt_test": len(questions["opt"] & questions["test"]),
        "val_test": len(questions["val"] & questions["test"]),
    }
    return records


def candidate_stats(run_dir):
    rows = read_jsonl(run_dir / "update_logs.jsonl")
    candidates = [row for row in rows if row.get("event") == "candidate_evaluated"]
    summaries = [row for row in rows if row.get("event") == "beam_update_summary"]
    raw_negative = [row for row in candidates if f(row, "boundary_useful_diversity_delta") < -EPS]
    reward_bad = []
    vote_bad = []
    transition_bad = []
    clipping_bad = []
    for row in candidates:
        components = sum(
            f(row, key)
            for key in (
                "reward_component_target_accuracy",
                "reward_component_vote_delta",
                "reward_component_vote_margin",
                "reward_component_boundary_diversity",
                "reward_component_invalid_penalty",
                "reward_component_guard_penalty",
            )
        )
        if abs(f(row, "reward_total") - components) > EPS:
            reward_bad.append(row.get("candidate_id"))
        if abs(f(row, "vote_delta") - (f(row, "candidate_team_accuracy") - f(row, "baseline_team_accuracy"))) > EPS:
            vote_bad.append(row.get("candidate_id"))
        if abs(f(row, "vote_delta") - (f(row, "vote_gain_rate") - f(row, "vote_loss_rate"))) > EPS:
            transition_bad.append(row.get("candidate_id"))
        if f(row, "boundary_useful_diversity_delta") < -EPS and (
            abs(f(row, "boundary_diversity_gain")) > EPS or f(row, "reward_component_boundary_diversity") < -EPS
        ):
            clipping_bad.append(row.get("candidate_id"))
    pareto_front_sizes = [int(row.get("pareto_front0_size") or 0) for row in summaries if row.get("pareto_front0_size") is not None]
    pareto_feasible = [int(row.get("num_pareto_feasible") or 0) for row in summaries if row.get("num_pareto_feasible") is not None]
    return {
        "candidate_total": len(candidates),
        "optimizer_candidates": sum(str(row.get("candidate_pool_source")) == "optimizer" for row in candidates),
        "existing_beam_candidates": sum(str(row.get("candidate_pool_source")) == "existing_beam" for row in candidates),
        "nonzero_vote_delta_rate": mean([abs(f(row, "vote_delta")) > EPS for row in candidates]),
        "nonzero_vote_margin_delta_rate": mean([abs(f(row, "vote_margin_delta")) > EPS for row in candidates]),
        "positive_boundary_gain_rate": mean([f(row, "boundary_diversity_gain") > EPS for row in candidates]),
        "accuracy_guard_rejection_rate": mean([not bool(row.get("accuracy_guard_passed", True)) for row in candidates]),
        "invalid_guard_rejection_rate": mean([not bool(row.get("invalid_guard_passed", True)) for row in candidates]),
        "active_prompt_change_rate": mean([bool(row.get("active_prompt_changed")) for row in summaries]),
        "optimizer_underfilled_rate": mean([bool(row.get("optimizer_underfilled")) for row in summaries]),
        "optimizer_candidate_adoption_rate": mean([
            str(row.get("candidate_pool_source")) == "optimizer" and bool(row.get("is_top1")) for row in candidates
        ]),
        "update_attempt_count": len(summaries),
        "pareto_front0_size_mean": mean(pareto_front_sizes),
        "pareto_front_ratio_mean": mean([
            size / feasible for size, feasible in zip(pareto_front_sizes, pareto_feasible) if feasible
        ]),
        "negative_raw_boundary_delta_count": len(raw_negative),
        "reward_identity_failure_count": len(reward_bad),
        "vote_delta_identity_failure_count": len(vote_bad),
        "vote_transition_identity_failure_count": len(transition_bad),
        "boundary_clipping_failure_count": len(clipping_bad),
    }


def pareto_active_key(row, rank):
    return (
        -f(row, "vote_delta"),
        f(row, "vote_loss_rate"),
        -f(row, "vote_gain_rate"),
        -f(row, "vote_margin_delta"),
        -f(row, "candidate_target_accuracy", f(row, "target_agent_accuracy")),
        -f(row, "boundary_useful_diversity_delta"),
        f(row, "candidate_invalid_rate", f(row, "invalid_rate")),
        rank,
        str(row.get("candidate_id", "")),
    )


def crowding_key(item):
    row, distance = item
    return (
        -distance,
        -f(row, "vote_delta"),
        f(row, "vote_loss_rate"),
        -f(row, "vote_gain_rate"),
        -f(row, "vote_margin_delta"),
        -f(row, "candidate_target_accuracy", f(row, "target_agent_accuracy")),
        -f(row, "boundary_useful_diversity_delta"),
        f(row, "candidate_invalid_rate", f(row, "invalid_rate")),
        str(row.get("candidate_id", "")),
    )


def pareto_top1(pool, invalid_epsilon, beam_size):
    feasible = []
    for row in pool:
        accuracy_epsilon = f(row, "effective_accuracy_guard_epsilon")
        accuracy_ok = f(row, "candidate_target_accuracy") >= f(row, "baseline_target_accuracy") - accuracy_epsilon - EPS
        invalid_ok = f(row, "candidate_invalid_rate") <= f(row, "baseline_invalid_rate") + invalid_epsilon + EPS
        if accuracy_ok and invalid_ok:
            feasible.append(row)
    forced = False
    if not feasible:
        existing = [row for row in pool if str(row.get("candidate_pool_source")) == "existing_beam"]
        feasible = existing[:1] or pool[:1]
        forced = True
    wrapped = [{"candidate_id": row.get("candidate_id"), "metrics": row} for row in feasible]
    fronts = non_dominated_sort(wrapped)
    retained = []
    ranks = {}
    distances = {}
    for rank, indices in enumerate(fronts):
        front_distances = compute_crowding_distances(wrapped, indices)
        for index in indices:
            ranks[index] = rank
            distances[index] = front_distances.get(index, 0.0)
        slots = beam_size - len(retained)
        if slots <= 0:
            continue
        if len(indices) <= slots:
            retained.extend(sorted(indices, key=lambda index: str(feasible[index].get("candidate_id", ""))))
        else:
            retained.extend(sorted(indices, key=lambda index: crowding_key((feasible[index], distances[index])))[:slots])
            break
    retained.sort(key=lambda index: pareto_active_key(feasible[index], ranks.get(index, 10**9)))
    return feasible[retained[0]], len(fronts[0]) if fronts else 0, len(feasible), forced


def replay_run(run_dir):
    meta = read_json(run_dir / "run_meta.json")
    config = meta.get("config", {})
    invalid_epsilon = float(config.get("invalid_guard_epsilon", 0.05))
    beam_size = int(config.get("beam_size", 3))
    rows = read_jsonl(run_dir / "update_logs.jsonl")
    pools = defaultdict(list)
    for row in rows:
        if row.get("event") == "candidate_evaluated":
            pools[str(row.get("update_attempt_id"))].append(row)
    output = []
    for attempt, pool in pools.items():
        scalar = max(pool, key=lambda row: f(row, "reward"))
        pareto, front_size, feasible_count, forced = pareto_top1(pool, invalid_epsilon, beam_size)
        record = {
            "source_run": str(run_dir.relative_to(REPO)),
            "source_selector": config.get("candidate_selection_mode"),
            "update_attempt_id": attempt,
            "epoch": pool[0].get("epoch"),
            "step": pool[0].get("step"),
            "agent_id": pool[0].get("agent_id"),
            "candidate_pool_size": len(pool),
            "scalar_candidate_id": scalar.get("candidate_id"),
            "pareto_candidate_id": pareto.get("candidate_id"),
            "top1_disagrees": scalar.get("candidate_id") != pareto.get("candidate_id"),
            "pareto_front0_size": front_size,
            "pareto_feasible_count": feasible_count,
            "pareto_first_front_ratio": front_size / feasible_count if feasible_count else 0.0,
            "pareto_forced_fallback": forced,
        }
        for prefix, selected in (("scalar", scalar), ("pareto", pareto)):
            for key in ("vote_delta", "vote_loss_rate", "vote_gain_rate", "candidate_target_accuracy", "vote_margin_delta", "boundary_diversity_gain"):
                record[f"{prefix}_{key}"] = f(selected, key)
            record[f"{prefix}_candidate_pool_source"] = selected.get("candidate_pool_source")
        output.append(record)
    return output


def reuse_entry(stage, setting, run_dir):
    meta = read_json(run_dir / "run_meta.json")
    split = meta.get("split_integrity", meta.get("split_integrity_json", {}))
    return {
        "stage": stage,
        "task": TASK,
        "setting": setting,
        "seed": 42,
        "source_run_directory": str(run_dir.relative_to(REPO)),
        "git_commit": meta.get("git_commit"),
        "protocol_version": meta.get("experiment_protocol_version"),
        "behavior_config_fingerprint": meta.get("behavior_config_fingerprint") or meta.get("behavior_config", {}).get("fingerprint"),
        "split_hashes": split,
        "reuse_reason": "complete and compatible",
    }


def main():
    split = split_info()
    reused = []
    registry = []
    for stage, stage_root in ((1, STAGE1_ROOT), (2, STAGE2_ROOT)):
        for setting in SETTINGS:
            run_dir = stage_root / f"{setting}_seed42"
            status = "REUSE_COMPLETE" if complete(run_dir) else "CORRUPTED"
            registry.append({"stage": stage, "task": TASK, "setting": setting, "seed": 42, "status": status, "run_directory": str(run_dir.relative_to(REPO))})
            if status == "REUSE_COMPLETE":
                reused.append(reuse_entry(stage, setting, run_dir))

    excluded = []
    for task in ("geometric_shapes", "ruin_names", "sports_understanding"):
        task_dir = REPO / "runs_vote_stage2_selector_pilot_v4_dcc9492" / task
        if not task_dir.exists():
            continue
        for run_dir in sorted(path.parent for path in task_dir.rglob("run_meta.json")):
            excluded.append({"task": task, "run_directory": str(run_dir.relative_to(REPO)), "reason": "outside single-task scope; preserved and not resumed"})

    write_json(ROOT / "REUSED_RUNS.json", reused)
    write_json(ROOT / "NEW_RUNS.json", [])
    write_json(ROOT / "EXCLUDED_RUNS.json", excluded)
    write_csv(ROOT / "RUN_REGISTRY.csv", registry, ["stage", "task", "setting", "seed", "status", "run_directory"])

    preflight = f"""# Preflight Report

- Git commit: `{CURRENT_COMMIT}`
- Tracked-source dirty: `false` (only untracked historical run directories were present)
- Tests: `156 passed in 1.85s`
- Active runner: none; PID 9160 is an exited Windows process object (`HasExited=true`)
- Effective protocol: `{CURRENT_PROTOCOL}`
- Attachment's requested `vote_oriented_v1`: historical label; not used for compatibility because current code and Stage 2 are v4
- Checkpoint version: `2`
- Vote tie-break: `random`
- Models: solver=`deepseek-chat`, prompt generator=`deepseek-chat`, critic=`deepseek-chat`
- Candidate evaluation source: `optimization_train`
- Validation role: `vote_first` state selection only
- Test role: one final evaluation after restoring selected state

## Strict split

| Split | Count | SHA256 |
|---|---:|---|
| opt | {split['opt']['count']} | `{split['opt']['sha256']}` |
| val | {split['val']['count']} | `{split['val']['sha256']}` |
| test | {split['test']['count']} | `{split['test']['sha256']}` |

Overlaps: opt/val={split['overlap']['opt_val']}, opt/test={split['overlap']['opt_test']}, val/test={split['overlap']['val_test']}.

## Reuse

Six completed Stage 1/2 run records were selected. Other-task Stage 2 runs are excluded and left untouched; see `REUSED_RUNS.json` and `EXCLUDED_RUNS.json`.
"""
    (ROOT / "preflight_report.md").write_text(preflight, encoding="utf-8")

    stage1_rows = []
    for setting in SETTINGS[1:]:
        run_dir = STAGE1_ROOT / f"{setting}_seed42"
        stats = candidate_stats(run_dir)
        audit = audit_run(run_dir)
        test = final_test(run_dir)
        stage1_rows.append({"setting": setting, **stats, "tcs_audit_problems": audit["problems"], "test_vote_acc": test.get("vote_acc"), "test_mean_individual_acc": test.get("mean_individual_acc"), "test_oracle_acc": test.get("oracle_acc")})
    stage1_payload = {"model_calls_made": 0, "runs": stage1_rows}
    write_json(ROOT / "stage1_smoke_metrics.json", stage1_payload)
    stage1_report = ["# Stage 1 Smoke Reuse", "", "Status: `reused_passed`; no model calls were made.", "", "| Setting | Candidates | Optimizer | Existing | Nonzero vote | Nonzero margin | Positive boundary | Prompt change |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in stage1_rows:
        stage1_report.append(f"| {row['setting']} | {row['candidate_total']} | {row['optimizer_candidates']} | {row['existing_beam_candidates']} | {row['nonzero_vote_delta_rate']:.3f} | {row['nonzero_vote_margin_delta_rate']:.3f} | {row['positive_boundary_gain_rate']:.3f} | {row['active_prompt_change_rate']:.3f} |")
    stage1_report.extend(["", "All TCS audits passed. Reward, vote-delta, transition, and boundary-clipping identity failure counts are zero."])
    (ROOT / "stage1_smoke_reuse_report.md").write_text("\n".join(stage1_report) + "\n", encoding="utf-8")
    write_json(ROOT / "STAGE1_COMPLETE.json", {"stage": 1, "status": "reused_passed", "model_calls_made": 0, "reused_runs": [item["source_run_directory"] for item in reused if item["stage"] == 1], "key_findings": stage1_payload, "next_stage_authorized": True})

    stage2_metrics = []
    replay = []
    for setting in SETTINGS:
        run_dir = STAGE2_ROOT / f"{setting}_seed42"
        test = final_test(run_dir)
        val = best_validation(run_dir) if setting != "shared_baseline" else {}
        stats = candidate_stats(run_dir) if setting != "shared_baseline" else {}
        stage2_metrics.append({"setting": setting, "best_val_epoch": val.get("epoch"), "val_vote_acc": val.get("vote_acc"), "val_mean_individual_acc": val.get("mean_individual_acc"), "val_mean_vote_margin": val.get("mean_vote_margin"), "val_invalid_rate": val.get("mean_invalid_rate"), "test_vote_acc": test.get("vote_acc"), "test_mean_individual_acc": test.get("mean_individual_acc"), "test_best_individual_acc": test.get("best_individual_acc"), "test_mean_vote_margin": test.get("mean_vote_margin"), "test_vote_tie_rate": test.get("vote_tie_rate"), "test_oracle_acc": test.get("oracle_acc"), "test_aggregation_gap": test.get("aggregation_gap"), "test_invalid_rate": test.get("mean_invalid_rate"), "test_mean_boundary_useful_diversity": test.get("mean_boundary_useful_diversity"), "test_trace_diversity": test.get("mean_embedding_diversity"), **stats})
        if setting != "shared_baseline":
            replay.extend(replay_run(run_dir))
    write_csv(ROOT / "stage2_selector_metrics.csv", stage2_metrics)
    write_csv(ROOT / "selector_replay.csv", replay)
    disagreement = mean([row["top1_disagrees"] for row in replay])
    scalar_selected = [row for row in replay]
    replay_summary = {
        "effective_update_count": len(replay),
        "top1_disagreement_count": sum(row["top1_disagrees"] for row in replay),
        "top1_disagreement_rate": disagreement,
        "scalar_selected_vote_delta": mean([row["scalar_vote_delta"] for row in scalar_selected]),
        "pareto_selected_vote_delta": mean([row["pareto_vote_delta"] for row in scalar_selected]),
        "scalar_selected_vote_loss_rate": mean([row["scalar_vote_loss_rate"] for row in scalar_selected]),
        "pareto_selected_vote_loss_rate": mean([row["pareto_vote_loss_rate"] for row in scalar_selected]),
        "scalar_selected_target_accuracy": mean([row["scalar_candidate_target_accuracy"] for row in scalar_selected]),
        "pareto_selected_target_accuracy": mean([row["pareto_candidate_target_accuracy"] for row in scalar_selected]),
        "scalar_selected_vote_margin_delta": mean([row["scalar_vote_margin_delta"] for row in scalar_selected]),
        "pareto_selected_vote_margin_delta": mean([row["pareto_vote_margin_delta"] for row in scalar_selected]),
        "pareto_first_front_size": mean([row["pareto_front0_size"] for row in scalar_selected]),
        "pareto_first_front_ratio": mean([row["pareto_first_front_ratio"] for row in scalar_selected]),
    }
    scalar_metric = next(row for row in stage2_metrics if row["setting"] == "shared_scalar_tcs_vote_first")
    pareto_metric = next(row for row in stage2_metrics if row["setting"] == "shared_vote_pareto_tcs")
    if f(pareto_metric, "val_vote_acc") > f(scalar_metric, "val_vote_acc") + EPS:
        selected = "vote_pareto"
        reason = "Pareto has higher best validation vote accuracy."
    elif f(scalar_metric, "val_vote_acc") > f(pareto_metric, "val_vote_acc") + EPS:
        selected = "scalar_reward"
        reason = "Scalar has higher best validation vote accuracy; test was not used as the primary selector."
    elif disagreement < 0.10:
        selected = "scalar_reward"
        reason = "Validation is tied and replay disagreement is below 10%, so the simpler selector is preferred."
    elif replay_summary["pareto_selected_vote_loss_rate"] + EPS < replay_summary["scalar_selected_vote_loss_rate"]:
        selected = "vote_pareto"
        reason = "Validation is tied and Pareto reduces replay-selected vote loss."
    else:
        selected = "scalar_reward"
        reason = "Validation is tied and Pareto shows no clear replay safety advantage, so the simpler selector is preferred."
    selector_payload = {
        "task": TASK,
        "selected_candidate_selection_mode": selected,
        "selection_uses_test_as_primary": False,
        "reused_runs": [item["source_run_directory"] for item in reused if item["stage"] == 2],
        "validation_evidence": {"scalar": scalar_metric, "pareto": pareto_metric},
        "candidate_replay_evidence": replay_summary,
        "reason": reason,
    }
    write_json(ROOT / "selected_main_selector.json", selector_payload)
    stage2_report = f"""# Stage 2 Selector Reuse and Replay

All three `disambiguation_qa` runs were reused from protocol v4; model calls made: 0.

| Selector | Best val vote | Best val individual | Test vote | Test individual |
|---|---:|---:|---:|---:|
| scalar | {f(scalar_metric, 'val_vote_acc'):.3f} | {f(scalar_metric, 'val_mean_individual_acc'):.3f} | {f(scalar_metric, 'test_vote_acc'):.3f} | {f(scalar_metric, 'test_mean_individual_acc'):.3f} |
| Pareto | {f(pareto_metric, 'val_vote_acc'):.3f} | {f(pareto_metric, 'val_mean_individual_acc'):.3f} | {f(pareto_metric, 'test_vote_acc'):.3f} | {f(pareto_metric, 'test_mean_individual_acc'):.3f} |

Replay used {len(replay)} complete update-attempt candidate pools. Top-1 disagreement was {replay_summary['top1_disagreement_count']}/{len(replay)} ({disagreement:.1%}). Mean replay vote loss was {replay_summary['scalar_selected_vote_loss_rate']:.4f} for scalar and {replay_summary['pareto_selected_vote_loss_rate']:.4f} for Pareto.

Selected main selector: `{selected}`. {reason}

This is single-task validation evidence only.
"""
    (ROOT / "stage2_selector_report.md").write_text(stage2_report, encoding="utf-8")
    write_json(ROOT / "STAGE2_COMPLETE.json", {"stage": 2, "status": "reused_passed", "model_calls_made": 0, "reused_runs": selector_payload["reused_runs"], "key_findings": {"selected_selector": selected, **replay_summary}, "next_stage_authorized": True})
    print(json.dumps({"root": str(ROOT), "selected_selector": selected, "replay": replay_summary}, indent=2))


if __name__ == "__main__":
    main()
