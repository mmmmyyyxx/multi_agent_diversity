import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.utils import infer_strategy_family_major

from prove_experiment_utils import (
    bootstrap_mean_ci,
    find_prediction_file,
    infer_probe_kind,
    read_json,
    read_jsonl,
    safe_float,
    safe_mean,
    spearman_corr,
    wilcoxon_signed_rank,
)


def _safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs]
    return float(statistics.mean(vals)) if vals else 0.0


def _load_probe_targets(run_dir: Path) -> Dict[int, List[str]]:
    probe = read_json(run_dir / "probe_prompts.json")
    targets: Dict[int, List[str]] = {}
    if not isinstance(probe, dict):
        return targets
    for agent in probe.get("agents", []):
        if not isinstance(agent, dict):
            continue
        aid = int(agent.get("agent_id", len(targets)))
        target = agent.get("target_family", [])
        if isinstance(target, str):
            target = [target]
        targets[aid] = [str(x) for x in target] if isinstance(target, list) else []
    return targets


def _target_hit(primary: str, secondary: str, targets: List[str]) -> Tuple[int, int]:
    if not targets:
        return 0, 0
    target_set = {str(x) for x in targets}
    exact = int(primary in target_set or secondary in target_set)
    target_majors = {infer_strategy_family_major(x) for x in target_set}
    major = infer_strategy_family_major(primary)
    same_major = int(bool(target_majors) and major in target_majors)
    return exact, max(exact, same_major)


def _prediction_metrics(preds: List[Dict[str, Any]], targets: Dict[int, List[str]]) -> Dict[str, Any]:
    if not preds:
        return {
            "eval_size": 0,
            "mean_family_diversity": 0.0,
            "mean_family_homogeneity_rate": 0.0,
            "all_same_pair_rate": 0.0,
            "disagreement_rate": 0.0,
            "vote_acc": 0.0,
            "target_exact_hit_rate": None,
            "target_same_major_hit_rate": None,
        }
    exact_hits = []
    major_hits = []
    for rec in preds:
        primary = rec.get("primary_family_labels", [])
        secondary = rec.get("secondary_family_labels", primary)
        if not isinstance(primary, list):
            primary = []
        if not isinstance(secondary, list):
            secondary = primary
        for aid, target in targets.items():
            p = str(primary[aid]) if aid < len(primary) else ""
            s = str(secondary[aid]) if aid < len(secondary) else p
            exact, same_major = _target_hit(p, s, target)
            exact_hits.append(exact)
            major_hits.append(same_major)
    disagreements = []
    for rec in preds:
        answers = rec.get("answers", [])
        if isinstance(answers, list) and answers:
            disagreements.append(int(len(set(str(x) for x in answers)) > 1))
    return {
        "eval_size": len(preds),
        "mean_family_diversity": _safe_mean(float(r.get("team_family_diversity", 0.0) or 0.0) for r in preds),
        "mean_family_homogeneity_rate": _safe_mean(float(r.get("team_family_homogeneity_rate", 0.0) or 0.0) for r in preds),
        "mean_major_family_diversity": _safe_mean(float(r.get("team_major_family_diversity", 0.0) or 0.0) for r in preds),
        "mean_intra_family_diversity": _safe_mean(float(r.get("team_intra_family_diversity", 0.0) or 0.0) for r in preds),
        "mean_family_confidence": _safe_mean(float(r.get("mean_family_confidence", 0.0) or 0.0) for r in preds),
        "low_confidence_share": _safe_mean(float(r.get("low_confidence_share", 0.0) or 0.0) for r in preds),
        "all_same_pair_rate": _safe_mean(int(bool(r.get("all_same_pair", False))) for r in preds),
        "disagreement_rate": _safe_mean(disagreements),
        "vote_acc": _safe_mean(float(r.get("vote_correct", 0.0) or 0.0) for r in preds),
        "target_exact_hit_rate": _safe_mean(exact_hits) if exact_hits else None,
        "target_same_major_hit_rate": _safe_mean(major_hits) if major_hits else None,
    }


def _latest_history_metrics(run_dir: Path) -> Dict[str, Any]:
    hist = read_json(run_dir / "history.json")
    if not isinstance(hist, list) or not hist:
        return {}
    latest = hist[-1] if isinstance(hist[-1], dict) else {}
    out = {}
    for split in ["train", "val", "test"]:
        block = latest.get(split, {}) if isinstance(latest.get(split, {}), dict) else {}
        for k, v in block.items():
            out[f"latest_{split}_{k}"] = v
    out["history_last_epoch"] = latest.get("epoch", "")
    out["selected_epoch"] = latest.get("selected_epoch", "")
    out["early_stopped"] = latest.get("early_stopped", "")
    return out


def _update_metrics(run_dir: Path) -> Dict[str, Any]:
    rows = read_jsonl(run_dir / "train_step_logs.jsonl")
    if not rows:
        return {
            "train_steps": 0,
            "update_applied_rate": None,
            "mean_updated_agents": None,
            "candidate_family_shift_rate": None,
            "candidate_rho_reduction": None,
            "candidate_invalid_delta": None,
            "candidate_summary_embedding_shift": None,
            "optimization_signal_rate": None,
        }
    applied = []
    updated_counts = []
    family_shift = []
    rho_reduction = []
    invalid_delta = []
    summary_shift = []
    signal_flags = []
    for row in rows:
        upd = row.get("update", {}) if isinstance(row.get("update", {}), dict) else {}
        updated = upd.get("updated_agent_ids", [])
        if not isinstance(updated, list):
            updated = []
        applied.append(int(len(updated) > 0))
        updated_counts.append(len(updated))

        records = []
        legacy = upd.get("candidate_behavior_records", [])
        if isinstance(legacy, list):
            records.extend([x for x in legacy if isinstance(x, dict)])
        top = row.get("candidate_behavior_diagnostics", {})
        if isinstance(top, dict) and top:
            records.append(top)
        nested = upd.get("candidate_behavior_diagnostics", {})
        if isinstance(nested, dict) and nested:
            records.append(nested)

        for record in records:
            if not isinstance(record, dict):
                continue
            if record.get("family_shift_rate") is not None:
                family_shift.append(float(record.get("family_shift_rate", 0.0) or 0.0))
            if record.get("rho_reduction") is not None:
                rho_reduction.append(float(record.get("rho_reduction", 0.0) or 0.0))
            if record.get("invalid_delta") is not None:
                invalid_delta.append(float(record.get("invalid_delta", 0.0) or 0.0))
            if record.get("summary_embedding_shift") is not None:
                summary_shift.append(float(record.get("summary_embedding_shift", 0.0) or 0.0))
            shift_ok = float(record.get("family_shift_rate", 0.0) or 0.0) > 0.0
            rho_ok = float(record.get("rho_reduction", 0.0) or 0.0) > 0.0
            invalid_ok = float(record.get("invalid_delta", 0.0) or 0.0) <= 0.1
            signal_flags.append(int((shift_ok or rho_ok) and invalid_ok))
    return {
        "train_steps": len(rows),
        "update_applied_rate": _safe_mean(applied),
        "mean_updated_agents": _safe_mean(updated_counts),
        "candidate_family_shift_rate": _safe_mean(family_shift) if family_shift else None,
        "candidate_rho_reduction": _safe_mean(rho_reduction) if rho_reduction else None,
        "candidate_invalid_delta": _safe_mean(invalid_delta) if invalid_delta else None,
        "candidate_summary_embedding_shift": _safe_mean(summary_shift) if summary_shift else None,
        "optimization_signal_rate": _safe_mean(signal_flags) if signal_flags else None,
    }


def _major_labels(rec: Dict[str, Any]) -> List[str]:
    labels = rec.get("primary_family_labels", [])
    if not isinstance(labels, list):
        return []
    return [infer_strategy_family_major(str(x)) for x in labels]


def _pairwise_major_disagreement(preds: List[Dict[str, Any]]) -> float:
    vals = []
    for rec in preds:
        majors = _major_labels(rec)
        n = len(majors)
        if n < 2:
            continue
        total = 0
        diff = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                diff += int(majors[i] != majors[j])
        if total:
            vals.append(diff / total)
    return _safe_mean(vals)


def _question_values(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    pred_file = find_prediction_file(run_dir)
    preds = read_jsonl(pred_file) if pred_file else []
    out: Dict[str, Dict[str, Any]] = {}
    for rec in preds:
        qh = str(rec.get("question_hash", ""))
        if not qh:
            continue
        out[qh] = {
            "team_family_diversity": safe_float(rec.get("team_family_diversity")),
            "team_family_homogeneity_rate": safe_float(rec.get("team_family_homogeneity_rate")),
            "team_major_family_diversity": safe_float(rec.get("team_major_family_diversity")),
            "team_intra_family_diversity": safe_float(rec.get("team_intra_family_diversity")),
            "mean_family_confidence": safe_float(rec.get("mean_family_confidence")),
            "major_disagreement": _pairwise_major_disagreement([rec]),
        }
    return out


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    meta = read_json(run_dir / "run_meta.json") or {}
    cfg = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    pred_file = find_prediction_file(run_dir)
    preds = read_jsonl(pred_file) if pred_file else []
    targets = _load_probe_targets(run_dir)
    probe = meta.get("probe", {}) if isinstance(meta.get("probe", {}), dict) else {}
    probe_name = probe.get("probe_name", "")
    out = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "probe_name": probe_name,
        "probe_kind": infer_probe_kind(probe_name, run_dir.name),
        "has_probe_targets": int(bool(targets)),
        "model": cfg.get("model", ""),
        "critic_model": cfg.get("critic_model", ""),
        "seed": cfg.get("seed", ""),
        "init_mode": meta.get("init_mode", cfg.get("init_mode", "")),
        "baseline_only": int(bool(cfg.get("baseline_only", False))),
        "lambda_diversity": cfg.get("lambda_diversity", ""),
        "lambda_homogeneity": cfg.get("lambda_homogeneity", ""),
        "same_major_family_weight": cfg.get("same_major_family_weight", ""),
        "macro_diversity_weight": cfg.get("macro_diversity_weight", ""),
        "prediction_file": str(pred_file) if pred_file else "",
        "major_disagreement_rate": _pairwise_major_disagreement(preds),
    }
    out.update(_prediction_metrics(preds, targets))
    out.update(_latest_history_metrics(run_dir))
    out.update(_update_metrics(run_dir))
    return out


def _paired_probe_stats(rows: List[Dict[str, Any]], metric: str, iterations: int, seed: int) -> Dict[str, Any]:
    same_rows = [r for r in rows if r.get("probe_kind") == "same"]
    mixed_rows = [r for r in rows if r.get("probe_kind") == "mixed"]
    same_by_model = defaultdict(list)
    mixed_by_model = defaultdict(list)
    for row in same_rows:
        same_by_model[str(row.get("model", ""))].append(row)
    for row in mixed_rows:
        mixed_by_model[str(row.get("model", ""))].append(row)

    deltas: List[float] = []
    matched_models: List[str] = []
    for model in sorted(set(same_by_model) & set(mixed_by_model)):
        same_run = Path(str(same_by_model[model][0].get("run_dir", "")))
        mixed_run = Path(str(mixed_by_model[model][0].get("run_dir", "")))
        same_vals = _question_values(same_run)
        mixed_vals = _question_values(mixed_run)
        for qh in sorted(set(same_vals) & set(mixed_vals)):
            deltas.append(safe_float(mixed_vals[qh].get(metric)) - safe_float(same_vals[qh].get(metric)))
        matched_models.append(model)

    ci = bootstrap_mean_ci(deltas, iterations=iterations, seed=seed)
    wilcox = wilcoxon_signed_rank(deltas)
    return {
        "metric": metric,
        "matched_models": ",".join(matched_models),
        "paired_question_count": len(deltas),
        "mean_delta": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "wilcoxon_z": wilcox["z"],
        "wilcoxon_p_approx": wilcox["p_approx"],
    }


def _model_identity_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    same_rows = [r for r in rows if r.get("probe_kind") == "same"]
    mixed_rows = [r for r in rows if r.get("probe_kind") == "mixed"]
    strategy_effects = []
    for model in sorted({str(r.get("model", "")) for r in rows}):
        same = [r for r in same_rows if str(r.get("model", "")) == model]
        mixed = [r for r in mixed_rows if str(r.get("model", "")) == model]
        if same and mixed:
            strategy_effects.append(safe_float(mixed[0].get("major_disagreement_rate")) - safe_float(same[0].get("major_disagreement_rate")))

    model_effects = []
    for group in [same_rows, mixed_rows]:
        if len(group) < 2:
            continue
        vals = [safe_float(r.get("major_disagreement_rate")) for r in group]
        if vals:
            model_effects.append(max(vals) - min(vals))

    return {
        "strategy_effect_major_disagreement": safe_mean(strategy_effects),
        "model_identity_effect_major_disagreement": safe_mean(model_effects),
        "strategy_gt_model_identity": int(bool(strategy_effects) and safe_mean(strategy_effects) > safe_mean(model_effects)),
    }


def _write_stats_json(rows: List[Dict[str, Any]], path: Path, iterations: int, seed: int):
    stats = {
        "paired_mixed_minus_same_family_diversity": _paired_probe_stats(rows, "team_family_diversity", iterations, seed),
        "paired_mixed_minus_same_homogeneity": _paired_probe_stats(rows, "team_family_homogeneity_rate", iterations, seed),
        "paired_mixed_minus_same_major_diversity": _paired_probe_stats(rows, "team_major_family_diversity", iterations, seed),
        "model_identity_check": _model_identity_stats(rows),
        "family_vs_major_disagreement_spearman": spearman_corr(
            [r.get("mean_family_diversity", 0.0) for r in rows],
            [r.get("major_disagreement_rate", 0.0) for r in rows],
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _write_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ["run_name"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if v is None:
        return ""
    return str(v)


def _write_md(rows: List[Dict[str, Any]], path: Path):
    columns = [
        "run_name",
        "probe_name",
        "model",
        "eval_size",
        "mean_family_diversity",
        "mean_family_homogeneity_rate",
        "mean_major_family_diversity",
        "low_confidence_share",
        "all_same_pair_rate",
        "target_exact_hit_rate",
        "target_same_major_hit_rate",
        "vote_acc",
        "lambda_diversity",
        "lambda_homogeneity",
        "same_major_family_weight",
        "update_applied_rate",
        "candidate_family_shift_rate",
        "candidate_invalid_delta",
        "optimization_signal_rate",
    ]
    lines = ["# Prove Experiment Summary", "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in columns) + " |")

    by_probe = defaultdict(list)
    for row in rows:
        by_probe[str(row.get("probe_name", "") or row.get("run_name", ""))].append(row)
    lines.extend(["", "## 自动对照提示", ""])
    names = {row.get("probe_name") or row.get("run_name"): row for row in rows}
    same = [r for r in rows if "same" in str(r.get("probe_name", r.get("run_name", ""))).lower()]
    mixed = [r for r in rows if "mixed" in str(r.get("probe_name", r.get("run_name", ""))).lower() or "bank" in str(r.get("run_name", "")).lower()]
    if same and mixed:
        same_div = _safe_mean(r.get("mean_family_diversity", 0.0) for r in same)
        mixed_div = _safe_mean(r.get("mean_family_diversity", 0.0) for r in mixed)
        lines.append(f"- mixed/same diversity delta: {mixed_div - same_div:.4f}；应为正，才支持显式策略干预有效。")
    training = [r for r in rows if not int(r.get("baseline_only", 0) or 0)]
    if training:
        no_div = [r for r in training if float(r.get("lambda_diversity", 0.0) or 0.0) == 0.0]
        div = [r for r in training if float(r.get("lambda_diversity", 0.0) or 0.0) > 0.0]
        if no_div and div:
            no_div_val = _safe_mean(float(r.get("latest_val_mean_family_diversity", r.get("mean_family_diversity", 0.0)) or 0.0) for r in no_div)
            best_div = max(float(r.get("latest_val_mean_family_diversity", r.get("mean_family_diversity", 0.0)) or 0.0) for r in div)
            lines.append(f"- best nonzero reward val diversity minus no_div: {best_div - no_div_val:.4f}；应为正，才支持 reward 可优化。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_md_clean(rows: List[Dict[str, Any]], path: Path):
    columns = [
        "run_name",
        "probe_name",
        "model",
        "eval_size",
        "mean_family_diversity",
        "mean_family_homogeneity_rate",
        "mean_major_family_diversity",
        "mean_intra_family_diversity",
        "low_confidence_share",
        "all_same_pair_rate",
        "target_exact_hit_rate",
        "target_same_major_hit_rate",
        "vote_acc",
    ]
    lines = [
        "# P3 证明实验汇总",
        "",
        "## 指标中文含义",
        "",
        "| 指标 | 中文含义 |",
        "|---|---|",
        "| `mean_family_diversity` | 策略树 leaf 层面的平均团队多样性。越高表示五个 agent 被判到的细策略越分散。 |",
        "| `mean_family_homogeneity_rate` | 平均同质性。越高表示 agent 之间策略越相似。 |",
        "| `mean_major_family_diversity` | 主类层面的平均团队多样性。越高表示五个 agent 更常落到不同主策略类。 |",
        "| `mean_intra_family_diversity` | 同一主类内部的 leaf 多样性。 |",
        "| `low_confidence_share` | judge 对策略标签低置信的比例。越高表示标签更不稳定。 |",
        "| `all_same_pair_rate` | 五个 agent 策略对完全相同的题目比例。越低表示策略差异更明显。 |",
        "| `target_exact_hit_rate` | `primary` 或 `secondary` leaf 精确命中 prompt 目标 leaf 的比例。 |",
        "| `target_same_major_hit_rate` | `primary` 所属主类命中目标主类，或 leaf 精确命中的比例。 |",
        "| `vote_acc` | 五个 agent 多数投票答案准确率。 |",
        "",
        "## Run 级汇总",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in columns) + " |")

    same = [r for r in rows if str(r.get("probe_kind", "")).lower() == "same"]
    mixed = [r for r in rows if str(r.get("probe_kind", "")).lower() == "mixed"]
    if same and mixed:
        same_div = _safe_mean(r.get("mean_family_diversity", 0.0) for r in same)
        mixed_div = _safe_mean(r.get("mean_family_diversity", 0.0) for r in mixed)
        same_major = _safe_mean(r.get("mean_major_family_diversity", 0.0) for r in same)
        mixed_major = _safe_mean(r.get("mean_major_family_diversity", 0.0) for r in mixed)
        same_homo = _safe_mean(r.get("mean_family_homogeneity_rate", 0.0) for r in same)
        mixed_homo = _safe_mean(r.get("mean_family_homogeneity_rate", 0.0) for r in mixed)
        lines.extend(
            [
                "",
                "## Same vs Mixed 总体对照",
                "",
                "| 对比项 | same 平均 | mixed 平均 | mixed - same |",
                "|---|---|---|---|",
                f"| leaf 策略多样性 | {same_div:.4f} | {mixed_div:.4f} | {mixed_div - same_div:.4f} |",
                f"| 主类策略多样性 | {same_major:.4f} | {mixed_major:.4f} | {mixed_major - same_major:.4f} |",
                f"| 策略同质性 | {same_homo:.4f} | {mixed_homo:.4f} | {mixed_homo - same_homo:.4f} |",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_stats_md(path: Path, stats: Dict[str, Any]):
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines.extend(["", "## 统计检验", ""])
    for name, block in stats.items():
        if not isinstance(block, dict):
            continue
        lines.append(f"### {name}")
        if "mean_delta" in block:
            lines.append(
                f"- paired n={block.get('paired_question_count', 0)}, mean_delta={safe_float(block.get('mean_delta')):.4f}, "
                f"95% bootstrap CI=[{safe_float(block.get('ci_low')):.4f}, {safe_float(block.get('ci_high')):.4f}], "
                f"Wilcoxon p~{safe_float(block.get('wilcoxon_p_approx'), 1.0):.4f}"
            )
        elif "rho" in block:
            lines.append(f"- n={block.get('n', 0)}, Spearman rho={safe_float(block.get('rho')):.4f}")
        else:
            for k, v in block.items():
                lines.append(f"- {k}: {_fmt(v)}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_stats_md_clean(path: Path, stats: Dict[str, Any]):
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines.extend(["", "## 统计检验", ""])
    name_map = {
        "paired_mixed_minus_same_family_diversity": "paired: mixed - same 的 leaf 策略多样性",
        "paired_mixed_minus_same_homogeneity": "paired: mixed - same 的策略同质性",
        "paired_mixed_minus_same_major_diversity": "paired: mixed - same 的主类策略多样性",
        "model_identity_check": "模型身份效应检查",
        "family_vs_major_disagreement_spearman": "leaf 多样性与主类分歧 Spearman 相关",
    }
    for name, block in stats.items():
        if not isinstance(block, dict):
            continue
        lines.append(f"### {name_map.get(name, name)}")
        if "mean_delta" in block:
            lines.append(
                f"- 配对样本数={block.get('paired_question_count', 0)}, mean_delta={safe_float(block.get('mean_delta')):.4f}, "
                f"95% bootstrap CI=[{safe_float(block.get('ci_low')):.4f}, {safe_float(block.get('ci_high')):.4f}], "
                f"Wilcoxon 近似 p={safe_float(block.get('wilcoxon_p_approx'), 1.0):.4f}"
            )
        elif "rho" in block:
            lines.append(f"- n={block.get('n', 0)}, Spearman rho={safe_float(block.get('rho')):.4f}")
        else:
            for k, v in block.items():
                lines.append(f"- {k}: {_fmt(v)}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze prove experiment runs.")
    parser.add_argument("--runs_root", type=str, default="prove_experiments/runs")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_md", type=str, default="")
    parser.add_argument("--out_stats_json", type=str, default="")
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.runs_root)
    run_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "run_meta.json").exists()] if root.exists() else []
    rows = [analyze_run(p) for p in run_dirs]
    out_csv = Path(args.out_csv) if args.out_csv else root / "prove_summary.csv"
    out_md = Path(args.out_md) if args.out_md else root / "prove_summary.md"
    out_stats = Path(args.out_stats_json) if args.out_stats_json else root / "prove_stats.json"
    _write_csv(rows, out_csv)
    _write_md_clean(rows, out_md)
    stats = _write_stats_json(rows, out_stats, args.bootstrap_iterations, args.seed)
    _append_stats_md_clean(out_md, stats)
    print(f"Analyzed runs: {len(rows)}")
    print(f"CSV: {out_csv}")
    print(f"Markdown: {out_md}")
    print(f"Stats JSON: {out_stats}")


if __name__ == "__main__":
    main()
