#!/usr/bin/env python
"""Analyze P3 with question-level valid trace filtering.

The script does not call any model API. It scans existing P3 run outputs,
checks whether all five agent traces on a question pass basic quality rules,
and reports both all-question and valid-question metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.utils import infer_strategy_family_major, normalize_spaces

from prove_experiment_utils import bootstrap_mean_ci, find_prediction_file, read_json, read_jsonl, safe_float, wilcoxon_signed_rank


INVALID_THRESHOLD = 0.35
MIN_TRACE_CHARS = 80
REPEAT_THRESHOLD = 0.35


def mean(values: Iterable[Any]) -> float:
    vals = [safe_float(v) for v in values]
    return float(statistics.mean(vals)) if vals else 0.0


def trace_repeat_ratio(text: str) -> float:
    tokens = re.findall(r"\w+", normalize_spaces(text).lower())
    if len(tokens) < 12:
        return 0.0
    grams = [" ".join(tokens[i : i + 4]) for i in range(max(0, len(tokens) - 3))]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def invalid_trace_flags(trace: str, answer: str) -> dict[str, Any]:
    cleaned = normalize_spaces(trace)
    flags: dict[str, Any] = {
        "empty": False,
        "short": False,
        "no_final_answer": False,
        "repeat": False,
        "answer_missing": False,
        "penalty": 0.0,
        "bad": False,
    }
    if not cleaned:
        flags.update({"empty": True, "penalty": 1.0, "bad": True})
        return flags

    penalty = 0.0
    if len(cleaned) < MIN_TRACE_CHARS:
        flags["short"] = True
        penalty += 0.35
    if not re.search(r"FINAL_ANSWER\s*:", cleaned, flags=re.IGNORECASE):
        flags["no_final_answer"] = True
        penalty += 0.25
    repeat_ratio = trace_repeat_ratio(cleaned)
    if repeat_ratio > REPEAT_THRESHOLD:
        flags["repeat"] = True
        penalty += min(0.25, repeat_ratio)
    if str(answer or "").strip() == "":
        flags["answer_missing"] = True
        penalty += 0.25
    flags["penalty"] = float(min(1.0, penalty))
    flags["bad"] = bool(flags["penalty"] >= INVALID_THRESHOLD)
    return flags


def load_targets(run_dir: Path) -> dict[int, list[str]]:
    probe = read_json(run_dir / "probe_prompts.json")
    targets: dict[int, list[str]] = {}
    if not isinstance(probe, dict):
        return targets
    for item in probe.get("agents", []):
        if not isinstance(item, dict):
            continue
        aid = int(item.get("agent_id", len(targets)))
        target = item.get("target_family", [])
        if isinstance(target, str):
            target = [target]
        targets[aid] = [str(x) for x in target] if isinstance(target, list) else []
    return targets


def target_hits(rec: dict[str, Any], targets: dict[int, list[str]]) -> tuple[list[int], list[int]]:
    primary = rec.get("primary_family_labels", [])
    secondary = rec.get("secondary_family_labels", primary)
    if not isinstance(primary, list):
        primary = []
    if not isinstance(secondary, list):
        secondary = primary

    exact_hits: list[int] = []
    major_hits: list[int] = []
    for aid, target in targets.items():
        p = str(primary[aid]) if aid < len(primary) else ""
        s = str(secondary[aid]) if aid < len(secondary) else p
        target_set = set(target)
        exact = int(bool(target_set) and (p in target_set or s in target_set))
        target_majors = {infer_strategy_family_major(x) for x in target}
        same_major = int(
            exact
            or (bool(target_majors) and infer_strategy_family_major(p) in target_majors)
            or (bool(target_majors) and infer_strategy_family_major(s) in target_majors)
        )
        exact_hits.append(exact)
        major_hits.append(same_major)
    return exact_hits, major_hits


def run_kind(run_name: str) -> str:
    text = run_name.lower()
    if "mixed" in text:
        return "mixed"
    if "same" in text:
        return "same"
    return "other"


def model_name(run_dir: Path) -> str:
    meta = read_json(run_dir / "run_meta.json")
    cfg = meta.get("config", {}) if isinstance(meta, dict) and isinstance(meta.get("config", {}), dict) else {}
    return str(cfg.get("model", ""))


def summarize_subset(records: list[dict[str, Any]], targets: dict[int, list[str]]) -> dict[str, Any]:
    exact: list[int] = []
    major: list[int] = []
    for rec in records:
        e, m = target_hits(rec, targets)
        exact.extend(e)
        major.extend(m)
    return {
        "n": len(records),
        "family_div": mean([r.get("team_family_diversity") for r in records]),
        "homogeneity": mean([r.get("team_family_homogeneity_rate") for r in records]),
        "major_div": mean([r.get("team_major_family_diversity") for r in records]),
        "intra_family_div": mean([r.get("team_intra_family_diversity") for r in records]),
        "all_same_pair_rate": mean([int(bool(r.get("all_same_pair", False))) for r in records]),
        "vote_acc": mean([r.get("vote_correct") for r in records]),
        "target_exact": mean(exact) if exact else math.nan,
        "target_same_major": mean(major) if major else math.nan,
    }


def analyze_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_path = find_prediction_file(run_dir)
    preds = read_jsonl(pred_path) if pred_path else []
    trace_rows = read_jsonl(run_dir / "test_trace_history.jsonl")
    trace_by_hash = {str(r.get("question_hash", "")): r for r in trace_rows if str(r.get("question_hash", ""))}
    targets = load_targets(run_dir)

    question_rows: list[dict[str, Any]] = []
    valid_preds: list[dict[str, Any]] = []
    for rec in preds:
        qh = str(rec.get("question_hash", ""))
        answers = rec.get("answers", [])
        answers = answers if isinstance(answers, list) else []
        agents = trace_by_hash.get(qh, {}).get("agents", [])
        agents = agents if isinstance(agents, list) else []
        bad_ids: list[int] = []
        flags_by_agent: list[dict[str, Any]] = []
        n = min(len(answers), len(agents))
        for i in range(n):
            flags = invalid_trace_flags(str(agents[i].get("trace", "")), str(answers[i] if i < len(answers) else ""))
            flags_by_agent.append(flags)
            if flags["bad"]:
                bad_ids.append(i)
        missing_agents = max(0, 5 - n)
        if missing_agents:
            bad_ids.extend(range(n, n + missing_agents))
        is_valid = int(len(bad_ids) == 0 and n >= 5)
        if is_valid:
            valid_preds.append(rec)
        question_rows.append(
            {
                "run_name": run_dir.name,
                "model": model_name(run_dir),
                "probe_kind": run_kind(run_dir.name),
                "question_hash": qh,
                "valid_all_agents": is_valid,
                "bad_agent_count": len(bad_ids),
                "bad_agent_ids": "|".join(str(x) for x in bad_ids),
                "family_div": safe_float(rec.get("team_family_diversity")),
                "homogeneity": safe_float(rec.get("team_family_homogeneity_rate")),
                "major_div": safe_float(rec.get("team_major_family_diversity")),
                "vote_correct": safe_float(rec.get("vote_correct")),
            }
        )

    all_stats = summarize_subset(preds, targets)
    valid_stats = summarize_subset(valid_preds, targets)
    row = {
        "run_name": run_dir.name,
        "model": model_name(run_dir),
        "probe_kind": run_kind(run_dir.name),
        "questions": len(preds),
        "valid_questions": len(valid_preds),
        "valid_question_rate": len(valid_preds) / len(preds) if preds else 0.0,
        "bad_questions": len(preds) - len(valid_preds),
        "all_family_div": all_stats["family_div"],
        "all_homogeneity": all_stats["homogeneity"],
        "all_major_div": all_stats["major_div"],
        "all_vote_acc": all_stats["vote_acc"],
        "valid_family_div": valid_stats["family_div"],
        "valid_homogeneity": valid_stats["homogeneity"],
        "valid_major_div": valid_stats["major_div"],
        "valid_vote_acc": valid_stats["vote_acc"],
        "valid_target_exact": valid_stats["target_exact"],
        "valid_target_same_major": valid_stats["target_same_major"],
    }
    return row, question_rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()}) if rows else ["id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(v) for v in row) + " |")
    return lines


def paired_stats(run_rows: list[dict[str, Any]], question_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_run = {r["run_name"]: r for r in run_rows}
    models = sorted({str(r["model"]) for r in run_rows})
    out: dict[str, Any] = {}

    q_by_model_kind: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in question_rows:
        if int(row.get("valid_all_agents", 0)) != 1:
            continue
        q_by_model_kind[(str(row["model"]), str(row["probe_kind"]))][str(row["question_hash"])] = row

    for metric in ["family_div", "homogeneity", "major_div"]:
        deltas: list[float] = []
        for model in models:
            same = q_by_model_kind.get((model, "same"), {})
            mixed = q_by_model_kind.get((model, "mixed"), {})
            for qh in sorted(set(same) & set(mixed)):
                deltas.append(safe_float(mixed[qh].get(metric)) - safe_float(same[qh].get(metric)))
        ci = bootstrap_mean_ci(deltas)
        wx = wilcoxon_signed_rank(deltas)
        out[f"valid_mixed_minus_same_{metric}"] = {
            "paired_question_count": len(deltas),
            "mean_delta": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "wilcoxon_z": wx["z"],
            "wilcoxon_p_approx": wx["p_approx"],
        }

    same_major_pairs: list[float] = []
    mixed_major_pairs: list[float] = []
    for model in models:
        for qh, same_row in q_by_model_kind.get((model, "same"), {}).items():
            mixed_row = q_by_model_kind.get((model, "mixed"), {}).get(qh)
            if mixed_row:
                same_major_pairs.append(safe_float(same_row.get("major_div")))
                mixed_major_pairs.append(safe_float(mixed_row.get("major_div")))

    out["valid_run_rows"] = by_run
    return out


def write_md(run_rows: list[dict[str, Any]], stats: dict[str, Any], path: Path) -> None:
    lines = [
        "# P3 有效 Trace 质量分析",
        "",
        "本文件只使用已存在的 P3 输出进行离线统计，不调用 API。有效 trace 的定义是：同一道题的 5 个 agent trace 全部非空、长度足够、包含 `FINAL_ANSWER`、没有明显重复，并且答案字段非空。",
        "",
        "## 按 run 汇总",
        "",
    ]
    lines.extend(
        md_table(
            [
                "run",
                "model",
                "condition",
                "valid questions",
                "valid rate",
                "valid family_div",
                "valid major_div",
                "valid homogeneity",
                "valid vote_acc",
                "valid target exact",
                "valid target same-major",
            ],
            [
                [
                    r["run_name"],
                    r["model"],
                    r["probe_kind"],
                    f"{r['valid_questions']}/{r['questions']}",
                    r["valid_question_rate"],
                    r["valid_family_div"],
                    r["valid_major_div"],
                    r["valid_homogeneity"],
                    r["valid_vote_acc"],
                    r["valid_target_exact"],
                    r["valid_target_same_major"],
                ]
                for r in sorted(run_rows, key=lambda x: (x["probe_kind"], x["model"]))
            ],
        )
    )

    lines.extend(["", "## 有效 trace 口径下的 paired 检验", ""])
    stat_rows = []
    labels = {
        "family_div": "team_family_diversity",
        "homogeneity": "team_family_homogeneity_rate",
        "major_div": "team_major_family_diversity",
    }
    for key, label in labels.items():
        s = stats.get(f"valid_mixed_minus_same_{key}", {})
        stat_rows.append(
            [
                label,
                s.get("paired_question_count", 0),
                s.get("mean_delta", 0.0),
                f"[{safe_float(s.get('ci_low')):.4f}, {safe_float(s.get('ci_high')):.4f}]",
                s.get("wilcoxon_p_approx", 1.0),
            ]
        )
    lines.extend(md_table(["metric", "paired n", "mixed - same", "95% CI", "Wilcoxon p"], stat_rows))

    lines.extend(
        [
            "",
            "## 读法",
            "",
            "- 如果有效 trace 口径下 mixed 仍高于 same，说明 P3 的提升不是由坏 trace 直接抬高。",
            "- 如果某些模型有效题数较低，它们仍可用于趋势观察，但不应单独承担强结论。",
            "- 正式结论应同时报告四模型总体结果、有效 trace 口径、目标命中拆解和 GPT-5.5 复核。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="prove_experiments/p3_analysis_runs")
    parser.add_argument("--out_dir", default="prove_experiments/p3_valid_trace_quality")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    run_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("P3_")):
        row, qrows = analyze_run(run_dir)
        run_rows.append(row)
        question_rows.extend(qrows)

    stats = paired_stats(run_rows, question_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_rows, out_dir / "p3_valid_trace_run_summary.csv")
    write_csv(question_rows, out_dir / "p3_valid_trace_question_rows.csv")
    (out_dir / "p3_valid_trace_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(run_rows, stats, out_dir / "p3_valid_trace_quality.md")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
