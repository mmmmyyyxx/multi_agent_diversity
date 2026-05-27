#!/usr/bin/env python
"""Factor contrast analysis for P4 model identity vs prompt effects."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute_experiment_metrics import (
    DEFAULT_SUMMARY_EMBEDDING_MODEL,
    DEFAULT_TRACE_EMBEDDING_CHUNK_OVERLAP,
    DEFAULT_TRACE_EMBEDDING_CHUNK_WORDS,
    SummaryEmbeddingEncoder,
    _pairwise_cosine_diversity,
    _pairwise_document_embedding_cosine_diversity,
    _pairwise_embedding_cosine_diversity,
)
from multi_dataset_diverse_rl.utils import compute_strategy_family_profile_metrics
from prove_experiment_utils import find_prediction_file, read_json, read_jsonl, safe_float


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def mean_optional(values: list[Any]) -> float | str:
    vals = [safe_float(v) for v in values if v not in (None, "")]
    return mean(vals) if vals else ""


def prompt_family(run_name: str) -> str:
    text = run_name.lower()
    if "same_prompt" in text:
        return "same_prompt"
    if "mixed_strategy" in text:
        return "mixed_strategy"
    if "same_definition" in text:
        return "same_definition"
    if "same_elimination" in text:
        return "same_elimination"
    return "other"


def prompt_signature(run_dir: Path) -> tuple[str, ...]:
    probe = read_json(run_dir / "probe_prompts.json")
    candidates: list[dict[str, Any]] = []
    if isinstance(probe, dict) and isinstance(probe.get("agents", []), list):
        candidates = [x for x in probe.get("agents", []) if isinstance(x, dict)]
    if not candidates:
        meta = read_json(run_dir / "run_meta.json")
        probe_meta = meta.get("probe", {}) if isinstance(meta, dict) and isinstance(meta.get("probe", {}), dict) else {}
        if isinstance(probe_meta.get("agents", []), list):
            candidates = [x for x in probe_meta.get("agents", []) if isinstance(x, dict)]
    prompts = [str(agent.get("prompt", "")).strip() for agent in candidates if str(agent.get("prompt", "")).strip()]
    if not prompts:
        return tuple()
    return tuple(prompts)


def model_name(run_dir: Path) -> str:
    meta = read_json(run_dir / "run_meta.json")
    cfg = meta.get("config", {}) if isinstance(meta, dict) and isinstance(meta.get("config", {}), dict) else {}
    return str(cfg.get("model", ""))


def parse_model_filter(text: str) -> set[str]:
    return {item.strip() for item in str(text or "").split(",") if item.strip()}


def model_allowed(model: str, include_models: set[str], exclude_models: set[str]) -> bool:
    if include_models and model not in include_models:
        return False
    if exclude_models and model in exclude_models:
        return False
    return True


def load_run_metric_maps(runs_root: Path) -> dict[str, dict[str, float | str]]:
    candidates = [
        runs_root / "p4_embedding_metrics_cleaned.csv",
        runs_root.parent / "p4_embedding_metrics.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {}

    metric_map: dict[str, dict[str, float | str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_name = str(row.get("run_name", "")).strip()
            if not run_name:
                continue
            metric_map[run_name] = {
                "run_name": run_name,
                "latest_prompt_embedding_cosine_diversity": safe_float(row.get("latest_prompt_embedding_cosine_diversity")),
                "latest_trace_embedding_cosine_diversity": safe_float(row.get("latest_trace_embedding_cosine_diversity")),
                "latest_prompt_cosine_diversity": safe_float(row.get("latest_prompt_cosine_diversity")),
                "latest_trace_cosine_diversity": safe_float(row.get("latest_trace_cosine_diversity")),
                "latest_summary_embedding_cosine_diversity": safe_float(row.get("latest_summary_embedding_cosine_diversity")),
                "latest_reasoning_summary_cosine_diversity": safe_float(row.get("latest_reasoning_summary_cosine_diversity")),
                "latest_test_mean_family_diversity": safe_float(row.get("latest_test_mean_family_diversity")),
                "latest_test_mean_family_homogeneity_rate": safe_float(row.get("latest_test_mean_family_homogeneity_rate")),
                "latest_test_mean_llm_direct_diversity_score": safe_float(row.get("latest_test_mean_llm_direct_diversity_score")),
                "latest_test_vote_acc": safe_float(row.get("latest_test_vote_acc")),
                "mean_family_diversity_from_preds": safe_float(row.get("mean_family_diversity_from_preds")),
                "mean_family_homogeneity_rate_from_preds": safe_float(row.get("mean_family_homogeneity_rate_from_preds")),
                "mean_llm_direct_diversity_score_from_preds": safe_float(row.get("mean_llm_direct_diversity_score_from_preds")),
                "mean_vote_acc_from_preds": safe_float(row.get("mean_vote_acc_from_preds")),
                "all_same_pair_rate_from_preds": safe_float(row.get("all_same_pair_rate_from_preds")),
                "eval_size": safe_float(row.get("eval_size")),
            }
    return metric_map


def distribution_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    return 0.5 * sum(abs(safe_float(a.get(k)) - safe_float(b.get(k))) for k in keys)


def trace_texts_by_question(run_dir: Path) -> dict[str, tuple[str, ...]]:
    rows = read_jsonl(run_dir / "test_trace_history.jsonl")
    out: dict[str, tuple[str, ...]] = {}
    for row in rows:
        if str(row.get("split", "")).lower() and not str(row.get("split", "")).lower().startswith("test"):
            continue
        qh = str(row.get("question_hash", ""))
        agents = row.get("agents", [])
        if not qh or not isinstance(agents, list):
            continue
        traces = []
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            trace = str(agent.get("trace", "") or agent.get("reasoning_summary", "")).strip()
            if trace:
                traces.append(trace)
        out[qh] = tuple(traces)
    return out


def prepare_embedding_cache(records: list[dict[str, Any]], encoder: SummaryEmbeddingEncoder) -> None:
    unique_texts: list[str] = []
    seen: set[str] = set()
    for rec in records:
        for text in list(rec.get("prompts", ())) + list(rec.get("traces", ())):
            cleaned = str(text).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique_texts.append(cleaned)
    if unique_texts:
        encoder.encode(unique_texts)


def combined_metrics(records: list[dict[str, Any]], encoder: SummaryEmbeddingEncoder) -> dict[str, Any]:
    return {
        "family_diversity": safe_float(
            compute_strategy_family_profile_metrics(
                [label for r in records for label in list(r.get("primary_family_labels", []))],
                [label for r in records for label in list(r.get("secondary_family_labels", r.get("primary_family_labels", [])))],
                use_dual_family=True,
                primary_weight=0.7,
                secondary_weight=0.3,
                same_major_weight=0.5,
                macro_diversity_weight=0.5,
            ).get("team_family_diversity")
        ),
        "major_diversity": safe_float(
            compute_strategy_family_profile_metrics(
                [label for r in records for label in list(r.get("primary_family_labels", []))],
                [label for r in records for label in list(r.get("secondary_family_labels", r.get("primary_family_labels", [])))],
                use_dual_family=True,
                primary_weight=0.7,
                secondary_weight=0.3,
                same_major_weight=0.5,
                macro_diversity_weight=0.5,
            ).get("team_major_family_diversity")
        ),
        "homogeneity": safe_float(
            compute_strategy_family_profile_metrics(
                [label for r in records for label in list(r.get("primary_family_labels", []))],
                [label for r in records for label in list(r.get("secondary_family_labels", r.get("primary_family_labels", [])))],
                use_dual_family=True,
                primary_weight=0.7,
                secondary_weight=0.3,
                same_major_weight=0.5,
                macro_diversity_weight=0.5,
            ).get("team_family_homogeneity_rate")
        ),
        "prompt_embedding_diversity": (
            _pairwise_embedding_cosine_diversity(
                [text for r in records for text in list(r.get("prompts", ()))],
                encoder,
            )[0]
        ),
        "trace_embedding_diversity": (
            _pairwise_document_embedding_cosine_diversity(
                [text for r in records for text in list(r.get("traces", ()))],
                encoder,
                chunk_words=DEFAULT_TRACE_EMBEDDING_CHUNK_WORDS,
                chunk_overlap=DEFAULT_TRACE_EMBEDDING_CHUNK_OVERLAP,
            )[0]
        ),
        "trace_token_diversity": _pairwise_cosine_diversity([text for r in records for text in list(r.get("traces", ()))])[0],
        "vote_acc": mean([safe_float(r.get("vote_correct")) for r in records]),
    }


def average_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_family_diversity": mean_optional([r.get("family_diversity") for r in rows]),
        "mean_major_diversity": mean_optional([r.get("major_diversity") for r in rows]),
        "mean_homogeneity": mean_optional([r.get("homogeneity") for r in rows]),
        "mean_prompt_embedding_diversity": mean_optional([r.get("prompt_embedding_diversity") for r in rows]),
        "mean_trace_embedding_diversity": mean_optional([r.get("trace_embedding_diversity") for r in rows]),
        "mean_trace_token_diversity": mean_optional([r.get("trace_token_diversity") for r in rows]),
        "mean_vote_acc": mean_optional([r.get("vote_acc") for r in rows]),
    }


def delta_rows(full_summary: list[dict[str, Any]], baseline_name: str = "same_model_same_prompt") -> list[dict[str, Any]]:
    baseline_rows = [row for row in full_summary if str(row.get("contrast", "")) == baseline_name]
    if not baseline_rows:
        return []
    baseline_metrics = average_metric_rows(baseline_rows)
    baseline = {
        **baseline_metrics,
        "mean_major_distribution_distance": mean([safe_float(r.get("mean_major_distribution_distance")) for r in baseline_rows]),
    }
    metrics = [
        "mean_family_diversity",
        "mean_major_diversity",
        "mean_homogeneity",
        "mean_prompt_embedding_diversity",
        "mean_trace_embedding_diversity",
        "mean_trace_token_diversity",
        "mean_vote_acc",
        "mean_major_distribution_distance",
    ]
    rows = []
    for row in full_summary:
        out = {
            "baseline": baseline_name,
            "contrast": row.get("contrast", ""),
            "left_model": row.get("left_model", ""),
            "right_model": row.get("right_model", ""),
            "left_prompt_family": row.get("left_prompt_family", ""),
            "right_prompt_family": row.get("right_prompt_family", ""),
            "prompt_mode": row.get("prompt_mode", ""),
            "unit": row.get("unit", ""),
            "n": row.get("n", ""),
        }
        for metric in metrics:
            value = row.get(metric, "")
            base = baseline.get(metric, "")
            out[f"delta_{metric}"] = safe_float(value) - safe_float(base) if value not in ("", None) and base not in ("", None) else ""
        rows.append(out)
    return rows


def run_records(
    runs_root: Path,
    include_models: set[str] | None = None,
    exclude_models: set[str] | None = None,
) -> list[dict[str, Any]]:
    include_models = include_models or set()
    exclude_models = exclude_models or set()
    out: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("P4_")):
        model = model_name(run_dir)
        if not model_allowed(model, include_models, exclude_models):
            continue
        signature = prompt_signature(run_dir)
        mode = "exact" if len(set(signature)) <= 1 and len(signature) > 0 else "family"
        traces_by_q = trace_texts_by_question(run_dir)
        pred_file = find_prediction_file(run_dir)
        preds = read_jsonl(pred_file) if pred_file else []
        for rec in preds:
            qh = str(rec.get("question_hash", ""))
            if not qh:
                continue
            dist = rec.get("major_family_distribution", {})
            if not isinstance(dist, dict):
                dist = {}
            out.append(
                {
                    "run_name": run_dir.name,
                    "model": model,
                    "prompt_family": prompt_family(run_dir.name),
                    "prompt_signature": signature,
                    "prompt_mode": mode,
                    "question_hash": qh,
                    "prompts": signature,
                    "traces": traces_by_q.get(qh, tuple()),
                    "primary_family_labels": rec.get("primary_family_labels", []),
                    "secondary_family_labels": rec.get("secondary_family_labels", rec.get("primary_family_labels", [])),
                    "major_family_distribution": dist,
                    "team_family_diversity": safe_float(rec.get("team_family_diversity")),
                    "team_major_family_diversity": safe_float(rec.get("team_major_family_diversity")),
                    "team_family_homogeneity_rate": safe_float(rec.get("team_family_homogeneity_rate")),
                    "vote_correct": safe_float(rec.get("vote_correct")),
                }
            )
    return out


def attach_run_level_metrics(records: list[dict[str, Any]], run_metric_map: dict[str, dict[str, float | str]]) -> None:
    for rec in records:
        metrics = run_metric_map.get(str(rec.get("run_name", "")), {})
        rec.update(metrics)


def contrast_kind(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    same_model = a["model"] == b["model"]
    same_prompt_family = a["prompt_family"] == b["prompt_family"]
    same_signature = a.get("prompt_signature", tuple()) == b.get("prompt_signature", tuple())
    same_mode = a.get("prompt_mode", "") == b.get("prompt_mode", "")
    if not same_mode:
        return None
    if a.get("prompt_mode") == "exact":
        if same_model:
            return None
        if same_signature:
            return "different_model_same_prompt"
        return None
    if same_model and same_prompt_family:
        return None
    if same_model and not same_prompt_family:
        return "same_model_different_prompt_family"
    if not same_model and same_prompt_family:
        return "different_model_same_prompt_family"
    return "different_model_different_prompt_family"


def summarize(records: list[dict[str, Any]], encoder: SummaryEmbeddingEncoder) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_question[rec["question_hash"]].append(rec)

    pairs: list[dict[str, Any]] = []
    full_metric_rows: list[dict[str, Any]] = []
    for qh, vals in by_question.items():
        for a, b in itertools.combinations(vals, 2):
            kind = contrast_kind(a, b)
            if kind is None:
                continue
            mode = "exact" if kind == "different_model_same_prompt" else "family"
            metrics = combined_metrics([a, b], encoder)
            pairs.append(
                {
                    "question_hash": qh,
                    "contrast": kind,
                    "left_run": a["run_name"],
                    "right_run": b["run_name"],
                    "left_model": a["model"],
                    "right_model": b["model"],
                    "left_prompt_family": a["prompt_family"],
                    "right_prompt_family": b["prompt_family"],
                    "major_distribution_distance": distribution_distance(
                        a["major_family_distribution"],
                        b["major_family_distribution"],
                    ),
                }
            )
            full_metric_rows.append(
                {
                    "question_hash": qh,
                    "contrast": kind,
                    "prompt_mode": mode,
                    "unit": "between_team_same_question",
                    "left_run": a["run_name"],
                    "right_run": b["run_name"],
                    "left_model": a["model"],
                    "right_model": b["model"],
                    "left_prompt_family": a["prompt_family"],
                    "right_prompt_family": b["prompt_family"],
                    "major_distribution_distance": distribution_distance(
                        a["major_family_distribution"],
                        b["major_family_distribution"],
                    ),
                    **metrics,
                }
            )

    within_rows: list[dict[str, Any]] = []
    for mode, label in [("family", "same_model_same_prompt_family"), ("exact", "same_model_same_prompt")]:
        subset = [r for r in records if r.get("prompt_mode") == mode]
        if not subset:
            continue
        within_rows.append(
            {
                "contrast": label,
                "prompt_mode": mode,
                "unit": "within_team",
                "n": len(subset),
                "mean_major_distribution_distance": mean([r["team_major_family_diversity"] for r in subset]),
                "mean_family_diversity": mean([r["team_family_diversity"] for r in subset]),
                "mean_homogeneity": mean([r["team_family_homogeneity_rate"] for r in subset]),
            }
        )
        for rec in subset:
            full_metric_rows.append(
                {
                    "question_hash": rec["question_hash"],
                    "contrast": label,
                    "prompt_mode": mode,
                    "unit": "within_team",
                    "left_run": rec["run_name"],
                    "right_run": "",
                    "left_model": rec["model"],
                    "right_model": rec["model"],
                    "left_prompt_family": rec["prompt_family"],
                    "right_prompt_family": rec["prompt_family"],
                    "major_distribution_distance": rec["team_major_family_diversity"],
                    **combined_metrics([rec], encoder),
                }
            )
    full_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in full_metric_rows:
        if row.get("unit", "") == "within_team":
            key = (
                row.get("contrast", ""),
                row.get("prompt_mode", ""),
                row.get("unit", ""),
                row.get("left_model", ""),
                row.get("left_prompt_family", ""),
            )
        else:
            key = (
                row.get("contrast", ""),
                row.get("prompt_mode", ""),
                row.get("unit", ""),
                row.get("left_model", ""),
                row.get("right_model", ""),
                row.get("left_prompt_family", ""),
                row.get("right_prompt_family", ""),
            )
        full_grouped[key].append(row)
    full_summary: list[dict[str, Any]] = []
    for key in sorted(full_grouped.keys()):
        rows = full_grouped[key]
        if not rows:
            continue
        base = average_metric_rows(rows)
        sample = rows[0]
        full_summary.append(
            {
                "contrast": sample.get("contrast", ""),
                "prompt_mode": sample.get("prompt_mode", ""),
                "unit": sample.get("unit", ""),
                "left_model": sample.get("left_model", ""),
                "right_model": sample.get("right_model", ""),
                "left_prompt_family": sample.get("left_prompt_family", ""),
                "right_prompt_family": sample.get("right_prompt_family", ""),
                "n": len(rows),
                "mean_major_distribution_distance": mean([safe_float(r["major_distribution_distance"]) for r in rows]),
                **base,
            }
        )

    summary = [
        {
            "contrast": row.get("contrast", ""),
            "prompt_mode": row.get("prompt_mode", ""),
            "unit": row.get("unit", ""),
            "left_model": row.get("left_model", ""),
            "right_model": row.get("right_model", ""),
            "left_prompt_family": row.get("left_prompt_family", ""),
            "right_prompt_family": row.get("right_prompt_family", ""),
            "n": row.get("n", ""),
            "mean_major_distribution_distance": row.get("mean_major_distribution_distance", ""),
            "mean_family_diversity": row.get("mean_family_diversity", ""),
            "mean_major_diversity": row.get("mean_major_diversity", ""),
            "mean_homogeneity": row.get("mean_homogeneity", ""),
            "mean_prompt_embedding_diversity": row.get("mean_prompt_embedding_diversity", ""),
            "mean_trace_embedding_diversity": row.get("mean_trace_embedding_diversity", ""),
            "mean_trace_token_diversity": row.get("mean_trace_token_diversity", ""),
            "mean_vote_acc": row.get("mean_vote_acc", ""),
        }
        for row in full_summary
    ]

    summary = [
        {
            "contrast": row.get("contrast", ""),
            "prompt_mode": row.get("prompt_mode", ""),
            "unit": row.get("unit", ""),
            "left_model": row.get("left_model", ""),
            "right_model": row.get("right_model", ""),
            "left_prompt_family": row.get("left_prompt_family", ""),
            "right_prompt_family": row.get("right_prompt_family", ""),
            "n": row.get("n", ""),
            "mean_major_distribution_distance": row.get("mean_major_distribution_distance", ""),
            "mean_family_diversity": row.get("mean_family_diversity", ""),
            "mean_major_diversity": row.get("mean_major_diversity", ""),
            "mean_homogeneity": row.get("mean_homogeneity", ""),
            "mean_prompt_embedding_diversity": row.get("mean_prompt_embedding_diversity", ""),
            "mean_trace_embedding_diversity": row.get("mean_trace_embedding_diversity", ""),
            "mean_trace_token_diversity": row.get("mean_trace_token_diversity", ""),
            "mean_vote_acc": row.get("mean_vote_acc", ""),
        }
        for row in full_summary
    ]

    return summary, pairs, full_summary, full_metric_rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()}) if rows else ["id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def signed_md_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.4f}"
    return md_cell(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(v) for v in row) + " |")
    return lines


def md_signed_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(signed_md_cell(v) for v in row) + " |")
    return lines


def write_md(summary: list[dict[str, Any]], full_summary: list[dict[str, Any]], deltas: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# P4 模型身份与 Prompt 因子对比",
        "",
        "本页按六个口径核对：`same_model_same_prompt` 是 exact baseline，`different_model_same_prompt` 是跨模型同 prompt 对照，其余四组是 prompt family / model identity 的组合。",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        md_table(
            [
                "contrast",
                "prompt_mode",
                "unit",
                "left_model",
                "right_model",
                "left_prompt_family",
                "right_prompt_family",
                "n",
                "family_div",
                "major_div",
                "homogeneity",
                "prompt_embedding_div",
                "trace_embedding_div",
                "trace_token_div",
                "vote_acc",
                "major_dist",
            ],
            [
                [
                    r.get("contrast", ""),
                    r.get("prompt_mode", ""),
                    r.get("unit", ""),
                    r.get("left_model", ""),
                    r.get("right_model", ""),
                    r.get("left_prompt_family", ""),
                    r.get("right_prompt_family", ""),
                    r.get("n", ""),
                    r.get("mean_family_diversity", ""),
                    r.get("mean_major_diversity", ""),
                    r.get("mean_homogeneity", ""),
                    r.get("mean_prompt_embedding_diversity", ""),
                    r.get("mean_trace_embedding_diversity", ""),
                    r.get("mean_trace_token_diversity", ""),
                    r.get("mean_vote_acc", ""),
                    r.get("mean_major_distribution_distance", ""),
                ]
                for r in summary
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Signed Delta vs exact same_model_same_prompt",
            "",
        ]
    )
    lines.extend(
        md_signed_table(
            [
                "contrast",
                "left_model",
                "right_model",
                "left_prompt_family",
                "right_prompt_family",
                "Δ family_div",
                "Δ major_div",
                "Δ homogeneity",
                "Δ prompt_embedding_div",
                "Δ trace_embedding_div",
                "Δ trace_token_div",
                "Δ vote_acc",
                "Δ major_dist",
            ],
            [
                [
                    r.get("contrast", ""),
                    r.get("left_model", ""),
                    r.get("right_model", ""),
                    r.get("left_prompt_family", ""),
                    r.get("right_prompt_family", ""),
                    r.get("delta_mean_family_diversity", ""),
                    r.get("delta_mean_major_diversity", ""),
                    r.get("delta_mean_homogeneity", ""),
                    r.get("delta_mean_prompt_embedding_diversity", ""),
                    r.get("delta_mean_trace_embedding_diversity", ""),
                    r.get("delta_mean_trace_token_diversity", ""),
                    r.get("delta_mean_vote_acc", ""),
                    r.get("delta_mean_major_distribution_distance", ""),
                ]
                for r in deltas
            ],
        )
    )
    lines.extend(
        [
            "",
            "signed delta 使用 `contrast_mean - same_model_same_prompt_mean`，不取绝对值。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="prove_experiments/cleaned_runs")
    parser.add_argument("--out_dir", default="prove_experiments/p4_factor_contrasts")
    parser.add_argument("--include_models", default="", help="Comma-separated model names to include.")
    parser.add_argument("--exclude_models", default="", help="Comma-separated model names to exclude.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    records = run_records(
        Path(args.runs_root),
        include_models=parse_model_filter(args.include_models),
        exclude_models=parse_model_filter(args.exclude_models),
    )
    attach_run_level_metrics(records, load_run_metric_maps(Path(args.runs_root)))
    encoder = SummaryEmbeddingEncoder(DEFAULT_SUMMARY_EMBEDDING_MODEL)
    prepare_embedding_cache(records, encoder)
    summary, pairs, full_summary, full_metric_rows = summarize(records, encoder)
    deltas = delta_rows(full_summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary, out_dir / "p4_factor_contrast_summary.csv")
    write_csv(pairs, out_dir / "p4_factor_contrast_pairs.csv")
    write_csv(full_summary, out_dir / "p4_factor_full_metric_summary.csv")
    write_csv(full_metric_rows, out_dir / "p4_factor_full_metric_rows.csv")
    write_csv(deltas, out_dir / "p4_factor_signed_delta_vs_exact_baseline.csv")
    (out_dir / "p4_factor_contrast_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "p4_factor_full_metric_summary.json").write_text(json.dumps(full_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "p4_factor_signed_delta_vs_exact_baseline.json").write_text(json.dumps(deltas, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(summary, full_summary, deltas, out_dir / "p4_factor_contrast_summary.md")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
