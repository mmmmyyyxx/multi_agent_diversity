import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _to_float_or_nan(x: str) -> float:
    try:
        text = str(x).strip()
        if not text or text.lower() in {"none", "nan"}:
            return math.nan
        return float(text)
    except Exception:
        return math.nan


def _setting_order(name: str) -> int:
    order = {
        "shared_div": 0,
        "bank_div": 1,
        "shared_baseline": 2,
        "bank_baseline": 3,
    }
    return order.get(name, 99)


def _setting_color(row: Dict[str, str]) -> str:
    init_mode = str(row.get("init_mode", "")).lower()
    if init_mode == "shared":
        return "#4e79a7"
    if init_mode == "bank":
        return "#f28e2b"
    return "#9c9c9c"


def _setting_hatch(row: Dict[str, str]) -> str:
    baseline_only = str(row.get("baseline_only", "0"))
    return "//" if baseline_only in {"1", "True", "true"} else ""


def _short_label(name: str) -> str:
    mapping = {
        "shared_div": "shared\ndiv",
        "bank_div": "bank\ndiv",
        "shared_baseline": "shared\ntest-only",
        "bank_baseline": "bank\ntest-only",
    }
    return mapping.get(name, name)


def _sort_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda r: (_setting_order(str(r.get("setting", ""))), str(r.get("setting", ""))))


def _metric_values(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [_to_float_or_nan(r.get(key, "")) for r in rows]


def _finite_values(values: List[float]) -> List[float]:
    return [v for v in values if isinstance(v, float) and not math.isnan(v)]


def _resolve_ylim(
    values: List[float],
    preset: Optional[tuple],
    pad_ratio: float = 0.08,
    tight: bool = True,
    include_zero: bool = False,
) -> Optional[tuple]:
    finite = _finite_values(values)
    if not finite:
        return preset

    data_min = min(finite)
    data_max = max(finite)

    if tight or preset is None:
        if abs(data_max - data_min) < 1e-9:
            center = data_min
            base = max(abs(center), 0.05)
            pad = max(0.01, 0.08 * base)
            lo = center - pad
            hi = center + pad
        else:
            span = data_max - data_min
            pad = max(1e-4, span * pad_ratio)
            lo = data_min - pad
            hi = data_max + pad
        if include_zero:
            lo = min(0.0, lo)
        if data_min >= 0.0:
            lo = max(0.0, lo)
        if data_max <= 1.0 and hi <= 1.08:
            hi = min(1.0, hi)
        if preset is not None:
            lo = max(float(preset[0]), lo)
            hi = min(float(preset[1]), hi)
            if hi <= lo:
                lo, hi = float(preset[0]), float(preset[1])
        return (lo, hi)

    if preset is None:
        if abs(data_max - data_min) < 1e-9:
            base = max(abs(data_max), 1.0)
            pad = 0.1 * base
            return (data_min - pad, data_max + pad)
        span = data_max - data_min
        pad = max(1e-6, span * pad_ratio)
        return (data_min - pad, data_max + pad)

    lo, hi = float(preset[0]), float(preset[1])
    if lo > hi:
        lo, hi = hi, lo
    span = max(1e-6, hi - lo)
    pad = span * pad_ratio
    lo2 = min(lo, data_min - pad)
    hi2 = max(hi, data_max + pad)
    return (lo2, hi2)


def _plot_panel_bar(
    ax,
    x,
    values,
    colors,
    hatches,
    labels,
    title,
    ylabel,
    ylim: Optional[tuple] = None,
    tight_ylim: bool = True,
    include_zero: bool = False,
):
    bars = ax.bar(x, values, color=colors)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    resolved_ylim = _resolve_ylim(values, ylim, tight=tight_ylim, include_zero=include_zero)
    if resolved_ylim is not None:
        ax.set_ylim(*resolved_ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _plot_grouped_trace_summary_bar(
    ax,
    x,
    trace_values,
    summary_values,
    hatches,
    labels,
    title,
    ylabel,
    ylim: Optional[tuple] = None,
    tight_ylim: bool = True,
    include_zero: bool = False,
):
    width = 0.36
    trace_x = [i - width / 2 for i in x]
    summary_x = [i + width / 2 for i in x]
    trace_bars = ax.bar(trace_x, trace_values, width=width, color="#e15759", label="Full trace")
    summary_bars = ax.bar(summary_x, summary_values, width=width, color="#59a14f", label="Reasoning summary")
    for bars in (trace_bars, summary_bars):
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    resolved_ylim = _resolve_ylim(trace_values + summary_values, ylim, tight=tight_ylim, include_zero=include_zero)
    if resolved_ylim is not None:
        ax.set_ylim(*resolved_ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()


def _save_panel_figure(rows: List[Dict[str, str]], out_dir: Path, file_name: str, title: str, specs: List[Dict[str, str]]):
    rows = _sort_rows(rows)
    settings = [str(r.get("setting", "")) for r in rows]
    labels = [_short_label(x) for x in settings]
    x = list(range(len(rows)))
    colors = [_setting_color(r) for r in rows]
    hatches = [_setting_hatch(r) for r in rows]

    n = max(1, len(specs))
    cols = 2
    rows_n = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(16, 5 * rows_n))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, spec in zip(axes_list, specs):
        values = _metric_values(rows, spec["key"])
        _plot_panel_bar(
            ax,
            x,
            values,
            colors,
            hatches,
            labels,
            spec["title"],
            spec["ylabel"],
            spec.get("ylim"),
            bool(spec.get("tight_ylim", True)),
            bool(spec.get("include_zero", False)),
        )

    for ax in axes_list[len(specs):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / file_name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


def _save_trace_summary_panel(rows: List[Dict[str, str]], out_dir: Path):
    rows = _sort_rows(rows)
    settings = [str(r.get("setting", "")) for r in rows]
    labels = [_short_label(x) for x in settings]
    x = list(range(len(rows)))
    hatches = [_setting_hatch(r) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    _plot_grouped_trace_summary_bar(
        axes[0],
        x,
        _metric_values(rows, "latest_trace_cosine_diversity"),
        _metric_values(rows, "latest_reasoning_summary_cosine_diversity"),
        hatches,
        labels,
        "Trace vs Summary Cosine Diversity",
        "Cosine Diversity",
        (0.0, 0.35),
        True,
        False,
    )
    _plot_grouped_trace_summary_bar(
        axes[1],
        x,
        _metric_values(rows, "latest_trace_cosine_similarity"),
        _metric_values(rows, "latest_reasoning_summary_cosine_similarity"),
        hatches,
        labels,
        "Trace vs Summary Cosine Similarity",
        "Cosine Similarity",
        (0.65, 1.0),
        True,
        False,
    )
    fig.suptitle("Unified Comparison: Text-Level Trace and Summary Metrics", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "experiment_trace_summary_panel.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_grouped_embedding_bar(
    ax,
    x,
    prompt_values,
    trace_values,
    summary_values,
    hatches,
    labels,
    title,
    ylabel,
    ylim: Optional[tuple] = None,
    tight_ylim: bool = True,
    include_zero: bool = False,
):
    width = 0.24
    prompt_x = [i - width for i in x]
    trace_x = list(x)
    summary_x = [i + width for i in x]
    prompt_bars = ax.bar(prompt_x, prompt_values, width=width, color="#4e79a7", label="Prompt embedding")
    trace_bars = ax.bar(trace_x, trace_values, width=width, color="#e15759", label="Trace embedding")
    summary_bars = ax.bar(summary_x, summary_values, width=width, color="#edc948", label="Summary embedding")
    for bars in (prompt_bars, trace_bars, summary_bars):
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    resolved_ylim = _resolve_ylim(prompt_values + trace_values + summary_values, ylim, tight=tight_ylim, include_zero=include_zero)
    if resolved_ylim is not None:
        ax.set_ylim(*resolved_ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()


def _save_embedding_panel(rows: List[Dict[str, str]], out_dir: Path):
    rows = _sort_rows(rows)
    settings = [str(r.get("setting", "")) for r in rows]
    labels = [_short_label(x) for x in settings]
    x = list(range(len(rows)))
    hatches = [_setting_hatch(r) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    _plot_grouped_embedding_bar(
        axes[0],
        x,
        _metric_values(rows, "latest_prompt_embedding_cosine_diversity"),
        _metric_values(rows, "latest_trace_embedding_cosine_diversity"),
        _metric_values(rows, "latest_summary_embedding_cosine_diversity"),
        hatches,
        labels,
        "Embedding Cosine Diversity",
        "Cosine Diversity",
        (0.0, 0.35),
        True,
        False,
    )
    _plot_grouped_embedding_bar(
        axes[1],
        x,
        _metric_values(rows, "latest_prompt_embedding_cosine_similarity"),
        _metric_values(rows, "latest_trace_embedding_cosine_similarity"),
        _metric_values(rows, "latest_summary_embedding_cosine_similarity"),
        hatches,
        labels,
        "Embedding Cosine Similarity",
        "Cosine Similarity",
        (0.65, 1.0),
        True,
        False,
    )
    fig.suptitle("Unified Comparison: Embedding Metrics", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "experiment_embedding_panel.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


def plot(rows: List[Dict[str, str]], out_dir: Path):
    # Remove stale structure figure to avoid confusion with the new metric set.
    structure_path = out_dir / "experiment_structure_panel.png"
    if structure_path.exists():
        structure_path.unlink()

    _save_panel_figure(
        rows,
        out_dir,
        "experiment_diversity_panel.png",
        "Unified Comparison: Diversity Metrics",
        [
            {"key": "latest_prompt_cosine_diversity", "title": "Prompt Cosine Diversity", "ylabel": "Cosine Diversity", "ylim": (0.0, 1.0)},
            {"key": "latest_trace_cosine_diversity", "title": "Trace Cosine Diversity", "ylabel": "Cosine Diversity", "ylim": (0.0, 0.4)},
            {"key": "latest_reasoning_summary_cosine_diversity", "title": "Reasoning Summary Cosine Diversity", "ylabel": "Cosine Diversity", "ylim": (0.0, 0.4)},
            {"key": "latest_test_mean_family_diversity", "title": "Test Family Diversity", "ylabel": "Family Diversity", "ylim": (0.0, 1.0)},
        ],
    )

    _save_panel_figure(
        rows,
        out_dir,
        "experiment_similarity_panel.png",
        "Unified Comparison: Homogeneity Metrics",
        [
            {"key": "latest_test_mean_family_homogeneity_rate", "title": "Test Family Homogeneity", "ylabel": "Family Homogeneity", "ylim": (0.0, 1.0)},
            {"key": "latest_prompt_cosine_similarity", "title": "Prompt Cosine Similarity", "ylabel": "Cosine Similarity", "ylim": (0.0, 1.0)},
            {"key": "latest_trace_cosine_similarity", "title": "Trace Cosine Similarity", "ylabel": "Cosine Similarity", "ylim": (0.0, 1.0)},
            {"key": "latest_reasoning_summary_cosine_similarity", "title": "Reasoning Summary Cosine Similarity", "ylabel": "Cosine Similarity", "ylim": (0.0, 1.0)},
            {"key": "all_same_pair_rate", "title": "All-Same Family Pair Rate", "ylabel": "Rate", "ylim": (0.0, 1.0)},
        ],
    )

    _save_trace_summary_panel(rows, out_dir)
    _save_embedding_panel(rows, out_dir)

    _save_panel_figure(
        rows,
        out_dir,
        "experiment_behavior_panel.png",
        "Unified Comparison: Behavior and Optimization Metrics",
        [
            {"key": "latest_test_vote_acc", "title": "Test Vote Accuracy", "ylabel": "Vote Accuracy", "ylim": (0.0, 1.0)},
            {"key": "disagreement_rate", "title": "Disagreement Rate", "ylabel": "Rate", "ylim": (0.0, 1.0)},
            {"key": "prompt_drift_cosine_distance", "title": "Prompt Drift Cosine Distance", "ylabel": "Distance", "ylim": (0.0, 1.0)},
            {"key": "update_applied_rate", "title": "Update Applied Rate", "ylabel": "Rate", "ylim": (0.0, 1.0)},
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Plot unified comparisons for the four experiment settings.")
    parser.add_argument("--csv", type=str, required=True, help="Path to analyzed CSV (e.g. runs_experiments/experiment_analysis.csv)")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for figures")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(rows, out_dir)


if __name__ == "__main__":
    main()

