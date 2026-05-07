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
        "A_shared_no_div": 0,
        "B_shared_div": 1,
        "C_bank_no_div": 2,
        "D_bank_div": 3,
        "E_shared_testonly": 4,
        "F_bank_testonly": 5,
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
        "A_shared_no_div": "A\nshared\nno-div",
        "B_shared_div": "B\nshared\ndiv",
        "C_bank_no_div": "C\nbank\nno-div",
        "D_bank_div": "D\nbank\ndiv",
        "E_shared_testonly": "E\nshared\ntest-only",
        "F_bank_testonly": "F\nbank\ntest-only",
    }
    return mapping.get(name, name)


def _sort_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda r: (_setting_order(str(r.get("setting", ""))), str(r.get("setting", ""))))


def _metric_values(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [_to_float_or_nan(r.get(key, "")) for r in rows]


def _finite_values(values: List[float]) -> List[float]:
    return [v for v in values if isinstance(v, float) and not math.isnan(v)]


def _resolve_ylim(values: List[float], preset: Optional[tuple], pad_ratio: float = 0.05) -> Optional[tuple]:
    finite = _finite_values(values)
    if not finite:
        return preset

    data_min = min(finite)
    data_max = max(finite)

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


def _plot_panel_bar(ax, x, values, colors, hatches, labels, title, ylabel, ylim: Optional[tuple] = None):
    bars = ax.bar(x, values, color=colors)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    resolved_ylim = _resolve_ylim(values, ylim)
    if resolved_ylim is not None:
        ax.set_ylim(*resolved_ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _save_panel_figure(rows: List[Dict[str, str]], out_dir: Path, file_name: str, title: str, specs: List[Dict[str, str]]):
    rows = _sort_rows(rows)
    settings = [str(r.get("setting", "")) for r in rows]
    labels = [_short_label(x) for x in settings]
    x = list(range(len(rows)))
    colors = [_setting_color(r) for r in rows]
    hatches = [_setting_hatch(r) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes_list = list(axes.flat)
    for ax, spec in zip(axes_list, specs):
        values = _metric_values(rows, spec["key"])
        _plot_panel_bar(ax, x, values, colors, hatches, labels, spec["title"], spec["ylabel"], spec.get("ylim"))

    for ax in axes_list[len(specs):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / file_name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


def plot(rows: List[Dict[str, str]], out_dir: Path):
    # Remove stale structure figure to avoid confusion with the new metric set.
    structure_path = out_dir / "ablation_with_baselines_structure_panel.png"
    if structure_path.exists():
        structure_path.unlink()

    _save_panel_figure(
        rows,
        out_dir,
        "ablation_with_baselines_diversity_panel.png",
        "Unified Comparison: Diversity Metrics",
        [
            {"key": "final_prompt_cosine_diversity", "title": "Final Prompt Cosine Diversity", "ylabel": "Cosine Diversity", "ylim": (0.35, 0.85)},
            {"key": "final_trace_cosine_diversity", "title": "Final Trace Cosine Diversity", "ylabel": "Cosine Diversity", "ylim": (0.0, 0.2)},
            {"key": "final_test_mean_family_diversity", "title": "Final Test Family Diversity", "ylabel": "Family Diversity", "ylim": (0.25, 0.45)},
        ],
    )

    _save_panel_figure(
        rows,
        out_dir,
        "ablation_with_baselines_homogeneity_panel.png",
        "Unified Comparison: Homogeneity Metrics",
        [
            {"key": "final_test_mean_family_homogeneity_rate", "title": "Final Test Family Homogeneity", "ylabel": "Family Homogeneity", "ylim": (0.7, 1.0)},
            {"key": "final_prompt_cosine_similarity", "title": "Final Prompt Cosine Similarity", "ylabel": "Cosine Similarity", "ylim": (0.15, 1.0)},
            {"key": "final_trace_cosine_similarity", "title": "Final Trace Cosine Similarity", "ylabel": "Cosine Similarity", "ylim": (0.8, 1.0)},
        ],
    )

    _save_panel_figure(
        rows,
        out_dir,
        "ablation_with_baselines_behavior_panel.png",
        "Unified Comparison: Behavior and Optimization Metrics",
        [
            {"key": "final_test_vote_acc", "title": "Final Test Vote Accuracy", "ylabel": "Vote Accuracy", "ylim": (0.7, 0.8)},
            {"key": "disagreement_rate", "title": "Disagreement Rate", "ylabel": "Rate", "ylim": (0.2, 0.35)},
            {"key": "prompt_drift_cosine_distance", "title": "Prompt Drift Cosine Distance", "ylabel": "Distance", "ylim": (0.0, 0.9)},
            {"key": "update_applied_rate", "title": "Update Applied Rate", "ylabel": "Rate", "ylim": (0.05, 0.2)},
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Plot unified A/B/C/D + test-only baseline comparisons.")
    parser.add_argument("--csv", type=str, required=True, help="Path to merged CSV (e.g. runs_abcd/abcd_plus_baselines.csv)")
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
