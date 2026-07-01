import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("vote_acc", "Vote Accuracy", "vote_acc"),
    ("mean_embedding_diversity", "Embedding Diversity", "embedding_diversity"),
    ("mean_embedding_overlap", "Embedding Overlap", "embedding_overlap"),
    ("mean_invalid_rate", "Invalid Rate", "invalid_rate"),
]


def _read_json(path: Path, default: Any):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _numeric_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in history if isinstance(row, dict) and isinstance(row.get("epoch"), int)]


def _latest_split(history: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
    for row in reversed(history):
        payload = row.get(split, {}) if isinstance(row, dict) else {}
        if isinstance(payload, dict) and payload:
            return payload
    return {}


def _best_val_epoch(history: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any], float]:
    best_epoch = 0
    best_payload: Dict[str, Any] = {}
    best_score = -1e30
    for row in _numeric_history(history):
        val = row.get("val", {}) if isinstance(row.get("val", {}), dict) else {}
        score = (
            float(val.get("vote_acc", 0.0) or 0.0)
            + 0.2 * float(val.get("mean_embedding_diversity", 0.0) or 0.0)
            - 0.1 * float(val.get("mean_invalid_rate", 0.0) or 0.0)
        )
        if score > best_score:
            best_score = score
            best_epoch = int(row.get("epoch", 0))
            best_payload = val
    return best_epoch, best_payload, best_score


def _metrics_from_prediction_file(path: Path) -> Dict[str, Any]:
    rows = _read_jsonl(path)
    if not rows:
        return {}
    return {
        "size": len(rows),
        "vote_acc": float(np.mean([int(r.get("vote_correct", 0) or 0) for r in rows])),
        "mean_embedding_diversity": float(np.mean([float(r.get("embedding_diversity", 0.0) or 0.0) for r in rows])),
        "mean_embedding_overlap": float(np.mean([float(r.get("mean_embedding_overlap", 0.0) or 0.0) for r in rows])),
        "mean_invalid_rate": float(np.mean([float(r.get("invalid_rate", 0.0) or 0.0) for r in rows])),
    }


def _test_metrics(run_dir: Path, history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    from_history = _latest_split(history, "test")
    if from_history:
        return from_history, "history:test"
    for name in ["test_final_predictions.jsonl", "test_epoch1_predictions.jsonl"]:
        path = run_dir / name
        payload = _metrics_from_prediction_file(path)
        if payload:
            return payload, name
    return {}, "missing"


def _tight_ylim(values: List[Optional[float]], metric_key: str) -> Tuple[float, float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return (0.0, 1.0)
    lo = min(vals)
    hi = max(vals)
    if abs(hi - lo) < 1e-9:
        pad = 0.01 if metric_key != "mean_embedding_overlap" else 0.005
    else:
        pad = max((hi - lo) * 0.18, 0.005)
    lower = lo - pad
    upper = hi + pad
    if metric_key in {"vote_acc", "mean_embedding_diversity", "mean_embedding_overlap", "mean_invalid_rate"}:
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    if upper - lower < 0.02:
        mid = (upper + lower) / 2
        lower = max(0.0, mid - 0.01)
        upper = min(1.0, mid + 0.01)
    return lower, upper


def _metric_value(payload: Dict[str, Any], key: str) -> Optional[float]:
    if key not in payload or payload.get(key) == "":
        return None
    return float(payload.get(key, 0.0) or 0.0)


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bar_label(ax, x, y: Optional[float], text: str, baseline: float):
    if y is None:
        ax.text(x, baseline, "N/A", ha="center", va="bottom", fontsize=8)
    else:
        ax.text(x, y, text, ha="center", va="bottom", fontsize=8)


def plot_shared_beam_dynamics(root: Path, out_dir: Path, histories: Dict[str, List[Dict[str, Any]]], test_rows: Dict[str, Dict[str, Any]]):
    history = _numeric_history(histories["shared_beam"])
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    axes = axes.ravel()
    shared_best_test = test_rows.get("shared_beam_best", {})
    shared_last_test = test_rows.get("shared_beam_last", {})
    shared_base_test = test_rows.get("shared_baseline", {})
    bank_base_test = test_rows.get("bank_baseline", {})
    for ax, (metric_key, title, _) in zip(axes, METRICS):
        train_vals = [_metric_value(row.get("train", {}), metric_key) for row in history]
        val_vals = [_metric_value(row.get("val", {}), metric_key) for row in history]
        best_test_val = _metric_value(shared_best_test, metric_key)
        last_test_val = _metric_value(shared_last_test, metric_key)
        shared_base_val = _metric_value(shared_base_test, metric_key)
        bank_base_val = _metric_value(bank_base_test, metric_key)
        ylim = _tight_ylim(train_vals + val_vals + [best_test_val, last_test_val, shared_base_val, bank_base_val], metric_key)
        ax.plot(epochs, train_vals, marker="o", linewidth=2, label="shared_beam train")
        ax.plot(epochs, val_vals, marker="s", linewidth=2, label="shared_beam val")
        if best_test_val is not None:
            ax.axhline(best_test_val, color="#1f77b4", linestyle="--", linewidth=1.8, label="shared_beam best test")
        if last_test_val is not None:
            ax.axhline(last_test_val, color="#17becf", linestyle=(0, (5, 2)), linewidth=1.8, label="shared_beam last test")
        if shared_base_val is not None:
            ax.axhline(shared_base_val, color="#f58518", linestyle=":", linewidth=1.8, label="shared_baseline test")
        if bank_base_val is not None:
            ax.axhline(bank_base_val, color="#54a24b", linestyle="-.", linewidth=1.6, label="bank_baseline test")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("shared_beam Train/Validation Dynamics with Best/Last Test Records (Scaled Axes)")
    fig.tight_layout()
    fig.savefig(out_dir / "scaled_shared_beam_train_val_test_dynamics.png", dpi=200)
    plt.close(fig)


def plot_record_comparison(out_dir: Path, summary_rows: List[Dict[str, Any]]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    axes = axes.ravel()
    record_order = [
        ("shared_beam", "train_latest", "SB train latest"),
        ("shared_beam", "val_latest", "SB val latest"),
        ("shared_beam", "val_best", "SB val best"),
        ("shared_beam", "test_best", "SB test best"),
        ("shared_beam", "test_last", "SB test last"),
        ("shared_baseline", "test", "Shared base test"),
        ("bank_baseline", "test", "Bank base test"),
    ]
    colors = {
        "train_latest": "#4c78a8",
        "val_latest": "#72b7b2",
        "val_best": "#59a14f",
        "test_best": "#1f77b4",
        "test_last": "#17becf",
        "test": "#f58518",
    }
    for ax, (metric_key, title, _) in zip(axes, METRICS):
        metric_col = {
            "vote_acc": "vote_acc",
            "mean_embedding_diversity": "embedding_diversity",
            "mean_embedding_overlap": "embedding_overlap",
            "mean_invalid_rate": "invalid_rate",
        }[metric_key]
        values = []
        labels = []
        bar_colors = []
        for setting, split, label in record_order:
            row = next((r for r in summary_rows if r["setting"] == setting and r["split"] == split), None)
            values.append(None if row is None or row.get(metric_col, "") == "" else float(row[metric_col]))
            labels.append(label)
            bar_colors.append(colors.get(split, "#999999"))
        ylim = _tight_ylim(values, metric_key)
        xs = np.arange(len(values))
        ax.bar(xs, [0.0 if v is None else v for v in values], color=bar_colors)
        for x, v in zip(xs, values):
            label = "" if v is None else f"{v:.3f}"
            _bar_label(ax, x, v, label, ylim[0])
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Train / Validation / Best-Test / Last-Test Comparison (Scaled Axes)")
    fig.tight_layout()
    fig.savefig(out_dir / "scaled_train_val_test_record_comparison.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot current MMLU shared_beam/baseline visuals with scaled axes.")
    parser.add_argument("--root", type=str, default="runs_mmlu_subject_balanced_default_size_4way")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures_current_scaled"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        "shared_beam": root / "shared_beam_seed42",
        "shared_baseline": root / "shared_baseline_seed42",
        "bank_baseline": root / "bank_baseline_seed42",
    }
    histories = {name: _read_json(run_dir / "history.json", []) for name, run_dir in run_dirs.items()}
    tests = {name: _test_metrics(run_dir, histories[name])[0] for name, run_dir in run_dirs.items()}
    test_sources = {name: _test_metrics(run_dir, histories[name])[1] for name, run_dir in run_dirs.items()}
    shared_beam_best_test = _metrics_from_prediction_file(run_dirs["shared_beam"] / "test_best_predictions.jsonl")
    shared_beam_last_test = _metrics_from_prediction_file(run_dirs["shared_beam"] / "test_last_predictions.jsonl")
    tests["shared_beam_best"] = shared_beam_best_test
    tests["shared_beam_last"] = shared_beam_last_test
    test_sources["shared_beam_best"] = "test_best_predictions.jsonl"
    test_sources["shared_beam_last"] = "test_last_predictions.jsonl"

    shared_history = _numeric_history(histories["shared_beam"])
    latest_epoch = max([int(row["epoch"]) for row in shared_history] + [0])
    latest_train = _latest_split(histories["shared_beam"], "train")
    latest_val = _latest_split(histories["shared_beam"], "val")
    best_val_epoch, best_val, best_score = _best_val_epoch(histories["shared_beam"])

    summary_rows: List[Dict[str, Any]] = []

    def add_row(setting: str, split: str, payload: Dict[str, Any], source: str = ""):
        summary_rows.append(
            {
                "setting": setting,
                "split": split,
                "source": source,
                "size": payload.get("size", ""),
                "vote_acc": payload.get("vote_acc", ""),
                "embedding_diversity": payload.get("mean_embedding_diversity", ""),
                "embedding_overlap": payload.get("mean_embedding_overlap", ""),
                "invalid_rate": payload.get("mean_invalid_rate", ""),
            }
        )

    add_row("shared_beam", "train_latest", latest_train, f"epoch_{latest_epoch}")
    add_row("shared_beam", "val_latest", latest_val, f"epoch_{latest_epoch}")
    add_row("shared_beam", "val_best", best_val, f"epoch_{best_val_epoch};score={best_score:.6f}")
    add_row("shared_beam", "test_best", tests["shared_beam_best"], test_sources["shared_beam_best"])
    add_row("shared_beam", "test_last", tests["shared_beam_last"], test_sources["shared_beam_last"])
    add_row("shared_baseline", "test", tests["shared_baseline"], test_sources["shared_baseline"])
    add_row("bank_baseline", "test", tests["bank_baseline"], test_sources["bank_baseline"])

    _write_csv(root / "current_scaled_split_metrics.csv", summary_rows)
    plot_shared_beam_dynamics(root, out_dir, histories, tests)
    plot_record_comparison(out_dir, summary_rows)

    print(f"Wrote {root / 'current_scaled_split_metrics.csv'}")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
