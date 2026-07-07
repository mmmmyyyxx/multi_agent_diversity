import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clamp_ylim(values, pad=0.015, floor=0.0, ceil=1.0):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return floor, ceil
    lo = max(floor, min(vals) - pad)
    hi = min(ceil, max(vals) + pad)
    if hi - lo < 0.04:
        center = (hi + lo) / 2
        lo = max(floor, center - 0.02)
        hi = min(ceil, center + 0.02)
    return lo, hi


def final_test_acc(history):
    for rec in reversed(history):
        if isinstance(rec, dict) and rec.get("epoch") == "final":
            test = rec.get("test", {}) if isinstance(rec.get("test", {}), dict) else {}
            return float(test.get("vote_acc", 0.0) or 0.0), int(test.get("size", 0) or 0), rec
    return 0.0, 0, {}


def baseline_test_acc(run_dir: Path):
    hist = load_json(run_dir / "history.json")
    for rec in hist:
        test = rec.get("test", {}) if isinstance(rec, dict) and isinstance(rec.get("test", {}), dict) else {}
        if test:
            return float(test.get("vote_acc", 0.0) or 0.0), int(test.get("size", 0) or 0)
    return 0.0, 0


def epoch_rows(history):
    rows = []
    for rec in history:
        if not isinstance(rec, dict) or not isinstance(rec.get("epoch"), int):
            continue
        train = rec.get("train", {}) if isinstance(rec.get("train", {}), dict) else {}
        val = rec.get("val", {}) if isinstance(rec.get("val", {}), dict) else {}
        rows.append(
            {
                "epoch": int(rec["epoch"]),
                "train_acc": float(train.get("vote_acc", 0.0) or 0.0),
                "val_acc": float(val.get("vote_acc", 0.0) or 0.0),
                "beam_refresh_changed": int((rec.get("beam_refresh", {}) or {}).get("active_prompt_changed_count", 0) or 0),
            }
        )
    return rows


def candidate_update_rows(update_logs):
    groups = defaultdict(list)
    refresh_rows = []
    for row in update_logs:
        if row.get("event") == "beam_refresh":
            refresh_rows.append(row)
            continue
        if "candidate_id" not in row:
            continue
        key = (int(row.get("epoch", 0) or 0), int(row.get("step", 0) or 0), int(row.get("agent_id", -1) or -1))
        groups[key].append(row)

    updates = []
    for idx, (key, rows) in enumerate(sorted(groups.items()), start=1):
        epoch, step, agent_id = key
        rows = sorted(rows, key=lambda r: float(r.get("reward", 0.0) or 0.0), reverse=True)
        top = rows[0]
        optimizer_rewards = [float(r.get("reward", 0.0) or 0.0) for r in rows if r.get("candidate_source") == "optimizer"]
        existing_rewards = [float(r.get("reward", 0.0) or 0.0) for r in rows if r.get("candidate_source") == "existing_beam"]
        updates.append(
            {
                "update_index": idx,
                "epoch": epoch,
                "step": step,
                "agent_id": agent_id,
                "top_reward": float(top.get("reward", 0.0) or 0.0),
                "top_team_accuracy": float(top.get("team_accuracy", 0.0) or 0.0),
                "top_target_agent_accuracy": float(top.get("target_agent_accuracy", 0.0) or 0.0),
                "top_source": str(top.get("candidate_source", "")),
                "top_is_optimizer": 1 if str(top.get("candidate_source", "")) == "optimizer" else 0,
                "best_optimizer_reward": max(optimizer_rewards) if optimizer_rewards else None,
                "best_existing_reward": max(existing_rewards) if existing_rewards else None,
                "optimizer_beats_existing": (
                    1
                    if optimizer_rewards and existing_rewards and max(optimizer_rewards) > max(existing_rewards)
                    else 0
                ),
                "candidate_count": len(rows),
            }
        )
    return updates, refresh_rows


def summarize_by_epoch(update_rows):
    grouped = defaultdict(list)
    for row in update_rows:
        grouped[int(row["epoch"])].append(row)
    summary = []
    for epoch in sorted(grouped):
        rows = grouped[epoch]
        n = len(rows)
        summary.append(
            {
                "epoch": epoch,
                "update_count": n,
                "mean_top_reward": sum(r["top_reward"] for r in rows) / n if n else 0.0,
                "mean_top_target_agent_accuracy": sum(r["top_target_agent_accuracy"] for r in rows) / n if n else 0.0,
                "top_optimizer_rate": sum(r["top_is_optimizer"] for r in rows) / n if n else 0.0,
                "optimizer_beats_existing_rate": sum(r["optimizer_beats_existing"] for r in rows) / n if n else 0.0,
            }
        )
    return summary


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_learning_curve(out_dir: Path, epochs, final_acc, selected_epoch):
    xs = [r["epoch"] for r in epochs]
    train = [r["train_acc"] for r in epochs]
    val = [r["val_acc"] for r in epochs]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(xs, train, marker="o", linewidth=2, label="Train vote accuracy")
    ax.plot(xs, val, marker="s", linewidth=2, label="Validation vote accuracy")
    if selected_epoch:
        best_val = next((r["val_acc"] for r in epochs if r["epoch"] == selected_epoch), None)
        if best_val is not None:
            ax.scatter([selected_epoch], [best_val], s=120, marker="*", color="#d62728", label=f"Selected epoch {selected_epoch}")
    ax.axhline(final_acc, color="#2ca02c", linestyle="--", linewidth=1.8, label=f"Final test acc {final_acc:.3f}")
    ax.set_title("Accuracy-only prompt optimization: train/validation trajectory")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(xs)
    ax.set_ylim(*clamp_ylim(train + val + [final_acc], pad=0.02))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_learning_curve.png", dpi=180)
    plt.close(fig)


def plot_test_comparison(out_dir: Path, labels, values):
    colors = ["#7f7f7f", "#9467bd", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(labels, values, color=colors[: len(values)], width=0.62)
    ax.set_title("Final test accuracy compared with baselines")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(*clamp_ylim(values, pad=0.02))
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    if len(values) >= 3:
        delta = values[2] - values[0]
        ax.text(2, values[2] - 0.006, f"vs shared: {delta:+.3f}", ha="center", va="top", color="white", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "test_accuracy_comparison.png", dpi=180)
    plt.close(fig)


def plot_candidate_behavior(out_dir: Path, update_rows, epoch_summary):
    if not update_rows:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=False)
    xs = [r["update_index"] for r in update_rows]
    rewards = [r["top_reward"] for r in update_rows]
    target_acc = [r["top_target_agent_accuracy"] for r in update_rows]
    colors = ["#1f77b4" if r["top_is_optimizer"] else "#7f7f7f" for r in update_rows]
    axes[0].scatter(xs, rewards, c=colors, s=22, alpha=0.8)
    axes[0].plot(xs, rewards, color="#1f77b4", alpha=0.25, linewidth=1)
    axes[0].set_title("Top-1 candidate reward per prompt update (reward = target-agent accuracy)")
    axes[0].set_ylabel("Top-1 reward")
    axes[0].set_ylim(*clamp_ylim(rewards + target_acc, pad=0.04))
    axes[0].grid(alpha=0.25)
    axes[0].text(0.01, 0.93, "Blue: optimizer proposal wins; gray: existing beam wins", transform=axes[0].transAxes, fontsize=9)

    ep = [r["epoch"] for r in epoch_summary]
    mean_reward = [r["mean_top_reward"] for r in epoch_summary]
    opt_rate = [r["top_optimizer_rate"] for r in epoch_summary]
    axes[1].plot(ep, mean_reward, marker="o", linewidth=2, label="Mean top reward")
    axes[1].bar(ep, opt_rate, width=0.55, alpha=0.35, label="Optimizer top-1 rate")
    axes[1].set_title("Candidate selection signal by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(ep)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_accuracy_selection.png", dpi=180)
    plt.close(fig)


def plot_overview(out_dir: Path, epochs, final_acc, selected_epoch, baseline_labels, baseline_values, epoch_summary):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    xs = [r["epoch"] for r in epochs]
    train = [r["train_acc"] for r in epochs]
    val = [r["val_acc"] for r in epochs]
    axes[0, 0].plot(xs, train, marker="o", label="Train")
    axes[0, 0].plot(xs, val, marker="s", label="Validation")
    axes[0, 0].axhline(final_acc, color="#2ca02c", linestyle="--", label="Final test")
    if selected_epoch:
        best_val = next((r["val_acc"] for r in epochs if r["epoch"] == selected_epoch), None)
        if best_val is not None:
            axes[0, 0].scatter([selected_epoch], [best_val], marker="*", s=110, color="#d62728")
    axes[0, 0].set_title("Learning curve")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(*clamp_ylim(train + val + [final_acc], pad=0.02))
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(loc="best")

    bars = axes[0, 1].bar(baseline_labels, baseline_values, color=["#7f7f7f", "#9467bd", "#1f77b4"], width=0.62)
    axes[0, 1].set_title("Test accuracy")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(*clamp_ylim(baseline_values, pad=0.02))
    axes[0, 1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, baseline_values):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", fontsize=9)

    ep = [r["epoch"] for r in epoch_summary]
    mean_reward = [r["mean_top_reward"] for r in epoch_summary]
    opt_rate = [r["top_optimizer_rate"] for r in epoch_summary]
    axes[1, 0].plot(ep, mean_reward, marker="o", linewidth=2, label="Mean top reward")
    axes[1, 0].bar(ep, opt_rate, width=0.55, alpha=0.35, label="Optimizer top-1 rate")
    axes[1, 0].set_title("Candidate selection")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(loc="best")

    changes = [r["beam_refresh_changed"] for r in epochs]
    axes[1, 1].bar(xs, changes, color="#ff7f0e", alpha=0.75)
    axes[1, 1].set_title("Active prompt changes at beam refresh")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Changed agents")
    axes[1, 1].set_xticks(xs)
    axes[1, 1].set_ylim(0, max(changes + [1]) + 1)
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle("Accuracy-only shared prompt optimization", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_only_overview.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy_run", type=str, default="runs_mmlu_accuracy_only_prompt_opt_retry/shared_accuracy_only_seed42")
    parser.add_argument("--baseline_root", type=str, default="runs_mmlu_subject_balanced_default_size_4way")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    acc_run = Path(args.accuracy_run)
    baseline_root = Path(args.baseline_root)
    out_dir = Path(args.out_dir) if args.out_dir else acc_run.parent / "figures_accuracy_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = load_json(acc_run / "history.json")
    epochs = epoch_rows(history)
    final_acc, final_size, final_record = final_test_acc(history)
    selected_epoch = int(final_record.get("selected_epoch", 0) or 0)

    shared_base_acc, shared_base_size = baseline_test_acc(baseline_root / "shared_baseline_seed42")
    bank_base_acc, bank_base_size = baseline_test_acc(baseline_root / "bank_baseline_seed42")

    updates, refresh_rows = candidate_update_rows(load_jsonl(acc_run / "update_logs.jsonl"))
    update_summary = summarize_by_epoch(updates)

    labels = ["Shared baseline", "Bank baseline", "Accuracy-only"]
    values = [shared_base_acc, bank_base_acc, final_acc]
    plot_learning_curve(out_dir, epochs, final_acc, selected_epoch)
    plot_test_comparison(out_dir, labels, values)
    plot_candidate_behavior(out_dir, updates, update_summary)
    plot_overview(out_dir, epochs, final_acc, selected_epoch, labels, values, update_summary)

    metric_rows = [
        {"metric": "accuracy_only_final_test_acc", "value": final_acc, "size": final_size},
        {"metric": "shared_baseline_test_acc", "value": shared_base_acc, "size": shared_base_size},
        {"metric": "bank_baseline_test_acc", "value": bank_base_acc, "size": bank_base_size},
        {"metric": "delta_vs_shared_baseline", "value": final_acc - shared_base_acc, "size": ""},
        {"metric": "delta_vs_bank_baseline", "value": final_acc - bank_base_acc, "size": ""},
        {"metric": "selected_epoch", "value": selected_epoch, "size": ""},
        {"metric": "best_validation_acc", "value": float(final_record.get("best_validation_score", 0.0) or 0.0), "size": ""},
        {"metric": "epoch_count", "value": len(epochs), "size": ""},
        {"metric": "update_count", "value": len(updates), "size": ""},
    ]
    write_csv(out_dir / "accuracy_metrics_summary.csv", metric_rows)
    write_csv(out_dir / "candidate_epoch_summary.csv", update_summary)

    report = (
        "# Accuracy-only prompt optimization visualization\n\n"
        "This run optimizes only team answer accuracy. Diversity, invalid-rate, and local-validity fields are not used as reward signals here.\n\n"
        f"- Selected epoch: {selected_epoch}\n"
        f"- Best validation accuracy: {float(final_record.get('best_validation_score', 0.0) or 0.0):.4f}\n"
        f"- Final test accuracy: {final_acc:.4f} ({int(round(final_acc * final_size))}/{final_size})\n"
        f"- Shared baseline test accuracy: {shared_base_acc:.4f}; delta: {final_acc - shared_base_acc:+.4f}\n"
        f"- Bank baseline test accuracy: {bank_base_acc:.4f}; delta: {final_acc - bank_base_acc:+.4f}\n"
        f"- Prompt-update candidate groups: {len(updates)}\n\n"
        "Figures:\n\n"
        "- `accuracy_only_overview.png`\n"
        "- `accuracy_learning_curve.png`\n"
        "- `test_accuracy_comparison.png`\n"
        "- `candidate_accuracy_selection.png`\n"
    )
    (out_dir / "accuracy_only_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote figures to {out_dir}")
    print(f"Final test accuracy: {final_acc:.4f} ({int(round(final_acc * final_size))}/{final_size})")
    print(f"Shared baseline: {shared_base_acc:.4f} delta={final_acc - shared_base_acc:+.4f}")
    print(f"Bank baseline: {bank_base_acc:.4f} delta={final_acc - bank_base_acc:+.4f}")


if __name__ == "__main__":
    main()
