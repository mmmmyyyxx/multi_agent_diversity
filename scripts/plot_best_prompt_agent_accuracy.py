import argparse
import csv
import json
from pathlib import Path
from textwrap import shorten
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def answer_correct(agent_row: Dict[str, Any], gold: Any) -> int:
    return int(str(agent_row.get("answer", "")).strip() == str(gold).strip())


def invalid_value(agent_row: Dict[str, Any]) -> int:
    invalid = agent_row.get("invalid", {})
    if isinstance(invalid, dict):
        return int(bool(invalid.get("invalid", 0)))
    return int(bool(invalid))


def get_agent(rows: Iterable[Dict[str, Any]], agent_id: int) -> Iterable[Dict[str, Any]]:
    for row in rows:
        for agent in row.get("agents", []):
            if int(agent.get("agent_id", -1)) == agent_id:
                yield row, agent
                break


def per_agent_metrics(rows: List[Dict[str, Any]], agent_id: int, expected_hash: Optional[str] = None) -> Dict[str, Any]:
    total = 0
    correct = 0
    invalid = 0
    prompt_hashes = set()
    mismatched_hash_count = 0

    for row, agent in get_agent(rows, agent_id):
        total += 1
        correct += answer_correct(agent, row.get("gold"))
        invalid += invalid_value(agent)
        prompt_hash = str(agent.get("prompt_hash", ""))
        if prompt_hash:
            prompt_hashes.add(prompt_hash)
        if expected_hash is not None and prompt_hash != expected_hash:
            mismatched_hash_count += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else None,
        "invalid": invalid,
        "invalid_rate": (invalid / total) if total else None,
        "observed_prompt_hashes": ";".join(sorted(prompt_hashes)),
        "mismatched_hash_count": mismatched_hash_count,
    }


def vote_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(int(row.get("vote_correct", 0) or 0) for row in rows)
    invalid_rates = [float(row.get("invalid_rate", 0.0) or 0.0) for row in rows]
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else None,
        "invalid_rate": float(np.mean(invalid_rates)) if invalid_rates else None,
    }


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value * 100:.1f}%"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "agent_id",
        "best_prompt_hash",
        "best_prompt_accuracy",
        "best_prompt_correct",
        "best_prompt_total",
        "best_prompt_invalid_rate",
        "baseline_agent_accuracy",
        "baseline_agent_correct",
        "baseline_agent_total",
        "baseline_agent_invalid_rate",
        "delta_accuracy",
        "matched_best_prompt_hash",
        "best_prompt_preview",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, Any]], team_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Best Prompt Agent Accuracy",
        "",
        "Scope: per-agent final test accuracy for prompts listed in `shared_beam_seed42/best_prompts.json`, evaluated from `shared_beam_seed42/test_best_predictions.jsonl` and compared with `shared_baseline_seed42/test_epoch1_predictions.jsonl`.",
        "",
        "## Team Vote",
        "",
        "| setting | accuracy | correct/total | invalid_rate |",
        "|---|---:|---:|---:|",
    ]
    for row in team_rows:
        lines.append(
            f"| {row['setting']} | {fmt_pct(row['accuracy'])} | {row['correct']}/{row['total']} | {fmt_pct(row['invalid_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Agents",
            "",
            "| agent | best_prompt | shared_baseline | delta | best correct/total | baseline correct/total | hash_match |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["agent_id"]),
                    fmt_pct(row["best_prompt_accuracy"]),
                    fmt_pct(row["baseline_agent_accuracy"]),
                    fmt_pct(row["delta_accuracy"]),
                    f"{row['best_prompt_correct']}/{row['best_prompt_total']}",
                    f"{row['baseline_agent_correct']}/{row['baseline_agent_total']}",
                    str(row["matched_best_prompt_hash"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_accuracy(out_path: Path, agent_rows: List[Dict[str, Any]], team_rows: List[Dict[str, Any]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 5.4))
    agents = [int(row["agent_id"]) for row in agent_rows]
    x = np.arange(len(agents))
    width = 0.36
    best = np.array([float(row["best_prompt_accuracy"]) for row in agent_rows])
    baseline = np.array([float(row["baseline_agent_accuracy"]) for row in agent_rows])

    best_color = "#4C78A8"
    baseline_color = "#F58518"

    ax.bar(x - width / 2, baseline, width, label="shared_baseline agent", color=baseline_color, alpha=0.88)
    ax.bar(x + width / 2, best, width, label="best_prompt agent", color=best_color, alpha=0.92)

    for i, (b0, b1) in enumerate(zip(baseline, best)):
        ax.text(i - width / 2, b0 + 0.006, f"{b0 * 100:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, b1 + 0.006, f"{b1 * 100:.1f}%", ha="center", va="bottom", fontsize=8)

    team_best = next(row for row in team_rows if row["setting"] == "best_prompt_vote")
    team_baseline = next(row for row in team_rows if row["setting"] == "shared_baseline_vote")
    ax.axhline(float(team_best["accuracy"]), color=best_color, linestyle="--", linewidth=1.5, label="best_prompt vote")
    ax.axhline(float(team_baseline["accuracy"]), color=baseline_color, linestyle=":", linewidth=1.8, label="shared_baseline vote")

    ymin = max(0.0, min(float(np.min(best)), float(np.min(baseline)), float(team_best["accuracy"]), float(team_baseline["accuracy"])) - 0.045)
    ymax = min(1.0, max(float(np.max(best)), float(np.max(baseline)), float(team_best["accuracy"]), float(team_baseline["accuracy"])) + 0.055)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Agent {agent}" for agent in agents])
    ax.set_ylabel("Test accuracy")
    ax.set_title("Per-Agent Final Accuracy: optimized best_prompts vs shared_baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, loc="lower right")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.9, bottom=0.14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-agent best_prompt final accuracy against shared_baseline.")
    parser.add_argument("--root", default="runs_mmlu_subject_balanced_default_size_4way")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures_best_prompt_agents"
    shared_beam_dir = root / "shared_beam_seed42"
    shared_baseline_dir = root / "shared_baseline_seed42"

    best_prompts = read_json(shared_beam_dir / "best_prompts.json")
    best_rows = read_jsonl(shared_beam_dir / "test_best_predictions.jsonl")
    baseline_rows = read_jsonl(shared_baseline_dir / "test_epoch1_predictions.jsonl")

    agent_rows: List[Dict[str, Any]] = []
    for agent in best_prompts.get("agents", []):
        agent_id = int(agent["agent_id"])
        best_hash = str(agent["prompt_hash"])
        best_metrics = per_agent_metrics(best_rows, agent_id, expected_hash=best_hash)
        baseline_metrics = per_agent_metrics(baseline_rows, agent_id)
        best_acc = best_metrics["accuracy"]
        baseline_acc = baseline_metrics["accuracy"]
        matched_hash = best_metrics["mismatched_hash_count"] == 0
        agent_rows.append(
            {
                "agent_id": agent_id,
                "best_prompt_hash": best_hash,
                "best_prompt_accuracy": best_acc,
                "best_prompt_correct": best_metrics["correct"],
                "best_prompt_total": best_metrics["total"],
                "best_prompt_invalid_rate": best_metrics["invalid_rate"],
                "baseline_agent_accuracy": baseline_acc,
                "baseline_agent_correct": baseline_metrics["correct"],
                "baseline_agent_total": baseline_metrics["total"],
                "baseline_agent_invalid_rate": baseline_metrics["invalid_rate"],
                "delta_accuracy": (best_acc - baseline_acc) if best_acc is not None and baseline_acc is not None else None,
                "matched_best_prompt_hash": matched_hash,
                "best_prompt_preview": shorten(" ".join(str(agent.get("prompt", "")).split()), width=160, placeholder="..."),
            }
        )

    best_vote = vote_metrics(best_rows)
    baseline_vote = vote_metrics(baseline_rows)
    team_rows = [
        {"setting": "best_prompt_vote", **best_vote},
        {"setting": "shared_baseline_vote", **baseline_vote},
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = root / "best_prompt_agent_accuracy.csv"
    md_path = root / "best_prompt_agent_accuracy.md"
    png_path = out_dir / "best_prompt_agent_accuracy_vs_shared_baseline.png"

    write_csv(csv_path, agent_rows)
    write_markdown(md_path, agent_rows, team_rows)
    plot_accuracy(png_path, agent_rows, team_rows)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
