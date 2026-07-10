import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: Dict[str, str], key: str):
    try:
        value = row.get(key, "")
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def _save_panel(rows: List[Dict[str, str]], out_dir: Path, filename: str, title: str, specs: List[Dict[str, str]]):
    if not rows:
        return
    labels = [r.get("run_name") or r.get("setting", "") for r in rows]
    fig, axes = plt.subplots(len(specs), 1, figsize=(max(8, len(labels) * 1.2), 2.8 * len(specs)))
    if len(specs) == 1:
        axes = [axes]
    for ax, spec in zip(axes, specs):
        key = spec["key"]
        vals = [_float(r, key) for r in rows]
        xs = list(range(len(rows)))
        ax.bar(xs, [v if v is not None else 0.0 for v in vals])
        ax.set_title(spec.get("title", key))
        ax.set_ylabel(spec.get("ylabel", ""))
        if spec.get("ylim"):
            ax.set_ylim(*spec["ylim"])
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def plot(rows: List[Dict[str, str]], out_dir: Path):
    _save_panel(
        rows,
        out_dir,
        "experiment_embedding_outcome_panel.png",
        "MAD Outcome Diagnostics",
        [
            {"key": "latest_test_vote_acc", "title": "Test Vote Accuracy", "ylabel": "Accuracy", "ylim": (0.0, 1.0)},
            {"key": "latest_test_embedding_diversity", "title": "Test Embedding Diversity", "ylabel": "Diversity", "ylim": (0.0, 1.0)},
            {"key": "latest_test_embedding_overlap", "title": "Test Embedding Overlap", "ylabel": "Overlap", "ylim": (0.0, 1.0)},
            {"key": "latest_test_invalid_rate", "title": "Test Invalid Rate", "ylabel": "Rate", "ylim": (0.0, 1.0)},
        ],
    )
    _save_panel(
        rows,
        out_dir,
        "experiment_beam_reward_panel.png",
        "Evolutionary Beam Candidate Reward Components",
        [
            {"key": "reward", "title": "Candidate Reward", "ylabel": "Reward", "ylim": (0.0, 1.0)},
            {"key": "target_agent_accuracy", "title": "Target Agent Accuracy", "ylabel": "Accuracy", "ylim": (0.0, 1.0)},
            {"key": "coverage_delta", "title": "Oracle Coverage Delta", "ylabel": "Delta"},
            {"key": "useful_diversity", "title": "Useful Diversity", "ylabel": "Score", "ylim": (0.0, 1.0)},
            {"key": "invalid_guard_pass_rate", "title": "Invalid Guard Pass Rate", "ylabel": "Rate", "ylim": (0.0, 1.0)},
            {"key": "team_accuracy", "title": "Team Vote Accuracy Diagnostic", "ylabel": "Mean", "ylim": (0.0, 1.0)},
            {"key": "student_final_failure_rate", "title": "Final Student Failure Rate", "ylabel": "Rate", "ylim": (0.0, 1.0)},
            {"key": "student_retry_recovery_rate", "title": "Student Retry Recovery Rate", "ylabel": "Rate", "ylim": (0.0, 1.0)},
            {"key": "invalid_rate", "title": "Invalid Rate", "ylabel": "Rate", "ylim": (0.0, 1.0)},
            {"key": "beam_refresh_count", "title": "Beam Refresh Count", "ylabel": "Count"},
            {"key": "active_prompt_changed_count", "title": "Refresh Prompt Changes", "ylabel": "Count"},
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Plot trace-overlap beam experiment results.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "figures"
    plot(load_rows(csv_path), out_dir)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
