import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


SETTING_ORDER = [
    "shared_div",
    "bank_div",
]


def read_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


def read_history(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        return []
    except Exception:
        return []


def read_run_meta(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def moving_average(xs: List[float], window: int) -> List[float]:
    if not xs:
        return []
    w = max(1, int(window))
    out: List[float] = []
    acc = 0.0
    q: List[float] = []
    for x in xs:
        q.append(x)
        acc += x
        if len(q) > w:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out


def split_into_run_segments(records: List[Dict]) -> List[List[Dict]]:
    """
    将 train_step_logs 文件顺序中的记录按 run 切段：
    当 (epoch, step) 回退/重置时视为新 run 开始。
    """
    if not records:
        return []

    segments: List[List[Dict]] = []
    cur: List[Dict] = []
    prev_key: Optional[Tuple[int, int]] = None

    for r in records:
        key = (int(r.get("epoch", 0)), int(r.get("step", 0)))
        if prev_key is not None and key <= prev_key:
            if cur:
                segments.append(cur)
            cur = [r]
        else:
            cur.append(r)
        prev_key = key

    if cur:
        segments.append(cur)
    return segments


def _expected_total_steps_from_meta(run_meta: Dict) -> Optional[int]:
    cfg = run_meta.get("config", {}) if isinstance(run_meta, dict) else {}
    if not isinstance(cfg, dict):
        return None
    try:
        epochs = int(cfg.get("epochs", 0))
        train_size = int(cfg.get("train_size", 0))
        if epochs > 0 and train_size > 0:
            return epochs * train_size
    except Exception:
        return None
    return None


def _is_complete_segment(seg: List[Dict], expected_total_steps: Optional[int]) -> bool:
    if not seg:
        return False
    if expected_total_steps is not None:
        return len(seg) >= expected_total_steps
    return False


def pick_last_complete_segment(run_dir: Path, records: List[Dict]) -> List[Dict]:
    segments = split_into_run_segments(records)
    if not segments:
        return []

    run_meta = read_run_meta(run_dir / "run_meta.json")
    expected_total_steps = _expected_total_steps_from_meta(run_meta)

    if expected_total_steps is not None:
        complete_idxs = [
            i for i, seg in enumerate(segments) if _is_complete_segment(seg, expected_total_steps)
        ]
        if complete_idxs:
            return segments[complete_idxs[-1]]

    # 回退策略：若无法判断完整性，则取最后一段（通常是最后一次运行）。
    return segments[-1]


def collect_training_series(run_dir: Path) -> Tuple[List[int], List[float], List[float]]:
    records = read_jsonl(run_dir / "train_step_logs.jsonl")
    if not records:
        return [], [], []

    records = pick_last_complete_segment(run_dir, records)
    if not records:
        return [], [], []

    # 对单段 run 按 (epoch, step) 排序，映射到全局 step
    records.sort(key=lambda r: (int(r.get("epoch", 0)), int(r.get("step", 0))))

    x: List[int] = []
    homo: List[float] = []
    div: List[float] = []

    g = 0
    for r in records:
        g += 1
        x.append(g)
        homo.append(float(r.get("team_family_homogeneity_rate", 0.0)))
        div.append(float(r.get("team_family_diversity", 0.0)))
    return x, homo, div


def collect_test_epoch_series(run_dir: Path) -> Tuple[List[int], List[float], List[float]]:
    hist = read_history(run_dir / "history.json")
    if not hist:
        return [], [], []

    epochs: List[int] = []
    test_div: List[float] = []
    test_homo: List[float] = []
    for i, rec in enumerate(hist, start=1):
        test = rec.get("test", {}) if isinstance(rec, dict) else {}
        if not isinstance(test, dict):
            test = {}
        epochs.append(int(rec.get("epoch", i)))
        test_div.append(float(test.get("mean_family_diversity", 0.0)))
        test_homo.append(float(test.get("mean_family_homogeneity_rate", 0.0)))
    return epochs, test_div, test_homo


def plot_train_curves(base_dir: Path, out_dir: Path, smooth_window: int):
    fig1, ax1 = plt.subplots(figsize=(11, 5.5))
    fig2, ax2 = plt.subplots(figsize=(11, 5.5))
    fig3, ax3 = plt.subplots(figsize=(11, 5.5))

    for setting in SETTING_ORDER:
        run_dir = base_dir / setting
        x, homo, div = collect_training_series(run_dir)
        if not x:
            continue
        homo_sm = moving_average(homo, smooth_window)
        div_sm = moving_average(div, smooth_window)

        ax1.plot(x, homo_sm, label=setting, linewidth=2)
        ax2.plot(x, div_sm, label=setting, linewidth=2)
        ax3.plot(x, moving_average(homo, 50), label=setting, linewidth=2)

    ax1.set_title(f"Training Homogeneity Dynamics (moving average, window={smooth_window})")
    ax1.set_xlabel("Global Training Step")
    ax1.set_ylabel("Train Homogeneity Rate (smoothed)")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(linestyle="--", alpha=0.35)
    ax1.legend()

    ax2.set_title(f"Training Diversity Dynamics (moving average, window={smooth_window})")
    ax2.set_xlabel("Global Training Step")
    ax2.set_ylabel("Train Diversity (smoothed)")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(linestyle="--", alpha=0.35)
    ax2.legend()

    fig1.tight_layout()
    fig2.tight_layout()
    ax3.set_title("Training Homogeneity Dynamics (moving average, window=50)")
    ax3.set_xlabel("Global Training Step")
    ax3.set_ylabel("Train Homogeneity Rate (MA50)")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(linestyle="--", alpha=0.35)
    ax3.legend()

    fig3.tight_layout()
    p1 = out_dir / "train_homogeneity_dynamics.png"
    p2 = out_dir / "train_diversity_dynamics_mmlu.png"
    p3 = out_dir / "train_homogeneity_dynamics_ma50.png"
    fig1.savefig(p1, dpi=170)
    fig2.savefig(p2, dpi=170)
    fig3.savefig(p3, dpi=170)
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")


def plot_test_epoch_curves(base_dir: Path, out_dir: Path):
    fig1, ax1 = plt.subplots(figsize=(8.5, 5))
    fig2, ax2 = plt.subplots(figsize=(8.5, 5))

    for setting in SETTING_ORDER:
        run_dir = base_dir / setting
        epochs, test_div, test_homo = collect_test_epoch_series(run_dir)
        if not epochs:
            continue
        ax1.plot(epochs, test_homo, marker="o", label=setting, linewidth=2)
        ax2.plot(epochs, test_div, marker="o", label=setting, linewidth=2)

    ax1.set_title("Test Family Homogeneity Rate Across Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Family Homogeneity Rate")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(linestyle="--", alpha=0.35)
    ax1.legend()

    ax2.set_title("Test Family Diversity Across Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Family Diversity")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(linestyle="--", alpha=0.35)
    ax2.legend()

    fig1.tight_layout()
    fig2.tight_layout()
    p1 = out_dir / "test_homogeneity_epoch_dynamics.png"
    p2 = out_dir / "test_diversity_epoch_dynamics_mmlu.png"
    fig1.savefig(p1, dpi=170)
    fig2.savefig(p2, dpi=170)
    plt.close(fig1)
    plt.close(fig2)
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")


def main():
    parser = argparse.ArgumentParser(description="Plot training/test dynamics for shared_div and bank_div settings.")
    parser.add_argument(
        "--runs_root",
        type=str,
        default="runs_experiments",
        help="Directory containing shared_div and bank_div run folders.",
    )
    parser.add_argument("--base_dir", type=str, default="", help="Alias for --runs_root.")
    parser.add_argument("--out_dir", type=str, default="runs_experiments/figures")
    parser.add_argument("--smooth_window", type=int, default=10)
    args = parser.parse_args()

    base_dir = Path(args.base_dir or args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_train_curves(base_dir, out_dir, args.smooth_window)
    plot_test_epoch_curves(base_dir, out_dir)


if __name__ == "__main__":
    main()
