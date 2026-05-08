import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def plot_ablation(rows: List[Dict[str, str]], out_dir: Path):
    # 按 setting 字母顺序展示 A/B/C/D
    order = [
        "A_shared_no_div",
        "B_shared_div",
        "C_bank_no_div",
        "D_bank_div",
    ]
    rows = sorted(rows, key=lambda r: order.index(r["setting"]) if r.get("setting") in order else 999)

    settings = [r.get("setting", "") for r in rows]
    prompt_div = [to_float(r.get("final_prompt_diversity_rate", "0")) for r in rows]
    prompt_cos_div = [to_float(r.get("final_prompt_cosine_diversity", "0")) for r in rows]
    trace_cos_div = [to_float(r.get("final_trace_cosine_diversity", "0")) for r in rows]
    summary_cos_div = [to_float(r.get("final_reasoning_summary_cosine_diversity", "0")) for r in rows]
    trace_cos_sim = [to_float(r.get("final_trace_cosine_similarity", "0")) for r in rows]
    summary_cos_sim = [to_float(r.get("final_reasoning_summary_cosine_similarity", "0")) for r in rows]
    train_div = [to_float(r.get("final_train_mean_family_diversity", "0")) for r in rows]
    test_div = [to_float(r.get("final_test_mean_family_diversity", "0")) for r in rows]
    train_homo = [to_float(r.get("final_train_mean_family_homogeneity_rate", "0")) for r in rows]
    test_homo = [to_float(r.get("final_test_mean_family_homogeneity_rate", "0")) for r in rows]
    test_vote_acc = [to_float(r.get("final_test_vote_acc", "0")) for r in rows]

    # 图0：最后 prompt 多样性
    fig0, ax0 = plt.subplots(figsize=(10, 5))
    ax0.bar(settings, prompt_div, color="#6c8ebf")
    ax0.set_ylim(0.0, 1.0)
    ax0.set_ylabel("Prompt Diversity")
    ax0.set_title("Ablation: Final Prompt Diversity by Setting")
    ax0.grid(axis="y", linestyle="--", alpha=0.3)
    fig0.tight_layout()
    fig0_path = out_dir / "ablation_prompt_diversity.png"
    fig0.savefig(fig0_path, dpi=160)
    plt.close(fig0)

    # 图0b：最后 prompt 余弦多样性
    fig0b, ax0b = plt.subplots(figsize=(10, 5))
    ax0b.bar(settings, prompt_cos_div, color="#59a14f")
    ax0b.set_ylim(0.0, 1.0)
    ax0b.set_ylabel("Prompt Cosine Diversity")
    ax0b.set_title("Ablation: Final Prompt Cosine Diversity by Setting")
    ax0b.grid(axis="y", linestyle="--", alpha=0.3)
    fig0b.tight_layout()
    fig0b_path = out_dir / "ablation_prompt_cosine_diversity.png"
    fig0b.savefig(fig0b_path, dpi=160)
    plt.close(fig0b)

    # 图0c：最后轨迹与 reasoning summary 余弦多样性
    fig0c, ax0c = plt.subplots(figsize=(10, 5))
    x = list(range(len(settings)))
    width = 0.36
    ax0c.bar([i - width / 2 for i in x], trace_cos_div, width=width, color="#e15759", label="Full trace")
    ax0c.bar([i + width / 2 for i in x], summary_cos_div, width=width, color="#59a14f", label="Reasoning summary")
    ax0c.set_xticks(x)
    ax0c.set_xticklabels(settings, rotation=20)
    ax0c.set_ylim(0.0, 1.0)
    ax0c.set_ylabel("Cosine Diversity")
    ax0c.set_title("Ablation: Final Trace and Summary Cosine Diversity by Setting")
    ax0c.legend()
    ax0c.grid(axis="y", linestyle="--", alpha=0.3)
    fig0c.tight_layout()
    fig0c_path = out_dir / "ablation_trace_summary_cosine_diversity.png"
    fig0c.savefig(fig0c_path, dpi=160)
    plt.close(fig0c)

    # 图0d：最后轨迹与 reasoning summary 余弦相似度
    fig0d, ax0d = plt.subplots(figsize=(10, 5))
    ax0d.bar([i - width / 2 for i in x], trace_cos_sim, width=width, color="#e15759", label="Full trace")
    ax0d.bar([i + width / 2 for i in x], summary_cos_sim, width=width, color="#59a14f", label="Reasoning summary")
    ax0d.set_xticks(x)
    ax0d.set_xticklabels(settings, rotation=20)
    ax0d.set_ylim(0.0, 1.0)
    ax0d.set_ylabel("Cosine Similarity")
    ax0d.set_title("Ablation: Final Trace and Summary Cosine Similarity by Setting")
    ax0d.legend()
    ax0d.grid(axis="y", linestyle="--", alpha=0.3)
    fig0d.tight_layout()
    fig0d_path = out_dir / "ablation_trace_summary_cosine_similarity.png"
    fig0d.savefig(fig0d_path, dpi=160)
    plt.close(fig0d)

    # 图1：轨迹多样性
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    width = 0.36
    ax1.bar([i - width / 2 for i in x], train_div, width=width, label="Train Diversity")
    ax1.bar([i + width / 2 for i in x], test_div, width=width, label="Test Diversity")
    ax1.set_xticks(x)
    ax1.set_xticklabels(settings, rotation=20)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Diversity")
    ax1.set_title("Ablation: Train/Test Family Diversity by Setting")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    fig1.tight_layout()
    fig1_path = out_dir / "ablation_diversity.png"
    fig1.savefig(fig1_path, dpi=160)
    plt.close(fig1)

    # 图2：同质化率（越低越好）
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(settings, train_homo, marker="o", label="Train Homogeneity Rate")
    ax2.plot(settings, test_homo, marker="o", label="Test Homogeneity Rate")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Metric Value")
    ax2.set_title("Ablation: Family Homogeneity Rate Across Settings")
    ax2.grid(linestyle="--", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2_path = out_dir / "ablation_homogeneity_rate.png"
    fig2.savefig(fig2_path, dpi=160)
    plt.close(fig2)

    # 图3：测试集 vote accuracy
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.bar(settings, test_vote_acc, color="#76b7b2")
    ax3.set_ylim(0.0, 1.0)
    ax3.set_ylabel("Test Vote Accuracy")
    ax3.set_title("Ablation: Test Vote Accuracy by Setting")
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    fig3.tight_layout()
    fig3_path = out_dir / "ablation_vote_acc.png"
    fig3.savefig(fig3_path, dpi=160)
    plt.close(fig3)

    print(f"Saved: {fig0_path}")
    print(f"Saved: {fig0b_path}")
    print(f"Saved: {fig0c_path}")
    print(f"Saved: {fig0d_path}")
    print(f"Saved: {fig1_path}")
    print(f"Saved: {fig2_path}")
    print(f"Saved: {fig3_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot family diversity and homogeneity metrics for 4 ablation settings.")
    parser.add_argument("--csv", type=str, required=True, help="Path to ablation_runs.csv")
    parser.add_argument("--dataset", type=str, default="", help="Dataset filter, e.g. mmlu; ignored if CSV has no dataset column")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for figures")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows = load_rows(csv_path)
    has_dataset_col = bool(rows) and ("dataset" in rows[0])
    if args.dataset and has_dataset_col:
        rows = [r for r in rows if r.get("dataset", "").lower() == args.dataset.lower()]
    if not rows:
        if args.dataset and has_dataset_col:
            raise ValueError(f"No rows found for dataset={args.dataset} in {csv_path}")
        raise ValueError(f"No rows found in {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_ablation(rows, out_dir)


if __name__ == "__main__":
    main()
