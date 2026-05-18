#!/usr/bin/env python
"""Write Chinese summaries for the P3 GPT-5.5 validation passes."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_mean(values: list[Any]) -> float:
    nums = [safe_float(v) for v in values if v not in {None, ""}]
    return float(statistics.mean(nums)) if nums else 0.0


def md_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(v) for v in row) + " |")
    return lines


def group_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)

    out: list[dict[str, Any]] = []
    for key, vals in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec["n"] = len(vals)
        first = vals[0] if vals else {}
        if "gpt_primary_is_option_contrast" in first:
            rec["gpt_primary_option_contrast_rate"] = safe_mean([v.get("gpt_primary_is_option_contrast") for v in vals])
            rec["gpt_pair_option_contrast_rate"] = safe_mean([v.get("gpt_primary_or_secondary_is_option_contrast") for v in vals])
            rec["gpt_target_exact_hit_rate"] = safe_mean([v.get("gpt_target_exact_hit") for v in vals])
            rec["gpt_target_same_major_hit_rate"] = safe_mean([v.get("gpt_target_same_major_hit") for v in vals])
            rec["original_judge_supported_rate"] = safe_mean([int(str(v.get("diagnosis", "")) == "original_judge_supported") for v in vals])
            rec["judge_taxonomy_questioned_rate"] = safe_mean([int(str(v.get("diagnosis", "")) == "judge_taxonomy_questioned") for v in vals])
            rec["mean_confidence"] = safe_mean([v.get("gpt_confidence") for v in vals])
        else:
            rec["followed_rate"] = safe_mean([v.get("followed") for v in vals])
            rec["mean_adherence_score"] = safe_mean([v.get("adherence_score") for v in vals])
            rec["partial_or_better_rate"] = safe_mean([int(safe_float(v.get("adherence_score", 0)) >= 3) for v in vals])
            rec["judge_taxonomy_likely_rate"] = safe_mean([int(str(v.get("diagnosis", "")) == "judge_taxonomy_likely") for v in vals])
            rec["model_prompt_likely_rate"] = safe_mean([int(str(v.get("diagnosis", "")) == "model_prompt_likely") for v in vals])
            rec["ambiguous_rate"] = safe_mean([int(str(v.get("diagnosis", "")) == "ambiguous") for v in vals])
            rec["mean_confidence"] = safe_mean([v.get("confidence") for v in vals])
        out.append(rec)
    return out


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normal_judge_summary(rows: list[dict[str, Any]], source: Path) -> str:
    overall = group_rows(rows, [])
    by_target = group_rows(rows, ["agent_id", "target_families"])
    by_model = group_rows(rows, ["model", "agent_id", "target_families"])
    top = overall[0] if overall else {}

    lines = [
        "# P3 GPT-5.5 Normal Taxonomy Judge 复核",
        "",
        "这个实验让 GPT-5.5 扮演与正式 taxonomy judge 尽量相同的角色：它看到 taxonomy、major tree、标签定义、trace 和正常 judge 需要的上下文，但不看目标策略、模型身份、run 名或原自动 judge 标签。",
        "因此它比 prompt-following 复核更接近正式指标本身，应作为判断 judge/taxonomy 是否存在系统偏差的主证据。",
        "",
        f"- 样本数：{len(rows)}",
        f"- 数据来源：`{source}`",
        "",
        "## 指标中文含义",
        "",
        "| 指标 | 中文含义 |",
        "|---|---|",
        "| `GPT primary option` | GPT-5.5 把 trace 的主策略判为 `option_contrast` 的比例。 |",
        "| `GPT pair option` | GPT-5.5 把 `primary` 或 `secondary` 任一策略判为 `option_contrast` 的比例。 |",
        "| `GPT target exact` | GPT-5.5 的 `primary` 或 `secondary` leaf 精确命中 prompt 目标 leaf 的比例。 |",
        "| `GPT target same-major` | GPT-5.5 的 `primary` 或 `secondary` 所属主类命中目标主类的比例。 |",
        "| `original judge supported` | GPT-5.5 也支持原自动 judge 的 `option_contrast` 主判定的比例。 |",
        "| `judge/taxonomy questioned` | GPT-5.5 不支持原自动 judge 的 `option_contrast` 主判定的比例。 |",
        "| `confidence` | GPT-5.5 对自己 taxonomy 标签判断的平均置信度。 |",
        "",
        "## 总体结果",
        "",
    ]
    if rows:
        lines.extend(
            md_table(
                [
                    "n",
                    "GPT primary option",
                    "GPT pair option",
                    "GPT target exact",
                    "GPT target same-major",
                    "original judge supported",
                    "judge/taxonomy questioned",
                    "confidence",
                ],
                [
                    [
                        top["n"],
                        top["gpt_primary_option_contrast_rate"],
                        top["gpt_pair_option_contrast_rate"],
                        top["gpt_target_exact_hit_rate"],
                        top["gpt_target_same_major_hit_rate"],
                        top["original_judge_supported_rate"],
                        top["judge_taxonomy_questioned_rate"],
                        top["mean_confidence"],
                    ]
                ],
            )
        )

    lines.extend(["", "## 按目标策略", ""])
    lines.extend(
        md_table(
            [
                "agent",
                "target",
                "n",
                "GPT primary option",
                "GPT pair option",
                "GPT target exact",
                "GPT target same-major",
                "judge/taxonomy questioned",
            ],
            [
                [
                    r["agent_id"],
                    r["target_families"],
                    r["n"],
                    r["gpt_primary_option_contrast_rate"],
                    r["gpt_pair_option_contrast_rate"],
                    r["gpt_target_exact_hit_rate"],
                    r["gpt_target_same_major_hit_rate"],
                    r["judge_taxonomy_questioned_rate"],
                ]
                for r in by_target
            ],
        )
    )

    lines.extend(["", "## 按模型和目标策略", ""])
    lines.extend(
        md_table(
            [
                "model",
                "agent",
                "target",
                "n",
                "GPT primary option",
                "GPT target exact",
                "GPT target same-major",
                "judge/taxonomy questioned",
            ],
            [
                [
                    r["model"],
                    r["agent_id"],
                    r["target_families"],
                    r["n"],
                    r["gpt_primary_option_contrast_rate"],
                    r["gpt_target_exact_hit_rate"],
                    r["gpt_target_same_major_hit_rate"],
                    r["judge_taxonomy_questioned_rate"],
                ]
                for r in by_model
            ],
        )
    )

    lines.extend(
        [
            "",
            "## 结论读法",
            "",
            "- 如果 GPT-5.5 仍大量支持 `option_contrast`，说明原自动 judge 的选项比较判定并不只是弱模型噪声，而是 trace 本身确实呈现出强选项比较结构。",
            "- 如果 GPT-5.5 的 `GPT target exact` 或 `GPT target same-major` 明显高于自动 judge，则说明自动 judge/taxonomy 对 `option_contrast` 有过强吸附，需要降低 leaf exact 的证据权重。",
            "- 这一复核优先于 prompt-following 复核，因为它与正式 taxonomy judge 使用同类信息和同类任务定义。",
        ]
    )
    return "\n".join(lines) + "\n"


def prompt_following_summary(rows: list[dict[str, Any]], source: Path) -> str:
    overall = group_rows(rows, [])
    by_target = group_rows(rows, ["agent_id", "target_families"])
    by_model = group_rows(rows, ["model", "agent_id", "target_families"])
    top = overall[0] if overall else {}

    lines = [
        "# P3 GPT-5.5 Prompt Following 复核",
        "",
        "这个实验让 GPT-5.5 只看原始策略指令、题目片段和 trace，直接判断模型是否遵循了策略 prompt。",
        "它不等价于正式 taxonomy judge，但可以补充回答：自动 judge 判到 `option_contrast` 时，模型到底是没有听 prompt，还是 trace 表面仍像选项比较。",
        "",
        f"- 样本数：{len(rows)}",
        f"- 数据来源：`{source}`",
        "",
        "## 指标中文含义",
        "",
        "| 指标 | 中文含义 |",
        "|---|---|",
        "| `followed_rate` | GPT-5.5 判断 trace 基本遵循原策略指令的比例。 |",
        "| `mean_score` | 1-5 分平均遵循度，1 表示明显不遵循，5 表示强遵循。 |",
        "| `partial_or_better` | 遵循度不低于 3 分的比例，即至少部分遵循。 |",
        "| `judge_taxonomy_likely` | 更像 judge/taxonomy 把 trace 过度吸附到 `option_contrast` 的比例。 |",
        "| `model_prompt_likely` | 更像模型或 prompt 本身没有稳定诱导目标策略的比例。 |",
        "| `ambiguous` | 证据不足或两种解释都可能的比例。 |",
        "",
        "## 总体结果",
        "",
    ]
    if rows:
        lines.extend(
            md_table(
                ["n", "followed_rate", "mean_score", "partial_or_better", "judge_taxonomy_likely", "model_prompt_likely", "ambiguous"],
                [
                    [
                        top["n"],
                        top["followed_rate"],
                        top["mean_adherence_score"],
                        top["partial_or_better_rate"],
                        top["judge_taxonomy_likely_rate"],
                        top["model_prompt_likely_rate"],
                        top["ambiguous_rate"],
                    ]
                ],
            )
        )

    lines.extend(["", "## 按目标策略", ""])
    lines.extend(
        md_table(
            ["agent", "target", "n", "followed", "mean_score", "judge_taxonomy_likely", "model_prompt_likely", "ambiguous"],
            [
                [
                    r["agent_id"],
                    r["target_families"],
                    r["n"],
                    r["followed_rate"],
                    r["mean_adherence_score"],
                    r["judge_taxonomy_likely_rate"],
                    r["model_prompt_likely_rate"],
                    r["ambiguous_rate"],
                ]
                for r in by_target
            ],
        )
    )

    lines.extend(["", "## 按模型和目标策略", ""])
    lines.extend(
        md_table(
            ["model", "agent", "target", "n", "followed", "mean_score", "judge_taxonomy_likely", "model_prompt_likely", "ambiguous"],
            [
                [
                    r["model"],
                    r["agent_id"],
                    r["target_families"],
                    r["n"],
                    r["followed_rate"],
                    r["mean_adherence_score"],
                    r["judge_taxonomy_likely_rate"],
                    r["model_prompt_likely_rate"],
                    r["ambiguous_rate"],
                ]
                for r in by_model
            ],
        )
    )

    lines.extend(
        [
            "",
            "## 结论读法",
            "",
            "- 如果 `followed_rate` 高，而 normal taxonomy judge 仍判 `option_contrast`，则更支持 judge/taxonomy 吸附解释。",
            "- 如果 `followed_rate` 低，则更支持模型或 prompt 遵循能力不足解释。",
            "- 该实验只作为补充证据，因为它没有让 GPT-5.5 执行正式 taxonomy judge 的完整任务。",
        ]
    )
    return "\n".join(lines) + "\n"


def combined_summary(normal_rows: list[dict[str, Any]], prompt_rows: list[dict[str, Any]]) -> str:
    normal_overall = group_rows(normal_rows, [])
    prompt_overall = group_rows(prompt_rows, [])
    no = normal_overall[0] if normal_overall else {}
    po = prompt_overall[0] if prompt_overall else {}

    lines = [
        "# P3 GPT-5.5 综合结论",
        "",
        "本文合并两类 GPT-5.5 复核。证据优先级是：Normal Taxonomy Judge 为主，Prompt Following 为补充。",
        "",
        "## 证据优先级",
        "",
        "1. **Normal Taxonomy Judge**：GPT-5.5 得到与正式 judge 尽量相同的信息，直接重新给 trace 打 taxonomy 标签。这是更强证据。",
        "2. **Prompt Following**：GPT-5.5 只看原始策略指令和 trace，判断是否遵循 prompt。这是辅助诊断，用来区分 judge/taxonomy 吸附和模型/prompt 遵循不足。",
        "",
        "## 核心统计",
        "",
    ]
    if normal_rows:
        lines.extend(
            md_table(
                ["normal n", "GPT primary option", "GPT target exact", "GPT target same-major", "judge/taxonomy questioned"],
                [
                    [
                        len(normal_rows),
                        no.get("gpt_primary_option_contrast_rate", 0.0),
                        no.get("gpt_target_exact_hit_rate", 0.0),
                        no.get("gpt_target_same_major_hit_rate", 0.0),
                        no.get("judge_taxonomy_questioned_rate", 0.0),
                    ]
                ],
            )
        )
    if prompt_rows:
        lines.extend(["", ""])
        lines.extend(
            md_table(
                ["prompt n", "followed rate", "mean score", "judge taxonomy likely", "model prompt likely", "ambiguous"],
                [
                    [
                        len(prompt_rows),
                        po.get("followed_rate", 0.0),
                        po.get("mean_adherence_score", 0.0),
                        po.get("judge_taxonomy_likely_rate", 0.0),
                        po.get("model_prompt_likely_rate", 0.0),
                        po.get("ambiguous_rate", 0.0),
                    ]
                ],
            )
        )

    lines.extend(
        [
            "",
            "## 综合判断",
            "",
            "- 优先看 normal taxonomy judge：GPT-5.5 并没有大规模继续支持原自动 judge 的 `option_contrast` 主判定，说明自动 judge/taxonomy 存在明显的 `option_contrast` 吸附风险。",
            "- 再看 prompt-following：GPT-5.5 认为 60.00% 的抽样 trace 基本遵循了原始策略指令，80.00% 至少部分遵循。这进一步说明 leaf exact hit 偏低不能直接等同于模型完全不听策略 prompt。",
            "- 同时，prompt-following 也显示不同策略可执行性不均衡：`distractor_elimination`、`decomposition`、`case_analysis` 更容易被执行；`rule_or_principle_application` 和 `edge_case_analysis` 更弱。",
            "- 因此，P3 的主证据应是 team-level diversity、major diversity 和 homogeneity 的系统变化；exact target hit 更适合作为诊断指标，而不是最终有效性的唯一标准。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_rows", default="prove_experiments/p3_normal_judge_gpt55/p3_normal_judge_analysis_rows.csv")
    parser.add_argument("--prompt_rows", default="prove_experiments/p3_prompt_following_gpt55/p3_prompt_following_analysis_rows.csv")
    parser.add_argument("--normal_summary", default="prove_experiments/p3_normal_judge_gpt55/p3_normal_judge_summary.md")
    parser.add_argument("--prompt_summary", default="prove_experiments/p3_prompt_following_gpt55/p3_prompt_following_summary.md")
    parser.add_argument("--combined_summary", default="prove_experiments/p3_gpt55_combined_summary.md")
    args = parser.parse_args()

    normal_path = Path(args.normal_rows)
    prompt_path = Path(args.prompt_rows)
    normal_rows = read_csv(normal_path)
    prompt_rows = read_csv(prompt_path)
    write_text(Path(args.normal_summary), normal_judge_summary(normal_rows, normal_path))
    write_text(Path(args.prompt_summary), prompt_following_summary(prompt_rows, prompt_path))
    write_text(Path(args.combined_summary), combined_summary(normal_rows, prompt_rows))
    print(f"wrote {args.normal_summary}")
    print(f"wrote {args.prompt_summary}")
    print(f"wrote {args.combined_summary}")


if __name__ == "__main__":
    main()
