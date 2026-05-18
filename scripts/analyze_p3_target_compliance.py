#!/usr/bin/env python
"""Analyze target-strategy compliance for P3 probe runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.utils import infer_strategy_family_major  # noqa: E402


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_prediction_file(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("test_epoch*_predictions.jsonl"))
    if candidates:
        return candidates[-1]
    candidates = sorted(run_dir.glob("test*_predictions.jsonl"))
    return candidates[-1] if candidates else None


def load_targets(run_dir: Path) -> dict[int, list[str]]:
    probe = read_json(run_dir / "probe_prompts.json")
    targets: dict[int, list[str]] = {}
    for agent in probe.get("agents", []):
        aid = int(agent.get("agent_id", len(targets)))
        target = agent.get("target_family", [])
        if isinstance(target, str):
            target = [target]
        targets[aid] = [str(x) for x in target]
    return targets


def infer_probe_kind(run_name: str) -> str:
    text = run_name.lower()
    if "mixed_strategy" in text:
        return "mixed"
    if "same_elimination" in text:
        return "same"
    return "other"


def load_model(run_dir: Path) -> str:
    meta = read_json(run_dir / "run_meta.json")
    cfg = meta.get("config", {}) if isinstance(meta.get("config"), dict) else {}
    return str(cfg.get("model", run_dir.name))


def target_label(targets: list[str]) -> str:
    return "|".join(targets)


def is_answer_correct(answer: Any, gold: Any) -> int | None:
    if gold is None:
        return None
    a = str(answer or "").strip().upper()
    g = str(gold or "").strip().upper()
    if not a or not g:
        return None
    return int(a == g)


def row_records(run_dir: Path) -> list[dict[str, Any]]:
    pred_file = find_prediction_file(run_dir)
    if pred_file is None:
        return []
    targets_by_agent = load_targets(run_dir)
    model = load_model(run_dir)
    run_name = run_dir.name
    probe_kind = infer_probe_kind(run_name)
    rows: list[dict[str, Any]] = []
    for rec in read_jsonl(pred_file):
        primary = rec.get("primary_family_labels", [])
        secondary = rec.get("secondary_family_labels", primary)
        answers = rec.get("answers", [])
        if not isinstance(primary, list):
            primary = []
        if not isinstance(secondary, list):
            secondary = primary
        if not isinstance(answers, list):
            answers = []
        gold = rec.get("gold")
        for aid, targets in targets_by_agent.items():
            p = str(primary[aid]) if aid < len(primary) else ""
            s = str(secondary[aid]) if aid < len(secondary) else p
            ans = answers[aid] if aid < len(answers) else ""
            target_set = set(targets)
            target_majors = {infer_strategy_family_major(x) for x in targets}
            p_major = infer_strategy_family_major(p)
            s_major = infer_strategy_family_major(s)
            exact = int(p in target_set or s in target_set)
            rows.append(
                {
                    "probe_kind": probe_kind,
                    "run_name": run_name,
                    "model": model,
                    "agent_id": aid,
                    "target_label": target_label(targets),
                    "target_major_label": "|".join(sorted(target_majors)),
                    "primary": p,
                    "secondary": s,
                    "primary_major": p_major,
                    "secondary_major": s_major,
                    "exact_hit": exact,
                    "primary_exact_hit": int(p in target_set),
                    "secondary_exact_hit": int(s in target_set),
                    "same_major_primary_hit": int(exact or p_major in target_majors),
                    "same_major_any_hit": int(exact or p_major in target_majors or s_major in target_majors),
                    "agent_answer_correct": is_answer_correct(ans, gold),
                }
            )
    return rows


def mean(values: list[Any]) -> float:
    nums = [float(x) for x in values if x is not None]
    return float(statistics.mean(nums)) if nums else 0.0


def summarize(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, vals in sorted(groups.items()):
        rec = {k: v for k, v in zip(group_keys, key)}
        primary_counts = Counter(v["primary"] for v in vals)
        secondary_counts = Counter(v["secondary"] for v in vals)
        primary_major_counts = Counter(v["primary_major"] for v in vals)
        top_primary, top_primary_n = primary_counts.most_common(1)[0]
        top_secondary, top_secondary_n = secondary_counts.most_common(1)[0]
        top_major, top_major_n = primary_major_counts.most_common(1)[0]
        n = len(vals)
        rec.update(
            {
                "n": n,
                "exact_hit_rate": mean([v["exact_hit"] for v in vals]),
                "primary_exact_hit_rate": mean([v["primary_exact_hit"] for v in vals]),
                "secondary_exact_hit_rate": mean([v["secondary_exact_hit"] for v in vals]),
                "same_major_primary_hit_rate": mean([v["same_major_primary_hit"] for v in vals]),
                "same_major_any_hit_rate": mean([v["same_major_any_hit"] for v in vals]),
                "agent_answer_acc": mean([v["agent_answer_correct"] for v in vals]),
                "top_primary": top_primary,
                "top_primary_share": top_primary_n / n,
                "top_secondary": top_secondary,
                "top_secondary_share": top_secondary_n / n,
                "top_primary_major": top_major,
                "top_primary_major_share": top_major_n / n,
                "primary_counts_json": json.dumps(dict(primary_counts.most_common()), ensure_ascii=False),
                "secondary_counts_json": json.dumps(dict(secondary_counts.most_common()), ensure_ascii=False),
            }
        )
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x).replace("|", "\\|")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    return lines


def build_markdown(overall: list[dict[str, Any]], by_model: list[dict[str, Any]]) -> str:
    mixed = [r for r in overall if r["probe_kind"] == "mixed"]
    same = [r for r in overall if r["probe_kind"] == "same"]
    lines = [
        "# P3 目标策略命中率拆解",
        "",
        "本分析把自动 taxonomy judge 的标签按目标策略、agent 和模型拆开。`exact_hit` 表示 primary 或 secondary leaf 标签精确命中 prompt 指定的目标 leaf；`same_major_any_hit` 表示 primary 或 secondary 至少落入目标 major family。",
        "",
        "## 指标中文含义",
        "",
        "| 指标/列名 | 中文含义 | 读数方向 |",
        "|---|---|---|",
        "| `agent` / `agent_id` | 五个 agent 的编号，0-4 分别对应 prompt 文件中的五条策略指令。 | 用于定位是哪条策略指令。 |",
        "| `target leaf` / `target_label` | prompt 显式要求 agent 使用的细粒度目标策略标签。多个标签用 `\\|` 连接，命中任意一个都算 exact hit。 | 这是目标定义，不是分数。 |",
        "| `target major` / `target_major_label` | 目标 leaf 映射到的粗粒度 major family。 | 用于计算宽松命中。 |",
        "| `n` | 该行统计的 agent-question 样本数。四模型、100 题时，每个 mixed target 通常为 400。 | 越大越稳定。 |",
        "| `exact` / `exact_hit_rate` | primary 或 secondary leaf 是否精确命中目标 leaf 的比例。 | 越高表示越严格地符合指定细策略。 |",
        "| `same-major(any)` / `same_major_any_hit_rate` | primary 或 secondary 的 major family 是否落入目标 major family 的比例。 | 越高表示至少落入相近粗策略。 |",
        "| `top primary` | 该组样本中最常见的 primary leaf 标签。 | 用来判断模型实际最常表现出的策略。 |",
        "| `top primary share` | 最常见 primary leaf 所占比例。 | 越高表示输出越被单一策略形态支配。 |",
        "| `top secondary` | 该组样本中最常见的 secondary leaf 标签。 | 用来观察辅助策略或次要策略。 |",
        "| `agent acc` / `agent_answer_acc` | 单个 agent 自己答案的准确率，不是五 agent 投票准确率。 | 越高表示该 agent 答题越准。 |",
        "| `primary_exact_hit_rate` | 只看 primary leaf 是否精确命中目标 leaf。 | 比 `exact` 更严格。 |",
        "| `secondary_exact_hit_rate` | 只看 secondary leaf 是否精确命中目标 leaf。 | 用于判断目标策略是否退到次策略位置。 |",
        "| `same_major_primary_hit_rate` | 只看 primary major 是否命中目标 major。 | 用于判断主策略大类是否对齐。 |",
        "| `top_primary_major` | 最常见的 primary major family。 | 用来判断输出主要落在哪个粗策略大类。 |",
        "| `top_primary_major_share` | 最常见 primary major family 所占比例。 | 越高表示粗策略越集中。 |",
        "| `primary_counts_json` | primary leaf 标签的完整计数字典。 | 用于追查被哪些标签吸走。 |",
        "| `secondary_counts_json` | secondary leaf 标签的完整计数字典。 | 用于追查目标策略是否作为次策略出现。 |",
        "",
        "注意：这里的 `exact` 是“策略树标签是否命中目标 leaf”，不是“答案是否正确”。答案正确率看 `agent acc`。",
        "",
        "## Mixed 策略目标整体情况",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "agent",
                "target leaf",
                "target major",
                "n",
                "exact",
                "same-major(any)",
                "top primary",
                "top primary share",
                "top secondary",
                "agent acc",
            ],
            [
                [
                    r["agent_id"],
                    r["target_label"],
                    r["target_major_label"],
                    r["n"],
                    r["exact_hit_rate"],
                    r["same_major_any_hit_rate"],
                    r["top_primary"],
                    r["top_primary_share"],
                    r["top_secondary"],
                    r["agent_answer_acc"],
                ]
                for r in mixed
            ],
        )
    )
    lines.extend(["", "## Same-elimination 对照整体情况", ""])
    lines.extend(
        markdown_table(
            [
                "agent",
                "target leaf",
                "n",
                "exact",
                "same-major(any)",
                "top primary",
                "top primary share",
            ],
            [
                [
                    r["agent_id"],
                    r["target_label"],
                    r["n"],
                    r["exact_hit_rate"],
                    r["same_major_any_hit_rate"],
                    r["top_primary"],
                    r["top_primary_share"],
                ]
                for r in same
            ],
        )
    )
    mixed_model = [r for r in by_model if r["probe_kind"] == "mixed"]
    lines.extend(["", "## Mixed 策略按模型拆解", ""])
    lines.extend(
        markdown_table(
            [
                "model",
                "agent",
                "target leaf",
                "exact",
                "same-major(any)",
                "top primary",
                "top primary share",
            ],
            [
                [
                    r["model"],
                    r["agent_id"],
                    r["target_label"],
                    r["exact_hit_rate"],
                    r["same_major_any_hit_rate"],
                    r["top_primary"],
                    r["top_primary_share"],
                ]
                for r in mixed_model
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 主要读法",
            "",
            "- P3 的 mixed 条件五个 target 分别来自五个不同 major，所以 `same-major(any)` 可以解释为粗粒度策略大类命中。",
            "- mixed 的 team-level 多样性提高，不等于每个 agent 都稳定落在指定 target leaf。需要同时看总体 diversity、exact hit、same-major hit 和 top primary。",
            "- 如果 mixed 的 top primary 大量集中在 `option_contrast`，说明 MMLU 多选题和当前 judge primary 规则仍有选项比较吸附；这更像 prompt 可执行性、题型适配和 judge evidence 标准的共同问题，不能只解释为模型完全不听指令。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_markdown_clean(overall: list[dict[str, Any]], by_model: list[dict[str, Any]]) -> str:
    """Build the Chinese report used by the official P3 analysis."""
    mixed = [r for r in overall if r["probe_kind"] == "mixed"]
    same = [r for r in overall if r["probe_kind"] == "same"]
    mixed_model = [r for r in by_model if r["probe_kind"] == "mixed"]
    same_model = [r for r in by_model if r["probe_kind"] == "same"]

    lines = [
        "# P3 目标策略命中分析",
        "",
        "本分析把自动 taxonomy judge 给出的 `primary` / `secondary` 策略标签，与 prompt 中显式指定的目标策略进行对齐。",
        "这里的命中率衡量的是“trace 被判到目标策略标签或目标主类的比例”，不是答案正确率。",
        "",
        "## 指标中文含义",
        "",
        "| 指标 | 中文含义 | 解读方向 |",
        "|---|---|---|",
        "| `agent_id` | 五个 agent 的编号，0-4 对应 prompt 文件中的五条策略指令。 | 用来定位是哪条策略指令。 |",
        "| `target_label` / `target leaf` | prompt 显式要求使用的细粒度策略标签。多个标签用 `|` 连接。 | 这是目标定义，不是分数。 |",
        "| `target_major_label` / `target major` | 目标 leaf 所属的主策略类。 | 用于判断粗粒度策略是否对齐。 |",
        "| `n` | 该行统计的 agent-question 样本数。 | 越大越稳定。 |",
        "| `exact_hit_rate` / `exact` | `primary` 或 `secondary` leaf 精确命中目标 leaf 的比例。 | 越高表示越严格遵循指定细策略。 |",
        "| `primary_exact_hit_rate` | 只看 `primary` leaf 是否精确命中目标 leaf。 | 比 `exact` 更严格。 |",
        "| `secondary_exact_hit_rate` | 只看 `secondary` leaf 是否精确命中目标 leaf。 | 用来判断目标策略是否退到次策略位置。 |",
        "| `same_major_primary_hit_rate` | 只看 `primary` 所属主类是否命中目标主类。 | 衡量主策略方向是否对齐。 |",
        "| `same_major_any_hit_rate` / `same-major(any)` | `primary` 或 `secondary` 所属主类是否命中目标主类。 | 衡量较宽松的粗粒度策略遵循。 |",
        "| `top_primary` | 该组样本中最常见的 `primary` leaf。 | 用来观察实际 trace 最常被判成哪类策略。 |",
        "| `top_primary_share` | 最常见 `primary` leaf 的占比。 | 越高说明越被单一策略形态支配。 |",
        "| `top_secondary` | 该组样本中最常见的 `secondary` leaf。 | 用来观察辅助或次要策略。 |",
        "| `top_primary_major` | 最常见 `primary` 所属主类。 | 用来观察粗粒度策略落点。 |",
        "| `agent_answer_acc` / `agent acc` | 单个 agent 自己答案的准确率。 | 与策略命中不同，衡量答题对错。 |",
        "| `primary_counts_json` | `primary` leaf 的完整计数字典。 | 用于追查被哪些标签吸走。 |",
        "| `secondary_counts_json` | `secondary` leaf 的完整计数字典。 | 用于追查目标策略是否作为次策略出现。 |",
        "",
        "## P3 目标策略设计确认",
        "",
        "| agent | 目标 leaf | 所属主类 | 策略含义 |",
        "|---|---|---|---|",
        "| 0 | `distractor_elimination` | `mmlu_option_semantics` | 逐项排除干扰项，保留最符合题干的选项。 |",
        "| 1 | `rule_or_principle_application` | `mmlu_domain_reasoning` | 先识别领域规则、定理、原则或机制，再把规则应用到题干。 |",
        "| 2 | `decomposition` | `representation_formalization` | 把题干拆成事实、约束和子问题，再合并得到答案。 |",
        "| 3 | `case_analysis` | `logical_proof` | 枚举相关条件、情形或分支，逐一检验。 |",
        "| 4 | `edge_case_analysis` | `optimization_boundary_meta` | 检查边界条件、限定词、例外或极端情形。 |",
        "",
        "这五个目标 leaf 分别属于五个不同主类，因此 P3 的 mixed 条件可以干净地检验“显式策略 prompt 是否能提高跨主类策略多样性”。",
        "",
        "## Mixed 策略目标整体情况",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "agent",
                "target leaf",
                "target major",
                "n",
                "exact",
                "same-major(any)",
                "top primary",
                "top primary share",
                "top secondary",
                "agent acc",
            ],
            [
                [
                    r["agent_id"],
                    r["target_label"],
                    r["target_major_label"],
                    r["n"],
                    r["exact_hit_rate"],
                    r["same_major_any_hit_rate"],
                    r["top_primary"],
                    r["top_primary_share"],
                    r["top_secondary"],
                    r["agent_answer_acc"],
                ]
                for r in mixed
            ],
        )
    )
    lines.extend(["", "## Same-elimination 对照整体情况", ""])
    lines.extend(
        markdown_table(
            [
                "agent",
                "target leaf",
                "n",
                "exact",
                "same-major(any)",
                "top primary",
                "top primary share",
                "agent acc",
            ],
            [
                [
                    r["agent_id"],
                    r["target_label"],
                    r["n"],
                    r["exact_hit_rate"],
                    r["same_major_any_hit_rate"],
                    r["top_primary"],
                    r["top_primary_share"],
                    r["agent_answer_acc"],
                ]
                for r in same
            ],
        )
    )
    lines.extend(["", "## Mixed 策略按模型拆解", ""])
    lines.extend(
        markdown_table(
            [
                "model",
                "agent",
                "target leaf",
                "target major",
                "exact",
                "same-major(any)",
                "top primary",
                "top primary share",
                "agent acc",
            ],
            [
                [
                    r["model"],
                    r["agent_id"],
                    r["target_label"],
                    r["target_major_label"],
                    r["exact_hit_rate"],
                    r["same_major_any_hit_rate"],
                    r["top_primary"],
                    r["top_primary_share"],
                    r["agent_answer_acc"],
                ]
                for r in mixed_model
            ],
        )
    )
    lines.extend(["", "## Same-elimination 对照按模型拆解", ""])
    lines.extend(
        markdown_table(
            [
                "model",
                "agent",
                "target leaf",
                "exact",
                "same-major(any)",
                "top primary",
                "top primary share",
                "agent acc",
            ],
            [
                [
                    r["model"],
                    r["agent_id"],
                    r["target_label"],
                    r["exact_hit_rate"],
                    r["same_major_any_hit_rate"],
                    r["top_primary"],
                    r["top_primary_share"],
                    r["agent_answer_acc"],
                ]
                for r in same_model
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 主要读法",
            "",
            "- `exact` 高，说明 trace 的细粒度 leaf 标签更接近目标策略；`same-major(any)` 高，说明至少在主类层面落到了目标策略方向。",
            "- mixed 条件下如果 `exact` 较低但 `same-major(any)` 较高，说明模型可能没有稳定执行到指定 leaf，但已经发生了较粗粒度的策略转向。",
            "- 如果 `top_primary` 大量集中到 `option_contrast`，说明 MMLU 多选题形式和 judge 的 primary 规则仍会把不同策略吸到选项比较形态；这需要结合 GPT-5.5 normal taxonomy judge 与 prompt-following 复核一起解释。",
            "- P3 的关键证据不是单个 leaf 是否完全可控，而是 same 与 mixed 在 team-level diversity、major diversity 和同质性上的系统差异。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="prove_experiments/p3_analysis_runs")
    parser.add_argument("--out_dir", default="prove_experiments/p3_target_compliance")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if "mixed_strategy" not in run_dir.name and "same_elimination" not in run_dir.name:
            continue
        if not (run_dir / "probe_prompts.json").exists():
            continue
        rows.extend(row_records(run_dir))

    overall = summarize(rows, ["probe_kind", "agent_id", "target_label", "target_major_label"])
    by_model = summarize(rows, ["probe_kind", "model", "agent_id", "target_label", "target_major_label"])

    write_csv(out_dir / "p3_target_compliance_detail.csv", rows)
    write_csv(out_dir / "p3_target_compliance_overall.csv", overall)
    write_csv(out_dir / "p3_target_compliance_by_model.csv", by_model)
    (out_dir / "p3_target_compliance.md").write_text(build_markdown_clean(overall, by_model), encoding="utf-8-sig")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
