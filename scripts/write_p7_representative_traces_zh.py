#!/usr/bin/env python
"""Write a Chinese P7 representative-trace report.

The raw traces are kept verbatim because they are the experimental evidence.
Chinese summaries are added around them so the report is readable in VS Code.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_IDS = ["P7G0013", "P7G0053", "P7G0003", "P7G0049"]


TAG_TRANSLATIONS = {
    "answer-only": "只给答案",
    "no-explicit-reasoning": "没有显式推理",
    "option-by-option-elimination": "逐项排除",
    "surface-level-relevance-check": "浅层相关性检查",
    "uncertainty-based-speculation": "基于不确定性的猜测",
    "incomplete-degenerate-repetition": "不完整且重复退化",
    "structured-option-analysis": "结构化选项分析",
    "physics-conceptual-explanation": "物理概念解释",
    "Maxwell/electromagnetic-wave reasoning": "麦克斯韦方程/电磁波推理",
    "attempted-generic-breakdown": "尝试做一般性拆解",
    "direct concept recall": "直接概念回忆",
    "answer-choice elimination": "答案选项排除",
    "direct rule application": "直接套用规则",
    "Newton's Third Law": "牛顿第三定律",
    "Newton's third law": "牛顿第三定律",
    "equal-and-opposite reaction": "大小相等方向相反的反作用力",
    "projectile-motion decomposition": "抛体运动分解",
    "vertical-motion kinematics": "竖直方向运动学",
    "formula substitution": "公式代入",
    "general kinematic equation": "一般运动学方程",
    "option contrast": "选项对比",
    "all-of-the-above heuristic": "all of the above 启发式判断",
    "holistic answer choice selection": "整体式选项选择",
}


CASE_NOTES: dict[str, dict[str, Any]] = {
    "P7G0013": {
        "title": "例1：GPT-5.5 认为多样性高，策略树也高",
        "one_line": "这是一个“二者一致为高”的样本，但高多样性里混入了无推理和退化输出。",
        "gpt_rationale_zh": (
            "GPT-5.5 认为这里至少有几类不同方法：A1 只给答案，A2 做浅层选项排除和相关性检查，"
            "A4 使用电场、磁场与光的物理机制解释，A3 和 A5 则是不完整、重复、退化的推理。"
            "因此它把这些 trace 判成较高方法多样性。"
        ),
        "tree_reading_zh": (
            "策略树也给高分，因为五个 agent 被分到 option_contrast、distractor_elimination、"
            "causal_mechanism_reasoning、decomposition 等不同标签。这个例子支持指标能感知明显行为差异，"
            "但也暴露一个问题：退化输出和 answer-only 会被算进“多样性”，不一定等价于有效策略多样性。"
        ),
        "agent_notes": {
            "A1": "中文解读：没有推理过程，只直接给出答案 D。",
            "A2": "中文解读：尝试逐项检查选项是否与电磁学相关，但推理很浅且语言退化，最后给 B。",
            "A3": "中文解读：围绕电磁关系做含糊猜测，随后大量重复，属于退化 trace。",
            "A4": "中文解读：逐项分析选项，并明确用“光是电磁波，振荡电场和磁场相互生成”来支持 D。",
            "A5": "中文解读：开头像是想分解问题，但很快进入大量重复词，属于退化输出。",
        },
    },
    "P7G0053": {
        "title": "例2：策略树分数高，但 GPT-5.5 认为几乎是同一种方法（牛顿第三定律）",
        "one_line": "这是一个关键反例：策略树高分，但真实解题方法几乎完全相同。",
        "gpt_rationale_zh": (
            "GPT-5.5 认为所有 agent 都在直接应用牛顿第三定律：手给木板 3000 N，木板给手大小相等、"
            "方向相反的 3000 N。选项排除只是表达形式上的附加步骤，不能构成真正不同的方法。"
        ),
        "tree_reading_zh": (
            "策略树把其中一个 trace 标成 causal_mechanism_reasoning，把其他 trace 标成 "
            "rule_or_principle_application，并在 secondary 标签里出现 option_contrast / "
            "answer_to_stem_backward_check，于是分数偏高。但从人类/GPT-5.5 视角看，核心策略都是"
            "同一条物理定律直接套用。这个例子说明当前 taxonomy 对“原则应用、机制解释、选项排除”"
            "的边界可能过细，容易把同一种解法的表述差异算成策略差异。"
        ),
        "agent_notes": {
            "A1": "中文解读：说明牛顿第三定律，再逐项排除，答案 B。",
            "A2": "中文解读：同样说明作用力和反作用力大小相等，再看选项，答案 B。",
            "A3": "中文解读：同样用牛顿第三定律推出木板对手也是 3000 N，答案 B。",
            "A4": "中文解读：同样用牛顿第三定律，并补充木板断裂不改变相互作用力大小，答案 B。",
            "A5": "中文解读：同样用牛顿第三定律和简单选项排除，答案 B。",
        },
    },
    "P7G0003": {
        "title": "例3：策略树分数高，但 GPT-5.5 认为几乎是同一种方法（水平抛公式）",
        "one_line": "这是另一个关键反例：decomposition、equation_solving、algebraic_derivation 标签不同，但解法实质相同。",
        "gpt_rationale_zh": (
            "GPT-5.5 认为所有 agent 都使用同一方法：把水平抛运动分解为竖直方向运动，指出初始竖直速度为 0，"
            "忽略水平速度，然后用恒加速度公式 y = 1/2 g t^2 算出 19.6 m。A5 写了更一般的运动学公式，"
            "但这只是表达更完整，不是另一种解题策略。"
        ),
        "tree_reading_zh": (
            "策略树高分来自 decomposition、equation_solving、algebraic_derivation 等 leaf 标签差异。"
            "但这些标签在这道题上实际描述的是同一公式解法的不同书写层次。这个例子说明，在数学/物理计算题上，"
            "当前 taxonomy 可能把“分解问题、列方程、代数推导、直接计算”拆得太细，导致策略树分数高估真实方法多样性。"
        ),
        "agent_notes": {
            "A1": "中文解读：指出竖直初速度为 0，用 y = 1/2 g t^2 计算，答案 D。",
            "A2": "中文解读：同样分解竖直运动，用同一公式计算，答案 D。",
            "A3": "中文解读：同样说明水平速度与高度无关，用同一公式计算，答案 D。",
            "A4": "中文解读：同样列 y = 1/2 g t^2 并代入，答案 D。",
            "A5": "中文解读：写成更一般的 y = v0y t + 1/2 g t^2，但 v0y=0，本质还是同一解法，答案 D。",
        },
    },
    "P7G0049": {
        "title": "例4：策略树分数低，但 GPT-5.5 认为存在一定行为/方法差异",
        "one_line": "这是反方向反例：策略树认为都属于 option_contrast，但 GPT-5.5 看到了推理深度和 answer-only 的差异。",
        "gpt_rationale_zh": (
            "GPT-5.5 认为 A1 和 A4 是相似的逐项排除，围绕对话中的概念问题判断选项；A2 是另一种较浅的整体判断，"
            "主要依赖“all of the above 最全面”；A3 和 A5 则只是给答案，没有推理。因此整体有一定方法/行为差异，"
            "但主要有推理的 trace 之间仍然重合较多。"
        ),
        "tree_reading_zh": (
            "策略树给低分，因为 primary 标签全是 option_contrast，major diversity 甚至为 0。"
            "但这个标签没有区分“完整逐项排除”“浅层 all-of-the-above 猜测”“answer-only”。"
            "这说明策略树在 option_contrast 内部可能过粗，尤其对推理深度、空答案、退化答案的区分不足。"
        ),
        "agent_notes": {
            "A1": "中文解读：逐项解释 A/B/D 为什么不合适，认为 C 最能回应苏格拉底的质疑。",
            "A2": "中文解读：没有逐项细证，主要以“all of the above 最全面”为理由选 D，较浅。",
            "A3": "中文解读：只给最终答案 D，没有推理过程。",
            "A4": "中文解读：较详细逐项分析 A/B/C/D，最后支持 C。",
            "A5": "中文解读：只给最终答案 D，没有推理过程。",
        },
    },
}


def read_jsonl_by_id(path: Path, id_field: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[str(obj[id_field])] = obj
    return rows


def read_csv_by_id(path: Path, id_field: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row[id_field]: row for row in csv.DictReader(f)}


def find_run_record(run_dir: Path, question_hash: str) -> dict[str, Any]:
    path = run_dir / "reasoning_summary_history.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("question_hash") == question_hash:
                return obj
    return {}


def find_prediction_record(run_dir: Path, question_hash: str) -> dict[str, Any]:
    path = run_dir / "test_epoch1_predictions.jsonl"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("question_hash") == question_hash:
                return obj
    return {}


def parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def fmt_float(value: Any, ndigits: int = 4) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return "NA"


def compact_answer(answer: Any) -> str:
    text = str(answer).strip()
    if not text:
        return "空"
    if len(text) <= 20 and re.fullmatch(r"[A-Za-z0-9(). _:-]+", text):
        return text
    return "无法稳定解析或输出退化"


def complete_strategy_record(run_record: dict[str, Any], pred_record: dict[str, Any]) -> dict[str, Any]:
    record = dict(run_record)
    for key in (
        "primary_family_labels",
        "secondary_family_labels",
        "answers",
        "vote_answer",
        "vote_correct",
        "question_excerpt",
    ):
        if key in pred_record:
            record[key] = pred_record[key]

    agents = run_record.get("agents") or []
    if not record.get("primary_family_labels") and agents:
        record["primary_family_labels"] = [agent.get("primary_family", "") for agent in agents]
    if not record.get("secondary_family_labels") and agents:
        record["secondary_family_labels"] = [agent.get("secondary_family", "") for agent in agents]
    return record


def format_tag_list_zh(tags: Any) -> str:
    if not isinstance(tags, list):
        return f"`{tags}`"
    chunks = []
    for tag in tags:
        text = str(tag)
        zh = TAG_TRANSLATIONS.get(text)
        if zh:
            chunks.append(f"`{text}`（{zh}）")
        else:
            chunks.append(f"`{text}`")
    return "，".join(chunks)


def write_case(
    lines: list[str],
    case_id: str,
    index: int,
    row: dict[str, str],
    packet: dict[str, Any],
    eval_row: dict[str, Any],
    run_record: dict[str, Any],
    question_excerpt: str,
) -> None:
    note = CASE_NOTES[case_id]
    lines.extend(
        [
            f"## {note['title']}",
            "",
            f"**一句话结论：**{note['one_line']}",
            "",
            f"- 盲评 ID: `{case_id}`",
            f"- 分桶: `{row.get('bucket', '')}`",
            f"- 来源 run: `{row.get('run_name', '')}`",
            f"- 题目 hash: `{row.get('question_hash', '')}`",
            f"- 策略树多样性: `{fmt_float(row.get('team_family_diversity'))}`",
            f"- major 策略树多样性: `{fmt_float(row.get('team_major_family_diversity'))}`",
            f"- trace 文本多样性: `{fmt_float(row.get('trace_token_cosine_diversity'))}`",
            f"- GPT-5.5 盲评分数: `{eval_row.get('gpt_method_diversity_score', row.get('gpt_method_diversity_score', 'NA'))}/5`",
            f"- GPT-5.5 认为的方法数: `{eval_row.get('gpt_distinct_methods_count', row.get('gpt_distinct_methods_count', 'NA'))}`",
            f"- GPT-5.5 置信度: `{eval_row.get('gpt_confidence', row.get('gpt_confidence', 'NA'))}`",
            f"- 投票是否正确: `{fmt_float(row.get('vote_correct'), 1)}`",
            "",
        ]
    )
    if question_excerpt:
        lines.extend(["题目摘录：", "", "```text", question_excerpt, "```", ""])

    lines.extend(
        [
            "GPT-5.5 盲评理由（中文）：",
            "",
            note["gpt_rationale_zh"],
            "",
            "策略树视角解读：",
            "",
            note["tree_reading_zh"],
            "",
        ]
    )

    tags = parse_json_maybe(eval_row.get("gpt_coarse_method_tags_by_agent"))
    if isinstance(tags, dict):
        lines.append("GPT-5.5 给出的粗粒度方法标签：")
        lines.append("")
        for alias in sorted(tags):
            lines.append(f"- `{alias}`: {format_tag_list_zh(tags[alias])}")
        lines.append("")

    primary = run_record.get("primary_family_labels", [])
    secondary = run_record.get("secondary_family_labels", [])
    answers = run_record.get("answers", [])
    lines.extend(
        [
            "策略树标签与答案摘要：",
            "",
            f"- primary_family_labels: `{json.dumps(primary, ensure_ascii=False)}`",
            f"- secondary_family_labels: `{json.dumps(secondary, ensure_ascii=False)}`",
            f"- answers 摘要: `{json.dumps([compact_answer(x) for x in answers], ensure_ascii=False)}`",
            "",
            "下面是五个 agent 的真实完整 trace。注意：trace 保留原始英文输出；每条 trace 前的“中文解读”是为方便阅读添加的，不参与原实验打分。",
            "",
        ]
    )

    for trace_obj in packet.get("traces", []):
        alias = trace_obj.get("agent_alias", "")
        agent_idx = int(alias[1:]) - 1 if re.fullmatch(r"A\d+", alias) else None
        label = ""
        if agent_idx is not None and agent_idx < len(primary):
            label = f"primary=`{primary[agent_idx]}`"
            if agent_idx < len(secondary):
                label += f", secondary=`{secondary[agent_idx]}`"
            if agent_idx < len(answers):
                label += f", answer=`{compact_answer(answers[agent_idx])}`"
        lines.extend(
            [
                f"### {case_id} / {alias}",
                "",
                note["agent_notes"].get(alias, "中文解读：无。"),
                "",
            ]
        )
        if label:
            lines.extend([f"策略树记录：{label}", ""])
        lines.extend(["原始 trace：", "", "```text", str(trace_obj.get("trace", "")).rstrip(), "```", ""])


def build_report(root: Path, ids: list[str]) -> str:
    p7_dir = root / "prove_experiments" / "p7_gpt55_blind"
    rows = read_csv_by_id(p7_dir / "p7_gpt55_analysis_rows.csv", "blinded_id")
    packets = read_jsonl_by_id(p7_dir / "p7_blind_annotation_packet.jsonl", "blinded_id")
    evals = read_jsonl_by_id(p7_dir / "p7_gpt55_evaluations.jsonl", "blinded_id")

    lines = [
        "# P7 盲评代表样本完整 Trace（中文解读版）",
        "",
        "这个文件包含 4 组真实的 P7 盲评样本，用来直观说明 GPT-5.5 认为的“真实方法多样性”和策略树分数之间的关系。",
        "",
        "重要说明：原始 trace 是实验中模型真实输出，通常是英文。为了不改变证据，本文件保留原文完整 trace；中文内容是对盲评理由、策略树标签和每条 trace 的解释。",
        "",
        "阅读说明：",
        "",
        "- `strategy_tree_diversity` / `策略树多样性`：基于 leaf 策略标签计算的多样性分数。",
        "- `major_tree_diversity` / `major 策略树多样性`：将 leaf 标签映射到 major family 后的多样性。",
        "- `trace_text_diversity` / `trace 文本多样性`：原始 trace 文本的 token-cosine 多样性。",
        "- `GPT-5.5 盲评分数`：GPT-5.5 对真实方法多样性的 1-5 分判断。",
        "",
    ]

    for idx, case_id in enumerate(ids, start=1):
        row = rows.get(case_id)
        packet = packets.get(case_id)
        eval_row = evals.get(case_id, {})
        if row is None or packet is None:
            raise KeyError(f"missing P7 case data for {case_id}")
        run_dir = root / row["run_dir"]
        run_record = find_run_record(run_dir, row["question_hash"])
        pred_record = find_prediction_record(run_dir, row["question_hash"])
        run_record = complete_strategy_record(run_record, pred_record)
        question_excerpt = str(run_record.get("question_excerpt", "")).strip()
        write_case(lines, case_id, idx, row, packet, eval_row, run_record, question_excerpt)

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--out",
        default="prove_experiments/p7_gpt55_blind/p7_representative_full_traces.md",
        help="Output Markdown path.",
    )
    parser.add_argument("--ids", default=",".join(DEFAULT_IDS), help="Comma-separated P7 IDs.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    text = build_report(root, ids)
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8-sig")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
