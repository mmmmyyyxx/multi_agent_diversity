#!/usr/bin/env python
"""Validate whether option_contrast labels mask prompt following.

This script samples traces whose target strategy does not include
``option_contrast`` but whose automatic primary family is ``option_contrast``.
It then optionally asks GPT-5.5 to judge prompt following from only:
the original strategy instruction, the question excerpt, and the raw trace.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.utils import infer_strategy_family_major


SYSTEM_PROMPT = """You are an independent evaluator of strategy-prompt following.

You will see one original strategy instruction and one model trace for a multiple-choice question.
You must judge whether the trace actually follows the given strategy instruction.

Important constraints:
- Do not use any taxonomy labels; none are shown to you.
- Do not judge answer correctness except when correctness is needed to understand whether a claimed method was actually used.
- Do not reward mere wording that names the strategy. Look for evidence in the trace.
- Distinguish "followed", "partially followed", and "not followed".
- A trace can compare answer options and still follow another strategy if the requested strategy is clearly the organizing method.

Return only valid JSON:
{
  "followed": true,
  "adherence_score": 4,
  "confidence": 0.0,
  "inferred_actual_method": "short method description",
  "evidence_for_following": ["short evidence span or paraphrase"],
  "evidence_against_following": ["short evidence span or paraphrase"],
  "diagnosis": "judge_taxonomy_likely | model_prompt_likely | ambiguous",
  "rationale": "one concise paragraph"
}

Use this scoring scale:
1 = clearly does not follow the instruction.
2 = mostly does not follow; only superficial or incidental overlap.
3 = partially follows but another method dominates.
4 = mostly follows; minor option-comparison or formatting elements are secondary.
5 = strongly follows; the requested strategy clearly organizes the trace.

Diagnosis rule:
- judge_taxonomy_likely: the trace follows the instruction despite possibly looking like option comparison.
- model_prompt_likely: the trace does not follow or only superficially follows the instruction.
- ambiguous: partial or insufficient evidence.
"""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()}) if rows else ["id"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_prediction_file(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("test_epoch*_predictions.jsonl"))
    if candidates:
        return candidates[-1]
    candidates = sorted(run_dir.glob("test*_predictions.jsonl"))
    return candidates[-1] if candidates else None


def load_probe(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "probe_prompts.json")


def load_model(run_dir: Path) -> str:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return run_dir.name
    meta = read_json(meta_path)
    cfg = meta.get("config", {}) if isinstance(meta.get("config"), dict) else {}
    return str(cfg.get("model", run_dir.name))


def truncate(text: str, max_chars: int) -> str:
    clean = str(text or "").strip()
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[:max_chars] + "\n[TRUNCATED]"


def prediction_by_question(run_dir: Path) -> dict[str, dict[str, Any]]:
    pred_file = find_prediction_file(run_dir)
    if pred_file is None:
        return {}
    return {str(row.get("question_hash", "")): row for row in read_jsonl(pred_file)}


def question_excerpt_by_question(run_dir: Path) -> dict[str, str]:
    excerpts: dict[str, str] = {}
    for path in [run_dir / "reasoning_summary_history.jsonl", run_dir / "test_trace_history.jsonl"]:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            qh = str(row.get("question_hash", ""))
            excerpt = str(row.get("question_excerpt", "")).strip()
            if qh and excerpt and qh not in excerpts:
                excerpts[qh] = excerpt
    return excerpts


def load_candidates(runs_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or "mixed_strategy" not in run_dir.name:
            continue
        if not (run_dir / "probe_prompts.json").exists():
            continue
        trace_path = run_dir / "test_trace_history.jsonl"
        if not trace_path.exists():
            continue
        probe = load_probe(run_dir)
        model = load_model(run_dir)
        pred_by_q = prediction_by_question(run_dir)
        excerpt_by_q = question_excerpt_by_question(run_dir)
        agent_cfg = {}
        for agent in probe.get("agents", []):
            aid = int(agent.get("agent_id", len(agent_cfg)))
            target = agent.get("target_family", [])
            if isinstance(target, str):
                target = [target]
            agent_cfg[aid] = {
                "target_families": [str(x) for x in target],
                "strategy_instruction": str(agent.get("prompt", "")),
            }
        for rec in read_jsonl(trace_path):
            qh = str(rec.get("question_hash", ""))
            pred = pred_by_q.get(qh, {})
            question_excerpt = str(
                pred.get("question_excerpt", "")
                or excerpt_by_q.get(qh, "")
                or rec.get("question_excerpt", "")
            ).strip()
            answers = pred.get("answers", [])
            vote_correct = pred.get("vote_correct", "")
            for agent in rec.get("agents", []):
                aid = int(agent.get("agent_id", -1))
                cfg = agent_cfg.get(aid)
                if not cfg:
                    continue
                targets = cfg["target_families"]
                if "option_contrast" in targets:
                    continue
                primary = str(agent.get("primary_family", ""))
                if primary != "option_contrast":
                    continue
                answer = answers[aid] if isinstance(answers, list) and aid < len(answers) else ""
                target_majors = sorted({infer_strategy_family_major(x) for x in targets})
                candidates.append(
                    {
                        "run_name": run_dir.name,
                        "run_dir": str(run_dir),
                        "model": model,
                        "question_hash": qh,
                        "question_excerpt": question_excerpt,
                        "agent_id": aid,
                        "target_families": targets,
                        "target_majors": target_majors,
                        "strategy_instruction": cfg["strategy_instruction"],
                        "auto_primary": primary,
                        "auto_secondary": str(agent.get("secondary_family", "")),
                        "auto_primary_major": infer_strategy_family_major(primary),
                        "auto_secondary_major": infer_strategy_family_major(str(agent.get("secondary_family", ""))),
                        "answer": answer,
                        "vote_correct": vote_correct,
                        "trace": str(agent.get("trace", "")),
                    }
                )
    return candidates


def sample_candidates(candidates: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        groups[(str(cand["model"]), int(cand["agent_id"]))].append(cand)
    for vals in groups.values():
        rng.shuffle(vals)
    keys = sorted(groups)
    selected: list[dict[str, Any]] = []
    idx = 0
    while len(selected) < sample_size and any(groups.values()):
        key = keys[idx % len(keys)]
        if groups[key]:
            selected.append(groups[key].pop())
        idx += 1
        if idx > sample_size * max(1, len(keys)) * 5:
            break
    if len(selected) < sample_size:
        remaining = [cand for vals in groups.values() for cand in vals]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])
    return selected[:sample_size]


def make_packet_and_key(sampled: list[dict[str, Any]], max_trace_chars: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packets: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for i, cand in enumerate(sampled, start=1):
        blinded_id = f"P3PF{i:04d}"
        packets.append(
            {
                "blinded_id": blinded_id,
                "strategy_instruction": cand["strategy_instruction"],
                "question_excerpt": cand["question_excerpt"],
                "trace": truncate(cand["trace"], max_trace_chars),
                "annotation_fields": {
                    "followed_yes_no": "",
                    "adherence_score_1_to_5": "",
                    "notes_optional": "",
                },
            }
        )
        key = {k: v for k, v in cand.items() if k != "trace"}
        key["blinded_id"] = blinded_id
        key["target_families"] = "|".join(cand["target_families"])
        key["target_majors"] = "|".join(cand["target_majors"])
        key_rows.append(key)
    return packets, key_rows


def extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def normalize_eval(blinded_id: str, model: str, raw_response: str) -> dict[str, Any]:
    obj = extract_json_object(raw_response)
    score = int(round(float(obj.get("adherence_score", obj.get("score", 0)) or 0)))
    score = max(1, min(5, score)) if score else 0
    followed_raw = obj.get("followed", False)
    followed = bool(followed_raw) if isinstance(followed_raw, bool) else str(followed_raw).strip().lower() in {"true", "yes", "1"}
    confidence = float(obj.get("confidence", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    diagnosis = str(obj.get("diagnosis", "")).strip()
    if diagnosis not in {"judge_taxonomy_likely", "model_prompt_likely", "ambiguous"}:
        if score >= 4 or followed:
            diagnosis = "judge_taxonomy_likely"
        elif score <= 2:
            diagnosis = "model_prompt_likely"
        else:
            diagnosis = "ambiguous"
    return {
        "blinded_id": blinded_id,
        "evaluator_model": model,
        "followed": int(followed),
        "adherence_score": score,
        "confidence": confidence,
        "inferred_actual_method": str(obj.get("inferred_actual_method", "")),
        "evidence_for_following": json.dumps(obj.get("evidence_for_following", []), ensure_ascii=False),
        "evidence_against_following": json.dumps(obj.get("evidence_against_following", []), ensure_ascii=False),
        "diagnosis": diagnosis,
        "rationale": str(obj.get("rationale", "")),
        "raw_response": raw_response,
        "parse_ok": int(score > 0),
    }


def build_user_prompt(packet: dict[str, Any], max_trace_chars: int) -> str:
    return "\n".join(
        [
            f"blinded_id: {packet.get('blinded_id', '')}",
            "",
            "Original strategy instruction:",
            str(packet.get("strategy_instruction", "")),
            "",
            "Question excerpt:",
            str(packet.get("question_excerpt", "")),
            "",
            "Model trace:",
            truncate(str(packet.get("trace", "")), max_trace_chars),
            "",
            "Evaluate whether the model trace follows the original strategy instruction. Return JSON only.",
        ]
    )


async def call_openai_chat(
    client: Any,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    retry_sleep: float,
) -> str:
    last_err = None
    for attempt in range(max(1, max_retries)):
        started = time.time()
        try:
            req = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            resp = await asyncio.wait_for(req, timeout=timeout) if timeout > 0 else await req
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            elapsed = time.time() - started
            print(f"[P3PF][WARN] attempt={attempt + 1}/{max_retries} elapsed={elapsed:.2f}s error={exc}", flush=True)
            if attempt + 1 < max_retries:
                await asyncio.sleep(max(0.0, retry_sleep) * (attempt + 1))
    raise RuntimeError(f"GPT prompt-following evaluator failed after {max_retries} attempts: {last_err}")


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("blinded_id", "")): row for row in read_jsonl(path) if row.get("blinded_id")}


async def run_gpt(packet_path: Path, out_jsonl: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
    packets = read_jsonl(packet_path)
    existing = read_existing(out_jsonl) if int(args.resume) else {}
    rows: list[dict[str, Any] | None] = [None] * len(packets)
    sem = asyncio.Semaphore(max(1, int(getattr(args, "eval_parallelism", 12))))

    async def _eval_one(idx: int, packet: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        blinded_id = str(packet.get("blinded_id", ""))
        if blinded_id in existing and int(existing[blinded_id].get("parse_ok", 0) or 0):
            return idx, existing[blinded_id]
        async with sem:
            print(f"[P3PF] {idx + 1}/{len(packets)} blinded_id={blinded_id} model={args.evaluator_model}", flush=True)
            try:
                raw = await call_openai_chat(
                    client,
                    args.evaluator_model,
                    build_user_prompt(packet, args.max_trace_chars),
                    args.temperature,
                    args.max_tokens,
                    args.llm_call_timeout,
                    args.max_retries,
                    args.retry_sleep,
                )
                return idx, normalize_eval(blinded_id, args.evaluator_model, raw)
            except Exception as exc:
                print(f"[P3PF][ERROR] blinded_id={blinded_id} skipped after retries: {exc}", flush=True)
                return idx, {
                    "blinded_id": blinded_id,
                    "evaluator_model": args.evaluator_model,
                    "followed": "",
                    "adherence_score": "",
                    "confidence": "",
                    "inferred_actual_method": "",
                    "evidence_for_following": "[]",
                    "evidence_against_following": "[]",
                    "diagnosis": "api_error",
                    "rationale": "",
                    "raw_response": "",
                    "parse_ok": 0,
                    "error": str(exc),
                }

    tasks = [asyncio.create_task(_eval_one(idx, packet)) for idx, packet in enumerate(packets)]
    for fut in asyncio.as_completed(tasks):
        idx, row = await fut
        rows[idx] = row
        write_jsonl(out_jsonl, [r for r in rows if r is not None])
    return [r for r in rows if r is not None]


def safe_mean(vals: list[Any]) -> float:
    nums = [float(v) for v in vals if v not in {None, ""}]
    return float(statistics.mean(nums)) if nums else 0.0


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    out = []
    for key, vals in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, key)}
        n = len(vals)
        rec.update(
            {
                "n": n,
                "followed_rate": safe_mean([v.get("followed") for v in vals]),
                "mean_adherence_score": safe_mean([v.get("adherence_score") for v in vals]),
                "partial_or_better_rate": safe_mean([int(float(v.get("adherence_score", 0) or 0) >= 3) for v in vals]),
                "judge_taxonomy_likely_rate": safe_mean([int(v.get("diagnosis") == "judge_taxonomy_likely") for v in vals]),
                "model_prompt_likely_rate": safe_mean([int(v.get("diagnosis") == "model_prompt_likely") for v in vals]),
                "ambiguous_rate": safe_mean([int(v.get("diagnosis") == "ambiguous") for v in vals]),
                "mean_confidence": safe_mean([v.get("confidence") for v in vals]),
            }
        )
        out.append(rec)
    return out


def join_key_eval(key_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["blinded_id"]): row for row in key_rows}
    rows = []
    for ev in eval_rows:
        bid = str(ev.get("blinded_id", ""))
        if bid in by_id and int(ev.get("parse_ok", 0) or 0):
            rows.append({**by_id[bid], **ev})
    return rows


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def build_summary_md(sampled_count: int, candidate_count: int, joined: list[dict[str, Any]]) -> str:
    lines = [
        "# P3 Prompt-following GPT-5.5 Validation",
        "",
        "目标：抽样 `target` 不包含 `option_contrast`、但自动 judge 的 primary 标签为 `option_contrast` 的 trace，让 GPT-5.5 只看原始策略指令和 trace，判断模型是否真的遵循了策略指令。",
        "",
        f"- candidate_count: {candidate_count}",
        f"- sampled_count: {sampled_count}",
        f"- evaluated_count: {len(joined)}",
        "",
        "判读规则：",
        "",
        "- GPT-5.5 认为遵循，而自动 judge 判 `option_contrast`：更像 judge/taxonomy 把选项形式过度吸附为 `option_contrast`。",
        "- GPT-5.5 也认为没有遵循：更像模型/prompt 没有稳定诱导目标策略。",
        "- GPT-5.5 认为部分遵循：说明两者都有可能，需要看具体 trace。",
        "",
    ]
    if not joined:
        lines.extend(
            [
                "当前只生成了待评审样本包，还没有 GPT-5.5 评审结果。",
                "",
                "运行评审命令：",
                "",
                "```powershell",
                "python scripts\\run_p3_prompt_following_validation.py --run_gpt 1 --sample_size 776",
                "```",
            ]
        )
        return "\n".join(lines) + "\n"

    overall = summarize(joined, [])
    by_target = summarize(joined, ["agent_id", "target_families"])
    by_model = summarize(joined, ["model", "agent_id", "target_families"])
    lines.extend(["## Overall", ""])
    lines.extend(
        md_table(
            ["n", "followed_rate", "mean_score", "partial_or_better", "judge_taxonomy_likely", "model_prompt_likely", "ambiguous"],
            [
                [
                    overall[0]["n"],
                    overall[0]["followed_rate"],
                    overall[0]["mean_adherence_score"],
                    overall[0]["partial_or_better_rate"],
                    overall[0]["judge_taxonomy_likely_rate"],
                    overall[0]["model_prompt_likely_rate"],
                    overall[0]["ambiguous_rate"],
                ]
            ],
        )
    )
    lines.extend(["", "## By Target", ""])
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
    lines.extend(["", "## By Model And Target", ""])
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
    return "\n".join(lines) + "\n"


def build_summary_md_clean(sampled_count: int, candidate_count: int, joined: list[dict[str, Any]]) -> str:
    lines = [
        "# P3 GPT-5.5 Prompt Following 复核",
        "",
        "目标：抽样 `target` 不包含 `option_contrast`、但自动 judge 的 primary 标签为 `option_contrast` 的 trace，让 GPT-5.5 只看原始策略指令和 trace，判断模型是否真的遵循了策略指令。",
        "",
        f"- candidate_count: {candidate_count}",
        f"- sampled_count: {sampled_count}",
        f"- evaluated_count: {len(joined)}",
        "",
        "判读规则：",
        "",
        "- GPT-5.5 认为遵循，而自动 judge 判 `option_contrast`：更像 judge/taxonomy 把选项形式过度吸附到 `option_contrast`。",
        "- GPT-5.5 也认为没有遵循：更像模型或 prompt 没有稳定诱导目标策略。",
        "- GPT-5.5 认为部分遵循：两种解释都可能，需要看具体 trace。",
        "",
    ]
    if not joined:
        lines.extend(
            [
                "当前只生成了待评审样本包，还没有 GPT-5.5 评审结果。",
                "",
                "运行评审命令：",
                "",
                "```powershell",
                "python scripts\\run_p3_prompt_following_validation.py --runs_root prove_experiments\\p3_analysis_runs --run_gpt 1 --sample_size 776",
                "```",
            ]
        )
        return "\n".join(lines) + "\n"

    overall = summarize(joined, [])
    by_target = summarize(joined, ["agent_id", "target_families"])
    by_model = summarize(joined, ["model", "agent_id", "target_families"])
    lines.extend(["## 总体结果", ""])
    lines.extend(
        md_table(
            ["n", "followed_rate", "mean_score", "partial_or_better", "judge_taxonomy_likely", "model_prompt_likely", "ambiguous"],
            [
                [
                    overall[0]["n"],
                    overall[0]["followed_rate"],
                    overall[0]["mean_adherence_score"],
                    overall[0]["partial_or_better_rate"],
                    overall[0]["judge_taxonomy_likely_rate"],
                    overall[0]["model_prompt_likely_rate"],
                    overall[0]["ambiguous_rate"],
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
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="prove_experiments/runs")
    parser.add_argument("--out_dir", default="prove_experiments/p3_prompt_following_gpt55")
    parser.add_argument("--sample_size", type=int, default=776)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_trace_chars", type=int, default=3500)
    parser.add_argument("--run_gpt", type=int, default=0)
    parser.add_argument("--evaluator_model", default="gpt-5.5")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=900)
    parser.add_argument("--llm_call_timeout", type=float, default=180.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--eval_parallelism", type=int, default=4)
    parser.add_argument("--resume", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(Path(args.runs_root))
    sampled = sample_candidates(candidates, args.sample_size, args.seed)
    packets, key_rows = make_packet_and_key(sampled, args.max_trace_chars)

    packet_path = out_dir / "p3_prompt_following_packet.jsonl"
    key_path = out_dir / "p3_prompt_following_key.csv"
    eval_path = out_dir / "p3_prompt_following_evaluations.jsonl"
    rows_path = out_dir / "p3_prompt_following_analysis_rows.csv"
    summary_path = out_dir / "p3_prompt_following_summary.md"

    write_jsonl(packet_path, packets)
    write_csv(key_path, key_rows)

    eval_rows: list[dict[str, Any]] = []
    if int(args.run_gpt):
        eval_rows = asyncio.run(run_gpt(packet_path, eval_path, args))
    elif eval_path.exists():
        eval_rows = read_jsonl(eval_path)

    joined = join_key_eval(key_rows, eval_rows)
    write_csv(rows_path, joined)
    summary_path.write_text(build_summary_md_clean(len(sampled), len(candidates), joined), encoding="utf-8-sig")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
