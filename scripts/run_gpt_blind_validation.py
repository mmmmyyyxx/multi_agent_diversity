import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in [ROOT, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from prepare_human_blind_validation import _bucket_groups, _collect_groups, _sample, _write_packet  # noqa: E402
from prove_experiment_utils import (  # noqa: E402
    bootstrap_mean_ci,
    read_jsonl,
    safe_float,
    safe_mean,
    spearman_corr,
    write_csv,
)


SYSTEM_PROMPT = """You are an independent blind evaluator of reasoning-method diversity.

You will see only anonymized traces from several agents solving the same problem. You must not infer from model identity, prompt wording, experiment condition, answer correctness, or automatic labels; none are shown to you.

Judge whether the agents used genuinely different reasoning methods, not whether they used different wording, length, verbosity, or formatting.

Score method diversity on this scale:
1 = same method; differences are mainly wording, length, or ordering.
2 = mostly same method with small variations.
3 = mixed, but methods still overlap substantially.
4 = several agents use clearly different methods.
5 = highly diverse; agents approach the problem through clearly different reasoning paths.

Return only valid JSON with these fields:
{
  "method_diversity_score": 1,
  "confidence": 0.0,
  "distinct_methods_count": 1,
  "coarse_method_tags_by_agent": {"A1": ["..."]},
  "rationale": "one concise paragraph"
}
"""


def _truncate_trace(text: str, max_chars: int) -> str:
    clean = str(text or "").strip()
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[:max_chars] + "\n[TRUNCATED]"


def _build_user_prompt(packet: Dict[str, Any], max_trace_chars: int) -> str:
    lines = [f"blinded_id: {packet.get('blinded_id', '')}", "", "Anonymized traces:"]
    traces = packet.get("traces", [])
    if not isinstance(traces, list):
        traces = []
    for item in traces:
        if not isinstance(item, dict):
            continue
        alias = str(item.get("agent_alias", "A?"))
        trace = _truncate_trace(str(item.get("trace", "")), max_trace_chars)
        lines.append(f"\n[{alias}]\n{trace}")
    lines.append("\nEvaluate only the diversity of reasoning methods across these traces. Return JSON only.")
    return "\n".join(lines)


def _extract_json_object(text: str) -> Dict[str, Any]:
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
            obj = json.loads(raw[start:end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_eval(blinded_id: str, model: str, raw_response: str) -> Dict[str, Any]:
    obj = _extract_json_object(raw_response)
    score = int(round(safe_float(obj.get("method_diversity_score", obj.get("score", 0.0)))))
    score = max(1, min(5, score)) if score else 0
    confidence = max(0.0, min(1.0, safe_float(obj.get("confidence", 0.0))))
    distinct = int(round(safe_float(obj.get("distinct_methods_count", 0.0))))
    tags = obj.get("coarse_method_tags_by_agent", {})
    if not isinstance(tags, dict):
        tags = {}
    return {
        "blinded_id": blinded_id,
        "evaluator_model": model,
        "gpt_method_diversity_score": score,
        "gpt_confidence": confidence,
        "gpt_distinct_methods_count": distinct,
        "gpt_coarse_method_tags_by_agent": json.dumps(tags, ensure_ascii=False),
        "gpt_rationale": str(obj.get("rationale", "")),
        "raw_response": raw_response,
        "parse_ok": int(score > 0),
    }


async def _call_openai_chat(
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
            print(f"[GPT-EVAL][WARN] attempt={attempt + 1}/{max_retries} elapsed={elapsed:.2f}s error={exc}", flush=True)
            if attempt + 1 < max_retries:
                await asyncio.sleep(max(0.0, retry_sleep) * (attempt + 1))
    raise RuntimeError(f"GPT evaluator failed after {max_retries} attempts: {last_err}")


def _read_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(r.get("blinded_id", "")): r for r in rows if str(r.get("blinded_id", ""))}


async def _run_gpt_evaluator(packet_path: Path, out_jsonl: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set; GPT-5.5 blind evaluation cannot run.")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
    packets = read_jsonl(packet_path)
    existing = _read_existing(out_jsonl) if int(args.resume) else {}
    rows: List[Dict[str, Any]] = []
    for idx, packet in enumerate(packets, start=1):
        blinded_id = str(packet.get("blinded_id", ""))
        if blinded_id in existing and int(existing[blinded_id].get("parse_ok", 0) or 0):
            rows.append(existing[blinded_id])
            continue
        user_prompt = _build_user_prompt(packet, args.max_trace_chars)
        print(f"[GPT-EVAL] {idx}/{len(packets)} blinded_id={blinded_id} model={args.evaluator_model}", flush=True)
        raw = await _call_openai_chat(
            client,
            args.evaluator_model,
            user_prompt,
            args.temperature,
            args.max_tokens,
            args.llm_call_timeout,
            args.max_retries,
            args.retry_sleep,
        )
        row = _normalize_eval(blinded_id, args.evaluator_model, raw)
        rows.append(row)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for item in rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return rows


def _key_by_id(key_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("blinded_id", "")): row for row in key_rows}


def _analysis_rows(key_rows: List[Dict[str, Any]], eval_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    key = _key_by_id(key_rows)
    rows = []
    for ev in eval_rows:
        bid = str(ev.get("blinded_id", ""))
        if bid not in key:
            continue
        rows.append({**key[bid], **ev})
    return rows


def _analyze(rows: List[Dict[str, Any]], bootstrap_iterations: int, seed: int) -> Dict[str, Any]:
    parsed = [r for r in rows if safe_float(r.get("gpt_method_diversity_score")) > 0]
    if not parsed:
        return {"matched_count": 0}
    scores = [safe_float(r["gpt_method_diversity_score"]) for r in parsed]
    corr_strategy = spearman_corr([r["team_family_diversity"] for r in parsed], scores)
    corr_major = spearman_corr([r["team_major_family_diversity"] for r in parsed], scores)
    corr_text = spearman_corr([r["trace_token_cosine_diversity"] for r in parsed], scores)
    high = [safe_float(r["gpt_method_diversity_score"]) for r in parsed if str(r.get("bucket")) in {"high_strategy", "low_text_high_strategy"}]
    low = [safe_float(r["gpt_method_diversity_score"]) for r in parsed if str(r.get("bucket")) in {"low_strategy", "high_text_low_strategy"}]
    delta_vals = [h - l for h, l in zip(high, low)]
    delta_ci = bootstrap_mean_ci(delta_vals, iterations=bootstrap_iterations, seed=seed) if delta_vals else {"n": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    return {
        "matched_count": len(parsed),
        "mean_gpt_method_diversity_score": safe_mean(scores),
        "strategy_tree_vs_gpt_spearman": corr_strategy,
        "major_tree_vs_gpt_spearman": corr_major,
        "trace_text_vs_gpt_spearman": corr_text,
        "high_strategy_minus_low_strategy_gpt_score_ci": delta_ci,
        "mean_high_strategy_gpt_score": safe_mean(high),
        "mean_low_strategy_gpt_score": safe_mean(low),
    }


def _write_summary(out_dir: Path, groups: List[Dict[str, Any]], bucketed: Dict[str, List[Dict[str, Any]]], key_rows: List[Dict[str, Any]], analysis: Dict[str, Any], args: argparse.Namespace):
    lines = [
        "# P7 GPT-5.5 Blind Validation",
        "",
        f"- evaluator_model: {args.evaluator_model}",
        f"- candidate_groups: {len(groups)}",
        f"- sampled_groups: {len(key_rows)}",
        f"- matched_evaluations: {analysis.get('matched_count', 0)}",
        "",
        "## Bucket Counts",
        "",
    ]
    for bucket, vals in bucketed.items():
        sampled = sum(1 for r in key_rows if r.get("bucket") == bucket)
        lines.append(f"- {bucket}: candidates={len(vals)}, sampled={sampled}")
    lines.extend(["", "## GPT Evaluation", ""])
    if analysis.get("matched_count", 0):
        lines.append(f"- mean_gpt_method_diversity_score: {safe_float(analysis.get('mean_gpt_method_diversity_score')):.4f}")
        for key in ["strategy_tree_vs_gpt_spearman", "major_tree_vs_gpt_spearman", "trace_text_vs_gpt_spearman"]:
            block = analysis.get(key, {})
            if isinstance(block, dict):
                lines.append(f"- {key}: rho={safe_float(block.get('rho')):.4f}, n={block.get('n', 0)}")
        ci = analysis.get("high_strategy_minus_low_strategy_gpt_score_ci", {})
        if isinstance(ci, dict):
            lines.append(
                f"- high_strategy_minus_low_strategy_gpt_score: mean={safe_float(ci.get('mean')):.4f}, "
                f"95% CI=[{safe_float(ci.get('ci_low')):.4f}, {safe_float(ci.get('ci_high')):.4f}]"
            )
    else:
        lines.append("- No parsed GPT evaluations yet.")
    lines.extend(
        [
            "",
            "判读：如果 strategy_tree_vs_gpt_spearman 为正，且 high_strategy 组 GPT 分数高于 low_strategy 组，说明策略树多样性与独立 GPT-5.5 盲评的构念判断一致。",
            "",
        ]
    )
    (out_dir / "p7_gpt55_summary.md").write_text("\n".join(lines), encoding="utf-8")


async def main_async():
    parser = argparse.ArgumentParser(description="P7: GPT-5.5 blind evaluator for method-diversity validation.")
    parser.add_argument("--runs_root", type=str, default="prove_experiments/runs")
    parser.add_argument("--out_dir", type=str, default="prove_experiments/p7_gpt55_blind")
    parser.add_argument("--per_bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluator_model", type=str, default="gpt-5.5")
    parser.add_argument("--evaluate", type=int, default=1, choices=[0, 1])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--max_trace_chars", type=int, default=3500)
    parser.add_argument("--llm_call_timeout", type=float, default=180.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--resume", type=int, default=1, choices=[0, 1])
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = _collect_groups(Path(args.runs_root))
    bucketed = _bucket_groups(groups)
    selected = _sample(bucketed, args.per_bucket, args.seed)
    packet_path, key_rows = _write_packet(selected, out_dir, args.seed)
    eval_path = out_dir / "p7_gpt55_evaluations.jsonl"

    if int(args.evaluate):
        eval_rows = await _run_gpt_evaluator(packet_path, eval_path, args)
    else:
        eval_rows = read_jsonl(eval_path)

    rows = _analysis_rows(key_rows, eval_rows)
    write_csv(rows, out_dir / "p7_gpt55_analysis_rows.csv")
    analysis = _analyze(rows, args.bootstrap_iterations, args.seed)
    (out_dir / "p7_gpt55_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(out_dir, groups, bucketed, key_rows, analysis, args)
    print(f"P7 GPT packet: {packet_path}")
    print(f"P7 GPT evaluations: {eval_path}")
    print(f"P7 GPT summary: {out_dir / 'p7_gpt55_summary.md'}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

