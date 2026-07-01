#!/usr/bin/env python
"""Evaluate traces with a small induced taxonomy and compare to induction labels."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.utils import ensure_dir, extract_json_obj, normalize_spaces  # noqa: E402
from scripts.run_inductive_taxonomy_judge import (  # noqa: E402
    call_openai_chat,
    read_jsonl,
    stable_hash,
    truncate,
    write_csv,
    write_jsonl,
)


SYSTEM_PROMPT = """You are a reasoning-strategy judge.

You will see a small induced taxonomy and one batch of blinded reasoning traces.
Assign each trace to the most appropriate taxonomy family.

Rules:
- Use only the provided taxonomy labels.
- Judge reasoning strategy only, not answer correctness.
- Do not create new labels, do not use "other", and do not use hidden metadata.
- Return strict JSON only.
"""


USER_PROMPT_TEMPLATE = """Evaluate these traces using the provided induced taxonomy.

Return JSON:
{{
  "batch_id": "{batch_id}",
  "trace_assignments": [
    {{
      "trace_id": "T000001",
      "primary_family": "one provided taxonomy label",
      "secondary_family": "one provided taxonomy label, or same as primary",
      "confidence": 0.0,
      "rationale": "one concise sentence",
      "evidence_spans": ["short exact span copied from the trace"]
    }}
  ]
}}

Provided taxonomy:
{taxonomy_json}

Batch ID: {batch_id}
Trace count: {trace_count}

{trace_block}
"""


SMALL_TAXONOMY = [
    {
        "family": "statement_truth_evaluation",
        "definition": "Evaluates one or more mathematical statements as true or false, then selects an option from the resulting truth-value pattern.",
        "decision_rule": "Use when the trace is organized around Statement 1/Statement 2, true/false judgments, or option selection from truth values.",
        "aliases_from_induction": ["truth_evaluation", "statement_analysis", "truth_value_analysis"],
    },
    {
        "family": "group_property_analysis",
        "definition": "Analyzes group-theoretic structure or subgroup properties such as normality, abelianness, element orders, cosets, or solvability.",
        "decision_rule": "Use when group properties are the main object and the trace is not mainly enumerating permutation cycle orders.",
        "aliases_from_induction": ["group_property_analysis", "group_properties_analysis"],
    },
    {
        "family": "relation_property_analysis",
        "definition": "Checks properties of a mathematical relation such as symmetry, antisymmetry, reflexivity, transitivity, or equivalence.",
        "decision_rule": "Use when the trace tests relation property definitions against listed ordered pairs.",
        "aliases_from_induction": ["relation_property_analysis", "relation_properties_analysis", "relation_analysis"],
    },
    {
        "family": "ring_characteristic_analysis",
        "definition": "Determines or reasons about the characteristic of a ring or product of rings.",
        "decision_rule": "Use when the trace computes ring characteristic from components, modular rings, or product rings.",
        "aliases_from_induction": ["ring_characteristic_analysis"],
    },
    {
        "family": "permutation_order_analysis",
        "definition": "Determines the order or maximum possible order of a permutation by analyzing cycle decompositions and least common multiples.",
        "decision_rule": "Use when the trace enumerates cycle structures or uses LCM of cycle lengths to find a permutation order.",
        "aliases_from_induction": ["permutation_order_analysis"],
    },
    {
        "family": "polynomial_evaluation",
        "definition": "Evaluates a polynomial over a field or ring, often by plugging in candidates or checking roots.",
        "decision_rule": "Use when the trace's main method is evaluating polynomial values, roots, degrees, or polynomial behavior.",
        "aliases_from_induction": ["polynomial_evaluation"],
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_family(value: Any) -> str:
    label = normalize_spaces(str(value or "")).lower()
    label = label.replace("-", "_").replace(" ", "_")
    return "".join(ch for ch in label if ch.isalnum() or ch == "_").strip("_")


def alias_map(taxonomy: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in taxonomy:
        family = normalize_family(entry.get("family", ""))
        if not family:
            continue
        mapping[family] = family
        for alias in entry.get("aliases_from_induction", []):
            mapping[normalize_family(alias)] = family
    return mapping


def load_base_packet(induction_dir: Path, base_batch_id: str) -> dict[str, Any]:
    packets = read_jsonl(induction_dir / "inductive_taxonomy_packets.jsonl")
    for packet in packets:
        if str(packet.get("base_batch_id", packet.get("batch_id", ""))) == base_batch_id:
            traces = packet.get("traces", [])
            return {
                "batch_id": base_batch_id,
                "base_batch_id": base_batch_id,
                "repeat_id": 1,
                "traces": traces,
            }
    raise ValueError(f"Cannot find base batch {base_batch_id} in {induction_dir}")


def repeat_packets(base_packet: dict[str, Any], repeats: int) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for repeat_id in range(1, max(1, repeats) + 1):
        packets.append(
            {
                "batch_id": f"{base_packet['base_batch_id']}_GT{repeat_id:02d}",
                "base_batch_id": base_packet["base_batch_id"],
                "repeat_id": repeat_id,
                "traces": [dict(trace) for trace in base_packet.get("traces", [])],
            }
        )
    return packets


def build_trace_block(packet: dict[str, Any], max_trace_chars: int) -> str:
    blocks: list[str] = []
    for trace in packet.get("traces", []):
        trace_id = str(trace.get("trace_id", ""))
        text = truncate(str(trace.get("trace", "")), max_trace_chars)
        blocks.append(f'<TRACE id="{trace_id}">\n{text}\n</TRACE>')
    return "\n\n".join(blocks)


def build_user_prompt(packet: dict[str, Any], taxonomy: list[dict[str, Any]], max_trace_chars: int) -> str:
    compact_taxonomy = [
        {
            "family": entry["family"],
            "definition": entry["definition"],
            "decision_rule": entry["decision_rule"],
        }
        for entry in taxonomy
    ]
    return USER_PROMPT_TEMPLATE.format(
        batch_id=str(packet.get("batch_id", "")),
        trace_count=len(packet.get("traces", [])),
        taxonomy_json=json.dumps(compact_taxonomy, ensure_ascii=False, indent=2),
        trace_block=build_trace_block(packet, max_trace_chars),
    )


def normalize_guided_result(packet: dict[str, Any], raw_response: str, allowed: set[str]) -> dict[str, Any]:
    batch_id = str(packet.get("batch_id", ""))
    parsed = extract_json_obj(raw_response)
    if not isinstance(parsed, dict):
        return {
            "batch_id": batch_id,
            "base_batch_id": str(packet.get("base_batch_id", batch_id)),
            "repeat_id": int(packet.get("repeat_id", 1) or 1),
            "parse_ok": 0,
            "trace_assignments": [],
            "raw_response": raw_response,
        }

    assignments = parsed.get("trace_assignments", [])
    if not isinstance(assignments, list):
        assignments = []
    clean_assignments: list[dict[str, Any]] = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        primary = normalize_family(assignment.get("primary_family", ""))
        secondary = normalize_family(assignment.get("secondary_family", "")) or primary
        try:
            confidence = float(assignment.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        clean_assignment = dict(assignment)
        clean_assignment.update(
            {
                "trace_id": str(assignment.get("trace_id", "")),
                "primary_family": primary,
                "secondary_family": secondary,
                "primary_in_taxonomy": int(primary in allowed),
                "secondary_in_taxonomy": int(secondary in allowed),
                "confidence": confidence,
            }
        )
        clean_assignments.append(clean_assignment)
    return {
        "batch_id": batch_id,
        "base_batch_id": str(packet.get("base_batch_id", batch_id)),
        "repeat_id": int(packet.get("repeat_id", 1) or 1),
        "judge_reported_batch_id": str(parsed.get("batch_id", "") or ""),
        "parse_ok": 1,
        "trace_assignments": clean_assignments,
        "raw_response": raw_response,
    }


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("batch_id", "")): row for row in read_jsonl(path) if row.get("batch_id")}


async def run_guided_judge(
    packets: list[dict[str, Any]],
    taxonomy: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    api_key_env = str(args.critic_api_key_env or "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is not set.")
    base_url = (
        os.getenv(str(args.critic_base_url_env))
        if str(args.critic_base_url_env or "").strip()
        else (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
    )
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    out_path = Path(args.out_dir) / "guided_taxonomy_results.jsonl"
    existing = read_existing(out_path) if int(args.resume) else {}
    allowed = {normalize_family(entry.get("family", "")) for entry in taxonomy}
    results: list[dict[str, Any] | None] = [None] * len(packets)
    sem = asyncio.Semaphore(max(1, int(args.eval_parallelism)))

    async def eval_one(index: int, packet: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        batch_id = str(packet.get("batch_id", ""))
        if batch_id in existing and int(existing[batch_id].get("parse_ok", 0) or 0):
            return index, existing[batch_id]
        async with sem:
            print(
                f"[guided_taxonomy] {index + 1}/{len(packets)} batch={batch_id} "
                f"traces={len(packet.get('traces', []))} model={args.critic_model}",
                flush=True,
            )
            raw = await call_openai_chat(
                client,
                model=args.critic_model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(packet, taxonomy, args.max_trace_chars),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.llm_call_timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
            return index, normalize_guided_result(packet, raw, allowed)

    tasks = [asyncio.create_task(eval_one(index, packet)) for index, packet in enumerate(packets)]
    for future in asyncio.as_completed(tasks):
        index, result = await future
        results[index] = result
        write_jsonl(out_path, [row for row in results if row is not None])
    return [row for row in results if row is not None]


def induction_majority_by_trace(induction_dir: Path, mapping: dict[str, str]) -> dict[str, dict[str, Any]]:
    rows = read_csv(induction_dir / "inductive_taxonomy_assignments.csv")
    grouped: dict[str, list[str]] = defaultdict(list)
    raw_grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        trace_id = str(row.get("trace_id", ""))
        raw_primary = normalize_family(row.get("primary_family", ""))
        mapped = mapping.get(raw_primary, raw_primary)
        if trace_id:
            grouped[trace_id].append(mapped)
            raw_grouped[trace_id].append(raw_primary)

    out: dict[str, dict[str, Any]] = {}
    for trace_id, labels in grouped.items():
        counts = Counter(labels)
        raw_counts = Counter(raw_grouped.get(trace_id, []))
        majority, majority_count = counts.most_common(1)[0]
        out[trace_id] = {
            "induction_mapped_majority": majority,
            "induction_mapped_majority_share": float(majority_count / len(labels)) if labels else 0.0,
            "induction_mapped_labels": "|".join(labels),
            "induction_mapped_counts": json.dumps(dict(counts.most_common()), ensure_ascii=False),
            "induction_raw_counts": json.dumps(dict(raw_counts.most_common()), ensure_ascii=False),
        }
    return out


def agreement_score(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    return float(counts.most_common(1)[0][1] / len(labels))


def safe_mean(values: list[Any]) -> float:
    nums: list[float] = []
    for value in values:
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            continue
    return float(sum(nums) / len(nums)) if nums else 0.0


def summarize(
    results: list[dict[str, Any]],
    induction_dir: Path,
    taxonomy: list[dict[str, Any]],
    packets: list[dict[str, Any]],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    mapping = alias_map(taxonomy)
    induction_ref = induction_majority_by_trace(induction_dir, mapping)
    key_rows = read_csv(induction_dir / "inductive_taxonomy_key.csv")
    key_by_trace = {row.get("trace_id", ""): row for row in key_rows}

    guided_rows: list[dict[str, Any]] = []
    by_trace_repeat: dict[str, dict[int, str]] = defaultdict(dict)
    family_counts: Counter[str] = Counter()
    for result in results:
        batch_id = str(result.get("batch_id", ""))
        repeat_id = int(result.get("repeat_id", 1) or 1)
        for assignment in result.get("trace_assignments", []):
            if not isinstance(assignment, dict):
                continue
            trace_id = str(assignment.get("trace_id", ""))
            primary = normalize_family(assignment.get("primary_family", ""))
            secondary = normalize_family(assignment.get("secondary_family", "")) or primary
            key = key_by_trace.get(trace_id, {})
            ref = induction_ref.get(trace_id, {})
            match_majority = int(primary == ref.get("induction_mapped_majority", ""))
            family_counts[primary] += 1
            by_trace_repeat[trace_id][repeat_id] = primary
            guided_rows.append(
                {
                    "batch_id": batch_id,
                    "repeat_id": repeat_id,
                    "trace_id": trace_id,
                    "primary_family": primary,
                    "secondary_family": secondary,
                    "confidence": assignment.get("confidence", 0.0),
                    "primary_in_taxonomy": assignment.get("primary_in_taxonomy", ""),
                    "secondary_in_taxonomy": assignment.get("secondary_in_taxonomy", ""),
                    "matches_induction_mapped_majority": match_majority,
                    "induction_mapped_majority": ref.get("induction_mapped_majority", ""),
                    "induction_mapped_majority_share": ref.get("induction_mapped_majority_share", ""),
                    "induction_mapped_labels": ref.get("induction_mapped_labels", ""),
                    "rationale": str(assignment.get("rationale", "")),
                    "evidence_spans": json.dumps(assignment.get("evidence_spans", []), ensure_ascii=False),
                    "source_run": key.get("source_run", ""),
                    "question_hash": key.get("question_hash", ""),
                    "agent_id": key.get("agent_id", ""),
                    "trace_hash": key.get("trace_hash", ""),
                    "original_primary_family_hidden_from_judge": key.get("original_primary_family_hidden_from_judge", ""),
                }
            )

    repeat_count = max(1, int(args.repeats))
    stability_rows: list[dict[str, Any]] = []
    for trace in packets[0].get("traces", []):
        trace_id = str(trace.get("trace_id", ""))
        labels = [by_trace_repeat.get(trace_id, {}).get(i, "__missing__") for i in range(1, repeat_count + 1)]
        ref = induction_ref.get(trace_id, {})
        key = key_by_trace.get(trace_id, {})
        stability_rows.append(
            {
                "trace_id": trace_id,
                "guided_primary_agreement": agreement_score(labels),
                "guided_unique_primary_count": len(set(labels)),
                "guided_primary_labels_by_repeat": "|".join(labels),
                "induction_mapped_majority": ref.get("induction_mapped_majority", ""),
                "induction_mapped_majority_share": ref.get("induction_mapped_majority_share", ""),
                "guided_majority_matches_induction_majority": int(Counter(labels).most_common(1)[0][0] == ref.get("induction_mapped_majority", "")),
                "source_run": key.get("source_run", ""),
                "question_hash": key.get("question_hash", ""),
                "agent_id": key.get("agent_id", ""),
            }
        )

    parse_ok = sum(int(result.get("parse_ok", 0) or 0) for result in results)
    assignment_match_rate = safe_mean([row["matches_induction_mapped_majority"] for row in guided_rows])
    trace_majority_match_rate = safe_mean([row["guided_majority_matches_induction_majority"] for row in stability_rows])
    guided_repeat_agreement = safe_mean([row["guided_primary_agreement"] for row in stability_rows])
    summary = {
        "critic_model": args.critic_model,
        "trace_count": len(packets[0].get("traces", [])) if packets else 0,
        "repeats": repeat_count,
        "parse_ok_batches": parse_ok,
        "batch_count": len(results),
        "taxonomy_path": str(out_dir / "small_induced_taxonomy.json"),
        "guided_family_counts": dict(family_counts.most_common()),
        "assignment_match_rate_vs_induction_mapped_majority": assignment_match_rate,
        "trace_guided_majority_match_rate_vs_induction_mapped_majority": trace_majority_match_rate,
        "mean_guided_primary_agreement_across_repeats": guided_repeat_agreement,
        "judge_saw_taxonomy": True,
        "taxonomy_families": [entry["family"] for entry in taxonomy],
    }

    write_csv(out_dir / "guided_taxonomy_assignments.csv", guided_rows)
    write_csv(out_dir / "guided_taxonomy_repeat_stability.csv", stability_rows)
    with (out_dir / "guided_taxonomy_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Guided Induced Taxonomy Consistency",
        "",
        f"- critic_model: `{args.critic_model}`",
        f"- trace_count: {summary['trace_count']}",
        f"- repeats: {repeat_count}",
        f"- parse_ok_batches: {parse_ok}/{len(results)}",
        f"- assignment_match_rate_vs_induction_mapped_majority: {assignment_match_rate:.4f}",
        f"- trace_guided_majority_match_rate_vs_induction_mapped_majority: {trace_majority_match_rate:.4f}",
        f"- mean_guided_primary_agreement_across_repeats: {guided_repeat_agreement:.4f}",
        "",
        "## Small Taxonomy",
        "",
        "| family | definition |",
        "| --- | --- |",
    ]
    for entry in taxonomy:
        md_lines.append(f"| `{entry['family']}` | {entry['definition']} |")

    md_lines.extend(["", "## Guided Family Counts", "", "| family | count |", "| --- | ---: |"])
    for family, count in family_counts.most_common():
        md_lines.append(f"| `{family}` | {count} |")

    md_lines.extend(["", "## Per Trace Stability", "", "| trace_id | guided_agreement | guided_labels | induction_majority | match |", "| --- | ---: | --- | --- | ---: |"])
    for row in stability_rows:
        md_lines.append(
            f"| `{row['trace_id']}` | {float(row['guided_primary_agreement']):.4f} | "
            f"`{row['guided_primary_labels_by_repeat']}` | `{row['induction_mapped_majority']}` | "
            f"{row['guided_majority_matches_induction_majority']} |"
        )
    (out_dir / "guided_taxonomy_summary.md").write_text("\n".join(md_lines), encoding="utf-8")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Evaluate traces with a small induced taxonomy and compare to induction outputs.")
    parser.add_argument("--induction_dir", type=str, default="prove_experiments/inductive_taxonomy_judge_20x5")
    parser.add_argument("--out_dir", type=str, default="prove_experiments/guided_induced_taxonomy_20x3")
    parser.add_argument("--base_batch_id", type=str, default="B0001")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--critic_model", type=str, default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=5000)
    parser.add_argument("--max_trace_chars", type=int, default=1800)
    parser.add_argument("--eval_parallelism", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--llm_call_timeout", type=float, default=180.0)
    parser.add_argument("--critic_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--critic_base_url_env", type=str, default="")
    parser.add_argument("--resume", type=int, default=1, choices=[0, 1])
    parser.add_argument("--packet_only", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    induction_dir = Path(args.induction_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    random.seed(42)

    taxonomy = [dict(entry) for entry in SMALL_TAXONOMY]
    with (out_dir / "small_induced_taxonomy.json").open("w", encoding="utf-8") as f:
        json.dump({"taxonomy": taxonomy}, f, ensure_ascii=False, indent=2)

    base_packet = load_base_packet(induction_dir, args.base_batch_id)
    packets = repeat_packets(base_packet, max(1, int(args.repeats)))
    write_jsonl(out_dir / "guided_taxonomy_packets.jsonl", packets)
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "trace_count": len(base_packet.get("traces", [])),
        "repeats": max(1, int(args.repeats)),
        "judge_saw_taxonomy": True,
        "taxonomy_families": [entry["family"] for entry in taxonomy],
    }
    with (out_dir / "guided_taxonomy_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"Prepared guided taxonomy eval: traces={len(base_packet.get('traces', []))} "
        f"repeats={max(1, int(args.repeats))} out_dir={out_dir}",
        flush=True,
    )
    if int(args.packet_only):
        print("packet_only=1, skipping judge calls.", flush=True)
        return

    results = await run_guided_judge(packets, taxonomy, args)
    summarize(results, induction_dir, taxonomy, packets, out_dir, args)
    print(f"Guided induced taxonomy consistency complete: {out_dir}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
