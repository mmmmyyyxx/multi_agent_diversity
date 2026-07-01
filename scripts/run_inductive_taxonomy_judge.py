#!/usr/bin/env python
"""Induce a reasoning-strategy taxonomy from trace batches without labels.

The judge is intentionally not shown any existing taxonomy, gold answers,
vote results, or original automatic family labels. It receives a batch of
blinded traces and returns a compact taxonomy discovered from that batch.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
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


SYSTEM_PROMPT = """You are a reasoning-strategy taxonomy discovery judge.

You will see one batch of blinded reasoning traces. No predefined taxonomy is provided.
Your job is to infer a compact, reusable taxonomy of reasoning strategy families from the traces alone.

Rules:
- Judge reasoning strategy only, not answer correctness.
- Do not assume or name any hidden taxonomy; none is available to you.
- Do not use gold answers, vote results, model identity, prompt identity, or source metadata; they are not provided.
- Prefer functional reasoning categories over surface wording categories.
- Create categories only when they describe reusable reasoning behavior.
- Avoid labels such as other, miscellaneous, unknown, good_reasoning, bad_reasoning, verbose, concise, or correct.
- Return strict JSON only.
"""


USER_PROMPT_TEMPLATE = """Induce a reasoning-strategy taxonomy for this batch of traces.

Output JSON schema:
{{
  "batch_id": "{batch_id}",
  "taxonomy": [
    {{
      "family": "short_snake_case_label",
      "major_family": "optional_coarser_group_label",
      "definition": "What reasoning behavior this family captures.",
      "decision_rule": "How to recognize this family in a new trace.",
      "representative_trace_ids": ["T000001"],
      "evidence_spans": [
        {{"trace_id": "T000001", "span": "short exact span copied from that trace"}}
      ]
    }}
  ],
  "trace_assignments": [
    {{
      "trace_id": "T000001",
      "primary_family": "one family label from taxonomy",
      "secondary_family": "another family label from taxonomy, or same as primary",
      "confidence": 0.0,
      "rationale": "one concise sentence about the reasoning strategy",
      "evidence_spans": ["short exact span copied from that trace"]
    }}
  ],
  "taxonomy_diagnosis": "one paragraph describing the main strategy axes visible in the batch",
  "possible_missing_categories": ["strategy family that might appear with more traces"]
}}

Requirements:
- The taxonomy should usually contain 3 to 10 leaf families unless the batch is extremely homogeneous.
- Every primary_family and secondary_family must be one of the family labels you define in taxonomy.
- Use exact trace IDs from the input.
- Evidence spans must be short contiguous substrings copied from traces.
- If traces are homogeneous, say so and create fewer categories rather than inventing fake distinctions.

Batch ID: {batch_id}
Trace count: {trace_count}

{trace_block}
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["id"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stable_hash(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:12]


def truncate(text: str, max_chars: int) -> str:
    clean = str(text or "").strip()
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "\n[TRUNCATED]"


def _add_item(
    items: list[dict[str, Any]],
    *,
    trace: str,
    source_file: Path,
    source_run: str,
    row_index: int,
    agent_id: Any = "",
    question_hash: str = "",
    split: str = "",
    step: Any = "",
    original_primary_family: str = "",
    original_secondary_family: str = "",
) -> None:
    clean = str(trace or "").strip()
    if not clean:
        return
    items.append(
        {
            "source_file": str(source_file),
            "source_run": source_run,
            "row_index": row_index,
            "split": split,
            "step": step,
            "question_hash": question_hash,
            "agent_id": agent_id,
            "trace": clean,
            "trace_hash": stable_hash(normalize_spaces(clean)),
            "original_primary_family_hidden_from_judge": original_primary_family,
            "original_secondary_family_hidden_from_judge": original_secondary_family,
        }
    )


def collect_trace_items_from_file(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    source_run = path.parent.name
    items: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        question_hash = str(row.get("question_hash", ""))
        split = str(row.get("split", ""))
        step = row.get("step", "")
        agents = row.get("agents", [])
        if isinstance(agents, list) and agents:
            for agent in agents:
                if not isinstance(agent, dict):
                    continue
                _add_item(
                    items,
                    trace=str(agent.get("trace", "")),
                    source_file=path,
                    source_run=source_run,
                    row_index=row_index,
                    agent_id=agent.get("agent_id", ""),
                    question_hash=question_hash,
                    split=split,
                    step=step,
                    original_primary_family=str(agent.get("primary_family", "")),
                    original_secondary_family=str(agent.get("secondary_family", "")),
                )
            continue

        traces = row.get("traces", [])
        if isinstance(traces, list) and traces:
            for agent_id, trace in enumerate(traces):
                _add_item(
                    items,
                    trace=str(trace),
                    source_file=path,
                    source_run=source_run,
                    row_index=row_index,
                    agent_id=agent_id,
                    question_hash=question_hash,
                    split=split,
                    step=step,
                )
            continue

        if row.get("trace"):
            _add_item(
                items,
                trace=str(row.get("trace", "")),
                source_file=path,
                source_run=source_run,
                row_index=row_index,
                agent_id=row.get("agent_id", ""),
                question_hash=question_hash,
                split=split,
                step=step,
                original_primary_family=str(row.get("primary_family", "")),
                original_secondary_family=str(row.get("secondary_family", "")),
            )
    return items


def collect_trace_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths: list[Path] = []
    if args.trace_path:
        for part in str(args.trace_path).split(","):
            part = part.strip()
            if part:
                paths.append(Path(part))
    if args.runs_root:
        root = Path(args.runs_root)
        if root.exists():
            if (root / args.trace_filename).exists():
                paths.append(root / args.trace_filename)
            for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                path = run_dir / args.trace_filename
                if path.exists():
                    paths.append(path)

    seen_paths: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            unique_paths.append(path)

    items: list[dict[str, Any]] = []
    for path in unique_paths:
        items.extend(collect_trace_items_from_file(path))

    if int(args.dedupe_traces):
        seen_hashes: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for item in items:
            trace_hash = str(item.get("trace_hash", ""))
            if trace_hash in seen_hashes:
                continue
            seen_hashes.add(trace_hash)
            deduped.append(item)
        items = deduped

    if args.max_per_source > 0:
        rng = random.Random(args.seed)
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_source[str(item.get("source_run", ""))].append(item)
        capped: list[dict[str, Any]] = []
        for source_items in by_source.values():
            rng.shuffle(source_items)
            capped.extend(source_items[: args.max_per_source])
        items = capped

    rng = random.Random(args.seed)
    rng.shuffle(items)
    if args.sample_size > 0:
        items = items[: args.sample_size]
    return items


def make_batches(items: list[dict[str, Any]], group_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packets: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    group_size = max(1, int(group_size))
    trace_counter = 1
    for batch_index, start in enumerate(range(0, len(items), group_size), start=1):
        batch_id = f"B{batch_index:04d}"
        packet_traces: list[dict[str, Any]] = []
        for item in items[start : start + group_size]:
            trace_id = f"T{trace_counter:06d}"
            trace_counter += 1
            packet_traces.append(
                {
                    "trace_id": trace_id,
                    "trace": item["trace"],
                }
            )
            key_row = {k: v for k, v in item.items() if k != "trace"}
            key_row.update({"batch_id": batch_id, "trace_id": trace_id})
            key_rows.append(key_row)
        packets.append({"batch_id": batch_id, "base_batch_id": batch_id, "repeat_id": 1, "traces": packet_traces})
    return packets, key_rows


def expand_repeat_packets(packets: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    repeats = max(1, int(repeats))
    if repeats == 1:
        return packets

    repeated: list[dict[str, Any]] = []
    for packet in packets:
        base_batch_id = str(packet.get("base_batch_id", packet.get("batch_id", "")))
        for repeat_id in range(1, repeats + 1):
            repeated.append(
                {
                    "batch_id": f"{base_batch_id}_R{repeat_id:02d}",
                    "base_batch_id": base_batch_id,
                    "repeat_id": repeat_id,
                    "traces": [dict(trace) for trace in packet.get("traces", [])],
                }
            )
    return repeated


def build_trace_block(packet: dict[str, Any], max_trace_chars: int) -> str:
    blocks: list[str] = []
    for trace in packet.get("traces", []):
        trace_id = str(trace.get("trace_id", ""))
        text = truncate(str(trace.get("trace", "")), max_trace_chars)
        blocks.append(f'<TRACE id="{trace_id}">\n{text}\n</TRACE>')
    return "\n\n".join(blocks)


def build_user_prompt(packet: dict[str, Any], max_trace_chars: int) -> str:
    return USER_PROMPT_TEMPLATE.format(
        batch_id=str(packet.get("batch_id", "")),
        trace_count=len(packet.get("traces", [])),
        trace_block=build_trace_block(packet, max_trace_chars),
    )


async def call_openai_chat(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    retry_sleep: float,
) -> str:
    last_err: Exception | None = None
    for attempt in range(max(1, int(max_retries))):
        started = time.time()
        try:
            request = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            resp = await asyncio.wait_for(request, timeout=timeout) if timeout > 0 else await request
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            elapsed = time.time() - started
            print(
                f"[inductive_taxonomy][WARN] attempt={attempt + 1}/{max_retries} "
                f"elapsed={elapsed:.2f}s error={exc}",
                flush=True,
            )
            if attempt + 1 < max(1, int(max_retries)):
                await asyncio.sleep(max(0.0, retry_sleep) * (attempt + 1))
    raise RuntimeError(f"Judge call failed after {max_retries} attempts: {last_err}")


def normalize_family_label(value: Any) -> str:
    label = normalize_spaces(str(value or "")).lower()
    label = label.replace("-", "_").replace(" ", "_")
    return "".join(ch for ch in label if ch.isalnum() or ch == "_").strip("_")


def normalize_result(packet: dict[str, Any], raw_response: str) -> dict[str, Any]:
    batch_id = str(packet.get("batch_id", ""))
    base_batch_id = str(packet.get("base_batch_id", batch_id))
    repeat_id = int(packet.get("repeat_id", 1) or 1)
    parsed = extract_json_obj(raw_response)
    if not isinstance(parsed, dict):
        return {
            "batch_id": batch_id,
            "base_batch_id": base_batch_id,
            "repeat_id": repeat_id,
            "parse_ok": 0,
            "taxonomy": [],
            "trace_assignments": [],
            "taxonomy_diagnosis": "",
            "raw_response": raw_response,
        }

    taxonomy = parsed.get("taxonomy", [])
    if not isinstance(taxonomy, list):
        taxonomy = []
    clean_taxonomy: list[dict[str, Any]] = []
    for entry in taxonomy:
        if not isinstance(entry, dict):
            continue
        family = normalize_family_label(entry.get("family", ""))
        if not family:
            continue
        clean_entry = dict(entry)
        clean_entry["family"] = family
        clean_entry["major_family"] = normalize_family_label(entry.get("major_family", ""))
        clean_taxonomy.append(clean_entry)

    allowed = {str(x.get("family", "")) for x in clean_taxonomy}
    assignments = parsed.get("trace_assignments", [])
    if not isinstance(assignments, list):
        assignments = []
    clean_assignments: list[dict[str, Any]] = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        primary = normalize_family_label(assignment.get("primary_family", ""))
        secondary = normalize_family_label(assignment.get("secondary_family", "")) or primary
        clean_assignment = dict(assignment)
        clean_assignment["trace_id"] = str(assignment.get("trace_id", ""))
        clean_assignment["primary_family"] = primary
        clean_assignment["secondary_family"] = secondary
        clean_assignment["primary_in_taxonomy"] = int(primary in allowed)
        clean_assignment["secondary_in_taxonomy"] = int(secondary in allowed)
        try:
            clean_assignment["confidence"] = float(assignment.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            clean_assignment["confidence"] = 0.0
        clean_assignments.append(clean_assignment)

    out = dict(parsed)
    out.update(
        {
            "batch_id": batch_id,
            "base_batch_id": base_batch_id,
            "repeat_id": repeat_id,
            "judge_reported_batch_id": str(parsed.get("batch_id", "") or ""),
            "parse_ok": 1,
            "taxonomy": clean_taxonomy,
            "trace_assignments": clean_assignments,
            "raw_response": raw_response,
        }
    )
    return out


def read_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        batch_id = str(row.get("batch_id", ""))
        if batch_id:
            existing[batch_id] = row
    return existing


async def run_judge(packets: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    api_key_env = str(args.critic_api_key_env or "OPENAI_API_KEY")
    base_url_env = str(args.critic_base_url_env or "")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is not set.")
    base_url = os.getenv(base_url_env) if base_url_env else (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    out_path = Path(args.out_dir) / "inductive_taxonomy_results.jsonl"
    existing = read_existing_results(out_path) if int(args.resume) else {}
    results: list[dict[str, Any] | None] = [None] * len(packets)
    sem = asyncio.Semaphore(max(1, int(args.eval_parallelism)))

    async def eval_one(index: int, packet: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        batch_id = str(packet.get("batch_id", ""))
        if batch_id in existing and int(existing[batch_id].get("parse_ok", 0) or 0):
            return index, existing[batch_id]
        async with sem:
            print(
                f"[inductive_taxonomy] {index + 1}/{len(packets)} "
                f"batch={batch_id} traces={len(packet.get('traces', []))} model={args.critic_model}",
                flush=True,
            )
            raw = await call_openai_chat(
                client,
                model=args.critic_model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(packet, args.max_trace_chars),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.llm_call_timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
            return index, normalize_result(packet, raw)

    tasks = [asyncio.create_task(eval_one(index, packet)) for index, packet in enumerate(packets)]
    for future in asyncio.as_completed(tasks):
        index, result = await future
        results[index] = result
        write_jsonl(out_path, [row for row in results if row is not None])
    return [row for row in results if row is not None]


def summarize_results(results: list[dict[str, Any]], key_rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    key_by_trace = {str(row.get("trace_id", "")): row for row in key_rows}
    family_counts: Counter[str] = Counter()
    major_counts: Counter[str] = Counter()
    assignment_rows: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []
    taxonomy_sets_by_base_repeat: dict[str, dict[int, set[str]]] = defaultdict(dict)

    for result in results:
        batch_id = str(result.get("batch_id", ""))
        base_batch_id = str(result.get("base_batch_id", batch_id))
        repeat_id = int(result.get("repeat_id", 1) or 1)
        taxonomy_set: set[str] = set()
        for entry in result.get("taxonomy", []):
            if not isinstance(entry, dict):
                continue
            family = str(entry.get("family", ""))
            if family:
                taxonomy_set.add(family)
            taxonomy_rows.append(
                {
                    "batch_id": batch_id,
                    "base_batch_id": base_batch_id,
                    "repeat_id": repeat_id,
                    "family": family,
                    "major_family": str(entry.get("major_family", "")),
                    "definition": str(entry.get("definition", "")),
                    "decision_rule": str(entry.get("decision_rule", "")),
                    "representative_trace_ids": json.dumps(entry.get("representative_trace_ids", []), ensure_ascii=False),
                }
            )
        taxonomy_sets_by_base_repeat[base_batch_id][repeat_id] = taxonomy_set
        for assignment in result.get("trace_assignments", []):
            if not isinstance(assignment, dict):
                continue
            trace_id = str(assignment.get("trace_id", ""))
            key = key_by_trace.get(trace_id, {})
            primary = str(assignment.get("primary_family", ""))
            secondary = str(assignment.get("secondary_family", ""))
            family_counts[primary] += 1
            major = ""
            for entry in result.get("taxonomy", []):
                if isinstance(entry, dict) and str(entry.get("family", "")) == primary:
                    major = str(entry.get("major_family", ""))
                    break
            if major:
                major_counts[major] += 1
            assignment_rows.append(
                {
                    "batch_id": batch_id,
                    "base_batch_id": base_batch_id,
                    "repeat_id": repeat_id,
                    "trace_id": trace_id,
                    "primary_family": primary,
                    "secondary_family": secondary,
                    "confidence": assignment.get("confidence", 0.0),
                    "primary_in_taxonomy": assignment.get("primary_in_taxonomy", ""),
                    "secondary_in_taxonomy": assignment.get("secondary_in_taxonomy", ""),
                    "rationale": str(assignment.get("rationale", "")),
                    "evidence_spans": json.dumps(assignment.get("evidence_spans", []), ensure_ascii=False),
                    "source_run": key.get("source_run", ""),
                    "question_hash": key.get("question_hash", ""),
                    "agent_id": key.get("agent_id", ""),
                    "trace_hash": key.get("trace_hash", ""),
                    "original_primary_family_hidden_from_judge": key.get("original_primary_family_hidden_from_judge", ""),
                    "original_secondary_family_hidden_from_judge": key.get("original_secondary_family_hidden_from_judge", ""),
                }
            )

    repeat_stability_rows = build_repeat_stability_rows(assignment_rows, key_rows, max(1, int(args.repeats)))
    taxonomy_jaccard_rows = build_taxonomy_jaccard_rows(taxonomy_sets_by_base_repeat)
    mean_primary_agreement = safe_mean([row.get("primary_agreement", 0.0) for row in repeat_stability_rows])
    mean_pair_agreement = safe_mean([row.get("pair_agreement", 0.0) for row in repeat_stability_rows])
    mean_taxonomy_jaccard = safe_mean([row.get("jaccard", 0.0) for row in taxonomy_jaccard_rows])
    fully_stable_primary = sum(1 for row in repeat_stability_rows if float(row.get("primary_agreement", 0.0) or 0.0) >= 1.0)

    parse_ok = sum(int(row.get("parse_ok", 0) or 0) for row in results)
    summary = {
        "critic_model": args.critic_model,
        "trace_count": len(key_rows),
        "batch_count": len(results),
        "base_batch_count": len({str(row.get("base_batch_id", row.get("batch_id", ""))) for row in results}),
        "repeats": max(1, int(args.repeats)),
        "parse_ok_batches": parse_ok,
        "group_size": args.group_size,
        "max_trace_chars": args.max_trace_chars,
        "judge_saw_existing_taxonomy": False,
        "judge_input_fields": ["trace_id", "trace"],
        "primary_family_counts": dict(family_counts.most_common()),
        "major_family_counts": dict(major_counts.most_common()),
        "unique_families": sorted(k for k in family_counts if k),
        "mean_primary_agreement_across_repeats": mean_primary_agreement,
        "mean_pair_agreement_across_repeats": mean_pair_agreement,
        "fully_stable_primary_trace_count": fully_stable_primary,
        "fully_stable_primary_trace_share": float(fully_stable_primary / len(repeat_stability_rows)) if repeat_stability_rows else 0.0,
        "mean_taxonomy_label_set_jaccard_across_repeats": mean_taxonomy_jaccard,
    }

    with (out_dir / "inductive_taxonomy_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(out_dir / "inductive_taxonomy_assignments.csv", assignment_rows)
    write_csv(out_dir / "inductive_taxonomy_taxonomy_rows.csv", taxonomy_rows)
    write_csv(out_dir / "inductive_taxonomy_repeat_stability.csv", repeat_stability_rows)
    write_csv(out_dir / "inductive_taxonomy_repeat_jaccard.csv", taxonomy_jaccard_rows)

    md_lines = [
        "# Inductive Taxonomy Judge Summary",
        "",
        f"- critic_model: `{args.critic_model}`",
        f"- trace_count: {len(key_rows)}",
        f"- batch_count: {len(results)}",
        f"- repeats: {max(1, int(args.repeats))}",
        f"- parse_ok_batches: {parse_ok}/{len(results)}",
        "- judge_input_fields: `trace_id`, `trace`",
        "- existing taxonomy shown to judge: no",
        f"- mean_primary_agreement_across_repeats: {mean_primary_agreement:.4f}",
        f"- mean_pair_agreement_across_repeats: {mean_pair_agreement:.4f}",
        f"- fully_stable_primary_trace_share: {summary['fully_stable_primary_trace_share']:.4f}",
        f"- mean_taxonomy_label_set_jaccard_across_repeats: {mean_taxonomy_jaccard:.4f}",
        "",
        "## Primary Family Counts",
        "",
    ]
    if family_counts:
        md_lines.extend(["| family | count |", "| --- | ---: |"])
        for family, count in family_counts.most_common():
            md_lines.append(f"| `{family}` | {count} |")
    else:
        md_lines.append("No parsed trace assignments.")

    md_lines.extend(["", "## Repeat Stability", ""])
    if repeat_stability_rows:
        md_lines.extend(["| trace_id | primary_agreement | pair_agreement | unique_primary_count | primary_labels_by_repeat |", "| --- | ---: | ---: | ---: | --- |"])
        for row in repeat_stability_rows:
            md_lines.append(
                f"| `{row.get('trace_id', '')}` | {float(row.get('primary_agreement', 0.0)):.4f} | "
                f"{float(row.get('pair_agreement', 0.0)):.4f} | {row.get('unique_primary_count', '')} | "
                f"`{row.get('primary_labels_by_repeat', '')}` |"
            )
    else:
        md_lines.append("No repeat stability rows.")

    md_lines.extend(["", "## Taxonomy Set Jaccard", ""])
    if taxonomy_jaccard_rows:
        md_lines.extend(["| base_batch_id | repeat_a | repeat_b | jaccard | shared | only_a | only_b |", "| --- | ---: | ---: | ---: | --- | --- | --- |"])
        for row in taxonomy_jaccard_rows:
            md_lines.append(
                f"| `{row.get('base_batch_id', '')}` | {row.get('repeat_a', '')} | {row.get('repeat_b', '')} | "
                f"{float(row.get('jaccard', 0.0)):.4f} | `{row.get('shared_families', '')}` | "
                f"`{row.get('only_a_families', '')}` | `{row.get('only_b_families', '')}` |"
            )
    else:
        md_lines.append("No taxonomy Jaccard rows.")

    md_lines.extend(["", "## Batch Taxonomies", ""])
    for result in results:
        batch_id = str(result.get("batch_id", ""))
        md_lines.append(f"### {batch_id}")
        if not int(result.get("parse_ok", 0) or 0):
            md_lines.append("")
            md_lines.append("Parse failed; see raw response in `inductive_taxonomy_results.jsonl`.")
            md_lines.append("")
            continue
        diagnosis = normalize_spaces(str(result.get("taxonomy_diagnosis", "")))
        if diagnosis:
            md_lines.extend(["", diagnosis, ""])
        taxonomy = [entry for entry in result.get("taxonomy", []) if isinstance(entry, dict)]
        if taxonomy:
            md_lines.extend(["| family | major | definition |", "| --- | --- | --- |"])
            for entry in taxonomy:
                family = str(entry.get("family", ""))
                major = str(entry.get("major_family", ""))
                definition = normalize_spaces(str(entry.get("definition", "")))
                md_lines.append(f"| `{family}` | `{major}` | {definition} |")
        else:
            md_lines.append("")
            md_lines.append("No taxonomy entries parsed.")
        md_lines.append("")

    (out_dir / "inductive_taxonomy_summary.md").write_text("\n".join(md_lines), encoding="utf-8")


def safe_mean(values: list[Any]) -> float:
    nums: list[float] = []
    for value in values:
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            continue
    return float(sum(nums) / len(nums)) if nums else 0.0


def agreement_score(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    return float(max(counts.values()) / len(labels))


def build_repeat_stability_rows(
    assignment_rows: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
    repeats: int,
) -> list[dict[str, Any]]:
    by_trace_repeat: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in assignment_rows:
        trace_id = str(row.get("trace_id", ""))
        repeat_id = int(row.get("repeat_id", 1) or 1)
        if trace_id:
            by_trace_repeat[trace_id][repeat_id] = row

    rows: list[dict[str, Any]] = []
    for key in key_rows:
        trace_id = str(key.get("trace_id", ""))
        primary_labels: list[str] = []
        secondary_labels: list[str] = []
        pair_labels: list[str] = []
        confidences: list[float] = []
        for repeat_id in range(1, repeats + 1):
            assignment = by_trace_repeat.get(trace_id, {}).get(repeat_id)
            if assignment is None:
                primary = "__missing__"
                secondary = "__missing__"
                confidence = 0.0
            else:
                primary = str(assignment.get("primary_family", "") or "__missing__")
                secondary = str(assignment.get("secondary_family", "") or primary)
                try:
                    confidence = float(assignment.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
            primary_labels.append(primary)
            secondary_labels.append(secondary)
            pair_labels.append(f"{primary}::{secondary}")
            confidences.append(confidence)

        unique_primary = sorted(set(primary_labels))
        unique_pairs = sorted(set(pair_labels))
        rows.append(
            {
                "trace_id": trace_id,
                "source_run": key.get("source_run", ""),
                "question_hash": key.get("question_hash", ""),
                "agent_id": key.get("agent_id", ""),
                "trace_hash": key.get("trace_hash", ""),
                "original_primary_family_hidden_from_judge": key.get("original_primary_family_hidden_from_judge", ""),
                "original_secondary_family_hidden_from_judge": key.get("original_secondary_family_hidden_from_judge", ""),
                "repeats": repeats,
                "primary_agreement": agreement_score(primary_labels),
                "secondary_agreement": agreement_score(secondary_labels),
                "pair_agreement": agreement_score(pair_labels),
                "unique_primary_count": len(unique_primary),
                "unique_pair_count": len(unique_pairs),
                "mean_confidence": safe_mean(confidences),
                "primary_labels_by_repeat": "|".join(primary_labels),
                "secondary_labels_by_repeat": "|".join(secondary_labels),
                "pair_labels_by_repeat": "|".join(pair_labels),
            }
        )
    return rows


def build_taxonomy_jaccard_rows(taxonomy_sets_by_base_repeat: dict[str, dict[int, set[str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_batch_id, by_repeat in sorted(taxonomy_sets_by_base_repeat.items()):
        repeat_ids = sorted(by_repeat)
        for i, repeat_a in enumerate(repeat_ids):
            for repeat_b in repeat_ids[i + 1 :]:
                set_a = by_repeat.get(repeat_a, set())
                set_b = by_repeat.get(repeat_b, set())
                union = set_a | set_b
                shared = set_a & set_b
                jaccard = float(len(shared) / len(union)) if union else 1.0
                rows.append(
                    {
                        "base_batch_id": base_batch_id,
                        "repeat_a": repeat_a,
                        "repeat_b": repeat_b,
                        "jaccard": jaccard,
                        "shared_families": "|".join(sorted(shared)),
                        "only_a_families": "|".join(sorted(set_a - set_b)),
                        "only_b_families": "|".join(sorted(set_b - set_a)),
                    }
                )
    return rows


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a judge to induce a reasoning-strategy taxonomy from batches of traces without showing an existing taxonomy."
    )
    parser.add_argument("--trace_path", type=str, default="runs_experiments/shared_div/test_trace_history.jsonl")
    parser.add_argument("--runs_root", type=str, default="")
    parser.add_argument("--trace_filename", type=str, default="test_trace_history.jsonl")
    parser.add_argument("--out_dir", type=str, default="prove_experiments/inductive_taxonomy_judge")
    parser.add_argument("--sample_size", type=int, default=24)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max_per_source", type=int, default=0)
    parser.add_argument("--dedupe_traces", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--critic_model", type=str, default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=5000)
    parser.add_argument("--max_trace_chars", type=int, default=3000)
    parser.add_argument("--eval_parallelism", type=int, default=2)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--llm_call_timeout", type=float, default=180.0)
    parser.add_argument("--critic_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--critic_base_url_env", type=str, default="")
    parser.add_argument("--resume", type=int, default=1, choices=[0, 1])
    parser.add_argument("--packet_only", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    random.seed(args.seed)

    items = collect_trace_items(args)
    if not items:
        raise ValueError("No traces found. Set --trace_path or --runs_root to existing trace history files.")

    base_packets, key_rows = make_batches(items, args.group_size)
    packets = expand_repeat_packets(base_packets, args.repeats)
    write_jsonl(out_dir / "inductive_taxonomy_packets.jsonl", packets)
    write_csv(out_dir / "inductive_taxonomy_key.csv", key_rows)

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "trace_count": len(key_rows),
        "batch_count": len(packets),
        "base_batch_count": len(base_packets),
        "repeats": max(1, int(args.repeats)),
        "judge_saw_existing_taxonomy": False,
        "judge_input_fields": ["trace_id", "trace"],
        "prompt_version": "inductive_batch_taxonomy_no_existing_taxonomy_v1",
    }
    with (out_dir / "inductive_taxonomy_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"Prepared {len(key_rows)} traces in {len(base_packets)} base batch(es), "
        f"{len(packets)} judge call packet(s) with repeats={max(1, int(args.repeats))}. out_dir={out_dir}",
        flush=True,
    )
    if int(args.packet_only):
        print("packet_only=1, skipping judge calls.", flush=True)
        return

    results = await run_judge(packets, args)
    summarize_results(results, key_rows, out_dir, args)
    print(f"Inductive taxonomy judge complete: {out_dir}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
