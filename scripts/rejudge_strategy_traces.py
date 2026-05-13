import argparse
import asyncio
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import TextualGradientRLSystem
from multi_dataset_diverse_rl.utils import ensure_dir, infer_strategy_family_major, set_seed


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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


def _collect_trace_items(runs_root: Path, max_per_run: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        trace_path = run_dir / "test_trace_history.jsonl"
        records = _read_jsonl(trace_path)
        run_items: List[Dict[str, Any]] = []
        for rec in records:
            qh = str(rec.get("question_hash", ""))
            for agent in rec.get("agents", []):
                if not isinstance(agent, dict):
                    continue
                trace = str(agent.get("trace", "")).strip()
                if not trace:
                    continue
                run_items.append(
                    {
                        "source_run": run_dir.name,
                        "question_hash": qh,
                        "agent_id": int(agent.get("agent_id", len(run_items))),
                        "trace": trace,
                        "original_primary_family": str(agent.get("primary_family", "")),
                        "original_secondary_family": str(agent.get("secondary_family", "")),
                    }
                )
        if max_per_run > 0 and len(run_items) > max_per_run:
            run_items = rng.sample(run_items, max_per_run)
        items.extend(run_items)
    rng.shuffle(items)
    return items


def _agreement(labels: List[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    return float(max(counts.values()) / len(labels))


def _write_summary(rows: List[Dict[str, Any]], out_dir: Path):
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trace_key"])].append(row)

    trace_summaries = []
    for key, reps in grouped.items():
        primaries = [str(r.get("primary_family", "")) for r in reps]
        secondaries = [str(r.get("secondary_family", "")) for r in reps]
        pairs = [f"{p}::{s}" for p, s in zip(primaries, secondaries)]
        majors = [infer_strategy_family_major(p) for p in primaries]
        trace_summaries.append(
            {
                "trace_key": key,
                "source_run": reps[0].get("source_run", ""),
                "question_hash": reps[0].get("question_hash", ""),
                "agent_id": reps[0].get("agent_id", ""),
                "original_primary_family": reps[0].get("original_primary_family", ""),
                "original_secondary_family": reps[0].get("original_secondary_family", ""),
                "primary_agreement": _agreement(primaries),
                "secondary_agreement": _agreement(secondaries),
                "pair_agreement": _agreement(pairs),
                "major_agreement": _agreement(majors),
                "mean_confidence": sum(float(r.get("confidence", 0.0) or 0.0) for r in reps) / len(reps),
                "repeats": len(reps),
                "primary_labels": ";".join(primaries),
                "secondary_labels": ";".join(secondaries),
            }
        )

    def mean(name: str) -> float:
        vals = [float(x.get(name, 0.0) or 0.0) for x in trace_summaries]
        return float(sum(vals) / len(vals)) if vals else 0.0

    summary = {
        "trace_count": len(trace_summaries),
        "judgment_count": len(rows),
        "mean_primary_agreement": mean("primary_agreement"),
        "mean_secondary_agreement": mean("secondary_agreement"),
        "mean_pair_agreement": mean("pair_agreement"),
        "mean_major_agreement": mean("major_agreement"),
        "mean_confidence": mean("mean_confidence"),
    }
    with (out_dir / "rejudge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = out_dir / "rejudge_trace_agreement.csv"
    fieldnames = [
        "trace_key",
        "source_run",
        "question_hash",
        "agent_id",
        "original_primary_family",
        "original_secondary_family",
        "primary_agreement",
        "secondary_agreement",
        "pair_agreement",
        "major_agreement",
        "mean_confidence",
        "repeats",
        "primary_labels",
        "secondary_labels",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace_summaries)

    md_lines = [
        "# Rejudge Reliability Summary",
        "",
        f"- trace_count: {summary['trace_count']}",
        f"- judgment_count: {summary['judgment_count']}",
        f"- mean_major_agreement: {summary['mean_major_agreement']:.4f}",
        f"- mean_primary_agreement: {summary['mean_primary_agreement']:.4f}",
        f"- mean_pair_agreement: {summary['mean_pair_agreement']:.4f}",
        f"- mean_confidence: {summary['mean_confidence']:.4f}",
        "",
        "Pass reference: major >= 0.85, primary >= 0.70.",
        "",
    ]
    (out_dir / "rejudge_summary.md").write_text("\n".join(md_lines), encoding="utf-8")


async def main_async():
    parser = argparse.ArgumentParser(description="Rejudge sampled traces repeatedly to test strategy-label stability.")
    parser.add_argument("--runs_root", type=str, default="runs_experiments")
    parser.add_argument("--out_dir", type=str, default="prove_experiments/rejudge_p1")
    parser.add_argument("--max_per_run", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_type", type=str, default="mmlu", choices=["auto", "gsm8k", "mmlu"])
    parser.add_argument("--critic_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_expansion_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_taxonomy_path", type=str, default="auto")
    parser.add_argument("--family_expansion_enabled", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_dual_family_labels", type=int, default=1, choices=[0, 1])
    parser.add_argument("--family_rejudge_on_low_confidence", type=int, default=0, choices=[0, 1])
    parser.add_argument("--critic_max_tokens", type=int, default=8000)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--transient_retry_forever", type=int, default=1, choices=[0, 1])
    parser.add_argument("--llm_call_timeout", type=float, default=120.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    set_seed(args.seed)
    items = _collect_trace_items(Path(args.runs_root), args.max_per_run, args.seed)
    if not items:
        raise ValueError(f"No traces found under {args.runs_root}")

    cfg = Config(
        task_type=args.task_type,
        model=args.critic_model,
        critic_model=args.critic_model,
        family_expansion_model=args.family_expansion_model,
        family_expansion_enabled=bool(int(args.family_expansion_enabled)),
        family_taxonomy_path=args.family_taxonomy_path,
        use_dual_family_labels=bool(int(args.use_dual_family_labels)),
        family_rejudge_on_low_confidence=bool(int(args.family_rejudge_on_low_confidence)),
        agents=1,
        baseline_only=True,
        out_dir=str(out_dir / "_judge_system"),
        critic_temperature=0.0,
        critic_max_tokens=args.critic_max_tokens,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        transient_retry_forever=bool(int(args.transient_retry_forever)),
        llm_call_timeout=args.llm_call_timeout,
        seed=args.seed,
    )
    system = TextualGradientRLSystem(cfg)
    system.strategy_family_cache = {}

    judgments_path = out_dir / "rejudge_records.jsonl"
    rows: List[Dict[str, Any]] = []
    with judgments_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            trace_key = f"{item['source_run']}::{item['question_hash']}::{item['agent_id']}::{idx}"
            for repeat in range(args.repeats):
                # Clear cache so repeated judgments test judge stability, not cache hits.
                system.strategy_family_cache = {}
                judgment = await system._judge_strategy_family_single(
                    int(item["agent_id"]),
                    str(item["trace"]),
                    answer="",
                    question="",
                )
                finalized = await system._finalize_strategy_family_judgment(
                    judgment,
                    str(item["trace"]),
                    answer="",
                    question="",
                )
                row = {
                    "trace_key": trace_key,
                    "repeat": repeat + 1,
                    "source_run": item["source_run"],
                    "question_hash": item["question_hash"],
                    "agent_id": item["agent_id"],
                    "original_primary_family": item.get("original_primary_family", ""),
                    "original_secondary_family": item.get("original_secondary_family", ""),
                    "primary_family": finalized.get("primary_family", ""),
                    "secondary_family": finalized.get("secondary_family", ""),
                    "confidence": finalized.get("confidence", 0.0),
                    "source": finalized.get("source", ""),
                    "trace_hash": system._strategy_family_cache_key(str(item["trace"])),
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (idx + 1) % 10 == 0:
                print(f"Rejudged {idx + 1}/{len(items)} traces")

    _write_summary(rows, out_dir)
    print(f"Rejudge complete: {out_dir}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
