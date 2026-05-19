import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in [ROOT, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from prove_experiment_utils import (  # noqa: E402
    bootstrap_mean_ci,
    find_prediction_file,
    DEFAULT_EMBEDDING_MODEL,
    SentenceEmbeddingEncoder,
    pairwise_document_embedding_cosine_diversity,
    pairwise_token_cosine_diversity,
    read_jsonl,
    safe_float,
    safe_mean,
    spearman_corr,
    write_csv,
)


def _trace_records(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    records = {}
    for rec in read_jsonl(run_dir / "test_trace_history.jsonl"):
        qh = str(rec.get("question_hash", ""))
        agents = rec.get("agents", [])
        if qh and isinstance(agents, list) and agents:
            records[qh] = rec
    return records


def _collect_groups(runs_root: Path) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]) if runs_root.exists() else []:
        pred_file = find_prediction_file(run_dir)
        if not pred_file:
            continue
        traces_by_qh = _trace_records(run_dir)
        for pred in read_jsonl(pred_file):
            qh = str(pred.get("question_hash", ""))
            trace_rec = traces_by_qh.get(qh)
            if not trace_rec:
                continue
            agents = trace_rec.get("agents", [])
            trace_texts = [str(a.get("trace", "")).strip() for a in agents if isinstance(a, dict) and str(a.get("trace", "")).strip()]
            if len(trace_texts) < 2:
                continue
            text_div, text_sim = pairwise_token_cosine_diversity(trace_texts)
            groups.append(
                {
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "question_hash": qh,
                    "team_family_diversity": safe_float(pred.get("team_family_diversity")),
                    "team_family_homogeneity_rate": safe_float(pred.get("team_family_homogeneity_rate")),
                    "team_major_family_diversity": safe_float(pred.get("team_major_family_diversity")),
                    "trace_token_cosine_diversity": text_div,
                    "trace_token_cosine_similarity": text_sim,
                    "vote_correct": safe_float(pred.get("vote_correct")),
                    "agents": [
                        {
                            "agent_id": int(a.get("agent_id", i)) if isinstance(a, dict) else i,
                            "trace": str(a.get("trace", "")) if isinstance(a, dict) else "",
                        }
                        for i, a in enumerate(agents)
                        if isinstance(a, dict)
                    ],
                }
            )
    return groups


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(max(0, min(len(vals) - 1, round(q * (len(vals) - 1)))))
    return vals[idx]


def _bucket_groups(groups: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    if not groups:
        return {}
    strategy_vals = [safe_float(g.get("team_family_diversity")) for g in groups]
    text_vals = [safe_float(g.get("trace_token_cosine_diversity")) for g in groups]
    s_low = _quantile(strategy_vals, 0.30)
    s_high = _quantile(strategy_vals, 0.70)
    t_low = _quantile(text_vals, 0.30)
    t_high = _quantile(text_vals, 0.70)
    return {
        "high_strategy": [g for g in groups if safe_float(g.get("team_family_diversity")) >= s_high],
        "low_strategy": [g for g in groups if safe_float(g.get("team_family_diversity")) <= s_low],
        "high_text_low_strategy": [
            g for g in groups
            if safe_float(g.get("trace_token_cosine_diversity")) >= t_high and safe_float(g.get("team_family_diversity")) <= s_low
        ],
        "low_text_high_strategy": [
            g for g in groups
            if safe_float(g.get("trace_token_cosine_diversity")) <= t_low and safe_float(g.get("team_family_diversity")) >= s_high
        ],
    }


def _sample(groups_by_bucket: Dict[str, List[Dict[str, Any]]], per_bucket: int, seed: int) -> List[Tuple[str, Dict[str, Any]]]:
    rng = random.Random(seed)
    selected: List[Tuple[str, Dict[str, Any]]] = []
    seen = set()
    for bucket, groups in groups_by_bucket.items():
        candidates = list(groups)
        rng.shuffle(candidates)
        take = []
        for g in candidates:
            key = (g["run_name"], g["question_hash"])
            if key in seen:
                continue
            seen.add(key)
            take.append(g)
            if len(take) >= per_bucket:
                break
        selected.extend((bucket, g) for g in take)
    rng.shuffle(selected)
    return selected


def _write_packet(selected: List[Tuple[str, Dict[str, Any]]], out_dir: Path, seed: int):
    packet_path = out_dir / "p7_blind_annotation_packet.jsonl"
    key_rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    encoder = SentenceEmbeddingEncoder(DEFAULT_EMBEDDING_MODEL, enabled=True)
    with packet_path.open("w", encoding="utf-8") as f:
        for idx, (bucket, group) in enumerate(selected, start=1):
            blinded_id = f"P7G{idx:04d}"
            agents = list(group.get("agents", []))
            rng.shuffle(agents)
            trace_texts = [
                str(agent.get("trace", "")).strip()
                for agent in agents
                if isinstance(agent, dict) and str(agent.get("trace", "")).strip()
            ]
            trace_emb_div, trace_emb_sim, trace_emb_count, trace_emb_chunks = pairwise_document_embedding_cosine_diversity(
                trace_texts,
                encoder,
            )
            blind_agents = [
                {
                    "agent_alias": f"A{j + 1}",
                    "trace": str(agent.get("trace", "")),
                }
                for j, agent in enumerate(agents)
            ]
            f.write(
                json.dumps(
                    {
                        "blinded_id": blinded_id,
                        "traces": blind_agents,
                        "annotation_fields": {
                            "human_method_diversity_score_1_to_5": "",
                            "coarse_method_tags_optional": "",
                            "notes_optional": "",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            key_rows.append(
                {
                    "blinded_id": blinded_id,
                    "bucket": bucket,
                    "run_name": group.get("run_name", ""),
                    "run_dir": group.get("run_dir", ""),
                    "question_hash": group.get("question_hash", ""),
                    "team_family_diversity": group.get("team_family_diversity", 0.0),
                    "team_family_homogeneity_rate": group.get("team_family_homogeneity_rate", 0.0),
                    "team_major_family_diversity": group.get("team_major_family_diversity", 0.0),
                    "trace_token_cosine_diversity": group.get("trace_token_cosine_diversity", 0.0),
                    "trace_token_cosine_similarity": group.get("trace_token_cosine_similarity", 0.0),
                    "trace_embedding_cosine_diversity": trace_emb_div,
                    "trace_embedding_cosine_similarity": trace_emb_sim,
                    "trace_embedding_text_count": trace_emb_count,
                    "mean_trace_embedding_chunks": trace_emb_chunks,
                    "trace_embedding_chunk_words": 320,
                    "trace_embedding_chunk_overlap": 40,
                    "trace_embedding_model": encoder.model_name,
                    "trace_embedding_status": encoder.status,
                    "vote_correct": group.get("vote_correct", 0.0),
                }
            )
    write_csv(key_rows, out_dir / "p7_annotation_key.csv")
    return packet_path, key_rows


def _read_annotations(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    out = {}
    for row in rows:
        bid = str(row.get("blinded_id", "")).strip()
        if bid:
            out[bid] = row
    return out


def _analyze_annotations(key_rows: List[Dict[str, Any]], annotations_path: str, out_dir: Path, bootstrap_iterations: int, seed: int):
    if not annotations_path:
        return None
    annotations = _read_annotations(Path(annotations_path))
    rows: List[Dict[str, Any]] = []
    for key in key_rows:
        ann = annotations.get(str(key.get("blinded_id", "")))
        if not ann:
            continue
        score = ann.get("human_method_diversity_score", ann.get("human_method_diversity_score_1_to_5", ann.get("score")))
        rows.append({**key, "human_method_diversity_score": safe_float(score), "annotation_notes": ann.get("notes_optional", "")})
    write_csv(rows, out_dir / "p7_human_annotation_analysis_rows.csv")
    if not rows:
        return {"matched_count": 0}
    corr_strategy = spearman_corr([r["team_family_diversity"] for r in rows], [r["human_method_diversity_score"] for r in rows])
    corr_text = spearman_corr([r["trace_token_cosine_diversity"] for r in rows], [r["human_method_diversity_score"] for r in rows])
    corr_embedding = spearman_corr([r["trace_embedding_cosine_diversity"] for r in rows], [r["human_method_diversity_score"] for r in rows])
    high = [r["human_method_diversity_score"] for r in rows if str(r.get("bucket")) in {"high_strategy", "low_text_high_strategy"}]
    low = [r["human_method_diversity_score"] for r in rows if str(r.get("bucket")) in {"low_strategy", "high_text_low_strategy"}]
    delta_ci = bootstrap_mean_ci([h - l for h, l in zip(high, low)], iterations=bootstrap_iterations, seed=seed) if high and low else {"n": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    summary = {
        "matched_count": len(rows),
        "strategy_tree_vs_human_spearman": corr_strategy,
        "trace_text_vs_human_spearman": corr_text,
        "trace_embedding_vs_human_spearman": corr_embedding,
        "high_strategy_minus_low_strategy_human_score_ci": delta_ci,
        "mean_high_strategy_human_score": safe_mean(high),
        "mean_low_strategy_human_score": safe_mean(low),
    }
    (out_dir / "p7_human_annotation_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _write_md(out_dir: Path, groups: List[Dict[str, Any]], bucketed: Dict[str, List[Dict[str, Any]]], key_rows: List[Dict[str, Any]], analysis: Any):
    lines = [
        "# P7 Human Blind Validation",
        "",
        f"- candidate_groups: {len(groups)}",
        f"- sampled_groups: {len(key_rows)}",
        "",
        "## Bucket Counts",
        "",
    ]
    for bucket, vals in bucketed.items():
        sampled = sum(1 for r in key_rows if r.get("bucket") == bucket)
        lines.append(f"- {bucket}: candidates={len(vals)}, sampled={sampled}")
    lines.extend(["", "## Files", "", "- p7_blind_annotation_packet.jsonl", "- p7_annotation_key.csv"])
    if analysis:
        lines.extend(
            [
                "",
                "## Human Annotation Analysis",
                "",
                f"- matched_count: {analysis.get('matched_count', 0)}",
            ]
        )
        if isinstance(analysis.get("strategy_tree_vs_human_spearman"), dict):
            lines.append(f"- strategy_tree_vs_human_spearman: {safe_float(analysis['strategy_tree_vs_human_spearman'].get('rho')):.4f}")
        if isinstance(analysis.get("trace_text_vs_human_spearman"), dict):
            lines.append(f"- trace_text_vs_human_spearman: {safe_float(analysis['trace_text_vs_human_spearman'].get('rho')):.4f}")
        if isinstance(analysis.get("trace_embedding_vs_human_spearman"), dict):
            lines.append(f"- trace_embedding_vs_human_spearman: {safe_float(analysis['trace_embedding_vs_human_spearman'].get('rho')):.4f}")
    (out_dir / "p7_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="P7: prepare blind human annotation packet and optionally analyze annotations.")
    parser.add_argument("--runs_root", type=str, default="prove_experiments/runs")
    parser.add_argument("--out_dir", type=str, default="prove_experiments/p7_human_blind")
    parser.add_argument("--per_bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotations", type=str, default="", help="Optional completed CSV/JSONL annotations with blinded_id and score.")
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = _collect_groups(Path(args.runs_root))
    bucketed = _bucket_groups(groups)
    selected = _sample(bucketed, args.per_bucket, args.seed)
    packet_path, key_rows = _write_packet(selected, out_dir, args.seed)
    analysis = _analyze_annotations(key_rows, args.annotations, out_dir, args.bootstrap_iterations, args.seed)
    _write_md(out_dir, groups, bucketed, key_rows, analysis)
    print(f"P7 packet: {packet_path}")
    print(f"P7 key: {out_dir / 'p7_annotation_key.csv'}")


if __name__ == "__main__":
    main()
