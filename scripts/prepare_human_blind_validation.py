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


def _parse_model_filter(text: str) -> set[str]:
    return {item.strip() for item in str(text or "").split(",") if item.strip()}


def _run_model_name(run_dir: Path) -> str:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return ""
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg = meta.get("config", {}) if isinstance(meta, dict) and isinstance(meta.get("config", {}), dict) else {}
    return str(cfg.get("model", ""))


def _model_allowed(model: str, include_models: set[str], exclude_models: set[str]) -> bool:
    if include_models and model not in include_models:
        return False
    if exclude_models and model in exclude_models:
        return False
    return True


def _trace_records(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    records = {}
    for rec in read_jsonl(run_dir / "test_trace_history.jsonl"):
        qh = str(rec.get("question_hash", ""))
        agents = rec.get("agents", [])
        if qh and isinstance(agents, list) and agents:
            records[qh] = rec
    return records


def _collect_groups(runs_root: Path, include_models: set[str] | None = None, exclude_models: set[str] | None = None) -> List[Dict[str, Any]]:
    include_models = include_models or set()
    exclude_models = exclude_models or set()
    groups: List[Dict[str, Any]] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]) if runs_root.exists() else []:
        model = _run_model_name(run_dir)
        if not _model_allowed(model, include_models, exclude_models):
            continue
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
                    "model": model,
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


def _trace_texts(group: Dict[str, Any]) -> List[str]:
    agents = group.get("agents", [])
    if not isinstance(agents, list):
        return []
    return [
        str(agent.get("trace", "")).strip()
        for agent in agents
        if isinstance(agent, dict) and str(agent.get("trace", "")).strip()
    ]


def _attach_group_trace_embedding_metrics(groups: List[Dict[str, Any]], model_name: str = DEFAULT_EMBEDDING_MODEL) -> List[Dict[str, Any]]:
    if not groups:
        return groups
    encoder = SentenceEmbeddingEncoder(model_name, enabled=True)
    for group in groups:
        emb_div, emb_sim, emb_count, emb_chunks = pairwise_document_embedding_cosine_diversity(
            _trace_texts(group),
            encoder,
        )
        group["trace_embedding_cosine_diversity"] = emb_div
        group["trace_embedding_cosine_similarity"] = emb_sim
        group["trace_embedding_text_count"] = emb_count
        group["mean_trace_embedding_chunks"] = emb_chunks
        group["trace_embedding_chunk_words"] = 320
        group["trace_embedding_chunk_overlap"] = 40
        group["trace_embedding_model"] = encoder.model_name
        group["trace_embedding_status"] = encoder.status
    return groups


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(max(0, min(len(vals) - 1, round(q * (len(vals) - 1)))))
    return vals[idx]


def _bucket_groups(
    groups: List[Dict[str, Any]],
    text_metric: str = "trace_embedding_cosine_diversity",
    include_text_only_buckets: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    if not groups:
        return {}
    strategy_vals = [safe_float(g.get("team_family_diversity")) for g in groups]
    text_vals = [safe_float(g.get(text_metric)) for g in groups]
    s_low = _quantile(strategy_vals, 0.30)
    s_high = _quantile(strategy_vals, 0.70)
    t_low = _quantile(text_vals, 0.30)
    t_high = _quantile(text_vals, 0.70)
    bucketed: Dict[str, List[Dict[str, Any]]] = {}
    if include_text_only_buckets:
        bucketed["high_text"] = [g for g in groups if safe_float(g.get(text_metric)) >= t_high]
        bucketed["low_text"] = [g for g in groups if safe_float(g.get(text_metric)) <= t_low]
    bucketed.update({
        "high_strategy": [g for g in groups if safe_float(g.get("team_family_diversity")) >= s_high],
        "low_strategy": [g for g in groups if safe_float(g.get("team_family_diversity")) <= s_low],
        "high_text_low_strategy": [
            g for g in groups
            if safe_float(g.get(text_metric)) >= t_high and safe_float(g.get("team_family_diversity")) <= s_low
        ],
        "low_text_high_strategy": [
            g for g in groups
            if safe_float(g.get(text_metric)) <= t_low and safe_float(g.get("team_family_diversity")) >= s_high
        ],
    })
    return bucketed


def _sort_bucket_candidates(bucket: str, groups: List[Dict[str, Any]], text_metric: str) -> List[Dict[str, Any]]:
    def strategy(g: Dict[str, Any]) -> float:
        return safe_float(g.get("team_family_diversity"))

    def major(g: Dict[str, Any]) -> float:
        return safe_float(g.get("team_major_family_diversity"))

    def text(g: Dict[str, Any]) -> float:
        return safe_float(g.get(text_metric))

    def stable(g: Dict[str, Any]) -> Tuple[str, str]:
        return str(g.get("run_name", "")), str(g.get("question_hash", ""))

    if bucket == "high_text":
        return sorted(groups, key=lambda g: (-text(g), -strategy(g), -major(g), stable(g)))
    if bucket == "low_text":
        return sorted(groups, key=lambda g: (text(g), strategy(g), major(g), stable(g)))
    if bucket == "high_text_low_strategy":
        return sorted(groups, key=lambda g: (-text(g), strategy(g), major(g), stable(g)))
    if bucket == "low_text_high_strategy":
        return sorted(groups, key=lambda g: (text(g), -strategy(g), -major(g), stable(g)))
    if bucket == "high_strategy":
        return sorted(groups, key=lambda g: (-strategy(g), -major(g), -text(g), stable(g)))
    if bucket == "low_strategy":
        return sorted(groups, key=lambda g: (strategy(g), major(g), text(g), stable(g)))
    return sorted(groups, key=stable)


def _sample(
    groups_by_bucket: Dict[str, List[Dict[str, Any]]],
    per_bucket: int,
    seed: int,
    sample_mode: str = "random",
    text_metric: str = "trace_embedding_cosine_diversity",
    dedupe_across_buckets: bool = True,
) -> List[Tuple[str, Dict[str, Any]]]:
    rng = random.Random(seed)
    selected: List[Tuple[str, Dict[str, Any]]] = []
    seen = set()
    for bucket, groups in groups_by_bucket.items():
        if sample_mode == "extreme":
            candidates = _sort_bucket_candidates(bucket, list(groups), text_metric)
        else:
            candidates = list(groups)
            rng.shuffle(candidates)
        take = []
        for g in candidates:
            key = (g["run_name"], g["question_hash"])
            if dedupe_across_buckets and key in seen:
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
                    "model": group.get("model", ""),
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
    high_strategy = [r["human_method_diversity_score"] for r in rows if str(r.get("bucket")) == "high_strategy"]
    low_strategy = [r["human_method_diversity_score"] for r in rows if str(r.get("bucket")) == "low_strategy"]
    high_text = [r["human_method_diversity_score"] for r in rows if str(r.get("bucket")) == "high_text"]
    low_text = [r["human_method_diversity_score"] for r in rows if str(r.get("bucket")) == "low_text"]
    delta_ci = bootstrap_mean_ci([h - l for h, l in zip(high_strategy, low_strategy)], iterations=bootstrap_iterations, seed=seed) if high_strategy and low_strategy else {"n": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    text_delta_ci = bootstrap_mean_ci([h - l for h, l in zip(high_text, low_text)], iterations=bootstrap_iterations, seed=seed) if high_text and low_text else {"n": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    summary = {
        "matched_count": len(rows),
        "strategy_tree_vs_human_spearman": corr_strategy,
        "trace_text_vs_human_spearman": corr_text,
        "trace_embedding_vs_human_spearman": corr_embedding,
        "high_strategy_minus_low_strategy_human_score_ci": delta_ci,
        "mean_high_strategy_human_score": safe_mean(high_strategy),
        "mean_low_strategy_human_score": safe_mean(low_strategy),
        "high_text_minus_low_text_human_score_ci": text_delta_ci,
        "mean_high_text_human_score": safe_mean(high_text),
        "mean_low_text_human_score": safe_mean(low_text),
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
    parser.add_argument("--include_models", type=str, default="", help="Comma-separated model names to include.")
    parser.add_argument("--exclude_models", type=str, default="", help="Comma-separated model names to exclude.")
    parser.add_argument("--text_diversity_metric", type=str, default="embedding", choices=["embedding", "token"])
    parser.add_argument("--include_text_only_buckets", type=int, default=1, choices=[0, 1])
    parser.add_argument("--sample_mode", type=str, default="random", choices=["random", "extreme"])
    parser.add_argument("--dedupe_across_buckets", type=int, default=1, choices=[0, 1])
    parser.add_argument("--annotations", type=str, default="", help="Optional completed CSV/JSONL annotations with blinded_id and score.")
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = _collect_groups(
        Path(args.runs_root),
        include_models=_parse_model_filter(args.include_models),
        exclude_models=_parse_model_filter(args.exclude_models),
    )
    text_metric = "trace_embedding_cosine_diversity" if args.text_diversity_metric == "embedding" else "trace_token_cosine_diversity"
    if args.text_diversity_metric == "embedding":
        groups = _attach_group_trace_embedding_metrics(groups, DEFAULT_EMBEDDING_MODEL)
    bucketed = _bucket_groups(groups, text_metric=text_metric, include_text_only_buckets=bool(args.include_text_only_buckets))
    selected = _sample(
        bucketed,
        args.per_bucket,
        args.seed,
        sample_mode=args.sample_mode,
        text_metric=text_metric,
        dedupe_across_buckets=bool(args.dedupe_across_buckets),
    )
    packet_path, key_rows = _write_packet(selected, out_dir, args.seed)
    analysis = _analyze_annotations(key_rows, args.annotations, out_dir, args.bootstrap_iterations, args.seed)
    _write_md(out_dir, groups, bucketed, key_rows, analysis)
    print(f"P7 packet: {packet_path}")
    print(f"P7 key: {out_dir / 'p7_annotation_key.csv'}")


if __name__ == "__main__":
    main()
