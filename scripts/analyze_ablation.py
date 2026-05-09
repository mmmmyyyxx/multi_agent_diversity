import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except json.JSONDecodeError:
                continue
    return out


def _find_latest_test_predictions(run_dir: Path) -> Optional[Path]:
    files = sorted(run_dir.glob("test_epoch*_predictions.jsonl"))
    if files:
        return files[-1]
    return None


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def _find_recent_trace_snapshots(run_dir: Path, limit: int = 10) -> List[List[Dict[str, Any]]]:
    candidates = [
        run_dir / "train_trace_history.jsonl",
        run_dir / "test_trace_history.jsonl",
        run_dir / "reasoning_summary_history.jsonl",
    ]
    records: List[Dict[str, Any]] = []
    for path in candidates:
        records = _read_jsonl(path)
        if records:
            break
    if not records:
        return []

    out: List[List[Dict[str, Any]]] = []
    for rec in records[-max(1, limit):]:
        if not isinstance(rec, dict):
            continue
        agents = rec.get("agents", [])
        if not isinstance(agents, list):
            continue
        out.append([x for x in agents if isinstance(x, dict)])
    return out


def _find_recent_summary_snapshots(run_dir: Path, limit: int = 10) -> List[List[Dict[str, Any]]]:
    records = _read_jsonl(run_dir / "reasoning_summary_history.jsonl")
    if not records:
        return []

    out: List[List[Dict[str, Any]]] = []
    for rec in records[-max(1, limit):]:
        if not isinstance(rec, dict):
            continue
        agents = rec.get("agents", [])
        if not isinstance(agents, list):
            continue
        out.append([x for x in agents if isinstance(x, dict)])
    return out


def _extract_final_prompt_strings(prompt_history: Any) -> List[str]:
    prompts: List[str] = []
    if not isinstance(prompt_history, dict):
        return prompts

    for agent in prompt_history.values():
        if not isinstance(agent, dict):
            continue

        events = agent.get("events", [])
        if isinstance(events, list) and events:
            last_prompt = None
            for event in reversed(events):
                if isinstance(event, dict) and event.get("current_prompt"):
                    last_prompt = str(event.get("current_prompt", ""))
                    break
            if last_prompt:
                prompts.append(last_prompt)
                continue

        current_prompt = str(agent.get("current_prompt", ""))
        if current_prompt:
            prompts.append(current_prompt)

    return prompts


def _pairwise_mismatch_rate(items: List[str]) -> float:
    n = len(items)
    if n < 2:
        return 0.0

    total_pairs = n * (n - 1) / 2
    mismatched_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            mismatched_pairs += int(items[i] != items[j])
    return float(mismatched_pairs / total_pairs)


def _extract_final_trace_strings(trace_agents: List[Dict[str, Any]]) -> List[str]:
    traces: List[str] = []
    for a in trace_agents:
        trace = str(a.get("trace", "")).strip()
        if not trace:
            trace = str(a.get("reasoning_summary", "")).strip()
        if trace:
            traces.append(trace)
    return traces


def _extract_final_reasoning_summary_strings(trace_agents: List[Dict[str, Any]]) -> List[str]:
    summaries: List[str] = []
    for a in trace_agents:
        summary = str(a.get("reasoning_summary", "")).strip()
        if summary:
            summaries.append(summary)
    return summaries


def _tokenize_for_cosine(text: str) -> List[str]:
    # 使用轻量 bag-of-words，避免额外依赖。
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _cosine_sim_from_tokens(xs: List[str], ys: List[str]) -> float:
    if not xs and not ys:
        return 1.0
    if not xs or not ys:
        return 0.0

    cx = Counter(xs)
    cy = Counter(ys)
    keys = set(cx.keys()) | set(cy.keys())
    dot = sum(float(cx[k] * cy[k]) for k in keys)
    nx = math.sqrt(sum(float(v * v) for v in cx.values()))
    ny = math.sqrt(sum(float(v * v) for v in cy.values()))
    if nx <= 0.0 or ny <= 0.0:
        return 0.0
    sim = float(dot / (nx * ny))
    return float(max(0.0, min(1.0, sim)))


def _pairwise_cosine_diversity(texts: List[str]) -> Tuple[float, float]:
    n = len(texts)
    if n < 2:
        return 0.0, 1.0

    tokenized = [_tokenize_for_cosine(t) for t in texts]
    sims: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(_cosine_sim_from_tokens(tokenized[i], tokenized[j]))

    mean_sim = _safe_mean(sims)
    mean_sim = float(max(0.0, min(1.0, mean_sim)))
    return float(max(0.0, min(1.0, 1.0 - mean_sim))), mean_sim


def _collect_eval_metrics(pred_records: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(pred_records)
    if n == 0:
        return {
            "eval_size": 0,
            "disagreement_rate": 0.0,
            "mean_family_diversity_from_preds": 0.0,
            "mean_family_homogeneity_rate_from_preds": 0.0,
            "mean_llm_direct_diversity_score_from_preds": 0.0,
            "mean_vote_acc_from_preds": 0.0,
            "all_same_primary_rate_from_preds": 0.0,
            "all_same_pair_rate_from_preds": 0.0,
            "mean_family_confidence_from_preds": 0.0,
            "low_confidence_share_from_preds": 0.0,
            "mean_rejudge_count_from_preds": 0.0,
            "mean_summary_words_from_preds": 0.0,
            "mean_summary_tokens_from_preds": 0.0,
            "mean_evidence_spans_from_preds": 0.0,
        }

    disagreement = []
    family_div_all = []
    family_homo_all = []
    direct_div_all = []
    vote_acc_all = []
    all_same_primary_all = []
    all_same_pair_all = []
    confidence_all = []
    low_conf_all = []
    rejudge_all = []
    summary_words_all = []
    summary_tokens_all = []
    evidence_spans_all = []

    for r in pred_records:
        answers = r.get("answers", [])
        if not isinstance(answers, list):
            answers = []
        div = float(r.get("team_family_diversity", 0.0))
        family_homo_rate = float(r.get("team_family_homogeneity_rate", 0.0))
        direct_div = r.get("llm_direct_diversity_score", None)
        vote_correct = float(r.get("vote_correct", 0.0))
        dg = int(len(set(map(str, answers))) > 1) if answers else 0
        disagreement.append(dg)
        family_div_all.append(div)
        family_homo_all.append(family_homo_rate)
        if direct_div is not None:
            direct_div_all.append(float(direct_div))
        vote_acc_all.append(vote_correct)
        all_same_primary_all.append(int(bool(r.get("all_same_primary", False))))
        all_same_pair_all.append(int(bool(r.get("all_same_pair", False))))
        confidence_all.append(float(r.get("mean_family_confidence", 0.0) or 0.0))
        low_conf_all.append(float(r.get("low_confidence_share", 0.0) or 0.0))
        rejudge_all.append(float(r.get("rejudge_count", 0.0) or 0.0))
        summary_words_all.append(float(r.get("mean_summary_words", 0.0) or 0.0))
        summary_tokens_all.append(float(r.get("mean_summary_tokens", 0.0) or 0.0))
        evidence_spans_all.append(float(r.get("mean_evidence_spans", 0.0) or 0.0))

    return {
        "eval_size": n,
        "disagreement_rate": _safe_mean(disagreement),
        "mean_family_diversity_from_preds": _safe_mean(family_div_all),
        "mean_family_homogeneity_rate_from_preds": _safe_mean(family_homo_all),
        "mean_llm_direct_diversity_score_from_preds": _safe_mean(direct_div_all),
        "mean_vote_acc_from_preds": _safe_mean(vote_acc_all),
        "all_same_primary_rate_from_preds": _safe_mean(all_same_primary_all),
        "all_same_pair_rate_from_preds": _safe_mean(all_same_pair_all),
        "mean_family_confidence_from_preds": _safe_mean(confidence_all),
        "low_confidence_share_from_preds": _safe_mean(low_conf_all),
        "mean_rejudge_count_from_preds": _safe_mean(rejudge_all),
        "mean_summary_words_from_preds": _safe_mean(summary_words_all),
        "mean_summary_tokens_from_preds": _safe_mean(summary_tokens_all),
        "mean_evidence_spans_from_preds": _safe_mean(evidence_spans_all),
    }


def _collect_update_metrics(step_records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not step_records:
        return {
            "train_steps": 0,
            "update_requested_rate": 0.0,
            "update_ready_rate": 0.0,
            "update_selected_rate": 0.0,
            "update_applied_rate": 0.0,
            "mean_selected_agents": 0.0,
            "mean_updated_agents": 0.0,
            "mean_generic_prompt_candidate_rate": 0.0,
            "mean_family_shift_rate_during_candidate_eval": 0.0,
            "mean_summary_embedding_shift_during_candidate_eval": 0.0,
        }

    update_requested = []
    update_ready = []
    update_selected = []
    update_applied = []
    num_selected = []
    num_updated = []
    generic_rates = []
    family_shift_rates = []
    summary_shifts = []

    for r in step_records:
        u = r.get("update", {}) if isinstance(r.get("update", {}), dict) else {}
        req = int(bool(u.get("update_requested", False)))
        ready = int(bool(u.get("update_ready", False)))
        selected_ids = u.get("selected_agent_ids", [])
        updated_ids = u.get("updated_agent_ids", [])
        if not isinstance(selected_ids, list):
            selected_ids = []
        if not isinstance(updated_ids, list):
            updated_ids = []

        update_requested.append(req)
        update_ready.append(ready)
        update_selected.append(int(len(selected_ids) > 0))
        update_applied.append(int(len(updated_ids) > 0))
        num_selected.append(float(len(selected_ids)))
        num_updated.append(float(len(updated_ids)))
        generic_rates.append(float(r.get("generic_prompt_candidate_rate", 0.0) or 0.0))
        diag = r.get("candidate_behavior_diagnostics", {})
        if isinstance(diag, dict):
            family_shift_rates.append(float(diag.get("family_shift_rate", 0.0) or 0.0))
            summary_shifts.append(float(diag.get("summary_embedding_shift", 0.0) or 0.0))

    return {
        "train_steps": len(step_records),
        "update_requested_rate": _safe_mean(update_requested),
        "update_ready_rate": _safe_mean(update_ready),
        "update_selected_rate": _safe_mean(update_selected),
        "update_applied_rate": _safe_mean(update_applied),
        "mean_selected_agents": _safe_mean(num_selected),
        "mean_updated_agents": _safe_mean(num_updated),
        "mean_generic_prompt_candidate_rate": _safe_mean(generic_rates),
        "mean_family_shift_rate_during_candidate_eval": _safe_mean(family_shift_rates),
        "mean_summary_embedding_shift_during_candidate_eval": _safe_mean(summary_shifts),
    }


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    run_meta = _read_json(run_dir / "run_meta.json") or {}
    history = _read_json(run_dir / "history.json") or []
    last_state = _read_json(run_dir / "last_state.json") or {}
    step_logs = _read_jsonl(run_dir / "train_step_logs.jsonl")
    trace_snapshots = _find_recent_trace_snapshots(run_dir, limit=10)
    summary_snapshots = _find_recent_summary_snapshots(run_dir, limit=10)
    pred_path = _find_latest_test_predictions(run_dir)
    pred_records = _read_jsonl(pred_path) if pred_path else []
    prompt_history = _read_json(run_dir / "prompt_history.json") or {}

    latest_train_metrics: Dict[str, Any] = {}
    latest_test_metrics: Dict[str, Any] = {}
    if isinstance(history, list) and history:
        latest = history[-1]
        if isinstance(latest, dict):
            if isinstance(latest.get("train", {}), dict) or isinstance(latest.get("test", {}), dict):
                latest_train_metrics = latest.get("train", {}) if isinstance(latest.get("train", {}), dict) else {}
                latest_test_metrics = latest.get("test", {}) if isinstance(latest.get("test", {}), dict) else {}

    cfg = run_meta.get("config", {}) if isinstance(run_meta.get("config", {}), dict) else {}
    lam_div = float(cfg.get("lambda_diversity", 0.0))
    diversity_reward_enabled = int(lam_div > 0.0)

    agents = last_state.get("agents", []) if isinstance(last_state.get("agents", []), list) else []
    prompt_drift_flags = []
    prompt_drift_cos_distances = []
    for a in agents:
        if not isinstance(a, dict):
            continue

        init_prompt = _normalize_spaces(str(a.get("initial_prompt", "")))
        cur_prompt = _normalize_spaces(str(a.get("current_prompt", "")))
        if init_prompt and cur_prompt:
            prompt_drift_flags.append(int(init_prompt != cur_prompt))
            sim = _cosine_sim_from_tokens(_tokenize_for_cosine(init_prompt), _tokenize_for_cosine(cur_prompt))
            prompt_drift_cos_distances.append(float(max(0.0, min(1.0, 1.0 - sim))))
            continue

        init_h = str(a.get("initial_prompt_hash", ""))
        cur_h = str(a.get("current_prompt_hash", ""))
        prompt_drift_flags.append(int(init_h != "" and cur_h != "" and init_h != cur_h))

    final_prompt_strings = _extract_final_prompt_strings(prompt_history)
    prompt_diversity_rate = _pairwise_mismatch_rate(final_prompt_strings)
    prompt_cos_div, prompt_cos_sim = _pairwise_cosine_diversity(final_prompt_strings)

    trace_cos_div_all: List[float] = []
    trace_cos_sim_all: List[float] = []
    final_trace_strings: List[str] = []
    for i, snapshot_agents in enumerate(trace_snapshots):
        trace_strings = _extract_final_trace_strings(snapshot_agents)
        div_i, sim_i = _pairwise_cosine_diversity(trace_strings)
        trace_cos_div_all.append(div_i)
        trace_cos_sim_all.append(sim_i)
        if i == len(trace_snapshots) - 1:
            final_trace_strings = trace_strings

    trace_cos_div = _safe_mean(trace_cos_div_all)
    trace_cos_sim = _safe_mean(trace_cos_sim_all)

    summary_cos_div_all: List[float] = []
    summary_cos_sim_all: List[float] = []
    final_summary_strings: List[str] = []
    for i, snapshot_agents in enumerate(summary_snapshots):
        summary_strings = _extract_final_reasoning_summary_strings(snapshot_agents)
        div_i, sim_i = _pairwise_cosine_diversity(summary_strings)
        summary_cos_div_all.append(div_i)
        summary_cos_sim_all.append(sim_i)
        if i == len(summary_snapshots) - 1:
            final_summary_strings = summary_strings

    summary_cos_div = _safe_mean(summary_cos_div_all)
    summary_cos_sim = _safe_mean(summary_cos_sim_all)

    setting = run_dir.name
    cfg_baseline_only = bool(cfg.get("baseline_only", False))
    baseline_only = int(setting.endswith("_testonly") or cfg_baseline_only)

    eval_metrics = _collect_eval_metrics(pred_records)
    final_test_mean_family_diversity = float(
        latest_test_metrics.get("mean_family_diversity", eval_metrics["mean_family_diversity_from_preds"]) or 0.0
    )
    final_test_mean_family_homogeneity_rate = float(
        latest_test_metrics.get("mean_family_homogeneity_rate", eval_metrics["mean_family_homogeneity_rate_from_preds"]) or 0.0
    )
    final_test_mean_llm_direct_diversity_score = float(
        latest_test_metrics.get("mean_llm_direct_diversity_score", eval_metrics["mean_llm_direct_diversity_score_from_preds"]) or 0.0
    )
    final_train_mean_llm_direct_diversity_score = float(latest_train_metrics.get("mean_llm_direct_diversity_score", 0.0) or 0.0) if latest_train_metrics else 0.0
    final_test_vote_acc = float(
        latest_test_metrics.get("vote_acc", eval_metrics["mean_vote_acc_from_preds"]) or 0.0
    )

    out = {
        "run_dir": str(run_dir),
        "setting": setting,
        "baseline_only": baseline_only,
        "init_mode": str(run_meta.get("init_mode", "unknown")),
        "all_agents_shared_origin": int(bool(run_meta.get("all_agents_shared_origin", False))),
        "agents": int(run_meta.get("agents", cfg.get("agents", 0) or 0)),
        "update_every": int(run_meta.get("update_every", cfg.get("update_every", 0) or 0)),
        "epochs": int(cfg.get("epochs", 0) or 0),
        "train_size": int(cfg.get("train_size", 0) or 0),
        "test_size": int(cfg.get("test_size", 0) or 0),
        "lambda_diversity": lam_div,
        "diversity_reward_enabled": diversity_reward_enabled,
        "final_test_mean_family_diversity": final_test_mean_family_diversity,
        "final_test_mean_family_homogeneity_rate": final_test_mean_family_homogeneity_rate,
        "final_train_mean_llm_direct_diversity_score": final_train_mean_llm_direct_diversity_score,
        "final_test_mean_llm_direct_diversity_score": final_test_mean_llm_direct_diversity_score,
        "final_train_vote_acc": float(latest_train_metrics.get("vote_acc", 0.0) or 0.0) if latest_train_metrics else None,
        "final_test_vote_acc": final_test_vote_acc,
        "all_same_primary_rate": eval_metrics.get("all_same_primary_rate_from_preds", 0.0),
        "all_same_pair_rate": eval_metrics.get("all_same_pair_rate_from_preds", 0.0),
        "mean_family_confidence": eval_metrics.get("mean_family_confidence_from_preds", 0.0),
        "low_confidence_share": eval_metrics.get("low_confidence_share_from_preds", 0.0),
        "mean_rejudge_count": eval_metrics.get("mean_rejudge_count_from_preds", 0.0),
        "mean_summary_words": eval_metrics.get("mean_summary_words_from_preds", 0.0),
        "mean_summary_tokens": eval_metrics.get("mean_summary_tokens_from_preds", 0.0),
        "mean_evidence_spans": eval_metrics.get("mean_evidence_spans_from_preds", 0.0),
        "prompt_drift_rate": _safe_mean(prompt_drift_flags),
        "prompt_drift_cosine_distance": _safe_mean(prompt_drift_cos_distances),
    }
    out.update(eval_metrics)
    out.update(_collect_update_metrics(step_logs))

    # For test-only baseline runs, these training/update metrics are not applicable.
    if baseline_only and not step_logs:
        out["train_steps"] = None
        out["update_requested_rate"] = None
        out["update_ready_rate"] = None
        out["update_selected_rate"] = None
        out["update_applied_rate"] = None
        out["mean_selected_agents"] = None
        out["mean_updated_agents"] = None

    if out.get("eval_size", 0) == 0 and isinstance(latest_test_metrics.get("size", None), (int, float)):
        out["eval_size"] = int(latest_test_metrics.get("size", 0) or 0)

    out["latest_test_prediction_file"] = str(pred_path) if pred_path else ""
    out["final_prompt_diversity_rate"] = prompt_diversity_rate
    out["final_prompt_exact_diversity_rate"] = prompt_diversity_rate
    out["final_prompt_count"] = len(final_prompt_strings)
    out["final_prompt_cosine_diversity"] = prompt_cos_div
    out["final_prompt_cosine_similarity"] = prompt_cos_sim
    out["final_trace_count"] = len(final_trace_strings)
    out["final_trace_cosine_diversity"] = trace_cos_div
    out["final_trace_cosine_similarity"] = trace_cos_sim
    out["trace_cosine_window_used"] = len(trace_snapshots)
    out["final_reasoning_summary_count"] = len(final_summary_strings)
    out["final_reasoning_summary_cosine_diversity"] = summary_cos_div
    out["final_reasoning_summary_cosine_similarity"] = summary_cos_sim
    out["reasoning_summary_cosine_window_used"] = len(summary_snapshots)
    return out


def _to_float_str(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_markdown(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        path.write_text("# Ablation Summary\n\nNo valid runs found.\n", encoding="utf-8")
        return

    columns = [
        "run_dir",
        "setting",
        "baseline_only",
        "init_mode",
        "diversity_reward_enabled",
        "final_prompt_cosine_diversity",
        "final_trace_cosine_diversity",
        "final_reasoning_summary_cosine_diversity",
        "final_test_mean_family_diversity",
        "final_test_mean_family_homogeneity_rate",
        "final_train_mean_llm_direct_diversity_score",
        "final_test_mean_llm_direct_diversity_score",
        "final_train_vote_acc",
        "final_test_vote_acc",
        "all_same_primary_rate",
        "all_same_pair_rate",
        "mean_family_confidence",
        "low_confidence_share",
        "mean_summary_words",
        "mean_evidence_spans",
        "disagreement_rate",
        "prompt_drift_cosine_distance",
        "update_applied_rate",
        "mean_generic_prompt_candidate_rate",
        "mean_family_shift_rate_during_candidate_eval",
        "mean_summary_embedding_shift_during_candidate_eval",
    ]
    lines = ["# Ablation Summary", "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_to_float_str(r.get(c, "")) for c in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Aggregate ablation experiment outputs.")
    parser.add_argument("--runs", nargs="*", default=[], help="Explicit run directories to analyze.")
    parser.add_argument("--runs_root", type=str, default="", help="Root directory containing run sub-directories.")
    parser.add_argument("--out_csv", type=str, default="", help="Output CSV path.")
    parser.add_argument("--out_md", type=str, default="", help="Output Markdown summary path.")
    args = parser.parse_args()

    run_dirs: List[Path] = []
    for x in args.runs:
        p = Path(x)
        if p.exists() and p.is_dir():
            run_dirs.append(p)
    if args.runs_root:
        root = Path(args.runs_root)
        if root.exists() and root.is_dir():
            for p in sorted(root.iterdir()):
                if p.is_dir() and (p / "run_meta.json").exists():
                    run_dirs.append(p)

    dedup = []
    seen = set()
    for p in run_dirs:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            dedup.append(p)
    run_dirs = dedup

    rows = [analyze_run(p) for p in run_dirs]
    rows = sorted(rows, key=lambda r: (r.get("init_mode", ""), int(r.get("diversity_reward_enabled", 0)), r.get("run_dir", "")))

    if args.out_csv:
        out_csv = Path(args.out_csv)
    elif args.runs_root:
        out_csv = Path(args.runs_root) / "ablation_summary.csv"
    else:
        out_csv = Path("ablation_summary.csv")

    if args.out_md:
        out_md = Path(args.out_md)
    elif args.runs_root:
        out_md = Path(args.runs_root) / "ablation_summary.md"
    else:
        out_md = Path("ablation_summary.md")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            f.write("run_dir\n")

    write_markdown(rows, out_md)

    print(f"Analyzed runs: {len(rows)}")
    print(f"CSV: {out_csv}")
    print(f"Markdown: {out_md}")


if __name__ == "__main__":
    main()
