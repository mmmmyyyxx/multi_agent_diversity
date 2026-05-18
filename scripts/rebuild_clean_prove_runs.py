import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import TextualGradientRLSystem
from multi_dataset_diverse_rl.utils import (
    ensure_dir,
    extract_question_answer,
    load_jsonl,
    normalize_spaces,
    parse_gold,
    set_seed,
)


INVALID_THRESHOLD = 0.35
MIN_TRACE_CHARS = 80
REPEAT_THRESHOLD = 0.35


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def _write_jsonl(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _trace_repeat_ratio(text: str) -> float:
    tokens = re.findall(r"\w+", normalize_spaces(text).lower())
    if len(tokens) < 12:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def _question_hash(question: str) -> str:
    return hashlib.sha1(normalize_spaces(question).encode("utf-8")).hexdigest()[:12]


def _config_from_run_meta(cfg_data: Dict[str, Any], out_dir: Path, args: argparse.Namespace) -> Config:
    allowed = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in dict(cfg_data).items() if k in allowed}
    filtered["out_dir"] = str(out_dir)

    # The cleaned rebuild should use one deterministic judge configuration, even
    # if the original P4 runs mixed low-confidence rejudge settings.
    overrides = {
        "task_type": args.task_type,
        "critic_model": args.critic_model,
        "family_expansion_model": args.family_expansion_model,
        "family_taxonomy_path": args.family_taxonomy_path,
        "family_expansion_enabled": bool(args.family_expansion_enabled),
        "use_dual_family_labels": bool(args.use_dual_family_labels),
        "family_rejudge_on_low_confidence": bool(args.family_rejudge_on_low_confidence),
        "family_confidence_threshold": args.family_confidence_threshold,
        "critic_max_tokens": args.critic_max_tokens,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "critic_temperature": args.critic_temperature,
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "llm_call_timeout": args.llm_call_timeout,
        "seed": args.seed,
    }
    for key, value in overrides.items():
        if key in allowed and str(value) != "auto":
            filtered[key] = value

    if args.solver_api_key_env:
        filtered["solver_api_key_env"] = args.solver_api_key_env
    elif str(filtered.get("solver_api_key_env", "") or "").strip() and not os.getenv(str(filtered.get("solver_api_key_env", ""))):
        filtered["solver_api_key_env"] = ""
    if filtered.get("solver_api_key_env", "") == "":
        filtered.pop("solver_api_key_env", None)
    if args.solver_base_url_env:
        filtered["solver_base_url_env"] = args.solver_base_url_env
    elif str(filtered.get("solver_base_url_env", "") or "").strip() and not os.getenv(str(filtered.get("solver_base_url_env", ""))):
        filtered["solver_base_url_env"] = ""
    if filtered.get("solver_base_url_env", "") == "":
        filtered.pop("solver_base_url_env", None)

    if args.critic_api_key_env:
        filtered["critic_api_key_env"] = args.critic_api_key_env
    if args.critic_base_url_env:
        filtered["critic_base_url_env"] = args.critic_base_url_env

    return Config(**filtered)


def _write_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _invalid_trace_flags(trace: str, answer: str) -> Dict[str, Any]:
    cleaned = normalize_spaces(trace)
    flags: Dict[str, Any] = {
        "empty": False,
        "short": False,
        "no_final_answer": False,
        "repeat": False,
        "answer_missing": False,
        "penalty": 0.0,
        "bad": False,
    }
    if not cleaned:
        flags["empty"] = True
        flags["penalty"] = 1.0
        flags["bad"] = True
        return flags

    penalty = 0.0
    if len(cleaned) < MIN_TRACE_CHARS:
        flags["short"] = True
        penalty += 0.35
    if not re.search(r"FINAL_ANSWER\s*:", cleaned, flags=re.IGNORECASE):
        flags["no_final_answer"] = True
        penalty += 0.25
    repeat_ratio = _trace_repeat_ratio(cleaned)
    if repeat_ratio > REPEAT_THRESHOLD:
        flags["repeat"] = True
        penalty += min(0.25, repeat_ratio)
    if str(answer or "").strip() == "":
        flags["answer_missing"] = True
        penalty += 0.25
    flags["penalty"] = float(min(1.0, penalty))
    flags["bad"] = bool(flags["penalty"] >= INVALID_THRESHOLD)
    return flags


def _latest_jsonl(path_candidates: List[Path]) -> Optional[Path]:
    files = [p for p in path_candidates if p.exists()]
    if not files:
        return None
    return sorted(files)[-1]


def _load_question_bank(test_path: str, test_size: int) -> Dict[str, Dict[str, str]]:
    bank: Dict[str, Dict[str, str]] = {}
    for row in load_jsonl(test_path, test_size):
        try:
            question, answer = extract_question_answer(row)
        except Exception:
            continue
        bank[_question_hash(question)] = {"question": question, "answer": answer}
    return bank


def _load_prompts(run_dir: Path, run_meta: Dict[str, Any]) -> List[str]:
    probe = _read_json(run_dir / "probe_prompts.json")
    candidates: List[Dict[str, Any]] = []
    if isinstance(probe, dict) and isinstance(probe.get("agents", []), list):
        candidates = [x for x in probe.get("agents", []) if isinstance(x, dict)]
    if not candidates:
        probe_meta = run_meta.get("probe", {}) if isinstance(run_meta.get("probe", {}), dict) else {}
        if isinstance(probe_meta.get("agents", []), list):
            candidates = [x for x in probe_meta.get("agents", []) if isinstance(x, dict)]
    if not candidates:
        last_state = _read_json(run_dir / "last_state.json")
        if isinstance(last_state, dict) and isinstance(last_state.get("agents", []), list):
            for item in last_state.get("agents", []):
                if isinstance(item, dict):
                    candidates.append(
                        {
                            "prompt": item.get("current_prompt")
                            or item.get("initial_prompt")
                            or "",
                        }
                    )
    prompts = [str(x.get("prompt", "")).strip() for x in candidates]
    return [p for p in prompts if p]


def _patch_meta_out_dir(payload: Any, out_dir: Path) -> Any:
    if not isinstance(payload, dict):
        return payload
    cfg = payload.get("config", {})
    if isinstance(cfg, dict):
        cfg = dict(cfg)
        cfg["out_dir"] = str(out_dir)
        payload["config"] = cfg
    probe = payload.get("probe", {})
    if isinstance(probe, dict):
        probe = dict(probe)
        p_cfg = probe.get("config", {})
        if isinstance(p_cfg, dict):
            p_cfg = dict(p_cfg)
            p_cfg["out_dir"] = str(out_dir)
            probe["config"] = p_cfg
        payload["probe"] = probe
    return payload


def _build_agent_record(
    agent_id: int,
    traces: List[str],
    family_metrics: Dict[str, Any],
    judgments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    primaries = list(family_metrics.get("primary_families", []))
    secondaries = list(family_metrics.get("secondary_families", primaries))
    dists = list(family_metrics.get("agent_family_distributions", []))
    summaries = list(family_metrics.get("reasoning_summaries", []))
    steps = list(family_metrics.get("strategy_steps", []))
    features = list(family_metrics.get("distinctive_features", []))
    evidences = list(family_metrics.get("evidence_spans", []))
    confidences = list(family_metrics.get("family_confidences", []))
    judgment = judgments[agent_id] if agent_id < len(judgments) and isinstance(judgments[agent_id], dict) else {}
    trace = traces[agent_id] if agent_id < len(traces) else ""
    return {
        "agent_id": agent_id,
        "primary_family": primaries[agent_id] if agent_id < len(primaries) else str(judgment.get("primary_family", "")),
        "secondary_family": secondaries[agent_id] if agent_id < len(secondaries) else str(judgment.get("secondary_family", "")),
        "family_distribution": dists[agent_id] if agent_id < len(dists) else {},
        "reasoning_summary": summaries[agent_id] if agent_id < len(summaries) else str(judgment.get("reasoning_summary", "")),
        "strategy_steps": steps[agent_id] if agent_id < len(steps) else [],
        "distinctive_features": features[agent_id] if agent_id < len(features) else [],
        "evidence_spans": evidences[agent_id] if agent_id < len(evidences) else [],
        "confidence": confidences[agent_id] if agent_id < len(confidences) else 0.0,
        "trace": trace,
    }


def _normalize_family_judgments(value: Any, size: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    labels: List[str] = [""] * size
    judgments: List[Dict[str, Any]] = [{} for _ in range(size)]
    if not isinstance(value, list):
        return labels, judgments
    for i, item in enumerate(value[:size]):
        if not isinstance(item, dict):
            continue
        labels[i] = str(item.get("primary_family", "") or "")
        judgments[i] = dict(item)
    return labels, judgments


def _normalize_label_list(value: Any, size: int) -> List[str]:
    labels = [""] * size
    if not isinstance(value, list):
        return labels
    for i, item in enumerate(value[:size]):
        labels[i] = str(item or "")
    return labels


async def _repair_question(
    system: TextualGradientRLSystem,
    question: str,
    prompts: List[str],
    traces: List[str],
    answers: List[str],
    bad_agent_ids: List[int],
    max_attempts: int,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[int], List[Dict[str, Any]]]:
    current_traces = list(traces)
    current_answers = list(answers)
    current_bad = list(bad_agent_ids)
    attempt_logs: List[Dict[str, Any]] = []

    for attempt in range(1, max(1, max_attempts) + 1):
        if not current_bad:
            break
        tasks = [system.solve_once(question, agent_id, prompts[agent_id]) for agent_id in current_bad]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        next_bad: List[int] = []
        for agent_id, result in zip(current_bad, results):
            if isinstance(result, Exception):
                attempt_logs.append(
                    {
                        "agent_id": agent_id,
                        "attempt": attempt,
                        "status": "api_error",
                        "error": str(result),
                    }
                )
                next_bad.append(agent_id)
                continue
            trace_text, answer_text = result
            current_traces[agent_id] = trace_text
            current_answers[agent_id] = answer_text
            flags = _invalid_trace_flags(trace_text, answer_text)
            attempt_logs.append(
                {
                    "agent_id": agent_id,
                    "attempt": attempt,
                    "status": "repaired" if not flags["bad"] else "still_bad",
                    "flags": flags,
                }
            )
            if flags["bad"]:
                next_bad.append(agent_id)
        current_bad = next_bad

    return current_traces, current_answers, attempt_logs, current_bad, bad_agent_ids


async def _rebuild_one_run(
    run_dir: Path,
    out_root: Path,
    args: argparse.Namespace,
    report_rows: List[Dict[str, Any]],
    detail_rows: List[Dict[str, Any]],
):
    run_meta = _read_json(run_dir / "run_meta.json")
    if not isinstance(run_meta, dict):
        return
    cfg_data = run_meta.get("config", {})
    if not isinstance(cfg_data, dict):
        return
    cfg_data = dict(cfg_data)
    test_path = str(cfg_data.get("test_path", ""))
    test_size = int(cfg_data.get("test_size", 0) or 0)
    question_bank = _load_question_bank(test_path, test_size)
    prompts = _load_prompts(run_dir, run_meta)
    pred_path = _latest_jsonl([p for p in run_dir.glob("test_epoch*_predictions.jsonl")])
    trace_path = _latest_jsonl([run_dir / "test_trace_history.jsonl"])
    summary_path = _latest_jsonl([run_dir / "reasoning_summary_history.jsonl"])
    if not pred_path or not trace_path:
        return

    pred_records = _read_jsonl(pred_path)
    trace_records = _read_jsonl(trace_path)
    summary_records = _read_jsonl(summary_path) if summary_path else []
    trace_by_hash = {str(r.get("question_hash", "")): r for r in trace_records if str(r.get("question_hash", ""))}
    summary_by_hash = {str(r.get("question_hash", "")): r for r in summary_records if str(r.get("question_hash", ""))}
    pred_by_hash = {str(r.get("question_hash", "")): r for r in pred_records if str(r.get("question_hash", ""))}

    cfg: Optional[Config] = None
    runtime_dir: Optional[Path] = None
    system: Optional[TextualGradientRLSystem] = None
    if not bool(args.scan_only):
        cfg = _config_from_run_meta(cfg_data, out_root / "_runtime" / run_dir.name, args)
        runtime_dir = Path(cfg.out_dir)
        ensure_dir(str(runtime_dir))
        system = TextualGradientRLSystem(cfg)

    cleaned_preds: List[Dict[str, Any]] = []
    cleaned_traces: List[Dict[str, Any]] = []
    cleaned_summaries: List[Dict[str, Any]] = []
    repair_log: List[Dict[str, Any]] = []
    question_results: List[Dict[str, Any]] = []

    question_parallelism = max(1, int(getattr(args, "question_parallelism", 4)))
    question_sem = asyncio.Semaphore(question_parallelism)

    async def _process_prediction(idx: int, pred: Dict[str, Any]) -> Dict[str, Any]:
        qh = str(pred.get("question_hash", ""))
        trace_rec = trace_by_hash.get(qh, {})
        summary_rec = summary_by_hash.get(qh, {})
        qbank = question_bank.get(qh, {})
        question = str(qbank.get("question", "")).strip() or str(pred.get("question_excerpt", "")).strip()
        gold_raw = str(qbank.get("answer", pred.get("gold", "")))
        task_type = str(cfg_data.get("task_type", args.task_type))
        if task_type == "auto":
            task_type = args.task_type if args.task_type != "auto" else "mmlu"
        gold = parse_gold(gold_raw, task_type, question=question) if question else str(pred.get("gold", ""))
        answers = list(pred.get("answers", [])) if isinstance(pred.get("answers", []), list) else []
        trace_agents = list(trace_rec.get("agents", [])) if isinstance(trace_rec.get("agents", []), list) else []
        traces = [str(agent.get("trace", "")) for agent in trace_agents]
        bad_agent_ids: List[int] = []
        if question and answers and trace_agents and len(prompts) >= len(answers):
            flags = [
                _invalid_trace_flags(traces[i] if i < len(traces) else "", answers[i] if i < len(answers) else "")
                for i in range(min(len(traces), len(answers)))
            ]
            bad_agent_ids = [i for i, flag in enumerate(flags) if flag["bad"]]

        original_labels, original_judgments = _normalize_family_judgments(pred.get("family_judgments", []), len(traces))
        if not any(original_labels):
            original_labels = _normalize_label_list(
                pred.get("primary_family_labels", pred.get("primary_families", [])),
                len(traces),
            )
        if not any(original_judgments):
            original_judgments = [dict() for _ in range(len(traces))]

        result: Dict[str, Any] = {
            "idx": idx,
            "question_hash": qh,
            "bad_agent_count": len(bad_agent_ids),
            "bad_agent_ids_before_repair": bad_agent_ids,
            "unresolved_agent_ids": [],
            "question_status": "clean" if not bad_agent_ids else "degraded",
            "repaired": False,
            "repair_attempts": int(args.repair_attempts),
            "repair_log": {
                "run_name": run_dir.name,
                "question_hash": qh,
                "bad_agent_ids_before_repair": bad_agent_ids,
                "bad_agent_count": len(bad_agent_ids),
                "repair_attempts": int(args.repair_attempts),
                "question_status": "clean" if not bad_agent_ids else "degraded",
            },
            "cleaned_pred": dict(pred),
            "cleaned_trace": dict(trace_rec) if isinstance(trace_rec, dict) and trace_rec else None,
            "cleaned_summary": dict(summary_rec) if isinstance(summary_rec, dict) and summary_rec else None,
        }

        if not question or not answers or not trace_agents:
            result["question_status"] = "copied_missing_data"
            result["repair_log"]["question_status"] = "copied_missing_data"
            return result

        if len(prompts) < len(answers):
            result["question_status"] = "skipped_prompt_mismatch"
            result["repair_log"]["question_status"] = "skipped_prompt_mismatch"
            result["repair_log"]["answers"] = len(answers)
            result["repair_log"]["prompts"] = len(prompts)
            return result

        if bool(args.scan_only):
            return result

        if not bad_agent_ids:
            return result

        assert system is not None
        current_traces = list(traces)
        current_answers = list(answers)
        current_traces, current_answers, attempt_logs, unresolved, _ = await _repair_question(
            system,
            question,
            prompts,
            current_traces,
            current_answers,
            bad_agent_ids,
            int(args.repair_attempts),
        )
        repaired_labels, repaired_judgments, _, _ = await system._judge_strategy_families(
            [current_traces[i] for i in bad_agent_ids],
            [current_answers[i] for i in bad_agent_ids],
            question,
        )
        for local_idx, agent_id in enumerate(bad_agent_ids):
            if agent_id < len(original_labels):
                original_labels[agent_id] = repaired_labels[local_idx]
            if agent_id < len(original_judgments):
                original_judgments[agent_id] = repaired_judgments[local_idx]

        family_labels = list(original_labels)
        judgments = list(original_judgments)
        reward_pack = system.compute_rewards(
            current_traces,
            current_answers,
            gold,
            primary_family_labels=family_labels,
            family_judgments=judgments,
            family_group_judgment={
                "llm_direct_diversity_score": pred.get("llm_direct_diversity_score"),
                "llm_direct_diversity_reason": pred.get("llm_direct_diversity_reason", ""),
            },
        )
        family_metrics = reward_pack.get("family_metrics", {}) if isinstance(reward_pack.get("family_metrics", {}), dict) else {}

        cleaned_pred = dict(pred)
        cleaned_pred.update(
            {
                "answers": current_answers,
                "vote_answer": reward_pack.get("vote_answer", pred.get("vote_answer", "")),
                "vote_correct": reward_pack.get("vote_correct", pred.get("vote_correct", 0)),
                "llm_direct_diversity_score": reward_pack.get("llm_direct_diversity_score"),
                "llm_direct_diversity_reason": reward_pack.get("llm_direct_diversity_reason", ""),
                "primary_family_labels": family_metrics.get("primary_families", family_labels),
                "secondary_family_labels": family_metrics.get("secondary_families", family_labels),
                "reasoning_summaries": family_metrics.get("reasoning_summaries", []),
                "strategy_steps": family_metrics.get("strategy_steps", []),
                "distinctive_features": family_metrics.get("distinctive_features", []),
                "evidence_spans": family_metrics.get("evidence_spans", []),
                "family_confidences": family_metrics.get("family_confidences", []),
                "agent_family_distributions": family_metrics.get("agent_family_distributions", []),
                "family_judgments": judgments,
                "primary_family_counts": family_metrics.get("primary_family_counts", {}),
                "weighted_family_distribution": family_metrics.get("weighted_family_distribution", {}),
                "major_family_distribution": family_metrics.get("major_family_distribution", {}),
                "team_family_homogeneity_rate": family_metrics.get("team_family_homogeneity_rate", 0.0),
                "team_family_diversity": family_metrics.get("team_family_diversity", 0.0),
                "team_major_family_diversity": family_metrics.get("team_major_family_diversity", 0.0),
                "team_intra_family_diversity": family_metrics.get("team_intra_family_diversity", 0.0),
                "family_judge_metric": family_metrics.get("family_judge_metric", "unknown"),
                "all_same_primary": bool(family_metrics.get("all_same_primary", False)),
                "all_same_pair": bool(family_metrics.get("all_same_pair", False)),
                "primary_dominant_share": family_metrics.get("primary_dominant_share", 0.0),
                "pair_dominant_share": family_metrics.get("pair_dominant_share", 0.0),
                "mean_family_confidence": family_metrics.get("mean_family_confidence", 0.0),
                "low_confidence_share": family_metrics.get("low_confidence_share", 0.0),
                "rejudge_count": family_metrics.get("rejudge_count", 0),
                "mean_summary_words": family_metrics.get("mean_summary_words", 0.0),
                "mean_summary_tokens": family_metrics.get("mean_summary_tokens", 0.0),
                "mean_evidence_spans": family_metrics.get("mean_evidence_spans", 0.0),
                "gold": gold,
            }
        )
        cleaning_meta = {
            "bad_agent_ids_before_repair": bad_agent_ids,
            "repair_attempts": int(args.repair_attempts),
            "attempt_logs": attempt_logs,
            "unresolved_agent_ids": unresolved,
        }
        cleaned_pred["trace_cleaning"] = cleaning_meta
        cleaned_trace = dict(trace_rec)
        cleaned_trace.update(
            {
                "agents": [_build_agent_record(i, current_traces, family_metrics, judgments) for i in range(len(current_traces))],
                "family_judge_metric": family_metrics.get("family_judge_metric", "unknown"),
                "team_family_homogeneity_rate": family_metrics.get("team_family_homogeneity_rate", 0.0),
                "team_family_diversity": family_metrics.get("team_family_diversity", 0.0),
                "team_major_family_diversity": family_metrics.get("team_major_family_diversity", 0.0),
                "team_intra_family_diversity": family_metrics.get("team_intra_family_diversity", 0.0),
                "trace_cleaning": cleaning_meta,
            }
        )
        cleaned_summary = system._build_reasoning_summary_history_record("test_epoch1", 0, int(trace_rec.get("step", idx)), question, family_metrics, current_traces)
        cleaned_summary["trace_cleaning"] = cleaning_meta

        result.update(
            {
                "repaired": True,
                "current_traces": current_traces,
                "current_answers": current_answers,
                "unresolved_agent_ids": unresolved,
                "attempt_logs": attempt_logs,
                "cleaned_pred": cleaned_pred,
                "cleaned_trace": cleaned_trace,
                "cleaned_summary": cleaned_summary,
                "repair_log": {
                    "run_name": run_dir.name,
                    "question_hash": qh,
                    "bad_agent_ids_before_repair": bad_agent_ids,
                    "unresolved_agent_ids": unresolved,
                    "repair_attempts": int(args.repair_attempts),
                    "question_status": "repaired",
                },
            }
        )
        return result

    async def _guarded_process(idx: int, pred: Dict[str, Any]) -> Dict[str, Any]:
        async with question_sem:
            return await _process_prediction(idx, pred)

    question_results = await asyncio.gather(*[asyncio.create_task(_guarded_process(i, pred)) for i, pred in enumerate(pred_records, start=1)])
    question_results.sort(key=lambda r: int(r.get("idx", 0)))

    total_bad_questions = sum(1 for r in question_results if int(r.get("bad_agent_count", 0)) > 0)
    total_bad_agents = sum(int(r.get("bad_agent_count", 0)) for r in question_results)
    repaired_questions = sum(1 for r in question_results if bool(r.get("repaired", False)))
    unresolved_agents = sum(len(r.get("unresolved_agent_ids", [])) for r in question_results)

    if bool(args.scan_only):
        report_rows.append(
            {
                "run_name": run_dir.name,
                "questions": len(pred_records),
                "degraded_questions": total_bad_questions,
                "degraded_agent_entries": total_bad_agents,
                "repaired_questions": 0,
                "unresolved_agent_entries": 0,
            }
        )
        for row in question_results:
            detail_rows.append(row.get("repair_log", {}))
        return

    if not question_results:
        return

    for row in question_results:
        cleaned_preds.append(row.get("cleaned_pred", {}))
        if row.get("cleaned_trace") is not None:
            cleaned_traces.append(row["cleaned_trace"])
        if row.get("cleaned_summary") is not None:
            cleaned_summaries.append(row["cleaned_summary"])

    # Overwrite cleaned run files.
    cleaned_dir = out_root / run_dir.name
    if cleaned_dir.exists():
        shutil.rmtree(cleaned_dir)
    shutil.copytree(run_dir, cleaned_dir)

    _write_jsonl(cleaned_preds, cleaned_dir / pred_path.name)
    _write_jsonl(cleaned_traces, cleaned_dir / trace_path.name)
    if summary_path:
        _write_jsonl(cleaned_summaries, cleaned_dir / summary_path.name)

    history = _read_json(cleaned_dir / "history.json")
    if not isinstance(history, list) or not history:
        history = []
    if history and isinstance(history[-1], dict):
        latest = dict(history[-1])
    else:
        latest = {"epoch": 1, "train": {}, "test": {}}
    train_hist = latest.get("train", {}) if isinstance(latest.get("train", {}), dict) else {}
    test_hist = {
        "mean_family_homogeneity_rate": _mean([float(r.get("team_family_homogeneity_rate", 0.0) or 0.0) for r in cleaned_preds]),
        "mean_family_diversity": _mean([float(r.get("team_family_diversity", 0.0) or 0.0) for r in cleaned_preds]),
        "mean_llm_direct_diversity_score": _mean(
            [
                float(r.get("llm_direct_diversity_score", 0.0) or 0.0)
                for r in cleaned_preds
                if r.get("llm_direct_diversity_score") is not None
            ]
        ),
        "vote_acc": _mean([float(r.get("vote_correct", 0.0) or 0.0) for r in cleaned_preds]),
        "size": len(cleaned_preds),
    }
    latest["train"] = train_hist
    latest["test"] = test_hist
    if history:
        history[-1] = latest
    else:
        history = [latest]
    _write_json(cleaned_dir / "history.json", history)

    # Keep state files aligned with the cleaned history.
    for state_name in ["last_state.json", "best_state.json"]:
        state_path = cleaned_dir / state_name
        state = _read_json(state_path)
        if not isinstance(state, dict):
            state = {}
        state["history"] = history
        state["extra"] = latest
        state = _patch_meta_out_dir(state, cleaned_dir)
        _write_json(state_path, state)

    run_meta = _read_json(cleaned_dir / "run_meta.json")
    if isinstance(run_meta, dict):
        run_meta = _patch_meta_out_dir(run_meta, cleaned_dir)
        _write_json(cleaned_dir / "run_meta.json", run_meta)
    probe_prompts = _read_json(cleaned_dir / "probe_prompts.json")
    if isinstance(probe_prompts, dict):
        probe_prompts = _patch_meta_out_dir(probe_prompts, cleaned_dir)
        _write_json(cleaned_dir / "probe_prompts.json", probe_prompts)

    report_rows.append(
        {
            "run_name": run_dir.name,
            "questions": len(pred_records),
            "degraded_questions": total_bad_questions,
            "degraded_agent_entries": total_bad_agents,
            "repaired_questions": repaired_questions,
            "unresolved_agent_entries": unresolved_agents,
        }
    )

    for row in repair_log:
        row["run_name"] = run_dir.name
        detail_rows.append(row)

    # Remove the runtime directory used by the judge/solver helper.
    if runtime_dir is not None and runtime_dir.exists():
        shutil.rmtree(runtime_dir, ignore_errors=True)


async def main_async():
    parser = argparse.ArgumentParser(description="Repair degraded prove_experiments traces by rerunning only the bad agents.")
    parser.add_argument("--runs_root", type=str, default="prove_experiments/runs")
    parser.add_argument("--out_root", type=str, default="prove_experiments/cleaned_runs")
    parser.add_argument("--target_runs", type=str, default="", help="Comma-separated subset of run directory names.")
    parser.add_argument("--task_type", type=str, default="mmlu", choices=["auto", "gsm8k", "mmlu"])
    parser.add_argument("--repair_attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--critic_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_expansion_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_taxonomy_path", type=str, default="auto")
    parser.add_argument("--family_expansion_enabled", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_dual_family_labels", type=int, default=1, choices=[0, 1])
    parser.add_argument("--family_rejudge_on_low_confidence", type=int, default=0, choices=[0, 1])
    parser.add_argument("--family_confidence_threshold", type=float, default=0.3)
    parser.add_argument("--critic_max_tokens", type=int, default=8000)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--critic_temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--llm_call_timeout", type=float, default=180.0)
    parser.add_argument("--solver_api_key_env", type=str, default="")
    parser.add_argument("--solver_base_url_env", type=str, default="")
    parser.add_argument("--critic_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--critic_base_url_env", type=str, default="OPENAI_BASE_URL")
    parser.add_argument("--scan_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--question_parallelism", type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)
    runs_root = Path(args.runs_root)
    out_root = Path(args.out_root)
    ensure_dir(str(out_root))

    wanted = {x.strip() for x in args.target_runs.split(",") if x.strip()}
    run_dirs = [p for p in sorted(runs_root.iterdir()) if p.is_dir() and (not wanted or p.name in wanted)]
    if not run_dirs:
        raise ValueError(f"No run directories found under {runs_root}")

    report_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        print(f"[repair] {run_dir.name}")
        try:
            await _rebuild_one_run(run_dir, out_root, args, report_rows, detail_rows)
        except Exception as exc:
            report_rows.append(
                {
                    "run_name": run_dir.name,
                    "questions": 0,
                    "degraded_questions": 0,
                    "degraded_agent_entries": 0,
                    "repaired_questions": 0,
                    "unresolved_agent_entries": 0,
                    "error": str(exc),
                }
            )
            print(f"[repair][WARN] {run_dir.name} failed: {exc}")

    report_csv = out_root / "degradation_report.csv"
    _write_csv(report_rows, report_csv)

    report_md = out_root / "degradation_report.md"
    md_lines = ["# Degradation Report", ""]
    for row in report_rows:
        if row.get("error"):
            md_lines.append(f"- {row['run_name']}: failed - {row['error']}")
            continue
        md_lines.append(
            f"- {row['run_name']}: degraded_questions={row.get('degraded_questions', 0)}/{row.get('questions', 0)}, "
            f"degraded_agent_entries={row.get('degraded_agent_entries', 0)}, repaired_questions={row.get('repaired_questions', 0)}, "
            f"unresolved_agent_entries={row.get('unresolved_agent_entries', 0)}"
        )
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    detail_path = out_root / "repair_log.jsonl"
    _write_jsonl(detail_rows, detail_path)

    print(f"Report written to {report_csv}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
