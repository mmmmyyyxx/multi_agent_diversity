import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from multi_dataset_diverse_rl.config import Config
    from scripts.experiment_config import (
        DEFAULT_DATASET_PATHS,
        DEFAULT_EXPERIMENT_SETTINGS,
        DEFAULT_SEED_BASELINES,
        ExperimentSetting,
        dataset_paths_from_args,
        parse_csv_list,
        select_settings,
    )
    from scripts.experiment_io import append_jsonl, read_json, read_jsonl, write_csv
except ModuleNotFoundError:
    from multi_dataset_diverse_rl.config import Config
    from experiment_config import (
        DEFAULT_DATASET_PATHS,
        DEFAULT_EXPERIMENT_SETTINGS,
        DEFAULT_SEED_BASELINES,
        ExperimentSetting,
        dataset_paths_from_args,
        parse_csv_list,
        select_settings,
    )
    from experiment_io import append_jsonl, read_json, read_jsonl, write_csv


SETTINGS = DEFAULT_EXPERIMENT_SETTINGS


def _load_history_metrics(history_path: Path) -> Dict[str, Any]:
    empty = {
        "epochs_completed": 0,
        "latest_train_embedding_diversity": None,
        "latest_train_embedding_overlap": None,
        "latest_train_invalid_rate": None,
        "latest_train_vote_acc": None,
        "latest_test_embedding_diversity": None,
        "latest_test_embedding_overlap": None,
        "latest_test_invalid_rate": None,
        "latest_test_vote_acc": None,
        "latest_test_vote_tie_rate": None,
    }
    if not history_path.exists():
        return empty
    hist = read_json(history_path)
    if not isinstance(hist, list) or not hist:
        return empty
    last = hist[-1] if isinstance(hist[-1], dict) else {}
    train = last.get("train", {}) if isinstance(last.get("train", {}), dict) else {}
    test = last.get("test", {}) if isinstance(last.get("test", {}), dict) else {}
    return {
        "epochs_completed": len(hist),
        "latest_train_embedding_diversity": train.get("mean_embedding_diversity"),
        "latest_train_embedding_overlap": train.get("mean_embedding_overlap"),
        "latest_train_invalid_rate": train.get("mean_invalid_rate"),
        "latest_train_vote_acc": train.get("vote_acc"),
        "latest_test_embedding_diversity": test.get("mean_embedding_diversity"),
        "latest_test_embedding_overlap": test.get("mean_embedding_overlap"),
        "latest_test_invalid_rate": test.get("mean_invalid_rate"),
        "latest_test_vote_acc": test.get("vote_acc"),
        "latest_test_vote_tie_rate": test.get("vote_tie_rate"),
    }


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _collect_run_log_metrics(run_dir: Path) -> Dict[str, Any]:
    update_rows = read_jsonl(run_dir / "update_logs.jsonl")
    candidate_rows = [r for r in update_rows if isinstance(r, dict) and "reward" in r and r.get("event") != "beam_refresh"]

    def vals(key: str) -> List[float]:
        out = []
        for row in candidate_rows:
            try:
                out.append(float(row.get(key, 0.0) or 0.0))
            except Exception:
                pass
        return out

    return {
        "reward": _safe_mean(vals("reward")),
        "candidate_embedding_diversity": _safe_mean(vals("embedding_diversity")),
        "candidate_invalid_rate": _safe_mean(vals("invalid_rate")),
        "solver_calls": _safe_mean(vals("solver_calls")),
        "solver_reuse_hit_rate": _safe_mean(vals("solver_reuse_hit_rate")),
    }


def _append_common_cli_args(cmd: List[str], args: argparse.Namespace, setting: ExperimentSetting, dataset_info: Dict[str, str], seed: int):
    reward_mode = args.force_reward_mode or setting.reward_mode
    cmd.extend(
        [
            "--task_type", dataset_info["task_type"],
            "--dataset_format", args.dataset_format,
            "--agent_model", args.agent_model,
            "--optimizer_model", args.optimizer_model,
            "--evaluator_model", args.evaluator_model,
            "--search_mode", args.search_mode,
            "--reward_mode", reward_mode,
            "--beam_size", str(args.beam_size),
            "--num_candidates_per_parent", str(args.num_candidates_per_parent),
            "--beam_refresh_each_epoch", str(int(args.beam_refresh_each_epoch)),
            "--homogeneity_overlap_threshold", str(args.homogeneity_overlap_threshold),
            "--homogeneity_pressure_tie_eps", str(args.homogeneity_pressure_tie_eps),
            "--max_homogeneous_cases_per_agent", str(args.max_homogeneous_cases_per_agent),
            "--random_window_cases_per_agent", str(args.random_window_cases_per_agent),
            "--hard_validity_cases_per_agent", str(args.hard_validity_cases_per_agent),
            "--invalid_repair_rate_threshold", str(args.invalid_repair_rate_threshold),
            "--accuracy_guard_epsilon", str(args.accuracy_guard_epsilon),
            "--reward_weight_div_delta", str(args.reward_weight_div_delta),
            "--reward_weight_invalid_delta", str(args.reward_weight_invalid_delta),
            "--reward_weight_coverage", str(args.reward_weight_coverage),
            "--reward_weight_useful_diversity", str(args.reward_weight_useful_diversity),
            "--invalid_guard_epsilon", str(args.invalid_guard_epsilon),
            "--use_baseline_relative_reward", str(int(args.use_baseline_relative_reward)),
            "--reward_schedule_mode", args.reward_schedule_mode,
            "--reward_diversity_warmup_updates", str(args.reward_diversity_warmup_updates),
            "--reward_weight_div_delta_early", str(args.reward_weight_div_delta_early),
            "--reward_weight_div_delta_late", str(args.reward_weight_div_delta_late),
            "--reward_weight_coverage_early", str(args.reward_weight_coverage_early),
            "--reward_weight_coverage_late", str(args.reward_weight_coverage_late),
            "--reward_weight_useful_diversity_early", str(args.reward_weight_useful_diversity_early),
            "--reward_weight_useful_diversity_late", str(args.reward_weight_useful_diversity_late),
            "--reward_weight_target_accuracy_early", str(args.reward_weight_target_accuracy_early),
            "--reward_weight_target_accuracy_late", str(args.reward_weight_target_accuracy_late),
            "--accuracy_guard_epsilon_early", str(args.accuracy_guard_epsilon_early),
            "--accuracy_guard_epsilon_late", str(args.accuracy_guard_epsilon_late),
            "--optimizer_architecture", args.optimizer_architecture,
            "--teacher_critic_max_rounds", str(args.teacher_critic_max_rounds),
            "--teacher_question_pass_threshold", str(args.teacher_question_pass_threshold),
            "--teacher_temperature", str(args.teacher_temperature),
            "--critic_temperature", str(args.critic_temperature),
            "--student_temperature", str(args.student_temperature),
            "--teacher_max_tokens", str(args.teacher_max_tokens),
            "--critic_max_tokens", str(args.critic_max_tokens),
            "--student_max_tokens", str(args.student_max_tokens),
            "--student_json_retry_on_parse_fail", str(int(args.student_json_retry_on_parse_fail)),
            "--student_json_max_retries", str(args.student_json_max_retries),
            "--student_json_repair_enabled", str(int(args.student_json_repair_enabled)),
            "--student_json_repair_max_tokens", str(args.student_json_repair_max_tokens),
            "--student_json_repair_temperature", str(args.student_json_repair_temperature),
            "--student_candidate_schema_mode", args.student_candidate_schema_mode,
            "--student_candidate_max_chars_per_field", str(args.student_candidate_max_chars_per_field),
            "--student_candidate_prompt_max_chars", str(args.student_candidate_prompt_max_chars),
            "--student_force_minified_json", str(int(args.student_force_minified_json)),
            "--teacher_critic_use_voting_failure", str(int(args.teacher_critic_use_voting_failure)),
            "--optimizer_fallback_mode", args.optimizer_fallback_mode,
            "--no_effective_evolution_patience", str(args.no_effective_evolution_patience),
            "--no_effective_evolution_min_optimizer_candidates", str(args.no_effective_evolution_min_optimizer_candidates),
            "--no_effective_evolution_stop_enabled", str(int(args.no_effective_evolution_stop_enabled)),
            "--diversity_metric", args.diversity_metric,
            "--use_joint_trace_diversity_evaluator", str(int(args.use_joint_trace_diversity_evaluator)),
            "--invalid_binary", str(int(args.invalid_binary)),
            "--embedding_model", args.embedding_model,
            "--trace_embedding_chunk_words", str(args.trace_embedding_chunk_words),
            "--trace_embedding_chunk_overlap", str(args.trace_embedding_chunk_overlap),
            "--max_retries", str(args.max_retries),
            "--retry_sleep", str(args.retry_sleep),
            "--transient_retry_forever", str(int(args.transient_retry_forever)),
            "--max_transient_retries", str(args.max_transient_retries),
            "--max_retry_backoff", str(args.max_retry_backoff),
            "--llm_call_logging", str(int(args.llm_call_logging)),
            "--llm_call_timeout", str(args.llm_call_timeout),
            "--candidate_eval_concurrency", str(args.candidate_eval_concurrency),
            "--candidate_eval_strategy", args.candidate_eval_strategy,
            "--candidate_eval_pool_size", str(args.candidate_eval_pool_size),
            "--candidate_eval_repeats", str(args.candidate_eval_repeats),
            "--candidate_eval_seed_offset", str(args.candidate_eval_seed_offset),
            "--train_rollout_concurrency", str(args.train_rollout_concurrency),
            "--eval_solver_call_concurrency", str(args.eval_solver_call_concurrency),
            "--vote_tie_break", args.vote_tie_break,
            "--aggregation_mode", args.aggregation_mode,
            "--agents", str(args.agents),
            "--test_size", str(args.test_size),
            "--eval_test_each_epoch", str(int(args.eval_test_each_epoch)),
            "--early_stopping_patience", str(args.early_stopping_patience),
            "--early_stopping_min_delta", str(args.early_stopping_min_delta),
            "--init_mode", setting.init_mode,
            "--shared_prompt", args.shared_prompt,
            "--max_tokens", str(args.max_tokens),
            "--optimizer_max_tokens", str(args.optimizer_max_tokens),
            "--evaluator_max_tokens", str(args.evaluator_max_tokens),
            "--seed", str(seed),
        ]
    )


def run_one(dataset: str, setting: ExperimentSetting, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    dataset_info = dataset_paths_from_args(args, dataset)
    run_name = f"{setting.name}_seed{seed}" if args.multi_seed_names else setting.name
    run_dir = Path(args.out_root) / dataset / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [args.python, "-m", "multi_dataset_diverse_rl.cli"]
    _append_common_cli_args(cmd, args, setting, dataset_info, seed)
    cmd.extend(
        [
            "--test_path", dataset_info["test"],
            "--out_dir", str(run_dir),
            "--baseline_only", "1" if setting.baseline_only else "0",
        ]
    )
    if not setting.baseline_only:
        cmd.extend(
            [
                "--train_path", dataset_info["train"],
                "--val_path", dataset_info["val"],
                "--train_size", str(args.train_size),
                "--val_size", str(args.val_size),
                "--val_split_ratio", str(args.val_split_ratio),
                "--epochs", str(args.epochs),
                "--update_every", str(args.update_every),
                "--candidate_eval_batch_size", str(args.candidate_eval_batch_size),
            ]
        )

    start = time.time()
    print(f"\n[RUN] dataset={dataset} setting={setting.name} seed={seed}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=args.workspace)
    elapsed = time.time() - start
    reward_mode = args.force_reward_mode or setting.reward_mode
    row = {
        "dataset": dataset,
        "setting": setting.name,
        "run_name": run_name,
        "seed": seed,
        "reward_mode": reward_mode,
        "init_mode": setting.init_mode,
        "baseline_only": int(setting.baseline_only),
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 2),
        "run_dir": str(run_dir),
        "agent_model": args.agent_model,
        "optimizer_model": args.optimizer_model,
        "evaluator_model": args.evaluator_model,
        "search_mode": args.search_mode,
        "beam_size": args.beam_size,
    }
    history_metrics = _load_history_metrics(run_dir / "history.json")
    row.update(history_metrics)
    log_metrics = _collect_run_log_metrics(run_dir)
    row.update(log_metrics)
    row["vote_acc"] = history_metrics.get("latest_test_vote_acc")
    row["embedding_diversity"] = history_metrics.get("latest_test_embedding_diversity")
    row["invalid_rate"] = history_metrics.get("latest_test_invalid_rate")
    row["vote_tie_rate"] = history_metrics.get("latest_test_vote_tie_rate")
    return row


def _selected_settings(raw: str) -> List[ExperimentSetting]:
    return select_settings(raw, SETTINGS)


def main():
    parser = argparse.ArgumentParser(description="Run shared/bank baselines and shared/bank guarded beam experiments.")
    cli_defaults = Config()
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--out_root", type=str, default="runs_trace_beam")
    parser.add_argument("--datasets", type=str, default="mmlu")
    parser.add_argument("--run_settings", type=str, default="all")
    parser.add_argument("--mars_result_path", type=str, default="")
    parser.add_argument("--summary_by_dataset", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dataset_format", type=str, default="legacy", choices=["legacy", "mars"])
    parser.add_argument("--task_type", type=str, default="mmlu", choices=["auto", "gsm8k", "mmlu", "bbh"])
    parser.add_argument("--train_path", type=str, default="mmlu_train.jsonl")
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="mmlu_test.jsonl")
    parser.add_argument("--mmlu_train_path", type=str, default=DEFAULT_DATASET_PATHS["mmlu"].train)
    parser.add_argument("--mmlu_val_path", type=str, default=DEFAULT_DATASET_PATHS["mmlu"].val)
    parser.add_argument("--mmlu_test_path", type=str, default=DEFAULT_DATASET_PATHS["mmlu"].test)
    parser.add_argument("--bbh_train_path", type=str, default=DEFAULT_DATASET_PATHS["bbh"].train)
    parser.add_argument("--bbh_val_path", type=str, default=DEFAULT_DATASET_PATHS["bbh"].val)
    parser.add_argument("--bbh_test_path", type=str, default=DEFAULT_DATASET_PATHS["bbh"].test)

    parser.add_argument("--agent_model", type=str, default="deepseek-chat")
    parser.add_argument("--optimizer_model", type=str, default="deepseek-chat")
    parser.add_argument("--evaluator_model", type=str, default="deepseek-chat")
    parser.add_argument("--search_mode", type=str, default="evolutionary_beam", choices=["evolutionary_beam"])
    parser.add_argument("--force_reward_mode", type=str, default="", choices=["", "accuracy_only", "guarded_diversity", "coverage_useful_diversity"])
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--num_candidates_per_parent", type=int, default=2)
    parser.add_argument("--beam_refresh_each_epoch", type=int, default=1, choices=[0, 1])
    parser.add_argument("--homogeneity_overlap_threshold", type=float, default=0.55)
    parser.add_argument("--homogeneity_pressure_tie_eps", type=float, default=0.03)
    parser.add_argument("--max_homogeneous_cases_per_agent", type=int, default=4)
    parser.add_argument("--random_window_cases_per_agent", type=int, default=2)
    parser.add_argument("--hard_validity_cases_per_agent", type=int, default=2)
    parser.add_argument("--invalid_repair_rate_threshold", type=float, default=0.25)
    parser.add_argument("--accuracy_guard_epsilon", type=float, default=cli_defaults.accuracy_guard_epsilon)
    parser.add_argument("--reward_weight_div_delta", type=float, default=cli_defaults.reward_weight_div_delta)
    parser.add_argument("--reward_weight_invalid_delta", type=float, default=cli_defaults.reward_weight_invalid_delta)
    parser.add_argument("--reward_weight_coverage", type=float, default=cli_defaults.reward_weight_coverage)
    parser.add_argument("--reward_weight_useful_diversity", type=float, default=cli_defaults.reward_weight_useful_diversity)
    parser.add_argument("--invalid_guard_epsilon", type=float, default=cli_defaults.invalid_guard_epsilon)
    parser.add_argument("--use_baseline_relative_reward", type=int, default=int(cli_defaults.use_baseline_relative_reward), choices=[0, 1])
    parser.add_argument("--reward_schedule_mode", type=str, default=cli_defaults.reward_schedule_mode, choices=["static", "phase_adaptive"])
    parser.add_argument("--reward_diversity_warmup_updates", type=int, default=cli_defaults.reward_diversity_warmup_updates)
    parser.add_argument("--reward_weight_div_delta_early", type=float, default=cli_defaults.reward_weight_div_delta_early)
    parser.add_argument("--reward_weight_div_delta_late", type=float, default=cli_defaults.reward_weight_div_delta_late)
    parser.add_argument("--reward_weight_coverage_early", type=float, default=cli_defaults.reward_weight_coverage_early)
    parser.add_argument("--reward_weight_coverage_late", type=float, default=cli_defaults.reward_weight_coverage_late)
    parser.add_argument("--reward_weight_useful_diversity_early", type=float, default=cli_defaults.reward_weight_useful_diversity_early)
    parser.add_argument("--reward_weight_useful_diversity_late", type=float, default=cli_defaults.reward_weight_useful_diversity_late)
    parser.add_argument("--reward_weight_target_accuracy_early", type=float, default=cli_defaults.reward_weight_target_accuracy_early)
    parser.add_argument("--reward_weight_target_accuracy_late", type=float, default=cli_defaults.reward_weight_target_accuracy_late)
    parser.add_argument("--accuracy_guard_epsilon_early", type=float, default=cli_defaults.accuracy_guard_epsilon_early)
    parser.add_argument("--accuracy_guard_epsilon_late", type=float, default=cli_defaults.accuracy_guard_epsilon_late)
    parser.add_argument("--optimizer_architecture", type=str, default=cli_defaults.optimizer_architecture, choices=["one_shot", "teacher_critic_student"])
    parser.add_argument("--teacher_critic_max_rounds", type=int, default=cli_defaults.teacher_critic_max_rounds)
    parser.add_argument("--teacher_question_pass_threshold", type=float, default=cli_defaults.teacher_question_pass_threshold)
    parser.add_argument("--teacher_temperature", type=float, default=cli_defaults.teacher_temperature)
    parser.add_argument("--critic_temperature", type=float, default=cli_defaults.critic_temperature)
    parser.add_argument("--student_temperature", type=float, default=cli_defaults.student_temperature)
    parser.add_argument("--teacher_max_tokens", type=int, default=cli_defaults.teacher_max_tokens)
    parser.add_argument("--critic_max_tokens", type=int, default=cli_defaults.critic_max_tokens)
    parser.add_argument("--student_max_tokens", type=int, default=cli_defaults.student_max_tokens)
    parser.add_argument("--student_json_retry_on_parse_fail", type=int, default=int(cli_defaults.student_json_retry_on_parse_fail), choices=[0, 1])
    parser.add_argument("--student_json_max_retries", type=int, default=cli_defaults.student_json_max_retries)
    parser.add_argument("--student_json_repair_enabled", type=int, default=int(cli_defaults.student_json_repair_enabled), choices=[0, 1])
    parser.add_argument("--student_json_repair_max_tokens", type=int, default=cli_defaults.student_json_repair_max_tokens)
    parser.add_argument("--student_json_repair_temperature", type=float, default=cli_defaults.student_json_repair_temperature)
    parser.add_argument("--student_candidate_schema_mode", type=str, default=cli_defaults.student_candidate_schema_mode, choices=["compact", "verbose"])
    parser.add_argument("--student_candidate_max_chars_per_field", type=int, default=cli_defaults.student_candidate_max_chars_per_field)
    parser.add_argument("--student_candidate_prompt_max_chars", type=int, default=cli_defaults.student_candidate_prompt_max_chars)
    parser.add_argument("--student_force_minified_json", type=int, default=int(cli_defaults.student_force_minified_json), choices=[0, 1])
    parser.add_argument("--teacher_critic_use_voting_failure", type=int, default=int(cli_defaults.teacher_critic_use_voting_failure), choices=[0, 1])
    parser.add_argument("--optimizer_fallback_mode", type=str, default=cli_defaults.optimizer_fallback_mode, choices=["none", "template"])
    parser.add_argument("--no_effective_evolution_patience", type=int, default=cli_defaults.no_effective_evolution_patience)
    parser.add_argument("--no_effective_evolution_min_optimizer_candidates", type=int, default=cli_defaults.no_effective_evolution_min_optimizer_candidates)
    parser.add_argument("--no_effective_evolution_stop_enabled", type=int, default=int(cli_defaults.no_effective_evolution_stop_enabled), choices=[0, 1])
    parser.add_argument("--diversity_metric", type=str, default="trace_embedding", choices=["trace_embedding"])
    parser.add_argument("--use_joint_trace_diversity_evaluator", type=int, default=0, choices=[0, 1])
    parser.add_argument("--invalid_binary", type=int, default=1, choices=[0, 1])
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--trace_embedding_chunk_words", type=int, default=320)
    parser.add_argument("--trace_embedding_chunk_overlap", type=int, default=40)

    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--val_size", type=int, default=150)
    parser.add_argument("--val_split_ratio", type=float, default=0.2)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval_test_each_epoch", type=int, default=0, choices=[0, 1])
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--update_every", type=int, default=10)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=20)
    parser.add_argument("--candidate_eval_strategy", type=str, default="fixed_pool", choices=["random", "fixed_pool", "stratified"])
    parser.add_argument("--candidate_eval_pool_size", type=int, default=100)
    parser.add_argument("--candidate_eval_repeats", type=int, default=1)
    parser.add_argument("--candidate_eval_seed_offset", type=int, default=1000)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--optimizer_max_tokens", type=int, default=1400)
    parser.add_argument("--evaluator_max_tokens", type=int, default=1200)
    parser.add_argument("--shared_prompt", type=str, default="You are a careful reasoning solver. Produce a compact, explicit reasoning trace, make your decision procedure visible, verify key logic, and give exactly one final answer.")

    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--transient_retry_forever", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_transient_retries", type=int, default=0)
    parser.add_argument("--max_retry_backoff", type=float, default=30.0)
    parser.add_argument("--llm_call_logging", type=int, default=1, choices=[0, 1])
    parser.add_argument("--llm_call_timeout", type=float, default=120.0)
    parser.add_argument("--candidate_eval_concurrency", type=int, default=0)
    parser.add_argument("--train_rollout_concurrency", type=int, default=0)
    parser.add_argument("--eval_solver_call_concurrency", type=int, default=225)
    parser.add_argument("--vote_tie_break", type=str, default="random", choices=["first", "random", "abstain"])
    parser.add_argument("--aggregation_mode", type=str, default="majority", choices=["majority", "weighted_vote", "verifier_select"])
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--seed_baselines", type=int, default=DEFAULT_SEED_BASELINES, choices=[0, 1])
    parser.add_argument("--multi_seed_names", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    for name in [
        "beam_refresh_each_epoch",
        "use_joint_trace_diversity_evaluator",
        "invalid_binary",
        "eval_test_each_epoch",
        "transient_retry_forever",
        "llm_call_logging",
        "use_baseline_relative_reward",
        "teacher_critic_use_voting_failure",
        "no_effective_evolution_stop_enabled",
        "summary_by_dataset",
        "seed_baselines",
        "multi_seed_names",
    ]:
        setattr(args, name, bool(int(getattr(args, name))))

    args.workspace = str(Path(args.workspace).resolve())
    args.out_root = str((Path(args.workspace) / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root).resolve())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    runs_jsonl = out_root / "experiment_runs.jsonl"
    runs_csv = out_root / "experiment_runs.csv"
    runs_jsonl.write_text("", encoding="utf-8")

    seeds = [int(x) for x in parse_csv_list(args.seeds)] or [42]
    datasets = [x.lower() for x in parse_csv_list(args.datasets)]
    settings = _selected_settings(args.run_settings)

    rows = []
    for dataset in datasets:
        for setting in settings:
            setting_seeds = seeds if (not setting.baseline_only or args.seed_baselines) else [seeds[0]]
            for seed in setting_seeds:
                row = run_one(dataset, setting, seed, args)
                rows.append(row)
                append_jsonl(runs_jsonl, row)
                write_csv(runs_csv, rows, empty_fieldnames=["dataset", "setting", "status"])
                if row["status"] != "ok":
                    raise SystemExit(row["returncode"])
    print(f"\n[DONE] Wrote {runs_jsonl} and {runs_csv}")
    if args.summary_by_dataset:
        summary_cmd = [
            args.python,
            str(Path(args.workspace) / "scripts" / "compute_experiment_metrics.py"),
            "--runs_root",
            str(out_root),
            "--out_csv",
            str(out_root / "experiment_metrics.csv"),
            "--out_md",
            str(out_root / "experiment_metrics.md"),
            "--out_group_csv",
            str(out_root / "experiment_metrics_grouped.csv"),
        ]
        if args.mars_result_path:
            summary_cmd.extend(["--mars_result_path", args.mars_result_path])
        print(f"[SUMMARY] {' '.join(summary_cmd)}", flush=True)
        subprocess.run(summary_cmd, cwd=args.workspace, check=False)


if __name__ == "__main__":
    main()
