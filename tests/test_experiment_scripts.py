import json
from argparse import Namespace

from multi_dataset_diverse_rl.config import Config
from scripts.analyze_student_failures import summarize_run
from scripts.compute_experiment_metrics import analyze_run
from scripts.experiment_config import DEFAULT_EXPERIMENT_SETTING_NAMES, DEFAULT_SEED_BASELINES, DEFAULT_EXPERIMENT_SETTINGS, ExperimentSetting, dataset_paths_from_args, select_settings, setting_names
from scripts.run_experiments import SETTINGS, _selected_settings
from scripts.run_task_level_accuracy import _append_common_cli_args, _completed_run_row, _explicit_cli_or_setting, _skip_row, _setting_reward_mode, is_completed_run_dir
from multi_dataset_diverse_rl.task_manifest import ComparisonTask


def test_run_experiments_default_settings_include_baselines_and_guarded_beams():
    names = [setting.name for setting in SETTINGS]
    assert names == [
        "shared_baseline",
        "bank_baseline",
        "shared_guarded_beam",
        "bank_guarded_beam",
        "shared_scalar_tcs_vote_first",
        "shared_vote_pareto_tcs",
        "shared_vote_pareto_tcs_static",
        "shared_vote_pareto_tcs_boundary_selector",
        "shared_vote_error_pareto_tcs",
        "shared_vote_error_pareto_tcs_residual_specialization",
        "shared_vote_error_pareto_tcs_residual_cycle_guard",
        "shared_legacy_coverage_useful_tcs_strict",
        "shared_vote_tcs_competence_schedule",
        "shared_vote_tcs_competence_depth2",
        "shared_vote_tcs_competence_depth2_progressive_residual",
        "shared_vote_tcs_competence_depth2_progressive_residual_hybrid",
        "shared_accuracy_only_tcs_vote_first",
        "shared_accuracy_rollout_embedding_tcs",
        "shared_vote_ready_rollout_diversity_tcs",
        "shared_v9_sequential_accuracy",
        "shared_v9_sequential_accuracy_state",
        "shared_v9_sequential_accuracy_state_vote",
        "shared_v9_sequential_accuracy_state_vote_diversity",
        "shared_guarded_diversity_tcs_vote_first",
        "shared_vote_no_margin_tcs_vote_first",
        "shared_vote_no_boundary_tcs_vote_first",
    ]
    assert {setting.name: setting.reward_mode for setting in SETTINGS}["shared_guarded_beam"] == "guarded_diversity"
    assert {setting.name: setting.reward_mode for setting in SETTINGS}["bank_guarded_beam"] == "guarded_diversity"
    vote_setting = {setting.name: setting for setting in SETTINGS}["shared_vote_pareto_tcs"]
    assert vote_setting.reward_mode == "vote_useful_diversity"
    assert vote_setting.candidate_selection_mode == "vote_pareto"
    assert vote_setting.best_state_selection_mode == "vote_first"
    assert SETTINGS == DEFAULT_EXPERIMENT_SETTINGS


def test_run_experiments_parser_seeds_baselines_by_default():
    assert DEFAULT_SEED_BASELINES == 1
    assert any(setting.baseline_only for setting in SETTINGS)


def test_selected_settings_filters_by_name():
    selected = _selected_settings("shared_baseline,bank_guarded_beam")
    assert [setting.name for setting in selected] == ["shared_baseline", "bank_guarded_beam"]
    assert select_settings("shared_baseline,bank_guarded_beam") == selected


def test_default_and_all_setting_sets_are_distinct():
    assert DEFAULT_EXPERIMENT_SETTING_NAMES == [
        "shared_baseline", "bank_baseline", "shared_guarded_beam", "bank_guarded_beam",
    ]
    assert setting_names(select_settings("all")) == [
        "shared_baseline", "bank_baseline", "shared_guarded_beam", "bank_guarded_beam",
        "shared_scalar_tcs_vote_first", "shared_vote_pareto_tcs",
        "shared_vote_pareto_tcs_static",
        "shared_vote_pareto_tcs_boundary_selector", "shared_vote_error_pareto_tcs",
        "shared_vote_error_pareto_tcs_residual_specialization", "shared_vote_error_pareto_tcs_residual_cycle_guard",
        "shared_legacy_coverage_useful_tcs_strict", "shared_vote_tcs_competence_schedule",
        "shared_vote_tcs_competence_depth2", "shared_vote_tcs_competence_depth2_progressive_residual",
        "shared_vote_tcs_competence_depth2_progressive_residual_hybrid",
        "shared_accuracy_only_tcs_vote_first", "shared_accuracy_rollout_embedding_tcs",
            "shared_vote_ready_rollout_diversity_tcs", "shared_v9_sequential_accuracy",
            "shared_v9_sequential_accuracy_state",
            "shared_v9_sequential_accuracy_state_vote",
            "shared_v9_sequential_accuracy_state_vote_diversity",
        "shared_guarded_diversity_tcs_vote_first",
        "shared_vote_no_margin_tcs_vote_first", "shared_vote_no_boundary_tcs_vote_first",
    ]


def test_dataset_paths_use_dataset_specific_defaults():
    args = Namespace(
        mmlu_train_path="mmlu_train.jsonl",
        mmlu_val_path="mmlu_val.jsonl",
        mmlu_test_path="mmlu_test.jsonl",
        bbh_train_path="bbh_train.jsonl",
        bbh_val_path="bbh_val.jsonl",
        bbh_test_path="bbh_test.jsonl",
        task_type="auto",
        train_path="train.jsonl",
        val_path="val.jsonl",
        test_path="test.jsonl",
    )
    assert dataset_paths_from_args(args, "mmlu") == {
        "task_type": "mmlu",
        "train": "mmlu_train.jsonl",
        "val": "mmlu_val.jsonl",
        "test": "mmlu_test.jsonl",
    }
    assert dataset_paths_from_args(args, "bbh") == {
        "task_type": "bbh",
        "train": "bbh_train.jsonl",
        "val": "bbh_val.jsonl",
        "test": "bbh_test.jsonl",
    }


def test_analyze_experiments_uses_shared_setting_order():
    import scripts.analyze_experiments as analyze_experiments

    assert analyze_experiments.SETTINGS == setting_names(DEFAULT_EXPERIMENT_SETTINGS)


def test_task_level_reward_mode_override_only_changes_non_baseline():
    args = Namespace(reward_mode="vote_useful_diversity")
    baseline = ExperimentSetting("shared_baseline", "shared", True, "guarded_diversity")
    beam = ExperimentSetting("shared_guarded_beam", "shared", False, "guarded_diversity")

    assert _setting_reward_mode(args, baseline) == "guarded_diversity"
    assert _setting_reward_mode(args, beam) == "vote_useful_diversity"


def test_task_level_high_accuracy_skip_row_records_precheck_reason():
    task = ComparisonTask(
        task_id="boolean_expressions",
        benchmark="BBH",
        task_type="bbh",
        answer_format="boolean",
        train_path="train.csv",
        val_path="val.csv",
        test_path="test.csv",
    )
    setting = ExperimentSetting("shared_guarded_beam", "shared", False, "guarded_diversity")
    args = Namespace(dataset_format="mars", reward_mode="vote_useful_diversity", precheck_steps=20, precheck_acc_threshold=0.95)
    row = _skip_row(task, setting, 42, args, {"precheck_vote_acc": 1.0})

    assert row["status"] == "skipped_high_baseline_acc"
    assert row["reward_mode"] == "vote_useful_diversity"
    assert row["precheck_vote_acc"] == 1.0
    assert row["precheck_steps"] == 20


def test_task_level_resume_completion_requires_history_cost_and_meta(tmp_path):
    run_dir = tmp_path / "task" / "shared_guarded_beam_seed42"
    run_dir.mkdir(parents=True)

    assert is_completed_run_dir(run_dir) is False

    (run_dir / "history.json").write_text(json.dumps([{"test": {"vote_acc": 0.5, "num_test_samples": 2}}]), encoding="utf-8")
    (run_dir / "run_meta.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    assert is_completed_run_dir(run_dir) is False

    (run_dir / "cost_summary.json").write_text(json.dumps({"total_llm_calls": 3}), encoding="utf-8")
    assert is_completed_run_dir(run_dir) is True


def test_task_level_completed_run_row_marks_resume_reuse(tmp_path):
    task = ComparisonTask(
        task_id="ruin_names",
        benchmark="BBH",
        task_type="bbh",
        answer_format="option_letter",
        train_path="train.csv",
        val_path="val.csv",
        test_path="test.csv",
    )
    setting = ExperimentSetting("shared_guarded_beam", "shared", False, "guarded_diversity")
    args = Namespace(out_root=str(tmp_path), dataset_format="mars", reward_mode="vote_useful_diversity")

    row = _completed_run_row(task, setting, 42, args)

    assert row["status"] == "reused_completed"
    assert row["returncode"] == 0
    assert row["resume_completed"] == 1
    assert row["reward_mode"] == "vote_useful_diversity"
    assert row["run_dir"].endswith("ruin_names\\shared_guarded_beam_seed42") or row["run_dir"].endswith("ruin_names/shared_guarded_beam_seed42")


def test_task_level_cli_args_pass_resume_checkpoint_flag():
    task = ComparisonTask(
        task_id="ruin_names",
        benchmark="BBH",
        task_type="bbh",
        answer_format="option_letter",
        train_path="train.csv",
        val_path="val.csv",
        test_path="test.csv",
    )
    setting = ExperimentSetting("shared_guarded_beam", "shared", False, "guarded_diversity")
    cfg = Config()
    args = Namespace(
        dataset_format="mars",
        reward_mode="vote_useful_diversity",
        agent_model=cfg.agent_model,
        optimizer_model=cfg.optimizer_model,
        evaluator_model=cfg.evaluator_model,
        agents=cfg.agents,
        shared_prompt=cfg.shared_prompt,
        beam_size=cfg.beam_size,
        num_candidates_per_parent=cfg.num_candidates_per_parent,
        optimizer_parent_concurrency=cfg.optimizer_parent_concurrency,
        beam_refresh_each_epoch=int(cfg.beam_refresh_each_epoch),
        accuracy_guard_epsilon=cfg.accuracy_guard_epsilon,
        reward_weight_div_delta=cfg.reward_weight_div_delta,
        reward_weight_invalid_delta=cfg.reward_weight_invalid_delta,
        reward_weight_vote_delta=cfg.reward_weight_vote_delta,
        reward_weight_vote_margin=cfg.reward_weight_vote_margin,
        reward_weight_boundary_diversity=cfg.reward_weight_boundary_diversity,
        invalid_guard_epsilon=cfg.invalid_guard_epsilon,
        use_baseline_relative_reward=int(cfg.use_baseline_relative_reward),
        reward_schedule_mode=cfg.reward_schedule_mode,
        reward_diversity_warmup_updates=cfg.reward_diversity_warmup_updates,
        reward_weight_div_delta_early=cfg.reward_weight_div_delta_early,
        reward_weight_div_delta_late=cfg.reward_weight_div_delta_late,
        reward_weight_vote_delta_early=cfg.reward_weight_vote_delta_early,
        reward_weight_vote_delta_late=cfg.reward_weight_vote_delta_late,
        reward_weight_vote_margin_early=cfg.reward_weight_vote_margin_early,
        reward_weight_vote_margin_late=cfg.reward_weight_vote_margin_late,
        reward_weight_boundary_diversity_early=cfg.reward_weight_boundary_diversity_early,
        reward_weight_boundary_diversity_late=cfg.reward_weight_boundary_diversity_late,
        reward_weight_target_accuracy_early=cfg.reward_weight_target_accuracy_early,
        reward_weight_target_accuracy_late=cfg.reward_weight_target_accuracy_late,
        accuracy_guard_epsilon_early=cfg.accuracy_guard_epsilon_early,
        accuracy_guard_epsilon_late=cfg.accuracy_guard_epsilon_late,
        optimizer_architecture=cfg.optimizer_architecture,
        teacher_critic_max_rounds=cfg.teacher_critic_max_rounds,
        teacher_question_pass_threshold=cfg.teacher_question_pass_threshold,
        teacher_temperature=cfg.teacher_temperature,
        critic_temperature=cfg.critic_temperature,
        student_temperature=cfg.student_temperature,
        teacher_max_tokens=cfg.teacher_max_tokens,
        critic_max_tokens=cfg.critic_max_tokens,
        student_max_tokens=cfg.student_max_tokens,
        student_json_retry_on_parse_fail=cfg.student_json_retry_on_parse_fail,
        student_json_max_retries=cfg.student_json_max_retries,
        student_json_repair_enabled=cfg.student_json_repair_enabled,
        student_json_repair_max_tokens=cfg.student_json_repair_max_tokens,
        student_json_repair_temperature=cfg.student_json_repair_temperature,
        student_candidate_schema_mode=cfg.student_candidate_schema_mode,
        student_candidate_max_chars_per_field=cfg.student_candidate_max_chars_per_field,
        student_candidate_prompt_max_chars=cfg.student_candidate_prompt_max_chars,
        student_force_minified_json=cfg.student_force_minified_json,
        teacher_critic_use_voting_failure=cfg.teacher_critic_use_voting_failure,
        optimizer_fallback_mode=cfg.optimizer_fallback_mode,
        no_effective_evolution_patience=cfg.no_effective_evolution_patience,
        no_effective_evolution_min_optimizer_candidates=cfg.no_effective_evolution_min_optimizer_candidates,
        no_effective_evolution_stop_enabled=cfg.no_effective_evolution_stop_enabled,
        candidate_eval_strategy=cfg.candidate_eval_strategy,
        candidate_eval_concurrency=cfg.candidate_eval_concurrency,
        candidate_eval_pool_size=cfg.candidate_eval_pool_size,
        candidate_eval_repeats=cfg.candidate_eval_repeats,
        candidate_eval_seed_offset=cfg.candidate_eval_seed_offset,
        candidate_reuse_recorded_rollouts=cfg.candidate_reuse_recorded_rollouts,
        resume_from_checkpoint=1,
        train_rollout_concurrency=cfg.train_rollout_concurrency,
        eval_solver_call_concurrency=cfg.eval_solver_call_concurrency,
        max_tokens=cfg.max_tokens,
        optimizer_max_tokens=cfg.optimizer_max_tokens,
        evaluator_max_tokens=cfg.evaluator_max_tokens,
        temperature=cfg.temperature,
        optimizer_temperature=cfg.optimizer_temperature,
        evaluator_temperature=cfg.evaluator_temperature,
        max_retries=cfg.max_retries,
        retry_sleep=cfg.retry_sleep,
        transient_retry_forever=cfg.transient_retry_forever,
        max_transient_retries=cfg.max_transient_retries,
        max_retry_backoff=cfg.max_retry_backoff,
        llm_call_logging=cfg.llm_call_logging,
        llm_call_timeout=cfg.llm_call_timeout,
        vote_tie_break=cfg.vote_tie_break,
        aggregation_mode=cfg.aggregation_mode,
        test_size=cfg.test_size,
        eval_test_each_epoch=int(cfg.eval_test_each_epoch),
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_min_delta=cfg.early_stopping_min_delta,
    )
    cmd = []

    _append_common_cli_args(cmd, args, task, setting, seed=42)

    idx = cmd.index("--resume_from_checkpoint")
    assert cmd[idx + 1] == "1"


def test_explicit_candidate_eval_budget_overrides_setting_defaults():
    setting = next(item for item in DEFAULT_EXPERIMENT_SETTINGS if item.name == "shared_scalar_tcs_vote_first")
    args = Namespace(
        candidate_eval_strategy="fixed_pool",
        candidate_eval_pool_size=20,
        candidate_eval_batch_size=10,
        candidate_eval_execution_mode="factorized_cached",
    )

    assert _explicit_cli_or_setting(args, setting, "candidate_eval_pool_size", 100) == 20
    assert _explicit_cli_or_setting(args, setting, "candidate_eval_batch_size", 20) == 10

    args.candidate_eval_pool_size = None
    args.candidate_eval_batch_size = None
    assert _explicit_cli_or_setting(args, setting, "candidate_eval_pool_size", 100) == 50
    assert _explicit_cli_or_setting(args, setting, "candidate_eval_batch_size", 20) == 24


def test_task_runner_passes_setting_name_to_run_metadata():
    cfg = Config()
    task = ComparisonTask(
        task_id="disambiguation_qa", benchmark="BBH", task_type="bbh",
        answer_format="option_letter", train_path="train.csv", val_path="val.csv", test_path="test.csv",
    )
    setting = next(
        item for item in DEFAULT_EXPERIMENT_SETTINGS
        if item.name == "shared_vote_tcs_competence_depth2_progressive_residual_hybrid"
    )
    args = Namespace(**{
        **vars(Namespace()),
        **cfg.to_flat_dict(),
        "dataset_format": "mars",
        "resume_from_checkpoint": 1,
    })
    cmd = []
    _append_common_cli_args(cmd, args, task, setting, seed=42)
    index = cmd.index("--experiment_setting")
    assert cmd[index + 1] == setting.name
    assert cmd[cmd.index("--beam_refresh_each_epoch") + 1] == "0"
    assert cmd[cmd.index("--teacher_critic_max_rounds") + 1] == "2"


def test_task_runner_passes_v9_ablation_switches_to_child_cli():
    cfg = Config()
    task = ComparisonTask(
        task_id="disambiguation_qa", benchmark="BBH", task_type="bbh",
        answer_format="option_letter", train_path="train.csv", val_path="val.csv", test_path="test.csv",
    )
    setting = next(
        item for item in DEFAULT_EXPERIMENT_SETTINGS
        if item.name == "shared_v9_sequential_accuracy_state"
    )
    args = Namespace(**{
        **cfg.to_flat_dict(),
        "dataset_format": "mars",
        "reward_mode": "",
        "resume_from_checkpoint": 1,
    })
    cmd = []
    _append_common_cli_args(cmd, args, task, setting, seed=42)

    assert cmd[cmd.index("--state_conditioned_enabled") + 1] == "1"
    assert cmd[cmd.index("--reward_mode") + 1] == "state_distribution_vote_reward"
    assert cmd[cmd.index("--candidate_selection_mode") + 1] == "sequential_accuracy_first_state_reward"
    assert cmd[cmd.index("--state_update_mode") + 1] == "sequential_single_agent"
    assert cmd[cmd.index("--state_distribution_reward_enabled") + 1] == "1"
    assert cmd[cmd.index("--state_vote_reward_enabled") + 1] == "0"
    assert cmd[cmd.index("--state_diversity_constraints_enabled") + 1] == "0"
    assert cmd[cmd.index("--candidate_batch_representative_size") + 1] == "12"
    assert cmd[cmd.index("--candidate_batch_coverage_size") + 1] == "6"
    assert cmd[cmd.index("--candidate_batch_conversion_size") + 1] == "6"


def test_compute_metrics_reads_vote_tie_rate_and_mars_delta(tmp_path):
    run_dir = tmp_path / "mmlu" / "shared_guarded_beam_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "config": {
                    "reward_mode": "guarded_diversity",
                    "baseline_only": False,
                    "init_mode": "shared",
                    "agents": 5,
                    "epochs": 1,
                    "train_size": 2,
                    "test_size": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(
        json.dumps(
            [
                {
                    "test": {
                        "vote_acc": 0.75,
                        "vote_tie_rate": 0.25,
                        "mean_embedding_diversity": 0.4,
                        "mean_invalid_rate": 0.1,
                    }
                }
            ]
        ),
        encoding="utf-8",
    )
    row = analyze_run(run_dir, {"mmlu": {"vote_acc": 0.7, "embedding_diversity": 0.3}})
    assert row["dataset"] == "mmlu"
    assert row["setting"] == "shared_guarded_beam"
    assert row["vote_tie_rate"] == 0.25
    assert round(row["vs_mars_delta_acc"], 6) == 0.05
    assert round(row["vs_mars_delta_diversity"], 6) == 0.1


def test_compute_metrics_uses_current_reward_component_names(tmp_path):
    run_dir = tmp_path / "bbh" / "shared_guarded_beam_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(
        json.dumps({"config": {"reward_mode": "vote_useful_diversity", "baseline_only": False}}),
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(
        json.dumps([{"test": {"vote_acc": 0.5, "oracle_acc": 0.75, "mean_useful_diversity": 0.2}}]),
        encoding="utf-8",
    )
    records = [
        {
            "event": "candidate_evaluated",
            "reward": 0.8,
            "target_agent_accuracy": 0.7,
            "baseline_target_accuracy": 0.4,
            "candidate_target_accuracy": 0.7,
            "coverage_delta": 0.25,
            "useful_diversity": 0.5,
            "invalid_guard_passed": True,
            "in_top_beam": True,
            "rank_in_beam": 1,
        },
        {
            "event": "beam_update_summary",
            "optimizer_architecture": "teacher_critic_student",
            "student_candidate_count_final": 0,
            "student_json_parse_failed": True,
            "student_json_retry_succeeded": False,
            "student_json_repair_succeeded": False,
            "optimizer_underfilled": True,
        },
    ]
    (run_dir / "update_logs.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    row = analyze_run(run_dir, {})
    assert row["target_agent_accuracy"] == 0.7
    assert row["candidate_target_accuracy"] == 0.7
    assert round(row["target_accuracy_delta"], 6) == 0.3
    assert row["coverage_delta"] == 0.25
    assert row["useful_diversity"] == 0.5
    assert row["student_final_failure_rate"] == 1.0
    assert row["student_json_parse_failure_rate"] == 1.0
    assert row["optimizer_underfilled_rate"] == 1.0


def test_student_failure_summary_uses_final_candidate_count_not_teacher_approval(tmp_path):
    run_dir = tmp_path / "task" / "shared_guarded_beam_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(json.dumps({"comparison_task_id": "task"}), encoding="utf-8")
    rows = [
        {
            "event": "beam_update_summary",
            "optimizer_architecture": "teacher_critic_student",
            "teacher_question_approved": False,
            "student_candidate_count_raw": 3,
            "student_candidate_count_final": 3,
            "student_failure_stage": "none",
        },
        {
            "event": "beam_update_summary",
            "optimizer_architecture": "teacher_critic_student",
            "teacher_question_approved": False,
            "student_candidate_count_raw": 1,
            "student_candidate_count_final": 0,
            "student_failure_stage": "all_candidates_filtered_schema",
            "student_json_parse_failed": False,
        },
    ]
    log = run_dir / "update_logs.jsonl"
    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = summarize_run(log)
    assert len(summary) == 1
    assert summary[0]["student_final_failure_count"] == 1
    assert summary[0]["student_candidate_count_raw"] == 1
    assert summary[0]["student_candidate_count_final"] == 0
    assert summary[0]["final_student_failure"] is True
    assert summary[0]["rate_within_update_summaries"] == 0.5


def test_student_failure_summary_marks_retry_recovered_as_non_final_failure(tmp_path):
    run_dir = tmp_path / "task" / "shared_guarded_beam_seed42"
    run_dir.mkdir(parents=True)
    rows = [
        {
            "event": "beam_update_summary",
            "optimizer_architecture": "teacher_critic_student",
            "student_candidate_count_raw": 0,
            "student_candidate_count_final": 2,
            "student_failure_stage": "none",
            "student_json_parse_failed": True,
            "student_json_retry_attempted": True,
            "student_json_retry_succeeded": True,
        }
    ]
    log = run_dir / "update_logs.jsonl"
    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = summarize_run(log)
    assert len(summary) == 1
    assert summary[0]["student_final_failure_count"] == 0
    assert summary[0]["student_retry_recovery_count"] == 1
    assert summary[0]["recovery_status"] == "retry_recovered"
    assert summary[0]["final_student_failure"] is False


def test_compute_metrics_does_not_count_non_student_summary_as_final_failure(tmp_path):
    run_dir = tmp_path / "bbh" / "shared_guarded_beam_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(json.dumps({"config": {"reward_mode": "guarded_diversity"}}), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps([{"test": {"vote_acc": 0.5}}]), encoding="utf-8")
    records = [
        {"event": "beam_update_summary", "optimizer_architecture": "one_shot", "active_prompt_changed": True},
    ]
    (run_dir / "update_logs.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")

    row = analyze_run(run_dir, {})
    assert row["student_final_failure_rate"] == 0.0
    assert row["student_json_parse_failure_rate"] == 0.0


def test_experiment_runner_defaults_match_config():
    import scripts.run_experiments as run_experiments
    import scripts.run_task_level_accuracy as run_task_level_accuracy

    defaults = Config()
    assert run_experiments.Config().agent_model == defaults.agent_model
    assert run_task_level_accuracy.Config().optimizer_model == defaults.optimizer_model
