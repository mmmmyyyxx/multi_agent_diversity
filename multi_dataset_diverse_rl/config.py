from dataclasses import dataclass
import argparse


DEFAULT_TEMPERATURE = 0.0
DEFAULT_OPTIMIZER_TEMPERATURE = 0.5
DEFAULT_EVALUATOR_TEMPERATURE = 0.0


@dataclass
class Config:
    task_type: str = "auto"
    dataset_format: str = "legacy"
    comparison_task_id: str = ""
    benchmark: str = ""
    answer_format: str = ""

    agent_model: str = "deepseek-chat"
    optimizer_model: str = "deepseek-chat"
    evaluator_model: str = "deepseek-chat"

    train_path: str = "train.jsonl"
    val_path: str = ""
    test_path: str = "test.jsonl"
    train_size: int = 200
    val_size: int = 100
    val_split_ratio: float = 0.2
    test_size: int = 200
    eval_test_each_epoch: bool = False

    agents: int = 5
    init_mode: str = "shared"
    shared_prompt: str = "You are a careful reasoning solver. Produce a compact, explicit reasoning trace, make your decision procedure visible, verify key logic, and give exactly one final answer."
    epochs: int = 2
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    update_every: int = 10
    candidate_eval_batch_size: int = 20
    baseline_only: bool = False

    search_mode: str = "evolutionary_beam"
    reward_mode: str = "vote_useful_diversity"
    candidate_selection_mode: str = "scalar_reward"
    best_state_selection_mode: str = "vote_first"
    beam_size: int = 3
    num_candidates_per_parent: int = 2
    optimizer_parent_concurrency: int = 2
    beam_refresh_each_epoch: bool = True
    homogeneity_overlap_threshold: float = 0.55
    homogeneity_pressure_tie_eps: float = 0.03
    max_homogeneous_cases_per_agent: int = 4
    random_window_cases_per_agent: int = 2
    hard_validity_cases_per_agent: int = 2
    invalid_repair_rate_threshold: float = 0.25

    accuracy_guard_epsilon: float = 0.02
    reward_weight_div_delta: float = 0.3
    reward_weight_invalid_delta: float = 0.5
    reward_weight_vote_delta: float = 0.3
    reward_weight_vote_margin: float = 0.2
    reward_weight_boundary_diversity: float = 0.2
    invalid_guard_epsilon: float = 0.05
    use_baseline_relative_reward: bool = True
    reward_schedule_mode: str = "phase_adaptive"
    reward_diversity_warmup_updates: int = 10
    reward_weight_div_delta_early: float = 0.8
    reward_weight_div_delta_late: float = 0.2
    reward_weight_vote_delta_early: float = 0.4
    reward_weight_vote_delta_late: float = 0.3
    reward_weight_vote_margin_early: float = 0.5
    reward_weight_vote_margin_late: float = 0.25
    reward_weight_boundary_diversity_early: float = 0.3
    reward_weight_boundary_diversity_late: float = 0.2
    reward_weight_target_accuracy_early: float = 0.9
    reward_weight_target_accuracy_late: float = 1.0
    accuracy_guard_epsilon_early: float = 0.03
    accuracy_guard_epsilon_late: float = 0.01
    optimizer_architecture: str = "teacher_critic_student"
    teacher_critic_max_rounds: int = 3
    teacher_question_pass_threshold: float = 0.75
    teacher_temperature: float = 0.4
    critic_temperature: float = 0.0
    student_temperature: float = 0.5
    teacher_max_tokens: int = 1200
    critic_max_tokens: int = 1000
    student_max_tokens: int = 1800
    student_json_retry_on_parse_fail: bool = True
    student_json_max_retries: int = 5
    student_json_repair_enabled: bool = True
    student_json_repair_max_tokens: int = 1200
    student_json_repair_temperature: float = 0.0
    student_candidate_schema_mode: str = "compact"
    student_candidate_max_chars_per_field: int = 320
    student_candidate_prompt_max_chars: int = 900
    student_force_minified_json: bool = True
    teacher_critic_use_voting_failure: bool = True
    optimizer_fallback_mode: str = "none"
    no_effective_evolution_patience: int = 10
    no_effective_evolution_min_optimizer_candidates: int = 1
    no_effective_evolution_stop_enabled: bool = True

    diversity_metric: str = "trace_embedding"
    use_joint_trace_diversity_evaluator: bool = False
    invalid_binary: bool = True
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    trace_embedding_chunk_words: int = 320
    trace_embedding_chunk_overlap: int = 40

    max_tokens: int = 1000
    optimizer_max_tokens: int = 1400
    evaluator_max_tokens: int = 1200
    temperature: float = DEFAULT_TEMPERATURE
    optimizer_temperature: float = DEFAULT_OPTIMIZER_TEMPERATURE
    evaluator_temperature: float = DEFAULT_EVALUATOR_TEMPERATURE

    out_dir: str = "runs_trace_beam"
    seed: int = 42
    resume_from_checkpoint: bool = False
    max_retries: int = 3
    retry_sleep: float = 1.5
    transient_retry_forever: bool = True
    max_transient_retries: int = 0
    max_retry_backoff: float = 30.0
    llm_call_logging: bool = True
    llm_call_timeout: float = 120.0
    candidate_eval_concurrency: int = 0
    candidate_eval_strategy: str = "random"
    candidate_eval_pool_size: int = 100
    candidate_eval_pool_actual_size: int = 0
    candidate_eval_repeats: int = 1
    candidate_eval_seed_offset: int = 1000
    candidate_reuse_recorded_rollouts: bool = True
    candidate_eval_execution_mode: str = "legacy"
    solver_rollout_singleflight: bool = True
    candidate_eval_prompt_dedup: bool = True
    candidate_eval_cache_logging: bool = True
    train_rollout_concurrency: int = 0
    eval_solver_call_concurrency: int = 225
    solver_api_key_env: str = ""
    solver_base_url_env: str = ""
    evaluator_api_key_env: str = ""
    evaluator_base_url_env: str = ""
    vote_tie_break: str = "random"
    aggregation_mode: str = "majority"

    def __post_init__(self):
        if not str(self.agent_model or "").strip():
            self.agent_model = "deepseek-chat"
        if not str(self.optimizer_model or "").strip():
            self.optimizer_model = "deepseek-chat"
        if not str(self.evaluator_model or "").strip():
            self.evaluator_model = "deepseek-chat"


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default=defaults.task_type, choices=["auto", "gsm8k", "mmlu", "bbh"])
    parser.add_argument("--dataset_format", type=str, default=defaults.dataset_format, choices=["legacy", "mars"])
    parser.add_argument("--comparison_task_id", type=str, default=defaults.comparison_task_id)
    parser.add_argument("--benchmark", type=str, default=defaults.benchmark)
    parser.add_argument("--answer_format", type=str, default=defaults.answer_format, choices=["", "option_letter", "boolean", "yes_no", "valid_invalid", "numeric", "free_text"])

    parser.add_argument("--agent_model", type=str, default=defaults.agent_model)
    parser.add_argument("--optimizer_model", type=str, default=defaults.optimizer_model)
    parser.add_argument("--evaluator_model", type=str, default=defaults.evaluator_model)

    parser.add_argument("--train_path", type=str, default=defaults.train_path)
    parser.add_argument("--val_path", type=str, default=defaults.val_path)
    parser.add_argument("--test_path", type=str, default=defaults.test_path)
    parser.add_argument("--train_size", type=int, default=defaults.train_size)
    parser.add_argument("--val_size", type=int, default=defaults.val_size)
    parser.add_argument("--val_split_ratio", type=float, default=defaults.val_split_ratio)
    parser.add_argument("--test_size", type=int, default=defaults.test_size)
    parser.add_argument("--eval_test_each_epoch", type=int, default=int(defaults.eval_test_each_epoch), choices=[0, 1])

    parser.add_argument("--agents", type=int, default=defaults.agents)
    parser.add_argument("--init_mode", type=str, default=defaults.init_mode, choices=["shared", "bank"])
    parser.add_argument(
        "--shared_prompt",
        type=str,
        default=defaults.shared_prompt,
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--early_stopping_patience", type=int, default=defaults.early_stopping_patience)
    parser.add_argument("--early_stopping_min_delta", type=float, default=defaults.early_stopping_min_delta)
    parser.add_argument("--update_every", type=int, default=defaults.update_every)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=defaults.candidate_eval_batch_size)
    parser.add_argument("--baseline_only", type=int, default=int(defaults.baseline_only), choices=[0, 1])

    parser.add_argument("--search_mode", type=str, default=defaults.search_mode, choices=["evolutionary_beam"])
    parser.add_argument("--reward_mode", type=str, default=defaults.reward_mode, choices=["accuracy_only", "guarded_diversity", "vote_useful_diversity"])
    parser.add_argument("--candidate_selection_mode", type=str, default=defaults.candidate_selection_mode, choices=["scalar_reward", "vote_pareto"])
    parser.add_argument("--best_state_selection_mode", type=str, default=defaults.best_state_selection_mode, choices=["existing", "vote_first"])
    parser.add_argument("--beam_size", type=int, default=defaults.beam_size)
    parser.add_argument("--num_candidates_per_parent", type=int, default=defaults.num_candidates_per_parent)
    parser.add_argument("--optimizer_parent_concurrency", type=int, default=defaults.optimizer_parent_concurrency)
    parser.add_argument("--beam_refresh_each_epoch", type=int, default=int(defaults.beam_refresh_each_epoch), choices=[0, 1])
    parser.add_argument("--homogeneity_overlap_threshold", type=float, default=defaults.homogeneity_overlap_threshold)
    parser.add_argument("--homogeneity_pressure_tie_eps", type=float, default=defaults.homogeneity_pressure_tie_eps)
    parser.add_argument("--max_homogeneous_cases_per_agent", type=int, default=defaults.max_homogeneous_cases_per_agent)
    parser.add_argument("--random_window_cases_per_agent", type=int, default=defaults.random_window_cases_per_agent)
    parser.add_argument("--hard_validity_cases_per_agent", type=int, default=defaults.hard_validity_cases_per_agent)
    parser.add_argument("--invalid_repair_rate_threshold", type=float, default=defaults.invalid_repair_rate_threshold)

    parser.add_argument("--accuracy_guard_epsilon", type=float, default=defaults.accuracy_guard_epsilon)
    parser.add_argument("--reward_weight_div_delta", type=float, default=defaults.reward_weight_div_delta)
    parser.add_argument("--reward_weight_invalid_delta", type=float, default=defaults.reward_weight_invalid_delta)
    parser.add_argument("--reward_weight_vote_delta", type=float, default=defaults.reward_weight_vote_delta)
    parser.add_argument("--reward_weight_vote_margin", type=float, default=defaults.reward_weight_vote_margin)
    parser.add_argument("--reward_weight_boundary_diversity", type=float, default=defaults.reward_weight_boundary_diversity)
    parser.add_argument("--invalid_guard_epsilon", type=float, default=defaults.invalid_guard_epsilon)
    parser.add_argument("--use_baseline_relative_reward", type=int, default=int(defaults.use_baseline_relative_reward), choices=[0, 1])
    parser.add_argument("--reward_schedule_mode", type=str, default=defaults.reward_schedule_mode, choices=["static", "phase_adaptive"])
    parser.add_argument("--reward_diversity_warmup_updates", type=int, default=defaults.reward_diversity_warmup_updates)
    parser.add_argument("--reward_weight_div_delta_early", type=float, default=defaults.reward_weight_div_delta_early)
    parser.add_argument("--reward_weight_div_delta_late", type=float, default=defaults.reward_weight_div_delta_late)
    parser.add_argument("--reward_weight_vote_delta_early", type=float, default=defaults.reward_weight_vote_delta_early)
    parser.add_argument("--reward_weight_vote_delta_late", type=float, default=defaults.reward_weight_vote_delta_late)
    parser.add_argument("--reward_weight_vote_margin_early", type=float, default=defaults.reward_weight_vote_margin_early)
    parser.add_argument("--reward_weight_vote_margin_late", type=float, default=defaults.reward_weight_vote_margin_late)
    parser.add_argument("--reward_weight_boundary_diversity_early", type=float, default=defaults.reward_weight_boundary_diversity_early)
    parser.add_argument("--reward_weight_boundary_diversity_late", type=float, default=defaults.reward_weight_boundary_diversity_late)
    parser.add_argument("--reward_weight_target_accuracy_early", type=float, default=defaults.reward_weight_target_accuracy_early)
    parser.add_argument("--reward_weight_target_accuracy_late", type=float, default=defaults.reward_weight_target_accuracy_late)
    parser.add_argument("--accuracy_guard_epsilon_early", type=float, default=defaults.accuracy_guard_epsilon_early)
    parser.add_argument("--accuracy_guard_epsilon_late", type=float, default=defaults.accuracy_guard_epsilon_late)
    parser.add_argument("--optimizer_architecture", type=str, default=defaults.optimizer_architecture, choices=["one_shot", "teacher_critic_student"])
    parser.add_argument("--teacher_critic_max_rounds", type=int, default=defaults.teacher_critic_max_rounds)
    parser.add_argument("--teacher_question_pass_threshold", type=float, default=defaults.teacher_question_pass_threshold)
    parser.add_argument("--teacher_temperature", type=float, default=defaults.teacher_temperature)
    parser.add_argument("--critic_temperature", type=float, default=defaults.critic_temperature)
    parser.add_argument("--student_temperature", type=float, default=defaults.student_temperature)
    parser.add_argument("--teacher_max_tokens", type=int, default=defaults.teacher_max_tokens)
    parser.add_argument("--critic_max_tokens", type=int, default=defaults.critic_max_tokens)
    parser.add_argument("--student_max_tokens", type=int, default=defaults.student_max_tokens)
    parser.add_argument("--student_json_retry_on_parse_fail", type=int, default=int(defaults.student_json_retry_on_parse_fail), choices=[0, 1])
    parser.add_argument("--student_json_max_retries", type=int, default=defaults.student_json_max_retries)
    parser.add_argument("--student_json_repair_enabled", type=int, default=int(defaults.student_json_repair_enabled), choices=[0, 1])
    parser.add_argument("--student_json_repair_max_tokens", type=int, default=defaults.student_json_repair_max_tokens)
    parser.add_argument("--student_json_repair_temperature", type=float, default=defaults.student_json_repair_temperature)
    parser.add_argument("--student_candidate_schema_mode", type=str, default=defaults.student_candidate_schema_mode, choices=["compact", "verbose"])
    parser.add_argument("--student_candidate_max_chars_per_field", type=int, default=defaults.student_candidate_max_chars_per_field)
    parser.add_argument("--student_candidate_prompt_max_chars", type=int, default=defaults.student_candidate_prompt_max_chars)
    parser.add_argument("--student_force_minified_json", type=int, default=int(defaults.student_force_minified_json), choices=[0, 1])
    parser.add_argument("--teacher_critic_use_voting_failure", type=int, default=int(defaults.teacher_critic_use_voting_failure), choices=[0, 1])
    parser.add_argument("--optimizer_fallback_mode", type=str, default=defaults.optimizer_fallback_mode, choices=["none", "template"])
    parser.add_argument("--no_effective_evolution_patience", type=int, default=defaults.no_effective_evolution_patience)
    parser.add_argument("--no_effective_evolution_min_optimizer_candidates", type=int, default=defaults.no_effective_evolution_min_optimizer_candidates)
    parser.add_argument("--no_effective_evolution_stop_enabled", type=int, default=int(defaults.no_effective_evolution_stop_enabled), choices=[0, 1])
    parser.add_argument("--diversity_metric", type=str, default=defaults.diversity_metric, choices=["trace_embedding"])
    parser.add_argument("--use_joint_trace_diversity_evaluator", type=int, default=int(defaults.use_joint_trace_diversity_evaluator), choices=[0, 1])
    parser.add_argument("--invalid_binary", type=int, default=int(defaults.invalid_binary), choices=[0, 1])
    parser.add_argument("--embedding_model", type=str, default=defaults.embedding_model)
    parser.add_argument("--trace_embedding_chunk_words", type=int, default=defaults.trace_embedding_chunk_words)
    parser.add_argument("--trace_embedding_chunk_overlap", type=int, default=defaults.trace_embedding_chunk_overlap)

    parser.add_argument("--max_tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--optimizer_max_tokens", type=int, default=defaults.optimizer_max_tokens)
    parser.add_argument("--evaluator_max_tokens", type=int, default=defaults.evaluator_max_tokens)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--optimizer_temperature", type=float, default=defaults.optimizer_temperature)
    parser.add_argument("--evaluator_temperature", type=float, default=defaults.evaluator_temperature)

    parser.add_argument("--out_dir", type=str, default=defaults.out_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--resume_from_checkpoint", type=int, default=int(defaults.resume_from_checkpoint), choices=[0, 1])
    parser.add_argument("--max_retries", type=int, default=defaults.max_retries)
    parser.add_argument("--retry_sleep", type=float, default=defaults.retry_sleep)
    parser.add_argument("--transient_retry_forever", type=int, default=int(defaults.transient_retry_forever), choices=[0, 1])
    parser.add_argument("--max_transient_retries", type=int, default=defaults.max_transient_retries)
    parser.add_argument("--max_retry_backoff", type=float, default=defaults.max_retry_backoff)
    parser.add_argument("--llm_call_logging", type=int, default=int(defaults.llm_call_logging), choices=[0, 1])
    parser.add_argument("--llm_call_timeout", type=float, default=defaults.llm_call_timeout)
    parser.add_argument("--candidate_eval_concurrency", type=int, default=defaults.candidate_eval_concurrency)
    parser.add_argument("--candidate_eval_strategy", type=str, default=defaults.candidate_eval_strategy, choices=["random", "fixed_pool", "stratified"])
    parser.add_argument("--candidate_eval_pool_size", type=int, default=defaults.candidate_eval_pool_size)
    parser.add_argument("--candidate_eval_repeats", type=int, default=defaults.candidate_eval_repeats)
    parser.add_argument("--candidate_eval_seed_offset", type=int, default=defaults.candidate_eval_seed_offset)
    parser.add_argument("--candidate_reuse_recorded_rollouts", type=int, default=int(defaults.candidate_reuse_recorded_rollouts), choices=[0, 1])
    parser.add_argument("--candidate_eval_execution_mode", type=str, default=defaults.candidate_eval_execution_mode, choices=["legacy", "factorized_cached"])
    parser.add_argument("--solver_rollout_singleflight", type=int, default=int(defaults.solver_rollout_singleflight), choices=[0, 1])
    parser.add_argument("--candidate_eval_prompt_dedup", type=int, default=int(defaults.candidate_eval_prompt_dedup), choices=[0, 1])
    parser.add_argument("--candidate_eval_cache_logging", type=int, default=int(defaults.candidate_eval_cache_logging), choices=[0, 1])
    parser.add_argument("--train_rollout_concurrency", type=int, default=defaults.train_rollout_concurrency)
    parser.add_argument("--eval_solver_call_concurrency", type=int, default=defaults.eval_solver_call_concurrency)
    parser.add_argument("--solver_api_key_env", type=str, default=defaults.solver_api_key_env)
    parser.add_argument("--solver_base_url_env", type=str, default=defaults.solver_base_url_env)
    parser.add_argument("--evaluator_api_key_env", type=str, default=defaults.evaluator_api_key_env)
    parser.add_argument("--evaluator_base_url_env", type=str, default=defaults.evaluator_base_url_env)
    parser.add_argument("--vote_tie_break", type=str, default=defaults.vote_tie_break, choices=["first", "random", "abstain"])
    parser.add_argument("--aggregation_mode", type=str, default=defaults.aggregation_mode, choices=["majority", "weighted_vote", "verifier_select"])
    return parser
