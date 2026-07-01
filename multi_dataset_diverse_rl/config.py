from dataclasses import dataclass
import argparse


DEFAULT_TEMPERATURE = 0.0
DEFAULT_OPTIMIZER_TEMPERATURE = 0.5
DEFAULT_EVALUATOR_TEMPERATURE = 0.0


@dataclass
class Config:
    task_type: str = "auto"
    dataset_format: str = "legacy"

    agent_model: str = "deepseek-chat"
    optimizer_model: str = "deepseek-v4-flash"
    evaluator_model: str = "deepseek-v4-flash"

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
    reward_mode: str = "guarded_diversity"
    beam_size: int = 3
    num_candidates_per_parent: int = 2
    beam_refresh_each_epoch: bool = True
    homogeneity_overlap_threshold: float = 0.55
    homogeneity_pressure_tie_eps: float = 0.03
    max_homogeneous_cases_per_agent: int = 4
    random_window_cases_per_agent: int = 2
    hard_validity_cases_per_agent: int = 2
    invalid_repair_rate_threshold: float = 0.25

    reward_weight_diversity: float = 0.5
    reward_weight_local_validity: float = 0.2
    reward_weight_team_accuracy: float = 0.1
    reward_weight_invalid_score: float = 0.2
    accuracy_guard_epsilon: float = 0.02
    reward_weight_div_delta: float = 0.3
    reward_weight_invalid_delta: float = 0.5
    use_baseline_relative_reward: bool = True

    diversity_metric: str = "trace_embedding"
    use_joint_trace_diversity_evaluator: bool = False
    local_validity_binary: bool = True
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
    train_rollout_concurrency: int = 0
    eval_solver_call_concurrency: int = 225
    local_evaluator_batch_size: int = 5
    solver_api_key_env: str = ""
    solver_base_url_env: str = ""
    evaluator_api_key_env: str = ""
    evaluator_base_url_env: str = ""
    vote_tie_break: str = "random"

    def __post_init__(self):
        if not str(self.agent_model or "").strip():
            self.agent_model = "deepseek-chat"
        if not str(self.optimizer_model or "").strip():
            self.optimizer_model = "deepseek-v4-flash"
        if not str(self.evaluator_model or "").strip():
            self.evaluator_model = "deepseek-v4-flash"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="auto", choices=["auto", "gsm8k", "mmlu", "bbh"])
    parser.add_argument("--dataset_format", type=str, default="legacy", choices=["legacy", "mars"])

    parser.add_argument("--agent_model", type=str, default="deepseek-chat")
    parser.add_argument("--optimizer_model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--evaluator_model", type=str, default="deepseek-v4-flash")

    parser.add_argument("--train_path", type=str, default="train.jsonl")
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="test.jsonl")
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--val_size", type=int, default=150)
    parser.add_argument("--val_split_ratio", type=float, default=0.2)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--eval_test_each_epoch", type=int, default=0, choices=[0, 1])

    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--init_mode", type=str, default="shared", choices=["shared", "bank"])
    parser.add_argument(
        "--shared_prompt",
        type=str,
        default="You are a careful reasoning solver. Produce a compact, explicit reasoning trace, make your decision procedure visible, verify key logic, and give exactly one final answer.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--update_every", type=int, default=10)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=20)
    parser.add_argument("--baseline_only", type=int, default=0, choices=[0, 1])

    parser.add_argument("--search_mode", type=str, default="evolutionary_beam", choices=["evolutionary_beam"])
    parser.add_argument("--reward_mode", type=str, default="guarded_diversity", choices=["embedding_local_acc_invalid", "accuracy_only", "guarded_diversity"])
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--num_candidates_per_parent", type=int, default=2)
    parser.add_argument("--beam_refresh_each_epoch", type=int, default=1, choices=[0, 1])
    parser.add_argument("--homogeneity_overlap_threshold", type=float, default=0.55)
    parser.add_argument("--homogeneity_pressure_tie_eps", type=float, default=0.03)
    parser.add_argument("--max_homogeneous_cases_per_agent", type=int, default=4)
    parser.add_argument("--random_window_cases_per_agent", type=int, default=2)
    parser.add_argument("--hard_validity_cases_per_agent", type=int, default=2)
    parser.add_argument("--invalid_repair_rate_threshold", type=float, default=0.25)

    parser.add_argument("--reward_weight_diversity", type=float, default=0.5)
    parser.add_argument("--reward_weight_local_validity", type=float, default=0.2)
    parser.add_argument("--reward_weight_team_accuracy", type=float, default=0.1)
    parser.add_argument("--reward_weight_invalid_score", type=float, default=0.2)
    parser.add_argument("--accuracy_guard_epsilon", type=float, default=0.02)
    parser.add_argument("--reward_weight_div_delta", type=float, default=0.3)
    parser.add_argument("--reward_weight_invalid_delta", type=float, default=0.5)
    parser.add_argument("--use_baseline_relative_reward", type=int, default=1, choices=[0, 1])
    parser.add_argument("--diversity_metric", type=str, default="trace_embedding", choices=["trace_embedding"])
    parser.add_argument("--use_joint_trace_diversity_evaluator", type=int, default=0, choices=[0, 1])
    parser.add_argument("--local_validity_binary", type=int, default=1, choices=[0, 1])
    parser.add_argument("--invalid_binary", type=int, default=1, choices=[0, 1])
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--trace_embedding_chunk_words", type=int, default=320)
    parser.add_argument("--trace_embedding_chunk_overlap", type=int, default=40)

    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--optimizer_max_tokens", type=int, default=6000)
    parser.add_argument("--evaluator_max_tokens", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--optimizer_temperature", type=float, default=DEFAULT_OPTIMIZER_TEMPERATURE)
    parser.add_argument("--evaluator_temperature", type=float, default=DEFAULT_EVALUATOR_TEMPERATURE)

    parser.add_argument("--out_dir", type=str, default="runs_trace_beam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=1.5)
    parser.add_argument("--transient_retry_forever", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_transient_retries", type=int, default=0)
    parser.add_argument("--max_retry_backoff", type=float, default=30.0)
    parser.add_argument("--llm_call_logging", type=int, default=1, choices=[0, 1])
    parser.add_argument("--llm_call_timeout", type=float, default=120.0)
    parser.add_argument("--candidate_eval_concurrency", type=int, default=0)
    parser.add_argument("--candidate_eval_strategy", type=str, default="random", choices=["random", "fixed_pool", "stratified"])
    parser.add_argument("--candidate_eval_pool_size", type=int, default=100)
    parser.add_argument("--candidate_eval_repeats", type=int, default=1)
    parser.add_argument("--candidate_eval_seed_offset", type=int, default=1000)
    parser.add_argument("--candidate_reuse_recorded_rollouts", type=int, default=1, choices=[0, 1])
    parser.add_argument("--train_rollout_concurrency", type=int, default=0)
    parser.add_argument("--eval_solver_call_concurrency", type=int, default=225)
    parser.add_argument("--local_evaluator_batch_size", type=int, default=5)
    parser.add_argument("--solver_api_key_env", type=str, default="")
    parser.add_argument("--solver_base_url_env", type=str, default="")
    parser.add_argument("--evaluator_api_key_env", type=str, default="")
    parser.add_argument("--evaluator_base_url_env", type=str, default="")
    parser.add_argument("--vote_tie_break", type=str, default="random", choices=["first", "random", "abstain"])
    return parser
