from dataclasses import dataclass
import argparse


@dataclass
class Config:
    task_type: str = "auto"
    model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o-mini"
    rewriter_model: str = "gpt-4o-mini"
    family_expansion_model: str = "deepseek-v4-pro"
    family_expansion_enabled: bool = True
    family_taxonomy_path: str = "family_taxonomy.json"
    use_dual_family_labels: bool = True
    primary_family_weight: float = 0.7
    secondary_family_weight: float = 0.3
    same_major_family_weight: float = 0.5
    macro_diversity_weight: float = 0.5
    family_confidence_threshold: float = 0.4
    family_rejudge_on_low_confidence: bool = True
    min_summary_words: int = 60
    max_summary_tokens: int = 512
    min_evidence_spans: int = 1
    reward_tie_eps: float = 0.03
    invalid_tolerance: float = 0.1

    train_path: str = "train.jsonl"
    test_path: str = "test.jsonl"
    train_size: int = 200
    test_size: int = 100

    agents: int = 4
    init_mode: str = "shared"
    shared_prompt: str = "You are a careful reasoning solver. Solve step by step, verify key logic, and output exactly one FINAL_ANSWER line in the required format."
    epochs: int = 2
    update_every: int = 5
    candidate_eval_batch_size: int = 3
    baseline_only: bool = False

    max_tokens: int = 1000
    critic_max_tokens: int = 8000
    rewriter_max_tokens: int = 1000

    temperature: float = 0.2
    critic_temperature: float = 0.3
    rewriter_temperature: float = 0.5

    out_dir: str = "runs_tg_rl"
    seed: int = 42
    max_retries: int = 3
    retry_sleep: float = 1.5
    transient_retry_forever: bool = True
    max_transient_retries: int = 0
    max_retry_backoff: float = 30.0
    llm_call_logging: bool = True
    llm_call_timeout: float = 120.0

    bandit_lr: float = 0.2
    baseline_momentum: float = 0.9
    homogeneity_window: int = 50

    lambda_diversity: float = 0.5
    lambda_homogeneity: float = 0.35
    lambda_invalid_trace: float = 0.30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="auto", choices=["auto", "gsm8k", "mmlu"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--critic_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--rewriter_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_expansion_model", type=str, default="deepseek-v4-pro")
    parser.add_argument("--family_expansion_enabled", type=int, default=1, choices=[0, 1])
    parser.add_argument("--family_taxonomy_path", type=str, default="family_taxonomy.json")
    parser.add_argument("--use_dual_family_labels", type=int, default=1, choices=[0, 1])
    parser.add_argument("--primary_family_weight", type=float, default=0.7)
    parser.add_argument("--secondary_family_weight", type=float, default=0.3)
    parser.add_argument("--same_major_family_weight", type=float, default=0.5)
    parser.add_argument("--macro_diversity_weight", type=float, default=0.5)
    parser.add_argument("--family_confidence_threshold", type=float, default=0.4)
    parser.add_argument("--family_rejudge_on_low_confidence", type=int, default=1, choices=[0, 1])
    parser.add_argument("--min_summary_words", type=int, default=60)
    parser.add_argument("--max_summary_tokens", type=int, default=512)
    parser.add_argument("--min_evidence_spans", type=int, default=1)
    parser.add_argument("--reward_tie_eps", type=float, default=0.03)
    parser.add_argument("--invalid_tolerance", type=float, default=0.1)

    parser.add_argument("--train_path", type=str, default="train.jsonl")
    parser.add_argument("--test_path", type=str, default="test.jsonl")
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--test_size", type=int, default=100)

    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--init_mode", type=str, default="shared", choices=["shared", "bank"])
    parser.add_argument(
        "--shared_prompt",
        type=str,
        default="You are a careful reasoning solver. Solve step by step, verify key logic, and output exactly one FINAL_ANSWER line in the required format.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--update_every", type=int, default=5)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=3)
    parser.add_argument("--baseline_only", type=int, default=0, choices=[0, 1])

    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--critic_max_tokens", type=int, default=8000)
    parser.add_argument("--rewriter_max_tokens", type=int, default=1000)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--critic_temperature", type=float, default=0.3)
    parser.add_argument("--rewriter_temperature", type=float, default=0.5)

    parser.add_argument("--out_dir", type=str, default="runs_tg_rl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=1.5)
    parser.add_argument("--transient_retry_forever", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_transient_retries", type=int, default=0)
    parser.add_argument("--max_retry_backoff", type=float, default=30.0)
    parser.add_argument("--llm_call_logging", type=int, default=1, choices=[0, 1])
    parser.add_argument("--llm_call_timeout", type=float, default=120.0)

    parser.add_argument("--bandit_lr", type=float, default=0.2)
    parser.add_argument("--baseline_momentum", type=float, default=0.9)
    # homogeneity_window is auto-aligned to update_every at runtime.

    parser.add_argument("--lambda_diversity", type=float, default=0.5)
    parser.add_argument("--lambda_homogeneity", type=float, default=0.35)
    parser.add_argument("--lambda_invalid_trace", type=float, default=0.30)
    return parser
