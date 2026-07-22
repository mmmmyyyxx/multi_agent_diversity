from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class DataConfig:
    task_type: str = "auto"
    dataset_format: str = "legacy"
    comparison_task_id: str = ""
    benchmark: str = ""
    answer_format: str = ""
    train_path: str = "train.jsonl"
    val_path: str = ""
    test_path: str = "test.jsonl"
    manifest_sha256: str = ""
    train_size: int = 200
    val_size: int = 100
    test_size: int = 200


@dataclass(frozen=True)
class ModelConfig:
    agent_model: str = "deepseek-chat"
    optimizer_model: str = "deepseek-chat"
    evaluator_model: str = "deepseek-chat"
    solver_api_key_env: str = ""
    solver_base_url_env: str = ""
    optimizer_api_key_env: str = ""
    optimizer_base_url_env: str = ""
    evaluator_api_key_env: str = ""
    evaluator_base_url_env: str = ""
    temperature: float = 0.0
    max_tokens: int = 1800


@dataclass(frozen=True)
class TrainingConfig:
    method_version: str = "peer_state_counterfactual_v1"
    experiment_setting: str = "shared_peer_state_full"
    agents: int = 5
    epochs: int = 3
    update_every: int = 10
    seed: int = 42
    shared_prompt: str = "You are a careful reasoning solver. Use an explicit decision procedure, verify the key inference, and end with exactly one FINAL_ANSWER line."
    initialization_mode: str = "shared_identical"
    provided_prompts_json: str = ""


@dataclass(frozen=True)
class TCSConfig:
    teacher_critic_max_rounds: int = 3
    critic_approval_threshold: float = 0.75
    teacher_temperature: float = 0.4
    critic_temperature: float = 0.0
    student_temperature: float = 0.5
    teacher_max_tokens: int = 1200
    critic_max_tokens: int = 1000
    student_max_tokens: int = 1800
    student_json_max_retries: int = 5
    num_candidates_per_parent: int = 2
    tcs_assigned_coverage_limit: int = 6
    tcs_assigned_conversion_limit: int = 6
    tcs_preservation_limit: int = 6
    tcs_representative_limit: int = 6
    tcs_context_max_chars: int = 24000


@dataclass(frozen=True)
class PeerStateConfig:
    aggregation_mode: str = "plurality"
    vote_tie_break: str = "abstain"
    soft_vote_tau: float = 1.0
    probe_version: str = "peer_state_fixed_probe_v1"
    parser_version: str = "task_parser_v1"


@dataclass(frozen=True)
class ResponsibilityConfig:
    responsibility_switch_margin: float = 0.05
    responsibility_max_wait_updates: int = 8


@dataclass(frozen=True)
class CandidateEvaluationConfig:
    candidate_eval_pool_size: int = 75
    eval_solver_call_concurrency: int = 20
    stage_a_representative_size: int = 12
    stage_a_coverage_size: int = 6
    stage_a_conversion_size: int = 6
    stage_a_preservation_size: int = 4
    stage_a_channel_top_k: int = 2
    stage_b_candidate_budget: int = 2


@dataclass(frozen=True)
class ConstraintConfig:
    local_accuracy_loss_epsilon: float = 0.0
    global_accuracy_loss_epsilon: float = 0.0
    invalid_guard_epsilon: float = 0.0
    vote_loss_limit: int = 0
    unique_correct_loss_limit: int = 0
    pivotal_loss_limit: int = 0
    min_soft_utility_gain: float = 0.005
    validation_accuracy_epsilon: float = 0.0
    validation_mean_epsilon: float = 0.0


@dataclass(frozen=True)
class PersistenceConfig:
    out_dir: str = "runs_peer_state"
    resume_from_checkpoint: bool = False
    max_retries: int = 3
    max_transient_retries: int = 20
    retry_sleep: float = 1.5
    max_retry_backoff: float = 60.0
    llm_call_timeout: float = 120.0
    max_total_llm_calls: int = 0
    max_total_tokens: int = 0


SECTION_TYPES = {
    "data": DataConfig, "models": ModelConfig, "training": TrainingConfig, "tcs": TCSConfig,
    "peer_state": PeerStateConfig, "responsibility": ResponsibilityConfig,
    "evaluation": CandidateEvaluationConfig, "constraints": ConstraintConfig,
    "persistence": PersistenceConfig,
}


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tcs: TCSConfig = field(default_factory=TCSConfig)
    peer_state: PeerStateConfig = field(default_factory=PeerStateConfig)
    responsibility: ResponsibilityConfig = field(default_factory=ResponsibilityConfig)
    evaluation: CandidateEvaluationConfig = field(default_factory=CandidateEvaluationConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    @classmethod
    def from_flat(cls, **values):
        unknown = set(values)
        sections = {}
        for name, section_type in SECTION_TYPES.items():
            field_names = set(section_type.__dataclass_fields__)
            section_values = {key: values[key] for key in list(unknown) if key in field_names}
            unknown -= set(section_values)
            sections[name] = section_type(**section_values)
        if unknown:
            raise TypeError(f"Unknown Config fields: {sorted(unknown)}")
        return cls(**sections)

    def to_flat_dict(self):
        result = {}
        for name in SECTION_TYPES:
            result.update(asdict(getattr(self, name)))
        return result

def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    defaults = Config().to_flat_dict()
    bool_fields = {key for key, value in defaults.items() if isinstance(value, bool)}
    for name, default in defaults.items():
        arg_type = int if name in bool_fields else type(default)
        kwargs = {"default": int(default) if name in bool_fields else default, "type": arg_type}
        if name in bool_fields:
            kwargs["choices"] = [0, 1]
        parser.add_argument(f"--{name}", **kwargs)
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    values = {name: getattr(args, name) for name in Config().to_flat_dict()}
    for name, default in Config().to_flat_dict().items():
        if isinstance(default, bool):
            values[name] = bool(int(values[name]))
    return Config.from_flat(**values)
