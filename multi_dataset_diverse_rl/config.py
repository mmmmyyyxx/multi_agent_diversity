from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field

from .provider_credentials import DASHSCOPE_API_KEY_ENV, DASHSCOPE_BASE_URL_ENV


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
    agent_model: str = "gpt-4o-mini"
    optimizer_model: str = "gpt-4o-mini"
    evaluator_model: str = "gpt-4o-mini"
    solver_api_key_env: str = DASHSCOPE_API_KEY_ENV
    solver_base_url_env: str = DASHSCOPE_BASE_URL_ENV
    optimizer_api_key_env: str = DASHSCOPE_API_KEY_ENV
    optimizer_base_url_env: str = DASHSCOPE_BASE_URL_ENV
    evaluator_api_key_env: str = DASHSCOPE_API_KEY_ENV
    evaluator_base_url_env: str = DASHSCOPE_BASE_URL_ENV
    temperature: float = 0.0
    solver_max_tokens: int = 1800
    solver_invalid_max_retries: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    method_version: str = "member_aware_peer_state_v10"
    experiment_setting: str = "shared_member_aware_full"
    agents: int = 5
    epochs: int = 3
    update_every: int = 10
    seed: int = 42
    shared_prompt: str = "You are a careful reasoning solver. Use an explicit decision procedure and verify the key inference before finalizing the decision."
    initialization_mode: str = "shared_identical"
    provided_prompts_json: str = ""


@dataclass(frozen=True)
class TCSConfig:
    proposal_memory_mode: str = "off"
    teacher_critic_max_rounds: int = 2
    teacher_json_max_retries: int = 1
    critic_json_max_retries: int = 1
    teacher_temperature: float = 0.4
    critic_temperature: float = 0.0
    student_temperature: float = 0.5
    student_invalid_max_retries: int = 3
    student_upstream_regeneration_max_count: int = 1
    num_candidates_per_parent: int = 2
    tcs_max_pattern_summaries: int = 3
    tcs_max_evidence_cases: int = 3
    tcs_context_max_chars: int = 10000
    teacher_total_max_chars: int = 1800
    teacher_field_max_chars: int = 800
    critic_feedback_max_chars: int = 500
    candidate_prompt_max_chars: int = 3000
    total_candidate_prompt_max_chars: int = 5000


@dataclass(frozen=True)
class PeerStateConfig:
    aggregation_mode: str = "plurality"
    vote_tie_break: str = "abstain"
    soft_vote_tau: float = 1.0
    probe_version: str = "peer_state_fixed_probe_v1"
    parser_version: str = "task_parser_v1"
    solver_output_contract_version: str = "task_output_contract_v1"


@dataclass(frozen=True)
class ResponsibilityConfig:
    responsibility_mode: str = "single_service_member_aware_v10"
    member_uplift_tolerance: int = 5


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
class PersistenceConfig:
    out_dir: str = "experiments/runs_peer_state"
    shared_solver_cache_path: str = ""
    resume_from_checkpoint: bool = False
    max_retries: int = 3
    max_transient_retries: int = 20
    retry_sleep: float = 1.5
    max_retry_backoff: float = 60.0
    llm_call_timeout: float = 120.0
    final_test_enabled: bool = True
    frozen_initialization_manifest_path: str = ""


SECTION_TYPES = {
    "data": DataConfig, "models": ModelConfig, "training": TrainingConfig, "tcs": TCSConfig,
    "peer_state": PeerStateConfig, "responsibility": ResponsibilityConfig,
    "evaluation": CandidateEvaluationConfig,
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
