from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import Config
from ..protocol import (
    candidate_budget_contract,
    canonical_experiment_setting,
    experiment_protocol,
)
from ..provider_credentials import resolve_base_url
from ..diagnosis_aggregation import (
    ANSWER_ROLE_ENCODING_VERSION,
    DIAGNOSIS_AGGREGATION_VERSION,
    PATTERN_SELECTION_VERSION,
)
from ..tcs import (
    CRITIC_SCHEMA_VERSION,
    ROLE_RETRY_POLICY_VERSION,
    STUDENT_SCHEMA_VERSION,
    TCS_PROTOCOL_VERSION,
    TEACHER_SCHEMA_VERSION,
    TEACHER_REVISION_PROTOCOL_VERSION,
)
from ..evaluation.output_contract import (
    SOLVER_REQUEST_TEMPLATE_VERSION,
    solver_output_contract,
    solver_system_prompt,
)
SOLVER_INVALID_RETRY_POLICY_VERSION = "retry_until_first_valid_v1"
PROMPT_QUESTION_EVALUATOR_VERSION = "prompt_question_recovered_invalid_v2"
NO_TEST_FILE_IDENTITY_SHA256 = hashlib.sha256(
    b"run-identity:no-test-file:not-applicable:v1"
).hexdigest()
from ..utils import normalize_spaces
from ..versions import (
    CANDIDATE_SELECTION_VERSION,
    CANDIDATE_ACCEPTANCE_VERSION,
    CANDIDATE_PROTOCOL_FILTER_VERSION,
    CHECKPOINT_SELECTION_VERSION,
    CHECKPOINT_VERSION,
    COALITION_CONTRIBUTION_VERSION,
    EVALUATION_PROTOCOL_VERSION,
    EXPERIMENT_MATRIX_VERSION,
    EXPERIMENTAL_MODULE2_VERSION,
    COMMON_UPDATE_POLICY_VERSION,
    DUAL_TARGET_SEARCH_VERSION,
    MUTABLE_PROMPT_CONTRACT_VERSION,
    MODEL_THINKING_MODE_VERSION,
    MINIMAL_EDIT_VERSION,
    PRESERVATION_POLICY_VERSION,
    PROPOSAL_MEMORY_VERSION,
    PROTOCOL_RESOLUTION_VERSION,
    REPAIRABILITY_VERSION,
    RESPONSIBILITY_VERSION,
    RESPONSIBILITY_UTILITY_VERSION,
    RCRU_VERSION,
    ROBUST_SUPPORT_VERSION,
    SERVICE_ROUTING_VERSION,
    STUDENT_INVALID_RECOVERY_VERSION,
    STUDENT_PROMPT_CONTRACT_VERSION,
    TARGET_SELECTION_VERSION,
    TCS_CONTEXT_VERSION,
    TEST_ISOLATION_VERSION,
)
from ..compatibility_repair import ONLINE_COMPATIBILITY_REPAIR_VERSION


@dataclass(frozen=True)
class RunIdentity:
    method_version: str
    experiment_setting: str
    git_commit: str
    git_dirty: bool
    config_fingerprint: str
    manifest_sha256: str
    train_file_sha256: str
    val_file_sha256: str
    test_file_sha256: str
    train_question_set_hash: str
    val_question_set_hash: str
    test_question_set_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: str) -> str:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Run identity input does not exist: {target}")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _test_file_identity_sha256(cfg: Config) -> str:
    """Represent an explicitly disabled, zero-row test split without touching it."""
    if not cfg.persistence.final_test_enabled and cfg.data.test_size == 0:
        return NO_TEST_FILE_IDENTITY_SHA256
    return _sha256_file(cfg.data.test_path)


def question_set_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    questions = sorted(
        hashlib.sha256(normalize_spaces(str(row["question"])).lower().encode("utf-8")).hexdigest()
        for row in rows
    )
    return hashlib.sha256(json.dumps(questions, separators=(",", ":")).encode("utf-8")).hexdigest()


def _git_identity(workspace: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=workspace, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip())
    return commit, dirty


def _protocol_for_config(cfg: Config):
    canonical_name = canonical_experiment_setting(
        cfg.training.experiment_setting,
        allow_legacy_setting=cfg.training.allow_legacy_setting,
        allow_auxiliary_setting=cfg.training.allow_auxiliary_setting,
    )
    return experiment_protocol(
        cfg.training.experiment_setting,
        initialization_mode=cfg.training.initialization_mode,
        tie_policy=cfg.peer_state.vote_tie_break,
        candidate_budget_contract=candidate_budget_contract(
            canonical_name,
            candidates_per_target_branch=cfg.tcs.num_candidates_per_parent,
            stage_b_budget_per_branch=cfg.evaluation.stage_b_candidate_budget,
            stage_a_channel_top_k=cfg.evaluation.stage_a_channel_top_k,
            representative_size=cfg.evaluation.stage_a_representative_size,
            coverage_size=cfg.evaluation.stage_a_coverage_size,
            conversion_size=cfg.evaluation.stage_a_conversion_size,
            preservation_size=cfg.evaluation.stage_a_preservation_size,
        ),
        allow_legacy_setting=cfg.training.allow_legacy_setting,
        allow_auxiliary_setting=cfg.training.allow_auxiliary_setting,
    )


def config_fingerprint(cfg: Config) -> str:
    values = cfg.to_flat_dict()
    protocol = _protocol_for_config(cfg)
    values["experiment_setting"] = protocol.name
    values["canonical_setting_name"] = protocol.name
    values["module_vector"] = protocol.module_vector
    values["candidate_acceptance_policy"] = (
        protocol.candidate_acceptance_policy
    )
    values["candidate_ranking_policy"] = protocol.candidate_ranking_policy
    values["stage_a_policy"] = protocol.stage_a_policy
    values["module2_context_variant"] = protocol.module2_context_variant
    values["module2_evolution_variant"] = protocol.module2_evolution_variant
    values["compatibility_repair_enabled"] = protocol.compatibility_repair_enabled
    values["generic_revision_enabled"] = protocol.generic_revision_enabled
    values["candidate_budget_contract"] = asdict(
        protocol.candidate_budget_contract
    )
    rcru_enabled = (
        protocol.candidate_acceptance_policy
        == "responsibility_robust_contribution"
    )
    for operational in ("out_dir", "resume_from_checkpoint"):
        values.pop(operational, None)
    values["endpoint_identity"] = {
        "solver": resolve_base_url(cfg.models.solver_base_url_env)[1],
        "optimizer": resolve_base_url(cfg.models.optimizer_base_url_env)[1],
        "evaluator": resolve_base_url(cfg.models.evaluator_base_url_env)[1],
    }
    values["behavior_versions"] = {
        "member_objective": "integer_vote_min_sum_v2",
        "experiment_matrix": EXPERIMENT_MATRIX_VERSION,
        "experimental_module2": EXPERIMENTAL_MODULE2_VERSION,
        "module2_context_variant": protocol.module2_context_variant,
        "module2_evolution_variant": protocol.module2_evolution_variant,
        "compatibility_repair_enabled": protocol.compatibility_repair_enabled,
        "generic_revision_enabled": protocol.generic_revision_enabled,
        "online_compatibility_repair": (
            ONLINE_COMPATIBILITY_REPAIR_VERSION
            if protocol.compatibility_repair_enabled else "off"
        ),
        "protocol_resolution": PROTOCOL_RESOLUTION_VERSION,
        "common_update_policy": COMMON_UPDATE_POLICY_VERSION,
        "responsibility": RESPONSIBILITY_VERSION,
        "service_routing": SERVICE_ROUTING_VERSION,
        "target_selection": TARGET_SELECTION_VERSION,
        "repairability": REPAIRABILITY_VERSION,
        "dual_target_search": DUAL_TARGET_SEARCH_VERSION,
        "candidate_selection": CANDIDATE_SELECTION_VERSION,
        "stage_a": protocol.stage_a_policy,
        "stage_b": RCRU_VERSION if rcru_enabled else CANDIDATE_ACCEPTANCE_VERSION,
        "candidate_acceptance": (
            RCRU_VERSION if rcru_enabled else CANDIDATE_ACCEPTANCE_VERSION
        ),
        "preservation_policy": PRESERVATION_POLICY_VERSION,
        "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_selection": CHECKPOINT_SELECTION_VERSION,
        "test_isolation": TEST_ISOLATION_VERSION,
        "tcs_context": TCS_CONTEXT_VERSION,
        "proposal_memory": PROPOSAL_MEMORY_VERSION,
        "proposal_memory_mode": cfg.tcs.proposal_memory_mode,
        "diagnosis_aggregation": DIAGNOSIS_AGGREGATION_VERSION,
        "answer_role_encoding": ANSWER_ROLE_ENCODING_VERSION,
        "pattern_selection": PATTERN_SELECTION_VERSION,
        "tcs_protocol": TCS_PROTOCOL_VERSION,
        "teacher_schema": TEACHER_SCHEMA_VERSION,
        "teacher_revision_protocol": TEACHER_REVISION_PROTOCOL_VERSION,
        "critic_schema": CRITIC_SCHEMA_VERSION,
        "student_schema": STUDENT_SCHEMA_VERSION,
        "role_retry_policy": ROLE_RETRY_POLICY_VERSION,
        "completion_policy": "provider_default",
        "teacher_total_max_chars": cfg.tcs.teacher_total_max_chars,
        "candidate_prompt_max_chars": cfg.tcs.candidate_prompt_max_chars,
        "total_candidate_prompt_max_chars": cfg.tcs.total_candidate_prompt_max_chars,
        "student_count_policy": "reject_excess_keep_individually_valid_v1",
        "student_invalid_recovery": STUDENT_INVALID_RECOVERY_VERSION,
        "mutable_prompt_contract": MUTABLE_PROMPT_CONTRACT_VERSION,
        "student_prompt_contract": STUDENT_PROMPT_CONTRACT_VERSION,
        "candidate_protocol_filter": CANDIDATE_PROTOCOL_FILTER_VERSION,
        "model_facing_payload_version": "audit_hash_isolated_v2",
        "model_thinking_mode": MODEL_THINKING_MODE_VERSION,
        "terminal_failure_version": "role_specific_terminal_failure_v1",
        "solver_request_template": SOLVER_REQUEST_TEMPLATE_VERSION,
        "solver_invalid_retry_policy": SOLVER_INVALID_RETRY_POLICY_VERSION,
        "prompt_question_evaluator": PROMPT_QUESTION_EVALUATOR_VERSION,
        "solver_invalid_max_retries": cfg.models.solver_invalid_max_retries,
        "max_pattern_count": cfg.tcs.tcs_max_pattern_summaries,
        "max_evidence_case_count": cfg.tcs.tcs_max_evidence_cases,
        "member_aware_repair_pattern_count": 1,
        "member_aware_repair_case_count": 2,
        "member_aware_preservation_case_count": 1,
        "member_aware_context_character_cap": min(
            cfg.tcs.tcs_context_max_chars, 6000
        ),
        "checkpoint": CHECKPOINT_VERSION,
    }
    if rcru_enabled:
        values["behavior_versions"].update({
            "rcru": RCRU_VERSION,
            "candidate_acceptance_policy": protocol.candidate_acceptance_policy,
            "candidate_ranking_policy": protocol.candidate_ranking_policy,
            "responsibility_utility": RESPONSIBILITY_UTILITY_VERSION,
            "coalition_contribution": COALITION_CONTRIBUTION_VERSION,
            "robust_support": ROBUST_SUPPORT_VERSION,
            "minimal_edit": MINIMAL_EDIT_VERSION,
        })
    encoded = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def solver_request_identity(cfg: Config) -> str:
    payload = solver_request_components(cfg)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def solver_request_components(cfg: Config) -> dict[str, Any]:
    endpoint = resolve_base_url(cfg.models.solver_base_url_env)[1]
    output_contract = solver_output_contract(cfg.data.answer_format)
    request_template = solver_system_prompt("<DECISION_PROCEDURE>", cfg.data.answer_format)
    return {
        "solver_model": cfg.models.agent_model,
        "enable_thinking": False,
        "endpoint_identity": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        "max_tokens": cfg.models.solver_max_tokens,
        "output_contract_version": cfg.peer_state.solver_output_contract_version,
        "answer_format": cfg.data.answer_format,
        "output_contract_sha256": hashlib.sha256(
            output_contract.encode("utf-8")
        ).hexdigest(),
        "request_template": SOLVER_REQUEST_TEMPLATE_VERSION,
        "request_template_sha256": hashlib.sha256(
            request_template.encode("utf-8")
        ).hexdigest(),
        "invalid_retry_policy": SOLVER_INVALID_RETRY_POLICY_VERSION,
        "prompt_question_evaluator": PROMPT_QUESTION_EVALUATOR_VERSION,
        "invalid_max_retries": cfg.models.solver_invalid_max_retries,
    }


def build_run_identity(
    cfg: Config,
    *,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    workspace: str | Path = ".",
) -> RunIdentity:
    commit, dirty = _git_identity(Path(workspace).resolve())
    return RunIdentity(
        method_version=cfg.training.method_version,
        experiment_setting=_protocol_for_config(cfg).name,
        git_commit=commit,
        git_dirty=dirty,
        config_fingerprint=config_fingerprint(cfg),
        manifest_sha256=cfg.data.manifest_sha256,
        train_file_sha256=_sha256_file(cfg.data.train_path),
        val_file_sha256=_sha256_file(cfg.data.val_path),
        test_file_sha256=_test_file_identity_sha256(cfg),
        train_question_set_hash=question_set_hash(train_rows),
        val_question_set_hash=question_set_hash(val_rows),
        test_question_set_hash=question_set_hash(test_rows),
    )


def validate_run_identity(expected: RunIdentity, actual: Mapping[str, Any]) -> None:
    expected_payload = expected.to_dict()
    missing = sorted(set(expected_payload) - set(actual))
    mismatches = {
        key: {"expected": expected_payload[key], "actual": actual[key]}
        for key in expected_payload
        if key in actual and actual[key] != expected_payload[key]
    }
    if missing or mismatches:
        raise ValueError(
            "Run identity mismatch; refusing resume or completed-run reuse: "
            + json.dumps({"missing": missing, "mismatches": mismatches}, sort_keys=True)
        )
