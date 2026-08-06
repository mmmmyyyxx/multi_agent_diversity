from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum


class InitializationMode(str, Enum):
    SHARED_IDENTICAL = "shared_identical"
    PROVIDED_PROMPT_SET = "provided_prompt_set"


@dataclass(frozen=True)
class CandidateBudgetContract:
    generated_per_update: int
    stage_a_channel_top_k: int
    stage_b_candidate_budget: int
    representative_size: int
    coverage_size: int
    conversion_size: int
    preservation_size: int


@dataclass(frozen=True)
class AblationModules:
    optimization: bool
    vote_state_diagnosis: bool
    member_aware_responsibility: bool
    responsibility_conditioned_evolution: bool
    robust_contribution_update: bool

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        return (
            self.optimization,
            self.vote_state_diagnosis,
            self.member_aware_responsibility,
            self.responsibility_conditioned_evolution,
            self.robust_contribution_update,
        )


@dataclass(frozen=True)
class ResolvedProtocolPolicies:
    optimization_enabled: bool
    target_selection_policy: str
    sample_pool_policy: str
    tcs_context_policy: str
    candidate_acceptance_policy: str
    candidate_ranking_policy: str
    stage_a_policy: str
    responsibility_refresh_policy: str
    repairability_freeze_enabled: bool
    service_routing_enabled: bool


@dataclass(frozen=True)
class ExperimentProtocol:
    name: str
    requested_name: str
    display_name: str
    modules: AblationModules
    optimization_enabled: bool
    target_selection_policy: str
    sample_pool_policy: str
    tcs_context_policy: str
    candidate_acceptance_policy: str
    candidate_ranking_policy: str
    stage_a_policy: str
    responsibility_refresh_policy: str
    repairability_freeze_enabled: bool
    service_routing_enabled: bool
    initialization_mode: InitializationMode
    tie_policy: str
    candidate_budget_contract: CandidateBudgetContract
    legacy_protocol: bool = False

    @property
    def candidate_selection_policy(self) -> str:
        """Compatibility view for old artifact readers.

        Active runtime decisions use the separate acceptance, ranking, and
        Stage-A policy fields.
        """

        return self.candidate_ranking_policy


MAIN_ABLATION_SETTINGS = (
    "shared_baseline",
    "shared_generic_evolution",
    "shared_vote_state_diagnosis",
    "shared_member_aware_responsibility",
    "shared_responsibility_conditioned_evolution",
    "shared_full_rcru",
)

SETTING_DISPLAY_NAMES = {
    "shared_baseline": "S0 Static Prompt Team",
    "shared_generic_evolution": "S1 Generic Prompt Evolution",
    "shared_vote_state_diagnosis": "S2 Vote-State Diagnosis",
    "shared_member_aware_responsibility": "S3 Member-Aware Responsibility",
    "shared_responsibility_conditioned_evolution": (
        "S4 Responsibility-Conditioned Evolution"
    ),
    "shared_full_rcru": "S5 Robust Contribution Update (Full)",
}

MAIN_ABLATION_MODULES = {
    "shared_baseline": AblationModules(False, False, False, False, False),
    "shared_generic_evolution": AblationModules(True, False, False, False, False),
    "shared_vote_state_diagnosis": AblationModules(
        True, True, False, False, False
    ),
    "shared_member_aware_responsibility": AblationModules(
        True, True, True, False, False
    ),
    "shared_responsibility_conditioned_evolution": AblationModules(
        True, True, True, True, False
    ),
    "shared_full_rcru": AblationModules(True, True, True, True, True),
}

EXPECTED_ADJACENT_MODULE = (
    ("shared_baseline", "shared_generic_evolution", "optimization"),
    (
        "shared_generic_evolution",
        "shared_vote_state_diagnosis",
        "vote_state_diagnosis",
    ),
    (
        "shared_vote_state_diagnosis",
        "shared_member_aware_responsibility",
        "member_aware_responsibility",
    ),
    (
        "shared_member_aware_responsibility",
        "shared_responsibility_conditioned_evolution",
        "responsibility_conditioned_evolution",
    ),
    (
        "shared_responsibility_conditioned_evolution",
        "shared_full_rcru",
        "robust_contribution_update",
    ),
)

LEGACY_CONTROL_SETTINGS = (
    "legacy_shared_independent_accuracy_v11",
    "legacy_shared_peer_state_vote_first_v11",
    "legacy_shared_peer_state_member_first_safe_v11",
    "legacy_shared_member_aware_full_v11",
)

HISTORICAL_SETTING_TO_LEGACY_CONTROL = {
    "shared_independent_accuracy": "legacy_shared_independent_accuracy_v11",
    "shared_peer_state_vote_first": "legacy_shared_peer_state_vote_first_v11",
    "shared_peer_state_member_first_safe": (
        "legacy_shared_peer_state_member_first_safe_v11"
    ),
    "shared_peer_state_member_pareto": (
        "legacy_shared_peer_state_member_first_safe_v11"
    ),
    "shared_member_aware_full": "legacy_shared_member_aware_full_v11",
}

CANDIDATE_SELECTION_POLICY_ALIASES = {
    "individual_accuracy": "individual_first_safe",
    "vote_first": "vote_first_safe",
    "member_aware_pareto": "member_first_safe",
}


def validate_ablation_modules(modules: AblationModules) -> None:
    if not modules.optimization and any(modules.as_tuple()[1:]):
        raise ValueError("optimization_disabled_requires_all_modules_disabled")
    if modules.member_aware_responsibility and not modules.vote_state_diagnosis:
        raise ValueError("responsibility_requires_vote_state_diagnosis")
    if (
        modules.responsibility_conditioned_evolution
        and not modules.member_aware_responsibility
    ):
        raise ValueError("conditioned_evolution_requires_responsibility")
    if (
        modules.robust_contribution_update
        and not modules.responsibility_conditioned_evolution
    ):
        raise ValueError("rcru_requires_conditioned_evolution")


def resolve_protocol_from_modules(
    modules: AblationModules,
) -> ResolvedProtocolPolicies:
    validate_ablation_modules(modules)
    if not modules.optimization:
        return ResolvedProtocolPolicies(
            optimization_enabled=False,
            target_selection_policy="none",
            sample_pool_policy="none",
            tcs_context_policy="none",
            candidate_acceptance_policy="none",
            candidate_ranking_policy="none",
            stage_a_policy="none",
            responsibility_refresh_policy="off",
            repairability_freeze_enabled=False,
            service_routing_enabled=False,
        )
    responsibility = modules.member_aware_responsibility
    if responsibility:
        target = "member_aware_responsibility"
        sample_pool = "member_aware_residuals"
        refresh = "online"
    else:
        target = "round_robin"
        sample_pool = (
            "global_peer_state"
            if modules.vote_state_diagnosis
            else "individual_errors"
        )
        refresh = "off"
    if modules.responsibility_conditioned_evolution:
        context = "member_aware_responsibility_conditioned"
    elif modules.vote_state_diagnosis:
        context = "generic_peer_state"
    else:
        context = "generic_accuracy"
    if modules.robust_contribution_update:
        acceptance = "responsibility_robust_contribution"
        ranking = "responsibility_contribution_pareto"
    else:
        acceptance = "fixed_peer_monotone_target_or_vote"
        ranking = "common_monotone_safe"
    return ResolvedProtocolPolicies(
        optimization_enabled=True,
        target_selection_policy=target,
        sample_pool_policy=sample_pool,
        tcs_context_policy=context,
        candidate_acceptance_policy=acceptance,
        candidate_ranking_policy=ranking,
        stage_a_policy="matched_all_generated",
        responsibility_refresh_policy=refresh,
        repairability_freeze_enabled=responsibility,
        service_routing_enabled=responsibility,
    )


def canonical_experiment_setting(
    name: str,
    *,
    allow_legacy_setting: bool = False,
) -> str:
    requested = str(name)
    if requested in MAIN_ABLATION_SETTINGS:
        return requested
    legacy = HISTORICAL_SETTING_TO_LEGACY_CONTROL.get(requested, requested)
    if legacy in LEGACY_CONTROL_SETTINGS:
        if not allow_legacy_setting:
            raise ValueError(
                f"Legacy experiment setting requires allow_legacy_setting=1: {requested}"
            )
        return legacy
    return requested


def canonical_candidate_selection_policy(name: str) -> str:
    return CANDIDATE_SELECTION_POLICY_ALIASES.get(str(name), str(name))


def changed_ablation_modules(
    left: ExperimentProtocol,
    right: ExperimentProtocol,
) -> tuple[str, ...]:
    return tuple(
        field.name
        for field in fields(AblationModules)
        if getattr(left.modules, field.name) != getattr(right.modules, field.name)
    )


def setting_index(name: str) -> int | None:
    return MAIN_ABLATION_SETTINGS.index(name) if name in MAIN_ABLATION_SETTINGS else None


def added_module_vs_previous(name: str) -> str | None:
    index = setting_index(name)
    if index is None or index == 0:
        return None
    left = MAIN_ABLATION_MODULES[MAIN_ABLATION_SETTINGS[index - 1]]
    right = MAIN_ABLATION_MODULES[name]
    changed = tuple(
        field.name
        for field in fields(AblationModules)
        if getattr(left, field.name) != getattr(right, field.name)
    )
    if len(changed) != 1:
        raise AssertionError("main_ablation_adjacency_is_not_single_module")
    return changed[0]


def _legacy_definition(name: str) -> tuple[str, AblationModules, dict]:
    common_modules = AblationModules(True, True, False, False, False)
    definitions = {
        "legacy_shared_independent_accuracy_v11": (
            "Legacy v11 Independent Accuracy Control",
            AblationModules(True, False, False, False, False),
            dict(
                optimization_enabled=True,
                target_selection_policy="round_robin",
                sample_pool_policy="individual_errors",
                tcs_context_policy="generic_accuracy",
                candidate_acceptance_policy="legacy_monotone_safe",
                candidate_ranking_policy="individual_first_safe",
                stage_a_policy="legacy_individual_first",
                responsibility_refresh_policy="off",
                repairability_freeze_enabled=False,
                service_routing_enabled=False,
            ),
        ),
        "legacy_shared_peer_state_vote_first_v11": (
            "Legacy v11 Peer-State Vote-First Control",
            common_modules,
            dict(
                optimization_enabled=True,
                target_selection_policy="round_robin",
                sample_pool_policy="global_peer_state",
                tcs_context_policy="generic_peer_state",
                candidate_acceptance_policy="legacy_monotone_safe",
                candidate_ranking_policy="vote_first_safe",
                stage_a_policy="legacy_vote_first",
                responsibility_refresh_policy="off",
                repairability_freeze_enabled=False,
                service_routing_enabled=False,
            ),
        ),
        "legacy_shared_peer_state_member_first_safe_v11": (
            "Legacy v11 Peer-State Member-First Control",
            common_modules,
            dict(
                optimization_enabled=True,
                target_selection_policy="round_robin",
                sample_pool_policy="global_peer_state",
                tcs_context_policy="generic_peer_state",
                candidate_acceptance_policy="legacy_monotone_safe",
                candidate_ranking_policy="member_first_safe",
                stage_a_policy="legacy_member_multichannel",
                responsibility_refresh_policy="off",
                repairability_freeze_enabled=False,
                service_routing_enabled=False,
            ),
        ),
        "legacy_shared_member_aware_full_v11": (
            "Legacy v11 Member-Aware Full Control",
            AblationModules(True, True, True, True, False),
            dict(
                optimization_enabled=True,
                target_selection_policy="member_aware_responsibility",
                sample_pool_policy="member_aware_residuals",
                tcs_context_policy="member_aware_responsibility_conditioned",
                candidate_acceptance_policy="legacy_monotone_safe",
                candidate_ranking_policy="member_first_safe",
                stage_a_policy="legacy_member_multichannel",
                responsibility_refresh_policy="online",
                repairability_freeze_enabled=True,
                service_routing_enabled=True,
            ),
        ),
    }
    return definitions[name]


def experiment_protocol(
    name: str,
    *,
    initialization_mode: str,
    tie_policy: str,
    candidate_budget_contract: CandidateBudgetContract,
    allow_legacy_setting: bool = False,
) -> ExperimentProtocol:
    requested_name = str(name)
    canonical_name = canonical_experiment_setting(
        requested_name,
        allow_legacy_setting=allow_legacy_setting,
    )
    if canonical_name in MAIN_ABLATION_SETTINGS:
        modules = MAIN_ABLATION_MODULES[canonical_name]
        resolved = resolve_protocol_from_modules(modules)
        if (
            resolved.optimization_enabled
            and candidate_budget_contract.stage_b_candidate_budget
            < candidate_budget_contract.generated_per_update
        ):
            raise ValueError(
                "matched_all_generated requires stage_b_candidate_budget "
                ">= generated_per_update"
            )
        return ExperimentProtocol(
            name=canonical_name,
            requested_name=requested_name,
            display_name=SETTING_DISPLAY_NAMES[canonical_name],
            modules=modules,
            initialization_mode=InitializationMode(initialization_mode),
            tie_policy=str(tie_policy),
            candidate_budget_contract=candidate_budget_contract,
            legacy_protocol=False,
            **resolved.__dict__,
        )
    if canonical_name in LEGACY_CONTROL_SETTINGS:
        display_name, modules, resolved = _legacy_definition(canonical_name)
        return ExperimentProtocol(
            name=canonical_name,
            requested_name=requested_name,
            display_name=display_name,
            modules=modules,
            initialization_mode=InitializationMode(initialization_mode),
            tie_policy=str(tie_policy),
            candidate_budget_contract=candidate_budget_contract,
            legacy_protocol=True,
            **resolved,
        )
    raise ValueError(f"Unknown experiment protocol: {requested_name}")
