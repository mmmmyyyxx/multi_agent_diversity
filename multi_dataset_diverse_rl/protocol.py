from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum

from .module2_context import (
    C0_CURRENT_V15,
    C2_BOUNDARY_PLUS_PRESERVATION,
    C3_COALITION_AWARE_PRESERVATION,
)


class InitializationMode(str, Enum):
    SHARED_IDENTICAL = "shared_identical"
    PROVIDED_PROMPT_SET = "provided_prompt_set"


@dataclass(frozen=True)
class CandidateBudgetContract:
    target_branch_count: int
    candidates_per_target_branch: int
    total_generated_candidates_per_update: int
    stage_b_budget_per_branch: int
    total_stage_b_candidate_budget: int
    stage_a_channel_top_k: int
    representative_size: int
    coverage_size: int
    conversion_size: int
    preservation_size: int

    def __post_init__(self) -> None:
        if self.target_branch_count == 0 or self.candidates_per_target_branch == 0:
            if any((
                self.target_branch_count,
                self.candidates_per_target_branch,
                self.total_generated_candidates_per_update,
                self.stage_b_budget_per_branch,
                self.total_stage_b_candidate_budget,
            )):
                raise ValueError("disabled candidate budget must be all zero")
            return
        if self.target_branch_count < 0 or self.candidates_per_target_branch < 0:
            raise ValueError("candidate branch budgets must be positive")
        if (
            self.total_generated_candidates_per_update
            != self.target_branch_count * self.candidates_per_target_branch
        ):
            raise ValueError("total generated candidate budget mismatch")
        if self.stage_b_budget_per_branch < self.candidates_per_target_branch:
            raise ValueError("all generated branch candidates must enter Stage B")
        if (
            self.total_stage_b_candidate_budget
            != self.target_branch_count * self.stage_b_budget_per_branch
        ):
            raise ValueError("total Stage B candidate budget mismatch")

    @property
    def generated_per_update(self) -> int:
        return self.total_generated_candidates_per_update

    @property
    def stage_b_candidate_budget(self) -> int:
        return self.total_stage_b_candidate_budget


@dataclass(frozen=True)
class AblationModules:
    member_aware_dual_target_search: bool
    responsibility_conditioned_evolution: bool

    def as_tuple(self) -> tuple[bool, bool]:
        return tuple(
            bool(getattr(self, field.name)) for field in fields(type(self))
        )


@dataclass(frozen=True)
class LegacyAblationModules:
    optimization: bool
    vote_state_diagnosis: bool
    member_aware_responsibility: bool
    dual_target_competition: bool
    responsibility_conditioned_evolution: bool
    robust_contribution_update: bool

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool, bool]:
        return tuple(
            bool(getattr(self, field.name)) for field in fields(type(self))
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
    modules: AblationModules | LegacyAblationModules
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
    module2_context_variant: str = C0_CURRENT_V15
    module2_evolution_variant: str = "m20_current_v15"
    compatibility_repair_enabled: bool = False
    generic_revision_enabled: bool = False
    legacy_protocol: bool = False
    auxiliary_protocol: bool = False

    @property
    def candidate_selection_policy(self) -> str:
        return self.candidate_ranking_policy

    @property
    def target_branch_count(self) -> int:
        return self.candidate_budget_contract.target_branch_count

    @property
    def candidates_per_target_branch(self) -> int:
        return self.candidate_budget_contract.candidates_per_target_branch

    @property
    def module_vector(self) -> dict[str, bool] | None:
        if self.name == "shared_static_reference":
            return None
        return {
            field.name: bool(getattr(self.modules, field.name))
            for field in fields(type(self.modules))
        }


MAIN_ABLATION_SETTINGS = (
    "shared_static_reference",
    "shared_generic_evolution",
    "shared_member_aware_dual_target",
    "shared_responsibility_conditioned_dual_target",
)

EXPERIMENTAL_V16_MODULE2_SETTINGS = (
    "experimental_v16_c0_current_v15",
    "experimental_v16_c2_boundary_plus_preservation",
    "experimental_v16_c3_coalition_aware_preservation",
    "experimental_v16_m20_current_v15",
    "experimental_v16_m2a_residual_diagnosis",
    "experimental_v16_m2b_diagnosis_minimal_edit",
    "experimental_v16_m2c_diagnosis_minimal_edit_relevance_critic",
    "experimental_v16_m2d_raw_responsibility_minimal_edit",
    "experimental_v16_m2e_scoped_behavioral_patch",
    "experimental_v16_m2f_online_compatibility_repair",
    "experimental_v16_efficacy_g_matched",
    "experimental_v16_efficacy_r_m20",
    "experimental_v16_efficacy_r_m2f",
)

EXPERIMENTAL_V16_MODULE2_VARIANTS = {
    "experimental_v16_c0_current_v15": C0_CURRENT_V15,
    "experimental_v16_c2_boundary_plus_preservation": (
        C2_BOUNDARY_PLUS_PRESERVATION
    ),
    "experimental_v16_c3_coalition_aware_preservation": (
        C3_COALITION_AWARE_PRESERVATION
    ),
}

EXPERIMENTAL_V16_EVOLUTION_VARIANTS = {
    "experimental_v16_m20_current_v15": "m20_current_v15",
    "experimental_v16_m2a_residual_diagnosis": "m2a_residual_diagnosis",
    "experimental_v16_m2b_diagnosis_minimal_edit": "m2b_diagnosis_minimal_edit",
    "experimental_v16_m2c_diagnosis_minimal_edit_relevance_critic": (
        "m2c_diagnosis_minimal_edit_relevance_critic"
    ),
    "experimental_v16_m2d_raw_responsibility_minimal_edit": (
        "m2d_raw_responsibility_minimal_edit"
    ),
    "experimental_v16_m2e_scoped_behavioral_patch": "m2e_scoped_behavioral_patch",
}

SETTING_DISPLAY_NAMES = {
    "shared_static_reference": "Static Reference",
    "shared_generic_evolution": "S0 Generic Prompt Evolution",
    "shared_member_aware_dual_target": "S1 Member-Aware Dual-Target Search",
    "shared_responsibility_conditioned_dual_target": (
        "S2 Responsibility-Conditioned Evolution (Full)"
    ),
}

MAIN_ABLATION_MODULES = {
    "shared_static_reference": AblationModules(False, False),
    "shared_generic_evolution": AblationModules(False, False),
    "shared_member_aware_dual_target": AblationModules(True, False),
    "shared_responsibility_conditioned_dual_target": AblationModules(True, True),
}

EXPECTED_ADJACENT_MODULE = (
    ("shared_static_reference", "shared_generic_evolution", "optimization"),
    (
        "shared_generic_evolution",
        "shared_member_aware_dual_target",
        "member_aware_dual_target_search",
    ),
    (
        "shared_member_aware_dual_target",
        "shared_responsibility_conditioned_dual_target",
        "responsibility_conditioned_evolution",
    ),
)

AUXILIARY_SEARCH_CONTROL_SETTINGS = (
    "aux_dual_target_budget_matched_2x1",
    "aux_single_target_compute_matched_1x4",
)

LEGACY_V12_SETTINGS = (
    "legacy_v12_shared_baseline",
    "legacy_v12_shared_generic_evolution",
    "legacy_v12_shared_vote_state_diagnosis",
    "legacy_v12_shared_member_aware_responsibility",
    "legacy_v12_shared_responsibility_conditioned_evolution",
    "legacy_v12_shared_full_rcru",
)

LEGACY_V13_SEVEN_SETTINGS = (
    "legacy_v13_shared_baseline",
    "legacy_v13_shared_generic_evolution",
    "legacy_v13_shared_vote_state_diagnosis",
    "legacy_v13_shared_repairability_adjusted_responsibility",
    "legacy_v13_shared_dual_target_competition",
    "legacy_v13_shared_responsibility_conditioned_dual_target",
    "legacy_v13_shared_full_dual_target_rcru",
)

LEGACY_V14_SETTINGS = (
    "legacy_v14_shared_full_dual_target_rcru",
)

LEGACY_V11_SETTINGS = (
    "legacy_shared_independent_accuracy_v11",
    "legacy_shared_peer_state_vote_first_v11",
    "legacy_shared_peer_state_member_first_safe_v11",
    "legacy_shared_member_aware_full_v11",
)
LEGACY_CONTROL_SETTINGS = (
    LEGACY_V14_SETTINGS
    + LEGACY_V13_SEVEN_SETTINGS
    + LEGACY_V12_SETTINGS
    + LEGACY_V11_SETTINGS
)

HISTORICAL_SETTING_TO_LEGACY_CONTROL = {
    "shared_full_dual_target_rcru": (
        "legacy_v14_shared_full_dual_target_rcru"
    ),
    "shared_baseline": "legacy_v13_shared_baseline",
    "shared_vote_state_diagnosis": (
        "legacy_v13_shared_vote_state_diagnosis"
    ),
    "shared_repairability_adjusted_responsibility": (
        "legacy_v13_shared_repairability_adjusted_responsibility"
    ),
    "shared_dual_target_competition": (
        "legacy_v13_shared_dual_target_competition"
    ),
    "shared_member_aware_responsibility": (
        "legacy_v12_shared_member_aware_responsibility"
    ),
    "shared_responsibility_conditioned_evolution": (
        "legacy_v12_shared_responsibility_conditioned_evolution"
    ),
    "shared_full_rcru": "legacy_v12_shared_full_rcru",
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
    dependencies = (
        (
            "responsibility_conditioned_evolution",
            "member_aware_dual_target_search",
        ),
    )
    for child, parent in dependencies:
        if getattr(modules, child) and not getattr(modules, parent):
            raise ValueError(f"{child}_requires_{parent}")


def resolve_protocol_from_modules(
    modules: AblationModules,
    *,
    optimization_enabled: bool = True,
) -> ResolvedProtocolPolicies:
    validate_ablation_modules(modules)
    if not optimization_enabled:
        if any(modules.as_tuple()):
            raise ValueError("optimization_disabled_requires_all_modules_disabled")
        return ResolvedProtocolPolicies(
            False, "none", "none", "none", "none", "none", "none",
            "off", False, False,
        )
    responsibility = modules.member_aware_dual_target_search
    target = (
        "repairability_adjusted_responsibility"
        if responsibility else "round_robin"
    )
    sample_pool = (
        "member_aware_residuals"
        if responsibility
        else "individual_errors"
    )
    context = (
        "member_aware_responsibility_conditioned"
        if modules.responsibility_conditioned_evolution
        else ("generic_peer_state" if responsibility else "generic_accuracy")
    )
    return ResolvedProtocolPolicies(
        optimization_enabled=True,
        target_selection_policy=target,
        sample_pool_policy=sample_pool,
        tcs_context_policy=context,
        candidate_acceptance_policy="fixed_peer_monotone_target_or_vote",
        candidate_ranking_policy="common_monotone_safe",
        stage_a_policy="matched_all_generated",
        responsibility_refresh_policy="online" if responsibility else "off",
        repairability_freeze_enabled=False,
        service_routing_enabled=responsibility,
    )


def canonical_experiment_setting(
    name: str,
    *,
    allow_legacy_setting: bool = False,
    allow_auxiliary_setting: bool = False,
) -> str:
    requested = str(name)
    if requested in MAIN_ABLATION_SETTINGS:
        return requested
    if requested in EXPERIMENTAL_V16_MODULE2_SETTINGS:
        return requested
    if requested in AUXILIARY_SEARCH_CONTROL_SETTINGS:
        if not allow_auxiliary_setting:
            raise ValueError(
                "Auxiliary experiment setting requires "
                f"allow_auxiliary_setting=1: {requested}"
            )
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
    if left.optimization_enabled != right.optimization_enabled:
        return ("optimization",)
    return tuple(
        field.name
        for field in fields(AblationModules)
        if getattr(left.modules, field.name) != getattr(right.modules, field.name)
    )


def setting_index(name: str) -> int | None:
    return (
        MAIN_ABLATION_SETTINGS.index(name)
        if name in MAIN_ABLATION_SETTINGS else None
    )


def added_module_vs_previous(name: str) -> str | None:
    index = setting_index(name)
    if index is None or index == 0:
        return None
    changed = (
        ("optimization",)
        if index == 1
        else tuple(
            field.name
            for field in fields(AblationModules)
            if (
                getattr(
                    MAIN_ABLATION_MODULES[MAIN_ABLATION_SETTINGS[index - 1]],
                    field.name,
                )
                != getattr(MAIN_ABLATION_MODULES[name], field.name)
            )
        )
    )
    if len(changed) != 1:
        raise AssertionError("main_ablation_adjacency_is_not_single_module")
    return changed[0]


def candidate_budget_contract(
    name: str,
    *,
    candidates_per_target_branch: int,
    stage_b_budget_per_branch: int,
    stage_a_channel_top_k: int,
    representative_size: int,
    coverage_size: int,
    conversion_size: int,
    preservation_size: int,
) -> CandidateBudgetContract:
    if name == "shared_static_reference":
        branch_count, candidate_count, stage_b = 0, 0, 0
    elif name == "aux_dual_target_budget_matched_2x1":
        branch_count, candidate_count, stage_b = 2, 1, 1
    elif name == "aux_single_target_compute_matched_1x4":
        branch_count, candidate_count, stage_b = 1, 4, 4
    elif name in LEGACY_V14_SETTINGS:
        branch_count, candidate_count, stage_b = 2, 2, 2
    elif name in LEGACY_V13_SEVEN_SETTINGS:
        branch_count = (
            2 if LEGACY_V13_SEVEN_SETTINGS.index(name) >= 4 else 1
        )
        candidate_count, stage_b = 2, 2
    elif name in MAIN_ABLATION_MODULES or name in EXPERIMENTAL_V16_MODULE2_SETTINGS:
        modules = (
            MAIN_ABLATION_MODULES[name]
            if name in MAIN_ABLATION_MODULES
            else MAIN_ABLATION_MODULES[
                "shared_responsibility_conditioned_dual_target"
            ]
        )
        branch_count = 2 if modules.member_aware_dual_target_search else 1
        candidate_count, stage_b = 2, 2
    else:
        branch_count = 1
        candidate_count = int(candidates_per_target_branch)
        stage_b = max(int(stage_b_budget_per_branch), candidate_count)
    return CandidateBudgetContract(
        target_branch_count=branch_count,
        candidates_per_target_branch=candidate_count,
        total_generated_candidates_per_update=branch_count * candidate_count,
        stage_b_budget_per_branch=stage_b,
        total_stage_b_candidate_budget=branch_count * stage_b,
        stage_a_channel_top_k=int(stage_a_channel_top_k),
        representative_size=int(representative_size),
        coverage_size=int(coverage_size),
        conversion_size=int(conversion_size),
        preservation_size=int(preservation_size),
    )


def _legacy_definition(
    name: str,
) -> tuple[str, LegacyAblationModules, dict]:
    if name in LEGACY_V14_SETTINGS:
        modules = LegacyAblationModules(True, True, True, True, True, True)
        return (
            "Legacy v14 S3 Robust Contribution Update (Full)",
            modules,
            dict(
                optimization_enabled=True,
                target_selection_policy="repairability_adjusted_responsibility",
                sample_pool_policy="member_aware_residuals",
                tcs_context_policy="member_aware_responsibility_conditioned",
                candidate_acceptance_policy="responsibility_robust_contribution",
                candidate_ranking_policy="responsibility_contribution_pareto",
                stage_a_policy="matched_all_generated",
                responsibility_refresh_policy="online",
                repairability_freeze_enabled=False,
                service_routing_enabled=True,
            ),
        )
    if name in LEGACY_V13_SEVEN_SETTINGS:
        index = LEGACY_V13_SEVEN_SETTINGS.index(name)
        modules = LegacyAblationModules(
            optimization=index >= 1,
            vote_state_diagnosis=index >= 2,
            member_aware_responsibility=index >= 3,
            dual_target_competition=index >= 4,
            responsibility_conditioned_evolution=index >= 5,
            robust_contribution_update=index >= 6,
        )
        responsibility = modules.member_aware_responsibility
        return (
            f"Legacy seven-setting v13 {name.removeprefix('legacy_v13_')}",
            modules,
            dict(
                optimization_enabled=modules.optimization,
                target_selection_policy=(
                    "repairability_adjusted_responsibility"
                    if responsibility else "round_robin"
                ) if modules.optimization else "none",
                sample_pool_policy=(
                    "member_aware_residuals"
                    if responsibility else (
                        "global_peer_state"
                        if modules.vote_state_diagnosis else "individual_errors"
                    )
                ) if modules.optimization else "none",
                tcs_context_policy=(
                    "member_aware_responsibility_conditioned"
                    if modules.responsibility_conditioned_evolution
                    else (
                        "generic_peer_state"
                        if modules.vote_state_diagnosis else "generic_accuracy"
                    )
                ) if modules.optimization else "none",
                candidate_acceptance_policy=(
                    "responsibility_robust_contribution"
                    if modules.robust_contribution_update
                    else "fixed_peer_monotone_target_or_vote"
                ) if modules.optimization else "none",
                candidate_ranking_policy=(
                    "responsibility_contribution_pareto"
                    if modules.robust_contribution_update
                    else "common_monotone_safe"
                ) if modules.optimization else "none",
                stage_a_policy=(
                    "matched_all_generated" if modules.optimization else "none"
                ),
                responsibility_refresh_policy=(
                    "online" if responsibility else "off"
                ),
                repairability_freeze_enabled=False,
                service_routing_enabled=responsibility,
            ),
        )
    is_v12 = name in LEGACY_V12_SETTINGS
    if is_v12:
        index = LEGACY_V12_SETTINGS.index(name)
        modules = (
            LegacyAblationModules(False, False, False, False, False, False)
            if index == 0
            else LegacyAblationModules(
                True,
                index >= 2,
                index >= 3,
                False,
                index >= 4,
                index >= 5,
            )
        )
        responsibility = modules.member_aware_responsibility
        payload = dict(
            optimization_enabled=modules.optimization,
            target_selection_policy="round_robin",
            sample_pool_policy=(
                "member_aware_residuals" if responsibility else (
                    "global_peer_state"
                    if modules.vote_state_diagnosis else "individual_errors"
                )
            ),
            tcs_context_policy=(
                "member_aware_responsibility_conditioned"
                if modules.responsibility_conditioned_evolution
                else (
                    "generic_peer_state"
                    if modules.vote_state_diagnosis else "generic_accuracy"
                )
            ),
            candidate_acceptance_policy=(
                "responsibility_robust_contribution"
                if modules.robust_contribution_update
                else "fixed_peer_monotone_target_or_vote"
            ),
            candidate_ranking_policy=(
                "responsibility_contribution_pareto"
                if modules.robust_contribution_update
                else "common_monotone_safe"
            ),
            stage_a_policy="matched_all_generated",
            responsibility_refresh_policy=(
                "online" if responsibility else "off"
            ),
            repairability_freeze_enabled=False,
            service_routing_enabled=responsibility,
        )
        if index >= 3:
            payload.update({
                "target_selection_policy": "member_aware_responsibility",
                "repairability_freeze_enabled": True,
            })
        return f"Legacy v12 {name.removeprefix('legacy_v12_')}", modules, payload

    common = LegacyAblationModules(True, True, False, False, False, False)
    definitions = {
        "legacy_shared_independent_accuracy_v11": (
            "Legacy v11 Independent Accuracy Control",
            LegacyAblationModules(True, False, False, False, False, False),
            ("round_robin", "individual_errors", "generic_accuracy",
             "individual_first_safe", "legacy_individual_first"),
        ),
        "legacy_shared_peer_state_vote_first_v11": (
            "Legacy v11 Peer-State Vote-First Control",
            common,
            ("round_robin", "global_peer_state", "generic_peer_state",
             "vote_first_safe", "legacy_vote_first"),
        ),
        "legacy_shared_peer_state_member_first_safe_v11": (
            "Legacy v11 Peer-State Member-First Control",
            common,
            ("round_robin", "global_peer_state", "generic_peer_state",
             "member_first_safe", "legacy_member_multichannel"),
        ),
        "legacy_shared_member_aware_full_v11": (
            "Legacy v11 Member-Aware Full Control",
            LegacyAblationModules(True, True, True, False, True, False),
            ("member_aware_responsibility", "member_aware_residuals",
             "member_aware_responsibility_conditioned",
             "member_first_safe", "legacy_member_multichannel"),
        ),
    }
    display, modules, values = definitions[name]
    target, pool, context, ranking, stage_a = values
    return display, modules, dict(
        optimization_enabled=True,
        target_selection_policy=target,
        sample_pool_policy=pool,
        tcs_context_policy=context,
        candidate_acceptance_policy="legacy_monotone_safe",
        candidate_ranking_policy=ranking,
        stage_a_policy=stage_a,
        responsibility_refresh_policy=(
            "online" if modules.member_aware_responsibility else "off"
        ),
        repairability_freeze_enabled=modules.member_aware_responsibility,
        service_routing_enabled=modules.member_aware_responsibility,
    )


def experiment_protocol(
    name: str,
    *,
    initialization_mode: str,
    tie_policy: str,
    candidate_budget_contract: CandidateBudgetContract,
    allow_legacy_setting: bool = False,
    allow_auxiliary_setting: bool = False,
) -> ExperimentProtocol:
    requested_name = str(name)
    canonical_name = canonical_experiment_setting(
        requested_name,
        allow_legacy_setting=allow_legacy_setting,
        allow_auxiliary_setting=allow_auxiliary_setting,
    )
    if canonical_name in MAIN_ABLATION_SETTINGS:
        modules = MAIN_ABLATION_MODULES[canonical_name]
        resolved = resolve_protocol_from_modules(
            modules,
            optimization_enabled=canonical_name != "shared_static_reference",
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
            auxiliary_protocol=False,
            **resolved.__dict__,
        )
    if canonical_name in EXPERIMENTAL_V16_MODULE2_SETTINGS:
        if canonical_name == "experimental_v16_efficacy_g_matched":
            modules = AblationModules(True, False)
        else:
            modules = MAIN_ABLATION_MODULES[
                "shared_responsibility_conditioned_dual_target"
            ]
        resolved = resolve_protocol_from_modules(modules)
        return ExperimentProtocol(
            name=canonical_name,
            requested_name=requested_name,
            display_name=canonical_name,
            modules=modules,
            initialization_mode=InitializationMode(initialization_mode),
            tie_policy=str(tie_policy),
            candidate_budget_contract=candidate_budget_contract,
            module2_context_variant=(
                EXPERIMENTAL_V16_MODULE2_VARIANTS.get(
                    canonical_name, C0_CURRENT_V15
                )
            ),
            module2_evolution_variant=(
                EXPERIMENTAL_V16_EVOLUTION_VARIANTS.get(
                    canonical_name, "m20_current_v15"
                )
            ),
            compatibility_repair_enabled=(
                canonical_name in {
                    "experimental_v16_m2f_online_compatibility_repair",
                    "experimental_v16_efficacy_r_m2f",
                }
            ),
            generic_revision_enabled=(
                canonical_name == "experimental_v16_efficacy_g_matched"
            ),
            legacy_protocol=False,
            auxiliary_protocol=False,
            **resolved.__dict__,
        )
    if canonical_name in AUXILIARY_SEARCH_CONTROL_SETTINGS:
        modules = MAIN_ABLATION_MODULES["shared_member_aware_dual_target"]
        resolved = resolve_protocol_from_modules(modules)
        return ExperimentProtocol(
            name=canonical_name,
            requested_name=requested_name,
            display_name=canonical_name,
            modules=modules,
            initialization_mode=InitializationMode(initialization_mode),
            tie_policy=str(tie_policy),
            candidate_budget_contract=candidate_budget_contract,
            auxiliary_protocol=True,
            **resolved.__dict__,
        )
    if canonical_name in LEGACY_CONTROL_SETTINGS:
        display, modules, resolved = _legacy_definition(canonical_name)
        return ExperimentProtocol(
            name=canonical_name,
            requested_name=requested_name,
            display_name=display,
            modules=modules,
            initialization_mode=InitializationMode(initialization_mode),
            tie_policy=str(tie_policy),
            candidate_budget_contract=candidate_budget_contract,
            legacy_protocol=True,
            **resolved,
        )
    raise ValueError(f"Unknown experiment protocol: {requested_name}")
