"""Backend and benchmark neutral values for one team-search loop."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
import json
from typing import Any, Mapping

from .. import versions


class SearchContractError(ValueError):
    """A scientific or access contract cannot be satisfied."""


@dataclass(frozen=True)
class BenchmarkCapabilities:
    supports_plurality: bool
    supports_current_responsibility: bool
    supports_boolean_member_success: bool
    supports_vote_classes: bool | None = None
    supports_plurality_margin: bool | None = None

    def __post_init__(self) -> None:
        # Preserve the three-field historical BBH constructor and identity.
        for name in ("supports_vote_classes", "supports_plurality_margin"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, self.supports_plurality)

    @property
    def binary_plurality_responsibility(self) -> bool:
        return bool(self.supports_current_responsibility and self.supports_vote_classes
                    and self.supports_boolean_member_success and self.supports_plurality_margin)


@dataclass(frozen=True)
class ParsedOutput:
    answer: str
    valid: bool


@dataclass(frozen=True)
class TeamStateSnapshot:
    team_state_id: str
    member_prompts: tuple[str, ...]
    member_outputs: tuple[tuple[ParsedOutput, ...], ...] = ()
    aggregated_outputs: tuple[ParsedOutput, ...] = ()
    member_scores: tuple[float, ...] = ()
    team_scores: Mapping[str, float] = field(default_factory=dict)
    residuals: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.team_state_id or not self.member_prompts:
            raise SearchContractError("team state identity and prompts are required")
        if self.member_outputs and len(self.member_outputs) != len(self.member_prompts):
            raise SearchContractError("member output shape does not match team")


@dataclass(frozen=True)
class Diagnosis:
    responsibility: Mapping[int, Any] = field(default_factory=dict)
    patterns: Mapping[str, Any] = field(default_factory=dict)
    transition_signals: Mapping[str, Any] = field(default_factory=dict)
    benchmark_signals: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceItem:
    example_id: str
    source_split: str
    roles: frozenset[str]
    signals: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id or not self.source_split or not self.roles:
            raise SearchContractError("evidence requires identity, split and roles")


@dataclass(frozen=True)
class EvidenceView:
    mutation_evidence: tuple[EvidenceItem, ...]
    search_validation_evidence: tuple[EvidenceItem, ...]
    team_probe_evidence: tuple[EvidenceItem, ...]
    full_evaluation_scope: str
    adaptive_gate_scope: str | None

    def __post_init__(self) -> None:
        if not self.full_evaluation_scope:
            raise SearchContractError("full evaluation scope is required")
        for role, rows in (
            ("mutation", self.mutation_evidence),
            ("search validation", self.search_validation_evidence),
            ("team probe", self.team_probe_evidence),
        ):
            if any(row.source_split != "optimize" for row in rows):
                raise SearchContractError(f"{role} evidence must come from Optimize")


@dataclass(frozen=True)
class OptimizationOpportunity:
    opportunity_id: str
    parent_state_id: str
    target_member: int
    parent_prompt: str
    objective: Mapping[str, Any]
    diagnosis: Diagnosis
    evidence: EvidenceView
    pattern_context: Mapping[str, Any] = field(default_factory=dict)
    memory_view: Mapping[str, Any] = field(default_factory=dict)
    search_budget: Mapping[str, int] = field(default_factory=dict)
    evaluation_plan: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.opportunity_id or not self.parent_state_id or not self.parent_prompt:
            raise SearchContractError("opportunity identity and parent are required")
        if self.target_member < 0:
            raise SearchContractError("negative target member")


@dataclass(frozen=True)
class SearchCandidate:
    candidate_id: str
    prompt: str
    search_score: float | None = None
    lineage: Mapping[str, Any] = field(default_factory=dict)
    backend_details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[SearchCandidate, ...]
    stop_reason: str
    search_state: Mapping[str, Any] = field(default_factory=dict)
    solver_calls: int = 0
    search_meta_calls: int = 0
    solver_tokens: int = 0
    search_meta_tokens: int = 0
    proposal_count: int = 0
    team_candidate_count: int = 0
    local_survival_update_count: int = 0
    strict_accepted_count: int = 0
    strict_rejected_exported_count: int = 0


@dataclass(frozen=True)
class TeamEvaluation:
    aggregate_score: float
    aggregate_success: bool | None
    member_scores: tuple[float, ...]
    oracle_score: float | None = None
    aggregation_diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatedCandidate:
    candidate: SearchCandidate
    team_probe: TeamEvaluation | None
    full: TeamEvaluation | None
    promoted: bool
    admissible: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionDecision:
    candidate: EvaluatedCandidate | None
    reason: str


@dataclass(frozen=True)
class TransitionRecord:
    opportunity_id: str
    parent_state_id: str
    child_state_id: str
    target_member: int
    candidate_id: str
    newly_fixed_ids: tuple[str, ...] = ()
    newly_broken_ids: tuple[str, ...] = ()
    parent_correctness: tuple[tuple[str, bool], ...] = ()
    child_correctness: tuple[tuple[str, bool], ...] = ()
    parent_prompt_hash: str = ""
    child_prompt_hash: str = ""


@dataclass(frozen=True)
class TeamCostAccounting:
    solver_calls: int = 0
    search_meta_calls: int = 0
    aggregation_calls: int = 0
    solver_tokens: int = 0
    search_meta_tokens: int = 0
    aggregation_tokens: int = 0


@dataclass(frozen=True)
class SearchStopConfig:
    identity: str = versions.UNIFIED_SEARCH_STOP_VERSION
    no_update_patience: int = 3

    def __post_init__(self) -> None:
        if self.no_update_patience <= 0:
            raise SearchContractError("search patience must be positive")


@dataclass(frozen=True)
class GlobalStopConfig:
    identity: str = versions.UNIFIED_GLOBAL_STOP_VERSION
    no_commit_patience: int = 2
    emergency_max_provider_calls: int = 100_000

    def __post_init__(self) -> None:
        if self.no_commit_patience <= 0 or self.emergency_max_provider_calls <= 0:
            raise SearchContractError("global stopping values must be positive")


@dataclass(frozen=True)
class SearchMethodConfig:
    method: str = versions.UNIFIED_TEAM_PROMPT_SEARCH_VERSION
    search_engine: str = versions.UNIFIED_GEPA_DERIVED_ENGINE_VERSION
    diagnosis_policy: str = versions.UNIFIED_PLURALITY_RESPONSIBILITY_VERSION
    target_policy: str = versions.UNIFIED_TARGET_POLICY_VERSION
    feasibility_policy: str = versions.UNIFIED_FEASIBILITY_POLICY_VERSION
    evidence_policy: str = versions.UNIFIED_EVIDENCE_POLICY_VERSION
    evaluation_policy: str = versions.UNIFIED_EVALUATION_POLICY_VERSION
    transition_policy: str = versions.UNIFIED_TRANSITION_POLICY_VERSION
    adaptive_gate_policy: str = versions.UNIFIED_ADAPTIVE_GATE_VERSION
    aggregation_policy: str = versions.UNIFIED_PLURALITY_AGGREGATION_VERSION
    memory_policy: str = versions.UNIFIED_NULL_MEMORY_VERSION
    pattern_policy: str = versions.UNIFIED_NULL_PATTERN_VERSION
    search_acceptance_policy: str = versions.UNIFIED_GEPA_ACCEPTANCE_VERSION
    search_stop: SearchStopConfig = field(default_factory=SearchStopConfig)
    global_stop: GlobalStopConfig = field(default_factory=GlobalStopConfig)

    @classmethod
    def v2(cls, **overrides: Any) -> "SearchMethodConfig":
        """V2 defaults; the zero-argument constructor retains the V1 replay identity."""
        values = dict(method=versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_VERSION,
                      search_engine=versions.UNIFIED_GEPA_EXPOSURE_V2_VERSION,
                      search_acceptance_policy=versions.UNIFIED_DECOUPLED_ACCEPTANCE_VERSION,
                      evidence_policy=versions.UNIFIED_VARIABLE_EVIDENCE_VERSION,
                      feasibility_policy=versions.UNIFIED_VARIABLE_FEASIBILITY_VERSION)
        values.update(overrides)
        return cls(**values)

    mechanism_config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def v2_1(cls, **overrides: Any) -> "SearchMethodConfig":
        """Frozen historical V2.1 contract; mechanisms remain explicit opt-ins."""
        values = dict(method=versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION,
                      evidence_policy=versions.UNIFIED_FOCUSED_EVIDENCE_VERSION,
                      transition_policy=versions.UNIFIED_COMPETENCE_TRANSITION_VERSION)
        values.update(overrides)
        return cls.v2(**values)

    @classmethod
    def v2_2(cls, **overrides: Any) -> "SearchMethodConfig":
        """V2.2 OR progress identity; current execution still requires the full bundle."""
        values = dict(method=versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_2_VERSION,
                      transition_policy=versions.UNIFIED_TARGET_OR_TEAM_TRANSITION_VERSION)
        values.update(overrides)
        return cls.v2_1(**values)

    def identity(self) -> str:
        from dataclasses import asdict
        payload = asdict(self)
        if self.method == versions.UNIFIED_TEAM_PROMPT_SEARCH_VERSION and not self.mechanism_config:
            payload.pop("mechanism_config")  # Exact historical V1 hash payload.
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SearchMethodConfig":
        allowed = {row.name for row in fields(cls)}
        if set(payload) - allowed:
            raise SearchContractError("unknown unified method component")
        if payload.get("method", cls.method) not in {versions.UNIFIED_TEAM_PROMPT_SEARCH_VERSION, versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_VERSION, versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION, versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_2_VERSION, versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_3_VERSION}:
            raise SearchContractError("unsupported unified method identity")
        from dataclasses import asdict
        factory = {versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_VERSION: cls.v2,
                   versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_1_VERSION: cls.v2_1,
                   versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_2_VERSION: cls.v2_2}.get(payload.get("method"))
        values = asdict(factory()) if factory else {}
        values.update(payload)
        for key, kind in (("search_stop", SearchStopConfig),
                          ("global_stop", GlobalStopConfig)):
            if key in values:
                raw = values[key]
                if not isinstance(raw, Mapping) or set(raw) - {row.name for row in fields(kind)}:
                    raise SearchContractError(f"invalid {key} config")
                values[key] = kind(**raw)
        return cls(**values)
