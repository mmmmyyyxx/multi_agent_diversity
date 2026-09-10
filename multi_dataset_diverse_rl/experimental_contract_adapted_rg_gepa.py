"""Pure primitives for the contract-adapted RG-GEPA fixed-parent pilot.

The reflection model selects a hypothesis from a closed vocabulary.  No model
text is copied into the mutable Solver prompt: a deterministic renderer maps
the selected symbols to program-owned clauses.  This makes immutable-output-
interface contamination unrepresentable in a rendered candidate.

This module is experimental and is not imported by the canonical runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Mapping

from .evaluation.mutable_prompt_contract import validate_mutable_decision_procedure
from .versions import COMMON_SOLVER_CONTRACT_V1_ID


CONTRACT_ADAPTED_PROTOCOL_VERSION = "contract_adapted_rg_gepa_fixed_parent_v1"
CONTRACT_ADAPTED_RENDERER_VERSION = "closed_vocabulary_renderer_v1"
HYPOTHESIS_INTERFACE_V2_VERSION = "rg_gepa_hypothesis_interface_v2"

FAILURE_PATTERNS: Mapping[str, str] = {
    "ambiguous_referent": "Resolve every ambiguous referent before comparing the choices.",
    "perspective_confusion": "Track each speaker and viewpoint explicitly before drawing a conclusion.",
    "negation_scope": "Normalize negation and scope before testing the choices.",
    "constraint_omission": "List all stated constraints and keep each one active during elimination.",
    "premature_choice": "Delay commitment until every plausible choice has been checked.",
    "evidence_conflict": "Reconcile conflicting clues by testing which interpretation satisfies all constraints.",
}

BEHAVIORAL_CHANGES: Mapping[str, str] = {
    "build_constraint_table": "Build a compact entity-to-constraint table before eliminating choices.",
    "compare_all_options": "Evaluate every choice against the same explicit set of constraints.",
    "trace_entities": "Trace entity references step by step and reject inconsistent assignments.",
    "resolve_negation_first": "Rewrite negative conditions into clear positive and excluded cases first.",
    "test_counterfactuals": "Test each plausible interpretation counterfactually against all evidence.",
    "cross_check_conclusion": "Cross-check the tentative conclusion against every stated relationship.",
}

PRESERVATION_PRIORITIES: Mapping[str, str] = {
    "explicit_evidence": "Preserve conclusions that follow directly from explicit evidence.",
    "valid_eliminations": "Preserve reliable option eliminations supported by stated constraints.",
    "ambiguity_checks": "Preserve explicit ambiguity checks before choosing an interpretation.",
    "global_consistency": "Preserve consistency across all entities, speakers, and constraints.",
}

AVOIDANCE_PRIORITIES: Mapping[str, str] = {
    "unsupported_assumptions": "Avoid adding assumptions not licensed by the question.",
    "surface_overlap": "Avoid choosing from superficial word overlap alone.",
    "single_clue_fixation": "Avoid relying on one clue while ignoring conflicting evidence.",
    "premature_commitment": "Avoid committing before alternatives have been compared symmetrically.",
}

HYPOTHESIS_FIELDS = (
    "failure_pattern",
    "behavioral_change",
    "preserve",
    "avoid",
)


@dataclass(frozen=True)
class ContractAdaptedProtocol:
    candidates_per_case: int = 2
    case_count: int = 6
    minibatch_size: int = 12
    max_full_eval_candidates_per_case: int = 2
    proposal_engine: str = "contract_adapted_reflection"
    renderer_version: str = CONTRACT_ADAPTED_RENDERER_VERSION
    evaluation_mode: str = "progressive"
    selection_arms: tuple[str, str] = ("B0_PRIME_CURRENT", "B1_PRIME_TEAM_PARETO")
    solver_contract_id: str = COMMON_SOLVER_CONTRACT_V1_ID
    commit_enabled: bool = False
    validation_enabled: bool = False
    test_enabled: bool = False
    memory_enabled: bool = False

    def __post_init__(self) -> None:
        if self.candidates_per_case != 2 or self.case_count != 6:
            raise ValueError("contract-adapted v1 freezes six cases and two candidates per case")
        if self.minibatch_size != 12 or self.max_full_eval_candidates_per_case != 2:
            raise ValueError("contract-adapted v1 freezes progressive evaluation budgets")
        if self.proposal_engine != "contract_adapted_reflection":
            raise ValueError("unknown contract-adapted proposal engine")
        if self.renderer_version != CONTRACT_ADAPTED_RENDERER_VERSION:
            raise ValueError("unknown contract-adapted renderer")
        if self.evaluation_mode != "progressive":
            raise ValueError("contract-adapted v1 freezes progressive evaluation")
        if self.selection_arms != ("B0_PRIME_CURRENT", "B1_PRIME_TEAM_PARETO"):
            raise ValueError("contract-adapted v1 freezes the two shared-pool selectors")
        if self.solver_contract_id != COMMON_SOLVER_CONTRACT_V1_ID:
            raise ValueError("contract-adapted v1 requires COMMON_SOLVER_CONTRACT_V1")
        if self.commit_enabled or self.validation_enabled or self.test_enabled or self.memory_enabled:
            raise ValueError("contract-adapted v1 is non-committing and excludes validation, test, and memory")

    def identity(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContractAdaptedProtocolV2:
    candidates_per_case: int = 2
    case_count: int = 6
    minibatch_size: int = 12
    max_full_eval_candidates_per_case: int = 2
    proposal_engine: str = "contract_adapted_reflection"
    renderer_version: str = CONTRACT_ADAPTED_RENDERER_VERSION
    hypothesis_interface_version: str = HYPOTHESIS_INTERFACE_V2_VERSION
    optimizer_model: str = "qwen3.7-flash"
    reflection_temperature: float = 0.0
    reflection_max_tokens: int = 64
    reflection_role: str = "reflection"
    evaluation_mode: str = "progressive"
    selection_arms: tuple[str, str] = ("B0_PRIME_CURRENT", "B1_PRIME_TEAM_PARETO")
    solver_contract_id: str = COMMON_SOLVER_CONTRACT_V1_ID
    commit_enabled: bool = False
    validation_enabled: bool = False
    test_enabled: bool = False
    memory_enabled: bool = False

    def __post_init__(self) -> None:
        if self.candidates_per_case != 2 or self.case_count != 6:
            raise ValueError("contract-adapted v2 freezes six cases and two candidates per case")
        if self.minibatch_size != 12 or self.max_full_eval_candidates_per_case != 2:
            raise ValueError("contract-adapted v2 freezes progressive evaluation budgets")
        if self.proposal_engine != "contract_adapted_reflection":
            raise ValueError("unknown contract-adapted v2 proposal engine")
        if self.renderer_version != CONTRACT_ADAPTED_RENDERER_VERSION:
            raise ValueError("unknown contract-adapted v2 renderer")
        if self.hypothesis_interface_version != HYPOTHESIS_INTERFACE_V2_VERSION:
            raise ValueError("contract-adapted v2 requires hypothesis interface V2")
        if self.optimizer_model != "qwen3.7-flash":
            raise ValueError("contract-adapted v2 freezes qwen3.7-flash reflection")
        if self.reflection_temperature != 0.0 or self.reflection_max_tokens != 64:
            raise ValueError("contract-adapted v2 decoding drift")
        if self.reflection_role != "reflection":
            raise ValueError("contract-adapted v2 reflection role drift")
        if self.evaluation_mode != "progressive":
            raise ValueError("contract-adapted v2 freezes progressive evaluation")
        if self.selection_arms != ("B0_PRIME_CURRENT", "B1_PRIME_TEAM_PARETO"):
            raise ValueError("contract-adapted v2 freezes shared-pool selectors")
        if self.solver_contract_id != COMMON_SOLVER_CONTRACT_V1_ID:
            raise ValueError("contract-adapted v2 requires COMMON_SOLVER_CONTRACT_V1")
        if self.commit_enabled or self.validation_enabled or self.test_enabled or self.memory_enabled:
            raise ValueError("contract-adapted v2 is non-committing and excludes validation, test, and memory")

    def identity(self) -> str:
        payload = json.dumps(
            {
                **asdict(self),
                "hypothesis_interface_hash": hypothesis_interface_v2_identity(),
                "renderer_vocabulary_hash": renderer_vocabulary_identity(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    @property
    def hypothesis_interface_hash(self) -> str:
        return hypothesis_interface_v2_identity()

    @property
    def renderer_vocabulary_hash(self) -> str:
        return renderer_vocabulary_identity()


@dataclass(frozen=True)
class EditHypothesis:
    failure_pattern: str
    behavioral_change: str
    preserve: str
    avoid: str

    def __post_init__(self) -> None:
        choices = (
            (self.failure_pattern, FAILURE_PATTERNS, "failure_pattern"),
            (self.behavioral_change, BEHAVIORAL_CHANGES, "behavioral_change"),
            (self.preserve, PRESERVATION_PRIORITIES, "preserve"),
            (self.avoid, AVOIDANCE_PRIORITIES, "avoid"),
        )
        for value, allowed, field in choices:
            if value not in allowed:
                raise ValueError(f"unknown contract-adapted {field}")

    def identity(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


def parse_edit_hypothesis(raw: str) -> EditHypothesis:
    """Parse one exact JSON object; prose, fences, missing, or extra keys fail."""

    try:
        payload = json.loads(str(raw).strip())
    except json.JSONDecodeError as exc:
        raise ValueError("contract-adapted reflection must be exact JSON") from exc
    if not isinstance(payload, dict) or set(payload) != set(HYPOTHESIS_FIELDS):
        raise ValueError("contract-adapted reflection schema mismatch")
    if any(not isinstance(payload[field], str) for field in HYPOTHESIS_FIELDS):
        raise ValueError("contract-adapted reflection fields must be strings")
    return EditHypothesis(**{field: payload[field] for field in HYPOTHESIS_FIELDS})


def render_contract_adapted_prompt(parent_prompt: str, hypothesis: EditHypothesis) -> str:
    """Render only program-owned text; raw reflection text never enters output."""

    parent = str(parent_prompt).rstrip()
    validate_mutable_decision_procedure(parent)
    rendered = (
        f"{parent}\n\n"
        "Additional decision discipline:\n"
        f"- Diagnostic focus: {FAILURE_PATTERNS[hypothesis.failure_pattern]}\n"
        f"- Behavioral change: {BEHAVIORAL_CHANGES[hypothesis.behavioral_change]}\n"
        f"- Preserve: {PRESERVATION_PRIORITIES[hypothesis.preserve]}\n"
        f"- Avoid: {AVOIDANCE_PRIORITIES[hypothesis.avoid]}"
    )
    validate_mutable_decision_procedure(rendered)
    return rendered


def renderer_vocabulary_identity() -> str:
    payload = {
        "version": CONTRACT_ADAPTED_RENDERER_VERSION,
        "failure_patterns": dict(FAILURE_PATTERNS),
        "behavioral_changes": dict(BEHAVIORAL_CHANGES),
        "preservation_priorities": dict(PRESERVATION_PRIORITIES),
        "avoidance_priorities": dict(AVOIDANCE_PRIORITIES),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# V2 eliminates model-authored field names.  The model returns an exact
# four-position JSON array of opaque IDs.  Array position supplies the program-
# owned field name, and the renderer expands only program-owned vocabulary.
FAILURE_IDS: Mapping[str, str] = {
    f"F{index}": key for index, key in enumerate(FAILURE_PATTERNS, start=1)
}
EDIT_IDS: Mapping[str, str] = {
    f"E{index}": key for index, key in enumerate(BEHAVIORAL_CHANGES, start=1)
}
PRESERVE_IDS: Mapping[str, str] = {
    f"P{index}": key for index, key in enumerate(PRESERVATION_PRIORITIES, start=1)
}
AVOID_IDS: Mapping[str, str] = {
    f"A{index}": key for index, key in enumerate(AVOIDANCE_PRIORITIES, start=1)
}


@dataclass(frozen=True)
class HypothesisSelectionV2:
    failure_id: str
    edit_id: str
    preserve_id: str
    avoid_id: str

    def __post_init__(self) -> None:
        for value, allowed, label in (
            (self.failure_id, FAILURE_IDS, "failure_id"),
            (self.edit_id, EDIT_IDS, "edit_id"),
            (self.preserve_id, PRESERVE_IDS, "preserve_id"),
            (self.avoid_id, AVOID_IDS, "avoid_id"),
        ):
            if value not in allowed:
                raise ValueError(f"unknown hypothesis-interface-v2 {label}")

    def identity(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def materialize(self) -> EditHypothesis:
        return EditHypothesis(
            failure_pattern=FAILURE_IDS[self.failure_id],
            behavioral_change=EDIT_IDS[self.edit_id],
            preserve=PRESERVE_IDS[self.preserve_id],
            avoid=AVOID_IDS[self.avoid_id],
        )


def parse_hypothesis_selection_v2(raw: str) -> HypothesisSelectionV2:
    """Parse exact ``[F#, E#, P#, A#]`` JSON with no cleanup or aliases."""

    try:
        payload = json.loads(str(raw).strip())
    except json.JSONDecodeError as exc:
        raise ValueError("hypothesis-interface-v2 must be exact JSON") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 4
        or any(not isinstance(value, str) for value in payload)
    ):
        raise ValueError("hypothesis-interface-v2 schema mismatch")
    return HypothesisSelectionV2(*payload)


def render_contract_adapted_prompt_v2(
    parent_prompt: str, selection: HypothesisSelectionV2
) -> str:
    return render_contract_adapted_prompt(parent_prompt, selection.materialize())


def hypothesis_interface_v2_identity() -> str:
    payload = {
        "version": HYPOTHESIS_INTERFACE_V2_VERSION,
        "wire_format": "exact_json_array_length_4",
        "position_0": dict(FAILURE_IDS),
        "position_1": dict(EDIT_IDS),
        "position_2": dict(PRESERVE_IDS),
        "position_3": dict(AVOID_IDS),
        "cleanup": False,
        "aliases": False,
        "fallback": False,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_hypothesis_request_v2(
    *,
    responsibility_lane: str,
    context_lines: tuple[str, ...],
    mutation_index: int,
) -> tuple[str, str]:
    """Authoritative V2 wire request shared by qualification and science paths."""

    if responsibility_lane not in {"coverage", "margin_support", "direct_flip"}:
        raise ValueError("unknown V2 responsibility lane")
    if not context_lines or any(not str(line).strip() for line in context_lines):
        raise ValueError("V2 reflection context must contain non-empty lines")
    if mutation_index not in {0, 1}:
        raise ValueError("V2 mutation index must be 0 or 1")
    system = (
        "Select one bounded reasoning-edit hypothesis. Return only an exact JSON array of four "
        "quoted IDs in this fixed order: failure, edit, preserve, avoid. Do not emit keys, prose, "
        "markdown, explanations, or any additional value."
    )
    user = (
        f"Allowed failure IDs: {','.join(FAILURE_IDS)}\n"
        f"Allowed edit IDs: {','.join(EDIT_IDS)}\n"
        f"Allowed preserve IDs: {','.join(PRESERVE_IDS)}\n"
        f"Allowed avoid IDs: {','.join(AVOID_IDS)}\n"
        f"Responsibility lane: {responsibility_lane}\n"
        + "\n".join(context_lines)
        + f"\nMutation index: {mutation_index}\n"
        "Required wire shape example: [\"F1\",\"E1\",\"P1\",\"A1\"]"
    )
    return system, user


@dataclass(frozen=True)
class SchemaQualificationV2Protocol:
    request_count: int = 12
    optimizer_model: str = "qwen3.7-flash"
    temperature: float = 0.0
    max_tokens: int = 64
    evidence_kind: str = "synthetic_non_evaluation_symbolic"
    strict_parser: bool = True
    cleanup_enabled: bool = False
    alias_mapping_enabled: bool = False
    schema_retry_enabled: bool = False
    solver_enabled: bool = False
    validation_enabled: bool = False
    test_enabled: bool = False
    scientific_evidence_eligible: bool = False

    def __post_init__(self) -> None:
        if self.request_count != 12 or self.optimizer_model != "qwen3.7-flash":
            raise ValueError("hypothesis-interface-v2 qualification freezes 12 qwen3.7-flash requests")
        if self.temperature != 0.0 or self.max_tokens != 64:
            raise ValueError("hypothesis-interface-v2 qualification decoding drift")
        if self.evidence_kind != "synthetic_non_evaluation_symbolic":
            raise ValueError("qualification may not use experiment evaluation evidence")
        if not self.strict_parser:
            raise ValueError("qualification parser must be strict")
        if any((
            self.cleanup_enabled,
            self.alias_mapping_enabled,
            self.schema_retry_enabled,
            self.solver_enabled,
            self.validation_enabled,
            self.test_enabled,
            self.scientific_evidence_eligible,
        )):
            raise ValueError("qualification must remain isolated and fail closed")

    def identity(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()
