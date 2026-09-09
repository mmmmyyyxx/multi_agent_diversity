"""Pure, opt-in primitives for the Responsibility-Guided GEPA pilot.

Nothing in this module is imported by the canonical v15 runtime.  It contains
the frozen policy decisions that an experimental fixed-parent runner needs, so
they can be unit-tested and hashed before a provider is ever contacted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable, Mapping, Sequence

from .versions import COMMON_SOLVER_CONTRACT_V1_ID


RG_GEPA_PROTOCOL_VERSION = "responsibility_guided_gepa_fixed_parent_v1"
RG_GEPA_LEDGER_VERSION = "rg_gepa_execution_ledger_v2"
PROPOSAL_ENGINES = ("current", "gepa_reflection")
EVALUATION_MODES = ("full", "progressive")
SELECTION_MODES = ("current", "team_pareto")
LEDGER_STAGES = (
    "reflection", "minibatch_parent", "minibatch_candidate", "full_member",
    "full_team", "shadow", "audit_only",
)


@dataclass(frozen=True)
class RGGEPAProtocol:
    proposal_engine: str = "gepa_reflection"
    evaluation_mode: str = "progressive"
    selection_mode: str = "team_pareto"
    minibatch_size: int = 12
    max_full_eval_candidates_per_opportunity: int = 2
    candidates_per_target: int = 2
    solver_contract_id: str = COMMON_SOLVER_CONTRACT_V1_ID
    commit_enabled: bool = False
    validation_enabled: bool = False
    test_enabled: bool = False

    def __post_init__(self) -> None:
        if self.proposal_engine not in PROPOSAL_ENGINES:
            raise ValueError("unknown RG-GEPA proposal engine")
        if self.evaluation_mode not in EVALUATION_MODES:
            raise ValueError("unknown RG-GEPA evaluation mode")
        if self.selection_mode not in SELECTION_MODES:
            raise ValueError("unknown RG-GEPA selection mode")
        if self.minibatch_size != 12:
            raise ValueError("RG-GEPA v1 freezes minibatch_size=12")
        if self.max_full_eval_candidates_per_opportunity != 2:
            raise ValueError("RG-GEPA v1 freezes max_full_eval_candidates_per_opportunity=2")
        if self.candidates_per_target != 2:
            raise ValueError("RG-GEPA v1 retains two candidates per target")
        if self.solver_contract_id != COMMON_SOLVER_CONTRACT_V1_ID:
            raise ValueError("RG-GEPA requires COMMON_SOLVER_CONTRACT_V1")
        if self.commit_enabled or self.validation_enabled or self.test_enabled:
            raise ValueError("fixed-parent RG-GEPA v1 is analytical only")

    def identity(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceItem:
    example_id: str
    source_type: str
    responsibility_type: str
    parent_member_correct: bool
    parent_vote_correct: bool
    parent_g: int
    filler_key: str = ""


@dataclass(frozen=True, order=True)
class TeamVector:
    """Larger is better in every field; invalid is represented separately."""
    vote_correct: int
    target_correct: int
    total_member_correct: int
    majority_preservation: int
    responsibility_recovery: int


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    vector: TeamVector
    invalid_count: int
    hard_gate_passed: bool = True


def dominates(left: TeamVector, right: TeamVector) -> bool:
    left_values = (left.vote_correct, left.target_correct, left.total_member_correct,
                   left.majority_preservation, left.responsibility_recovery)
    right_values = (right.vote_correct, right.target_correct, right.total_member_correct,
                    right.majority_preservation, right.responsibility_recovery)
    return all(a >= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a > b for a, b in zip(left_values, right_values, strict=True)
    )


def pareto_frontier(rows: Sequence[CandidateScore]) -> list[CandidateScore]:
    """Return the deterministic non-dominated frontier, including exact ties."""
    return sorted(
        [row for row in rows if not any(
            other.candidate_id != row.candidate_id and dominates(other.vector, row.vector)
            for other in rows
        )],
        key=lambda row: row.candidate_id,
    )


def deterministic_candidate_key(row: CandidateScore) -> tuple[int, int, int, int, int, int, str]:
    vector = row.vector
    return (
        vector.vote_correct, vector.target_correct, vector.total_member_correct,
        vector.majority_preservation, vector.responsibility_recovery,
        -row.invalid_count, row.candidate_id,
    )


def deterministic_candidate_order(row: CandidateScore) -> tuple[int, int, int, int, int, int, str]:
    """Ascending sort key: metrics descending, then stable ID ascending."""
    vector = row.vector
    return (-vector.vote_correct, -vector.target_correct, -vector.total_member_correct,
            -vector.majority_preservation, -vector.responsibility_recovery,
            row.invalid_count, row.candidate_id)


def progressive_promotions(
    *, parent: TeamVector, candidates: Sequence[CandidateScore], max_promotions: int = 2
) -> list[CandidateScore]:
    """Frozen minibatch gate: hard-valid, not parent-dominated, and improved."""
    if max_promotions != 2:
        raise ValueError("v1 freezes two expensive promotions")
    survivors = [
        row for row in candidates
        if row.hard_gate_passed and not dominates(parent, row.vector)
        and dominates(row.vector, parent)
    ]
    frontier = pareto_frontier(survivors)
    return sorted(frontier, key=deterministic_candidate_order)[:max_promotions]


def team_pareto_winner(candidates: Sequence[CandidateScore]) -> tuple[CandidateScore | None, list[CandidateScore]]:
    feasible = [row for row in candidates if row.hard_gate_passed]
    frontier = pareto_frontier(feasible)
    winner = min(frontier, key=deterministic_candidate_order, default=None)
    return winner, frontier


def current_selector_winner(candidates: Sequence[CandidateScore]) -> CandidateScore | None:
    """A transparent fixed-parent approximation of the current common key.

    Full runtime selection remains authoritative.  The explicit tie-break is
    used only for B0/B1 comparison of the same already-feasible candidates.
    """
    feasible = [row for row in candidates if row.hard_gate_passed]
    return max(
        feasible,
        key=lambda row: (row.vector.vote_correct, row.vector.target_correct,
                         row.vector.total_member_correct, -row.invalid_count,
                         row.candidate_id),
        default=None,
    )


def deterministic_minibatch(
    evidence: Iterable[EvidenceItem], *, size: int = 12
) -> list[EvidenceItem]:
    """Prefer responsibility evidence, then correctness-preserving fillers.

    Input may contain duplicates from overlapping responsibility categories;
    canonical example IDs are deduplicated before deterministic selection.
    """
    if size != 12:
        raise ValueError("v1 freezes a 12-example minibatch")
    by_id: dict[str, EvidenceItem] = {}
    source_rank = {"coverage": 0, "conversion": 1, "preservation": 2, "representative": 3, "filler": 4}
    for item in evidence:
        prior = by_id.get(item.example_id)
        if prior is None or (source_rank.get(item.source_type, 99), item.filler_key) < (
            source_rank.get(prior.source_type, 99), prior.filler_key
        ):
            by_id[item.example_id] = item
    values = list(by_id.values())
    primary = [item for item in values if item.source_type != "filler"]
    filler = [item for item in values if item.source_type == "filler"]
    primary.sort(key=lambda item: (source_rank.get(item.source_type, 99), item.responsibility_type, item.example_id))
    # Correct parent examples are first among filler to make collateral visible.
    filler.sort(key=lambda item: (not item.parent_member_correct, not item.parent_vote_correct,
                                  item.parent_g, item.filler_key, item.example_id))
    selected = [*primary[:size], *filler[:max(0, size - len(primary))]]
    if len(selected) != size:
        raise ValueError("insufficient Optimize-only examples for RG-GEPA minibatch")
    if len({item.example_id for item in selected}) != size:
        raise AssertionError("minibatch deduplication failed")
    return selected


def validate_ledger_record(row: Mapping[str, object]) -> None:
    required = {
        "ledger_version",
        "seed", "parent_id", "update_index", "candidate_id", "proposal_engine",
        "evaluation_stage", "input_tokens", "output_tokens", "total_tokens",
        "provider_attempt_id", "logical_call_id", "attempt_index", "record_kind",
        "provider_attempts", "successful_provider_calls", "cache_hit",
        "logical_role", "client_role", "success",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"RG-GEPA ledger missing fields: {sorted(missing)}")
    if str(row["ledger_version"]) != RG_GEPA_LEDGER_VERSION:
        raise ValueError("unknown RG-GEPA ledger version")
    if str(row["evaluation_stage"]) not in LEDGER_STAGES:
        raise ValueError("unknown RG-GEPA ledger stage")
    if str(row["proposal_engine"]) not in PROPOSAL_ENGINES:
        raise ValueError("unknown RG-GEPA proposal engine in ledger")
    if str(row["record_kind"]) not in {
        "solver_logical_invocation", "optimizer_provider_attempt", "cache_reuse",
    }:
        raise ValueError("unknown RG-GEPA ledger record kind")
    if not str(row["logical_call_id"]) or not str(row["provider_attempt_id"]):
        raise ValueError("RG-GEPA ledger identities must be non-empty")
    if int(row["attempt_index"]) < 0:
        raise ValueError("RG-GEPA ledger attempt index cannot be negative")
    attempts = int(row["provider_attempts"])
    successes = int(row["successful_provider_calls"])
    if attempts < 0 or successes < 0 or successes > attempts:
        raise ValueError("invalid RG-GEPA provider-attempt accounting")
    kind = str(row["record_kind"])
    if kind == "optimizer_provider_attempt" and (attempts != 1 or bool(row["cache_hit"])):
        raise ValueError("optimizer attempt rows must describe one non-cache provider attempt")
    if kind == "optimizer_provider_attempt" and successes != int(bool(row.get("success", False))):
        raise ValueError("optimizer attempt success accounting is inconsistent")
    if kind == "cache_reuse" and (attempts != 0 or successes != 0 or not bool(row["cache_hit"])):
        raise ValueError("cache-reuse rows cannot describe provider attempts")
    if kind == "solver_logical_invocation" and bool(row["cache_hit"]) != (attempts == 0):
        raise ValueError("solver invocation cache/provider semantics are inconsistent")
    if kind == "solver_logical_invocation" and (
        int(row["attempt_index"]) != 0
        or not bool(row["success"])
        or successes != int(not bool(row["cache_hit"]))
    ):
        raise ValueError("solver logical-invocation accounting is inconsistent")
    if kind == "cache_reuse" and (int(row["attempt_index"]) != 0 or not bool(row["success"])):
        raise ValueError("cache-reuse success accounting is inconsistent")
    prompt, completion, total = (int(row[key]) for key in ("input_tokens", "output_tokens", "total_tokens"))
    if prompt < 0 or completion < 0 or total < 0 or prompt + completion != total:
        raise ValueError("invalid provider token accounting")
    if bool(row["cache_hit"]) and total != 0:
        raise ValueError("cache ledger rows cannot carry provider tokens")
