"""Backend-neutral data transferred across the two-layer boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class OpaqueOptimizerState:
    """Serialized backend state whose contents Layer 2 must never inspect."""

    backend_name: str
    backend_version: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalOptimizerBudget:
    max_metric_calls: int
    reflection_minibatch_size: int = 3
    max_returned_candidates: int = 4

    def __post_init__(self) -> None:
        if self.max_metric_calls <= 0:
            raise ValueError("local optimizer metric budget must be positive")
        if self.reflection_minibatch_size <= 0:
            raise ValueError("reflection minibatch size must be positive")
        if self.max_returned_candidates <= 0:
            raise ValueError("local candidate return budget must be positive")


@dataclass(frozen=True)
class LocalEvidenceExample:
    example_id: str
    input_payload: str
    gold: str
    parent_output: str | None = None
    textual_feedback: str | None = None
    tags: tuple[str, ...] = ()
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.example_id or not self.input_payload or not self.gold:
            raise ValueError("local evidence identity, input, and gold are required")
        if self.weight <= 0:
            raise ValueError("local evidence weight must be positive")


@dataclass(frozen=True)
class LocalOptimizationTask:
    task_id: str
    parent_prompt: str
    search_examples: tuple[LocalEvidenceExample, ...]
    local_validation_examples: tuple[LocalEvidenceExample, ...]
    optimization_context: str
    solver_contract_id: str
    output_contract_id: str
    seed: int
    budget: LocalOptimizerBudget
    backend_state: OpaqueOptimizerState | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.parent_prompt:
            raise ValueError("local task identity and parent prompt are required")
        if not self.search_examples or not self.local_validation_examples:
            raise ValueError("local search and validation evidence cannot be empty")
        search_ids = [row.example_id for row in self.search_examples]
        validation_ids = [row.example_id for row in self.local_validation_examples]
        if len(search_ids) != len(set(search_ids)) or len(validation_ids) != len(set(validation_ids)):
            raise ValueError("local evidence ids must be unique within each split")


@dataclass(frozen=True)
class LocalPromptCandidate:
    candidate_id: str
    prompt: str
    local_score: float | None
    per_example_scores: Mapping[str, float]
    parent_ids: tuple[str, ...]
    generation: int
    local_rank_metadata: Mapping[str, Any] = field(default_factory=dict)
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.prompt:
            raise ValueError("local candidate identity and prompt are required")
        if self.generation < 0:
            raise ValueError("local candidate generation cannot be negative")


@dataclass(frozen=True)
class LocalOptimizationResult:
    candidates: tuple[LocalPromptCandidate, ...]
    backend_name: str
    backend_version: str
    optimizer_state: OpaqueOptimizerState | None
    solver_calls: int
    optimizer_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    termination_reason: str

    def __post_init__(self) -> None:
        counts = (
            self.solver_calls,
            self.optimizer_calls,
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
        )
        if any(value < 0 for value in counts):
            raise ValueError("local optimizer accounting cannot be negative")
        if self.input_tokens + self.output_tokens != self.total_tokens:
            raise ValueError("local optimizer token arithmetic mismatch")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("local candidate ids must be unique")
