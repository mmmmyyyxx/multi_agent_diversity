"""Single-member COMMON_SOLVER_CONTRACT_V1 adapter for official GEPA."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Protocol, Sequence

from ..evaluation.mutable_prompt_contract import validate_mutable_decision_procedure
from ..versions import COMMON_SOLVER_CONTRACT_V1_ID
from .gepa_runtime import import_frozen_gepa
from .schemas import LocalEvidenceExample

import_frozen_gepa()
from gepa.core.adapter import EvaluationBatch  # type: ignore[import-not-found]  # noqa: E402


@dataclass(frozen=True)
class LocalSolverObservation:
    parsed_answer: str | None
    raw_output: str
    correct: bool
    valid: bool
    failure_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    provider_called: bool = True


class LocalSolverEvaluator(Protocol):
    solver_contract_id: str
    output_contract_id: str

    def evaluate(self, prompt: str, example: LocalEvidenceExample) -> LocalSolverObservation:
        ...


@dataclass(frozen=True)
class LocalGEPATrajectory:
    example: LocalEvidenceExample
    observation: LocalSolverObservation


def _normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def contains_supplied_example_text(prompt: str, examples: Sequence[LocalEvidenceExample]) -> bool:
    """Reject long verbatim example fragments without importing team TCS code."""

    normalized_prompt = " ".join(_normalized_tokens(prompt))
    for example in examples:
        source = " ".join(_normalized_tokens(example.input_payload))
        words = source.split()
        for width in (12, 10):
            if len(words) >= width and any(
                " ".join(words[index : index + width]) in normalized_prompt
                for index in range(len(words) - width + 1)
            ):
                return True
    return False


def validate_complete_compact_prompt(
    prompt: str, *, parent_prompt: str, examples: Sequence[LocalEvidenceExample], max_chars: int
) -> None:
    if not prompt.strip() or len(prompt) > max_chars:
        raise ValueError("GEPA candidate is empty or exceeds the compact prompt limit")
    validate_mutable_decision_procedure(prompt)
    if prompt != parent_prompt and prompt.startswith(parent_prompt.rstrip()):
        raise ValueError("append-only GEPA mutation is forbidden")
    if contains_supplied_example_text(prompt, examples):
        raise ValueError("GEPA candidate copies supplied example text")


class DiversityGEPAAdapter:
    """GEPA sees one prompt and one local example; never a five-member team."""

    propose_new_texts = None

    def __init__(
        self,
        evaluator: LocalSolverEvaluator,
        *,
        parent_prompt: str,
        all_examples: Sequence[LocalEvidenceExample],
        optimization_context: str,
        output_contract_id: str,
        max_prompt_chars: int = 3000,
    ) -> None:
        if evaluator.solver_contract_id != COMMON_SOLVER_CONTRACT_V1_ID:
            raise ValueError("GEPA adapter requires COMMON_SOLVER_CONTRACT_V1")
        if evaluator.output_contract_id != output_contract_id:
            raise ValueError("GEPA adapter output contract mismatch")
        self.evaluator = evaluator
        self.parent_prompt = parent_prompt
        self.all_examples = tuple(all_examples)
        self.optimization_context = optimization_context
        self.output_contract_id = output_contract_id
        self.max_prompt_chars = max_prompt_chars
        self.solver_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @staticmethod
    def _prompt(candidate: Mapping[str, str]) -> str:
        if set(candidate) != {"system_prompt"} or not isinstance(candidate.get("system_prompt"), str):
            raise ValueError("GEPA candidate must contain exactly one system_prompt")
        return candidate["system_prompt"]

    def evaluate(
        self,
        batch: list[LocalEvidenceExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[LocalGEPATrajectory, dict[str, Any]]:
        prompt = self._prompt(candidate)
        try:
            validate_complete_compact_prompt(
                prompt,
                parent_prompt=self.parent_prompt,
                examples=self.all_examples,
                max_chars=self.max_prompt_chars,
            )
        except ValueError as exc:
            outputs = [
                {"example_id": row.example_id, "valid": False, "correct": False, "failure": str(exc)}
                for row in batch
            ]
            observations = [
                LocalSolverObservation(None, "", False, False, str(exc), provider_called=False)
                for _ in batch
            ]
        else:
            observations = [self.evaluator.evaluate(prompt, row) for row in batch]
            outputs = [
                {
                    "example_id": row.example_id,
                    "parsed_answer": observation.parsed_answer,
                    "valid": observation.valid,
                    "correct": observation.correct,
                    "failure": observation.failure_reason,
                }
                for row, observation in zip(batch, observations, strict=True)
            ]
            self.solver_calls += sum(observation.provider_called for observation in observations)
            self.input_tokens += sum(observation.input_tokens for observation in observations)
            self.output_tokens += sum(observation.output_tokens for observation in observations)
        trajectories = (
            [
                LocalGEPATrajectory(example=row, observation=observation)
                for row, observation in zip(batch, observations, strict=True)
            ]
            if capture_traces
            else None
        )
        return EvaluationBatch(
            outputs=outputs,
            scores=[1.0 if observation.correct and observation.valid else 0.0 for observation in observations],
            trajectories=trajectories,
            objective_scores=None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[LocalGEPATrajectory, dict[str, Any]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        self._prompt(candidate)
        if components_to_update != ["system_prompt"]:
            raise ValueError("only system_prompt may be evolved")
        if eval_batch.trajectories is None:
            raise ValueError("GEPA reflection requires captured local trajectories")
        records: list[dict[str, Any]] = []
        for trajectory in eval_batch.trajectories:
            example = trajectory.example
            observation = trajectory.observation
            feedback = example.textual_feedback or (
                "Correct; preserve this reasoning capability."
                if observation.correct
                else f"Incorrect or invalid. Expected label: {example.gold}. Failure: {observation.failure_reason or 'wrong answer'}."
            )
            if self.optimization_context:
                feedback = f"{feedback}\nController context: {self.optimization_context}"
            records.append(
                {
                    "Inputs": {"question": example.input_payload},
                    "Generated Outputs": observation.raw_output,
                    "Feedback": feedback,
                    "example_id": example.example_id,
                    "tags": list(example.tags),
                }
            )
        return {"system_prompt": records}


# Public project-facing name; it structurally implements gepa.core.adapter.GEPAAdapter.
GEPAAdapter = DiversityGEPAAdapter
