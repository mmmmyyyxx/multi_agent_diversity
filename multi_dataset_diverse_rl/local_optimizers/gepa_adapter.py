"""Single-member COMMON_SOLVER_CONTRACT_V1 adapter for official GEPA."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Protocol, Sequence

from ..evaluation.mutable_prompt_contract import (
    mutable_prompt_violation_reasons,
    validate_mutable_decision_procedure,
)
from ..versions import COMMON_SOLVER_CONTRACT_V1_ID
from ..evaluation.solver_output import FINAL_ANSWER_LINE
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

    def evaluate(
        self, decision_procedure: str, example: LocalEvidenceExample
    ) -> LocalSolverObservation:
        ...


@dataclass(frozen=True)
class LocalGEPATrajectory:
    example: LocalEvidenceExample
    observation: LocalSolverObservation


_REFLECTION_OUTPUT_TOPIC = re.compile(
    r"(?i)\b(?:output|response|answer)[\s_-]*"
    r"(?:contract|interface|format|schema|protocol|instruction)s?\b"
)


def reasoning_evidence_from_output(raw_output: str) -> str:
    """Keep reasoning evidence while excluding the immutable Solver interface.

    Reflection optimizes only ``decision_procedure``.  Provider response
    markers and formatting commentary therefore are not evidence for this
    mutable component and must not enter GEPA's ``side_info``.
    """

    normalized = str(raw_output or "").replace("\r\n", "\n").replace("\r", "\n")
    candidate_lines = [
        line.rstrip()
        for line in normalized.split("\n")
        if not FINAL_ANSWER_LINE.fullmatch(line)
    ]
    rejected: set[int] = set()
    for index, line in enumerate(candidate_lines):
        if mutable_prompt_violation_reasons(line) or _REFLECTION_OUTPUT_TOPIC.search(line):
            rejected.add(index)
    # Interface language can be split across provider line wrapping (for
    # example, ``response`` followed by ``format``).  Evaluate adjacent lines
    # as one sentence so line wrapping cannot bypass the same immutable-shell
    # exclusion rule.  This remains a projection of evidence, not a new
    # reflective-data field or optimizer signal.
    for index in range(len(candidate_lines) - 1):
        if index in rejected or index + 1 in rejected:
            continue
        joined = " ".join(candidate_lines[index : index + 2]).strip()
        if mutable_prompt_violation_reasons(joined) or _REFLECTION_OUTPUT_TOPIC.search(joined):
            rejected.update((index, index + 1))
    retained = [
        line for index, line in enumerate(candidate_lines) if index not in rejected
    ]
    evidence = "\n".join(retained).strip()
    return evidence or "No reusable reasoning trace was available."


def _reasoning_focus(
    tags: Sequence[str], optimization_context: str
) -> Mapping[str, str]:
    """Project controller evidence onto an allowlisted reasoning-only schema."""

    allowed_groups = ("responsibility", "coalition", "preservation")
    allowed_lanes = ("direct_flip", "near_margin", "coverage", "fallback")
    group = next((value for value in tags if value in allowed_groups), "general")
    lane = next((value for value in tags if value in allowed_lanes), None)
    if lane is None:
        match = re.search(
            r"primary_responsibility_lane=(direct_flip|near_margin|coverage|fallback)",
            optimization_context,
        )
        lane = match.group(1) if match else "general"
    return {"evidence_group": group, "reasoning_lane": lane}


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


def compact_prompt_failed_checks(
    prompt: str,
    *,
    parent_prompt: str,
    examples: Sequence[LocalEvidenceExample],
    max_chars: int,
) -> tuple[str, ...]:
    """Return sanitized, stable failed checks without retaining proposal text."""

    checks: list[str] = []
    if not prompt.strip():
        checks.append("empty")
    if len(prompt) > max_chars:
        checks.append("over_length")
    checks.extend(mutable_prompt_violation_reasons(prompt))
    if prompt != parent_prompt and prompt.startswith(parent_prompt.rstrip()):
        checks.append("append_only")
    if contains_supplied_example_text(prompt, examples):
        checks.append("example_copying")
    return tuple(dict.fromkeys(checks))


def primary_prompt_rejection_category(failed_checks: Sequence[str]) -> str | None:
    """Map multi-hot failed checks to one preregistered primary category."""

    checks = set(failed_checks)
    if "over_length" in checks:
        return "over_length"
    if checks.intersection(
        {"forbidden_final_answer_marker", "copied_solver_interface", "fixed_answer_payload"}
    ):
        return "output_contract_contamination"
    if "example_copying" in checks:
        return "example_copying"
    if "append_only" in checks:
        return "append_only"
    if checks:
        return "other_failed_check"
    return None


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
        self.solver_reached_proposal_hashes: set[str] = set()

    @staticmethod
    def _decision_procedure(candidate: Mapping[str, str]) -> str:
        if set(candidate) != {"decision_procedure"} or not isinstance(
            candidate.get("decision_procedure"), str
        ):
            raise ValueError("GEPA candidate must contain exactly one decision_procedure")
        return candidate["decision_procedure"]

    def evaluate(
        self,
        batch: list[LocalEvidenceExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[LocalGEPATrajectory, dict[str, Any]]:
        decision_procedure = self._decision_procedure(candidate)
        try:
            validate_complete_compact_prompt(
                decision_procedure,
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
            # The evaluator is the sole owner of composing the mutable decision
            # procedure with COMMON_SOLVER_CONTRACT_V1's immutable task shell and
            # output interface. The adapter must never append a second contract.
            observations = [
                self.evaluator.evaluate(decision_procedure, row) for row in batch
            ]
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
            if decision_procedure != self.parent_prompt and any(
                observation.provider_called for observation in observations
            ):
                self.solver_reached_proposal_hashes.add(
                    hashlib.sha256(decision_procedure.encode("utf-8")).hexdigest()
                )
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
        self._decision_procedure(candidate)
        if components_to_update != ["decision_procedure"]:
            raise ValueError("only decision_procedure may be evolved")
        if eval_batch.trajectories is None:
            raise ValueError("GEPA reflection requires captured local trajectories")
        records: list[dict[str, Any]] = []
        for trajectory in eval_batch.trajectories:
            example = trajectory.example
            observation = trajectory.observation
            outcome = (
                "correct"
                if observation.valid and observation.correct
                else "incorrect"
                if observation.valid
                else "invalid"
            )
            records.append(
                {
                    "Problem": example.input_payload,
                    "Reasoning Evidence": reasoning_evidence_from_output(
                        observation.raw_output
                    ),
                    "Evaluation Outcome": outcome,
                    "Reasoning Focus": _reasoning_focus(
                        example.tags, self.optimization_context
                    ),
                    "example_id": example.example_id,
                }
            )
        return {"decision_procedure": records}


# Public project-facing name; it structurally implements gepa.core.adapter.GEPAAdapter.
GEPAAdapter = DiversityGEPAAdapter
