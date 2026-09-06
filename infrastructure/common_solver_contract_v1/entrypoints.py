"""Repository boundary adapters; all terminate in one V1 serializer."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contract import canonical_question_payload, serialize_solver_request


def from_diversity(*, prompt: str, raw_question: str) -> dict[str, Any]:
    return serialize_solver_request(decision_procedure=prompt, question=raw_question)


def from_mars(*, prompt: str, rendered_question: str) -> dict[str, Any]:
    return serialize_solver_request(decision_procedure=prompt, question=rendered_question)


def from_gepa(*, prompt: str, example: Mapping[str, Any]) -> dict[str, Any]:
    labels: Sequence[str] = example["option_labels"]
    choices: Sequence[str] = example["choices"]
    rendered = "\n".join(
        [
            str(example["question"]).strip(),
            "Options:",
            *(f"({label}) {choice}" for label, choice in zip(labels, choices, strict=True)),
        ]
    )
    return serialize_solver_request(
        decision_procedure=prompt,
        question=canonical_question_payload(rendered),
    )
