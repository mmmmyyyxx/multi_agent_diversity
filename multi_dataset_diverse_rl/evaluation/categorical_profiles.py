"""Sanitized, replayable evidence for full team evaluations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from .endpoint_identifiability import (
    endpoint_structural_identifiability,
    plurality_outcome,
)
from .fixed_probe import ProbeExample
from .prompt_question import PromptAnswer


SCHEMA_VERSION = "sanitized_team_categorical_profile_v1"
STATE_SCHEMA_VERSION = "endpoint_identifiability_state_v1"
INVALID_CHOICE = "INVALID"


def _sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def legal_option_labels(question: str) -> tuple[str, ...]:
    """Extract only option labels; option text never enters an artifact."""
    parenthesized = re.findall(
        r"(?:^|\n)\s*\(([A-Z])\)\s+", str(question), flags=re.MULTILINE
    )
    labels = parenthesized or re.findall(
        r"(?:^|\n)\s*([A-Z])[\).]\s+", str(question), flags=re.MULTILINE
    )
    return tuple(dict.fromkeys(label.upper() for label in labels))


def _choice_category(
    value: str,
    *,
    normalize_answer: Callable[[str], str],
    legal_labels: Sequence[str],
) -> str:
    normalized = str(normalize_answer(value)).strip()
    if normalized in legal_labels:
        return normalized
    return f"VALUE_SHA256:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def categorical_choice(
    answer: PromptAnswer,
    *,
    question: str,
    normalize_answer: Callable[[str], str],
) -> str:
    if not answer.valid:
        return INVALID_CHOICE
    labels = tuple(
        str(normalize_answer(label)).strip() for label in legal_option_labels(question)
    )
    return _choice_category(
        answer.answer,
        normalize_answer=normalize_answer,
        legal_labels=labels,
    )


def sanitized_categorical_profile(
    *,
    examples: Sequence[ProbeExample],
    profile: Sequence[PromptAnswer],
    normalize_answer: Callable[[str], str],
    match_answer: Callable[[str, str], bool],
    parent_team_hash: str,
    update_index: int,
    target_member: int,
    candidate_hash: str,
    candidate_id: str,
    evaluation_stage: str,
) -> dict[str, Any]:
    """Build a prompt/question/answer-free profile persisted after a Full evaluation."""
    if len(examples) != len(profile):
        raise ValueError("categorical profile length differs from fixed probe")
    rows = []
    for example, answer in zip(examples, profile, strict=True):
        rows.append(
            {
                "example_id": example.question_hash,
                "normalized_choice": categorical_choice(
                    answer,
                    question=example.question,
                    normalize_answer=normalize_answer,
                ),
                "correct": bool(
                    answer.valid and match_answer(answer.answer, example.gold_answer)
                ),
                "terminal_invalid": bool(answer.terminal_invalid),
                "candidate_hash": candidate_hash,
            }
        )
    profile_sha256 = _sha256(rows)
    identity = _sha256(
        {
            "parent_team_hash": parent_team_hash,
            "update_index": int(update_index),
            "target_member": int(target_member),
            "candidate_hash": candidate_hash,
            "candidate_id": candidate_id,
            "evaluation_stage": evaluation_stage,
            "profile_sha256": profile_sha256,
        }
    )
    return {
        "artifact_schema_version": SCHEMA_VERSION,
        "evaluation_identity": identity,
        "profile_sha256": profile_sha256,
        "parent_team_hash": parent_team_hash,
        "update_index": int(update_index),
        "target_member": int(target_member),
        "candidate_hash": candidate_hash,
        "candidate_id": candidate_id,
        "evaluation_stage": evaluation_stage,
        "row_count": len(rows),
        "rows": rows,
    }


def endpoint_identifiability_snapshot(
    *,
    examples: Sequence[ProbeExample],
    profiles: Sequence[Sequence[PromptAnswer]],
    normalize_answer: Callable[[str], str],
    team_prompt_state_hash: str,
    update_index: int,
    trigger: str,
    committed_target_member: int | None = None,
    committed_candidate_hash: str | None = None,
) -> dict[str, Any]:
    """Compute P_0..P_4 and gold-correctable subsets from one realized state."""
    if len(profiles) != 5:
        raise ValueError("endpoint snapshot requires exactly five member profiles")
    if any(len(profile) != len(examples) for profile in profiles):
        raise ValueError("endpoint snapshot profile length differs from fixed probe")

    categorical_profiles: list[list[str | None]] = [[] for _ in range(5)]
    gold_categories: list[str] = []
    legal_domains: list[tuple[str | None, ...]] = []
    domain_sources: list[str] = []
    for row_index, example in enumerate(examples):
        labels = tuple(
            str(normalize_answer(label)).strip()
            for label in legal_option_labels(example.question)
        )
        for member in range(5):
            choice = categorical_choice(
                profiles[member][row_index],
                question=example.question,
                normalize_answer=normalize_answer,
            )
            categorical_profiles[member].append(
                None if choice == INVALID_CHOICE else choice
            )
        gold = _choice_category(
            example.gold_answer,
            normalize_answer=normalize_answer,
            legal_labels=labels,
        )
        gold_categories.append(gold)
        if labels:
            legal_domains.append(tuple([*labels, None]))
            domain_sources.append("parsed_option_labels_plus_invalid")
        else:
            observed = [
                value
                for value in (profile[row_index] for profile in categorical_profiles)
                if value is not None
            ]
            legal_domains.append(
                tuple(
                    dict.fromkeys(
                        [
                            *observed,
                            gold,
                            "UNOBSERVED_VALID_CATEGORY",
                            None,
                        ]
                    )
                )
            )
            domain_sources.append("outcome_complete_categorical_domain")

    audit = endpoint_structural_identifiability(
        categorical_profiles,
        legal_target_outputs_by_row=legal_domains,
    )
    for target in range(5):
        correctable_ids: list[str] = []
        for row_index, example in enumerate(examples):
            current = plurality_outcome(
                [categorical_profiles[member][row_index] for member in range(5)]
            )
            proposed = [categorical_profiles[member][row_index] for member in range(5)]
            proposed[target] = gold_categories[row_index]
            if (
                current != gold_categories[row_index]
                and plurality_outcome(proposed) == gold_categories[row_index]
            ):
                correctable_ids.append(example.question_hash)
        member = audit["by_member"][str(target)]
        capable_indices = member.pop("pivotal_capable_row_indices")
        member["pivotal_capable_example_ids"] = [
            examples[index].question_hash for index in capable_indices
        ]
        member["pivotal_correctable_count"] = len(correctable_ids)
        member["pivotal_correctable_example_ids"] = correctable_ids

    profile_state_sha256 = _sha256(
        {
            "example_ids": [example.question_hash for example in examples],
            "profiles": categorical_profiles,
        }
    )
    state_identity = _sha256(
        {
            "team_prompt_state_hash": team_prompt_state_hash,
            "profile_state_sha256": profile_state_sha256,
            "update_index": int(update_index),
            "trigger": trigger,
            "committed_target_member": committed_target_member,
            "committed_candidate_hash": committed_candidate_hash,
        }
    )
    source_counts = {
        source: domain_sources.count(source) for source in sorted(set(domain_sources))
    }
    p_i = {
        str(member): int(audit["by_member"][str(member)]["pivotal_capable_count"])
        for member in range(5)
    }
    return {
        "artifact_schema_version": STATE_SCHEMA_VERSION,
        "state_identity": state_identity,
        "team_prompt_state_hash": team_prompt_state_hash,
        "profile_state_sha256": profile_state_sha256,
        "update_index": int(update_index),
        "trigger": trigger,
        "committed_target_member": committed_target_member,
        "committed_candidate_hash": committed_candidate_hash,
        "legal_output_domain_source_counts": source_counts,
        "p_i": p_i,
        **audit,
        "immediate_vote_change_possible": bool(
            audit["total_member_row_opportunities"] > 0
        ),
    }
