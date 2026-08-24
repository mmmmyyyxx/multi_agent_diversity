from __future__ import annotations

import json
from typing import Any, Sequence

from .utils import extract_json_obj, normalize_prompt_text


ONLINE_COMPATIBILITY_REPAIR_VERSION = (
    "candidate_specific_counterfactual_compatibility_repair_v1"
)
EXTENDED_TRAIN_VOTE_LOSS_TRIGGER_VERSION = (
    "common_safe_train_vote_loss_extended_trigger_v1"
)
LOSS_BLIND_GENERIC_REVISION_VERSION = "loss_blind_generic_revision_v1"
LOSS_BLIND_GENERIC_REVISION_SYSTEM_PROMPT = (
    "Revise one candidate member prompt. Return strict JSON only with exactly "
    "one field: repaired_prompt. Do not quote or memorize examples. Do not add "
    "answer lookup rules, hashes, labels, or the immutable output contract."
)
REPAIR_MAX_TOKENS = 3000
REPAIR_SYSTEM_PROMPT = (
    "You repair one already-targeted member prompt. Return strict JSON only "
    "with exactly one field: repaired_prompt. Do not quote, identify, or "
    "memorize examples. Do not add answer lookup rules, hashes, labels, or "
    "the immutable output contract."
)
REPAIR_INSTRUCTION = """The SOURCE CANDIDATE successfully corrected the assigned responsibility examples below, but introduced new failures relative to the parent prompt.

Revise the SOURCE CANDIDATE, not the parent from scratch.
Primary requirement: retain the responsibility-specific behavior that produced the successful repairs.
Secondary requirement: remove, narrow, or revert only broader behavioral changes likely to have caused the observed competence losses.
Do not change the assigned responsibility objective. Do not memorize question text, option letters, gold answers, or individual examples. Do not add a universal rule merely to satisfy loss examples. Prefer restoring parent behavior outside the targeted responsibility mechanism.
Return one complete revised member prompt as strict JSON: {"repaired_prompt":"..."}."""
LOSS_BLIND_GENERIC_REVISION_INSTRUCTION = """Revise the SOURCE CANDIDATE once to improve its global compatibility with the PARENT PROMPT while retaining any useful improvement.

You are not given candidate-specific loss examples, responsibility examples, identities, labels, hashes, or outcome counts. Do not infer or invent them. Revise the source candidate, not the parent from scratch. Prefer a generally applicable decision procedure and avoid memorizing examples or answer patterns.
Return one complete revised member prompt as strict JSON: {"repaired_prompt":"..."}."""

COLLATERAL_REJECTION_REASONS = frozenset({
    "target_regression",
    "team_vote_regression",
    "terminal_invalid_regression",
})


def repair_eligible(
    *, responsibility_gain_count: int, rejection_reasons: Sequence[str],
    loss_evidence_count: int,
) -> bool:
    return bool(
        int(responsibility_gain_count) > 0
        and int(loss_evidence_count) > 0
        and COLLATERAL_REJECTION_REASONS.intersection(map(str, rejection_reasons))
    )


def extended_repair_eligible(
    *,
    responsibility_gain_count: int,
    rejection_reasons: Sequence[str],
    loss_evidence_count: int,
    source_common_safe: bool,
    source_vote_loss_count: int,
) -> bool:
    """Experimental M2F trigger extension used by the frozen V18 pilot.

    The original M2F trigger remains unchanged.  This extension adds exactly
    one alternative trigger for a Common-Safe source with observed train Vote
    loss; it does not change repair generation, evaluation, or write-back.
    """

    return bool(
        repair_eligible(
            responsibility_gain_count=responsibility_gain_count,
            rejection_reasons=rejection_reasons,
            loss_evidence_count=loss_evidence_count,
        )
        or (bool(source_common_safe) and int(source_vote_loss_count) > 0)
    )


def build_repair_request(
    *, parent_prompt: str, source_candidate_prompt: str,
    repair_evidence: Sequence[dict[str, Any]],
    loss_evidence: Sequence[dict[str, Any]], numeric_summary: dict[str, int],
) -> str:
    payload = {
        "parent_member_prompt": parent_prompt,
        "source_m20_candidate_prompt": source_candidate_prompt,
        "successful_assigned_responsibility_repairs": list(repair_evidence),
        "candidate_specific_competence_losses": list(loss_evidence),
        "numeric_summary": dict(numeric_summary),
    }
    return REPAIR_INSTRUCTION + "\n\nRepairInput:\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    )


def build_loss_blind_generic_revision_request(
    *, parent_prompt: str, source_candidate_prompt: str,
) -> str:
    return LOSS_BLIND_GENERIC_REVISION_INSTRUCTION + "\n\nRevisionInput:\n" + json.dumps(
        {
            "parent_member_prompt": parent_prompt,
            "source_candidate_prompt": source_candidate_prompt,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def parse_repair_output(
    text: str, *, source_candidate_prompt: str,
    supplied_evidence: Sequence[dict[str, Any]],
) -> str:
    payload = extract_json_obj(text)
    if not isinstance(payload, dict) or set(payload) != {"repaired_prompt"}:
        raise ValueError("repair response must contain exactly repaired_prompt")
    raw = payload["repaired_prompt"]
    prompt = normalize_prompt_text(raw if isinstance(raw, str) else "")
    if not prompt:
        raise ValueError("repair prompt is empty")
    if prompt == normalize_prompt_text(source_candidate_prompt):
        raise ValueError("repair prompt is unchanged")
    lowered = prompt.lower()
    forbidden = (
        "final_answer:", "question_hash", "gold answer", "answer choice a",
        "answer choice b", "answer choice c", "answer choice d",
    )
    if any(token in lowered for token in forbidden):
        raise ValueError("repair prompt contains forbidden answer or protocol material")
    compact = " ".join(lowered.split())
    for row in supplied_evidence:
        question = " ".join(str(row.get("question", "")).lower().split())
        if len(question) >= 32 and question in compact:
            raise ValueError("repair prompt memorizes supplied question")
    return prompt
