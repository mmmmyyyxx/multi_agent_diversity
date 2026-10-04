"""Observational replay of private proposals; never used for candidate admission.

Semantic labels are an owner's explicit, hash-bound annotation, not an inferred
ground truth or a replacement validator. Counterfactual edits stay in memory.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Any, Mapping, Sequence

from . import mutable_prompt_contract as mutable
from ..local_optimizers.gepa_adapter import compact_prompt_failed_checks
from ..local_optimizers.gepa_runtime import import_frozen_gepa
from ..local_optimizers.schemas import LocalEvidenceExample

import_frozen_gepa()
from gepa.strategies.instruction_proposal import InstructionProposalSignature  # noqa: E402


SEMANTIC_CLASSES = {
    "OUTPUT_PRESENTATION_CONTROL", "ANSWER_REPRESENTATION_GUIDANCE", "BENIGN_REASONING",
}


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_proposal(raw: str) -> str:
    return InstructionProposalSignature.output_extractor(raw)["new_instruction"]


def marker_spans(prompt: str) -> list[dict[str, Any]]:
    normalized = mutable._normalized_for_contract_check(prompt)
    patterns = [
        ("final_answer_phrase", mutable._FINAL_ANSWER_MARKER),
        *[("interface_literal", p) for p in mutable._COPIED_INTERFACE_MARKERS],
        *[("formatting_directive", p) for p in mutable._COPIED_FORMATTING_DIRECTIVES],
    ]
    return sorted([
        {"category": category, "start": match.start(), "end": match.end(),
         "span_sha256": text_hash(match.group())}
        for category, pattern in patterns for match in pattern.finditer(normalized)
    ], key=lambda row: (row["start"], row["category"]))


def verify_annotation(prompt: str, annotation: Mapping[str, Any]) -> None:
    if text_hash(prompt) != annotation["proposal_sha256"]:
        raise ValueError("semantic annotation proposal hash mismatch")
    if annotation["semantic_class"] not in SEMANTIC_CLASSES:
        raise ValueError("unknown semantic annotation class")
    normalized = mutable._normalized_for_contract_check(prompt)
    spans = annotation["evidence_spans"]
    if not spans:
        raise ValueError("semantic annotation requires evidence spans")
    for span in spans:
        start, end = span["start"], span["end"]
        if not (0 <= start < end <= len(normalized)):
            raise ValueError("semantic annotation span out of bounds")
        if text_hash(normalized[start:end]) != span["span_sha256"]:
            raise ValueError("semantic annotation span hash mismatch")


def replay_boundary(prompt: str, parent: str, examples: Sequence[LocalEvidenceExample]) -> dict:
    checks = compact_prompt_failed_checks(
        prompt, parent_prompt=parent, examples=examples, max_chars=3000,
    )
    # Diagnostic intervention only: remove the lexical phrase, retaining all
    # other bytes, checks and length limits. This is not a semantic validator.
    masked = re.sub(mutable._FINAL_ANSWER_MARKER.pattern, "derived result", prompt, flags=re.I)
    masked_checks = compact_prompt_failed_checks(
        masked, parent_prompt=parent, examples=examples, max_chars=3000,
    )
    return {"failed_checks": list(checks), "marker_spans": marker_spans(prompt),
            "phrase_mask_failed_checks": list(masked_checks),
            "phrase_mask_boundary_accepts": not masked_checks}


def paired_identity_audit(paired: Sequence[Mapping], expected_count: int) -> dict:
    """Read the persisted validation stage names; never reopen held-out inputs."""
    if any(r["team"] not in {"validation_initial", "validation_final"} for r in paired):
        raise ValueError("unknown paired score stage")
    initial = {r["example_id"]: r for r in paired if r["team"] == "validation_initial"}
    final = {r["example_id"]: r for r in paired if r["team"] == "validation_final"}
    if (len(paired) != 2 * expected_count or len(initial) != expected_count
            or set(initial) != set(final)):
        raise ValueError("paired score inventory mismatch")
    fields = ("member_correct", "member_valid", "oracle_correct", "vote_correct", "request_output_hashes")
    if any(any(initial[k][f] != final[k][f] for f in fields) for k in initial):
        raise ValueError("identical-team paired realizations differ")
    return {"paired_realizations_equal": True,
            "vote_correct": sum(r["vote_correct"] for r in initial.values()),
            "oracle_correct": sum(r["oracle_correct"] for r in initial.values()),
            "oracle_correct_vote_wrong": sum(r["oracle_correct"] and not r["vote_correct"] for r in initial.values())}


def audit_proposals(
    trace: Sequence[Mapping], trajectory: Sequence[Mapping], ledger: Sequence[Mapping],
    examples_by_id: Mapping[str, LocalEvidenceExample], annotations: Sequence[Mapping],
    broker: Any,
) -> dict:
    """Join physical provider responses, temporal cache reuse and GEPA callbacks."""
    physical = [row for row in trace if row["role"] == "reflection"]
    if len({r["request_sha256"] for r in physical}) != len(physical):
        raise ValueError("physical reflection request identities are not unique")
    by_hash, by_key = {}, {}
    for row in physical:
        request, key = broker._request_identity(
            role="reflection", split="optimize", messages=row["request"]["messages"],
        )
        if request != row["request"] or key != row["request_sha256"]:
            raise ValueError("physical reflection request identity mismatch")
        prompt = extract_proposal(row["response"]["text"])
        digest = text_hash(prompt)
        if digest in by_hash:
            raise ValueError("physical responses have duplicate proposal text")
        by_hash[digest] = row, prompt
        by_key[key] = digest
    labeled = {row["proposal_sha256"]: row for row in annotations}
    if len(labeled) != len(annotations) or set(labeled) != set(by_hash):
        raise ValueError("annotation coverage differs from physical proposal inventory")

    # Check cache replay against earlier physical successes in the same role/split.
    successful, uses, cache_counts = set(), [], Counter()
    for row in ledger:
        if row.get("role") != "reflection":
            continue
        key = row.get("request_sha256")
        if row.get("split") != "optimize":
            raise ValueError("reflection accessed a held-out split")
        if row["kind"] == "SUCCESS":
            if key not in by_key or key in successful:
                raise ValueError("unexpected reflection success")
            raw = by_hash[by_key[key]][0]["response"]["text"]
            if row["response_sha256"] != text_hash(raw):
                raise ValueError("reflection response ledger hash mismatch")
            successful.add(key)
            uses.append(by_key[key])
        elif row["kind"] == "CACHE_HIT":
            if key not in successful:
                raise ValueError("cache hit preceded its physical realization")
            cache_counts[by_key[key]] += 1
            uses.append(by_key[key])
    if successful != set(by_key):
        raise ValueError("physical response inventory and ledger successes differ")

    opportunities = {r["opportunity"]["opportunity_id"]: r["opportunity"]
                     for r in trajectory if r["stage"] == "OPPORTUNITY"}
    evaluations = [r for r in trajectory if r["stage"] == "EVALUATION"]
    if len(opportunities) != len(evaluations):
        raise ValueError("opportunity/evaluation inventory mismatch")
    logical_rows, member_by_hash, parent_by_hash, replayed_by_hash = [], {}, {}, {}
    flat_hashes = []
    for evaluation in evaluations:
        op = opportunities[evaluation["opportunity_id"]]
        evidence_ids = {x["example_id"] for role in ("mutation_evidence", "search_validation_evidence")
                        for x in op["evidence"][role]}
        examples = tuple(examples_by_id[x] for x in sorted(evidence_ids))
        callbacks = [r for r in evaluation["search"]["search_state"]["callback_events"]
                     if r["event_type"] == "proposal_end"]
        for recorded in callbacks:
            digest = recorded["proposal_hash"]
            _, prompt = by_hash[digest]
            boundary = replay_boundary(prompt, op["parent_prompt"], examples)
            if boundary["failed_checks"] != recorded["failed_checks"]:
                raise ValueError("callback full failed-check replay mismatch")
            if digest in member_by_hash and (
                member_by_hash[digest] != op["target_member"]
                or parent_by_hash[digest] != text_hash(op["parent_prompt"])
            ):
                raise ValueError("proposal reused under a different parent or member")
            member_by_hash[digest] = op["target_member"]
            parent_by_hash[digest] = text_hash(op["parent_prompt"])
            replayed_by_hash[digest] = boundary
            logical_rows.append({"proposal_sha256": digest, "target_member": op["target_member"],
                                 "parent_state_sha256": text_hash(op["parent_state_id"]),
                                 "iteration": recorded["iteration"],
                                 "failed_checks": recorded["failed_checks"]})
            flat_hashes.append(digest)
        if evaluation["search"]["candidates"] or evaluation["evaluated"]:
            raise ValueError("this audit requires the recorded zero-candidate endpoint")
    if flat_hashes != uses:
        raise ValueError("logical proposal order differs from provider/cache ledger")
    if set(flat_hashes) != set(by_hash):
        raise ValueError("callback inventory differs from physical proposals")

    rows = []
    for index, physical_row in enumerate(physical, 1):
        digest = by_key[physical_row["request_sha256"]]
        _, prompt = by_hash[digest]
        annotation = labeled[digest]
        verify_annotation(prompt, annotation)
        response = physical_row["response"]
        diagnostics = response["optimizer_generation_diagnostics"]
        body = physical_row["request"]
        normalized = mutable._normalized_for_contract_check(prompt)
        rows.append({
            "physical_index": index, "target_member": member_by_hash[digest],
            "proposal_sha256": digest, "request_sha256": physical_row["request_sha256"],
            "response_sha256": text_hash(response["text"]), "characters": len(prompt),
            "logical_occurrences": flat_hashes.count(digest), "cache_replays": cache_counts[digest],
            "parent_component_sha256": parent_by_hash[digest],
            "finish_reason": response["finish_reason"], "output_tokens": response["output_tokens"],
            "thinking_false_dispatched": body.get("extra_body", {}).get("enable_thinking") is False,
            "nonthinking_wire_confirmed": diagnostics["nonthinking_wire_confirmed"],
            "reasoning_tokens": diagnostics["reasoning_tokens"],
            "complete_single_fence": diagnostics["diagnostics"]["complete_single_fence"],
            "repetition_pathology": diagnostics["diagnostics"]["repetition_pathology"],
            "request_final_answer_phrase_count": sum(
                len(list(mutable._FINAL_ANSWER_MARKER.finditer(
                    mutable._normalized_for_contract_check(m["content"])))) for m in body["messages"]),
            "literal_wire_marker_present": bool(re.search(r"(?i)\bFINAL_ANSWER\s*:", prompt)),
            "copied_interface_literal_present": any(p.search(normalized) for p in mutable._COPIED_INTERFACE_MARKERS),
            "semantic_class": annotation["semantic_class"],
            "owner_annotation": annotation,
            **replayed_by_hash[digest],
        })
    semantic_counts = Counter(r["semantic_class"] for r in rows)
    # "Semantic admissible" below applies the proposer's stated ban on answer
    # formatting plus every remaining production check; it grants no admission.
    hypothetical = [r for r in rows if r["semantic_class"] == "BENIGN_REASONING"
                    and not set(r["failed_checks"]) - {"forbidden_final_answer_marker"}]
    accepted_masked = [r for r in rows if r["phrase_mask_boundary_accepts"]]
    return {
        "identity": "MATH_PROPOSAL_SEMANTIC_POSTMORTEM_V1", "provider_calls": 0,
        "candidate_admission_changed": False, "semantic_labels_source": "OWNER_ANNOTATION_NOT_INDEPENDENT_GROUND_TRUTH",
        "physical_proposals": len(rows), "logical_proposals": len(logical_rows),
        "cache_replays": sum(cache_counts.values()), "opportunities": len(opportunities),
        "opportunity_counts_by_member": [sum(op["target_member"] == i for op in opportunities.values()) for i in range(5)],
        "distinct_parent_states": len({op["parent_state_id"] for op in opportunities.values()}),
        "physical_failed_check_counts": dict(Counter(c for r in rows for c in r["failed_checks"])),
        "logical_failed_check_counts": dict(Counter(c for r in logical_rows for c in r["failed_checks"])),
        "semantic_class_counts": dict(semantic_counts),
        "strict_proposer_semantic_counterfactual_admissible": len(hypothetical),
        "phrase_mask_boundary_accepts": len(accepted_masked),
        "phrase_mask_accepts_owner_semantic_invalid": sum(r["semantic_class"] != "BENIGN_REASONING" for r in accepted_masked),
        "rows": rows, "logical_rows": logical_rows,
        "candidate_exported": 0, "candidate_team_evaluated": 0,
        "interpretation": "ZERO_INTERVENTION_NO_ADMISSIBLE_CANDIDATE",
        "downstream_efficacy": "NOT_IDENTIFIED",
    }
