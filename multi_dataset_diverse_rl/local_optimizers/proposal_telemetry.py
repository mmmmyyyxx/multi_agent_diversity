"""Opt-in observational telemetry for a preregistered local acceptance pilot.

Nothing here selects a GEPA parent, samples evidence, or accepts a candidate.
Only the supported stopper and a pre-evaluation hard budget guard stop work.
"""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
import re
from statistics import fmean, median
from typing import Any

from .gepa_adapter import compact_prompt_failed_checks, primary_prompt_rejection_category
from .gepa_callbacks import GEPALineageCallback
from ..versions import LOCAL_GEPA_TOKEN_EDIT_SIMILARITY_VERSION


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def edit_similarity(parent: str, proposal: str) -> dict[str, Any]:
    """v1: Unicode casefold, ASCII [a-z0-9]+ ordered tokens, autojunk off."""
    a, b = (re.findall(r"[a-z0-9]+", value.casefold()) for value in (parent, proposal))
    return {
        "similarity_version": LOCAL_GEPA_TOKEN_EDIT_SIMILARITY_VERSION,
        "similarity": SequenceMatcher(None, a, b, autojunk=False).ratio(),
        "parent_token_count": len(a),
        "proposal_token_count": len(b),
        "absolute_token_count_delta": abs(len(a) - len(b)),
    }


def wilson(accepted: int, evaluated: int) -> dict[str, Any]:
    if not 0 <= accepted <= evaluated:
        raise ValueError("invalid acceptance counts")
    if not evaluated:
        return {"numerator": accepted, "denominator": 0, "rate": None,
                "wilson_95": None, "interpretation": "NAIVE_IID_REFERENCE_ONLY"}
    z = 1.959963984540054
    p = accepted / evaluated
    scale = 1 + z * z / evaluated
    center = (p + z * z / (2 * evaluated)) / scale
    half = z * math.sqrt(p * (1 - p) / evaluated + z * z / (4 * evaluated**2)) / scale
    return {"numerator": accepted, "denominator": evaluated, "rate": p,
            "wilson_95": [max(0.0, center - half), min(1.0, center + half)],
            "interpretation": "NAIVE_IID_REFERENCE_ONLY"}


def descriptive_rate(accepted: int, evaluated: int) -> dict[str, Any]:
    if not 0 <= accepted <= evaluated:
        raise ValueError("invalid acceptance counts")
    return {"numerator": accepted, "denominator": evaluated,
            "rate": accepted / evaluated if evaluated else None,
            "estimand": "DESCRIPTIVE_NON_IID"}


def concentration(patterns: Counter) -> dict[str, Any]:
    count = sum(patterns.values())
    dominant = min(patterns, key=lambda key: (-patterns[key], key)) if count else None
    return {"parent_failure_count": count,
            "dominant_failure_pattern": list(dominant) if dominant else None,
            "dominant_failure_pattern_share": patterns[dominant] / count if count else None}


EVALUATION_FIELDS = {
    "evaluation_sequence_id", "candidate_hash", "example_ids", "binary_scores",
    "provider_called", "capture_traces", "evidence_group", "reasoning_lane",
}


def evaluation_schema() -> dict[str, Any]:
    def array(items):
        return {"type": "array", "items": items}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object", "additionalProperties": False,
            "required": sorted(EVALUATION_FIELDS), "properties": {
                "evaluation_sequence_id": {"type": "integer", "minimum": 0},
                "candidate_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "example_ids": array({"type": "string", "minLength": 1}),
                "binary_scores": array({"type": "integer", "enum": [0, 1]}),
                "provider_called": array({"type": "boolean"}),
                "capture_traces": {"type": "boolean"},
                "evidence_group": array({"enum": ["repair", "preservation", "team_hard", "general"]}),
                "reasoning_lane": array({"enum": ["direct_flip", "near_margin", "coverage", "fallback", "general"]}),
            }}


class MetricCeilingReached(RuntimeError):
    pass


class ProposalTelemetry(GEPALineageCallback):
    """Align exact adapter rows to official callbacks, including evolved parents."""

    def __init__(self, *args, search_examples, metric_ceiling: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_examples = tuple(search_examples)
        self.metric_ceiling = metric_ceiling
        self.evaluations: list[dict[str, Any]] = []
        self.selected: dict[int, dict[str, Any]] = {}
        self.pairs: dict[int, dict[str, dict[str, Any]]] = {}
        self.records: dict[int, dict[str, Any]] = {}
        self.depths = {0: 0}
        self.accepted: dict[int, int] = {}
        self.skipped = 0
        self.integrity_errors: list[str] = []
        self._selected_text = self.parent_prompt or ""
        self._seen_hashes: set[str] = set()
        self.metric_rows = 0
        self.finished_comparisons: set[int] = set()

    def _append(self, event_type: str, **values) -> None:
        try:
            super()._append(event_type, **values)
        except Exception:
            # Official GEPA swallows callback exceptions. Surface persistence
            # failure at the next stopper/evaluation boundary and final audit.
            self.integrity_errors.append("callback_persistence_failure")
            raise

    def before_evaluation(self, batch_size: int) -> None:
        if self.integrity_errors:
            raise ValueError("proposal telemetry integrity failure")
        if self.metric_rows + batch_size > self.metric_ceiling:
            raise MetricCeilingReached("PARENT_PROPOSAL_QUOTA_INCOMPLETE")

    def observe_evaluation(self, row: dict[str, Any]) -> None:
        row = {"evaluation_sequence_id": len(self.evaluations), **row}
        if set(row) != EVALUATION_FIELDS:
            raise ValueError("evaluation telemetry schema mismatch")
        length = len(row["example_ids"])
        if not length or any(len(row[key]) != length for key in (
            "binary_scores", "provider_called", "evidence_group", "reasoning_lane"
        )) or any(score not in (0, 1) for score in row["binary_scores"]):
            raise ValueError("evaluation telemetry vector mismatch")
        self.metric_rows += len(row["example_ids"])
        self.evaluations.append(row)
        if self.event_path is not None:
            path = self.event_path.with_suffix(".evaluations.jsonl")
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def on_candidate_selected(self, event):
        iteration = int(event["iteration"])
        self._selected_text = event["candidate"]["decision_procedure"]
        row = {"parent_candidate_index": int(event["candidate_idx"]),
               "parent_hash": text_hash(self._selected_text)}
        self.selected[iteration] = row
        self._append("candidate_selected", iteration=iteration, **row)

    def on_minibatch_sampled(self, event):
        # Official ListDataLoader emits positional integers, NOT example hashes.
        ids = [self.search_examples[value].example_id if isinstance(value, int)
               else str(value) for value in event["minibatch_ids"]]
        super().on_minibatch_sampled({**event, "minibatch_ids": ids})

    def on_proposal_end(self, event):
        self.proposal_count += 1
        iteration = int(event["iteration"])
        instructions = event.get("new_instructions", {})
        prompt = instructions.get("decision_procedure") if isinstance(instructions, dict) else None
        valid_mapping = isinstance(prompt, str) and set(instructions) == {"decision_procedure"}
        # Contract validation remains against the same frozen root as the adapter.
        checks = compact_prompt_failed_checks(
            prompt, parent_prompt=self.parent_prompt, examples=self.examples,
            max_chars=self.max_prompt_chars,
        ) if valid_mapping else ("invalid_component_mapping",)
        proposal_hash = text_hash(prompt) if valid_mapping else None
        selected = self.selected.get(iteration)
        if selected is None:
            self.integrity_errors.append("missing_selected_parent")
            selected = {"parent_candidate_index": 0, "parent_hash": None}
        row = {"iteration": iteration, **selected, "proposal_hash": proposal_hash,
               "changed": valid_mapping and prompt != self._selected_text,
               "duplicate": proposal_hash in self._seen_hashes,
               "contract_invalid": bool(checks), "failed_checks": list(checks),
               "primary_rejection_category": primary_prompt_rejection_category(checks),
               **edit_similarity(self._selected_text, prompt if valid_mapping else "")}
        if proposal_hash:
            self._seen_hashes.add(proposal_hash)
        self.records[iteration] = row
        self._append("proposal_end", **row)

    def on_evaluation_end(self, event):
        iteration = int(event["iteration"])
        if not self.evaluations:
            self.integrity_errors.append("missing_adapter_evaluation")
            return
        row = self.evaluations[-1]
        role = "child" if event.get("candidate_idx") is None else "parent"
        expected_hash = (self.records.get(iteration, {}).get("proposal_hash") if role == "child"
                         else self.selected.get(iteration, {}).get("parent_hash"))
        if (row["candidate_hash"] != expected_hash
                or row["example_ids"] != list(self.reflection_minibatches.get(iteration, ()))
                or row["binary_scores"] != list(event["scores"])
                or row["capture_traces"] != (role == "parent")):
            self.integrity_errors.append("evaluation_alignment_mismatch")
        if role == "parent" and event["candidate_idx"] != self.selected.get(iteration, {}).get("parent_candidate_index"):
            self.integrity_errors.append("selected_parent_index_mismatch")
        self.pairs.setdefault(iteration, {})[role] = row
        self._append("evaluation_link", iteration=iteration, role=role,
                     evaluation_sequence_id=row["evaluation_sequence_id"])

    def on_candidate_accepted(self, event):
        super().on_candidate_accepted(event)
        iteration = int(event["iteration"])
        self.finished_comparisons.add(iteration)
        parents = event["parent_ids"]
        selected = self.selected.get(iteration, {}).get("parent_candidate_index")
        if parents != [selected]:
            self.integrity_errors.append("accepted_parent_mismatch")
        depth = 1 + self.depths.get(selected, 0)
        self.depths[int(event["new_candidate_idx"])] = depth
        self.accepted[iteration] = depth
        pair = self.pairs.get(iteration, {})
        if "parent" not in pair or "child" not in pair:
            self.integrity_errors.append("accepted_without_paired_evaluation")
        elif sum(pair["child"]["binary_scores"]) <= sum(pair["parent"]["binary_scores"]):
            self.integrity_errors.append("acceptance_aggregate_mismatch")

    def on_candidate_rejected(self, event):
        # Persist a stable category, never free-form provider or engine text.
        super().on_candidate_rejected({**event, "reason": "no_strict_improvement"})
        self.finished_comparisons.add(int(event["iteration"]))
        pair = self.pairs.get(int(event["iteration"]), {})
        if ("parent" not in pair or "child" not in pair
                or sum(pair["parent"]["binary_scores"]) != event["old_score"]
                or sum(pair["child"]["binary_scores"]) != event["new_score"]
                or event["new_score"] > event["old_score"]):
            self.integrity_errors.append("rejection_aggregate_mismatch")

    def on_evaluation_skipped(self, event):
        self.skipped += 1
        super().on_evaluation_skipped({**event, "reason": "official_skip"})

    def scientific_summary(self, quota: int) -> dict[str, Any]:
        outcomes = []
        patterns: Counter = Counter()
        seen_failures: set[str] = set()
        reused = 0
        batches = []
        prior_batches: list[set[str]] = []
        for proposal_index, (iteration, record) in enumerate(sorted(self.records.items()), start=1):
            pair = self.pairs.get(iteration, {})
            before, after = pair.get("parent"), pair.get("child")
            sampled_ids = list(self.reflection_minibatches.get(iteration, ()))
            sampled_set = set(sampled_ids)
            prior_overlap = [len(sampled_set & prior) / len(sampled_set | prior)
                             for prior in prior_batches if sampled_set | prior]
            prior_batches.append(sampled_set)
            row = {**record, "proposal_index": proposal_index,
                   "solver_reached": after is not None and not record["contract_invalid"],
                   "gepa_accepted": iteration in self.accepted,
                   "lineage_depth": self.depths.get(record["parent_candidate_index"], 0) + 1,
                   "parent_lineage_depth": self.depths.get(record["parent_candidate_index"], 0),
                   "sampled_example_ids": sampled_ids,
                   "sampled_minibatch_max_jaccard_with_prior": max(prior_overlap) if prior_overlap else None,
                   "sampled_local_score_parent": None, "sampled_local_score_proposal": None,
                   "newly_fixed": None, "newly_broken": None, "delta_local_count": None,
                   "delta_local_rate": None, "preservation_loss": None,
                   "preservation_loss_count": None, "outcome_class": None,
                   "zero_delta_decomposition": None}
            for group in ("repair", "preservation", "team_hard"):
                row[group + "_newly_fixed"] = None
                row[group + "_newly_broken"] = None
            local_patterns: Counter = Counter()
            failures = set()
            if before:
                for ident, score, group, lane in zip(before["example_ids"], before["binary_scores"],
                                                    before["evidence_group"], before["reasoning_lane"], strict=True):
                    if score == 0:
                        local_patterns[(group, lane)] += 1
                        failures.add(ident)
                reused += bool(failures & seen_failures)
                seen_failures.update(failures)
            patterns.update(local_patterns)
            row.update(concentration(local_patterns))
            batches.append(set(row["sampled_example_ids"]))
            row["primary_denominator_eligible"] = bool(row["changed"] and row["solver_reached"])
            if before and after and row["solver_reached"]:
                if (before["example_ids"] != after["example_ids"]
                        or before["candidate_hash"] != row["parent_hash"]
                        or after["candidate_hash"] != row["proposal_hash"]):
                    raise ValueError("paired identity mismatch")
                old, new = before["binary_scores"], after["binary_scores"]
                fixes = [a == 0 and b == 1 for a, b in zip(old, new, strict=True)]
                breaks = [a == 1 and b == 0 for a, b in zip(old, new, strict=True)]
                fixed, broken = sum(fixes), sum(breaks)
                delta = fixed - broken
                outcome = "STRICT_POSITIVE" if delta > 0 else "STRICT_NEGATIVE" if delta < 0 else "EXACT_EQUAL"
                zero_kind = None
                if delta == 0:
                    zero_kind = ("BEHAVIORAL_NO_OP" if fixed == broken == 0
                                 else "REPAIR_PRESERVATION_CANCELLATION")
                row.update(sampled_local_score_parent=sum(old), sampled_local_score_proposal=sum(new),
                           newly_fixed=fixed, newly_broken=broken,
                           delta_local_count=delta,
                           delta_local_rate=delta / len(old), outcome_class=outcome,
                           zero_delta_decomposition=zero_kind,
                           parent_evaluation_sequence_id=before["evaluation_sequence_id"],
                           child_evaluation_sequence_id=after["evaluation_sequence_id"])
                for group in ("repair", "preservation", "team_hard"):
                    indices = [i for i, value in enumerate(before["evidence_group"]) if value == group]
                    row[group + "_newly_fixed"] = sum(fixes[i] for i in indices) if indices else None
                    row[group + "_newly_broken"] = sum(breaks[i] for i in indices) if indices else None
                row["preservation_loss"] = row["preservation_newly_broken"]
                row["preservation_loss_count"] = row["preservation_loss"]
                if iteration in self.finished_comparisons and row["gepa_accepted"] != (row["delta_local_count"] > 0):
                    self.integrity_errors.append("official_acceptance_delta_mismatch")
            outcomes.append(row)
        if self.integrity_errors:
            raise ValueError("proposal telemetry integrity: " + ",".join(sorted(set(self.integrity_errors))))
        overlap = [len(a & b) / len(a | b) for i, a in enumerate(batches) for b in batches[i + 1:]]
        eligible = [row for row in outcomes if row["primary_denominator_eligible"]]
        numerator = sum(row["gepa_accepted"] for row in eligible)
        funnel = {"proposal_opportunities_requested": quota, "proposal_attempts_observed": len(outcomes),
                  "changed": sum(row["changed"] for row in outcomes),
                  "unchanged": sum(not row["changed"] for row in outcomes),
                  "duplicate": sum(row["duplicate"] for row in outcomes),
                  "contract_invalid": sum(row["contract_invalid"] for row in outcomes),
                  "solver_reached": sum(row["solver_reached"] for row in outcomes),
                  "strict_positive_sampled_delta": sum(row["delta_local_count"] > 0 for row in eligible),
                  "strict_equal_sampled_delta": sum(row["delta_local_count"] == 0 for row in eligible),
                  "strict_negative_sampled_delta": sum(row["delta_local_count"] < 0 for row in eligible),
                  "gepa_accepted": len(self.accepted),
                  "full_local_validation_evaluated": sum(e["event_type"] == "valset_evaluated" and e["candidate_index"] != 0 for e in self.events)}
        unfinished = len(set(self.records) - self.finished_comparisons)
        similarities = [row["similarity"] for row in outcomes]
        evaluated = [row for row in outcomes if row["primary_denominator_eligible"]]
        return {"funnel": funnel, "primary": descriptive_rate(numerator, len(eligible)),
                "naive_iid_reference": wilson(numerator, len(eligible)), "proposals": outcomes,
                "quota_status": "COMPLETE" if len(outcomes) == quota and not unfinished else "PARENT_PROPOSAL_QUOTA_INCOMPLETE",
                "unfinished_comparison_count": unfinished,
                "proposal_shortfall": max(0, quota - len(outcomes)), "skipped_iterations": self.skipped,
                "metric_calls": self.metric_rows, "at_least_one_accepted": bool(self.accepted),
                "accepted_generations": list(self.accepted.values()), "candidate_lineage_depth": dict(self.depths),
                "parent_level_summary": {
                    "proposal_attempts": len(outcomes),
                    "contract_valid": sum(not row["contract_invalid"] for row in outcomes),
                    "solver_reached": sum(row["solver_reached"] for row in outcomes),
                    "positive": sum(row["outcome_class"] == "STRICT_POSITIVE" for row in evaluated),
                    "equal": sum(row["outcome_class"] == "EXACT_EQUAL" for row in evaluated),
                    "negative": sum(row["outcome_class"] == "STRICT_NEGATIVE" for row in evaluated),
                    "accepted": numerator,
                    "total_newly_fixed": sum(row["newly_fixed"] or 0 for row in evaluated),
                    "total_newly_broken": sum(row["newly_broken"] or 0 for row in evaluated),
                    "preservation_losses": sum(row["preservation_loss_count"] or 0 for row in evaluated),
                    "behavioral_no_op": sum(row["zero_delta_decomposition"] == "BEHAVIORAL_NO_OP" for row in evaluated),
                    "repair_preservation_cancellation": sum(row["zero_delta_decomposition"] == "REPAIR_PRESERVATION_CANCELLATION" for row in evaluated),
                    "mean_edit_similarity": fmean(similarities) if similarities else None,
                    "median_edit_similarity": median(similarities) if similarities else None,
                    "unique_failure_patterns": len(patterns),
                    "dominant_failure_pattern_concentration": concentration(patterns)["dominant_failure_pattern_share"],
                    "max_lineage_depth": max((row["lineage_depth"] for row in outcomes), default=0),
                    "accepted_generation_count": len(self.accepted),
                },
                "failure_concentration": concentration(patterns),
                "pairwise_minibatch_jaccard": overlap,
                "fraction_proposals_reusing_previous_failure": reused / len(outcomes) if outcomes else None,
                "unique_failure_examples_seen": len(seen_failures)}

    def proposal_diagnostics(self):
        # Preserve the host optimizer's legacy payload interface. Pilot analysis
        # exclusively consumes scientific_summary, not legacy heuristic fields.
        result = super().proposal_diagnostics()
        result.update(proposal_attempts=self.proposal_count,
                      proposal_hashes=[r["proposal_hash"] for r in self.records.values()],
                      proposal_changed=sum(r["changed"] for r in self.records.values()),
                      proposal_unchanged=sum(not r["changed"] for r in self.records.values()),
                      proposal_duplicate=sum(r["duplicate"] for r in self.records.values()),
                      proposal_contract_invalid=sum(r["contract_invalid"] for r in self.records.values()))
        return result


class ExactProposalStopper:
    """Structural StopperProtocol through the official stop_callbacks seam."""
    def __init__(self, callback: ProposalTelemetry, quota: int, skip_allowance: int):
        if quota <= 0 or skip_allowance <= 0:
            raise ValueError("positive frozen quota and skip allowance required")
        self.callback, self.quota, self.skip_allowance = callback, quota, skip_allowance

    def __call__(self, _state) -> bool:
        if self.callback.integrity_errors:
            raise ValueError("proposal telemetry integrity failure")
        return self.callback.proposal_count >= self.quota or self.callback.skipped >= self.skip_allowance


def cost_envelope(quota: int, validation_size: int, minibatch_size: int, skip_allowance: int):
    if min(quota, validation_size, minibatch_size, skip_allowance) <= 0:
        raise ValueError("positive budget dimensions required")
    normal = validation_size + quota * (2 * minibatch_size + validation_size)
    with_skips = normal + skip_allowance * minibatch_size
    return {"seed_local_validation": validation_size,
            "parent_child_sampled": quota * 2 * minibatch_size,
            "accepted_full_local_validation": quota * validation_size,
            "bounded_skip_evaluations": skip_allowance * minibatch_size,
            "normal_max_metric_calls": normal,
            "max_metric_calls_with_skips": with_skips,
            "metric_ceiling": with_skips + 1,
            "reflection_logical_calls": quota}
