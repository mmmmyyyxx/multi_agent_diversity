"""Observational GEPA callbacks for auditable lifecycle and lineage events."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .gepa_adapter import (
    compact_prompt_failed_checks,
    primary_prompt_rejection_category,
)
from .schemas import LocalEvidenceExample
from ..persistence.durable_io import append_jsonl


def _hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GEPALineageCallback:
    """Capture only hashes, ids, scores, and lifecycle facts—not raw prompts."""

    def __init__(
        self,
        event_path: Path | None = None,
        *,
        parent_prompt: str | None = None,
        examples: Sequence[LocalEvidenceExample] = (),
        max_prompt_chars: int = 3000,
    ) -> None:
        self.events: list[dict[str, Any]] = []
        self.reflection_minibatches: dict[int, tuple[str, ...]] = {}
        self.proposal_count = 0
        self.event_path = event_path
        self.parent_prompt = parent_prompt
        self.examples = tuple(examples)
        self.max_prompt_chars = max_prompt_chars
        self._proposal_records: list[dict[str, Any]] = []
        self._seen_changed_hashes: set[str] = set()
        self._proposal_token_sets: list[set[str]] = []
        self._parent_scores_by_iteration: dict[int, tuple[float, ...]] = {}
        self._proposal_scores_by_iteration: dict[int, tuple[float, ...]] = {}
        self._accepted_iterations: set[int] = set()

    def _append(self, event_type: str, **values: Any) -> None:
        row = {"event_index": len(self.events), "event_type": event_type, **values}
        self.events.append(row)
        if self.event_path is not None:
            append_jsonl(self.event_path, row)

    def on_optimization_start(self, event: dict[str, Any]) -> None:
        self._append(
            "optimization_start",
            trainset_size=int(event["trainset_size"]),
            valset_size=int(event["valset_size"]),
            seed_candidate_hash=_hash(event["seed_candidate"]),
        )

    def on_minibatch_sampled(self, event: dict[str, Any]) -> None:
        iteration = int(event["iteration"])
        ids = tuple(str(value) for value in event["minibatch_ids"])
        self.reflection_minibatches[iteration] = ids
        self._append("minibatch_sampled", iteration=iteration, minibatch_ids=list(ids))

    def on_candidate_selected(self, event: dict[str, Any]) -> None:
        self._append(
            "candidate_selected",
            iteration=int(event["iteration"]),
            candidate_index=int(event["candidate_idx"]),
            candidate_hash=_hash(event["candidate"]),
            score=float(event["score"]),
        )

    def on_proposal_end(self, event: dict[str, Any]) -> None:
        self.proposal_count += 1
        instructions = event.get("new_instructions")
        prompt = (
            instructions.get("decision_procedure")
            if isinstance(instructions, dict)
            and set(instructions) == {"decision_procedure"}
            and isinstance(instructions.get("decision_procedure"), str)
            else None
        )
        failed_checks: tuple[str, ...]
        if prompt is None or self.parent_prompt is None:
            proposal_hash = _hash(instructions)
            changed = prompt != self.parent_prompt
            failed_checks = ("invalid_component_mapping",) if prompt is None else ()
        else:
            proposal_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            changed = prompt != self.parent_prompt
            failed_checks = compact_prompt_failed_checks(
                prompt,
                parent_prompt=self.parent_prompt,
                examples=self.examples,
                max_chars=self.max_prompt_chars,
            )
        duplicate = changed and proposal_hash in self._seen_changed_hashes
        if changed:
            self._seen_changed_hashes.add(proposal_hash)
        primary_category = primary_prompt_rejection_category(failed_checks)
        parent_tokens = set(re.findall(r"[a-z0-9]+", (self.parent_prompt or "").casefold()))
        proposal_tokens = set(re.findall(r"[a-z0-9]+", (prompt or "").casefold()))
        token_union = parent_tokens | proposal_tokens
        diagnostic = {
            "iteration": int(event["iteration"]),
            "proposal_hash": proposal_hash,
            "changed": bool(changed),
            "duplicate": bool(duplicate),
            "contract_invalid": bool(failed_checks),
            "primary_rejection_category": primary_category,
            "failed_checks": list(failed_checks),
            "parent_token_jaccard": (
                len(parent_tokens & proposal_tokens) / len(token_union)
                if token_union else 1.0
            ),
            "added_token_count": len(proposal_tokens - parent_tokens),
            "removed_token_count": len(parent_tokens - proposal_tokens),
        }
        self._proposal_records.append(diagnostic)
        self._proposal_token_sets.append(proposal_tokens)
        self._append(
            "proposal_end",
            **diagnostic,
        )

    def proposal_diagnostics(self) -> dict[str, Any]:
        """Return authoritative sanitized proposal-event counters."""

        categories = {
            "over_length": 0,
            "output_contract_contamination": 0,
            "example_copying": 0,
            "append_only": 0,
            "other_failed_check": 0,
        }
        failed_checks: dict[str, int] = {}
        for row in self._proposal_records:
            primary = row["primary_rejection_category"]
            if primary is not None:
                categories[primary] += 1
            for check in row["failed_checks"]:
                failed_checks[check] = failed_checks.get(check, 0) + 1
        example_by_id = {row.example_id: row for row in self.examples}
        outcomes: list[dict[str, Any]] = []
        pattern_counts: dict[str, int] = {}
        allowlisted_tags = {
            "repair", "preservation", "team_hard",
            "direct_flip", "near_margin", "coverage", "fallback",
        }
        for row in self._proposal_records:
            iteration = int(row["iteration"])
            before = self._parent_scores_by_iteration.get(iteration)
            after = self._proposal_scores_by_iteration.get(iteration)
            minibatch_ids = self.reflection_minibatches.get(iteration, ())
            for example_id in minibatch_ids:
                example = example_by_id.get(example_id)
                tags = tuple(sorted(set(example.tags) & allowlisted_tags)) if example else ()
                signature = "+".join(tags) if tags else "general"
                pattern_counts[signature] = pattern_counts.get(signature, 0) + 1
            evaluable = before is not None and after is not None and len(before) == len(after)
            newly_fixed = newly_broken = preservation_loss = None
            delta = None
            if evaluable:
                assert before is not None and after is not None
                newly_fixed = sum(old < 1.0 and new >= 1.0 for old, new in zip(before, after, strict=True))
                newly_broken = sum(old >= 1.0 and new < 1.0 for old, new in zip(before, after, strict=True))
                delta = float(sum(after) - sum(before))
                preservation_loss = sum(
                    old >= 1.0 and new < 1.0
                    and "preservation" in example_by_id[example_id].tags
                    for example_id, old, new in zip(minibatch_ids, before, after, strict=True)
                    if example_id in example_by_id
                )
            outcomes.append({
                "iteration": iteration,
                "proposal_hash": row["proposal_hash"],
                "evaluable": evaluable,
                "official_strict_improvement": iteration in self._accepted_iterations,
                "before_correct": None if before is None else int(sum(before)),
                "after_correct": None if after is None else int(sum(after)),
                "delta_local": delta,
                "newly_fixed": newly_fixed,
                "newly_broken": newly_broken,
                "preservation_loss": preservation_loss,
                "parent_token_jaccard": row["parent_token_jaccard"],
                "added_token_count": row["added_token_count"],
                "removed_token_count": row["removed_token_count"],
                "contract_invalid": row["contract_invalid"],
                "duplicate": row["duplicate"],
            })
        pairwise_similarity: list[float] = []
        for left_index, left in enumerate(self._proposal_token_sets):
            for right in self._proposal_token_sets[left_index + 1 :]:
                union = left | right
                pairwise_similarity.append(
                    len(left & right) / len(union) if union else 1.0
                )
        pattern_total = sum(pattern_counts.values())
        return {
            "proposal_attempts": len(self._proposal_records),
            "proposal_changed": sum(row["changed"] for row in self._proposal_records),
            "proposal_unchanged": sum(
                not row["changed"] for row in self._proposal_records
            ),
            "proposal_duplicate": sum(row["duplicate"] for row in self._proposal_records),
            "proposal_contract_invalid": sum(
                row["contract_invalid"] for row in self._proposal_records
            ),
            "primary_rejection_category_counts": categories,
            "failed_check_counts": dict(sorted(failed_checks.items())),
            "proposal_hashes": [row["proposal_hash"] for row in self._proposal_records],
            "proposal_outcomes": outcomes,
            "mean_pairwise_proposal_token_jaccard": (
                sum(pairwise_similarity) / len(pairwise_similarity)
                if pairwise_similarity else None
            ),
            "reflection_pattern_counts": dict(sorted(pattern_counts.items())),
            "reflection_failure_pattern_concentration": (
                max(pattern_counts.values()) / pattern_total if pattern_total else None
            ),
        }

    def on_evaluation_end(self, event: dict[str, Any]) -> None:
        """Capture only binary score vectors needed for local effect accounting."""

        iteration = int(event["iteration"])
        scores = tuple(float(value) for value in event["scores"])
        if event.get("candidate_idx") is None:
            self._proposal_scores_by_iteration[iteration] = scores
        else:
            self._parent_scores_by_iteration[iteration] = scores

    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
        self._accepted_iterations.add(int(event["iteration"]))
        self._append(
            "candidate_accepted",
            iteration=int(event["iteration"]),
            candidate_index=int(event["new_candidate_idx"]),
            score=float(event["new_score"]),
            parent_indices=[int(value) for value in event["parent_ids"] if value is not None],
        )

    def on_candidate_rejected(self, event: dict[str, Any]) -> None:
        self._append(
            "candidate_rejected",
            iteration=int(event["iteration"]),
            old_score=float(event["old_score"]),
            new_score=float(event["new_score"]),
            reason=str(event["reason"]),
        )

    def on_evaluation_skipped(self, event: dict[str, Any]) -> None:
        self._append(
            "evaluation_skipped",
            iteration=int(event["iteration"]),
            candidate_index=int(event["candidate_idx"]),
            reason=str(event["reason"]),
            is_seed_candidate=bool(event["is_seed_candidate"]),
        )

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        self._append(
            "valset_evaluated",
            iteration=int(event["iteration"]),
            candidate_index=int(event["candidate_idx"]),
            num_examples_evaluated=int(event["num_examples_evaluated"]),
            total_valset_size=int(event["total_valset_size"]),
            is_best_program=bool(event["is_best_program"]),
        )

    def on_pareto_front_updated(self, event: dict[str, Any]) -> None:
        self._append(
            "local_gepa_front_updated",
            iteration=int(event["iteration"]),
            local_gepa_front=[int(value) for value in event["new_front"]],
            displaced=[int(value) for value in event["displaced_candidates"]],
        )

    def on_optimization_end(self, event: dict[str, Any]) -> None:
        self._append(
            "optimization_end",
            best_candidate_index=int(event["best_candidate_idx"]),
            total_iterations=int(event["total_iterations"]),
            total_metric_calls=int(event["total_metric_calls"]),
        )
