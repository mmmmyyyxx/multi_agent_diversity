"""Observational GEPA callbacks for auditable lifecycle and lineage events."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .gepa_adapter import (
    compact_prompt_failed_checks,
    primary_prompt_rejection_category,
)
from .schemas import LocalEvidenceExample


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

    def _append(self, event_type: str, **values: Any) -> None:
        row = {"event_index": len(self.events), "event_type": event_type, **values}
        self.events.append(row)
        if self.event_path is not None:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

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
        diagnostic = {
            "proposal_hash": proposal_hash,
            "changed": bool(changed),
            "duplicate": bool(duplicate),
            "contract_invalid": bool(failed_checks),
            "primary_rejection_category": primary_category,
            "failed_checks": list(failed_checks),
        }
        self._proposal_records.append(diagnostic)
        self._append(
            "proposal_end",
            iteration=int(event["iteration"]),
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
        }

    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
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
