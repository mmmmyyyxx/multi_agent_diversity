"""Grounded, deterministic strategy experience; no raw text enters storage.

Only recognized changes in reasoning actions are distilled. Unclassified
rewrites teach nothing, rather than manufacturing generic success lessons.
This closed abstraction vocabulary is versioned with the memory policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re

from .. import versions
from .memory import MemoryDelta, StructuredLongTermMemoryProviderV1, RISK_CODES
from .schemas import SearchContractError


# Procedures, not benchmark topics, entities, questions or final answers.
STRATEGIES = {
    "explicit_constraint_check": r"\b(?:check|verify|compare|enforce|track|list|respect)\b[^.!?\n]{0,80}\bconstraints?\b",
    "case_analysis": r"\b(?:enumerate|consider|split|analy[sz]e|check)\b[^.!?\n]{0,80}\bcases?\b",
    "boundary_check": r"\b(?:check|test|verify|consider)\b[^.!?\n]{0,80}\b(?:boundar(?:y|ies)|edge cases?|endpoints?)\b",
    "independent_verification": r"\b(?:verify|check)\b[^.!?\n]{0,80}\b(?:independent(?:ly)?|substitut(?:e|ion)|original equation|consistency)\b",
    "relational_consistency": r"\b(?:check|verify|compare|track|resolve)\b[^.!?\n]{0,80}\b(?:relational|relations?|referents?|semantic roles?)\b",
    "subproblem_decomposition": r"\b(?:decompose|break down|separate)\b[^.!?\n]{0,80}\b(?:steps?|subproblems?|goals?)\b",
    "assumption_check": r"\b(?:check|verify|identify|state|track)\b[^.!?\n]{0,80}\bassumptions?\b",
}


def abstract_actions(text):
    # Do not interpret "never check constraints" as an added positive check.
    # Ambiguous/negated clauses are omitted rather than guessed from keywords.
    clauses = re.split(r"[.!?\n]", text)
    text = ". ".join(c for c in clauses if not re.search(
        r"\b(?:not|never|avoid|skip|omit|without|don't|doesn't|no)\b", c, re.I))
    return frozenset(k for k, expression in STRATEGIES.items()
                     if re.search(expression, text, re.I))


@dataclass(frozen=True)
class StrategyExperience:
    memory_id: str
    scope: str
    owner_member: int | None
    kind: str
    pattern_id: str | None
    situation: dict
    action: dict
    outcome: dict
    lesson: dict
    risk_code: str | None
    source_opportunity_id: str
    source_candidate_hash: str
    created_update: int
    lane: str


class StrategyExperienceMemoryV2(StructuredLongTermMemoryProviderV1):
    identity = versions.UNIFIED_EXPERIENCE_MEMORY_VERSION

    def read_for_opportunity(self, opportunity):
        # The inherited ranked/top-k/bounded read compares pattern_id. Supply
        # the actual structural identity directly, not a hash of provider label.
        focus = opportunity.pattern_context.get("focus_mechanism_id")
        lane = opportunity.diagnosis.responsibility[opportunity.target_member].primary_lane
        key = lambda e: (-(focus is not None and e.pattern_id == focus),
                         -(e.lane == lane), -e.created_update, e.memory_id)
        private = sorted((e for e in self.private if e.owner_member == opportunity.target_member), key=key)[:self.limits["top_k_private"]]
        shared = sorted(self.shared, key=key)[:self.limits["top_k_shared"]]
        def project(entry):
            # Provenance hashes/counters live in storage/audit, not repeated
            # in every optimizer context. Keep the complete four-part lesson.
            row = asdict(entry)
            return {k: row[k] for k in ("memory_id", "owner_member", "situation", "action", "outcome", "lesson")}
        def view():
            return {"private": [project(e) for e in private], "shared": [project(e) for e in shared]}
        while (private or shared) and len(json.dumps(view(), sort_keys=True)) > self.limits["max_context_chars"]:
            (shared if shared else private).pop()
        self.read_private_count, self.read_shared_count = len(private), len(shared)
        return view() if private or shared else {}

    @staticmethod
    def _entry(outcome, candidate, kind, risk=None):
        opportunity = outcome.opportunity
        before = abstract_actions(opportunity.parent_prompt)
        after = abstract_actions(candidate.candidate.prompt)
        added, removed = tuple(sorted(after - before)), tuple(sorted(before - after))
        if not added and not removed:
            return None
        focus_id = opportunity.pattern_context.get("focus_mechanism_id")
        focus = next((p for p in opportunity.pattern_context.get("patterns", ())
                      if p["pattern_id"] == focus_id), None)
        # Vocabulary projection prevents a provider's copied question, answer,
        # prompt, entity or reasoning trace from becoming memory text.
        concepts = tuple(sorted(abstract_actions(" ".join(
            (focus["failure_mechanism"], focus["corrective_principle"]))) if focus else ()))
        lane = opportunity.diagnosis.responsibility[opportunity.target_member].primary_lane
        cid = hashlib.sha256(candidate.candidate.prompt.encode()).hexdigest()
        oid = hashlib.sha256(opportunity.opportunity_id.encode()).hexdigest()
        mid = hashlib.sha256(f"{oid}:{cid}:{kind}:{risk}".encode()).hexdigest()
        measurement = candidate.full or candidate.team_probe
        if measurement is None:
            raise SearchContractError("EXPERIENCE_WITHOUT_MEASURED_OUTCOME")
        d = measurement.aggregation_diagnostics
        fixed, broken = d.get("team_newly_fixed_count"), d.get("team_newly_broken_count")
        if any(v is not None and (not isinstance(v, int) or v < 0) for v in (fixed, broken)):
            raise SearchContractError("INVALID_EXPERIENCE_COUNTS")
        metrics = dict(scope="FULL" if candidate.full is not None else "TEAM_PROBE",
                       team_score=measurement.aggregate_score,
                       target_score=measurement.member_scores[opportunity.target_member],
                       newly_fixed=fixed, newly_broken=broken,
                       committed=kind == "SUCCESS", rejection_stage=risk,
                       adaptive_gate_passed=outcome.gate_passed if
                       candidate.candidate.candidate_id == outcome.selected_candidate_id else None)
        probe = d.get("team_probe_metrics")
        metrics.update(team_score_delta=d.get("team_score_delta", getattr(probe, "vote_delta", None)),
                       target_score_delta=d.get("target_score_delta", getattr(probe, "target_delta", None)),
                       target_initial_margin=d.get("target_initial_margin"))
        return StrategyExperience(mid, "PRIVATE" if kind == "SUCCESS" else "SHARED_RISK",
            opportunity.target_member if kind == "SUCCESS" else None, kind, focus_id,
            dict(primary_lane=lane, mechanism_concepts=concepts, mechanism_identity=focus_id),
            dict(added_checks=added, removed_checks=removed), metrics,
            dict(applicability={"primary_lane": lane, "mechanism_identity": focus_id},
                 successful_strategy={"added_checks": added, "removed_checks": removed} if kind == "SUCCESS" else None,
                 avoid_strategy={"added_checks": added, "removed_checks": removed} if kind == "RISK" else None,
                 measured_boundary="INITIAL_COMPETENCE_AND_STRICT_TEAM_GAIN" if kind == "SUCCESS" else risk,
                 transfer_status="OBSERVED_ONCE_NOT_CAUSAL_OR_GENERALIZATION_PROOF"),
            risk, oid, cid, outcome.update_index, lane)

    def prepare_outcome(self, outcome):
        if not outcome.complete or outcome.operational_failure:
            return MemoryDelta(self.revision, self.private, self.shared)
        private, shared = list(self.private), list(self.shared)
        for row in outcome.evaluated:
            if outcome.committed and row.candidate.candidate_id == outcome.selected_candidate_id:
                entry = self._entry(outcome, row, "SUCCESS")
                if entry is not None:
                    private.append(entry)
                continue
            risk = row.diagnostics.get("scientific_risk_code")
            if row.candidate.candidate_id == outcome.selected_candidate_id and outcome.gate_passed is False:
                risk = "SHADOW_REJECTION"
            if risk is not None:
                if risk not in RISK_CODES:
                    raise SearchContractError("nonstructural memory risk")
                entry = self._entry(outcome, row, "RISK", risk)
                if entry is not None:
                    shared.append(entry)
        private = list({e.memory_id: e for e in private}.values())
        kept = []
        for member in sorted({e.owner_member for e in private}):
            kept.extend(sorted((e for e in private if e.owner_member == member),
                               key=lambda e: (e.created_update, e.memory_id))[-self.limits["private_storage_limit"]:])
        shared = sorted({e.memory_id: e for e in shared}.values(),
                        key=lambda e: (e.created_update, e.memory_id))[-self.limits["shared_storage_limit"]:]
        return MemoryDelta(self.revision, tuple(kept), tuple(shared))
