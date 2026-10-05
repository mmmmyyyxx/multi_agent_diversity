"""Deterministic in-process private successes and shared structural risks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ... import versions
from ..schemas import SearchContractError
from ..memory_records import MemoryEntry, OpportunityOutcome, MemoryDelta

RISK_CODES = frozenset({"TEAM_PROBE_REJECTION", "COMMON_SAFE_REJECTION", "SHADOW_REJECTION"})
PRINCIPLES = {
    "SUCCESS": "Preserve fixed-peer team competence while repairing the selected responsibility.",
    "TEAM_PROBE_REJECTION": "Avoid edits that regress fixed-peer team probe behavior.",
    "COMMON_SAFE_REJECTION": "Preserve target, team and valid-output competence on the complete optimization scope.",
    "SHADOW_REJECTION": "Avoid deploying an edit that fails the adaptive safety gate.",
}











class StructuredLongTermMemoryProviderV1:
    identity = versions.UNIFIED_STRUCTURED_MEMORY_VERSION

    def __init__(self, *, top_k_private, top_k_shared, max_context_chars,
                 private_storage_limit, shared_storage_limit):
        limits = (top_k_private, top_k_shared, max_context_chars, private_storage_limit, shared_storage_limit)
        if any(not isinstance(x, int) or x <= 0 for x in limits):
            raise SearchContractError("explicit positive memory limits required")
        self.limits = dict(zip(("top_k_private", "top_k_shared", "max_context_chars", "private_storage_limit", "shared_storage_limit"), limits))
        self.private = (); self.shared = (); self.revision = 0
        self.read_private_count = self.read_shared_count = self.write_count = 0

    def read_for_opportunity(self, opportunity):
        focus = opportunity.pattern_context.get("dominant_pattern_id")
        pattern = hashlib.sha256(focus.encode()).hexdigest() if focus else None
        lane = opportunity.diagnosis.responsibility[opportunity.target_member].primary_lane
        key = lambda e: (-(pattern is not None and e.pattern_id == pattern), -(e.lane == lane), -e.created_update, e.memory_id)
        private = sorted((e for e in self.private if e.owner_member == opportunity.target_member), key=key)[:self.limits["top_k_private"]]
        shared = sorted(self.shared, key=key)[:self.limits["top_k_shared"]]
        view = lambda: {"private": [asdict(e) for e in private], "shared": [asdict(e) for e in shared]}
        # Drop lowest-ranked rows, preserving complete structured entries.
        while (private or shared) and len(json.dumps(view(), sort_keys=True)) > self.limits["max_context_chars"]:
            if shared: shared.pop()
            else: private.pop()
        self.read_private_count = len(private); self.read_shared_count = len(shared)
        return view() if private or shared else {}

    @staticmethod
    def _entry(outcome, candidate, kind, risk=None):
        opportunity = outcome.opportunity
        focus = opportunity.pattern_context.get("dominant_pattern_id")
        pattern = hashlib.sha256(focus.encode()).hexdigest() if focus else None
        cid = hashlib.sha256(candidate.candidate.prompt.encode()).hexdigest()
        oid = hashlib.sha256(opportunity.opportunity_id.encode()).hexdigest()
        identity = hashlib.sha256(f"{oid}:{cid}:{kind}:{risk}".encode()).hexdigest()
        d = candidate.diagnostics
        # Fixed/broken counts describe team Full effects, never local GEPA delta.
        fixed = int(d.get("team_newly_fixed_count", 0)); broken = int(d.get("team_newly_broken_count", 0))
        if min(fixed, broken) < 0: raise SearchContractError("negative memory counts")
        return MemoryEntry(identity, "PRIVATE" if kind == "SUCCESS" else "SHARED_RISK",
                           opportunity.target_member if kind == "SUCCESS" else None,
                           kind, pattern, PRINCIPLES[kind if kind == "SUCCESS" else risk], risk,
                           oid, cid, identity if kind == "SUCCESS" else None,
                           fixed, broken, int(opportunity.pattern_context.get("support_count", 0)),
                           outcome.update_index, opportunity.diagnosis.responsibility[opportunity.target_member].primary_lane)

    def prepare_outcome(self, outcome):
        private = list(self.private); shared = list(self.shared)
        if not outcome.complete or outcome.operational_failure:
            return MemoryDelta(self.revision, self.private, self.shared)
        for row in outcome.evaluated:
            cid = row.candidate.candidate_id
            if outcome.committed and cid == outcome.selected_candidate_id:
                private.append(self._entry(outcome, row, "SUCCESS"))
                continue
            risk = row.diagnostics.get("scientific_risk_code")
            if cid == outcome.selected_candidate_id and outcome.gate_passed is False:
                risk = "SHADOW_REJECTION"
            if risk is not None:
                if risk not in RISK_CODES: raise SearchContractError("nonstructural memory risk")
                shared.append(self._entry(outcome, row, "RISK", risk))
        # All validation and allocation happen before any team mutation.
        private = list({e.memory_id:e for e in private}.values())
        kept = []
        for member in sorted({e.owner_member for e in private}):
            kept.extend(sorted((e for e in private if e.owner_member == member),
                               key=lambda e: (e.created_update, e.memory_id))[-self.limits["private_storage_limit"]:])
        shared = sorted({e.memory_id:e for e in shared}.values(), key=lambda e: (e.created_update, e.memory_id))[-self.limits["shared_storage_limit"]:]
        return MemoryDelta(self.revision, tuple(kept), tuple(shared))

    def validate_delta(self, delta):
        if delta.revision != self.revision: raise SearchContractError("stale memory outcome")
        if not isinstance(delta.private, tuple) or not isinstance(delta.shared, tuple):
            raise SearchContractError("invalid memory delta")

    def apply_outcome(self, delta):
        # Validated prepared immutable tuples; no provider, persistence or parsing.
        changed = (self.private, self.shared) != (delta.private, delta.shared)
        self.private, self.shared = delta.private, delta.shared
        self.revision += int(changed)
        self.write_count += int(changed)

    def audit(self):
        return dict(private_memory_count_by_member={str(i):sum(e.owner_member == i for e in self.private) for i in range(5)},
                    shared_risk_count=len(self.shared), read_private_count=self.read_private_count,
                    read_shared_count=self.read_shared_count,
                    memory_ids=[e.memory_id for e in (*self.private, *self.shared)],
                    memory_policy_identity=self.identity, stateful_write_count=self.write_count)
