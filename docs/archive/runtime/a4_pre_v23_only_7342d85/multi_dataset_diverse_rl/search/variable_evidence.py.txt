"""V2 variable evidence; capacities are operational, never scientific quotas."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from dataclasses import replace

from .. import versions
from .policies import ResponsibilitySignal, _stable_key
from .schemas import EvidenceItem, EvidenceView, SearchContractError


def validation_capacity(metric_budget=36, reflection_minibatch_size=3):
    if metric_budget <= 0 or reflection_minibatch_size <= 0:
        raise SearchContractError("positive backend capacities required")
    return max(0, (metric_budget - 4 * reflection_minibatch_size) // 2)


class PatternCapableVariableEvidencePolicyV1:
    identity = versions.UNIFIED_VARIABLE_EVIDENCE_VERSION

    def __init__(self, *, metric_budget=36, reflection_minibatch_size=3):
        self.metric_budget = metric_budget
        self.minimum = reflection_minibatch_size
        self.capacity = validation_capacity(metric_budget, reflection_minibatch_size)

    @staticmethod
    def _priority(row, lane):
        primary = "REPAIR" in row.roles and (lane == "fallback" or lane in row.roles or row.signals.get("lane") == lane)
        if primary: rank = 0
        elif "TRANSITION_FOCUS" in row.roles: rank = 1
        elif "TRANSITION_ANCHOR" in row.roles: rank = 2
        elif "PRESERVATION" in row.roles and row.signals.get("mutation_sensitive"): rank = 3
        elif "TEAM_HARD" in row.roles and row.signals.get("team_disagreement", 0) > 1: rank = 4
        else: rank = 5
        return (rank, -int(row.signals.get("team_disagreement", 0)), *_stable_key(row))

    def compose(self, state, diagnosis, member_id, rows, pattern_context=None):
        del state
        signal = diagnosis.responsibility.get(member_id)
        if not isinstance(signal, ResponsibilitySignal) or signal.raw_value <= 0:
            raise SearchContractError("positive raw responsibility required")
        if any(r.source_split != "optimize" for r in rows):
            raise SearchContractError("variable evidence is Optimize only")
        unique = {}
        for row in rows:
            if row.example_id in unique and unique[row.example_id] != row:
                raise SearchContractError("conflicting Optimize evidence identity")
            unique[row.example_id] = row
        ordered = sorted(unique.values(), key=lambda r: self._priority(r, signal.primary_lane))
        if self.capacity < 1 or len(ordered) < self.minimum:
            raise SearchContractError("backend technical minimum infeasible")
        context = pattern_context or {}
        focus_id = context.get("dominant_pattern_id")
        focus = next((p for p in context.get("patterns", ()) if p["pattern_id"] == focus_id), None)
        support = set(focus["support_ids"]) if focus else set()
        risks = set((*focus["risk_ids"], *focus["counterexample_ids"])) if focus else set()
        if not (support | risks) <= set(unique):
            raise SearchContractError("pattern evidence outside Optimize")
        if focus:
            selected = sorted((r for r in ordered if r.example_id in support | risks),
                              key=lambda r: (r.example_id not in support, self._priority(r, signal.primary_lane)))
            # Other mechanisms are not fillers. Only a technical shortage may
            # draw generic rows, with an explicit backfill audit.
            validation = selected + [r for r in ordered if "PRESERVATION" in r.roles
                                      and r.signals.get("mutation_sensitive") and r.example_id not in support | risks]
        else:
            # Low-value ordinary rows never fill the operational maximum.
            selected = [r for r in ordered if self._priority(r, signal.primary_lane)[0] < 5]
            validation = list(selected)
        backfill = []
        used = {r.example_id for r in selected}
        for row in ordered:
            if len(selected) >= self.minimum: break
            if row.example_id not in used:
                selected.append(row); used.add(row.example_id); backfill.append(row.example_id)
        # Mutation view and validation/probe capacity are independent.
        mutation = selected[:max(self.capacity, self.minimum)]
        if len(mutation) < self.minimum:
            raise SearchContractError("backend technical mutation minimum infeasible")
        if not validation:
            validation = mutation[:1]
        validation = list({r.example_id:r for r in validation}.values())[:self.capacity]
        mutation = tuple(replace(r, roles=r.roles | {"MUTATION"}) for r in mutation)
        valid = tuple(replace(r, roles=r.roles | {"SEARCH_VALIDATION"}) for r in validation)
        probe = tuple(replace(r, roles=r.roles | {"TEAM_PROBE"}) for r in validation)
        ids = {"mutation": [r.example_id for r in mutation], "validation": [r.example_id for r in valid], "probe": [r.example_id for r in probe]}
        audit = dict(pattern_enabled=bool(context), pattern_count=len(context.get("patterns", ())),
                     focus_pattern_id=focus_id, DPR=context.get("dominant_pattern_ratio", 0.0),
                     normalized_entropy=context.get("normalized_entropy", 0.0),
                     mutation_count=len(mutation), search_validation_count=len(valid), team_probe_count=len(probe),
                     role_counts=dict(sorted(Counter(role for r in mutation for role in r.roles).items())),
                     pattern_support_count=sum(r.example_id in support for r in mutation),
                     pattern_risk_count=sum(r.example_id in risks for r in mutation),
                     generic_backfill_count=len(backfill), operational_validation_capacity=self.capacity,
                     evidence_ids_hash=hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        return EvidenceView(mutation, valid, probe, "optimize_full", "shadow_adaptive"), audit

    def build(self, state, diagnosis, member_id, rows):
        return self.compose(state, diagnosis, member_id, rows)[0]


class VariableEvidenceFeasibilityV1:
    identity = versions.UNIFIED_VARIABLE_FEASIBILITY_VERSION

    def __init__(self, *, metric_budget=36, reflection_minibatch_size=3):
        self.evidence = PatternCapableVariableEvidencePolicyV1(
            metric_budget=metric_budget, reflection_minibatch_size=reflection_minibatch_size)

    def feasible(self, state, diagnosis, member_id, evidence):
        if not any("REPAIR" in r.roles for r in evidence): return False
        try:
            self.evidence.build(state, diagnosis, member_id, evidence)
        except SearchContractError:
            return False
        return True
