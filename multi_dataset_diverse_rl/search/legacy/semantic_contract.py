"""Versioned team-specialization and single-mechanism semantics.

Historical V2 components retain their behavior. This module implements the
current semantic contract without changing responsibility or GEPA survival.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import unicodedata

from ... import versions
from ..patterns import PatternDiagnosticV1
from ..schemas import SearchContractError, TransitionDecision
from ..variable_evidence import PatternCapableVariableEvidencePolicyV1


def mechanism_identity(failure_mechanism, corrective_principle):
    """Structural description identity, independent of provider cluster labels.

    Normalization handles case/spacing, not speculative paraphrase equivalence.
    """
    values = [" ".join(unicodedata.normalize("NFKC", v).casefold().split())
              for v in (failure_mechanism, corrective_principle)]
    if not all(values):
        raise SearchContractError("EMPTY_MECHANISM_DESCRIPTION")
    return hashlib.sha256(json.dumps(values, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


class FocusedPatternDiagnosticV2(PatternDiagnosticV1):
    identity = versions.UNIFIED_FOCUSED_PATTERN_VERSION

    def analyze(self, state, diagnosis, target_member, evidence_rows, history):
        # Reuse all Optimize, partition, confidence and provider validation.
        result = super().analyze(state, diagnosis, target_member, evidence_rows, history)
        grouped = {}
        for row in result["patterns"]:
            mid = mechanism_identity(row["failure_mechanism"], row["corrective_principle"])
            if mid in grouped:
                before = grouped[mid]
                for key in ("support_ids", "counterexample_ids", "risk_ids"):
                    before[key] = tuple(sorted(set(before[key]) | set(row[key])))
                before["confidence"] = max(before["confidence"], row["confidence"])
            else:
                grouped[mid] = {**row, "pattern_id": mid}
        active = [r for r in grouped.values() if r["support_ids"]]
        # An empty provider partition is a conformance failure, not a
        # concentration threshold or permission to run generic mixed repair.
        if not active:
            raise SearchContractError("PATTERN_FOCUS_NOT_IDENTIFIED")
        lane = diagnosis.responsibility[target_member].primary_lane
        primary = {r.example_id for r in evidence_rows if "REPAIR" in r.roles and
                   (lane == "fallback" or lane in r.roles or r.signals.get("lane") == lane)}
        focus = min(active, key=lambda r: (-len(primary & set(r["support_ids"])),
                    -len(r["support_ids"]), -r["confidence"], r["pattern_id"]))
        assigned = sum(len(r["support_ids"]) for r in active)
        total = result["total_residual_count"]
        support = len(focus["support_ids"])
        weights = [len(r["support_ids"]) / assigned for r in active]
        entropy = -sum(w * math.log(w) for w in weights) / math.log(len(weights)) if len(weights) > 1 else 0.0
        return dict(patterns=tuple(grouped[k] for k in sorted(grouped)),
                    dominant_pattern_id=focus["pattern_id"], focus_mechanism_id=focus["pattern_id"],
                    support_count=support, total_residual_count=total,
                    assigned_residual_count=assigned,
                    DPR_all=support / total, ConditionalDPR=support / assigned,
                    Coverage=assigned / total,
                    dominant_pattern_ratio=support / total,
                    normalized_entropy=min(1.0, max(0.0, entropy)),
                    mixed_pattern_count=len(active),
                    unassigned_residual_ids=result["unassigned_residual_ids"],
                    focus_contract="single_mechanism_repair_boundary_only_v2")


class FocusedEvidencePolicyV2(PatternCapableVariableEvidencePolicyV1):
    identity = versions.UNIFIED_FOCUSED_EVIDENCE_VERSION

    def compose(self, state, diagnosis, member_id, rows, pattern_context=None):
        context = pattern_context or {}
        if not context:
            return super().compose(state, diagnosis, member_id, rows, context)
        focus = next((p for p in context.get("patterns", ())
                      if p["pattern_id"] == context.get("focus_mechanism_id")), None)
        if focus is None or not focus["support_ids"]:
            raise SearchContractError("PATTERN_FOCUS_NOT_IDENTIFIED")
        # The old variable-capacity algorithm still handles scope, quotas and
        # minima. Non-focus rows are explicitly boundary evidence, never repair.
        support = set(focus["support_ids"])
        boundaries = set((*focus["counterexample_ids"], *focus["risk_ids"])) - support
        qualified = []
        for row in rows:
            # Do not disguise another residual's repair as boundary backfill.
            # Only analyzer-declared risks/counterexamples or genuine non-repair
            # preservation/team context may accompany the focus gradient.
            if "REPAIR" in row.roles and row.example_id not in support | boundaries:
                continue
            if row.example_id in support:
                qualified.append(replace(row, roles=row.roles | {"FOCUS_REPAIR"},
                    signals={**row.signals, "legacy_tags": ("focus_repair_v2",),
                             "feedback": "Repair only the selected failure mechanism."}))
            else:
                roles = row.roles - {"REPAIR", "direct_flip", "near_margin", "coverage", "pure_coverage"}
                qualified.append(replace(row, roles=roles | {"SAFETY_BOUNDARY"},
                    signals={**row.signals, "legacy_group": "boundary",
                             "legacy_tags": ("safety_boundary_v2",),
                             "feedback": "Safety context only: do not introduce a second repair objective."}))
        if len({r.example_id for r in qualified}) < self.minimum:
            raise SearchContractError("FOCUSED_BACKEND_MINIMUM_WITHOUT_LEGAL_BOUNDARIES")
        view, audit = super().compose(state, diagnosis, member_id, qualified, context)
        audit.update(DPR=context["DPR_all"], DPR_all=context["DPR_all"],
                     ConditionalDPR=context["ConditionalDPR"], Coverage=context["Coverage"],
                     focus_mechanism_id=focus["pattern_id"],
                     boundary_count=sum("SAFETY_BOUNDARY" in r.roles for r in view.mutation_evidence),
                     nonfocus_repair_count=sum("REPAIR" in r.roles and r.example_id not in support
                                              for r in view.mutation_evidence),
                     pattern_risk_count=sum(r.example_id in boundaries for r in view.mutation_evidence))
        return view, audit





from ..initial_competence_transition import InitialCompetenceTransitionV2
