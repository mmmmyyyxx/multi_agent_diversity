"""Zero-provider, read-only decision evidence for the frozen V3 blockers.

This is an offline analysis tool, not an experiment runner. It prints only
counts/categories. Its fake initial-state rehearsal uses the existing test
fixture and deletes its temporary artifacts on exit.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import sys
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import local_gepa_budget_capacity
from multi_dataset_diverse_rl.persistence.durable_io import io_path
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (
    build_primary_responsibility_summary_from_counts,
    select_primary_responsibility_targets,
)
from multi_dataset_diverse_rl.team_search.task_builder import Layer2EvidenceRequestBuilder


def initial_state_capacity() -> dict[str, object]:
    """Capture the first real V3 composition gate with a fake provider."""
    source = ROOT / "tests/test_online_transfer_full_fake.py"
    spec = importlib.util.spec_from_file_location("v3_full_fake_for_capacity", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from multi_dataset_diverse_rl import production_transfer_diagnostic as runner
    from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (
        PrimaryResponsibilityPersistentRealizabilityScheduler,
    )

    observed: dict[str, object] = {}
    patch = pytest.MonkeyPatch()
    original_freeze = runner.freeze_current_responsibility
    original_select = PrimaryResponsibilityPersistentRealizabilityScheduler.select
    original_build = Layer2EvidenceRequestBuilder.build

    def capture_freeze(system, *, update_index):
        snapshot = original_freeze(system, update_index=update_index)
        memberships: dict[str, int] = {}
        for rows in snapshot.assigned.values():
            for row in rows:
                memberships[row.question_hash] = memberships.get(row.question_hash, 0) + 1
        observed["responsibility_source"] = snapshot.source_version
        observed["raw_legal_residual_count"] = len(memberships)
        observed["overlapping_residual_count"] = sum(n > 1 for n in memberships.values())
        return snapshot

    def capture_select(self, **kwargs):
        decision = original_select(self, **kwargs)
        selected = decision.selected_member_ids[0]
        summary = next(row for row in decision.summaries if row.member_id == selected)
        observed.update({
            "selected_member": selected,
            "primary_lane": summary.primary_lane,
            "D": summary.direct_count,
            "N": summary.near_margin_count,
            "C": summary.coverage_count,
            "V": summary.primary_score,
            "f": summary.failure_count,
            "target_score": summary.target_score,
        })
        return decision

    def capture_build(self, request, assignment):
        repair = tuple(row for row in assignment.evidence
                       if row.evidence_group == "repair" and
                       self._matches_primary_lane(row, assignment.primary_responsibility_lane))
        transition = assignment.latest_transition
        focus = () if transition is None else transition.newly_broken_ids
        anchor = () if transition is None else transition.newly_fixed_ids
        budget = request.local_metric_budget
        batches = max(1, budget // 3)
        slots = batches * 3
        needed = len(repair) + len(focus) + len(anchor)
        observed.update({
            "aligned_responsibility_rows": len(repair),
            "focus_rows": len(focus),
            "anchor_rows": len(anchor),
            "search_packet_item_count": needed,
            "local_eval_ids": len(assignment.local_validation_example_ids),
            "ordered_schedule_batches": batches,
            "batch_size": 3,
            "schedule_delivery_slots": slots,
            "local_metric_budget": budget,
            "minimum_metric_budget_for_coverage": 3 * ((needed + 2) // 3),
            "assignment_responsibility_value": assignment.responsibility_value,
        })
        return original_build(self, request, assignment)

    patch.setattr(runner, "freeze_current_responsibility", capture_freeze)
    patch.setattr(PrimaryResponsibilityPersistentRealizabilityScheduler, "select", capture_select)
    patch.setattr(Layer2EvidenceRequestBuilder, "build", capture_build)
    temporary = Path(tempfile.mkdtemp(prefix="v3_scientific_blocker_"))
    try:
        try:
            module.rehearse(temporary / ("capacity-" + "x" * 40), patch, 0, v3=True)
        except ValueError as exc:
            observed["current_failure"] = str(exc)
        else:
            raise AssertionError("expected V3 packet schedule failure")
    finally:
        patch.undo()
        # The fixture deliberately creates >260-character paths. Restrict
        # cleanup to this verified, freshly created task-local temp root.
        resolved = temporary.resolve()
        temp_parent = Path(tempfile.gettempdir()).resolve()
        if resolved.parent != temp_parent or not resolved.name.startswith("v3_scientific_blocker_"):
            raise RuntimeError("unexpected temporary audit root")
        shutil.rmtree(io_path(resolved))
    capacity = local_gepa_budget_capacity(
        metric_budget=36, validation_size=12, reflection_minibatch_size=3,
    )
    observed.update({
        "gepa_seed_evaluation_calls": capacity.seed_evaluation_calls,
        "gepa_proposal_minibatch_cost": capacity.proposal_attempt_calls,
        "gepa_accepted_full_validation_cost": capacity.accepted_full_evaluation_calls,
        "max_rejected_proposal_batches_at_budget_36": capacity.max_rejected_proposals,
        "max_distinct_deliveries_if_every_proposal_rejected": (
            capacity.max_rejected_proposals * 3
        ),
        "metric_budget_for_17_rejected_proposal_batches": (
            capacity.seed_evaluation_calls + 17 * capacity.proposal_attempt_calls
        ),
        "real_api_calls": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    })
    assert observed["search_packet_item_count"] > observed["schedule_delivery_slots"]
    return observed


def _summary(member: int, counts: tuple[int, int, int], failure: int):
    return build_primary_responsibility_summary_from_counts(
        member_id=member, direct_count=counts[0], near_margin_count=counts[1],
        coverage_count=counts[2], failure_count=failure,
    )


def quota_matrix() -> dict[str, object]:
    from multi_dataset_diverse_rl.team_search.schemas import TeamEvidenceCase

    def row(index: int, group: str, lane: str = "direct_flip") -> TeamEvidenceCase:
        return TeamEvidenceCase(
            example_id=f"synthetic-{index}", input_payload="synthetic",
            gold="A", target_output="B", feedback=None,
            evidence_group=group,
            tags=("repair", lane, "team_hard") if group == "repair" else (group,),
        )

    cases: list[dict[str, object]] = []
    builder = Layer2EvidenceRequestBuilder()
    for lane in ("direct_flip", "near_margin", "coverage"):
        for count in range(7):
            counts = tuple(count if name == lane else 0 for name in (
                "direct_flip", "near_margin", "coverage"))
            summaries = tuple(_summary(member, counts if member == 2 else (0, 0, 0), 0)
                              for member in range(5))
            decision = select_primary_responsibility_targets(
                summaries, seed=81, update_index=0, target_count=1,
            )
            selected = decision.selected_member_ids[0]
            chosen = summaries[selected]
            evidence = tuple(row(index, "repair", lane) for index in range(count))
            evidence += tuple(row(20 + index, "preservation") for index in range(4))
            evidence += tuple(row(40 + index, "team_hard") for index in range(4))
            if chosen.primary_lane == "fallback":
                status = "NO_ALIGNED_RESPONSIBILITY_TARGET_NOT_REACHED"
            else:
                try:
                    builder.select_team_minibatch(evidence, primary_responsibility_lane=lane)
                except ValueError as exc:
                    status = str(exc)
                else:
                    status = "CONSTRUCTIBLE"
            cases.append({
                "lane_input": lane, "lane_count": count,
                "D": counts[0], "N": counts[1], "C": counts[2],
                "selected_member": selected, "primary_lane": chosen.primary_lane,
                "V": chosen.primary_score, "target_score": chosen.target_score,
                "team_minibatch_status": status,
            })
    other_shortages: list[dict[str, object]] = []
    for group in ("preservation", "team_hard"):
        for count in range(5):
            evidence = tuple(row(index, "repair") for index in range(4))
            evidence += tuple(row(20 + index, "preservation")
                              for index in range(count if group == "preservation" else 4))
            evidence += tuple(row(40 + index, "team_hard")
                              for index in range(count if group == "team_hard" else 4))
            try:
                builder.select_team_minibatch(evidence, primary_responsibility_lane="direct_flip")
            except ValueError as exc:
                status = str(exc)
            else:
                status = "CONSTRUCTIBLE"
            other_shortages.append({"group": group, "available_extra_rows": count,
                                    "status": status})
    return {"lane_cases": cases, "other_quota_cases": other_shortages,
            "note": "Synthetic structural states; no real-data frequency inference."}


def cross_product_grid() -> dict[str, object]:
    """Compare raw target with quota/capacity-filtered target in five-member grids.

    Two members are potentially actionable, three are zero-score controls.
    All variables are deterministic and synthetic, not sampled from the task.
    """
    patterns = (
        (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0),
        (4, 0, 0), (5, 0, 0), (0, 1, 0), (0, 2, 0),
        (0, 4, 0), (0, 0, 1), (0, 0, 4), (2, 2, 0),
        (0, 3, 5), (40, 0, 0), (0, 0, 40),
    )
    totals = {key: 0 for key in (
        "valid_grid_states", "raw_positive_states", "same_target",
        "different_target", "no_feasible_target", "raw_target_quota_infeasible",
        "raw_target_capacity_infeasible", "selected_target_truncated",
        "selected_target_not_truncated", "raw_target_truncated_if_selected",
    )}
    by_overlap = {str(degree): {key: 0 for key in (
        "valid_grid_states", "different_target", "no_feasible_target",
        "selected_target_truncated",
    )} for degree in (0, 1)}
    witnesses: dict[str, dict[str, object]] = {}
    for a, b, fa, fb, preservation, hard, overlap, transition, sensitivity in itertools.product(
        patterns, patterns, (0, 2), (0, 2), (3, 4, 8), (6, 8, 20, 60),
        (0, 1), ((0, 0), (4, 4), (16, 16), (20, 20)), (0, 1),
    ):
        total_a, total_b = sum(a), sum(b)
        shared = min(total_a, total_b) if overlap else 0
        if total_a + total_b - shared > hard:
            continue  # A legal residual universe must contain all assigned rows.
        totals["valid_grid_states"] += 1
        by_overlap[str(overlap)]["valid_grid_states"] += 1
        summaries = tuple(_summary(member, a if member == 0 else b if member == 1 else (0, 0, 0),
                                   fa if member == 0 else fb if member == 1 else 0)
                          for member in range(5))
        raw = select_primary_responsibility_targets(
            summaries, seed=81, update_index=0, target_count=1,
        ).selected_member_ids[0]
        if summaries[raw].primary_score == 0:
            continue
        totals["raw_positive_states"] += 1
        available_slots = 36 - sum(transition)
        primary_rows = lambda member: (
            summaries[member].direct_count if summaries[member].primary_lane == "direct_flip"
            else summaries[member].near_margin_count if summaries[member].primary_lane == "near_margin"
            else summaries[member].coverage_count
        )
        quota_feasible = lambda member: (
            summaries[member].primary_score > 0 and primary_rows(member) >= 4
            and preservation >= 4 and hard >= 8
        )
        if not quota_feasible(raw):
            totals["raw_target_quota_infeasible"] += 1
        if available_slots < 4:
            totals["raw_target_capacity_infeasible"] += 1
        eligible = [member for member in (0, 1)
                    if quota_feasible(member) and available_slots >= 4]
        eligible.sort(key=lambda member: (-summaries[member].target_score, member))
        selected = eligible[0] if eligible else None
        witness = {
            "member0_DNC": list(a), "member1_DNC": list(b),
            "f0": fa, "f1": fb, "preservation_rows": preservation,
            "global_hard_rows": hard, "overlap_degree": overlap,
            "focus_rows": transition[0], "anchor_rows": transition[1],
            "margin_pivotality_variant": sensitivity,
            "raw_target": raw, "feasible_target": selected,
        }
        if selected is None:
            totals["no_feasible_target"] += 1
            by_overlap[str(overlap)]["no_feasible_target"] += 1
            witnesses.setdefault("no_feasible_target", witness)
        elif selected == raw:
            totals["same_target"] += 1
        else:
            totals["different_target"] += 1
            by_overlap[str(overlap)]["different_target"] += 1
            witnesses.setdefault("different_target", witness)
        if selected is not None:
            if primary_rows(selected) > available_slots:
                totals["selected_target_truncated"] += 1
                by_overlap[str(overlap)]["selected_target_truncated"] += 1
                witnesses.setdefault("selected_target_truncated", witness)
            else:
                totals["selected_target_not_truncated"] += 1
            if selected == raw and primary_rows(raw) > available_slots:
                totals["raw_target_truncated_if_selected"] += 1
    return {
        "grid_definition": {
            "member_count": 5, "actionable_member_ids": [0, 1],
            "DNC_patterns_per_actionable_member": [list(pattern) for pattern in patterns],
            "failure_counts": [0, 2], "preservation_rows": [3, 4, 8],
            "global_hard_rows": [6, 8, 20, 60], "overlap_degree": [0, 1],
            "focus_anchor_pairs": [[0, 0], [4, 4], [16, 16], [20, 20]],
            "margin_pivotality_variants": [0, 1],
            "structural_quota_rule": "primary>=4; preservation>=4; >=8 unique hard IDs including four selected repair IDs",
            "bounded_curriculum_rule": "reserve every focus/anchor item; select min(primary,36-focus-anchor) responsibility items; require >=4 slots",
            "overlap_rule": "member repair universes share 0 or min(total_a,total_b) IDs; skip impossible hard-universe states",
            "margins_pivotality_role": "preservation ordering negative-control axis; does not change counts or admission feasibility",
        },
        "counts": totals,
        "by_overlap_degree": by_overlap,
        "witnesses": witnesses,
        "evidence_type": "SYNTHETIC_STRUCTURAL_COUNTERFACTUAL_ONLY",
    }


def analyze() -> dict[str, object]:
    return {
        "schedule_capacity_analysis": initial_state_capacity(),
        "team_minibatch_feasibility_analysis": quota_matrix(),
        "cross_product_analysis": cross_product_grid(),
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), sort_keys=True, indent=2))
