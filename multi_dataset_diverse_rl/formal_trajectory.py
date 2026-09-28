"""Read-only, sanitized trajectory projection from a frozen Formal V3 summary.

This module is used only after execution. It cannot select a target, candidate,
checkpoint, or final state and must never be imported by the search runtime.
"""

from __future__ import annotations

from typing import Any, Mapping

from .team_search.evidence_audit import sanitize_v4_evidence_trace


_CANDIDATE_FIELDS = (
    "candidate_id", "candidate_hash", "generation", "local_parent_score",
    "local_candidate_score", "local_acceptance_delta",
    "local_full_validation_delta", "local_newly_fixed", "local_newly_broken",
    "local_preservation_loss", "team_minibatch", "full",
    "ordinary_common_safe", "ordinary_shadow", "committed", "shadow_gate",
)


def derive_formal_trajectory_records(summary: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Project only recorded decisions and counts; reject incomplete chains."""

    if summary.get("validation50_calls") != 0 or summary.get("test50_calls") != 0:
        raise ValueError("adaptive Formal V3 trajectory accessed a held-out split")
    mode = summary.get("mode_id")
    if mode not in {"GEPA_NATIVE", "GEPA_LAYER2_V4"}:
        raise ValueError("unsupported Formal V3 mode")
    seed = summary.get("seed")
    events = tuple(summary.get("events", ()))
    common = {
        "schema_version": "formal_responsibility_only_trajectory_v1",
        "experiment_id": summary["experiment_id"],
        "mode_id": mode,
        "seed": seed,
    }
    if mode == "GEPA_NATIVE":
        return tuple({
            **common,
            "record_type": "NATIVE_OPTIMIZATION_UNIT",
            "event_index": event["index"],
            "event_kind": event["kind"],
            "candidate_ids": list(event["candidate_ids"]),
            "state_hash": event["state_hash"],
            "stop_reason": event["stop_reason"],
            "backend_termination_reason": event["telemetry"].get("backend_termination_reason"),
            "backend_saturation": event["telemetry"].get("backend_saturation"),
            "final_native_candidate_hash": summary["final_native_candidate_hash"],
        } for event in events)

    feasibility_rows = summary["feasibility_trace"]
    evidence_rows = summary["evidence_view_trace"]
    transition_rows = summary["transition_trace"]
    feasibility = {row["update_index"]: row for row in feasibility_rows}
    evidence = {row["update_index"]: row for row in evidence_rows}
    transitions = {row["update_index"]: row for row in transition_rows}
    if (len(feasibility) != len(feasibility_rows)
            or len(evidence) != len(evidence_rows)
            or len(transitions) != len(transition_rows)):
        raise ValueError("duplicate formal opportunity telemetry index")
    candidates: dict[int, list[Mapping[str, Any]]] = {}
    for row in summary["candidate_diagnostics"]:
        candidates.setdefault(row["update_index"], []).append(row)
    opportunity_events = tuple(event for event in events if event["kind"] == "TEAM_OPPORTUNITY")
    if len(opportunity_events) != len(evidence) or set(evidence) != set(transitions):
        raise ValueError("formal opportunity telemetry is incomplete")
    if not set(evidence).issubset(feasibility):
        raise ValueError("formal opportunity lacks parent feasibility trace")
    records: list[dict[str, Any]] = []
    parent = summary["initial_team_hash"]
    for event in opportunity_events:
        update = event["index"] - 1
        packet = evidence[update]
        transition = transitions[update]
        target = feasibility[update]["selected_member"]
        successor = transition["successor_team_hash"]
        if (packet["parent_team_hash"] != parent
                or feasibility[update]["parent_team_hash"] != parent
                or transition["parent_team_hash"] != parent
                or packet["successor_team_hash"] != successor
                or packet["target_member"] != target
                or transition["selected_member"] != target
                or event["state_hash"] != successor):
            raise ValueError("formal parent/selection/successor chain mismatch")
        if (packet["raw_V"] != packet["assignment_V"]
                or packet["raw_V"] != packet["packet_V"]
                or packet["team_minibatch_ids"] != packet["local_eval_ids"]
                or len(set(packet["team_minibatch_ids"])) != 12
                or sum(len(batch) for batch in packet["nominal_schedule"]) > 36):
            raise ValueError("formal V4 packet identity mismatch")
        sanitize_v4_evidence_trace(packet)
        selected = next((row for row in feasibility[update]["members"]
                         if row["member_id"] == target), None)
        if (selected is None or selected["raw_V"] != packet["raw_V"]
                or selected["raw_target_score"] != selected["raw_V"] / (1 + selected["failure_count"])):
            raise ValueError("formal selected responsibility differs from frozen parent")
        committed = transition["committed_candidate_id"] is not None
        if committed == (successor == parent):
            raise ValueError("formal commit/state transition mismatch")
        if event["committed_candidate_id"] != transition["committed_candidate_id"]:
            raise ValueError("formal event/commit mismatch")
        rows = candidates.get(update, [])
        if (set(event["candidate_ids"]) != {row["candidate_id"] for row in rows}
                or len(event["candidate_ids"]) != len(rows)
                or (committed and sum(row["candidate_id"] == transition["committed_candidate_id"]
                                      and row["committed"] for row in rows) != 1)):
            raise ValueError("formal local candidate/event identity mismatch")
        if any(row["parent_team_hash"] != parent or row["target_member"] != target
               or row["full"] is None for row in rows):
            raise ValueError("formal accepted candidate lacks diagnostic evidence")
        records.append({
            **common,
            "record_type": "LAYER2_OPPORTUNITY",
            "update_index": update,
            "parent_team_hash": parent,
            "selected_member": target,
            "feasibility": feasibility[update],
            "evidence": packet,
            "transition": transition,
            "accepted_candidate_ids": [row["candidate_id"] for row in rows],
            "saturation": event["telemetry"].get("saturation"),
            "epoch_end_reason": event["telemetry"].get("epoch_end_reason"),
            "stop_reason": event["stop_reason"],
        })
        for row in rows:
            if any(field not in row for field in _CANDIDATE_FIELDS):
                raise ValueError("formal candidate trace lacks required observation")
            records.append({
                **common,
                "record_type": "LOCAL_ACCEPTED_CANDIDATE",
                "update_index": update,
                **{field: row[field] for field in _CANDIDATE_FIELDS},
            })
        parent = successor
    extra_feasibility = set(feasibility) - set(evidence)
    if extra_feasibility:
        if (len(extra_feasibility) != 1 or extra_feasibility != {len(opportunity_events)}
                or summary["stop_reason"] != "NO_FEASIBLE_LAYER2_OPPORTUNITY"):
            raise ValueError("formal feasibility trace has an unmatched parent")
        row = feasibility[len(opportunity_events)]
        if (row["parent_team_hash"] != parent or row["selected_member"] is not None
                or row["eligible_member_ids"]):
            raise ValueError("formal no-feasible terminal trace mismatch")
        records.append({
            **common, "record_type": "LAYER2_NO_FEASIBLE_STOP",
            "update_index": len(opportunity_events),
            "parent_team_hash": parent, "feasibility": row,
            "stop_reason": summary["stop_reason"],
        })
    if parent != summary["final_team_hash"]:
        raise ValueError("formal final state differs from trajectory")
    if set(candidates) - set(evidence):
        raise ValueError("formal candidate lies outside an opportunity")
    return tuple(records)
