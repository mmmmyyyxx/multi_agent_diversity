"""Sanitized, deterministic hashes for V4 Layer-2 evidence provenance."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


def _hash_ids(ids: Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps(ids, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sanitize_v4_evidence_trace(row: Mapping[str, Any]) -> dict[str, Any]:
    """Publish counts/hashes, never BBH IDs, prompts, questions, or answers."""

    universe = row["responsibility_universe"]
    scheduled = row.get("responsibility_scheduled", {})
    scheduled_ids = tuple(scheduled.get("ids", row.get("responsibility_scheduled_ids", ())))
    focus = tuple(row["focus_ids"])
    anchor = tuple(row["anchor_ids"])
    nominal = tuple(tuple(batch) for batch in row["nominal_schedule"])
    delivered = row["evidence_delivered"]
    batches = tuple(tuple(batch) for batch in delivered.get("batch_ids", delivered.get("delivered_batch_ids", ())))
    role_items = tuple(delivered.get("role_item_ids", delivered.get("evidence_delivered_role_item_ids", ())))
    sources = tuple(delivered.get("source_ids", delivered.get("evidence_delivered_source_ids", ())))
    not_delivered = delivered.get(
        "scheduled_but_not_delivered_count",
        delivered.get("scheduled_but_not_delivered_role_item_count"),
    )
    if not_delivered is None:
        raise ValueError("V4 delivered evidence is missing schedule reconciliation")
    if int(universe["responsibility_universe_count"]) < len(scheduled_ids):
        raise ValueError("scheduled responsibility exceeds the full legal universe")
    if len(focus) + len(anchor) + len(scheduled_ids) > 36:
        raise ValueError("bounded V4 nominal evidence exceeds 36 role items")
    if len(set(role_items)) + int(not_delivered) != len(set(
        (f"responsibility:{item}" for item in scheduled_ids)
    ).union(f"focus:{item}" for item in focus).union(f"anchor:{item}" for item in anchor)):
        # The packet can role-qualify repeated source IDs; the public audit
        # checks identity conservation without exposing any raw source ID.
        raise ValueError("V4 scheduled/delivered role-item accounting mismatch")
    return {
        "update_index": int(row["update_index"]),
        "target_member": int(row["target_member"]),
        "parent_team_hash": row["parent_team_hash"],
        "successor_team_hash": row.get("successor_team_hash"),
        "raw_V": row["raw_V"],
        "responsibility_universe_count": int(universe["responsibility_universe_count"]),
        "responsibility_universe_ids_sha256": universe["responsibility_universe_ids_sha256"],
        "responsibility_scheduled_count": len(scheduled_ids),
        "responsibility_scheduled_ids_sha256": _hash_ids(scheduled_ids),
        "focus_count": len(focus), "focus_ids_sha256": _hash_ids(focus),
        "anchor_count": len(anchor), "anchor_ids_sha256": _hash_ids(anchor),
        "nominal_batch_count": len(nominal),
        "nominal_role_item_slots": sum(len(batch) for batch in nominal),
        "nominal_schedule_sha256": _hash_ids(nominal),
        "packet_hash": row["packet_hash"],
        "delivered_batch_count": len(batches),
        "delivered_batch_ids_sha256": _hash_ids(batches),
        "delivered_role_item_count": len(role_items),
        "delivered_role_item_ids_sha256": _hash_ids(role_items),
        "delivered_source_count": len(sources),
        "delivered_source_ids_sha256": _hash_ids(sources),
        "delivered_role_counts": {
            role: sum(str(item).startswith(f"{role}:") for item in role_items)
            for role in ("responsibility", "focus", "anchor")
        },
        "scheduled_but_not_delivered_count": int(not_delivered),
    }
