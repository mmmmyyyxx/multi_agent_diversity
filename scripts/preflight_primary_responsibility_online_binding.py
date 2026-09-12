"""Zero-API source and semantics preflight for the opt-in online binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (
    PersistentRealizabilityState,
    PrimaryResponsibilityPersistentRealizabilityScheduler,
    build_primary_responsibility_summary_from_counts,
    select_primary_responsibility_targets,
)
from multi_dataset_diverse_rl.versions import (
    PERSISTENT_REALIZABILITY_SEMANTICS_VERSION,
    PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
)


MANIFEST = ROOT / "experiments/manifests/primary_responsibility_persistent_realizability_v1.yaml"
PROTOCOL = ROOT / "experiments/primary_responsibility_persistent_realizability_v1/PROTOCOL.md"
SOURCE_PATHS = (
    "multi_dataset_diverse_rl/versions.py",
    "multi_dataset_diverse_rl/team_search/controller.py",
    "multi_dataset_diverse_rl/team_search/primary_responsibility_binding.py",
    "multi_dataset_diverse_rl/team_search/primary_responsibility_scheduler.py",
    "multi_dataset_diverse_rl/team_search/task_builder.py",
    "multi_dataset_diverse_rl/team_search/candidate_selector.py",
    "experiments/primary_responsibility_persistent_realizability_v1/PROTOCOL.md",
    "experiments/manifests/primary_responsibility_persistent_realizability_v1.yaml",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    protocol = PROTOCOL.read_text(encoding="utf-8")

    one_positive = build_primary_responsibility_summary_from_counts(
        member_id=0,
        direct_count=1,
        near_margin_count=0,
        coverage_count=0,
        failure_count=0,
    )
    zero_rows = tuple(
        build_primary_responsibility_summary_from_counts(
            member_id=member_id,
            direct_count=0,
            near_margin_count=0,
            coverage_count=0,
            failure_count=0,
        )
        for member_id in range(1, 5)
    )
    selection = select_primary_responsibility_targets(
        (one_positive, *zero_rows),
        seed=19,
        update_index=4,
        rr_cursor=2,
    )
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler()
    decision = scheduler.select(
        assigned={},
        current_margin_by_question={},
        seed=19,
        update_index=4,
    )
    scheduler.record_outcome(
        decision=decision,
        update_index=4,
        committed_member_id=None,
        valid_outcome=False,
    )
    payload = scheduler.state.checkpoint_payload()
    restored = PersistentRealizabilityState.from_checkpoint_payload(
        payload, member_ids=scheduler.member_ids
    )
    duplicate_failed_closed = False
    try:
        scheduler.record_outcome(
            decision=decision,
            update_index=4,
            committed_member_id=None,
            valid_outcome=False,
        )
    except ValueError as exc:
        duplicate_failed_closed = "already recorded" in str(exc)

    assertions = {
        "api_authorized_false": manifest["api_authorization"]["authorized"] is False,
        "budget_not_frozen": manifest["budget"]["frozen_before_run"] is False,
        "duplicate_outcome_fails_closed": duplicate_failed_closed,
        "exact_once_survives_checkpoint": restored.completed_update_indices == {4},
        "manifest_status_implemented": manifest["status"] == "IMPLEMENTED",
        "models_not_frozen": manifest["model"]["solver"] is None,
        "prospective_seeds_not_frozen": manifest["seeds"] == [],
        "protocol_freezes_always_two": "always_two_targets" in protocol,
        "protocol_uses_write_back_term": "eventual write-back realizability" in protocol,
        "runtime_scheduler_identity": (
            selection.scheduler_version
            == PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION
        ),
        "semantics_identity_persisted": (
            payload["realizability_semantics"]
            == PERSISTENT_REALIZABILITY_SEMANTICS_VERSION
        ),
        "single_positive_still_has_two_targets": (
            len(selection.selected_member_ids) == 2
            and selection.selected_member_ids[0] == 0
        ),
        "test_policy_prohibited": manifest["data"]["test_policy"] == "prohibited; zero calls",
    }
    result = {
        "schema_version": "primary_responsibility_online_binding_preflight_v1",
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "assertions": assertions,
        "identities": {
            "scheduler": PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
            "realizability_semantics": PERSISTENT_REALIZABILITY_SEMANTICS_VERSION,
            "target_count_contract": "always_two_targets",
            "outcome_accounting": "exactly_once_per_update_index",
        },
        "source_sha256": {relative: sha256(ROOT / relative) for relative in SOURCE_PATHS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["status"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
