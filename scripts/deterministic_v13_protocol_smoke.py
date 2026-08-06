from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.protocol import (
    AUXILIARY_SEARCH_CONTROL_SETTINGS,
    MAIN_ABLATION_MODULES,
    MAIN_ABLATION_SETTINGS,
    candidate_budget_contract,
    canonical_experiment_setting,
    experiment_protocol,
)
from multi_dataset_diverse_rl.versions import (
    CHECKPOINT_VERSION,
    EXPERIMENT_MATRIX_VERSION,
    METHOD_VERSION,
    PROTOCOL_RESOLUTION_VERSION,
)


def _protocol(name: str):
    budget = candidate_budget_contract(
        name,
        candidates_per_target_branch=2,
        stage_b_budget_per_branch=2,
        stage_a_channel_top_k=2,
        representative_size=12,
        coverage_size=6,
        conversion_size=6,
        preservation_size=4,
    )
    return experiment_protocol(
        name,
        initialization_mode="shared_identical",
        tie_policy="abstain",
        candidate_budget_contract=budget,
        allow_auxiliary_setting=name in AUXILIARY_SEARCH_CONTROL_SETTINGS,
    )


def main() -> None:
    protocols = [_protocol(name) for name in MAIN_ABLATION_SETTINGS]
    assert METHOD_VERSION == "member_aware_peer_state_v13"
    assert CHECKPOINT_VERSION == 22
    assert EXPERIMENT_MATRIX_VERSION == "reduced_three_module_ablation_v1"
    assert PROTOCOL_RESOLUTION_VERSION == (
        "reduced_three_module_protocol_resolution_v1"
    )
    assert len(protocols) == 5
    assert [row.target_branch_count for row in protocols] == [
        0, 1, 2, 2, 2
    ]
    assert protocols[0].module_vector is None
    assert [
        tuple(int(value) for value in MAIN_ABLATION_MODULES[name].as_tuple())
        for name in MAIN_ABLATION_SETTINGS[1:]
    ] == [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
    ]
    for name, expected in zip(
        AUXILIARY_SEARCH_CONTROL_SETTINGS,
        ((2, 1, 2), (1, 4, 4)),
        strict=True,
    ):
        budget = _protocol(name).candidate_budget_contract
        assert (
            budget.target_branch_count,
            budget.candidates_per_target_branch,
            budget.total_generated_candidates_per_update,
        ) == expected
        assert name not in MAIN_ABLATION_SETTINGS
    try:
        canonical_experiment_setting("shared_full_rcru")
    except ValueError as exc:
        assert "allow_legacy_setting=1" in str(exc)
    else:
        raise AssertionError("v12 setting did not fail closed")
    print(json.dumps({
        "ok": True,
        "settings": list(MAIN_ABLATION_SETTINGS),
        "branch_counts": [row.target_branch_count for row in protocols],
        "auxiliary_settings": list(AUXILIARY_SEARCH_CONTROL_SETTINGS),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
