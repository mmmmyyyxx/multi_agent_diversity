from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.protocol import (
    MAIN_ABLATION_MODULES,
    MAIN_ABLATION_SETTINGS,
    candidate_budget_contract,
    canonical_experiment_setting,
    experiment_protocol,
)
from multi_dataset_diverse_rl.responsibility import (
    MemberAwareRepairOpportunity,
    RepairLane,
    ResponsibilityState,
    initialize_repairability_state,
    repairability_adjusted_target_scores,
)
from multi_dataset_diverse_rl.versions import (
    CHECKPOINT_VERSION,
    EXPERIMENT_MATRIX_VERSION,
    METHOD_VERSION,
    PROTOCOL_RESOLUTION_VERSION,
    TARGET_SELECTION_VERSION,
)


def _budget(name: str):
    return candidate_budget_contract(
        name,
        candidates_per_target_branch=2,
        stage_b_budget_per_branch=2,
        stage_a_channel_top_k=2,
        representative_size=12,
        coverage_size=6,
        conversion_size=6,
        preservation_size=4,
    )


def _protocol(name: str):
    return experiment_protocol(
        name,
        initialization_mode="shared_identical",
        tie_policy="abstain",
        candidate_budget_contract=_budget(name),
    )


def _opportunity(agent_id: int, question_hash: str, margin: int):
    return MemberAwareRepairOpportunity(
        agent_id=agent_id,
        question_hash=question_hash,
        vote_flip_gain=0,
        margin_gain=margin,
        member_error=True,
        coverage_opportunity=False,
        conversion_opportunity=True,
        dominant_wrong_member=False,
        unique_correct=False,
        pivotal_correct=False,
        oracle_soft_utility_gain=0.0,
    )


def _assert_w1() -> float:
    state = ResponsibilityState()
    initialize_repairability_state(state, range(5))
    state.branch_failure_count_by_agent[0] = 8
    state.updates_since_selected_by_agent.update({0: 10, 1: 0})
    active = {
        0: (_opportunity(0, "target", 1),),
        1: (_opportunity(1, "maximum", 9),),
    }
    scores = repairability_adjusted_target_scores(
        active_assignments=active,
        state=state,
        seed=46,
        current_member_correct_counts=(45, 55, 45, 45, 45),
        initial_member_correct_counts=(45, 45, 45, 45, 45),
        member_uplift_tolerance=5,
        legal_assignments=active,
        service_portfolios=active,
        active_lane_by_agent={
            0: RepairLane.MARGIN_SUPPORT,
            1: RepairLane.MARGIN_SUPPORT,
        },
    )
    target = next(row for row in scores if row.agent_id == 0)
    expected = (0.23333333333333334 + 0.05) / 9
    assert abs(target.expected_update_value - expected) < 1e-12
    assert abs(target.expected_update_value - (0.23333333333333334 / 9 + 0.05)) > 1e-6
    return target.expected_update_value


def main() -> None:
    assert METHOD_VERSION == "member_aware_peer_state_v15"
    assert CHECKPOINT_VERSION == 25
    assert TARGET_SELECTION_VERSION == (
        "repairability_adjusted_expected_update_value_wait_coupled_v2"
    )
    assert EXPERIMENT_MATRIX_VERSION == "reduced_two_module_ablation_v1"
    assert PROTOCOL_RESOLUTION_VERSION == (
        "reduced_two_module_protocol_resolution_v1"
    )
    assert MAIN_ABLATION_SETTINGS == (
        "shared_static_reference",
        "shared_generic_evolution",
        "shared_member_aware_dual_target",
        "shared_responsibility_conditioned_dual_target",
    )
    protocols = [_protocol(name) for name in MAIN_ABLATION_SETTINGS]
    assert [row.target_branch_count for row in protocols] == [0, 1, 2, 2]
    assert protocols[0].module_vector is None
    assert [
        MAIN_ABLATION_MODULES[name].as_tuple()
        for name in MAIN_ABLATION_SETTINGS[1:]
    ] == [(False, False), (True, False), (True, True)]
    for protocol in protocols[1:]:
        assert protocol.candidate_acceptance_policy == (
            "fixed_peer_monotone_target_or_vote"
        )
        assert protocol.candidate_ranking_policy == "common_monotone_safe"
        assert protocol.stage_a_policy == "matched_all_generated"
    full = protocols[-1]
    assert full.name == "shared_responsibility_conditioned_dual_target"
    assert full.target_branch_count == 2
    assert full.tcs_context_policy == (
        "member_aware_responsibility_conditioned"
    )
    try:
        canonical_experiment_setting("shared_full_dual_target_rcru")
    except ValueError as exc:
        assert "allow_legacy_setting=1" in str(exc)
    else:
        raise AssertionError("v14 RCRU setting did not fail closed")
    legacy = canonical_experiment_setting(
        "shared_full_dual_target_rcru",
        allow_legacy_setting=True,
    )
    assert legacy == "legacy_v14_shared_full_dual_target_rcru"
    print(json.dumps({
        "ok": True,
        "settings": list(MAIN_ABLATION_SETTINGS),
        "full_method_setting": full.name,
        "w1_expected_update_value": _assert_w1(),
        "main_rcru_enabled": False,
        "legacy_rcru_setting": legacy,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
