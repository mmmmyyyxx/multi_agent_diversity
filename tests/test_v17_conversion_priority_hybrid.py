from __future__ import annotations

from scripts.build_v17_conversion_priority_hybrid_registry import PARENTS, build_registry
from scripts.v17_conversion_priority_hybrid_support import (
    ARMS,
    BASE,
    BREADTH,
    DIRECT,
    arm_specs,
    classify,
)


def test_registry_is_exact_five_parent_three_arm_design() -> None:
    registry = build_registry("0" * 40)
    assert [(row["source_seed"], row["source_update_index"]) for row in registry["cases"]] == list(PARENTS)
    assert registry["case_count"] == 5
    assert registry["cell_count"] == 15
    assert registry["conceptual_branch_count"] == 30
    assert registry["deduplicated_branch_count"] == 15
    assert registry["actual_source_slot_budget"] == 30
    assert registry["arms"] == list(ARMS)


def test_target1_is_shared_and_target2_interventions_are_exact() -> None:
    registry = build_registry("0" * 40)
    breadth = direct = cross = 0
    for case in registry["cases"]:
        arms = arm_specs(case)
        assert len({rows[0]["target_member"] for rows in arms.values()}) == 1
        breadth += arms[BREADTH][1]["target_member"] != arms[BASE][1]["target_member"]
        direct += arms[DIRECT][1]["target_member"] != arms[BASE][1]["target_member"]
        cross += arms[BREADTH][1]["target_member"] != arms[DIRECT][1]["target_member"]
    assert (breadth, direct, cross) == (3, 3, 2)


def test_priority_rules_use_single_counts_with_rr_ties() -> None:
    registry = build_registry("0" * 40)
    for case in registry["cases"]:
        arms = arm_specs(case)
        scores = case["selector_scores_by_agent"]
        for arm, field in (
            (BREADTH, "conversion_responsibility_count"),
            (DIRECT, "direct_vote_flip_count"),
        ):
            selected = str(arms[arm][1]["target_member"])
            assert scores[selected][field] == max(
                int(row[field]) for agent, row in scores.items()
                if int(agent) != arms[arm][0]["target_member"]
            )


def test_classifier_covers_breadth_direct_both_and_null() -> None:
    def rows(bd, bv, dv, dc):
        return [
            {
                BASE: {"deeper_support_gain_count": 0, "validation_vote_delta": 0, "vote_conversion_count": 0, "validation_oracle_delta": 0},
                BREADTH: {"deeper_support_gain_count": bd[i], "validation_vote_delta": bv[i], "vote_conversion_count": 0, "validation_oracle_delta": 0},
                DIRECT: {"deeper_support_gain_count": 0, "validation_vote_delta": dv[i], "vote_conversion_count": dc[i], "validation_oracle_delta": 0},
            }
            for i in range(3)
        ]
    funnel = {arm: {"feasible_branch_count": 2, "would_commit_count": 2} for arm in ARMS}
    assert classify(rows([1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]), funnel)["final_diagnosis"] == "BREADTH_LOCAL_SIGNAL"
    assert classify(rows([0, 0, 0], [0, 0, 0], [1, 1, 0], [1, 1, 0]), funnel)["final_diagnosis"] == "DIRECT_FLIP_LOCAL_SIGNAL"
    assert classify(rows([1, 1, 0], [0, 0, 0], [1, 1, 0], [1, 1, 0]), funnel)["final_diagnosis"] == "BOTH_LOCAL_SIGNALS"
    assert classify(rows([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]), funnel)["final_diagnosis"] == "NO_CLEAR_PRIORITY_SIGNAL"


def test_runner_has_no_test_or_commit_path_and_freezes_validation_order() -> None:
    from pathlib import Path

    source = Path("scripts/run_v17_conversion_priority_hybrid_pilot.py").read_text(encoding="utf-8")
    assert source.index("decisions[arm]") < source.index("parent_validation, validation")
    assert "evaluate_final_test" not in source
    assert "test_path" not in source
    assert '"team_prompt_commit_count": 0' in source
    assert '"trajectory_mutation_count": 0' in source
    assert '"decision_frozen_before_validation": True' in source


def test_runner_candidate_log_does_not_persist_raw_content() -> None:
    from pathlib import Path

    source = Path("scripts/run_v17_conversion_priority_hybrid_pilot.py").read_text(encoding="utf-8")
    candidate_block = source[source.index("candidates.append({"):source.index("cell = {")]
    assert '"prompt"' not in candidate_block
    assert '"question"' not in candidate_block
    assert '"answer"' not in candidate_block
