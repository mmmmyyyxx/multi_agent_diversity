from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.qd.quality_anchors import (
    build_quality_anchor,
    real_anchor_feasible,
    update_quality_anchor_archive,
)
from multi_dataset_diverse_rl.quality_diversity import hierarchical_quality_bands


def team(name, vote, total, bottom2, c1, c2, per_agent):
    size = 10
    return {
        "prompt_hashes": [name] * len(per_agent), "answer_vectors": [["A"] * size] * len(per_agent),
        "vote_correct_count": vote, "total_agent_correct_count": total,
        "bottom2_correct_count": bottom2, "coverage_depth_c1_correct_count": c1,
        "coverage_depth_c2_correct_count": c2, "per_agent_correct_count": list(per_agent),
        "vote_acc": vote / size, "mean_individual_acc": total / (size * len(per_agent)),
        "bottom2_mean_acc": bottom2 / 20, "coverage_depth_c1": c1 / size,
        "coverage_depth_c2": c2 / size, "per_agent_acc": [value / size for value in per_agent],
    }


def strict_config():
    return Config(
        joint_allowed_vote_loss_questions=0, joint_allowed_total_agent_correct_loss=0,
        joint_allowed_bottom2_correct_loss=0, joint_allowed_c1_loss_questions=0,
        joint_allowed_c2_loss_questions=0, joint_allowed_per_agent_correct_loss=0,
    )


def test_tradeoff_real_teams_do_not_form_synthetic_componentwise_anchor():
    vote_anchor = build_quality_anchor(team("vote", 9, 32, 10, 9, 6, [8, 7, 6, 6, 5]), epoch=1, created_order=1)
    individual_anchor = build_quality_anchor(team("individual", 7, 40, 15, 8, 8, [8, 8, 8, 8, 8]), epoch=2, created_order=2)
    archive = update_quality_anchor_archive([], [vote_anchor, individual_anchor])
    assert {anchor.anchor_id for anchor in archive} == {vote_anchor.anchor_id, individual_anchor.anchor_id}
    assert not any(anchor.counts.vote == 9 and anchor.counts.total_agent_correct == 40 for anchor in archive)


def test_candidate_only_needs_to_satisfy_one_real_anchor():
    cfg = strict_config()
    first = build_quality_anchor(team("first", 9, 30, 10, 9, 5, [6] * 5), epoch=1, created_order=1)
    second = build_quality_anchor(team("second", 7, 40, 14, 8, 8, [8] * 5), epoch=2, created_order=2)
    candidate = team("candidate", 7, 40, 14, 8, 8, [8] * 5)
    assert real_anchor_feasible(candidate, [first, second], cfg)


def test_old_real_anchor_blocks_cumulative_epoch_regression():
    cfg = Config(
        joint_allowed_vote_loss_questions=1, joint_allowed_total_agent_correct_loss=2,
        joint_allowed_bottom2_correct_loss=1, joint_allowed_c1_loss_questions=1,
        joint_allowed_c2_loss_questions=1, joint_allowed_per_agent_correct_loss=1,
    )
    initial = build_quality_anchor(team("initial", 9, 40, 15, 9, 8, [8] * 5), epoch=0, created_order=0)
    incumbent = team("incumbent", 8, 38, 14, 8, 7, [8, 8, 8, 7, 7])
    cumulative = team("cumulative", 7, 36, 13, 7, 6, [7, 7, 7, 7, 8])
    bands = hierarchical_quality_bands([cumulative], incumbent, cfg, quality_anchors=[initial])
    assert bands["quality_floor"] == []
    assert bands["quality_anchor_fallback_reason"] == "no_real_anchor_feasible"
    assert bands["final"] == [incumbent]


def test_dominated_anchor_is_removed_and_capacity_is_limited():
    anchors = [
        build_quality_anchor(team(f"a{i}", 5 + i, 25 + i, 8 + i, 5 + i, 4 + i, [5 + i] * 5), epoch=i, created_order=i)
        for i in range(7)
    ]
    archive = update_quality_anchor_archive([], anchors, capacity=5)
    assert len(archive) == 1
    assert archive[0].prompt_hashes == ["a6"] * 5

    tradeoffs = [
        build_quality_anchor(team(f"t{i}", 10 - i, 30 + i, 10 + i, 9 - i // 2, 5 + i // 2, [6 + (i % 2)] * 5), epoch=i, created_order=i)
        for i in range(8)
    ]
    assert len(update_quality_anchor_archive([], tradeoffs, capacity=5)) <= 5
