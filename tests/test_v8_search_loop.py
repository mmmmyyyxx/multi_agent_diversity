from types import SimpleNamespace

from multi_dataset_diverse_rl.behavior_profiles import behavior_distance, build_team_behavior_profiles
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.lineage import empty_lineage_state, update_lineage_state
from multi_dataset_diverse_rl.quality_diversity import deterministic_probe_folds, hierarchical_quality_bands, select_stable_joint_team
from multi_dataset_diverse_rl.search_archive import (
    candidate_quality_bucket,
    cheap_prescreen,
    refill_requirements,
    select_joint_representatives,
    select_reproduction_parent,
    select_safe_archive,
)


def candidate(name, candidate_type="task_specific_repair", accuracy_delta=0.0, c1_delta=0, c2_delta=0, sequence=("hard_elimination",)):
    return {
        "prompt": f"{name}.", "prompt_hash": name, "candidate_id": name,
        "metrics": {
            "candidate_type": candidate_type, "accuracy_delta": accuracy_delta,
            "depth1_net_delta": c1_delta, "depth2_net_delta": c2_delta,
            "candidate_target_accuracy": 0.7, "penalized_reward": 0.1,
            "mechanism_representation": {"normalized_operation_sequence": list(sequence), "mechanism_embedding": [1.0, 0.0]},
        },
    }


def test_refill_requirements_trigger_then_stop_when_safe_types_arrive():
    cfg = Config()
    failed = [candidate("bad", accuracy_delta=-0.1), candidate("duplicate", candidate_type="mechanism_alternative", sequence=())]
    for item in failed:
        item["archive_bucket"] = candidate_quality_bucket(item, cfg)
    assert not refill_requirements(failed, cfg)["met"]
    safe = [candidate("repair"), candidate("alternative", "mechanism_alternative", sequence=("weighted_scoring",))]
    for item in safe:
        item["archive_bucket"] = "safe"
    result = refill_requirements(safe, cfg)
    assert result["met"]
    assert result["safe_non_incumbent_count"] == 2


def test_probation_is_not_safe_but_retains_small_novel_regression():
    cfg = Config()
    item = candidate("probation", accuracy_delta=-0.02, c1_delta=-1, sequence=("weighted_scoring",))
    assert candidate_quality_bucket(item, cfg) == "probation"
    archive = select_safe_archive([item], "none", 6)
    assert archive == []


def test_candidate_bucket_distinguishes_safe_probation_and_catastrophic():
    cfg = Config()
    assert candidate_quality_bucket(candidate("safe"), cfg) == "safe"
    assert candidate_quality_bucket(candidate("probation", accuracy_delta=-0.02, c1_delta=-1), cfg) == "probation"
    assert candidate_quality_bucket(candidate("catastrophic", accuracy_delta=-0.06), cfg) == "catastrophic"


def test_cheap_prescreen_rejects_duplicate_and_incomplete_candidates():
    item = candidate("duplicate")
    item["prompt"] = "unfinished"
    item["proposal"] = {"candidate_type": "task_specific_repair", "mechanism_steps": ["hard_elimination"]}
    assert {"incomplete_prompt", "duplicate_prompt"} <= set(cheap_prescreen(item, "parent", {"duplicate"}))


def test_probation_parent_is_chosen_before_safe_niche_without_opportunity():
    active = candidate("active")
    probation = candidate("probation", accuracy_delta=-0.02, c1_delta=-1, sequence=("weighted_scoring",))
    parent, source, _ = select_reproduction_parent(
        active, [active], [probation], {}, epoch=1, min_opportunities=1, allow_probation=True,
    )
    assert parent is probation
    assert source == "probation_niche"


def test_safe_niche_receives_round_robin_parent_opportunity():
    active = candidate("active", sequence=("hard_elimination",))
    niche = candidate("niche", sequence=("counterfactual_check",))
    parent, source, _ = select_reproduction_parent(
        active, [active, niche], [], {}, epoch=1, min_opportunities=1, allow_probation=True,
    )
    assert parent is niche
    assert source == "safe_niche"


def test_long_archive_and_representatives_are_separate():
    rows = []
    for index in range(6):
        item = candidate(f"n{index}", sequence=(f"operation_{index}",))
        item["archive_bucket"] = "safe"
        rows.append(item)
    archive = select_safe_archive(rows, "n0", 6)
    representatives = select_joint_representatives(archive, "n0", 3)
    assert len(archive) == 6
    assert len(representatives) == 3


def test_team_dependent_rescue_changes_with_peers():
    focal = [1, 0, 1]
    team_a = build_team_behavior_profiles([['A', 'B', 'A'], ['B', 'B', 'A'], ['B', 'A', 'A']], [focal, [0, 0, 1], [0, 0, 1]])
    team_b = build_team_behavior_profiles([['A', 'B', 'A'], ['A', 'B', 'A'], ['A', 'B', 'A']], [focal, [1, 1, 1], [1, 0, 1]])
    assert team_a[0]["rescue_vector"] != team_b[0]["rescue_vector"]


def test_two_fold_order_is_deterministic_and_gap_penalizes_stability():
    folds = deterministic_probe_folds(["q0", "q1", "q2", "q3"], seed=42)
    assert folds == deterministic_probe_folds(["q0", "q1", "q2", "q3"], seed=42)
    assert sorted(folds[0] + folds[1]) == [0, 1, 2, 3]


def test_hierarchical_band_rejects_vote_below_band():
    cfg = Config(joint_vote_band_questions=1)
    incumbent = {"vote_acc": 1.0, "mean_individual_acc": 1.0, "bottom2_mean_acc": 1.0, "coverage_depth_c1": 1.0, "coverage_depth_c2": 1.0, "per_agent_acc": [1.0] * 5, "answer_vectors": [["A"] * 4] * 5, "vote_correct_count": 4, "total_agent_correct_count": 20, "bottom2_correct_count": 8, "per_agent_correct_count": [4] * 5}
    lower = {**incumbent, "vote_correct_count": 2, "vote_acc": 0.5}
    bands = hierarchical_quality_bands([incumbent, lower], incumbent, cfg)
    assert bands["final"] == [incumbent]


def test_same_wrong_dispersion_rewards_different_wrong_answers():
    same = behavior_distance({"correctness_vector": [0, 0], "error_vector": [1, 1], "rescue_vector": [0, 0], "answer_vector": ["B", "B"]}, {"correctness_vector": [0, 0], "error_vector": [1, 1], "rescue_vector": [0, 0], "answer_vector": ["B", "B"]})
    different = behavior_distance({"correctness_vector": [0, 0], "error_vector": [1, 1], "rescue_vector": [0, 0], "answer_vector": ["B", "C"]}, {"correctness_vector": [0, 0], "error_vector": [1, 1], "rescue_vector": [0, 0], "answer_vector": ["C", "B"]})
    assert different["wrong_answer_dispersion"] > same["wrong_answer_dispersion"]


def test_two_stable_snapshots_commit_lineage():
    cfg = Config(lineage_commit_required_snapshots=2)
    selected = {"prompt_hash": "p", "prompt": "p", "mechanism_representation": {"normalized_operation_sequence": ["hard_elimination"], "mechanism_embedding": [1.0]}, "behavior_profile": {"correctness_vector": [1, 0], "error_vector": [0, 1], "rescue_vector": [0, 0], "accuracy": 0.5}, "cross_fold_diversity_gap": 0.0}
    first = update_lineage_state(empty_lineage_state(), selected, epoch=1, quality_gate_passed=True, config=cfg)
    second = update_lineage_state({key: value for key, value in first.items() if key not in {"old_status", "new_status", "reason"}}, selected, epoch=2, quality_gate_passed=True, config=cfg)
    assert first["new_status"] == "provisional"
    assert second["new_status"] == "committed"
