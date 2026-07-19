import asyncio
import json
from types import SimpleNamespace

from multi_dataset_diverse_rl.behavior_profiles import behavior_distance, build_team_behavior_profiles
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.lineage import empty_lineage_state, update_lineage_state
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.quality_diversity import deterministic_probe_folds, hierarchical_quality_bands, select_stable_joint_team
from multi_dataset_diverse_rl.search_archive import (
    candidate_quality_bucket,
    cheap_prescreen,
    mechanism_is_novel,
    refill_requirements,
    select_joint_representatives,
    select_reproduction_parent,
    select_safe_archive,
)
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def candidate(name, candidate_type="task_specific_repair", accuracy_delta=0.0, c1_delta=0, c2_delta=0, sequence=("hard_elimination",)):
    return {
        "prompt": f"{name}.", "prompt_hash": name, "candidate_id": name,
        "metrics": {
            "candidate_type": candidate_type, "accuracy_delta": accuracy_delta,
            "depth1_net_delta": c1_delta, "depth2_net_delta": c2_delta,
            "candidate_target_accuracy": 0.7, "penalized_reward": 0.1,
            "mechanism_representation": {"normalized_operation_sequence": list(sequence), "mechanism_embedding": [1.0, 0.0]},
            "mechanism_novel": bool(sequence),
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


def test_stable_qd_config_rejects_invalid_search_relationships():
    invalid = (
        {"candidate_refill_max_rounds": -1},
        {"candidate_refill_candidates_per_round": 0},
        {"candidate_refill_max_unique_candidates_per_parent": 1, "num_candidates_per_parent": 2},
        {"joint_representative_beam_size": 7, "qd_archive_size_per_agent": 6},
        {"probation_max_accuracy_loss": 0.06, "catastrophic_target_accuracy_loss_epsilon": 0.05},
        {"probation_max_c1_loss_questions": 2, "candidate_c1_catastrophic_loss_questions": 2},
        {"probe_stability_fold_count": 3},
    )
    for overrides in invalid:
        try:
            Config(**overrides)
        except ValueError:
            continue
        raise AssertionError(f"invalid Stable-QD configuration was accepted: {overrides}")


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


def test_fractional_coverage_rate_uses_real_loss_counts():
    cfg = Config()
    item = candidate("one-c1-loss", sequence=("weighted_scoring",))
    item["metrics"].update({
        "depth1_net_delta": -0.1,
        "depth1_net_count": -1,
        "depth1_loss_count": 1,
        "depth2_net_delta": 0.0,
        "depth2_net_count": 0,
        "depth2_loss_count": 0,
    })
    assert candidate_quality_bucket(item, cfg) == "probation"


def test_fractional_coverage_rate_without_count_is_never_truncated_to_safe():
    cfg = Config()
    item = candidate("legacy-rate", sequence=("weighted_scoring",))
    item["metrics"].update({"depth1_net_delta": -0.1, "num_eval_samples": 10})
    assert candidate_quality_bucket(item, cfg) == "probation"


def test_candidate_type_cannot_self_report_mechanism_novelty_for_probation():
    cfg = Config()
    item = candidate("claimed-novel", "mechanism_alternative", accuracy_delta=-0.02, c1_delta=-1)
    item["metrics"]["mechanism_novel"] = False
    assert candidate_quality_bucket(item, cfg) == "catastrophic"


def test_non_novel_mechanism_alternative_is_catastrophic_even_without_quality_loss():
    cfg = Config()
    item = candidate("duplicate-mechanism", "mechanism_alternative")
    item["metrics"]["mechanism_novel"] = False
    assert candidate_quality_bucket(item, cfg) == "catastrophic"


def test_cheap_prescreen_rejects_duplicate_and_incomplete_candidates():
    item = candidate("duplicate")
    item["prompt"] = "unfinished"
    item["proposal"] = {"candidate_type": "task_specific_repair", "mechanism_steps": ["hard_elimination"]}
    assert {"incomplete_prompt", "duplicate_prompt"} <= set(cheap_prescreen(item, "parent", {"duplicate"}))


def test_mechanism_alternative_requires_observed_operation_change():
    parent = candidate("parent", sequence=("hard_elimination",))
    unchanged = candidate("unchanged", "mechanism_alternative", sequence=("hard_elimination",))
    unchanged["proposal"] = {"candidate_type": "mechanism_alternative", "mechanism_steps": ["hard_elimination"]}
    assert "mechanism_operation_unchanged" in cheap_prescreen(unchanged, "parent", set(), parent=parent)
    assert not mechanism_is_novel(unchanged, parent)
    changed = candidate("changed", "mechanism_alternative", sequence=("weighted_scoring",))
    assert mechanism_is_novel(changed, parent)


def test_generic_mechanism_steps_are_rejected_before_solver_evaluation():
    item = candidate("generic", "mechanism_alternative", sequence=())
    item["proposal"] = {
        "candidate_type": "mechanism_alternative",
        "mechanism_steps": [
            "Produce a compact reasoning trace",
            "Make the decision procedure visible",
            "Proceed with logical reasoning carefully",
            "Give exactly one final answer",
        ],
    }
    reasons = cheap_prescreen(item, "parent", set())
    assert "missing_substantive_mechanism_operation" in reasons


def test_probation_parent_is_chosen_before_safe_niche_without_opportunity():
    active = candidate("active")
    probation = candidate("probation", accuracy_delta=-0.02, c1_delta=-1, sequence=("weighted_scoring",))
    parent, source, _ = select_reproduction_parent(
        active, [active], [probation], {}, epoch=1, min_opportunities=1, allow_probation=True,
    )
    assert parent is probation
    assert source == "probation_niche"


def test_expired_probation_branch_is_removed_before_parent_selection():
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(probation_archive_ttl_updates=2)
    system.probation_expired_count = 0
    agent = AgentState("active")
    active = candidate("active")
    probation = candidate("probation", "mechanism_alternative", accuracy_delta=-0.02, c1_delta=-1)
    probation["probation_created_update"] = 1
    agent.safe_qd_archive = [active]
    agent.probation_archive = [probation]
    agent.optimizer_update_count_by_epoch = {"1": 3}
    system.agents = [agent]
    system._expire_probation_branches(1)
    parent, source, _ = select_reproduction_parent(
        active, agent.safe_qd_archive, agent.probation_archive, {},
        epoch=1, min_opportunities=1, allow_probation=True,
    )
    assert agent.probation_archive == []
    assert system.probation_expired_count == 1
    assert parent is None
    assert source == "active"


def test_stable_qd_archive_snapshot_records_real_archive_state(tmp_path):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(out_dir=str(tmp_path), method_version="v8_stable_qd_lineage")
    agent = AgentState("active")
    active = {
        "prompt": "active", "prompt_hash": "active-hash", "archive_bucket": "safe",
        "metrics": {"mechanism_representation": {"normalized_operation_sequence": ["verify"]}},
    }
    alternate = {
        "prompt": "alternate", "prompt_hash": "alternate-hash", "archive_bucket": "safe",
        "metrics": {"mechanism_representation": {"normalized_operation_sequence": ["compare"]}},
    }
    agent.safe_qd_archive = [active, alternate]
    agent.probation_archive = []
    agent.prompt_beam = [active, alternate]
    system.agents = [agent]
    system.quality_diversity_archive_history = []

    system._record_stable_qd_archive_snapshot(
        agent_id=0,
        epoch=2,
        step=10,
        evaluated=[active, alternate, {"archive_bucket": "catastrophic"}],
        parent_sources=["active", "safe_niche"],
    )

    row = system.quality_diversity_archive_history[0]
    assert row["epoch"] == 2 and row["step"] == 10 and row["agent_id"] == 0
    assert row["safe_archive_size"] == 2
    assert row["representative_count"] == 2
    assert row["safe_candidate_count"] == 2
    assert row["catastrophic_candidate_count"] == 1
    assert row["parent_sources"] == ["active", "safe_niche"]
    assert row["safe_prompt_hashes"] == ["active-hash", "alternate-hash"]
    system._flush_jsonl("quality_diversity_archive.jsonl", system.quality_diversity_archive_history)
    written = json.loads((tmp_path / "quality_diversity_archive.jsonl").read_text(encoding="utf-8"))
    assert written == row


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


def test_long_archive_keeps_incumbent_and_diverse_niches_when_over_capacity():
    rows = []
    for index in range(8):
        item = candidate(f"n{index}", sequence=(f"operation_{index}",))
        item["archive_bucket"] = "safe"
        item["metrics"]["mechanism_representation"]["mechanism_embedding"] = [float(index == 0), float(index != 0)]
        rows.append(item)
    archive = select_safe_archive(rows, "n0", 6)
    assert len(archive) == 6
    assert "n0" in {item["prompt_hash"] for item in archive}


def test_team_dependent_rescue_changes_with_peers():
    focal = [1, 0, 1]
    team_a = build_team_behavior_profiles([['A', 'B', 'A'], ['B', 'B', 'A'], ['B', 'A', 'A']], [focal, [0, 0, 1], [0, 0, 1]])
    team_b = build_team_behavior_profiles([['A', 'B', 'A'], ['A', 'B', 'A'], ['A', 'C', 'A']], [focal, [1, 1, 1], [1, 0, 1]])
    assert team_a[0]["rescue_vector"] != team_b[0]["rescue_vector"]
    assert team_a[0]["same_wrong_vector"] != team_b[0]["same_wrong_vector"]


def test_stable_probe_and_mechanism_caches_report_real_hits():
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(method_version="v8_stable_qd_lineage", agents=1, aggregation_mode="plurality")
    system.prompt_probe_cache = {}
    system.mechanism_embedding_cache = {}
    system.full_probe_cache_hit_count = 0
    system.full_probe_missing_pair_evaluation_count = 0
    system.mechanism_embedding_cache_hit_count = 0
    system.mechanism_embedding_cache_miss_count = 0
    system.solver_call_semaphore = asyncio.Semaphore(1)
    system.task_spec = SimpleNamespace(
        parse_gold=lambda answer, question=None: str(answer),
        match_answer=lambda left, right: left == right,
    )
    calls = {"solver": 0, "embedding": 0}

    async def solve_once(question, agent_id, prompt):
        calls["solver"] += 1
        return "trace", "A"

    class Encoder:
        def encode(self, rows, normalize_embeddings=True):
            calls["embedding"] += 1
            return [[1.0, 0.0] for _ in rows]

    system.solve_once = solve_once
    system._record_solver_rollout = lambda **kwargs: None
    system._load_embedding_model = lambda: Encoder()
    system._normalize_vector = lambda value: list(value)
    probe = [{"question": "q", "answer": "A"}]
    first = asyncio.run(system._evaluate_prompt_on_stable_probe(0, "prompt", probe, ["Hard elimination"]))
    second = asyncio.run(system._evaluate_prompt_on_stable_probe(0, "prompt", probe, ["Hard elimination"]))
    assert first["answer_vector"] == second["answer_vector"] == ["A"]
    assert calls == {"solver": 1, "embedding": 1}
    assert system.full_probe_missing_pair_evaluation_count == 1
    assert system.full_probe_cache_hit_count == 1
    assert system.mechanism_embedding_cache_miss_count == 1
    assert system.mechanism_embedding_cache_hit_count == 1


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


def test_fold_quality_failure_cannot_advance_lineage():
    cfg = Config(lineage_commit_required_snapshots=2)
    selected = {
        "prompt_hash": "p", "prompt": "p",
        "mechanism_representation": {"normalized_operation_sequence": ["hard_elimination"], "mechanism_embedding": [1.0]},
        "behavior_profile": {"correctness_vector": [1, 0], "error_vector": [0, 1], "rescue_vector": [0, 0], "accuracy": 0.5},
        "cross_fold_diversity_gap": 0.0, "fold_quality_gate_passed": False,
    }
    state = update_lineage_state(empty_lineage_state(), selected, epoch=1, quality_gate_passed=True, config=cfg)
    assert state["new_status"] == "uncommitted"
    assert state["reason"] == "quality_gate_failed"


def test_single_fold_agent_behavior_cannot_advance_lineage():
    cfg = Config(lineage_commit_required_snapshots=2)
    selected = {
        "prompt_hash": "p", "prompt": "p",
        "mechanism_representation": {"normalized_operation_sequence": ["hard_elimination"], "mechanism_embedding": [1.0]},
        "behavior_profile": {"correctness_vector": [1, 0], "error_vector": [0, 1], "rescue_vector": [0, 0], "accuracy": 0.5},
        "cross_fold_diversity_gap": 0.0, "fold_quality_gate_passed": True,
        "fold_behavior_stable": False,
    }
    state = update_lineage_state(empty_lineage_state(), selected, epoch=1, quality_gate_passed=True, config=cfg)
    assert state["new_status"] == "uncommitted"
    assert state["reason"] == "unstable_single_fold_specialization"
