import asyncio
from types import SimpleNamespace

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.quality_diversity import (
    active_prompt_change_count,
    enumerate_joint_teams,
    epsilon_quality_frontier,
    quality_feasible,
    select_stable_joint_team,
    team_diversity_metrics,
)
from multi_dataset_diverse_rl.utils import plurality_vote_with_diagnostics
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def profile(name, correctness, sequence, embedding):
    return {
        "prompt_hash": name,
        "answer_vector": ["A" if value else "B" for value in correctness],
        "correctness_vector": list(correctness),
        "mechanism_representation": {
            "normalized_operation_sequence": list(sequence),
            "mechanism_embedding": list(embedding),
        },
    }


def exact(left, right):
    return left == right


def test_enumerates_all_243_five_agent_three_slot_teams_offline():
    beams = []
    for agent_id in range(5):
        beams.append([
            profile(f"a{agent_id}p{slot}", [1, slot % 2], [f"mechanism_{slot}"], [float(slot), 1.0])
            for slot in range(3)
        ])
    teams = enumerate_joint_teams(
        beams, ["A", "A"], ["q0", "q1"],
        vote_fn=plurality_vote_with_diagnostics, match_fn=exact,
        tie_break_method="random", seed=42,
    )
    assert len(teams) == 3 ** 5


def test_active_change_limit_uses_prompt_hashes_not_beam_indices():
    incumbent_profiles = [profile(f"p{i}", [1, 1], ["hard_elimination"], [1.0, 0.0]) for i in range(5)]
    incumbent = {"beam_indices": [2] * 5, "prompt_profiles": incumbent_profiles}
    same_hashes = {"beam_indices": [0] * 5, "prompt_profiles": [dict(item) for item in incumbent_profiles]}
    changed_hash = {"beam_indices": [2] * 5, "prompt_profiles": [{**incumbent_profiles[0], "prompt_hash": "new"}, *incumbent_profiles[1:]]}
    assert active_prompt_change_count(same_hashes, incumbent) == 0
    assert active_prompt_change_count(changed_hash, incumbent) == 1


def test_quality_constraints_exclude_diverse_but_degraded_team():
    incumbent = {
        "vote_acc": 0.8, "mean_individual_acc": 0.8, "bottom2_mean_acc": 0.75,
        "coverage_depth_c1": 0.9, "coverage_depth_c2": 0.8,
        "per_agent_acc": [0.8] * 5,
    }
    degraded = {**incumbent, "mean_individual_acc": 0.6, "coverage_depth_c1": 0.6, "per_agent_acc": [0.6] * 5}
    feasible = {**incumbent, "vote_acc": 0.79, "per_agent_acc": [0.79] * 5}
    eps = {key: 0.02 for key in ("vote_acc", "mean_individual_acc", "bottom2_mean_acc", "coverage_depth_c1", "coverage_depth_c2")}
    assert not quality_feasible(degraded, incumbent, [0.8] * 5, [-1.0] * 5, eps, 0.03)
    assert quality_feasible(feasible, incumbent, [0.8] * 5, [-1.0] * 5, eps, 0.03)


def test_diversity_selects_complementary_team_only_within_quality_frontier():
    config = Config()
    collapsed_profiles = [profile(str(i), [1, 0, 1, 0], ["hard_elimination"], [1.0, 0.0]) for i in range(5)]
    complementary_profiles = [
        profile("c0", [1, 0, 1, 0], ["hard_elimination"], [1.0, 0.0]),
        profile("c1", [0, 1, 1, 0], ["weighted_scoring"], [0.0, 1.0]),
        profile("c2", [1, 0, 0, 1], ["counterfactual_check"], [0.5, 0.5]),
        profile("c3", [0, 1, 0, 1], ["timeline_construction"], [0.2, 0.8]),
        profile("c4", [1, 0, 1, 0], ["semantic_role_check"], [0.8, 0.2]),
    ]
    collapsed = {
        "prompt_profiles": collapsed_profiles,
        "answer_vectors": [row["answer_vector"] for row in collapsed_profiles],
        "correctness_vectors": [row["correctness_vector"] for row in collapsed_profiles],
    }
    complementary = {
        "prompt_profiles": complementary_profiles,
        "answer_vectors": [row["answer_vector"] for row in complementary_profiles],
        "correctness_vectors": [row["correctness_vector"] for row in complementary_profiles],
    }
    for team in (collapsed, complementary):
        team.update({
            "vote_acc": 0.75, "mean_individual_acc": 0.5, "bottom2_mean_acc": 0.5,
            "coverage_depth_c1": 1.0, "coverage_depth_c2": 0.75,
        })
    frontier = epsilon_quality_frontier([collapsed, complementary], {key: 0.0 for key in (
        "vote_acc", "mean_individual_acc", "bottom2_mean_acc", "coverage_depth_c1", "coverage_depth_c2"
    )})
    assert len(frontier) == 2
    assert team_diversity_metrics(complementary, config)["team_diversity_score"] > team_diversity_metrics(collapsed, config)["team_diversity_score"]


def test_mechanism_niche_candidate_can_become_jointly_active(tmp_path):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(
        out_dir=str(tmp_path), agents=5, method_version="v8_stable_qd_lineage",
        beam_size=3, vote_tie_break="random", aggregation_mode="plurality",
    )
    system.task_spec = SimpleNamespace(match_answer=exact)
    system.specialization_strength = 0.0
    system.initial_competence_probe_metrics = {"per_agent_acc": [0.5] * 5}
    system.behavior_profile_by_prompt_hash = {}
    system.joint_team_selection_history = []
    system.lineage_history = []
    system.quality_diversity_archive_history = []
    system.latest_joint_team_metrics = {}
    system.peer_collapse_soft_count = 0
    system.peer_collapse_hard_rejection_count = 0
    system.agents = [AgentState("incumbent") for _ in range(5)]
    for agent_id, agent in enumerate(system.agents):
        agent.current_prompt = f"inc-{agent_id}"
        agent.prompt_beam = [{
            "prompt": agent.current_prompt,
            "metrics": {"beam_slot": "incumbent", "mechanism_signature": ["hard_elimination"]},
        }]
    system.agents[0].prompt_beam.append({
        "prompt": "alternative-0",
        "metrics": {"beam_slot": "mechanism_niche", "mechanism_signature": ["weighted_scoring"]},
    })

    async def fake_profile(agent_id, prompt_text, probe_data, mechanism_steps=()):
        alternative = prompt_text == "alternative-0"
        correctness = [0, 1, 1, 0] if alternative else [1, 0, 1, 0]
        sequence = ["weighted_scoring"] if alternative else ["hard_elimination"]
        return profile(prompt_text, correctness, sequence, [0.0, 1.0] if alternative else [1.0, 0.0]) | {
            "accuracy": 0.5,
            "question_hashes": ["q0", "q1", "q2", "q3"],
            "gold_answers": ["A"] * 4,
        }

    system._evaluate_prompt_on_stable_probe = fake_profile
    result = asyncio.run(system.select_joint_active_team(
        [{"question": f"q{i}", "answer": "A"} for i in range(4)], epoch=1,
    ))
    assert system.agents[0].current_prompt == "alternative-0"
    assert result["selected_beam_sources"][0] == "mechanism_niche"
    assert result["combination_count"] == 2
    assert result["actual_combination_count"] == (
        result["theoretical_combination_count"]
        - result["combination_rejected_by_change_limit_count"]
    )
    assert result["vote_band_remaining_count"] == result["hierarchical_band_count_by_name"]["vote"]
    assert result["fold_a_diversity"] == result["fold_diversities"][0]
    assert result["fold_b_diversity"] == result["fold_diversities"][1]
    assert result["quality_constraints_passed"] is True
    assert result["quality_constraint_violation"] is False
    assert result["peer_collapse_penalty_mean"] == 0.0


def test_near_duplicate_peer_collapse_cannot_replace_incumbent_team():
    config = Config()
    profiles = [profile(f"p{i}", [1, 0], ["hard_elimination"], [1.0, 0.0]) for i in range(5)]
    for item in profiles:
        item["accuracy"] = 0.5
    quality = {
        "vote_acc": 0.5, "mean_individual_acc": 0.5, "bottom2_mean_acc": 0.5,
        "coverage_depth_c1": 0.5, "coverage_depth_c2": 0.5,
        "per_agent_acc": [0.5] * 5,
        "answer_vectors": [item["answer_vector"] for item in profiles],
        "correctness_vectors": [item["correctness_vector"] for item in profiles],
    }
    incumbent = {"beam_indices": [0] * 5, "prompt_profiles": profiles, **quality}
    collapsed = {
        "beam_indices": [1, 0, 0, 0, 0],
        "prompt_profiles": [{**profiles[0], "prompt_hash": "peer-copy"}, *profiles[1:]],
        **quality,
    }
    states = [AgentState("p").lineage_state for _ in range(5)]
    states[1].update({
        "lineage_status": "committed",
        "lineage_anchor_mechanism_signature": ["hard_elimination"],
        "lineage_anchor_mechanism_embedding": [1.0, 0.0],
        "lineage_anchor_correctness_vector": [1, 0],
        "lineage_anchor_rescue_vector": [0, 0],
        "lineage_anchor_accuracy": 0.5,
    })
    result = select_stable_joint_team([incumbent, collapsed], incumbent, [0.5] * 5, states, 2, config)
    assert result["selected"] is incumbent
    assert result["hard_rejection_count"] >= 1


def test_fold_catastrophic_quality_regression_is_rejected():
    config = Config(
        joint_allowed_total_agent_correct_loss=0,
        joint_allowed_c1_loss_questions=0,
        joint_allowed_c2_loss_questions=0,
    )
    incumbent_profiles = [profile(f"p{i}", [1, 1, 0, 0], ["hard_elimination"], [1.0, 0.0]) for i in range(5)]
    candidate_profiles = [
        profile(f"c{i}", [1, 1, 0, 0] if i else [0, 1, 1, 0], ["weighted_scoring"], [0.0, 1.0])
        for i in range(5)
    ]
    teams = enumerate_joint_teams(
        [[incumbent_profiles[i], candidate_profiles[i]] if i == 0 else [incumbent_profiles[i]] for i in range(5)],
        ["A"] * 4, ["q0", "q1", "q2", "q3"],
        vote_fn=plurality_vote_with_diagnostics, match_fn=exact, tie_break_method="random", seed=42,
    )
    incumbent = teams[0]
    result = select_stable_joint_team(
        teams, incumbent, [0.5] * 5, [AgentState("p").lineage_state for _ in range(5)], 4, config,
        gold_answers=["A"] * 4, question_hashes=["q0", "q1", "q2", "q3"],
        vote_fn=plurality_vote_with_diagnostics, match_fn=exact, tie_break_method="random", seed=42,
    )
    assert result["fold_quality_rejection_count"] >= 1
    assert result["selected"]["prompt_hashes"] == [item["prompt_hash"] for item in incumbent_profiles]
