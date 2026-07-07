import asyncio

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem
from multi_dataset_diverse_rl.tasks import get_task_spec


def _system_without_init(cfg: Config):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = cfg
    system.task_spec = get_task_spec("mmlu")
    return system


def test_coverage_rescue_reward_uses_target_accuracy_coverage_and_useful_diversity_only():
    system = _system_without_init(
        Config(
            reward_mode="coverage_rescue_diversity",
            reward_schedule_mode="static",
            reward_weight_coverage=0.3,
            reward_weight_useful_diversity=0.2,
        )
    )
    result = system._candidate_reward_coverage_rescue_diversity(
        baseline_team_accuracy=0.0,
        candidate_team_accuracy=1.0,
        baseline_target_accuracy=0.4,
        candidate_target_accuracy=0.4,
        baseline_invalid_rate=0.0,
        candidate_invalid_rate=0.0,
        rescue_rate=1.0,
        coverage_delta=0.25,
        useful_diversity=0.5,
    )
    assert result["invalid_guard_passed"] is True
    assert round(result["reward"], 6) == round(0.4 + 0.3 * 0.25 + 0.2 * 0.5, 6)
    assert result["coverage_delta"] == 0.25
    assert result["rescue_rate"] == 1.0


def test_coverage_rescue_reward_does_not_use_rescue_or_vote_delta():
    system = _system_without_init(Config(reward_mode="coverage_rescue_diversity"))
    with_rescue = system._candidate_reward_coverage_rescue_diversity(
        baseline_team_accuracy=0.0,
        candidate_team_accuracy=1.0,
        baseline_target_accuracy=0.4,
        candidate_target_accuracy=0.5,
        baseline_invalid_rate=0.0,
        candidate_invalid_rate=0.0,
        rescue_rate=1.0,
        coverage_delta=0.2,
        useful_diversity=0.3,
    )
    without_rescue = system._candidate_reward_coverage_rescue_diversity(
        baseline_team_accuracy=1.0,
        candidate_team_accuracy=0.0,
        baseline_target_accuracy=0.4,
        candidate_target_accuracy=0.5,
        baseline_invalid_rate=0.0,
        candidate_invalid_rate=0.0,
        rescue_rate=0.0,
        coverage_delta=0.2,
        useful_diversity=0.3,
    )
    assert with_rescue["reward"] == without_rescue["reward"]


def test_coverage_rescue_reward_penalizes_invalid_guard_failure():
    system = _system_without_init(Config(reward_mode="coverage_rescue_diversity", invalid_guard_epsilon=0.05))
    result = system._candidate_reward_coverage_rescue_diversity(
        baseline_team_accuracy=0.5,
        candidate_team_accuracy=0.7,
        baseline_target_accuracy=0.5,
        candidate_target_accuracy=0.8,
        baseline_invalid_rate=0.0,
        candidate_invalid_rate=0.2,
        rescue_rate=1.0,
        coverage_delta=0.5,
        useful_diversity=1.0,
    )
    assert result["invalid_guard_passed"] is False
    assert result["reward"] == -1.0


def test_coverage_rescue_reward_has_no_target_accuracy_guard():
    system = _system_without_init(Config(reward_mode="coverage_rescue_diversity", reward_schedule_mode="static"))
    result = system._candidate_reward_coverage_rescue_diversity(
        baseline_team_accuracy=0.8,
        candidate_team_accuracy=0.7,
        baseline_target_accuracy=0.8,
        candidate_target_accuracy=0.6,
        baseline_invalid_rate=0.0,
        candidate_invalid_rate=0.0,
        rescue_rate=1.0,
        coverage_delta=0.5,
        useful_diversity=1.0,
    )
    assert "target_guard_passed" not in result
    assert round(result["reward"], 6) == round(0.6 + 0.3 * 0.5 + 0.2 * 1.0, 6)


def test_coverage_useful_diversity_alias():
    new_system = _system_without_init(Config(reward_mode="coverage_useful_diversity"))
    old_system = _system_without_init(Config(reward_mode="coverage_rescue_diversity"))
    assert new_system._is_coverage_useful_diversity_mode()
    assert old_system._is_coverage_useful_diversity_mode()
    kwargs = {
        "baseline_team_accuracy": 0.0,
        "candidate_team_accuracy": 0.0,
        "baseline_target_accuracy": 0.2,
        "candidate_target_accuracy": 0.7,
        "baseline_invalid_rate": 0.0,
        "candidate_invalid_rate": 0.0,
        "rescue_rate": 1.0,
        "coverage_delta": 0.1,
        "useful_diversity": 0.4,
    }
    assert (
        new_system._candidate_reward_coverage_useful_diversity(**kwargs)["reward"]
        == old_system._candidate_reward_coverage_rescue_diversity(**kwargs)["reward"]
    )


def test_reward_agent_selection_prefers_error_agent():
    system = _system_without_init(Config(agents=3))
    system.agents = [object(), object(), object()]
    diagnosis = {
        "per_agent_error_count": [0, 2, 0],
        "per_agent_team_wrong_error_count": [0, 1, 0],
        "per_agent_invalid_rate": [0.0, 0.0, 0.0],
        "per_agent_overlap_pressure": [5.0, 0.0, 0.0],
        "homogeneous_case_counts": [0, 0, 0],
    }
    selected = system.select_reward_agents_for_update(diagnosis, metrics={})
    assert selected[0] == 1


def test_accuracy_only_reward_uses_target_agent_accuracy():
    system = _system_without_init(Config(reward_mode="accuracy_only", agents=5))
    system.agents = [object() for _ in range(5)]

    def active_prompts():
        return [f"p{i}" for i in range(5)]

    async def fake_solve(q, prompts, source=""):
        return (
            ["t0", "t1", "t2", "t3", "t4"],
            ["A", "B", "B", "B", "B"],
            {"solver_reuse_hits": 0, "solver_reuse_misses": 0, "solver_calls": 0, "solver_reuse_total": 0},
        )

    def fake_rollout(traces, answers, gold, prompts=None, question_hash=""):
        return {
            "vote_correct": 0,
            "vote_answer": "B",
            "vote_tie": False,
            "tie_candidates": [],
            "vote_counts": {"B": 4, "A": 1},
            "tie_break_method": "first",
            "majority_vote_answer": "B",
            "weighted_vote_answer": "B",
            "majority_vote_correct": 0,
            "weighted_vote_correct": 0,
            "aggregation_mode": "majority",
        }

    system._active_prompt_list = active_prompts
    system.solve_with_prompts_reusing_records = fake_solve
    system.compute_rollout_metrics = fake_rollout
    system._hash = lambda value: "hash"
    result = asyncio.run(
        system._evaluate_candidate_prompt_accuracy_only(
            agent_id=0,
            candidate_prompt="candidate",
            peer_prompts=["p0", "p1", "p2", "p3", "p4"],
            eval_batch=[{"question": f"q{i}", "answer": "A"} for i in range(5)],
        )
    )
    assert result["team_accuracy"] == 0.0
    assert result["target_agent_accuracy"] == 1.0
    assert result["reward"] == 1.0
    assert result["accuracy_only_reward_basis"] == "target_agent_accuracy"


def test_weighted_vote_can_select_valid_independent_minority():
    cfg = Config(reward_mode="accuracy_only", aggregation_mode="weighted_vote", vote_tie_break="first")
    system = _system_without_init(cfg)
    metrics = system.compute_rollout_metrics(
        traces=["trace a", "trace b", "trace c"],
        answers=["A", "A", "B"],
        gold="B",
        prompts=["p0", "p1", "p2"],
        question_hash="q-weighted",
    )
    assert metrics["majority_vote_answer"] == "A"
    assert metrics["weighted_vote_answer"] == "A"

    weighted = system._weighted_vote_with_diagnostics(
        ["A", "A", "B"],
        invalid_flags=[1, 1, 0],
        per_agent_overlap=[0.0, 0.0, 0.0],
        question_hash="q-weighted",
    )
    assert weighted["weighted_vote_answer"] == "B"
    assert weighted["weighted_vote_scores"]["B"] > weighted["weighted_vote_scores"].get("A", 0.0)


def test_dataset_utility_summary_reports_oracle_gap_and_rescue():
    system = _system_without_init(Config())
    rows = [
        {
            "vote_correct": 0,
            "individual_correct": [0, 1, 0],
            "vote_counts": {"A": 2, "B": 1},
            "useful_diversity": 0.2,
            "vote_tie": False,
        },
        {
            "vote_correct": 1,
            "individual_correct": [1, 0, 0],
            "vote_counts": {"C": 3},
            "useful_diversity": 0.0,
            "vote_tie": False,
        },
    ]
    summary = system._summarize_rollout_rows(rows)
    assert summary["vote_acc"] == 0.5
    assert summary["oracle_acc"] == 1.0
    assert summary["aggregation_gap"] == 0.5
    assert summary["rescue_available_rate"] == 0.5
    assert summary["correct_disagreement_rate"] == 0.5
    assert summary["mean_useful_diversity"] == 0.1
