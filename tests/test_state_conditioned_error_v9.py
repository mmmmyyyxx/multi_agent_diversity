from types import SimpleNamespace

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.cli import select_state_conditioned_candidate_eval_batch
from multi_dataset_diverse_rl.persistence.checkpoint import (
    build_training_checkpoint,
    checkpoint_behavior_config_fingerprint,
    checkpoint_incompatibility_reasons,
)
from multi_dataset_diverse_rl.state_conditioned import (
    STATE_CONDITIONED_CHECKPOINT_VERSION,
    c2_dispersion_rescuability,
    compute_c2_wrong_split_metrics,
    coverage_case_assignees,
    paired_c0_metrics,
    select_state_conditioned_archive,
    select_state_conditioned_team,
    state_conditioned_transition_metrics,
    state_dataset_metrics,
)


def _transition_row(
    baseline_answers,
    candidate_answers,
    *,
    gold="A",
    baseline_vote=False,
    candidate_vote=False,
    option_count=4,
    target=4,
):
    baseline_correct = [answer == gold for answer in baseline_answers]
    candidate_correct = [answer == gold for answer in candidate_answers]

    def largest_wrong(answers):
        counts = {}
        for answer in answers:
            if answer != gold:
                counts[answer] = counts.get(answer, 0) + 1
        return max(counts.values(), default=0)

    return {
        "baseline_answers": baseline_answers,
        "candidate_answers": candidate_answers,
        "baseline_individual_correct": baseline_correct,
        "candidate_individual_correct": candidate_correct,
        "baseline_gold_vote_count": sum(baseline_correct),
        "candidate_gold_vote_count": sum(candidate_correct),
        "baseline_largest_wrong_vote_count": largest_wrong(baseline_answers),
        "candidate_largest_wrong_vote_count": largest_wrong(candidate_answers),
        "baseline_target_correct": baseline_correct[target],
        "target_agent_correct": candidate_correct[target],
        "baseline_vote_correct": baseline_vote,
        "candidate_vote_correct": candidate_vote,
        "option_count": option_count,
        "question_hash": "q",
    }


def _candidate(prompt_hash, accuracy, **metrics):
    return {
        "prompt_hash": prompt_hash,
        "prompt": prompt_hash,
        "generation": 1,
        "metrics": {
            "state_quality_guard_passed": True,
            "candidate_target_accuracy": accuracy,
            "candidate_invalid_rate": 0.0,
            **metrics,
        },
    }


def test_c2_strict_tie_and_failure_structures():
    strict = state_dataset_metrics([{
        "question_hash": "strict", "gold_vote_count": 2,
        "largest_wrong_vote_count": 1, "vote_correct": True,
        "vote_counts": {"A": 2, "B": 1, "C": 1, "D": 1}, "option_count": 4,
    }])
    tie = state_dataset_metrics([{
        "question_hash": "tie", "gold_vote_count": 2,
        "largest_wrong_vote_count": 2, "vote_correct": False,
        "vote_counts": {"A": 2, "B": 2, "C": 1}, "option_count": 3,
    }])
    failure = state_dataset_metrics([{
        "question_hash": "failure", "gold_vote_count": 2,
        "largest_wrong_vote_count": 3, "vote_correct": False,
        "vote_counts": {"A": 2, "B": 3}, "option_count": 4,
    }])
    assert strict["c2_strict_win_count"] == 1
    assert strict["c2_vote_correct_count"] == 1
    assert tie["c2_tie_count"] == 1
    assert tie["c2_strict_win_count"] == 0
    assert failure["c2_vote_fail_count"] == 1
    assert failure["c2_mean_largest_wrong_vote"] == 3


def test_c2_wrong_split_only_applies_when_target_remains_wrong_at_c2():
    split = _transition_row(
        ["A", "A", "B", "B", "B"],
        ["A", "A", "B", "B", "C"],
        candidate_vote=False,
    )
    metrics = state_conditioned_transition_metrics([split])
    assert metrics["c2_wrong_cluster_reduction"] == 1
    assert metrics["c2_wrong_split_tie_gain_count"] == 1
    assert metrics["c2_wrong_split_strict_gain_count"] == 0

    corrected = _transition_row(
        ["A", "A", "B", "B", "B"],
        ["A", "A", "B", "B", "A"],
        candidate_vote=True,
    )
    corrected_metrics = state_conditioned_transition_metrics([corrected])
    assert corrected_metrics["c2_to_c3_count"] == 1
    assert corrected_metrics["c2_wrong_cluster_reduction"] == 0


def test_c0_and_c3plus_wrong_dispersion_have_zero_task_gain():
    c0 = _transition_row(
        ["B", "B", "B", "C", "D"],
        ["B", "C", "D", "B", "C"],
        target=0,
    )
    c3 = _transition_row(
        ["A", "A", "A", "B", "B"],
        ["A", "A", "A", "B", "C"],
        target=4,
    )
    assert compute_c2_wrong_split_metrics(c0)["wrong_answer_diversity_task_gain"] == 0.0
    assert compute_c2_wrong_split_metrics(c3)["wrong_answer_diversity_task_gain"] == 0.0


def test_correct_coverage_is_asymmetric():
    gain = state_conditioned_transition_metrics([_transition_row(
        ["B", "B", "B", "B", "B"],
        ["B", "B", "B", "B", "A"],
        target=4,
    )])
    loss = state_conditioned_transition_metrics([_transition_row(
        ["B", "B", "B", "B", "A"],
        ["B", "B", "B", "B", "B"],
        target=4,
    )])
    assert gain["c0_to_c1_count"] == 1
    assert gain["target_wrong_to_correct_count"] == 1
    assert loss["c1_to_c0_count"] == 1
    assert loss["target_correct_to_wrong_count"] == 1


def test_c2_rescuability_depends_on_option_count():
    assert c2_dispersion_rescuability(4)["c2_strictly_rescuable_by_dispersion"] is True
    assert c2_dispersion_rescuability(3)["c2_tie_only_rescuable_by_dispersion"] is True
    assert c2_dispersion_rescuability(2)["c2_unrescuable_by_dispersion"] is True


def test_accuracy_band_blocks_low_accuracy_utility_candidate():
    cfg = Config(
        method_version="v9_state_conditioned_error",
        state_accuracy_tie_epsilon=0.02,
        state_representative_capacity=4,
    )
    incumbent = _candidate("incumbent", 0.86)
    best = _candidate("best", 0.90)
    coverage = _candidate("coverage", 0.89, c0_to_c1_count=2)
    conversion = _candidate("conversion", 0.89, c2_to_c3_count=2)
    tempting_but_weak = _candidate("weak", 0.70, c0_to_c1_count=99, c2_to_c3_count=99)
    archive = select_state_conditioned_archive(
        [incumbent, best, coverage, conversion, tempting_but_weak],
        "incumbent",
        4,
        cfg,
    )
    hashes = {item["prompt_hash"] for item in archive}
    assert hashes == {"incumbent", "best", "coverage", "conversion"}
    assert "weak" not in hashes


def test_archive_keeps_coverage_and_conversion_as_separate_slots():
    cfg = Config(method_version="v9_state_conditioned_error")
    archive = select_state_conditioned_archive(
        [
            _candidate("incumbent", 0.90),
            _candidate("accuracy", 0.92),
            _candidate("coverage", 0.91, c0_to_c1_count=3),
            _candidate("conversion", 0.91, c2_to_c3_count=3),
        ],
        "incumbent",
        4,
        cfg,
    )
    slots = {item["prompt_hash"]: item["state_archive_slot"] for item in archive}
    assert slots["coverage"] == "coverage_repair"
    assert slots["conversion"] == "c2_correct"


def test_higher_diversity_cannot_replace_higher_quality_team():
    cfg = Config(
        method_version="v9_state_conditioned_error",
        state_joint_total_correct_slack_rate=0.0,
    )
    selected = select_state_conditioned_team(
        [
            {"prompt_hashes": ["quality"], "total_agent_correct_count": 101, "trace_diversity_tiebreak": 0.0},
            {"prompt_hashes": ["diverse"], "total_agent_correct_count": 100, "trace_diversity_tiebreak": 1.0},
        ],
        cfg,
        probe_size=20,
        num_agents=5,
    )["selected"]
    assert selected["prompt_hashes"] == ["quality"]


def test_accuracy_only_team_ablation_ignores_state_and_trace_keys():
    cfg = Config(
        method_version="v9_state_conditioned_error",
        state_vote_objective_enabled=False,
        state_coverage_enabled=False,
        state_c2_correct_conversion_enabled=False,
        state_c2_wrong_split_enabled=False,
        state_trace_tiebreak_enabled=False,
        state_joint_total_correct_slack_rate=0.0,
    )
    common = {
        "total_agent_correct_count": 100,
        "bottom2_correct_count": 30,
        "mean_gold_plurality_margin": 0.1,
        "invalid_count": 0,
    }
    selected = select_state_conditioned_team(
        [
            {**common, "prompt_hashes": ["a"], "c0_count": 0, "c2_vote_correct_count": 10, "trace_diversity_tiebreak": 1.0},
            {**common, "prompt_hashes": ["z"], "c0_count": 10, "c2_vote_correct_count": 0, "trace_diversity_tiebreak": 0.0},
        ],
        cfg,
        probe_size=20,
        num_agents=5,
    )["selected"]
    assert selected["prompt_hashes"] == ["z"]


def test_residual_assignment_and_paired_c0_metrics_are_deterministic():
    first = coverage_case_assignees("question", 5, seed=42)
    assert first == coverage_case_assignees("question", 5, seed=42)
    assert len(first) == 2
    assert len(set(first)) == 2
    assert paired_c0_metrics(
        {"a": "C0", "b": "C0", "c": "C1"},
        {"a": "C0", "b": "C1", "c": "C0"},
    ) == {"persistent_c0_count": 1, "new_c0_count": 1, "resolved_c0_count": 1}


def test_candidate_batch_preserves_three_disjoint_pool_budgets():
    import hashlib

    source = [{"question": f"question {index}", "answer": "A"} for index in range(9)]
    states = [0, 1, 2, 0, 1, 2, 3, 3, 3]
    records = [
        {
            "question_hash": hashlib.sha1(row["question"].encode("utf-8")).hexdigest()[:12],
            "metrics": {"gold_vote_count": state},
        }
        for row, state in zip(source, states)
    ]
    cfg = Config(
        candidate_eval_batch_size=6,
        candidate_batch_representative_size=2,
        candidate_batch_coverage_size=2,
        candidate_batch_conversion_size=2,
        seed=42,
    )
    batch = select_state_conditioned_candidate_eval_batch(
        source, source, cfg, epoch=1, step=1, state_records=records
    )
    pools = [row["_candidate_pool"] for row in batch]
    assert len(batch) == 6
    assert len({row["question"] for row in batch}) == 6
    assert sum(bool(row["_candidate_pool_primary"]) for row in batch) == 2
    assert pools.count("coverage") == 2
    assert pools.count("conversion") == 1
    assert sum(row["_candidate_pool_fallback_for"] == "conversion" for row in batch) == 1


def test_v9_checkpoint_has_version_and_rejects_missing_v9_marker():
    cfg = Config(
        method_version="v9_state_conditioned_error",
        reward_mode="rollout_state_conditioned",
        candidate_selection_mode="state_conditioned_accuracy_first",
        best_state_selection_mode="state_conditioned_vote_first",
        epochs=1,
        train_size=1,
    )
    agent = SimpleNamespace(
        initial_prompt="p", current_prompt="p", prompt_beam=[], history=["p"],
        accept_count=0, reject_count=0, trajectory_state_dict=lambda: {},
    )
    system = SimpleNamespace(
        agents=[agent], recent_window_records=[], specialization_strength=0.0,
    )
    payload = build_training_checkpoint(
        cfg, system, epoch_index=0, cursor=0, order=[0], train_accumulators={},
        best_score=0.0, best_epoch=0, epochs_without_improvement=0, stopped_early=False,
        no_effective_evolution_counter=0, no_effective_evolution_stopped=False,
        no_effective_evolution_reason="",
    )
    assert payload["state_conditioned_checkpoint_version"] == STATE_CONDITIONED_CHECKPOINT_VERSION
    assert not checkpoint_incompatibility_reasons(payload, cfg, [{}])
    payload.pop("state_conditioned_checkpoint_version")
    assert any("state-conditioned checkpoint fingerprint" in reason for reason in checkpoint_incompatibility_reasons(payload, cfg, [{}]))


def test_v8_default_fingerprint_is_unchanged():
    assert checkpoint_behavior_config_fingerprint(Config()) == (
        "48c2f27cdcda64d2f7b32d008957b4903c683f49012988c4e5cab301ed29d5fa"
    )
