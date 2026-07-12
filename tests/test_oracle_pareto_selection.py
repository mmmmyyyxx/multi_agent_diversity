import math

from multi_dataset_diverse_rl.cli import is_better_validation_state, oracle_first_validation_key
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import (
    TraceBeamSearchSystem,
    compute_crowding_distances,
    compute_oracle_coverage_transitions,
    non_dominated_sort,
    pareto_dominates,
)


def _candidate(candidate_id, *, gain, loss, acc, baseline_acc=0.7, invalid=0.0, baseline_invalid=0.0, useful=0.0):
    return {
        "candidate_id": candidate_id,
        "prompt": candidate_id,
        "metrics": {
            "coverage_gain_rate": gain,
            "coverage_loss_rate": loss,
            "net_coverage_delta": gain - loss,
            "candidate_target_accuracy": acc,
            "baseline_target_accuracy": baseline_acc,
            "candidate_invalid_rate": invalid,
            "baseline_invalid_rate": baseline_invalid,
            "useful_diversity": useful,
        },
        "reward": 0.0,
    }


def _system():
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(
        candidate_selection_mode="oracle_pareto",
        reward_schedule_mode="static",
        accuracy_guard_epsilon=0.02,
        invalid_guard_epsilon=0.05,
    )
    system.agents = []
    return system


def test_oracle_coverage_transitions_and_delta_identity():
    transitions = compute_oracle_coverage_transitions(
        [[False, False], [True, False], [True, False], [False, False]],
        [[False, True], [False, False], [False, True], [False, False]],
    )

    assert transitions["coverage_gain_count"] == 1
    assert transitions["coverage_loss_count"] == 1
    assert transitions["net_coverage_count"] == 0
    assert transitions["coverage_gain_rate"] == 0.25
    assert transitions["coverage_loss_rate"] == 0.25
    assert transitions["net_coverage_delta"] == 0.0
    assert transitions["net_coverage_delta"] == transitions["candidate_oracle_accuracy"] - transitions["baseline_oracle_accuracy"]


def test_pareto_dominance_and_non_dominance():
    a = _candidate("a", gain=0.3, loss=0.0, acc=0.8)
    b = _candidate("b", gain=0.2, loss=0.1, acc=0.7)
    c = _candidate("c", gain=0.4, loss=0.2, acc=0.75)
    d = _candidate("d", gain=0.2, loss=0.0, acc=0.85)

    assert pareto_dominates(a, b)
    assert not pareto_dominates(b, a)
    assert not pareto_dominates(c, d)
    assert not pareto_dominates(d, c)


def test_non_dominated_sort_and_crowding_are_deterministic():
    candidates = [
        _candidate("d", gain=0.2, loss=0.0, acc=0.85),
        _candidate("b", gain=0.2, loss=0.1, acc=0.7),
        _candidate("a", gain=0.3, loss=0.0, acc=0.8),
        _candidate("c", gain=0.4, loss=0.2, acc=0.75),
    ]
    fronts = non_dominated_sort(candidates)
    assert [candidates[index]["candidate_id"] for index in fronts[0]] == ["a", "c", "d"]
    assert [candidates[index]["candidate_id"] for index in fronts[1]] == ["b"]
    distances = compute_crowding_distances(candidates, fronts[0])
    assert any(math.isinf(value) for value in distances.values())

    reordered = list(reversed(candidates))
    reordered_fronts = non_dominated_sort(reordered)
    assert {reordered[index]["candidate_id"] for index in reordered_fronts[0]} == {"a", "c", "d"}


def test_oracle_pareto_guards_exclude_infeasible_max_gain_candidate():
    system = _system()
    current = _candidate("current", gain=0.0, loss=0.0, acc=0.7)
    high_gain_but_bad_acc = _candidate("bad_acc", gain=0.9, loss=0.0, acc=0.6)
    high_gain_but_invalid = _candidate("bad_invalid", gain=0.8, loss=0.0, acc=0.8, invalid=0.2)
    good = _candidate("good", gain=0.2, loss=0.0, acc=0.8, useful=0.3)

    selected, summary = system._select_oracle_pareto_beam(
        [high_gain_but_bad_acc, current, high_gain_but_invalid, good], beam_size=2, current_prompt="current"
    )

    assert [item["candidate_id"] for item in selected] == ["good", "current"]
    assert high_gain_but_bad_acc["pareto_feasible"] is False
    assert high_gain_but_invalid["pareto_feasible"] is False
    assert high_gain_but_bad_acc["pareto_selected"] is False
    assert summary["num_pareto_feasible"] == 2


def test_oracle_pareto_beam_keeps_tradeoffs_and_active_prompt_is_first():
    system = _system()
    current = _candidate("current", gain=0.0, loss=0.0, acc=0.7)
    gain = _candidate("gain", gain=0.4, loss=0.15, acc=0.72)
    safe = _candidate("safe", gain=0.1, loss=0.0, acc=0.86)

    selected, _ = system._select_oracle_pareto_beam([safe, current, gain], beam_size=3, current_prompt="current")

    assert {item["candidate_id"] for item in selected} == {"current", "gain", "safe"}
    assert selected[0]["candidate_id"] == "gain"
    assert selected[0]["pareto_selected"] is True

    reverse_selected, _ = system._select_oracle_pareto_beam([gain, current, safe], beam_size=2, current_prompt="current")
    assert [item["candidate_id"] for item in reverse_selected] == ["gain", "safe"]


def test_oracle_first_validation_ignores_vote_accuracy_and_uses_specified_ties():
    epoch_a = {"epoch": 1, "val": {"oracle_acc": 0.90, "vote_acc": 0.70, "mean_individual_acc": 0.7, "mean_invalid_rate": 0.1, "mean_useful_diversity": 0.2}}
    epoch_b = {"epoch": 2, "val": {"oracle_acc": 0.88, "vote_acc": 0.90, "mean_individual_acc": 0.9, "mean_invalid_rate": 0.0, "mean_useful_diversity": 0.9}}
    assert is_better_validation_state(epoch_a, epoch_b, 0.0, "coverage_useful_diversity", "oracle_first")

    higher_mean = {"epoch": 2, "val": {"oracle_acc": 0.90, "mean_individual_acc": 0.8, "mean_invalid_rate": 0.2, "mean_useful_diversity": 0.1}}
    assert is_better_validation_state(higher_mean, epoch_a, 0.0, "coverage_useful_diversity", "oracle_first")

    lower_invalid = {"epoch": 3, "val": {"oracle_acc": 0.90, "mean_individual_acc": 0.8, "mean_invalid_rate": 0.05, "mean_useful_diversity": 0.0}}
    assert is_better_validation_state(lower_invalid, higher_mean, 0.0, "coverage_useful_diversity", "oracle_first")

    earlier = {"epoch": 1, "val": {"oracle_acc": 0.90, "mean_individual_acc": 0.8, "mean_invalid_rate": 0.05, "mean_useful_diversity": 0.0}}
    assert oracle_first_validation_key(earlier) < oracle_first_validation_key(lower_invalid)
