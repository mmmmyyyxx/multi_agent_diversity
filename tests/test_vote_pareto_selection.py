import math

from multi_dataset_diverse_rl.cli import is_better_validation_state, vote_first_validation_key
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import (
    TraceBeamSearchSystem,
    compute_crowding_distances,
    compute_vote_transitions,
    non_dominated_sort,
    pareto_dominates,
)


def _candidate(candidate_id, *, gain, loss, acc, vote_delta=None, margin=0.0, boundary=0.0, baseline_acc=0.7, invalid=0.0, baseline_invalid=0.0):
    return {
        "candidate_id": candidate_id,
        "prompt": candidate_id,
        "metrics": {
            "vote_gain_rate": gain,
            "vote_loss_rate": loss,
            "vote_delta": gain - loss if vote_delta is None else vote_delta,
            "candidate_target_accuracy": acc,
            "baseline_target_accuracy": baseline_acc,
            "candidate_invalid_rate": invalid,
            "baseline_invalid_rate": baseline_invalid,
            "vote_margin_delta": margin,
            "boundary_useful_diversity_delta": boundary,
        },
        "reward": 0.0,
    }


def _system():
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(candidate_selection_mode="vote_pareto", reward_schedule_mode="static", accuracy_guard_epsilon=0.02, invalid_guard_epsilon=0.05)
    system.agents = []
    return system


def test_vote_transitions_and_delta_identity():
    transitions = compute_vote_transitions([False, True, True, False], [True, False, True, False])
    assert transitions["vote_gain_count"] == 1
    assert transitions["vote_loss_count"] == 1
    assert transitions["net_vote_count"] == 0
    assert transitions["net_vote_delta"] == transitions["vote_gain_rate"] - transitions["vote_loss_rate"]


def test_pareto_dominance_and_non_dominance():
    a = _candidate("a", gain=0.3, loss=0.0, acc=0.8)
    b = _candidate("b", gain=0.2, loss=0.1, acc=0.7)
    c = _candidate("c", gain=0.4, loss=0.2, acc=0.75)
    d = _candidate("d", gain=0.2, loss=0.0, acc=0.85)
    assert pareto_dominates(a, b)
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
    assert any(math.isinf(value) for value in compute_crowding_distances(candidates, fronts[0]).values())


def test_vote_pareto_guards_and_deterministic_tie_break():
    system = _system()
    current = _candidate("current", gain=0.0, loss=0.0, acc=0.7)
    bad_acc = _candidate("bad_acc", gain=0.9, loss=0.0, acc=0.6)
    bad_invalid = _candidate("bad_invalid", gain=0.8, loss=0.0, acc=0.8, invalid=0.2)
    gain = _candidate("gain", gain=0.4, loss=0.1, acc=0.8, margin=0.1)
    selected, summary = system._select_vote_pareto_beam([bad_acc, current, bad_invalid, gain], beam_size=2, current_prompt="current")
    assert [item["candidate_id"] for item in selected] == ["gain", "current"]
    assert bad_acc["pareto_feasible"] is False
    assert bad_invalid["pareto_feasible"] is False
    assert summary["num_pareto_feasible"] == 2


def test_vote_first_validation_key_prioritizes_vote_metrics():
    lower_vote = {"epoch": 1, "val": {"vote_acc": 0.70, "mean_individual_acc": 0.9, "mean_vote_margin": 0.8, "mean_boundary_useful_diversity": 0.9, "mean_invalid_rate": 0.0, "oracle_acc": 1.0}}
    higher_vote = {"epoch": 2, "val": {"vote_acc": 0.80, "mean_individual_acc": 0.1, "mean_vote_margin": -0.5, "mean_boundary_useful_diversity": 0.0, "mean_invalid_rate": 1.0, "oracle_acc": 0.0}}
    assert is_better_validation_state(higher_vote, lower_vote, 0.0, "vote_useful_diversity", "vote_first")
    equal_later = {"epoch": 2, "val": {"vote_acc": 0.8, "mean_individual_acc": 0.5, "mean_vote_margin": 0.1, "mean_boundary_useful_diversity": 0.2, "mean_invalid_rate": 0.1}}
    equal_earlier = {"epoch": 1, "val": dict(equal_later["val"])}
    assert vote_first_validation_key(equal_earlier) < vote_first_validation_key(equal_later)
    higher_boundary_later = {"epoch": 2, "val": {**equal_earlier["val"], "mean_boundary_useful_diversity": 1.0}}
    assert vote_first_validation_key(equal_earlier) < vote_first_validation_key(higher_boundary_later)


def test_vote_first_applies_min_delta_to_vote_before_tiebreaks():
    best = {"epoch": 1, "val": {"vote_acc": 0.70, "mean_individual_acc": 0.1, "mean_vote_margin": 0.0, "mean_invalid_rate": 0.1}}
    tiny_vote_gain = {"epoch": 2, "val": {"vote_acc": 0.705, "mean_individual_acc": 0.2, "mean_vote_margin": 0.1, "mean_invalid_rate": 0.0}}

    assert is_better_validation_state(tiny_vote_gain, best, 0.0, "vote_useful_diversity", "vote_first", min_delta=0.01)
