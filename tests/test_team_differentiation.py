from multi_dataset_diverse_rl.evaluation.fixed_probe import ProbeExample, PromptAnswer
from multi_dataset_diverse_rl.tasks import get_task_spec, normalize_bbh_answer
from multi_dataset_diverse_rl.team_differentiation import (
    team_behavior_metrics,
    vote_transition_decomposition,
)


def _profiles(answers):
    return [
        (PromptAnswer(answer, f"FINAL_ANSWER: {answer}", True),)
        for answer in answers
    ]


def _metrics(answers):
    spec = get_task_spec("bbh")
    return team_behavior_metrics(
        examples=(ProbeExample("q", "q", "A"),),
        profiles=_profiles(answers),
        normalize_answer=normalize_bbh_answer,
        match_answer=spec.match_answer,
        tie_break="abstain",
        seed=42,
    )


def test_same_wrong_and_error_coalition_metrics_distinguish_concentration():
    concentrated = _metrics(["B", "B", "B", "B", "B"])
    dispersed = _metrics(["B", "C", "D", "E", "F"])
    assert concentrated["all_wrong_rate"] == dispersed["all_wrong_rate"] == 1.0
    assert concentrated["mean_H"] == 5
    assert dispersed["mean_H"] == 1
    assert concentrated["mean_off_diagonal_double_fault"] == 1.0
    assert concentrated["same_wrong_unconditional_matrix"][0][1] == 1.0
    assert dispersed["same_wrong_unconditional_matrix"][0][1] == 0.0
    assert concentrated["error_coalition_histogram"]["5"] == 1


def test_vote_transition_decomposition_captures_concentration_driven_flip():
    spec = get_task_spec("bbh")
    result = vote_transition_decomposition(
        examples=(ProbeExample("q", "q", "A"),),
        incumbent_profiles=_profiles(["A", "A", "B", "B", "B"]),
        candidate_profiles=_profiles(["A", "A", "B", "C", "D"]),
        normalize_answer=normalize_bbh_answer,
        match_answer=spec.match_answer,
        tie_break="abstain",
        seed=42,
    )
    assert result["wrong_to_correct_vote_flip_count"] == 1
    assert result["vote_gain_source_counts"] == {"concentration_driven": 1}
    assert result["mean_delta_G"] == 0
    assert result["mean_delta_H"] == -2
    assert result["boundary_vote_gains"] == 1
