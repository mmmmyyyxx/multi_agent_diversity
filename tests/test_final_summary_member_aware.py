from multi_dataset_diverse_rl.cli import _final_payload
from multi_dataset_diverse_rl.evaluation.validation import DatasetMetrics


def metrics(vote_count, member_counts):
    size = 10
    return DatasetMetrics(
        vote_correct_count=vote_count,
        per_agent_correct_counts=tuple(member_counts),
        plurality_vote_acc=vote_count / size,
        vote_acc=vote_count / size,
        mean_individual_acc=sum(member_counts) / (size * 5),
        min_individual_acc=min(member_counts) / size,
        per_agent_acc=tuple(value / size for value in member_counts),
        mean_soft_vote_utility=0.5,
        c0_count=0,
        mean_invalid_rate=0.0,
        tie_count=0,
        tie_rate=0.0,
        rows=(),
    )


def test_final_payload_keeps_initial_selected_and_member_gain():
    initial = metrics(5, (5, 5, 5, 5, 5))
    selected = metrics(7, (6, 7, 5, 8, 6))
    payload = _final_payload(
        initial, selected, selection_summary={"selected_epoch": 2}
    )
    assert set(payload) == {
        "initial_test", "selected_test", "member_gain", "selection_summary"
    }
    assert payload["member_gain"] == {
        "gain_counts": (1, 2, 0, 3, 1),
        "minimum_gain_count": 0,
        "total_gain_count": 7,
        "mean_gain": 1.4,
        "improved_agent_count": 4,
        "regressed_agent_count": 0,
        "all_members_non_regressed": True,
        "all_members_improved": False,
    }
