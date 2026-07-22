from multi_dataset_diverse_rl.candidate_selection import stage_a_multichannel_shortlist


def row(name, accuracy=0, vote=0, soft=0.0, coverage=0, residual=0.0):
    return {"prompt_hash": name, "metrics": {
        "candidate_target_correct_count": accuracy, "candidate_invalid_count": 0,
        "net_vote_delta": vote, "vote_loss_count": 0, "soft_vote_utility_delta": soft,
        "coverage_gain_count": coverage, "assigned_residual_utility_delta": residual,
        "unique_correct_loss_count": 0, "pivotal_vote_correct_loss_count": 0,
    }}


def test_three_channels_union_and_deduplicate_with_fixed_budget():
    rows = [
        row("accuracy", accuracy=9), row("vote", vote=3), row("coverage", coverage=4),
        row("shared", accuracy=10, vote=4, coverage=5), row("other", accuracy=1),
    ]
    selected = stage_a_multichannel_shortlist(rows, channel_top_k=2, total_budget=4)
    hashes = [item["prompt_hash"] for item in selected]
    assert len(hashes) == len(set(hashes)) == 4
    assert {"accuracy", "vote", "coverage", "shared"} == set(hashes)


def test_each_channel_can_retain_its_specialist():
    rows = [row("accuracy", accuracy=9), row("vote", vote=2), row("coverage", residual=2.0)]
    assert {item["prompt_hash"] for item in stage_a_multichannel_shortlist(rows, channel_top_k=1)} == {
        "accuracy", "vote", "coverage",
    }
