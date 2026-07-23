from multi_dataset_diverse_rl.cli import _progress_line


def test_progress_line_is_compact_and_contains_only_requested_metrics():
    line = _progress_line(
        epoch="1/1",
        step="10/75",
        vote_acc=0.456,
        individual_acc=0.4321,
    )

    assert line == "epoch=1/1 step=10/75 vote_acc=0.4560 individual_acc=0.4321"
    assert "{" not in line
    assert "[" not in line
    assert "plurality_vote_acc" not in line
