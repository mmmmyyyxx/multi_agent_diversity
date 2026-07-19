from multi_dataset_diverse_rl.behavior_profiles import behavior_distance, build_team_behavior_profiles


def test_no_rescue_support_has_zero_rescue_distance():
    left = {"correctness_vector": [0, 0], "error_vector": [1, 1], "rescue_vector": [0, 0]}
    right = {"correctness_vector": [0, 0], "error_vector": [1, 1], "rescue_vector": [0, 0]}
    result = behavior_distance(left, right)
    assert result["rescue_distance"] == 0.0
    assert result["rescue_reliability"] == 0.0


def test_team_behavior_profiles_capture_rescue_unique_and_shared_error():
    profiles = build_team_behavior_profiles(
        [["A", "X", "X"], ["B", "X", "Y"], ["B", "Z", "Y"]],
        [[1, 0, 0], [0, 0, 1], [0, 0, 1]],
    )
    assert profiles[0]["unique_correct_vector"] == [1, 0, 0]
    assert profiles[0]["rescue_vector"] == [1, 0, 0]
    assert profiles[0]["shared_error_vector"] == [0, 1, 0]
