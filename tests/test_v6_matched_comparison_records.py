from scripts.build_v6_matched_comparison_records import (
    acceptance_summary,
    build_pair_manifest,
    longest_rejection_streak,
)


def decision(update, target_gain=None, vote_before=None, vote_after=None):
    if target_gain is None:
        return {"update_index": update, "accepted_prompt_hash": "", "candidates": []}
    return {
        "update_index": update,
        "accepted_prompt_hash": f"candidate-{update}",
        "candidates": [{
            "prompt_hash": f"candidate-{update}",
            "constraint": {
                "target_gain": target_gain,
                "vote_correct_incumbent": vote_before,
                "vote_correct_candidate": vote_after,
            },
        }],
    }


def test_acceptance_transition_summary_separates_v6_progress_types():
    rows = [
        decision(0, 2, 10, 10), decision(1), decision(2),
        decision(3, 0, 10, 11), decision(4, 1, 11, 12),
    ]
    summary = acceptance_summary(rows)
    assert summary["update_indexes"] == {
        "target_only": [0], "vote_only": [3], "target_and_vote": [4], "other": [],
    }
    assert longest_rejection_streak(rows) == 2


def test_pair_manifest_never_calls_different_initial_states_matched():
    identity = {
        "git_commit": "same", "git_dirty": False, "config_fingerprint": "fingerprint",
        "manifest_sha256": "manifest", "train_file_sha256": "train",
        "val_file_sha256": "val", "test_file_sha256": "test",
    }
    base_meta = {
        "run_identity": identity,
        "proposal_memory_mode": "off",
        "initial_prompt_hashes": ["prompt"] * 5,
        "prompt_question_evaluator_identity": ["solver", "contract", "parser", 0.0, 44],
        "probe_hash": "probe",
        "planned_update_count": 32,
        "config": {},
    }
    memory_meta = {**base_meta, "proposal_memory_mode": "state_local_v1"}
    manifest = build_pair_manifest(44, base_meta, memory_meta, {"vote": 29}, {"vote": 27})
    assert manifest["runtime_config_match"] is True
    assert manifest["initial_state_match"] is False
    assert manifest["matched_status"] == "unmatched"
