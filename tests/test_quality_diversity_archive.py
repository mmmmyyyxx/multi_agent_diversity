from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.quality_diversity import select_quality_diversity_archive


def candidate(prompt_hash, sequence, accuracy, reward=0.0, rejection=""):
    return {
        "prompt": prompt_hash,
        "prompt_hash": prompt_hash,
        "candidate_id": prompt_hash,
        "metrics": {
            "candidate_target_accuracy": accuracy,
            "penalized_reward": reward,
            "rejection_reason": rejection,
            "mechanism_representation": {
                "normalized_operation_sequence": sequence,
                "mechanism_embedding": [1.0, 0.0] if sequence[0] == "hard_elimination" else [0.0, 1.0],
            },
        },
    }


def test_qd_archive_keeps_quality_elite_per_niche_and_distinct_niches():
    rows = [
        candidate("inc", ["hard_elimination"], 0.8),
        candidate("same-low", ["hard_elimination"], 0.5, reward=10.0),
        candidate("repair", ["weighted_scoring"], 0.75),
        candidate("alternative", ["counterfactual_check"], 0.74),
    ]
    retained, summary = select_quality_diversity_archive(rows, 3, "inc", Config())
    hashes = [row["prompt_hash"] for row in retained]
    assert hashes[0] == "inc"
    assert "same-low" not in hashes
    assert len(set(hashes)) == 3
    assert summary["niche_count"] == 3
    assert summary["mechanism_niche_occupancy"] == 1
    assert all(row["metrics"]["beam_slot"] == row["beam_slot"] for row in retained)
    assert all(row["metrics"]["qd_niche_key"] for row in retained)


def test_qd_archive_rejects_hard_guard_failures_and_works_at_zero_specialization():
    rows = [
        candidate("inc", ["hard_elimination"], 0.8),
        candidate("bad", ["weighted_scoring"], 1.0, rejection="c1_guard"),
        candidate("good", ["counterfactual_check"], 0.79),
    ]
    retained, _ = select_quality_diversity_archive(rows, 3, "inc", Config())
    assert "bad" not in [row["prompt_hash"] for row in retained]
    assert "good" in [row["prompt_hash"] for row in retained]
