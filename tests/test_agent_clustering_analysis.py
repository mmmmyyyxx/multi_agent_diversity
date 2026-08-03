from __future__ import annotations

import json

import numpy as np

from scripts.analyze_final_method_agent_clusters import (
    _bootstrap,
    canonical_labels,
    correctness_distance,
    normalize_partition,
    pairwise_partition_agreement,
    partitions_equal,
)


def test_partition_normalization() -> None:
    assert normalize_partition([1, 1, 0, 1, 0]) == [[0, 1, 3], [2, 4]]
    assert canonical_labels([1, 1, 0, 1, 0]) == [0, 0, 1, 0, 1]


def test_label_invariant_exact_match() -> None:
    assert partitions_equal([0, 0, 1, 0, 1], [1, 1, 0, 1, 0])


def test_pairwise_partition_agreement_uses_all_ten_pairs() -> None:
    left = [0, 0, 0, 1, 1]
    right = [0, 0, 1, 1, 1]
    manual = sum(
        (left[i] == left[j]) == (right[i] == right[j])
        for i in range(5) for j in range(i + 1, 5)
    ) / 10
    assert pairwise_partition_agreement(left, right) == manual


def test_bootstrap_reproducibility() -> None:
    correctness = np.asarray([
        [1, 1, 1, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 1, 0, 1],
    ], dtype=bool)
    _, _, distance, _ = correctness_distance(correctness)
    labels = canonical_labels([0, 0, 1, 1, 0])
    left = _bootstrap(correctness, replicates=50, bootstrap_seed=123, final_labels=labels)
    right = _bootstrap(correctness, replicates=50, bootstrap_seed=123, final_labels=labels)
    assert left == right
    assert distance.shape == (5, 5)


def test_undefined_correlation_switches_entire_split_to_hamming() -> None:
    correctness = [
        [True] * 8,
        [True, False] * 4,
        [False, True] * 4,
        [True, True, False, False] * 2,
        [False, False, True, True] * 2,
    ]
    distance_type, correlations, distance, undefined = correctness_distance(correctness)
    assert undefined is True
    assert distance_type == "correctness_normalized_hamming"
    assert correlations[0][1] is None
    assert distance[0, 1] == 0.5


def test_publication_payload_has_no_raw_leakage() -> None:
    payload = {
        "task": "disambiguation_qa",
        "source_paths_sanitized": ["runs/example/run_meta.json"],
        "source_hashes": {"run_meta.json": "a" * 64},
        "pairwise_correctness_correlation": [[1.0, 0.0], [0.0, 1.0]],
    }
    encoded = json.dumps(payload)
    forbidden = ["question\"", "gold\"", "prompt\"", "raw response", "D:\\\\", "C:\\\\"]
    assert not any(token in encoded for token in forbidden)
