from typing import Any, Dict, List, Sequence

import numpy as np


def build_team_behavior_profiles(answer_vectors: Sequence[Sequence[str]], correctness_vectors: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    agent_count = len(correctness_vectors)
    size = max((len(values) for values in correctness_vectors), default=0)
    profiles = []
    for agent_id in range(agent_count):
        correctness = [int(value) for value in correctness_vectors[agent_id]]
        error = [1 - value for value in correctness]
        rescue, unique, shared, wrong_clusters = [], [], [], []
        for index in range(size):
            own = correctness[index] if index < len(correctness) else 0
            other_correct = sum(
                int(correctness_vectors[peer][index])
                for peer in range(agent_count) if peer != agent_id and index < len(correctness_vectors[peer])
            )
            other_wrong = max(0, agent_count - 1 - other_correct)
            rescue.append(int(own == 1 and other_correct <= 1))
            unique.append(int(own == 1 and other_correct == 0))
            shared.append(int(own == 0 and other_wrong >= 2))
            answer = str(answer_vectors[agent_id][index]) if agent_id < len(answer_vectors) and index < len(answer_vectors[agent_id]) else ""
            wrong_clusters.append(answer if own == 0 else "")
        profiles.append({
            "answer_vector": list(answer_vectors[agent_id]) if agent_id < len(answer_vectors) else [],
            "correctness_vector": correctness,
            "error_vector": error,
            "rescue_vector": rescue,
            "unique_correct_vector": unique,
            "shared_error_vector": shared,
            "wrong_answer_cluster_vector": wrong_clusters,
            "accuracy": float(np.mean(correctness)) if correctness else 0.0,
        })
    return profiles


def _jaccard_distance(left: Sequence[int], right: Sequence[int], *, empty_distance: float) -> float:
    a = {index for index, value in enumerate(left) if int(value)}
    b = {index for index, value in enumerate(right) if int(value)}
    union = a | b
    if not union:
        return float(empty_distance)
    return 1.0 - len(a & b) / len(union)


def behavior_distance(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    correct_set_weight: float = 0.50,
    rescue_weight: float = 0.35,
    shared_wrong_weight: float = 0.15,
    support_shrinkage: float = 5.0,
) -> Dict[str, float]:
    correct = _jaccard_distance(left.get("correctness_vector", []), right.get("correctness_vector", []), empty_distance=0.0)
    left_rescue, right_rescue = left.get("rescue_vector", []), right.get("rescue_vector", [])
    support = sum(int(value) for value in left_rescue) + sum(int(value) for value in right_rescue)
    reliability = support / (support + max(0.0, float(support_shrinkage))) if support else 0.0
    rescue = reliability * _jaccard_distance(left_rescue, right_rescue, empty_distance=0.0)
    left_error, right_error = left.get("error_vector", []), right.get("error_vector", [])
    size = max(len(left_error), len(right_error), 1)
    overlap = sum(
        int(index < len(left_error) and int(left_error[index]) and index < len(right_error) and int(right_error[index]))
        for index in range(size)
    ) / size
    shared_complementarity = 1.0 - overlap
    distance = (
        float(correct_set_weight) * correct
        + float(rescue_weight) * rescue
        + float(shared_wrong_weight) * shared_complementarity
    )
    return {
        "correct_set_distance": float(correct),
        "rescue_distance": float(rescue),
        "rescue_reliability": float(reliability),
        "shared_wrong_complementarity": float(shared_complementarity),
        "behavior_distance": float(np.clip(distance, 0.0, 1.0)),
    }
