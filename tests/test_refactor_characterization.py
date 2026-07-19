import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.behavior_profiles import behavior_distance
from multi_dataset_diverse_rl.cli import checkpoint_behavior_config_fingerprint
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.lineage import empty_lineage_state, update_lineage_state
from multi_dataset_diverse_rl.mechanisms import mechanism_distance, normalize_mechanism_representation
from multi_dataset_diverse_rl.search_archive import (
    candidate_quality_bucket,
    select_joint_representatives,
    select_safe_archive,
)
from scripts.experiment_config import select_settings


SNAPSHOT = json.loads(
    (Path(__file__).parent / "fixtures" / "refactor_characterization_v8.json").read_text(encoding="utf-8")
)


def record(name, *, candidate_type="task_specific_repair", accuracy_delta=0.0, c1_loss=0, sequence=("hard_elimination",)):
    return {
        "id": name, "candidate_id": name, "prompt": f"{name}.", "prompt_hash": name,
        "archive_bucket": "safe",
        "metrics": {
            "candidate_type": candidate_type,
            "accuracy_delta": accuracy_delta,
            "depth1_loss_count": c1_loss,
            "depth2_loss_count": 0,
            "candidate_target_accuracy": 0.7,
            "penalized_reward": 1.0,
            "mechanism_novel": candidate_type == "task_specific_repair" or name == "alternative",
            "mechanism_representation": {
                "normalized_operation_sequence": list(sequence),
                "mechanism_embedding": [1.0, 0.0] if sequence == ("hard_elimination",) else [0.0, 1.0],
            },
        },
    }


def test_default_fingerprint_and_v8_preset_are_characterized():
    assert checkpoint_behavior_config_fingerprint(Config()) == SNAPSHOT["default_behavior_fingerprint"]
    setting = select_settings(SNAPSHOT["v8_setting"]["name"])[0]
    actual = {key: getattr(setting, key) for key in SNAPSHOT["v8_setting"]}
    assert actual == SNAPSHOT["v8_setting"]


def test_distance_and_archive_behavior_are_characterized():
    left = normalize_mechanism_representation("", ["Eliminate impossible candidates"])
    right = normalize_mechanism_representation("", ["Use weighted scoring for each candidate"])
    left["mechanism_embedding"], right["mechanism_embedding"] = [1.0, 0.0], [0.0, 1.0]
    assert mechanism_distance(left, right) == SNAPSHOT["mechanism_distance"]

    left_behavior = {
        "correctness_vector": [1, 0, 1, 0], "error_vector": [0, 1, 0, 1],
        "rescue_vector": [0, 1, 0, 0], "wrong_answer_vector": ["", "B", "", "C"],
        "same_wrong_vector": [0, 1, 0, 0],
    }
    right_behavior = {
        "correctness_vector": [1, 1, 0, 0], "error_vector": [0, 0, 1, 1],
        "rescue_vector": [0, 0, 1, 0], "wrong_answer_vector": ["", "", "D", "C"],
        "same_wrong_vector": [0, 0, 0, 1],
    }
    distance = behavior_distance(
        left_behavior, right_behavior, correct_set_weight=0.4, rescue_weight=0.3,
        shared_wrong_weight=0.15, wrong_answer_dispersion_weight=0.15,
        support_shrinkage=5.0, wrong_support_shrinkage=5.0,
    )["behavior_distance"]
    assert distance == pytest.approx(SNAPSHOT["behavior_distance"])

    cfg = Config()
    safe = record("safe")
    probation = record("probation", accuracy_delta=-0.02, c1_loss=1, sequence=("weighted_scoring",))
    catastrophic = record("catastrophic", accuracy_delta=-0.06)
    assert [candidate_quality_bucket(item, cfg) for item in (safe, probation, catastrophic)] == SNAPSHOT["candidate_buckets"]
    rows = [record("incumbent"), record("repair", sequence=("binding_resolution",)), record("alternative", candidate_type="mechanism_alternative", sequence=("weighted_scoring",))]
    archive = select_safe_archive(rows, "incumbent", 6)
    representatives = select_joint_representatives(archive, "incumbent", 3)
    assert [item["prompt_hash"] for item in archive] == SNAPSHOT["archive_prompt_hashes"]
    assert [item["prompt_hash"] for item in representatives] == SNAPSHOT["representative_prompt_hashes"]


def test_lineage_transition_is_characterized():
    cfg = Config(lineage_commit_required_snapshots=2)
    selected = {
        "prompt_hash": "p", "prompt": "prompt", "fold_behavior_stable": True,
        "mechanism_representation": {"normalized_operation_sequence": ["hard_elimination"], "mechanism_embedding": [1.0, 0.0]},
        "behavior_profile": {"correctness_vector": [1, 0], "rescue_vector": [0, 1], "accuracy": 0.5},
    }
    first = update_lineage_state(empty_lineage_state(), selected, epoch=1, quality_gate_passed=True, config=cfg)
    state = {key: value for key, value in first.items() if key not in {"old_status", "new_status", "reason"}}
    second = update_lineage_state(state, selected, epoch=2, quality_gate_passed=True, config=cfg)
    assert [first["new_status"], second["new_status"]] == SNAPSHOT["lineage_statuses"]
