from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.v17_formal_support import ARMS, SEEDS, classify_three_seed, formal_target_schedule


def main() -> None:
    protocols = {
        arm: PromptEnsembleOptimizationSystem(Config.from_flat(
            experiment_setting=setting, out_dir="runs/v17_smoke",
        )).protocol
        for arm, setting in ARMS.items()
    }
    assert protocols["S0"].target_branch_count == 0
    assert protocols["S1"].target_selection_policy == "round_robin_dual_formal"
    assert protocols["S1"].generic_revision_enabled
    assert not protocols["S1"].service_routing_enabled
    assert all(len(schedule) == 8 for schedule in map(formal_target_schedule, SEEDS))
    assert all(a != b for seed in SEEDS for a, b in formal_target_schedule(seed))
    assert classify_three_seed((1, 2, 3))["label"] == "CONSISTENT_POSITIVE"
    assert classify_three_seed((1, 2, 0))["label"] == "MAJORITY_POSITIVE"
    assert classify_three_seed((5, -1, -1))["label"] == "POSITIVE_MEAN_HETEROGENEOUS"
    assert classify_three_seed((2, 1, -4))["label"] == "MIXED_NONPOSITIVE"
    assert classify_three_seed((1, -1, 0))["label"] == "NOT_SUPPORTED"
    print(json.dumps({"status": "PASS", "api_calls": 0, "model_calls": 0}))


if __name__ == "__main__":
    main()
