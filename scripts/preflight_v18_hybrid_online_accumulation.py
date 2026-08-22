from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_v18_hybrid_online_accumulation_registry import build_registry
from scripts.v18_hybrid_online_accumulation_support import ARMS, HYBRID, SEEDS, UPDATES, W1


def preflight(registry: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "INITIAL_STATE_GATE": (
            registry["protocol"]["initialization_mode"] == "shared_identical"
            and len(registry["protocol"]["shared_prompt_hashes"]) == 5
            and len(set(registry["protocol"]["shared_prompt_hashes"])) == 1
        ),
        "ONLINE_HORIZON_GATE": (
            registry["update_opportunities_per_trajectory"] == UPDATES
            and registry["epochs"] == 1
            and registry["update_every"] == 10
            and registry["train_size"] == 75
        ),
        "SEED_FREEZE_GATE": (
            tuple(registry["seeds"]) == SEEDS
            and registry["seed_freeze"]["gate"] == "PASS"
        ),
        "COMPUTE_MATCH_GATE": (
            registry["source_candidates_per_target"] == 2
            and registry["loss_blind_revision_per_valid_source"] == 1
            and registry["protocol"]["target_branch_count"] == 2
            and registry["protocol"]["generic_revision_enabled"] is True
            and registry["protocol"]["compatibility_repair_enabled"] is False
        ),
        "SELECTOR_FREEZE_GATE": (
            tuple(registry["arms"]) == ARMS
            and registry["arm_contract"][W1] == "W1 rank1 plus W1 rank2"
            and "responsibility-constrained RR" in registry["arm_contract"][HYBRID]
            and registry["protocol"]["target_selection_policy"]
            == "repairability_adjusted_responsibility"
        ),
        "LONGITUDINAL_LOGGING_GATE": (
            registry["validation_schedule"] == "initial_plus_changed_state_only"
            and registry["validation_role"] == "analysis_only_no_selection"
            and registry["new_test_calls"] == 0
        ),
        "CLASSIFIER_FREEZE_GATE": (
            registry["classifier"]["classifier_version"]
            == "v18_hybrid_online_accumulation_classifier_v1"
            and len(registry["classifier"]["allowed_final_diagnoses"]) == 5
        ),
        "SANITIZATION_GATE": (
            registry["split_freeze"]["gate"] == "PASS"
            and registry["phase_a_zero_api"] is True
        ),
    }
    return {
        "gate": "PASS" if all(gates.values()) else "FAIL",
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
        "errors": [key for key, value in gates.items() if not value],
        "api_calls": 0,
        "model_calls": 0,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "trajectory_count": 6,
        "online_horizon": UPDATES,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    registry = (
        json.loads(args.registry.read_text(encoding="utf-8"))
        if args.registry else build_registry()
    )
    result = preflight(registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
