from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.compatibility_repair import (
    LOSS_BLIND_GENERIC_REVISION_VERSION,
    ONLINE_COMPATIBILITY_REPAIR_VERSION,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.versions import (
    CHECKPOINT_VERSION,
    METHOD_VERSION,
)
from scripts.v17_formal_support import (
    ARMS,
    CALL_CEILING,
    EXECUTION_ORDER,
    EXPERIMENT_ROOT,
    SEEDS,
    TOKEN_CEILING,
    UPDATES,
    formal_target_schedule,
    formal_seed_prior_use,
    sha256_json,
    source_semantics_diff,
    split_freeze,
    write_json,
)


def protocol_payload(setting: str) -> dict:
    protocol = PromptEnsembleOptimizationSystem(Config.from_flat(
        experiment_setting=setting, out_dir="runs/v17_preregistration",
    )).protocol
    return {
        "setting": setting,
        "optimization_enabled": protocol.optimization_enabled,
        "module_vector": protocol.module_vector,
        "target_selection_policy": protocol.target_selection_policy,
        "sample_pool_policy": protocol.sample_pool_policy,
        "tcs_context_policy": protocol.tcs_context_policy,
        "responsibility_refresh_policy": protocol.responsibility_refresh_policy,
        "service_routing_enabled": protocol.service_routing_enabled,
        "target_branch_count": protocol.target_branch_count,
        "candidates_per_target_branch": protocol.candidates_per_target_branch,
        "total_generated_candidates_per_update": (
            protocol.candidate_budget_contract.total_generated_candidates_per_update
        ),
        "generic_revision_enabled": protocol.generic_revision_enabled,
        "compatibility_repair_enabled": protocol.compatibility_repair_enabled,
        "candidate_acceptance_policy": protocol.candidate_acceptance_policy,
        "candidate_ranking_policy": protocol.candidate_ranking_policy,
        "module2_context_variant": protocol.module2_context_variant,
        "module2_evolution_variant": protocol.module2_evolution_variant,
    }


def main() -> None:
    if EXPERIMENT_ROOT.exists():
        raise FileExistsError("fresh V17 preregistration directory required")
    data = split_freeze()
    seed_audit = formal_seed_prior_use()
    semantics = source_semantics_diff()
    protocols = {arm: protocol_payload(setting) for arm, setting in ARMS.items()}
    classifier = {
        "version": "v17_three_seed_contrast_classifier_v1",
        "seed_count": 3,
        "definitions": {
            "CONSISTENT_POSITIVE": "mean_delta>0 and wins==3",
            "MAJORITY_POSITIVE": "mean_delta>0 and wins>losses and wins<3",
            "POSITIVE_MEAN_HETEROGENEOUS": "mean_delta>0 and wins<=losses",
            "MIXED_NONPOSITIVE": "mean_delta<=0 and wins>losses",
            "NOT_SUPPORTED": "mean_delta<=0 and wins<=losses",
        },
        "primary_contrasts": {
            "C01": "S1-S0", "C12": "S2-S1", "C23": "S3-S2",
            "C34": "S4-S3", "C14": "S4-S1",
        },
        "headline_total_contrast": {"C04": "S4-S0"},
        "primary_metric": "final_test_plurality_team_vote_accuracy",
        "support_labels": ["CONSISTENT_POSITIVE", "MAJORITY_POSITIVE"],
    }
    classifier["classifier_hash"] = sha256_json(classifier)
    matrix = {
        "version": "v17_formal_five_arm_matrix_v1",
        "arms": protocols,
        "isolation_contract": {
            "S1_S2": "complete Module1 mechanism comparison under common generic revision opportunity",
            "S2_S3": "compute-controlled generic mechanism package versus M20 package",
            "S3_S4": "frozen conditional M2F path only",
        },
        "loss_blind_generic_revision_version": LOSS_BLIND_GENERIC_REVISION_VERSION,
        "compatibility_repair_version": ONLINE_COMPATIBILITY_REPAIR_VERSION,
    }
    order = {
        "version": "v17_formal_execution_order_v1",
        "training": {str(seed): list(EXECUTION_ORDER[seed]) for seed in SEEDS},
        "validation": {str(seed): list(EXECUTION_ORDER[seed]) for seed in SEEDS},
        "test": {str(seed): list(EXECUTION_ORDER[seed]) for seed in SEEDS},
        "S1_target_schedule": {
            str(seed): formal_target_schedule(seed) for seed in SEEDS
        },
    }
    prereg = {
        "experiment_version": "v17_formal_5arm_3seed_v1",
        "starting_head": "f4be41c960aa9f052ac7d1de2a9cf23bde4fd95f",
        "method_version": METHOD_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "task": "disambiguation_qa",
        "benchmark": "BBH",
        "dataset_format": "mars",
        "formal_seeds": list(SEEDS),
        "development_seeds_excluded": list(range(47, 56)),
        "formal_seed_prior_use_count": seed_audit["prior_use_count"],
        "formal_seed_prior_use_gate": seed_audit["gate"],
        "arms": list(ARMS),
        "settings": ARMS,
        "formal_cells": 15,
        "model": "qwen3-14b",
        "thinking": False,
        "solver_temperature": 0,
        "solver_max_tokens": 1800,
        "agents": 5,
        "initialization_mode": "shared_identical",
        "proposal_memory_mode": "off",
        "train_size": 75,
        "validation_size": 50,
        "test_size": 125,
        "epochs": 1,
        "update_every": 10,
        "updates_per_optimized_run": UPDATES,
        "source_candidates_per_update": 4,
        "second_stage_slots_per_update": 4,
        "max_provider_calls_per_run": CALL_CEILING,
        "max_total_tokens_per_run": TOKEN_CEILING,
        "budget_exhaustion_policy": "INVALIDATE_AND_HOLD",
        "paper_untouched_test": False,
        "historical_test_exposure": True,
        "test_interpretation": "frozen-split final generalization evaluation",
        "untouched_paper_heldout_claim_authorized": False,
        "validation_role": "post-training generalization audit",
        "validation_selection_enabled": False,
        "test_used_for_selection": False,
        "phase_authorizations": {
            "train": "V17_FORMAL_TRAIN_AUTHORIZED",
            "validation": "V17_FORMAL_VALIDATION_AUTHORIZED",
            "test": "V17_FORMAL_TEST_AUTHORIZED",
        },
        "primary_metric": classifier["primary_metric"],
        "classifier_hash": classifier["classifier_hash"],
        "dataset_freeze_gate": data["gate"],
        "frozen_source_semantics_gate": semantics["gate"],
        "no_method_redesign_after_training_starts": True,
    }
    design = """# V17 Formal Five-Arm Experiment\n\nThis preregistration freezes a 5-arm by 3-seed comparison on BBH\n`disambiguation_qa`. Seeds 56-58 are the formal seeds; Seeds 47-55 are\ndevelopment evidence and are excluded from formal averages.\n\nValidation is a post-training generalization audit only. It cannot select or\nmutate a final state. Test evaluates the same frozen final states once after a\npre-test seal. The test split was historically exposed during development, so\nthis experiment is not described as an untouched paper heldout evaluation.\n\nS0 is static. S1 is the new explicit generic 2x2 dual-round-robin baseline.\nS2, S3, and S4 reuse the frozen G-Matched, R-M20, and R-M2F settings. Equal\nopportunity ceilings and common hard run ceilings are used; realized compute\nneed not be identical. No outcomes may change the method, cases, seeds, order,\nmetric, classifier, or thresholds.\n"""
    EXPERIMENT_ROOT.mkdir(parents=True)
    (EXPERIMENT_ROOT / "DESIGN_SPEC.md").write_text(design, encoding="utf-8")
    write_json(EXPERIMENT_ROOT / "dataset_freeze.json", data)
    write_json(EXPERIMENT_ROOT / "formal_matrix.json", matrix)
    write_json(EXPERIMENT_ROOT / "success_classifier.json", classifier)
    write_json(EXPERIMENT_ROOT / "execution_order.json", order)
    write_json(EXPERIMENT_ROOT / "preregistration.json", prereg)
    print(json.dumps({
        "status": "PASS" if data["gate"] == semantics["gate"] == "PASS" else "FAIL",
        "formal_seeds": list(SEEDS), "formal_cells": 15,
        "api_calls": 0, "model_calls": 0,
        "validation_calls": 0, "test_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
