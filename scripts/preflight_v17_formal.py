from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.v17_formal_support import (
    ARMS,
    EXECUTION_ORDER,
    EXPERIMENT_ROOT,
    ROOT,
    SEEDS,
    formal_target_schedule,
    formal_seed_prior_use,
    git,
    read_json,
    recursive_sanitize,
    source_semantics_diff,
    split_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow_dirty", type=int, choices=(0, 1), default=1)
    parser.add_argument("--expected_commit", default="")
    args = parser.parse_args()
    errors: list[str] = []
    data = split_freeze()
    if data["gate"] != "PASS":
        errors.extend(data["errors"])
    semantics = source_semantics_diff()
    if semantics["gate"] != "PASS":
        errors.append("frozen_v16_semantics_changed")
    head = git("rev-parse", "HEAD")
    if args.expected_commit and head != args.expected_commit:
        errors.append("execution_commit")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if dirty and not args.allow_dirty:
        errors.append("dirty_worktree")
    required = (
        "DESIGN_SPEC.md", "preregistration.json", "formal_matrix.json",
        "success_classifier.json", "dataset_freeze.json", "execution_order.json",
    )
    for name in required:
        if not (EXPERIMENT_ROOT / name).is_file():
            errors.append(f"missing_definition:{name}")
    prereg = read_json(EXPERIMENT_ROOT / "preregistration.json")
    if prereg.get("formal_seeds") != list(SEEDS) or prereg.get("formal_cells") != 15:
        errors.append("seed_or_cell_contract")
    if prereg.get("paper_untouched_test") is not False:
        errors.append("test_exposure_contract")
    protocols = {}
    for arm, setting in ARMS.items():
        protocol = PromptEnsembleOptimizationSystem(Config.from_flat(
            experiment_setting=setting, out_dir="runs/v17_preflight",
        )).protocol
        protocols[arm] = protocol
    s0, s1, s2, s3, s4 = (protocols[name] for name in ARMS)
    if s0.optimization_enabled or s0.target_branch_count != 0:
        errors.append("S0_contract")
    if not (
        not s1.modules.member_aware_dual_target_search
        and s1.target_selection_policy == "round_robin_dual_formal"
        and s1.sample_pool_policy == "individual_errors"
        and s1.tcs_context_policy == "generic_accuracy"
        and s1.responsibility_refresh_policy == "off"
        and not s1.service_routing_enabled
        and s1.target_branch_count == 2
        and s1.candidates_per_target_branch == 2
        and s1.generic_revision_enabled
        and not s1.compatibility_repair_enabled
    ):
        errors.append("S1_contract")
    if not (s2.generic_revision_enabled and not s2.compatibility_repair_enabled):
        errors.append("S2_contract")
    if s3.generic_revision_enabled or s3.compatibility_repair_enabled:
        errors.append("S3_contract")
    if s4.generic_revision_enabled or not s4.compatibility_repair_enabled:
        errors.append("S4_contract")
    common = {
        (p.target_branch_count, p.candidates_per_target_branch,
         p.candidate_acceptance_policy, p.candidate_ranking_policy)
        for p in (s1, s2, s3, s4)
    }
    if len(common) != 1:
        errors.append("optimized_budget_or_acceptance_mismatch")
    order = read_json(EXPERIMENT_ROOT / "execution_order.json")
    for phase in ("training", "validation", "test"):
        expected = {str(seed): list(EXECUTION_ORDER[seed]) for seed in SEEDS}
        if order.get(phase) != expected:
            errors.append(f"execution_order:{phase}")
    if order.get("S1_target_schedule") != {
        str(seed): formal_target_schedule(seed) for seed in SEEDS
    }:
        errors.append("S1_target_schedule")
    seed_audit = formal_seed_prior_use()
    if seed_audit["gate"] != "PASS":
        errors.append("formal_seed_prior_use")
    sanitation = []
    for name in required:
        path = EXPERIMENT_ROOT / name
        if path.suffix == ".json":
            sanitation.extend(recursive_sanitize(read_json(path)))
    if sanitation:
        errors.append("definition_sanitization")
    payload = {
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "git_head": head,
        "formal_seeds": list(SEEDS),
        "formal_seed_prior_use_count": seed_audit["prior_use_count"],
        "dataset_hash_gate": data["gate"],
        "frozen_v16_semantics_gate": semantics["gate"],
        "arms": list(ARMS),
        "train_cells": 15,
        "api_calls": 0,
        "model_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "all_artifacts_repo_local": all(
            ROOT.resolve() in (EXPERIMENT_ROOT / name).resolve().parents
            for name in required
        ),
    }
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
