from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.v17_formal_support import split_freeze
from scripts.v18_hybrid_online_accumulation_support import (
    ARMS,
    HYBRID,
    SEEDS,
    UPDATES,
    W1,
    canonical_json,
    hybrid_targets,
    sha256_json,
)


BASELINE_COMMIT = "c25a593d8f17e28844acab64d7a65c65fc73a972"
V17_PREREG = ROOT / "experiments/v17_formal_5arm_3seed_20260813/preregistration.json"
CLASSIFIER = ROOT / "experiments/v18_hybrid_online_accumulation_pilot_20260822/classifier_spec.json"


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_prior_use() -> dict[str, Any]:
    matches: dict[str, list[str]] = {}
    for seed in SEEDS:
        pattern = rf"(seed|seeds|experimental_seed|formal_seeds?)[^0-9]{{0,8}}{seed}([^0-9]|$)|seed{seed}([^0-9]|$)"
        process = subprocess.run(
            [
                "git", "grep", "-n", "-i", "-E", pattern,
                BASELINE_COMMIT, "--", ":!Dataset/**", ":!strict_splits*/**",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.returncode not in (0, 1):
            raise RuntimeError(process.stderr or "seed inventory failed")
        matches[str(seed)] = [line for line in process.stdout.splitlines() if line]
    return {
        "inventory_commit": BASELINE_COMMIT,
        "seeds": list(SEEDS),
        "prior_use_count_by_seed": {
            seed: len(rows) for seed, rows in matches.items()
        },
        "gate": "PASS" if not any(matches.values()) else "FAIL",
    }


def _protocol_snapshot() -> dict[str, Any]:
    cfg = Config.from_flat(
        experiment_setting="experimental_v16_efficacy_g_matched",
        out_dir="runs/v18_protocol_probe",
        final_test_enabled=False,
        agents=5,
        epochs=1,
        update_every=10,
        num_candidates_per_parent=2,
        candidate_eval_pool_size=75,
        stage_b_candidate_budget=2,
        proposal_memory_mode="off",
    )
    system = PromptEnsembleOptimizationSystem(cfg)
    protocol = system.protocol
    return {
        "underlying_setting": protocol.name,
        "target_selection_policy": protocol.target_selection_policy,
        "sample_pool_policy": protocol.sample_pool_policy,
        "tcs_context_policy": protocol.tcs_context_policy,
        "responsibility_refresh_policy": protocol.responsibility_refresh_policy,
        "target_branch_count": protocol.target_branch_count,
        "candidates_per_target_branch": protocol.candidates_per_target_branch,
        "total_generated_candidates_per_update": (
            protocol.candidate_budget_contract.total_generated_candidates_per_update
        ),
        "generic_revision_enabled": protocol.generic_revision_enabled,
        "compatibility_repair_enabled": protocol.compatibility_repair_enabled,
        "candidate_acceptance_policy": protocol.candidate_acceptance_policy,
        "candidate_ranking_policy": protocol.candidate_ranking_policy,
        "proposal_memory_mode": cfg.tcs.proposal_memory_mode,
        "initialization_mode": cfg.training.initialization_mode,
        "shared_prompt_hashes": [
            system.prompt_hash(agent.current_prompt) for agent in system.agents
        ],
        "w1_selector_source_hash": hashlib.sha256(
            inspect.getsource(PromptEnsembleOptimizationSystem.select_targets).encode()
        ).hexdigest(),
        "hybrid_selector_source_hash": hashlib.sha256(
            inspect.getsource(hybrid_targets).encode()
        ).hexdigest(),
    }


def build_registry() -> dict[str, Any]:
    prereg = json.loads(V17_PREREG.read_text(encoding="utf-8"))
    classifier = json.loads(CLASSIFIER.read_text(encoding="utf-8"))
    if int(prereg["updates_per_optimized_run"]) != UPDATES:
        raise ValueError("canonical V17 online horizon changed")
    if int(prereg["source_candidates_per_update"]) != 4:
        raise ValueError("canonical V17 source budget changed")
    if int(prereg["second_stage_slots_per_update"]) != 4:
        raise ValueError("canonical V17 second-stage budget changed")
    result = {
        "registry_version": "v18_hybrid_online_accumulation_registry_v1",
        "execution_commit": git("rev-parse", "HEAD"),
        "baseline_commit": BASELINE_COMMIT,
        "task": "disambiguation_qa",
        "dataset_format": "mars",
        "model": "qwen3-14b",
        "thinking": False,
        "agents": 5,
        "aggregation": "equal_weight_plurality_tie_abstain",
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "execution_order": {
            "59": [W1, HYBRID],
            "60": [HYBRID, W1],
            "61": [W1, HYBRID],
        },
        "trajectory_count": len(SEEDS) * len(ARMS),
        "update_opportunities_per_trajectory": UPDATES,
        "epochs": 1,
        "update_every": 10,
        "train_size": 75,
        "validation_size": 50,
        "test_size_identity_only": 125,
        "source_candidates_per_target": 2,
        "loss_blind_revision_per_valid_source": 1,
        "conceptual_source_branch_budget_per_trajectory": UPDATES * 2,
        "conceptual_source_candidate_budget_per_trajectory": UPDATES * 4,
        "validation_schedule": "initial_plus_changed_state_only",
        "validation_role": "analysis_only_no_selection",
        "new_test_calls": 0,
        "branch_reuse_policy": "none_across_executed_trajectories",
        "generation_key_fields": [
            "experiment_seed", "update_index", "target_member",
            "source_slot", "candidate_stage", "parent_team_hash",
        ],
        "arm_contract": {
            W1: "W1 rank1 plus W1 rank2",
            HYBRID: "W1 rank1 plus frozen responsibility-constrained RR",
        },
        "protocol": _protocol_snapshot(),
        "split_freeze": split_freeze(),
        "seed_freeze": _seed_prior_use(),
        "v17_preregistration_sha256": sha256_file(V17_PREREG),
        "classifier": classifier,
        "classifier_hash": sha256_json(classifier),
        "phase_a_zero_api": True,
    }
    result["registry_content_hash"] = sha256_json(result)
    return result


def main() -> None:
    print(json.dumps(build_registry(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
