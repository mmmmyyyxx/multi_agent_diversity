"""Deterministic identity and policy audits for native-feed branches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


LAYER2_CONTRACT_FILES = (
    "multi_dataset_diverse_rl/native_feed.py",
    "multi_dataset_diverse_rl/native_feed_audit.py",
    "multi_dataset_diverse_rl/team_search/schemas.py",
    "multi_dataset_diverse_rl/team_search/task_builder.py",
    "multi_dataset_diverse_rl/team_search/controller.py",
    "multi_dataset_diverse_rl/team_search/candidate_evaluator.py",
    "multi_dataset_diverse_rl/team_search/candidate_selector.py",
    "multi_dataset_diverse_rl/team_search/progressive_evaluation.py",
    "multi_dataset_diverse_rl/team_search/primary_responsibility_scheduler.py",
    "multi_dataset_diverse_rl/team_search/primary_responsibility_binding.py",
    "multi_dataset_diverse_rl/team_search/system_runtime.py",
    "multi_dataset_diverse_rl/team_search/run_record.py",
    "multi_dataset_diverse_rl/local_optimizers/base.py",
    "multi_dataset_diverse_rl/local_optimizers/schemas.py",
    "multi_dataset_diverse_rl/responsibility.py",
    "multi_dataset_diverse_rl/candidate_selection.py",
    "multi_dataset_diverse_rl/shadow_gate.py",
    "multi_dataset_diverse_rl/peer_state.py",
    "multi_dataset_diverse_rl/evaluation/categorical_profiles.py",
    "multi_dataset_diverse_rl/evaluation/endpoint_identifiability.py",
)


def normalized_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def layer2_contract_manifest(root: Path) -> dict[str, Any]:
    files = []
    for relative in LAYER2_CONTRACT_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        files.append(
            {
                "path": relative,
                "sha256_normalized_lf": sha256_bytes(normalized_bytes(path)),
            }
        )
    payload = {"version": "unified_layer2_transition_evidence_contract_v3", "files": files}
    digest = sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return {**payload, "layer2_contract_hash": digest}


def data_access_matrix() -> list[dict[str, Any]]:
    return [
        {
            "variant": "Native GEPA",
            "optimize_train": True,
            "optimize_internal_val_pareto": True,
            "validation50": False,
            "test50": False,
        },
        {
            "variant": "GEPA search core + Layer2-owned evidence",
            "optimize_train": True,
            "optimize_internal_val_pareto": True,
            "validation50": False,
            "test50": False,
        },
        {
            "variant": "Native MARS",
            "optimize_train": True,
            "optimize_internal_val_pareto": False,
            "validation50": False,
            "test50": False,
        },
        {
            "variant": "MARS search core + Layer2-owned evidence",
            "optimize_train": True,
            "optimize_internal_val_pareto": False,
            "validation50": False,
            "test50": False,
        },
    ]


def claim_registry() -> dict[str, Any]:
    return {
        "allowed": [
            "GEPA_SEARCH_CORE_PLUS_LAYER2_VS_NATIVE_GEPA",
            "MARS_SEARCH_CORE_PLUS_LAYER2_VS_NATIVE_MARS",
        ],
        "forbidden_cross_backend_claim": "GEPA_OUTPERFORMS_MARS",
        "reason": (
            "Treatment replaces optimizer-local example selection and adds the "
            "complete Layer2 allocation, curriculum and team-admission system. "
            "It does not isolate individual Layer2 components or compare backends."
        ),
    }


def budget_semantics() -> dict[str, Any]:
    return {
        "GEPA": {
            "native_units": [
                "metric calls",
                "train minibatch rollouts",
                "full optimizer-val evaluations",
                "reflection calls",
            ]
        },
        "MARS": {
            "native_units": [
                "Planner calls",
                "Teacher calls",
                "Critic calls",
                "Student calls",
                "full Optimize Target evaluations",
            ]
        },
        "within_optimizer_pair_budget_identical": True,
        "cross_optimizer_budget_equality_claimed": False,
        "realized_reporting": ["solver calls", "optimizer calls", "tokens", "wall time"],
    }
