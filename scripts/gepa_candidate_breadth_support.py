from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.candidate_selection import common_monotone_safe_key
from multi_dataset_diverse_rl.protocol import CandidateBudgetContract


AUTHORIZATION_ENV = "GEPA_PROPOSAL_BREADTH_API_AUTHORIZED"
PROBE_VERSION = "gepa_proposal_breadth_fixed_parent_v1"
BREADTHS = (2, 4)
REQUESTED_SOURCE_COUNT = 4
CASES = ((59, 3), (61, 5))
ALLOWED_LABELS = (
    "PROPOSAL_BREADTH_SUPPORTED",
    "PROPOSAL_BREADTH_THROUGHPUT_ONLY",
    "NO_PROPOSAL_BREADTH_SIGNAL",
    "PROPOSAL_BREADTH_HARMFUL",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def widen_candidate_budget(system: Any, count: int = REQUESTED_SOURCE_COUNT) -> None:
    """Apply an experiment-local candidate-count override after v15 resolution."""
    if system.protocol.candidates_per_target_branch != 2:
        raise ValueError("expected frozen v15 two-candidate source budget")
    system.cfg = replace(
        system.cfg,
        tcs=replace(system.cfg.tcs, num_candidates_per_parent=count),
        evaluation=replace(system.cfg.evaluation, stage_b_candidate_budget=count),
    )
    old = system.protocol.candidate_budget_contract
    contract = CandidateBudgetContract(
        target_branch_count=old.target_branch_count,
        candidates_per_target_branch=count,
        total_generated_candidates_per_update=old.target_branch_count * count,
        stage_b_budget_per_branch=count,
        total_stage_b_candidate_budget=old.target_branch_count * count,
        stage_a_channel_top_k=old.stage_a_channel_top_k,
        representative_size=old.representative_size,
        coverage_size=old.coverage_size,
        conversion_size=old.conversion_size,
        preservation_size=old.preservation_size,
    )
    system.protocol = replace(system.protocol, candidate_budget_contract=contract)
    if system.protocol.candidates_per_target_branch != count:
        raise AssertionError("experiment-local breadth override failed")


def candidate_row(runtime: Any, *, stage: str, source_slot: int | None) -> dict[str, Any]:
    constraint = runtime.constraint
    evaluation = runtime.final_evaluation
    if constraint is None or evaluation is None:
        raise ValueError("candidate is not fully evaluated")
    source_hash = (
        str((runtime.module2_diagnostics or {}).get("source_candidate_hash", ""))
        if stage == "revision" else runtime.prompt_hash
    )
    return {
        "candidate_hash": runtime.prompt_hash,
        "candidate_stage": stage,
        "source_candidate_hash": source_hash,
        "source_slot": source_slot,
        "valid": True,
        "feasible": bool(constraint.passed),
        "train_target_gain": int(constraint.target_gain),
        "train_vote_gain": int(constraint.vote_gain_count),
        "train_vote_loss": int(constraint.vote_loss_count),
        "train_vote_net": int(constraint.vote_net_gain),
        "assigned_residual_repair_count": int(
            evaluation.marginal.assigned_residual_repair_count
        ),
        "quality_key": list(common_monotone_safe_key(evaluation, runtime.generation)),
    }


def choose_pool(rows: Sequence[dict[str, Any]], breadth: int) -> dict[str, Any]:
    source_hashes = {
        str(row["candidate_hash"])
        for row in rows
        if row["candidate_stage"] == "source"
        and row["source_slot"] is not None
        and int(row["source_slot"]) <= breadth
    }
    pool = [
        row for row in rows
        if (
            row["candidate_stage"] == "source"
            and str(row["candidate_hash"]) in source_hashes
        ) or (
            row["candidate_stage"] == "revision"
            and str(row["source_candidate_hash"]) in source_hashes
        )
    ]
    feasible = [row for row in pool if bool(row["feasible"])]
    winner = max(feasible, key=lambda row: tuple(row["quality_key"]), default=None)
    best_loss = min((int(row["train_vote_loss"]) for row in feasible), default=None)
    best_net = max((int(row["train_vote_net"]) for row in feasible), default=None)
    return {
        "breadth": breadth,
        "source_candidate_count": len(source_hashes),
        "valid_candidate_count": len(pool),
        "feasible_candidate_count": len(feasible),
        "zero_loss_feasible_count": sum(
            int(row["train_vote_loss"]) == 0 for row in feasible
        ),
        "best_feasible_train_vote_loss": best_loss,
        "best_feasible_vote_net": best_net,
        "winner_hash": str(winner["candidate_hash"]) if winner else "",
        "winner_train_target_gain": int(winner["train_target_gain"]) if winner else None,
        "winner_train_vote_loss": int(winner["train_vote_loss"]) if winner else None,
        "winner_train_vote_net": int(winner["train_vote_net"]) if winner else None,
        "pool_candidate_hashes": sorted(str(row["candidate_hash"]) for row in pool),
    }


def finalize_pool_comparison(n2: dict[str, Any], n4: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    n2_hashes = set(map(str, n2["pool_candidate_hashes"]))
    n4_only = [row for row in rows if str(row["candidate_hash"]) not in n2_hashes]
    n4_only_feasible = [row for row in n4_only if bool(row["feasible"])]
    reference_loss = n2["best_feasible_train_vote_loss"]
    n4["n4_only_valid_candidate_count"] = len(n4_only)
    n4["n4_only_feasible_candidate_count"] = len(n4_only_feasible)
    n4["n4_only_zero_loss_feasible_count"] = sum(
        int(row["train_vote_loss"]) == 0 for row in n4_only_feasible
    )
    n4["n4_only_lower_than_n2_best_loss_feasible_count"] = sum(
        reference_loss is not None
        and int(row["train_vote_loss"]) < int(reference_loss)
        for row in n4_only_feasible
    )
    n4["n2_pool_is_subset"] = set(n2_hashes).issubset(
        set(map(str, n4["pool_candidate_hashes"]))
    )


def classify(case_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    novel_safer = sum(
        int(row["n4"]["n4_only_zero_loss_feasible_count"])
        + int(row["n4"]["n4_only_lower_than_n2_best_loss_feasible_count"])
        for row in case_rows
    )
    feasible_delta = sum(
        int(row["n4"]["feasible_candidate_count"])
        - int(row["n2"]["feasible_candidate_count"])
        for row in case_rows
    )
    validation_vote_delta = sum(
        int(row["n4"]["validation_vote_delta"])
        - int(row["n2"]["validation_vote_delta"])
        for row in case_rows
    )
    selected_loss_reduction = sum(
        int(row["n2"]["winner_train_vote_loss"])
        - int(row["n4"]["winner_train_vote_loss"])
        for row in case_rows
        if row["n2"]["winner_train_vote_loss"] is not None
        and row["n4"]["winner_train_vote_loss"] is not None
    )
    if novel_safer > 0 and selected_loss_reduction > 0 and validation_vote_delta >= 0:
        label = "PROPOSAL_BREADTH_SUPPORTED"
    elif validation_vote_delta < 0 and selected_loss_reduction <= 0:
        label = "PROPOSAL_BREADTH_HARMFUL"
    elif feasible_delta > 0 and novel_safer == 0:
        label = "PROPOSAL_BREADTH_THROUGHPUT_ONLY"
    else:
        label = "NO_PROPOSAL_BREADTH_SIGNAL"
    if label not in ALLOWED_LABELS:
        raise AssertionError("invalid proposal breadth label")
    return {
        "classifier_version": "gepa_proposal_breadth_classifier_v1",
        "rules_frozen_before_validation_readout": True,
        "novel_safer_feasible_candidate_count": novel_safer,
        "aggregate_feasible_candidate_delta": feasible_delta,
        "aggregate_selected_train_vote_loss_reduction": selected_loss_reduction,
        "aggregate_n4_minus_n2_validation_vote_delta": validation_vote_delta,
        "final_label": label,
    }
