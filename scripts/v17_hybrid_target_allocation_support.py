from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generic_m20_probe_support import state_hash, system_for


W1 = "W1_TOP2"
RR = "RR_TOP2"
HYBRID = "HYBRID_EXPLOIT_EXPLORE"
ARMS = (W1, RR, HYBRID)
AUTHORIZATION_ENV = "V17_HYBRID_TARGET_LOW_API_AUTHORIZED"
MEMBER_AWARE_SETTING = "experimental_v16_efficacy_g_matched"
ALLOWED_DIAGNOSES = (
    "HYBRID_RECOVERY_SUPPORTED",
    "HYBRID_THROUGHPUT_ONLY",
    "HYBRID_VALUE_ONLY",
    "HYBRID_NOT_SUPPORTED",
    "HYBRID_HARMFUL",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def rr_eligible_order(seed: int, update_index: int, eligible: Iterable[int]) -> list[int]:
    pool = set(map(int, eligible))
    start = (int(seed) + 2 * int(update_index)) % 5
    return [agent for offset in range(5) if (agent := (start + offset) % 5) in pool]


def arm_specs(case: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    ordered = [int(row["agent_id"]) for row in case["w1_priority_rows"]]
    eligible = set(map(int, case["responsibility_eligible_ids"]))
    if len(eligible) < 2 or not set(ordered).issubset(eligible):
        raise ValueError("invalid responsibility-eligible W1 pool")
    w1_targets = ordered[:2]
    rr_order = rr_eligible_order(case["source_seed"], case["source_update_index"], eligible)
    if len(rr_order) < 2:
        raise ValueError("RR requires two responsibility-eligible targets")
    rr_targets = rr_order[:2]
    exploit = w1_targets[0]
    explore = next(agent for agent in rr_order if agent != exploit)
    result = {
        W1: [
            {"target_member": w1_targets[0], "branch_type": "exploit"},
            {"target_member": w1_targets[1], "branch_type": "w1_second"},
        ],
        RR: [
            {"target_member": rr_targets[0], "branch_type": "rr_first"},
            {"target_member": rr_targets[1], "branch_type": "rr_second"},
        ],
        HYBRID: [
            {"target_member": exploit, "branch_type": "exploit"},
            {"target_member": explore, "branch_type": "explore"},
        ],
    }
    for rows in result.values():
        targets = [row["target_member"] for row in rows]
        if len(targets) != 2 or len(set(targets)) != 2 or not set(targets).issubset(eligible):
            raise ValueError("arm target contract failed")
    return result


def context_hashes(case: dict[str, Any], target: int) -> set[str]:
    hashes = set(map(str, case["active_residual_hashes_by_agent"][str(target)]))
    if not hashes:
        raise ValueError("member-aware target has no active residual")
    return hashes


def generation_config(case: dict[str, Any]) -> dict[str, Any]:
    config = case["base_config"]
    return {
        "agent_model": "qwen3-14b",
        "optimizer_model": "qwen3-14b",
        "evaluator_model": "qwen3-14b",
        "temperature": config["temperature"],
        "solver_max_tokens": config["solver_max_tokens"],
        "teacher_temperature": config["teacher_temperature"],
        "critic_temperature": config["critic_temperature"],
        "student_temperature": config["student_temperature"],
        "source_candidates_per_target": 2,
        "loss_blind_revision_per_valid_source": 1,
        "context_mode": "member_aware_residual_search",
        "proposal_mode": "generic_evolution",
        "proposal_memory_mode": "off",
        "final_test_enabled": False,
    }


def branch_key(case: dict[str, Any], target: int) -> str:
    payload = {
        "parent_id": case["case_id"],
        "target_member": int(target),
        "context_mode": "member_aware_residual_search",
        "proposal_mode": "generic_evolution",
        "generation_config_hash": canonical_hash(generation_config(case)),
        "source_slot_definition": "two_sources_one_revision_each_v1",
    }
    return canonical_hash(payload)


def probe_system(case: dict[str, Any], *, target: int, out_dir: Path, cache_path: Path | str):
    local = dict(case)
    local["target_agent_id"] = int(target)
    local["active_lane"] = case["active_lane_by_agent"][str(target)]
    system = system_for(
        local,
        setting=MEMBER_AWARE_SETTING,
        out_dir=out_dir,
        cache_path=cache_path,
        evolution_variant="m20_current_v15",
    )
    if system.protocol.compatibility_repair_enabled:
        raise AssertionError("compatibility repair is forbidden")
    if not system.protocol.generic_revision_enabled:
        raise AssertionError("loss-blind generic revision must be enabled")
    if system.cfg.tcs.proposal_memory_mode != "off":
        raise AssertionError("proposal memory must remain off")
    return system


def choose_would_commit(evaluator: Any, branch_winners: Iterable[Any]) -> Any | None:
    winners = [
        row for row in branch_winners
        if row is not None and getattr(row, "accepted", None) is not None
    ]
    return max(winners, key=evaluator._cross_branch_key, default=None)


def branch_object(target: int, rank: int, accepted: Any, incumbent: Any) -> Any:
    return SimpleNamespace(
        accepted=accepted,
        incumbent=incumbent,
        target_selection_rank=int(rank),
        target_agent_id=int(target),
    )


def realized_delta(would_commit: bool, parent: int, hypothetical: int) -> int:
    return int(hypothetical - parent) if would_commit else 0


def immutable_state_hash(system: Any) -> str:
    return state_hash(system)


def _wtl(hybrid: list[int], reference: list[int]) -> dict[str, int]:
    deltas = [left - right for left, right in zip(hybrid, reference, strict=True)]
    return {
        "wins": sum(value > 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
        "losses": sum(value < 0 for value in deltas),
    }


def classify(parent_rows: list[dict[str, Any]], feasible: dict[str, int]) -> dict[str, Any]:
    def values(arm: str, metric: str) -> list[int]:
        return [int(row[arm][metric]) for row in parent_rows]

    hybrid_vote = values(HYBRID, "validation_vote_delta")
    w1_vote = values(W1, "validation_vote_delta")
    rr_vote = values(RR, "validation_vote_delta")
    hybrid_oracle = values(HYBRID, "validation_oracle_delta")
    w1_oracle = values(W1, "validation_oracle_delta")
    vote_wtl = _wtl(hybrid_vote, w1_vote)
    rr_wtl = _wtl(hybrid_vote, rr_vote)
    oracle_wtl = _wtl(hybrid_oracle, w1_oracle)
    vote_diff = sum(hybrid_vote) - sum(w1_vote)
    oracle_diff = sum(hybrid_oracle) - sum(w1_oracle)
    vote_supported = vote_diff > 0 and vote_wtl["wins"] > vote_wtl["losses"]
    feasibility_supported = feasible[HYBRID] > feasible[W1]
    oracle_supported = oracle_diff > 0 and oracle_wtl["wins"] > oracle_wtl["losses"]
    recovered_fraction = None
    if feasible[RR] > feasible[W1]:
        recovered_fraction = (
            (feasible[HYBRID] - feasible[W1])
            / (feasible[RR] - feasible[W1])
        )
    if vote_supported and feasibility_supported:
        diagnosis = "HYBRID_RECOVERY_SUPPORTED"
    elif feasibility_supported and not vote_supported and vote_diff >= 0:
        diagnosis = "HYBRID_THROUGHPUT_ONLY"
    elif vote_supported and not feasibility_supported:
        diagnosis = "HYBRID_VALUE_ONLY"
    elif vote_diff < 0 and vote_wtl["losses"] > vote_wtl["wins"]:
        diagnosis = "HYBRID_HARMFUL"
    else:
        diagnosis = "HYBRID_NOT_SUPPORTED"
    assert diagnosis in ALLOWED_DIAGNOSES
    return {
        "classifier_version": "v17_hybrid_target_allocation_classifier_v1",
        "hybrid_vote_benefit_supported": vote_supported,
        "hybrid_feasibility_recovery_supported": feasibility_supported,
        "hybrid_oracle_benefit_supported": oracle_supported,
        "hybrid_minus_w1_vote_sum": vote_diff,
        "hybrid_minus_w1_oracle_sum": oracle_diff,
        "hybrid_minus_w1_vote_wtl": vote_wtl,
        "hybrid_minus_rr_vote_wtl": rr_wtl,
        "hybrid_minus_w1_oracle_wtl": oracle_wtl,
        "recovered_fraction": recovered_fraction,
        "final_pilot_diagnosis": diagnosis,
    }
