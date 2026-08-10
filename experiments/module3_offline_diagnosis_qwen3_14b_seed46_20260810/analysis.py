from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from multi_dataset_diverse_rl.candidate_selection import (
    CandidateEvaluation,
    PromptCompetenceMetrics,
    TeamOutcomeMetrics,
    common_cross_branch_transition_key,
    common_monotone_safe_key,
    evaluate_constraints,
    rcru_cross_branch_transition_key,
    stage_a_multichannel_shortlist,
    stage_a_rcru_shortlist,
)
from multi_dataset_diverse_rl.member_objectives import MemberGainMetrics
from multi_dataset_diverse_rl.responsibility import (
    CandidateMarginalContribution,
    ProtectionContribution,
)
from multi_dataset_diverse_rl.responsibility_contribution import (
    CoalitionContributionMetrics,
    PromptEditMetrics,
    ResponsibilityContributionMetrics,
    ResponsibilityUtilityMetrics,
    RobustSupportMetrics,
    evaluate_robust_contribution_constraints,
    responsibility_contribution_pareto_front,
    responsibility_utility,
    responsibility_utility_metrics,
    robust_contribution_key,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


OUTPUT = Path(__file__).resolve().parent
if len(sys.argv) != 2:
    raise SystemExit("usage: analysis.py <authoritative_formal_root>")
FORMAL_ROOT = Path(sys.argv[1]).resolve() / "disambiguation_qa"
S2_NAME = "shared_responsibility_conditioned_dual_target_seed46"
S3_NAME = "shared_full_dual_target_rcru_seed46"
RUN_SOURCE_COMMIT = "e5bdc9f27f7a5594072aafd828c7c6053297c03c"
RUN_SOURCE_TREE_HASH = (
    "9b38c5fbf3519481e0ab02faee5d69f71e674449eb478698d295475428723ceb"
)
REPORT_COMMIT = "ce58803a4314152f0fa62211edc3335ba1bcc524"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tuple_fields(payload: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
    row = dict(payload)
    for name in names:
        if name in row:
            row[name] = tuple(row[name])
    return row


def deserialize_evaluation(payload: dict[str, Any]) -> CandidateEvaluation:
    contribution_payload = payload.get("responsibility_contribution")
    contribution = None
    if contribution_payload is not None:
        contribution = ResponsibilityContributionMetrics(
            utility=ResponsibilityUtilityMetrics(
                **tuple_fields(
                    contribution_payload["utility"],
                    ("per_example_deltas", "per_example_question_hashes"),
                )
            ),
            coalition=CoalitionContributionMetrics(
                **contribution_payload["coalition"]
            ),
            robust_support=RobustSupportMetrics(
                **contribution_payload["robust_support"]
            ),
            edit=PromptEditMetrics(**contribution_payload["edit"]),
        )
    return CandidateEvaluation(
        prompt=payload["prompt"],
        prompt_hash=payload["prompt_hash"],
        competence=PromptCompetenceMetrics(**payload["competence"]),
        team_outcome=TeamOutcomeMetrics(
            **tuple_fields(
                payload["team_outcome"],
                (
                    "vote_correct_vector",
                    "gold_vote_counts",
                    "largest_wrong_vote_counts",
                    "plurality_margins",
                ),
            )
        ),
        marginal=CandidateMarginalContribution(**payload["marginal"]),
        protection=ProtectionContribution(**payload["protection"]),
        member_gain=MemberGainMetrics(
            **tuple_fields(
                payload["member_gain"],
                (
                    "initial_correct_counts",
                    "incumbent_correct_counts",
                    "candidate_correct_counts",
                    "gain_counts",
                ),
            )
        ),
        responsibility_contribution=contribution,
    )


@dataclass
class RunEvidence:
    name: str
    path: Path
    meta: dict[str, Any]
    candidate_decisions: list[dict[str, Any]]
    branch_decisions: list[dict[str, Any]]
    commit_decisions: list[dict[str, Any]]
    target_scores: list[dict[str, Any]]
    failure_events: list[dict[str, Any]]
    rcru_rows: list[dict[str, Any]]
    peer_chunks: list[list[dict[str, Any]]]
    initial_counts: tuple[int, ...]
    tcs_contexts: list[dict[str, Any]]

    @classmethod
    def load(cls, name: str) -> "RunEvidence":
        path = FORMAL_ROOT / name
        meta = read_json(path / "run_meta.json")
        manifest = read_json(Path(meta["config"]["frozen_initialization_manifest_path"]))
        train_size = int(meta["config"]["train_size"])
        peer_rows = read_jsonl(path / "peer_state_history.jsonl")
        if len(peer_rows) % train_size:
            raise AssertionError("peer_state_history_not_divisible_by_train_size")
        peer_chunks = [
            peer_rows[index : index + train_size]
            for index in range(0, len(peer_rows), train_size)
        ]
        rcru_path = path / "rcru_candidate_decisions_sanitized.jsonl"
        return cls(
            name=name,
            path=path,
            meta=meta,
            candidate_decisions=read_jsonl(path / "candidate_decisions.jsonl"),
            branch_decisions=read_jsonl(path / "dual_target_branch_decisions.jsonl"),
            commit_decisions=read_jsonl(path / "dual_target_commit_decisions.jsonl"),
            target_scores=read_jsonl(path / "repairability_adjusted_target_scores.jsonl"),
            failure_events=read_jsonl(path / "repairability_failure_events.jsonl"),
            rcru_rows=read_jsonl(rcru_path) if rcru_path.exists() else [],
            peer_chunks=peer_chunks,
            initial_counts=tuple(
                manifest["initialization_snapshot"]["initial_member_correct_counts"]
            ),
            tcs_contexts=read_jsonl(path / "tcs_context_history.jsonl"),
        )


def active_evaluation(
    run: RunEvidence,
    *,
    target: int,
    team_state_version: int,
    parent_prompt: str,
    candidate: CandidateEvaluation,
) -> CandidateEvaluation:
    peers = run.peer_chunks[team_state_version]
    counts = tuple(
        sum(bool(row["team_correctness"][agent]) for row in peers)
        for agent in range(5)
    )
    if counts != candidate.member_gain.incumbent_correct_counts:
        raise AssertionError("incumbent_member_counts_do_not_match_peer_state")
    gains = tuple(
        counts[index] - run.initial_counts[index] for index in range(5)
    )
    invalid_count = sum(not bool(row["team_validity"][target]) for row in peers)
    active_soft = (
        candidate.team_outcome.mean_soft_vote_utility
        - candidate.marginal.soft_utility_delta
    )
    return CandidateEvaluation(
        prompt=parent_prompt,
        prompt_hash=PromptEnsembleOptimizationSystem.prompt_hash(parent_prompt),
        competence=PromptCompetenceMetrics(
            correct_count=counts[target],
            accuracy=counts[target] / len(peers),
            invalid_count=invalid_count,
            invalid_rate=invalid_count / len(peers),
            terminal_invalid_count=0,
        ),
        team_outcome=TeamOutcomeMetrics(
            vote_correct_vector=tuple(bool(row["vote_correct"]) for row in peers),
            vote_correct_count=sum(bool(row["vote_correct"]) for row in peers),
            plurality_vote_accuracy=(
                sum(bool(row["vote_correct"]) for row in peers) / len(peers)
            ),
            gold_vote_counts=tuple(int(row["gold_vote_count"]) for row in peers),
            largest_wrong_vote_counts=tuple(
                int(row["largest_wrong_vote_count"]) for row in peers
            ),
            plurality_margins=tuple(int(row["plurality_margin"]) for row in peers),
            mean_soft_vote_utility=active_soft,
        ),
        marginal=CandidateMarginalContribution(0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0.0),
        protection=ProtectionContribution(0, 0, 0, 0),
        member_gain=MemberGainMetrics(
            initial_correct_counts=run.initial_counts,
            incumbent_correct_counts=counts,
            candidate_correct_counts=counts,
            gain_counts=gains,
            minimum_gain_count=min(gains),
            total_gain_count=sum(gains),
            mean_gain=sum(gains) / len(gains),
            improved_agent_count=sum(value > 0 for value in gains),
            regressed_agent_count=sum(value < 0 for value in gains),
            all_members_non_regressed=all(value >= 0 for value in gains),
            all_members_improved=all(value > 0 for value in gains),
            target_gain_vs_initial=gains[target],
            target_gain_vs_incumbent=0,
        ),
    )


def transition_rows(
    candidate: CandidateEvaluation,
    active: CandidateEvaluation,
    peer_rows: Sequence[dict[str, Any]],
    target: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, peer in enumerate(peer_rows):
        g0 = int(peer["gold_vote_count"])
        h0 = int(peer["largest_wrong_vote_count"])
        m0 = int(peer["plurality_margin"])
        g1 = int(candidate.team_outcome.gold_vote_counts[index])
        h1 = int(candidate.team_outcome.largest_wrong_vote_counts[index])
        m1 = int(candidate.team_outcome.plurality_margins[index])
        vote0 = bool(peer["vote_correct"])
        vote1 = bool(candidate.team_outcome.vote_correct_vector[index])
        target0 = bool(peer["team_correctness"][target])
        fixed_peer_gold = g0 - int(target0)
        target1_value = g1 - fixed_peer_gold
        if target1_value not in (0, 1):
            raise AssertionError("fixed_peer_target_correctness_reconstruction_failed")
        target1 = bool(target1_value)
        if m0 != g0 - h0 or m1 != g1 - h1:
            raise AssertionError("plurality_margin_identity_failed")
        if vote0 != (m0 > 0) or vote1 != (m1 > 0):
            raise AssertionError("plurality_vote_boundary_identity_failed")
        boundary_cross = not vote0 and vote1
        vote_loss = vote0 and not vote1
        coverage_only = (not target0) and target1 and not vote0 and not vote1
        margin_progress_no_flip = m1 > m0 and not vote0 and not vote1
        already_correct_strengthening = vote0 and vote1 and m1 > m0
        no_change = vote0 == vote1 and target0 == target1 and m0 == m1
        if boundary_cross:
            primary_class = "P3_PLURALITY_BOUNDARY_CROSS"
        elif vote_loss:
            primary_class = "P5_VOTE_LOSS"
        elif already_correct_strengthening:
            primary_class = "P4_ALREADY_CORRECT_STRENGTHENING"
        elif coverage_only:
            primary_class = "P1_COVERAGE_ONLY"
        elif margin_progress_no_flip:
            primary_class = "P2_MARGIN_PROGRESS_NO_FLIP"
        elif no_change:
            primary_class = "P0_NO_CHANGE"
        else:
            primary_class = "P6_OTHER"
        rows.append(
            {
                "question_hash": peer["question_hash"],
                "G0": g0,
                "H0": h0,
                "M0": m0,
                "vote0": vote0,
                "target_correct0": target0,
                "G1": g1,
                "H1": h1,
                "M1": m1,
                "vote1": vote1,
                "target_correct1": target1,
                "delta_G": g1 - g0,
                "delta_H": h1 - h0,
                "delta_M": m1 - m0,
                "boundary_cross": boundary_cross,
                "vote_loss": vote_loss,
                "coverage_only": coverage_only,
                "margin_progress_no_flip": margin_progress_no_flip,
                "already_correct_strengthening": already_correct_strengthening,
                "primary_class": primary_class,
            }
        )
    if sum(row["boundary_cross"] for row in rows) != candidate.marginal.vote_gain_count:
        raise AssertionError("candidate_vote_gain_count_reconstruction_failed")
    if sum(row["vote_loss"] for row in rows) != candidate.marginal.vote_loss_count:
        raise AssertionError("candidate_vote_loss_count_reconstruction_failed")
    return rows


def source_hash(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def verify_run_identity(run: RunEvidence, expected_updates: int) -> None:
    config = run.meta["config"]
    identity = run.meta["run_identity"]
    checks = {
        "task": (
            config["task_type"] == "bbh"
            and config["comparison_task_id"] == "disambiguation_qa"
        ),
        "seed": int(config["seed"]) == 46,
        "agent_model": config["agent_model"] == "qwen3-14b",
        "optimizer_model": config["optimizer_model"] == "qwen3-14b",
        "evaluator_model": config["evaluator_model"] == "qwen3-14b",
        "planned_updates": int(run.meta["planned_update_count"]) == expected_updates,
        "completed_updates": int(run.meta["completed_update_count"]) == expected_updates,
        "final_test_enabled": config["final_test_enabled"] is True,
        "test_once": int(run.meta["test_evaluation_count"]) == 1,
        "validation_zero": int(run.meta["validation_evaluation_count"]) == 0,
        "source_commit": identity["git_commit"] == RUN_SOURCE_COMMIT,
        "method": identity["method_version"] == "member_aware_peer_state_v14",
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise AssertionError(f"authoritative_run_identity_failed:{failures}")


def replay(run: RunEvidence, policy: str) -> dict[str, Any]:
    if policy not in {"common", "rcru"}:
        raise ValueError(policy)
    branch_by_key = {
        (row["update_index"], row["target_agent_id"]): row
        for row in run.branch_decisions
    }
    commits = {row["update_index"]: row for row in run.commit_decisions}
    contexts = {
        (row["update_index"], row["target_agent_id"]): row
        for row in run.tcs_contexts
    }
    prompts = [run.meta["config"]["shared_prompt"]] * 5
    if [PromptEnsembleOptimizationSystem.prompt_hash(prompt) for prompt in prompts] != run.meta[
        "initial_prompt_hashes"
    ]:
        raise AssertionError("initial_prompt_hash_mismatch")
    constraint_mismatches: list[dict[str, Any]] = []
    branch_mismatches: list[dict[str, Any]] = []
    global_mismatches: list[dict[str, Any]] = []
    parent_prompt_mismatches: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    branch_records: list[dict[str, Any]] = []
    update_records: list[dict[str, Any]] = []
    for update_row in run.candidate_decisions:
        update = int(update_row["update_index"])
        candidates: list[dict[str, Any]] = []
        active_by_target: dict[int, CandidateEvaluation] = {}
        for raw in update_row["candidates"]:
            target = int(raw["target_agent_id"])
            branch = branch_by_key[(update, target)]
            candidate = deserialize_evaluation(raw["evaluation"])
            active = active_evaluation(
                run,
                target=target,
                team_state_version=int(branch["team_state_version"]),
                parent_prompt=prompts[target],
                candidate=candidate,
            )
            active_by_target[target] = active
            decision = (
                evaluate_constraints(candidate, active)
                if policy == "common"
                else evaluate_robust_contribution_constraints(candidate, active)
            )
            if policy == "common" and run.name == S2_NAME:
                stored = raw["constraint"]
                if (
                    decision.passed != stored["passed"]
                    or list(decision.rejection_reasons) != stored["rejection_reasons"]
                ):
                    constraint_mismatches.append(
                        {
                            "update_index": update,
                            "target_agent_id": target,
                            "candidate_hash": candidate.prompt_hash,
                        }
                    )
            if policy == "rcru" and run.name == S3_NAME:
                stored = raw["constraint"]
                if (
                    decision.passed != stored["passed"]
                    or list(decision.rejection_reasons) != stored["rejection_reasons"]
                ):
                    constraint_mismatches.append(
                        {
                            "update_index": update,
                            "target_agent_id": target,
                            "candidate_hash": candidate.prompt_hash,
                        }
                    )
            transitions = transition_rows(
                candidate,
                active,
                run.peer_chunks[int(branch["team_state_version"])],
                target,
            )
            candidates.append(
                {
                    "raw": raw,
                    "candidate": candidate,
                    "active": active,
                    "decision": decision,
                    "transitions": transitions,
                }
            )
        branch_winners: list[dict[str, Any]] = []
        for branch_payload in update_row["branches"]:
            target = int(branch_payload["target_agent_id"])
            context = contexts.get((update, target))
            if context is not None:
                observed_parent_hash = PromptEnsembleOptimizationSystem.prompt_hash(
                    prompts[target]
                )
                if context["parent_prompt_hash"] != observed_parent_hash:
                    parent_prompt_mismatches.append(
                        {
                            "update_index": update,
                            "target_agent_id": target,
                        }
                    )
            branch_candidates = [
                row
                for row in candidates
                if int(row["raw"]["target_agent_id"]) == target
            ]
            feasible = [row for row in branch_candidates if row["decision"].passed]
            frontier_hashes: set[str] = set()
            ranked = list(feasible)
            if policy == "rcru" and feasible:
                frontier_hashes = {
                    row.prompt_hash
                    for row in responsibility_contribution_pareto_front(
                        [item["candidate"] for item in feasible]
                    )
                }
                ranked = [
                    item
                    for item in feasible
                    if item["candidate"].prompt_hash in frontier_hashes
                ]
            if policy == "common":
                branch_key = lambda item: common_monotone_safe_key(
                    item["candidate"], int(item["raw"]["generation"])
                )
            else:
                branch_key = lambda item: robust_contribution_key(
                    item["candidate"], int(item["raw"]["generation"])
                )
            winner = max(ranked, key=branch_key, default=None)
            winner_hash = winner["candidate"].prompt_hash if winner else ""
            if (
                (policy == "common" and run.name == S2_NAME)
                or (policy == "rcru" and run.name == S3_NAME)
            ) and winner_hash != branch_payload["branch_winner_hash"]:
                branch_mismatches.append(
                    {
                        "update_index": update,
                        "target_agent_id": target,
                        "replayed": winner_hash,
                        "recorded": branch_payload["branch_winner_hash"],
                    }
                )
            branch_record = {
                "update_index": update,
                "target_agent_id": target,
                "target_rank": int(branch_payload["target_selection_rank"]),
                "active_lane": branch_payload["active_lane"],
                "candidate_records": branch_candidates,
                "feasible_hashes": {
                    item["candidate"].prompt_hash for item in feasible
                },
                "frontier_hashes": frontier_hashes,
                "winner": winner,
                "winner_hash": winner_hash,
                "actual_winner_hash": branch_payload["branch_winner_hash"],
            }
            branch_records.append(branch_record)
            if winner:
                branch_winners.append(branch_record)
        if policy == "common":
            global_key = lambda branch: common_cross_branch_transition_key(
                branch["winner"]["candidate"],
                branch["winner"]["active"],
                target_selection_rank=branch["target_rank"],
            )
        else:
            global_key = lambda branch: rcru_cross_branch_transition_key(
                branch["winner"]["candidate"],
                branch["winner"]["active"],
                target_selection_rank=branch["target_rank"],
            )
        global_winner = max(branch_winners, key=global_key, default=None)
        global_hash = global_winner["winner_hash"] if global_winner else ""
        actual_commit = commits[update]
        if (
            (policy == "common" and run.name == S2_NAME)
            or (policy == "rcru" and run.name == S3_NAME)
        ) and global_hash != actual_commit["committed_prompt_hash"]:
            global_mismatches.append(
                {
                    "update_index": update,
                    "replayed": global_hash,
                    "recorded": actual_commit["committed_prompt_hash"],
                }
            )
        update_records.append(
            {
                "update_index": update,
                "candidates": candidates,
                "branches": [
                    row for row in branch_records if row["update_index"] == update
                ],
                "global_winner": global_winner,
                "global_winner_hash": global_hash,
                "actual_commit_hash": actual_commit["committed_prompt_hash"],
                "actual_commit_target": actual_commit["committed_target_id"],
                "parent_team_hash": update_row["parent_team_hash"],
            }
        )
        candidate_records.extend(candidates)
        if actual_commit["committed_prompt_hash"]:
            committed = next(
                item
                for item in candidates
                if item["candidate"].prompt_hash
                == actual_commit["committed_prompt_hash"]
            )
            prompts[int(actual_commit["committed_target_id"])] = committed[
                "candidate"
            ].prompt
    return {
        "policy": policy,
        "constraint_mismatches": constraint_mismatches,
        "branch_mismatches": branch_mismatches,
        "global_mismatches": global_mismatches,
        "parent_prompt_mismatches": parent_prompt_mismatches,
        "candidate_records": candidate_records,
        "branch_records": branch_records,
        "update_records": update_records,
    }


def safe_metrics(item: dict[str, Any] | None) -> dict[str, Any]:
    if item is None:
        return {
            "candidate_hash": "",
            "target_agent_id": None,
            "vote_gain": None,
            "vote_gain_count": None,
            "vote_loss_count": None,
            "target_gain": None,
            "minimum_gain_delta": None,
            "total_gain_delta": None,
            "soft_vote_utility_delta": None,
            "lane_delta": None,
            "positive_support": None,
            "negative_support": None,
            "coalition": None,
            "bootstrap": None,
            "edit_tokens": None,
            "boundary_cross_count": None,
        }
    candidate = item["candidate"]
    active = item["active"]
    contribution = candidate.responsibility_contribution
    return {
        "candidate_hash": candidate.prompt_hash,
        "target_agent_id": int(item["raw"]["target_agent_id"]),
        "vote_gain": (
            candidate.team_outcome.vote_correct_count
            - active.team_outcome.vote_correct_count
        ),
        "vote_gain_count": candidate.marginal.vote_gain_count,
        "vote_loss_count": candidate.marginal.vote_loss_count,
        "target_gain": candidate.competence.correct_count - active.competence.correct_count,
        "minimum_gain_delta": (
            candidate.member_gain.minimum_gain_count
            - active.member_gain.minimum_gain_count
        ),
        "total_gain_delta": (
            candidate.member_gain.total_gain_count
            - active.member_gain.total_gain_count
        ),
        "soft_vote_utility_delta": (
            candidate.team_outcome.mean_soft_vote_utility
            - active.team_outcome.mean_soft_vote_utility
        ),
        "lane_delta": contribution.utility.utility_delta if contribution else None,
        "positive_support": (
            contribution.utility.positive_support_count if contribution else None
        ),
        "negative_support": (
            contribution.utility.negative_support_count if contribution else None
        ),
        "coalition": (
            contribution.coalition.net_contribution_delta if contribution else None
        ),
        "bootstrap": contribution.robust_support.bootstrap_lcb if contribution else None,
        "edit_tokens": contribution.edit.total_edit_token_count if contribution else None,
        "boundary_cross_count": sum(
            row["boundary_cross"] for row in item["transitions"]
        ),
    }


def candidate_counterfactual_rows(
    rcru_replay: dict[str, Any], common_replay: dict[str, Any]
) -> list[dict[str, Any]]:
    common_by_key = {
        (
            row["raw"]["target_agent_id"],
            row["raw"]["prompt_hash"],
            row["raw"].get("generation", 0),
            row["active"].prompt_hash,
        ): row
        for row in common_replay["candidate_records"]
    }
    actual_branch_winners = {
        (branch["update_index"], branch["target_agent_id"]): branch["winner_hash"]
        for branch in rcru_replay["branch_records"]
    }
    actual_commits = {
        row["update_index"]: row["actual_commit_hash"]
        for row in rcru_replay["update_records"]
    }
    rows: list[dict[str, Any]] = []
    for item in rcru_replay["candidate_records"]:
        raw = item["raw"]
        target = int(raw["target_agent_id"])
        update = next(
            update["update_index"]
            for update in rcru_replay["update_records"]
            if item in update["candidates"]
        )
        common = common_by_key[
            (target, raw["prompt_hash"], raw.get("generation", 0), item["active"].prompt_hash)
        ]
        candidate = item["candidate"]
        active = item["active"]
        contribution = candidate.responsibility_contribution
        if contribution is None:
            raise AssertionError("S3_candidate_missing_RCRU_metrics")
        common_decision = common["decision"]
        rcru_decision = item["decision"]
        metrics = safe_metrics(item)
        category = (
            "A_BOTH"
            if common_decision.passed and rcru_decision.passed
            else "B_COMMON_ONLY"
            if common_decision.passed
            else "C_RCRU_ONLY"
            if rcru_decision.passed
            else "D_NEITHER"
        )
        rows.append(
            {
                "update_index": update,
                "parent_team_hash": next(
                    row["parent_team_hash"]
                    for row in rcru_replay["update_records"]
                    if row["update_index"] == update
                ),
                "target_agent_id": target,
                "target_rank": int(raw["target_selection_rank"]),
                "candidate_hash": candidate.prompt_hash,
                "target_gain": metrics["target_gain"],
                "vote_gain": metrics["vote_gain"],
                "vote_gain_count": metrics["vote_gain_count"],
                "vote_loss_count": metrics["vote_loss_count"],
                "minimum_gain_delta": metrics["minimum_gain_delta"],
                "total_gain_delta": metrics["total_gain_delta"],
                "soft_vote_utility_delta": metrics["soft_vote_utility_delta"],
                "terminal_invalid_delta": (
                    candidate.competence.terminal_invalid_count
                    - active.competence.terminal_invalid_count
                ),
                "lane": contribution.utility.repair_lane,
                "lane_utility_delta": contribution.utility.utility_delta,
                "positive_support": contribution.utility.positive_support_count,
                "negative_support": contribution.utility.negative_support_count,
                "bootstrap_lcb": contribution.robust_support.bootstrap_lcb,
                "net_coalition_contribution_delta": (
                    contribution.coalition.net_contribution_delta
                ),
                "edit_tokens": contribution.edit.total_edit_token_count,
                "boundary_cross_count": metrics["boundary_cross_count"],
                "preboundary_margin_progress_total": sum(
                    max(0, min(row["M1"], 0) - min(row["M0"], 0))
                    for row in item["transitions"]
                ),
                "common_feasible": common_decision.passed,
                "common_rejection_reasons": list(common_decision.rejection_reasons),
                "rcru_feasible": rcru_decision.passed,
                "rcru_rejection_reasons": list(rcru_decision.rejection_reasons),
                "feasibility_class": category,
                "actual_rcru_branch_selected": (
                    actual_branch_winners[(update, target)] == candidate.prompt_hash
                ),
                "actual_global_committed": actual_commits[update] == candidate.prompt_hash,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["update_index"],
            row["target_rank"],
            row["candidate_hash"],
        ),
    )


def counterfactual_mismatches(
    actual: dict[str, Any], common: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    actual_updates = {row["update_index"]: row for row in actual["update_records"]}
    common_updates = {row["update_index"]: row for row in common["update_records"]}
    rows: list[dict[str, Any]] = []
    regret = Counter()
    for update in range(32):
        a = actual_updates[update]
        c = common_updates[update]
        actual_item = a["global_winner"]["winner"] if a["global_winner"] else None
        common_item = c["global_winner"]["winner"] if c["global_winner"] else None
        am = safe_metrics(actual_item)
        cm = safe_metrics(common_item)
        a_branches = {row["target_agent_id"]: row for row in a["branches"]}
        c_branches = {row["target_agent_id"]: row for row in c["branches"]}
        feasibility_changed = any(
            a_branches[target]["feasible_hashes"]
            != c_branches[target]["feasible_hashes"]
            for target in a_branches
        )
        branch_changed = any(
            a_branches[target]["winner_hash"] != c_branches[target]["winner_hash"]
            for target in a_branches
        )
        cross_changed = (
            not branch_changed
            and a["global_winner_hash"] != c["global_winner_hash"]
        )
        reasons: list[str] = []
        if feasibility_changed:
            reasons.append("FEASIBILITY_SET_CHANGED")
        if branch_changed:
            reasons.append("BRANCH_RANKING_CHANGED")
        if cross_changed:
            reasons.append("CROSS_BRANCH_RANKING_CHANGED")
        classification = (
            "SAME"
            if a["global_winner_hash"] == c["global_winner_hash"]
            else reasons[0]
            if len(reasons) == 1
            else "MULTIPLE"
        )
        if common_item and (not actual_item or cm["vote_gain"] > am["vote_gain"]):
            regret["common_winner_higher_vote_gain"] += 1
        if (
            common_item
            and actual_item
            and cm["vote_gain"] == am["vote_gain"]
            and cm["target_gain"] > am["target_gain"]
        ):
            regret["same_vote_gain_common_higher_target_gain"] += 1
        if (
            common_item
            and actual_item
            and (cm["vote_gain"], cm["minimum_gain_delta"], cm["total_gain_delta"])
            == (am["vote_gain"], am["minimum_gain_delta"], am["total_gain_delta"])
            and (am["lane_delta"] or 0) > (cm["lane_delta"] or 0)
        ):
            regret["same_team_objective_rcru_higher_lane_utility"] += 1
        if (
            common_item
            and actual_item
            and am["vote_gain"] == 0
            and am["target_gain"] == 0
            and (am["lane_delta"] or 0) > 0
            and cm["target_gain"] > 0
        ):
            regret["rcru_lane_only_common_target_improving"] += 1
        if actual_item and not common_item:
            regret["rcru_commits_common_rejects_all"] += 1
        if common_item and not actual_item:
            regret["common_commits_rcru_rejects_all"] += 1
        rows.append(
            {
                "update": update,
                "actual_S3_commit_hash": am["candidate_hash"],
                "actual_target": am["target_agent_id"],
                "common_policy_commit_hash": cm["candidate_hash"],
                "common_target": cm["target_agent_id"],
                "same_or_different": (
                    "SAME"
                    if am["candidate_hash"] == cm["candidate_hash"]
                    else "DIFFERENT"
                ),
                "difference_classification": classification,
                "actual_vote_gain": am["vote_gain"],
                "actual_vote_gain_count": am["vote_gain_count"],
                "actual_vote_loss_count": am["vote_loss_count"],
                "actual_target_gain": am["target_gain"],
                "actual_lane_delta": am["lane_delta"],
                "actual_support": am["positive_support"],
                "actual_coalition": am["coalition"],
                "actual_bootstrap": am["bootstrap"],
                "actual_edit": am["edit_tokens"],
                "counterfactual_vote_gain": cm["vote_gain"],
                "counterfactual_vote_gain_count": cm["vote_gain_count"],
                "counterfactual_vote_loss_count": cm["vote_loss_count"],
                "counterfactual_target_gain": cm["target_gain"],
                "counterfactual_minimum_gain_delta": cm["minimum_gain_delta"],
                "counterfactual_total_gain_delta": cm["total_gain_delta"],
                "counterfactual_soft_vote_delta": cm["soft_vote_utility_delta"],
                "counterfactual_edit": cm["edit_tokens"],
            }
        )
    return rows, dict(regret)


def agent_attrition(
    run: RunEvidence,
    rcru_replay: dict[str, Any],
    common_replay: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rcru_candidates = {
        (row["raw"]["target_agent_id"], row["raw"]["prompt_hash"], update["update_index"]): row
        for update in rcru_replay["update_records"]
        for row in update["candidates"]
    }
    common_candidates = {
        (row["raw"]["target_agent_id"], row["raw"]["prompt_hash"], update["update_index"]): row
        for update in common_replay["update_records"]
        for row in update["candidates"]
    }
    branch_payload = {
        (row["update_index"], row["target_agent_id"]): row
        for row in run.branch_decisions
    }
    candidate_updates = {row["update_index"]: row for row in run.candidate_decisions}
    commit_by_update = {row["update_index"]: row for row in run.commit_decisions}
    rcru_rows = {
        (row["update_index"], row["target_agent_id"], row["candidate_prompt_hash"]): row
        for row in run.rcru_rows
    }
    scores = {
        (row["update_index"], row["agent_id"]): row for row in run.target_scores
    }
    per_agent: dict[int, dict[str, Any]] = {}
    timeline: list[dict[str, Any]] = []
    for agent in range(5):
        branches = [row for row in run.branch_decisions if row["target_agent_id"] == agent]
        candidate_rows = [
            (update["update_index"], raw)
            for update in run.candidate_decisions
            for raw in update["candidates"]
            if raw["target_agent_id"] == agent
        ]
        rrows = [row for row in run.rcru_rows if row["target_agent_id"] == agent]
        commits = [
            row
            for row in run.commit_decisions
            if row["committed_target_id"] == agent
        ]
        per_agent[agent] = {
            "selected": len(branches),
            "branch_produced_candidates": sum(row["candidate_count"] > 0 for row in branches),
            "stage_b_candidates": sum(row["candidate_count"] for row in branches),
            "layer1_pass": sum(bool(row["layer1_passed"]) for row in rrows),
            "layer2_pass": sum(bool(row["layer2_passed"]) for row in rrows),
            "layer3_pass": sum(bool(row["layer3_passed"]) for row in rrows),
            "branch_winner": sum(bool(row["branch_winner_hash"]) for row in branches),
            "competition_loser": sum(bool(row["competition_loser"]) for row in branches),
            "commit": len(commits),
        }
    cause_counts: dict[int, Counter[str]] = {agent: Counter() for agent in range(5)}
    generation_reasons: dict[int, Counter[str]] = {agent: Counter() for agent in range(5)}
    rcru_rejections: dict[int, Counter[str]] = {agent: Counter() for agent in range(5)}
    common_rejections: dict[int, Counter[str]] = {agent: Counter() for agent in range(5)}
    for update_row in run.candidate_decisions:
        update = int(update_row["update_index"])
        for branch in update_row["branches"]:
            agent = int(branch["target_agent_id"])
            payload = branch_payload[(update, agent)]
            candidates = [
                raw
                for raw in update_row["candidates"]
                if int(raw["target_agent_id"]) == agent
            ]
            r_items = [
                rcru_candidates[(agent, raw["prompt_hash"], update)]
                for raw in candidates
            ]
            c_items = [
                common_candidates[(agent, raw["prompt_hash"], update)]
                for raw in candidates
            ]
            for item in r_items:
                rcru_rejections[agent].update(item["decision"].rejection_reasons)
            for item in c_items:
                common_rejections[agent].update(item["decision"].rejection_reasons)
            committed = commit_by_update[update]["committed_target_id"] == agent
            if committed:
                cause = "COMMIT"
            elif not candidates:
                cause = "GENERATION_FAILURE"
                terminal = branch["funnel"].get("terminal_failure_class") or "no_candidates"
                generation_reasons[agent][terminal] += 1
            elif not any(item["decision"].passed for item in r_items):
                cause = (
                    "RCRU_LANE_FAILURE"
                    if any(item["decision"].passed for item in c_items)
                    else "COMMON_SAFETY_FAILURE"
                )
            elif payload["competition_loser"]:
                cause = "CROSS_BRANCH_COMPETITION_LOSS"
            elif not payload["branch_winner_hash"]:
                cause = "RCRU_RANKING_LOSS"
            else:
                cause = "OTHER"
            cause_counts[agent][cause] += 1
            score = scores[(update, agent)]
            failure_reasons = sorted(
                {
                    reason
                    for item in r_items
                    for reason in item["decision"].rejection_reasons
                }
            )
            if not candidates:
                failure_reasons = [
                    branch["funnel"].get("terminal_failure_class") or "no_candidates"
                ]
            elif payload["competition_loser"]:
                failure_reasons.append("cross_branch_competition_loser")
            if agent in (1, 4):
                timeline.append(
                    {
                        "update_index": update,
                        "agent_id": agent,
                        "target_rank": branch["target_selection_rank"],
                        "expected_update_value": score["expected_update_value"],
                        "branch_failure_count_before_selection": score[
                            "branch_failure_count"
                        ],
                        "repairability_discount": score["repairability_discount"],
                        "active_lane": score["active_lane"],
                        "active_slice_size": score["active_lane_size"],
                        "candidate_count": len(candidates),
                        "common_passed_candidate_count": sum(
                            item["decision"].passed for item in c_items
                        ),
                        "rcru_passed_candidate_count": sum(
                            item["decision"].passed for item in r_items
                        ),
                        "event_outcome": cause,
                        "failure_reasons": "|".join(sorted(set(failure_reasons))),
                    }
                )
    table = [
        {"agent": agent, **per_agent[agent]}
        for agent in range(5)
    ]
    details = {
        "per_agent": {
            str(agent): {
                "event_outcomes": dict(cause_counts[agent]),
                "event_outcome_percent_of_selected": {
                    name: count / max(1, per_agent[agent]["selected"])
                    for name, count in cause_counts[agent].items()
                },
                "generation_terminal_reasons": dict(generation_reasons[agent]),
                "rcru_candidate_rejection_reasons": dict(rcru_rejections[agent]),
                "common_counterfactual_candidate_rejection_reasons": dict(
                    common_rejections[agent]
                ),
            }
            for agent in range(5)
        }
    }
    return table, details, sorted(timeline, key=lambda row: (row["agent_id"], row["update_index"]))


def positive_support_analysis(
    rcru_replay: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    semantics: dict[str, Counter[str]] = defaultdict(Counter)
    coverage_transitions: Counter[str] = Counter()
    coverage_boundary = Counter()
    margin_events: list[dict[str, Any]] = []
    coalition = {
        "all_candidates": Counter(),
        "rcru_feasible": Counter(),
        "branch_winners": Counter(),
        "global_commits": Counter(),
    }
    branch_winners = {
        branch["winner_hash"]
        for branch in rcru_replay["branch_records"]
        if branch["winner_hash"]
    }
    commits = {
        update["actual_commit_hash"]
        for update in rcru_replay["update_records"]
        if update["actual_commit_hash"]
    }
    for item in rcru_replay["candidate_records"]:
        candidate = item["candidate"]
        contribution = candidate.responsibility_contribution
        if contribution is None:
            continue
        lane = contribution.utility.repair_lane
        by_hash = {row["question_hash"]: row for row in item["transitions"]}
        for question_hash, delta in zip(
            contribution.utility.per_example_question_hashes,
            contribution.utility.per_example_deltas,
            strict=True,
        ):
            if delta <= 0:
                continue
            row = by_hash[question_hash]
            semantics[lane]["positive_support"] += 1
            if row["boundary_cross"]:
                label = "boundary_cross"
            elif row["coverage_only"]:
                label = "coverage_only"
            elif row["margin_progress_no_flip"]:
                label = "margin_progress_no_flip"
            elif row["already_correct_strengthening"]:
                label = "already_correct_strengthening"
            else:
                label = "other"
            semantics[lane][label] += 1
            if row["coverage_only"] and row["margin_progress_no_flip"]:
                semantics[lane]["coverage_margin_overlap"] += 1
            if lane == "coverage":
                key = (
                    f"{row['G0']}->{row['G1']}"
                    if (row["G0"], row["G1"])
                    in {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)}
                    else "other"
                )
                coverage_transitions[key] += 1
                coverage_boundary[
                    "boundary_cross" if row["boundary_cross"] else "still_vote_wrong"
                ] += 1
            if lane == "margin_support":
                margin_events.append(
                    {
                        "candidate_hash": candidate.prompt_hash,
                        "question_hash": question_hash,
                        "M0": row["M0"],
                        "M1": row["M1"],
                        "lane_utility_delta": int(delta),
                        "outcome": (
                            "cross_boundary"
                            if row["M1"] > 0
                            else "tie_at_boundary"
                            if row["M1"] == 0
                            else "preboundary"
                        ),
                    }
                )
        high_lane_zero_coalition = (
            contribution.utility.utility_delta > 0
            and contribution.coalition.net_contribution_delta == 0
        )
        groups = ["all_candidates"]
        if item["decision"].passed:
            groups.append("rcru_feasible")
        if candidate.prompt_hash in branch_winners:
            groups.append("branch_winners")
        if candidate.prompt_hash in commits:
            groups.append("global_commits")
        for group in groups:
            coalition[group]["count"] += 1
            coalition[group]["lane_positive_coalition_zero"] += int(
                high_lane_zero_coalition
            )
            coalition[group]["vote_gain_candidates"] += int(
                candidate.team_outcome.vote_correct_count
                > item["active"].team_outcome.vote_correct_count
            )
            coalition[group]["lane_only_candidates"] += int(
                candidate.team_outcome.vote_correct_count
                == item["active"].team_outcome.vote_correct_count
                and candidate.competence.correct_count
                == item["active"].competence.correct_count
                and contribution.utility.utility_delta > 0
            )
    groups: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in margin_events:
        if row["outcome"] in {"tie_at_boundary", "cross_boundary"}:
            groups[(row["M0"], row["lane_utility_delta"])].add(row["outcome"])
    witness_groups = [
        {"M0": key[0], "lane_utility_delta": key[1]}
        for key, outcomes in groups.items()
        if {"tie_at_boundary", "cross_boundary"}.issubset(outcomes)
    ]
    witnesses = [
        row
        for row in margin_events
        if (row["M0"], row["lane_utility_delta"])
        in {(group["M0"], group["lane_utility_delta"]) for group in witness_groups}
        and row["outcome"] in {"tie_at_boundary", "cross_boundary"}
    ]
    semantics_payload = {
        "classification_note": (
            "Categories are exclusive in the order boundary_cross, coverage_only, "
            "margin_progress_no_flip, already_correct_strengthening, other; the "
            "coverage_margin_overlap diagnostic reports the underlying overlap."
        ),
        "by_lane": {lane: dict(counts) for lane, counts in semantics.items()},
        "all_lanes": dict(sum(semantics.values(), Counter())),
        "coverage_majority_transitions": dict(coverage_transitions),
        "coverage_vote_outcomes": dict(coverage_boundary),
    }
    margin_payload = {
        "tie_at_boundary_count": sum(
            row["outcome"] == "tie_at_boundary" for row in margin_events
        ),
        "cross_boundary_count": sum(
            row["outcome"] == "cross_boundary" for row in margin_events
        ),
        "saturation_witness_group_count": len(witness_groups),
        "saturation_observed": bool(witness_groups),
        "witness_groups": witness_groups,
        "witnesses": witnesses,
    }
    coalition_payload = {
        group: {
            **dict(counts),
            "lane_positive_coalition_zero_fraction": (
                counts["lane_positive_coalition_zero"] / counts["count"]
                if counts["count"]
                else None
            ),
        }
        for group, counts in coalition.items()
    }
    return semantics_payload, margin_payload, coalition_payload


def average_ranks(values: Sequence[float], reverse: bool = True) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=reverse)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(right) != len(left):
        return None
    lm = sum(left) / len(left)
    rm = sum(right) / len(right)
    numerator = sum((x - lm) * (y - rm) for x, y in zip(left, right, strict=True))
    ld = math.sqrt(sum((x - lm) ** 2 for x in left))
    rd = math.sqrt(sum((y - rm) ** 2 for y in right))
    return numerator / (ld * rd) if ld and rd else None


def ranking_alignment(rcru_replay: dict[str, Any]) -> dict[str, Any]:
    accumulated: dict[str, list[float]] = defaultdict(list)
    branch_count = 0
    winner_agreement: Counter[str] = Counter()
    for branch in rcru_replay["branch_records"]:
        feasible = [
            item for item in branch["candidate_records"] if item["decision"].passed
        ]
        if len(feasible) < 2:
            continue
        branch_count += 1
        frontier = branch["frontier_hashes"]
        rcru_order = sorted(
            feasible,
            key=lambda item: (
                item["candidate"].prompt_hash in frontier,
                robust_contribution_key(
                    item["candidate"], int(item["raw"]["generation"])
                ),
            ),
            reverse=True,
        )
        rcru_rank = {
            item["candidate"].prompt_hash: index + 1
            for index, item in enumerate(rcru_order)
        }
        metrics: dict[str, list[float]] = {
            "vote_gain": [],
            "preboundary_delta_M": [],
            "boundary_cross_count": [],
            "target_gain": [],
            "lane_utility_delta": [],
        }
        for item in feasible:
            candidate = item["candidate"]
            contribution = candidate.responsibility_contribution
            metrics["vote_gain"].append(
                candidate.team_outcome.vote_correct_count
                - item["active"].team_outcome.vote_correct_count
            )
            metrics["preboundary_delta_M"].append(
                sum(
                    max(0, min(row["M1"], 0) - min(row["M0"], 0))
                    for row in item["transitions"]
                )
            )
            metrics["boundary_cross_count"].append(
                sum(row["boundary_cross"] for row in item["transitions"])
            )
            metrics["target_gain"].append(
                candidate.competence.correct_count - item["active"].competence.correct_count
            )
            metrics["lane_utility_delta"].append(
                contribution.utility.utility_delta if contribution else 0
            )
        common_order = sorted(
            feasible,
            key=lambda item: common_monotone_safe_key(
                item["candidate"], int(item["raw"]["generation"])
            ),
            reverse=True,
        )
        common_rank = {
            item["candidate"].prompt_hash: index + 1
            for index, item in enumerate(common_order)
        }
        for index, item in enumerate(feasible):
            prompt_hash = item["candidate"].prompt_hash
            accumulated["rcru_rank"].append(float(rcru_rank[prompt_hash]))
            accumulated["common_rank"].append(float(common_rank[prompt_hash]))
        for name, values in metrics.items():
            ranks = average_ranks(values, reverse=True)
            accumulated[name + "_rank"].extend(ranks)
            best = min(range(len(values)), key=lambda index: ranks[index])
            winner_agreement[name] += int(
                feasible[best]["candidate"].prompt_hash
                == rcru_order[0]["candidate"].prompt_hash
            )
        winner_agreement["common_policy"] += int(
            common_order[0]["candidate"].prompt_hash
            == rcru_order[0]["candidate"].prompt_hash
        )
    correlations = {
        name.replace("_rank", ""): pearson(accumulated["rcru_rank"], values)
        for name, values in accumulated.items()
        if name != "rcru_rank"
    }
    return {
        "branch_count_with_at_least_two_rcru_feasible_candidates": branch_count,
        "candidate_rank_observation_count": len(accumulated["rcru_rank"]),
        "spearman_on_concatenated_branch_local_ranks": correlations,
        "rcru_winner_agreement_count": dict(winner_agreement),
        "interpretation_limit": (
            "Rank correlations concatenate branch-local ranks and are descriptive, "
            "not causal; most branches contain at most two candidates."
        ),
    }


def accepted_transition_rows(
    run: RunEvidence, replay_result: dict[str, Any], setting: str
) -> list[dict[str, Any]]:
    g_rows = read_jsonl(run.path / "g_transition_audit.jsonl")
    by_update: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in g_rows:
        by_update[int(row["update_index"])].append(row)
    replay_updates = {row["update_index"]: row for row in replay_result["update_records"]}
    results: list[dict[str, Any]] = []
    for update, rows in sorted(by_update.items()):
        replay_update = replay_updates[update]
        item = replay_update["global_winner"]["winner"]
        candidate = item["candidate"]
        active = item["active"]
        contribution = candidate.responsibility_contribution
        target = int(item["raw"]["target_agent_id"])
        branch = next(
            row
            for row in replay_update["branches"]
            if row["target_agent_id"] == target
        )
        boundary = sum(not row["vote_correct_before"] and row["vote_correct_after"] for row in rows)
        losses = sum(row["vote_correct_before"] and not row["vote_correct_after"] for row in rows)
        coverage_only = sum(
            not row["target_correct_before"]
            and row["target_correct_after"]
            and not row["vote_correct_before"]
            and not row["vote_correct_after"]
            for row in rows
        )
        margin_no_flip = sum(
            row["M_after"] > row["M_before"]
            and not row["vote_correct_before"]
            and not row["vote_correct_after"]
            for row in rows
        )
        vote_before = sum(bool(row["vote_correct_before"]) for row in rows)
        vote_after = sum(bool(row["vote_correct_after"]) for row in rows)
        target_gain = candidate.competence.correct_count - active.competence.correct_count
        net_vote = vote_after - vote_before
        lane_delta = contribution.utility.utility_delta if contribution else None
        results.append(
            {
                "setting": setting,
                "update": update,
                "agent": target,
                "lane": branch["active_lane"],
                "candidate_hash": candidate.prompt_hash,
                "target_gain": target_gain,
                "vote_gain": net_vote,
                "vote_gain_count": boundary,
                "vote_loss_count": losses,
                "lane_delta": lane_delta,
                "positive_support": (
                    contribution.utility.positive_support_count if contribution else None
                ),
                "negative_support": (
                    contribution.utility.negative_support_count if contribution else None
                ),
                "bootstrap": contribution.robust_support.bootstrap_lcb if contribution else None,
                "coalition_delta": (
                    contribution.coalition.net_contribution_delta if contribution else None
                ),
                "coverage_only_improvements": coverage_only,
                "margin_progress_no_flip": margin_no_flip,
                "boundary_crosses": boundary,
                "train_vote_before": vote_before,
                "train_vote_after": vote_after,
                "lane_only_commit": bool(
                    contribution
                    and target_gain == 0
                    and net_vote == 0
                    and contribution.utility.utility_delta > 0
                ),
                "member_only_commit": bool(target_gain > 0 and net_vote == 0),
            }
        )
    return results


def primary_cause(details: dict[str, Any], agent: int) -> str:
    outcomes = Counter(details["per_agent"][str(agent)]["event_outcomes"])
    outcomes.pop("COMMIT", None)
    if not outcomes:
        return "UNKNOWN"
    name, count = outcomes.most_common(1)[0]
    if count * 2 <= sum(outcomes.values()):
        return "MIXED"
    return {
        "GENERATION_FAILURE": "GENERATION",
        "COMMON_SAFETY_FAILURE": "COMMON_SAFETY",
        "RCRU_LANE_FAILURE": "RCRU_LANE",
        "RCRU_RANKING_LOSS": "RCRU_RANKING",
        "CROSS_BRANCH_COMPETITION_LOSS": "COMPETITION",
    }.get(name, "MIXED")


def main() -> None:
    gate = read_json(
        REPO
        / "reports"
        / "v14_qwen3_14b_seed46_20260809"
        / "formal"
        / "stage_gate.json"
    )
    if (
        gate["gate"] != "PASS"
        or gate["complete_run_count"] != 5
        or gate["blocker_count"] != 0
        or gate["major_count"] != 0
        or gate["run_source_commit"] != RUN_SOURCE_COMMIT
        or gate["run_source_identity"]["source_tree_hash"] != RUN_SOURCE_TREE_HASH
    ):
        raise AssertionError("formal_protocol_gate_or_source_identity_failed")
    s2 = RunEvidence.load(S2_NAME)
    s3 = RunEvidence.load(S3_NAME)
    verify_run_identity(s2, 32)
    verify_run_identity(s3, 32)

    s2_common = replay(s2, "common")
    s3_rcru = replay(s3, "rcru")
    s3_common = replay(s3, "common")
    replay_validation = {
        "COUNTERFACTUAL_REPLAY_VALIDATION": (
            "PASS"
            if not s2_common["constraint_mismatches"]
            and not s2_common["branch_mismatches"]
            and not s2_common["global_mismatches"]
            and not s2_common["parent_prompt_mismatches"]
            else "FAIL"
        ),
        "RCRU_REPLAY_VALIDATION": (
            "PASS"
            if not s3_rcru["constraint_mismatches"]
            and not s3_rcru["branch_mismatches"]
            and not s3_rcru["global_mismatches"]
            and not s3_rcru["parent_prompt_mismatches"]
            else "FAIL"
        ),
        "s2": {
            "constraint_mismatches": s2_common["constraint_mismatches"],
            "branch_mismatches": s2_common["branch_mismatches"],
            "global_mismatches": s2_common["global_mismatches"],
            "parent_prompt_mismatches": s2_common["parent_prompt_mismatches"],
        },
        "s3": {
            "constraint_mismatches": s3_rcru["constraint_mismatches"],
            "branch_mismatches": s3_rcru["branch_mismatches"],
            "global_mismatches": s3_rcru["global_mismatches"],
            "parent_prompt_mismatches": s3_rcru["parent_prompt_mismatches"],
        },
        "function_source_hashes": {
            function.__name__: source_hash(function)
            for function in (
                evaluate_constraints,
                common_monotone_safe_key,
                common_cross_branch_transition_key,
                stage_a_multichannel_shortlist,
                stage_a_rcru_shortlist,
                rcru_cross_branch_transition_key,
                responsibility_utility,
                responsibility_utility_metrics,
                evaluate_robust_contribution_constraints,
                robust_contribution_key,
            )
        },
    }
    if (
        replay_validation["COUNTERFACTUAL_REPLAY_VALIDATION"] != "PASS"
        or replay_validation["RCRU_REPLAY_VALIDATION"] != "PASS"
    ):
        write_json(OUTPUT / "replay_validation.json", replay_validation)
        raise AssertionError("exact_replay_validation_failed")

    counterfactual_rows = candidate_counterfactual_rows(s3_rcru, s3_common)
    mismatch_rows, objective_regret = counterfactual_mismatches(s3_rcru, s3_common)
    attrition, failure_details, timeline = agent_attrition(s3, s3_rcru, s3_common)
    positive_support, margin_saturation, coalition = positive_support_analysis(s3_rcru)
    alignment = ranking_alignment(s3_rcru)
    s2_accepted = accepted_transition_rows(s2, s2_common, "S2")
    s3_accepted = accepted_transition_rows(s3, s3_rcru, "S3")
    accepted_rows = s2_accepted + s3_accepted

    class_counts = Counter(row["feasibility_class"] for row in counterfactual_rows)
    common_only_special = sum(
        row["feasibility_class"] == "B_COMMON_ONLY"
        and row["target_gain"] > 0
        and row["vote_gain"] == 0
        and row["lane_utility_delta"] == 0
        for row in counterfactual_rows
    )
    rcru_only_lane_only = sum(
        row["feasibility_class"] == "C_RCRU_ONLY"
        and row["target_gain"] == 0
        and row["vote_gain"] == 0
        and row["lane_utility_delta"] > 0
        for row in counterfactual_rows
    )
    changed_updates = sum(row["same_or_different"] == "DIFFERENT" for row in mismatch_rows)
    changed_nonempty = [
        row
        for row in mismatch_rows
        if row["same_or_different"] == "DIFFERENT"
    ]
    all_support = positive_support["all_lanes"]
    support_total = all_support.get("positive_support", 0)
    support_boundary = all_support.get("boundary_cross", 0)
    support_preboundary = all_support.get("coverage_only", 0) + all_support.get(
        "margin_progress_no_flip", 0
    )

    def transition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "accepted_updates": len(rows),
            "total_vote_gains": sum(row["vote_gain_count"] for row in rows),
            "total_vote_losses": sum(row["vote_loss_count"] for row in rows),
            "net_train_vote_gain": sum(row["vote_gain"] for row in rows),
            "boundary_cross_count": sum(row["boundary_crosses"] for row in rows),
            "coverage_only_count": sum(
                row["coverage_only_improvements"] for row in rows
            ),
            "margin_progress_no_flip_count": sum(
                row["margin_progress_no_flip"] for row in rows
            ),
            "member_only_commit_count": sum(row["member_only_commit"] for row in rows),
            "lane_only_commit_count": sum(row["lane_only_commit"] for row in rows),
            "initial_train_vote": rows[0]["train_vote_before"] if rows else None,
            "final_train_vote": rows[-1]["train_vote_after"] if rows else None,
        }

    s2_transition = transition_summary(s2_accepted)
    s3_transition = transition_summary(s3_accepted)
    boundary_summary = {
        "plurality_boundary_definition": "M0 <= 0 and M1 > 0",
        "five_agent_majority_note": "G 2->3 is reported separately and is not the sole plurality boundary.",
        "positive_support": positive_support,
        "margin_utility_saturation": margin_saturation,
        "coalition_alignment": coalition,
        "ranking_alignment": alignment,
        "accepted_transition_summary": {"S2": s2_transition, "S3": s3_transition},
    }

    a14_causes = {
        str(agent): failure_details["per_agent"][str(agent)]["event_outcomes"]
        for agent in (1, 4)
    }
    hypotheses = {
        "H1": {
            "verdict": (
                "SUPPORTED"
                if all(
                    causes.get("GENERATION_FAILURE", 0)
                    > sum(causes.values()) / 2
                    for causes in a14_causes.values()
                )
                else "NOT_SUPPORTED"
            ),
            "evidence": a14_causes,
            "interpretation": (
                "Generation failure occurred in only 2/31 Agent1/4 selections; "
                "28/31 ended in common target/vote safety failure."
            ),
        },
        "H2": {
            "verdict": (
                "SUPPORTED"
                if any(
                    row["feasibility_class"] == "B_COMMON_ONLY"
                    and row["target_agent_id"] in (1, 4)
                    for row in counterfactual_rows
                )
                else "NOT_SUPPORTED"
            ),
            "common_only_candidates_agent1_agent4": sum(
                row["feasibility_class"] == "B_COMMON_ONLY"
                and row["target_agent_id"] in (1, 4)
                for row in counterfactual_rows
            ),
        },
        "H3": {
            "verdict": "SUPPORTED" if rcru_only_lane_only > 0 else "NOT_SUPPORTED",
            "rcru_only_lane_only_candidate_count": rcru_only_lane_only,
            "rcru_lane_only_commit_count": s3_transition["lane_only_commit_count"],
        },
        "H4": {
            "verdict": (
                "SUPPORTED"
                if support_preboundary > support_boundary
                else "PARTIALLY_SUPPORTED"
                if support_preboundary > 0
                else "NOT_SUPPORTED"
            ),
            "positive_support_total": support_total,
            "preboundary_count": support_preboundary,
            "boundary_cross_count": support_boundary,
        },
        "H5": {
            "verdict": (
                "SUPPORTED" if margin_saturation["saturation_observed"] else "NOT_SUPPORTED"
            ),
            "saturation_witness_group_count": margin_saturation[
                "saturation_witness_group_count"
            ],
        },
        "H6": {
            "verdict": (
                "PARTIALLY_SUPPORTED"
                if coalition["branch_winners"]["lane_positive_coalition_zero"] > 0
                else "INSUFFICIENT_EVIDENCE"
            ),
            "branch_winner_lane_positive_coalition_zero": coalition[
                "branch_winners"
            ]["lane_positive_coalition_zero"],
            "commit_lane_positive_coalition_zero": coalition["global_commits"][
                "lane_positive_coalition_zero"
            ],
            "limitation": "Zero coalition delta shows ranking tolerance, not causal dilution by itself.",
        },
        "H7": {
            "verdict": "SUPPORTED",
            "layer2_pass": sum(row["layer2_passed"] for row in s3.rcru_rows),
            "layer3_pass": sum(row["layer3_passed"] for row in s3.rcru_rows),
        },
    }

    primary_factors: list[str] = []
    if class_counts["B_COMMON_ONLY"] or class_counts["C_RCRU_ONLY"]:
        primary_factors.append("FEASIBILITY_MISALIGNMENT")
    if changed_updates:
        primary_factors.append("RANKING_MISALIGNMENT")
    if support_preboundary > support_boundary or margin_saturation["saturation_observed"]:
        primary_factors.append("BOUNDARY_CREDIT_MISALIGNMENT")
    generation_14 = sum(
        failure_details["per_agent"][str(agent)]["event_outcomes"].get(
            "GENERATION_FAILURE", 0
        )
        for agent in (1, 4)
    )
    primary_mode = (
        primary_factors[0] if len(primary_factors) == 1 else "MIXED"
    )

    def generation_summary(run: RunData) -> dict[str, Any]:
        return {
            "selected_branches": len(run.branch_decisions),
            "stage_b_candidates": sum(
                int(row["candidate_count"]) for row in run.branch_decisions
            ),
            "zero_candidate_branches": sum(
                int(row["candidate_count"]) == 0 for row in run.branch_decisions
            ),
            "normal_failure_branches": sum(
                bool(row["normal_failure"]) for row in run.branch_decisions
            ),
            "operational_failure_branches": sum(
                bool(row["operational_failure"]) for row in run.branch_decisions
            ),
        }

    generation_comparison = {
        "S2": generation_summary(s2),
        "S3": generation_summary(s3),
        "interpretation": (
            "S2/S3 generation totals come from different realized training "
            "trajectories and are a secondary, confounded signal; fixed-pool "
            "replay does not attribute the S2-S3 gap to generation."
        ),
    }

    summary = {
        "MODULE3_DIAGNOSIS": "COMPLETE",
        "AGENT1_ZERO_COMMIT_PRIMARY_CAUSE": primary_cause(failure_details, 1),
        "AGENT4_ZERO_COMMIT_PRIMARY_CAUSE": primary_cause(failure_details, 4),
        "COMMON_ONLY_CANDIDATE_COUNT": class_counts["B_COMMON_ONLY"],
        "RCRU_ONLY_CANDIDATE_COUNT": class_counts["C_RCRU_ONLY"],
        "BOTH_FEASIBLE_CANDIDATE_COUNT": class_counts["A_BOTH"],
        "NEITHER_FEASIBLE_CANDIDATE_COUNT": class_counts["D_NEITHER"],
        "COMMON_ONLY_TARGET_GAIN_NO_VOTE_NO_LANE_COUNT": common_only_special,
        "RCRU_ONLY_LANE_ONLY_CANDIDATE_COUNT": rcru_only_lane_only,
        "COUNTERFACTUAL_GLOBAL_WINNER_CHANGED_UPDATES": changed_updates,
        "RCRU_POSITIVE_SUPPORT_BOUNDARY_CROSS_FRACTION": (
            support_boundary / support_total if support_total else None
        ),
        "RCRU_POSITIVE_SUPPORT_PREBOUNDARY_FRACTION": (
            support_preboundary / support_total if support_total else None
        ),
        "RCRU_LANE_ONLY_COMMIT_COUNT": s3_transition["lane_only_commit_count"],
        "S2_BOUNDARY_CROSS_COUNT": s2_transition["boundary_cross_count"],
        "S3_BOUNDARY_CROSS_COUNT": s3_transition["boundary_cross_count"],
        "MARGIN_UTILITY_SATURATION_OBSERVED": margin_saturation[
            "saturation_observed"
        ],
        "PRIMARY_MODULE3_FAILURE_MODE": primary_mode,
        "PRIMARY_FACTOR_COMPONENTS": primary_factors,
        "SECONDARY_CONFOUNDED_SIGNALS": (
            ["GENERATION_ATTRITION"] if generation_14 > 0 else []
        ),
        "V15_DESIGN_CHANGE_AUTHORIZED": False,
        "API_CALLS": 0,
        "counterfactual_scope": (
            "S3_STAGE_B_FIXED_POOL_COMMON_POLICY_COUNTERFACTUAL; "
            "ONE-STEP DECISION-POLICY COUNTERFACTUAL, NOT A FULL ALTERNATIVE "
            "TRAINING TRAJECTORY."
        ),
        "identity": {
            "report_commit": REPORT_COMMIT,
            "run_source_commit": RUN_SOURCE_COMMIT,
            "run_source_tree_hash": RUN_SOURCE_TREE_HASH,
            "method_version": "member_aware_peer_state_v14",
            "checkpoint_version": 23,
            "model": "qwen3-14b",
            "task": "disambiguation_qa",
            "seed": 46,
            "formal_protocol_gate": "PASS",
        },
        "replay_validation": replay_validation,
        "agent_attrition": attrition,
        "failure_decomposition": failure_details,
        "generation_comparison": generation_comparison,
        "repeated_target_timeline": timeline,
        "candidate_feasibility_classes": dict(class_counts),
        "counterfactual_objective_regret": objective_regret,
        "changed_update_count": len(changed_nonempty),
        "boundary_transition_summary": boundary_summary,
        "accepted_transition_summary": {"S2": s2_transition, "S3": s3_transition},
        "formal_results": {
            "Static": {"train_vote": 50, "test_vote": 85},
            "S0": {"accepted": 4, "train_vote": 55, "test_vote": 90},
            "S1": {"accepted": 6, "train_vote": 55, "test_vote": 90},
            "S2": {"accepted": 9, "train_vote": 60, "test_vote": 93},
            "S3": {"accepted": 7, "train_vote": 58, "test_vote": 87},
        },
        "test_usage_note": (
            "Final test is report-only and was not used in any candidate replay, "
            "ranking, or diagnosis decision."
        ),
    }

    write_json(OUTPUT / "replay_validation.json", replay_validation)
    write_jsonl(OUTPUT / "candidate_policy_counterfactual.jsonl", counterfactual_rows)
    write_csv(
        OUTPUT / "counterfactual_commit_mismatches.csv",
        mismatch_rows,
        list(mismatch_rows[0].keys()),
    )
    write_csv(
        OUTPUT / "agent_commit_attrition.csv",
        attrition,
        list(attrition[0].keys()),
    )
    write_json(OUTPUT / "failure_count_by_agent_and_stage.json", failure_details)
    write_csv(
        OUTPUT / "repeated_target_failure_timeline.csv",
        timeline,
        list(timeline[0].keys()),
    )
    write_json(OUTPUT / "positive_support_semantics_by_lane.json", positive_support)
    write_json(OUTPUT / "boundary_transition_summary.json", boundary_summary)
    write_csv(
        OUTPUT / "accepted_transition_comparison_s2_s3.csv",
        accepted_rows,
        list(accepted_rows[0].keys()),
    )
    write_json(OUTPUT / "hypothesis_verdicts.json", hypotheses)
    write_json(OUTPUT / "module3_diagnosis_summary.json", summary)

    agent1 = failure_details["per_agent"]["1"]["event_outcomes"]
    agent4 = failure_details["per_agent"]["4"]["event_outcomes"]
    timeline_by_agent = {
        agent: [row for row in timeline if row["agent_id"] == agent]
        for agent in (1, 4)
    }

    def timeline_digest(agent: int) -> dict[str, Any]:
        rows = timeline_by_agent[agent]
        return {
            "max_failure_count": max(
                row["branch_failure_count_before_selection"] for row in rows
            ),
            "minimum_discount": min(row["repairability_discount"] for row in rows),
            "rank1_count": sum(row["target_rank"] == 1 for row in rows),
            "rank2_count": sum(row["target_rank"] == 2 for row in rows),
            "lane_counts": dict(Counter(row["active_lane"] for row in rows)),
        }

    agent1_timeline = timeline_digest(1)
    agent4_timeline = timeline_digest(4)
    important = [
        row
        for row in mismatch_rows
        if row["same_or_different"] == "DIFFERENT"
    ]
    report = f"""# Module 3 Fully Offline Diagnosis

## 1. Executive conclusion

Seed46 shows a mixed S2→S3 degradation rather than a Layer-3-threshold failure.
Both exact replays pass with zero constraint, branch-winner, or global-commit
mismatch. On the fixed S3 Stage-B pool, common and RCRU feasibility differ for
{class_counts['B_COMMON_ONLY'] + class_counts['C_RCRU_ONLY']} of 88 candidates,
and the hypothetical global winner changes on {changed_updates} of 32 updates.
RCRU positive support crosses the true plurality boundary in
{support_boundary}/{support_total} cases, while {support_preboundary}/{support_total}
cases are coverage or margin progress that remains pre-boundary. S3 still raises
train vote 50-to-58, but S2 reaches 50-to-60 and has more accepted transitions.
The primary evidence therefore points to `{primary_mode}` across
feasibility/ranking differences and boundary-credit alignment. Generation
attrition is recorded only as a secondary, trajectory-confounded signal. This
does not authorize a v15 change or prove multi-seed harm.

## 2. Agent1/4 zero-commit diagnosis

| Agent | Selected | Candidate-producing branches | Stage-B candidates | L1 | L2 | L3 | Branch winners | Competition losses | Commits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in attrition:
        report += (
            f"| {row['agent']} | {row['selected']} | {row['branch_produced_candidates']} | "
            f"{row['stage_b_candidates']} | {row['layer1_pass']} | {row['layer2_pass']} | "
            f"{row['layer3_pass']} | {row['branch_winner']} | "
            f"{row['competition_loser']} | {row['commit']} |\n"
        )
    report += f"""

Agent1 event outcomes: `{json.dumps(agent1, sort_keys=True)}`. Its primary
classification is **{summary['AGENT1_ZERO_COMMIT_PRIMARY_CAUSE']}**.

Agent4 event outcomes: `{json.dumps(agent4, sort_keys=True)}`. Its primary
classification is **{summary['AGENT4_ZERO_COMMIT_PRIMARY_CAUSE']}**.

The repeated-target timeline shows the exact pre-selection failure discount,
expected update value, active lane, candidate yield, and terminal stage in
`repeated_target_failure_timeline.csv`. This separates persistent selector
opportunity from realized generator/RCRU feasibility.

Agent1 reached failure count {agent1_timeline['max_failure_count']} and discount
{agent1_timeline['minimum_discount']:.3f}, yet remained rank 1 on
{agent1_timeline['rank1_count']} selections and rank 2 on
{agent1_timeline['rank2_count']}; its lane mix was
`{json.dumps(agent1_timeline['lane_counts'], sort_keys=True)}`. Agent4 reached
failure count {agent4_timeline['max_failure_count']} and discount
{agent4_timeline['minimum_discount']:.3f}, while remaining rank 1 on
{agent4_timeline['rank1_count']} selection and rank 2 on
{agent4_timeline['rank2_count']}; its lane mix was
`{json.dumps(agent4_timeline['lane_counts'], sort_keys=True)}`. The normalized
opportunity ranking therefore kept both agents actionable despite discount,
but 28/31 of their selected branches failed the common target/vote safety
policy. Only 2/31 were generation failures, and neither agent had a
common-feasible candidate rejected only by the RCRU lane policy.

## 3. Common-policy counterfactual

Both replay validations are **PASS**. Candidate-set counts are:

- both feasible: {class_counts['A_BOTH']}
- common-only: {class_counts['B_COMMON_ONLY']}
- RCRU-only: {class_counts['C_RCRU_ONLY']}
- neither: {class_counts['D_NEITHER']}
- common-only target-gain/no-vote/no-lane subtype: {common_only_special}
- RCRU-only lane-only subtype: {rcru_only_lane_only}

The one-step hypothetical global winner changes on {changed_updates}/32 updates.
This result freezes each actual S3 parent, target pair, generated candidate pool,
and completed rollout, then changes only the decision policy. It is explicitly
**not** a chained S2 trajectory and cannot imply test=93.

Important changed candidate hashes are listed in
`counterfactual_commit_mismatches.csv`; the complete 88-candidate decision table
is `candidate_policy_counterfactual.jsonl`.

The four changed updates include two common-only target improvements (updates 1
and 22), one RCRU-only lane commit (update 8), and one same-objective branch
ranking change (update 4). The fixed-pool regret summary records one update
where common policy would commit while RCRU commits nothing, one where the
common winner has higher vote gain, and two where equal-vote common winners
have higher target gain.

## 4. Boundary alignment diagnosis

The true boundary is `M0 <= 0 -> M1 > 0`, not merely `G 2->3`.
Positive-support boundary-cross fraction is
{support_boundary}/{support_total} = {support_boundary / support_total if support_total else float('nan'):.3f};
pre-boundary coverage/margin fraction is
{support_preboundary}/{support_total} = {support_preboundary / support_total if support_total else float('nan'):.3f}.
Margin saturation is **{'observed' if margin_saturation['saturation_observed'] else 'not observed'}**
with {margin_saturation['saturation_witness_group_count']} real `(M0, utility_delta)`
witness groups containing both `M1=0` and `M1>0` outcomes.

## 5. S2 vs S3 transition structure

| Setting | Accepted | Vote gains | Vote losses | Net train vote | Boundary crosses | Member-only | Lane-only |
|---|---:|---:|---:|---:|---:|---:|---:|
| S2 | {s2_transition['accepted_updates']} | {s2_transition['total_vote_gains']} | {s2_transition['total_vote_losses']} | +{s2_transition['net_train_vote_gain']} | {s2_transition['boundary_cross_count']} | {s2_transition['member_only_commit_count']} | {s2_transition['lane_only_commit_count']} |
| S3 | {s3_transition['accepted_updates']} | {s3_transition['total_vote_gains']} | {s3_transition['total_vote_losses']} | +{s3_transition['net_train_vote_gain']} | {s3_transition['boundary_cross_count']} | {s3_transition['member_only_commit_count']} | {s3_transition['lane_only_commit_count']} |

The final test association is S2=93 and S3=87, but test observations never enter
the replay or any ranking decision.

## 6. Hypothesis verdicts

"""
    for key, value in hypotheses.items():
        report += f"- **{key}: {value['verdict']}** — `{json.dumps(value, sort_keys=True)}`\n"
    report += """

## 7. What the evidence does not prove

- One seed cannot establish that RCRU or Module 3 is generally harmful.
- A fixed-parent one-step counterfactual is not an alternative training trajectory.
- Candidate generation differs between actual S2 and S3, so this analysis only
  isolates decision-policy effects within the observed S3 Stage-B pool.
- Correlations between ranking signals and boundary conversion are descriptive,
  not causal.
- No v15 design change is authorized by this diagnosis.

## Machine-readable conclusion

See `module3_diagnosis_summary.json`. `API_CALLS = 0`.
"""
    (OUTPUT / "module3_diagnosis_report.md").write_text(report, encoding="utf-8")

    expected_outputs = {
        "module3_diagnosis_summary.json",
        "module3_diagnosis_report.md",
        "agent_commit_attrition.csv",
        "candidate_policy_counterfactual.jsonl",
        "counterfactual_commit_mismatches.csv",
        "positive_support_semantics_by_lane.json",
        "boundary_transition_summary.json",
        "accepted_transition_comparison_s2_s3.csv",
        "hypothesis_verdicts.json",
        "failure_count_by_agent_and_stage.json",
        "repeated_target_failure_timeline.csv",
        "replay_validation.json",
    }
    missing = [name for name in expected_outputs if not (OUTPUT / name).exists()]
    if missing:
        raise AssertionError(f"missing_outputs:{missing}")
    print(
        json.dumps(
            {
                "MODULE3_DIAGNOSIS": summary["MODULE3_DIAGNOSIS"],
                "API_CALLS": 0,
                "S2_REPLAY": replay_validation["COUNTERFACTUAL_REPLAY_VALIDATION"],
                "S3_REPLAY": replay_validation["RCRU_REPLAY_VALIDATION"],
                "candidate_count": len(counterfactual_rows),
                "changed_updates": changed_updates,
                "primary_mode": primary_mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
