"""Pure read-only analysis for accepted local-mutation team-transfer replay."""

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

from ..candidate_selection import CandidateEvaluation, ConstraintDecision, evaluate_constraints
from ..evaluation.fixed_probe import ProbeExample
from ..evaluation.prompt_question import PromptAnswer
from .progressive_evaluation import has_promotion_signal, is_catastrophic
from .schemas import TeamMiniBatchMetrics


def minibatch_pass(metrics: TeamMiniBatchMetrics) -> bool:
    return not is_catastrophic(metrics) and has_promotion_signal(metrics)


def full_positive(evaluation: CandidateEvaluation, active: CandidateEvaluation) -> bool:
    return (
        evaluation.team_outcome.vote_correct_count > active.team_outcome.vote_correct_count
        or evaluation.competence.correct_count > active.competence.correct_count
    )


def common_safe(evaluation: CandidateEvaluation, active: CandidateEvaluation) -> ConstraintDecision:
    return evaluate_constraints(evaluation, active)


def transfer_decomposition(
    *,
    examples: Sequence[ProbeExample],
    active_profiles: Sequence[Sequence[PromptAnswer]],
    candidate_profile: Sequence[PromptAnswer],
    target_member: int,
    assigned_example_ids: set[str],
    match_answer: Callable[[str, str], bool],
    active: CandidateEvaluation,
    candidate: CandidateEvaluation,
) -> dict[str, object]:
    if len(active_profiles) != 5 or len(candidate_profile) != len(examples):
        raise ValueError("complete five-member fixed-probe profiles required")
    new_team = [
        row.question_hash for row, before, after in zip(
            examples, active.team_outcome.vote_correct_vector,
            candidate.team_outcome.vote_correct_vector, strict=True,
        ) if not before and after
    ]
    lost_team = [
        row.question_hash for row, before, after in zip(
            examples, active.team_outcome.vote_correct_vector,
            candidate.team_outcome.vote_correct_vector, strict=True,
        ) if before and not after
    ]
    collateral = []
    for index, row in enumerate(examples):
        incumbent = active_profiles[target_member][index]
        proposed = candidate_profile[index]
        if (
            row.question_hash not in assigned_example_ids
            and incumbent.valid
            and match_answer(incumbent.answer, row.gold_answer)
            and not (
                proposed.valid and match_answer(proposed.answer, row.gold_answer)
            )
        ):
            collateral.append(row.question_hash)
    return {
        "target_member_full_delta": (
            candidate.competence.correct_count - active.competence.correct_count
        ),
        "team_vote_delta": candidate.marginal.net_vote_delta,
        "oracle_coverage_delta": (
            candidate.marginal.coverage_gain_count - candidate.marginal.coverage_loss_count
        ),
        "team_correct_votes_gain": candidate.marginal.vote_gain_count,
        "team_correct_votes_loss": candidate.marginal.vote_loss_count,
        "new_team_correct_cases": new_team,
        "lost_team_correct_cases": lost_team,
        "pivotal_flip_gain": candidate.protection.pivotal_correct_gain_count,
        "pivotal_flip_loss": candidate.protection.pivotal_correct_loss_count,
        "unique_correct_gain": candidate.protection.unique_correct_gain_count,
        "unique_correct_loss": candidate.protection.unique_correct_loss_count,
        "collateral_loss_count": len(collateral),
        "collateral_loss_cases": collateral,
        "non_target_member_behavior_delta": 0,
    }


def classify_transfer(rows: Sequence[dict[str, object]]) -> str:
    if len(rows) != 5 or any(row.get("execution_status") != "COMPLETE" for row in rows):
        return "TEAM_TRANSFER_REPLAY_NOT_EVALUABLE"
    full_gains = sum(
        row.get("full_status") == "PASS"
        and int(row.get("team_vote_delta", 0)) > 0
        for row in rows
    )
    if full_gains >= 2:
        return "LOCAL_IMPROVEMENT_CAN_TRANSFER_TO_TEAM_GAIN"
    if full_gains == 1:
        return "SINGLE_TEAM_GAIN_OBSERVED_REPLICATION_NEEDED"
    if any(row.get("team_minibatch_status") == "PASS" for row in rows):
        return "NO_FULL_TEAM_GAIN_WITH_SOME_MINIBATCH_SIGNAL"
    return "LOCAL_TEAM_OBJECTIVE_MISALIGNMENT_SIGNAL"


def full_team_result(team_vote_delta: int) -> str:
    if team_vote_delta > 0:
        return "TEAM_POSITIVE"
    if team_vote_delta < 0:
        return "TEAM_NEGATIVE"
    return "TEAM_EQUAL"


def summarize_mandatory_full_transfer(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Describe all five mandatory Full replays without an IID interpretation."""
    if (
        len(rows) != 5
        or any(row.get("execution_status") != "COMPLETE" for row in rows)
        or any(row.get("full_status") != "PASS" for row in rows)
    ):
        return {
            "evaluable": False,
            "primary_interpretation": "TEAM_TRANSFER_REPLAY_NOT_EVALUABLE",
            "interpretation_states": ["TEAM_TRANSFER_REPLAY_NOT_EVALUABLE"],
        }
    results = [full_team_result(int(row["team_vote_delta"])) for row in rows]
    counts = {name: results.count(name) for name in (
        "TEAM_POSITIVE", "TEAM_EQUAL", "TEAM_NEGATIVE",
    )}
    states: list[str] = []
    if counts["TEAM_POSITIVE"] >= 2:
        states.append("LOCAL_IMPROVEMENT_CAN_TRANSFER_TO_TEAM_GAIN_WITHIN_THIS_STATE")
    if counts["TEAM_EQUAL"] >= 3:
        states.append("LOCAL_IMPROVEMENT_OFTEN_FAILS_TO_CHANGE_TEAM_OUTCOME")
    if counts["TEAM_NEGATIVE"] >= 2:
        states.append("LOCAL_TEAM_OBJECTIVE_MISALIGNMENT_OBSERVED")
    if sum(value > 0 for value in counts.values()) >= 2:
        states.append("MEMBER_SPECIFIC_TRANSFER_HETEROGENEITY")
    if not states:
        # This is reachable only for a homogeneous five-case outcome not covered
        # by the directional labels above (for example five TEAM_POSITIVE rows).
        states.append("LOCAL_IMPROVEMENT_CAN_TRANSFER_TO_TEAM_GAIN_WITHIN_THIS_STATE")
    return {
        "evaluable": True,
        "single_baseline_state": True,
        "iid_inference": False,
        "team_positive_count": counts["TEAM_POSITIVE"],
        "team_equal_count": counts["TEAM_EQUAL"],
        "team_negative_count": counts["TEAM_NEGATIVE"],
        "team_positive_denominator": 5,
        "primary_interpretation": states[0],
        "interpretation_states": states,
    }


def minibatch_full_gate_audit(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    """Raw 2x2 counts for the simulated MiniBatch gate against mandatory Full."""
    if len(rows) != 5 or any(row.get("full_status") != "PASS" for row in rows):
        raise ValueError("gate audit requires five mandatory Full results")
    counts = {
        "correct_promotion": 0,
        "false_negative_filtering": 0,
        "false_positive_promotion": 0,
        "correct_filtering": 0,
    }
    for row in rows:
        passed = row.get("team_minibatch_status") == "PASS"
        positive = int(row["team_vote_delta"]) > 0
        key = (
            "correct_promotion" if passed and positive
            else "false_negative_filtering" if not passed and positive
            else "false_positive_promotion" if passed
            else "correct_filtering"
        )
        counts[key] += 1
    return counts
