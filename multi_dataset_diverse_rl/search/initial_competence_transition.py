"""Immutable initial competence floors with versioned transition policies."""
import math
import hashlib

from .. import versions
from .schemas import SearchContractError, TransitionDecision


def _validate_transition_measurement(parent, full):
    if "terminal_invalid_delta" not in full.aggregation_diagnostics:
        raise SearchContractError("INVALID_OUTPUT_GUARD_NOT_MEASURED")
    if len(parent.member_scores) != 5 or len(full.member_scores) != 5:
        raise SearchContractError("INVALID_TRANSITION_TEAM_SHAPE")
    if any(not math.isfinite(v) for v in (*full.member_scores, full.aggregate_score, parent.aggregate_score)):
        raise SearchContractError("NONFINITE_TRANSITION_MEASUREMENT")


class InitialCompetenceTransitionV2:
    """Historical V2.1: strict ensemble gain above an immutable initial floor."""
    identity = versions.UNIFIED_COMPETENCE_TRANSITION_VERSION

    def __init__(self, *, invalid_predictions_are_incorrect=False):
        self.invalid_predictions_are_incorrect = invalid_predictions_are_incorrect
        self.initial_scores = None
        self.initial_state_id = None

    def bind_initial(self, scores, state_id):
        scores = tuple(scores)
        if len(scores) != 5 or any(not math.isfinite(x) or x < 0 for x in scores) or not state_id:
            raise SearchContractError("INITIAL_COMPETENCE_NOT_FROZEN")
        if self.initial_scores is not None and (scores, state_id) != (self.initial_scores, self.initial_state_id):
            raise SearchContractError("INITIAL_COMPETENCE_CANNOT_REBASE")
        self.initial_scores, self.initial_state_id = scores, state_id

    def select(self, parent, candidates):
        if self.initial_scores is None:
            raise SearchContractError("INITIAL_COMPETENCE_NOT_FROZEN")
        feasible = []
        for row in candidates:
            full = row.full
            if not row.promoted or full is None:
                continue
            target = row.diagnostics.get("target_member")
            if not isinstance(target, int) or target not in range(5):
                raise SearchContractError("INVALID_TRANSITION_TARGET")
            if self.allows(parent, full, target):
                feasible.append(row)
        winner = max(feasible, key=lambda r: (
            r.full.aggregate_score,
            -r.full.aggregation_diagnostics.get("team_newly_broken_count", 0),
            r.full.aggregation_diagnostics.get("mean_soft_vote_utility", 0),
            0 if self.invalid_predictions_are_incorrect else -r.full.aggregation_diagnostics.get("target_invalid_count", 0),
            hashlib.sha256(r.candidate.prompt.encode()).hexdigest(),
        )) if feasible else None
        return TransitionDecision(winner, "INITIAL_COMPETENCE_TEAM_GAIN" if winner else "NO_TEAM_GAIN_WINNER")

    def allows(self, parent, full, target):
        if self.initial_scores is None:
            raise SearchContractError("INITIAL_COMPETENCE_NOT_FROZEN")
        _validate_transition_measurement(parent, full)
        return (
            full.member_scores[target] >= self.initial_scores[target]
            and full.aggregate_score > parent.aggregate_score
            and (
                self.invalid_predictions_are_incorrect
                or full.aggregation_diagnostics["terminal_invalid_delta"] <= 0
            )
        )


class InitialCompetenceTargetOrTeamTransitionV3(InitialCompetenceTransitionV2):
    """Current V2.2: safe strict progress may come from target competence or team Vote.

    The immutable initial target floor is always enforced. Team Vote may never
    regress. A candidate is deployable only if either the target member strictly
    improves versus the current parent or the team Vote strictly improves.
    """

    identity = versions.UNIFIED_TARGET_OR_TEAM_TRANSITION_VERSION

    def select(self, parent, candidates):
        if self.initial_scores is None:
            raise SearchContractError("INITIAL_COMPETENCE_NOT_FROZEN")
        feasible = []
        for row in candidates:
            full = row.full
            if not row.promoted or full is None:
                continue
            target = row.diagnostics.get("target_member")
            if not isinstance(target, int) or target not in range(5):
                raise SearchContractError("INVALID_TRANSITION_TARGET")
            if self.allows(parent, full, target):
                feasible.append(row)
        # Keep team Vote primary, but once target-only progress is admissible,
        # target competence must participate in the winner key. Otherwise two
        # Vote-neutral candidates (e.g. 23 vs 25) could be ordered only by a
        # soft-vote/hash tie-breaker.
        winner = max(feasible, key=lambda r: (
            r.full.aggregate_score,
            r.full.member_scores[int(r.diagnostics["target_member"])],
            -r.full.aggregation_diagnostics.get("team_newly_broken_count", 0),
            r.full.aggregation_diagnostics.get("mean_soft_vote_utility", 0),
            0 if self.invalid_predictions_are_incorrect else -r.full.aggregation_diagnostics.get("target_invalid_count", 0),
            hashlib.sha256(r.candidate.prompt.encode()).hexdigest(),
        )) if feasible else None
        return TransitionDecision(
            winner,
            "INITIAL_COMPETENCE_TARGET_OR_TEAM_PROGRESS"
            if winner else "NO_SAFE_TARGET_OR_TEAM_PROGRESS_WINNER",
        )

    def allows(self, parent, full, target):
        if self.initial_scores is None:
            raise SearchContractError("INITIAL_COMPETENCE_NOT_FROZEN")
        _validate_transition_measurement(parent, full)
        target_floor = full.member_scores[target] >= self.initial_scores[target]
        vote_nonregression = full.aggregate_score >= parent.aggregate_score
        target_strict = full.member_scores[target] > parent.member_scores[target]
        vote_strict = full.aggregate_score > parent.aggregate_score
        invalid_guard = (
            self.invalid_predictions_are_incorrect
            or full.aggregation_diagnostics["terminal_invalid_delta"] <= 0
        )
        return target_floor and vote_nonregression and (target_strict or vote_strict) and invalid_guard
