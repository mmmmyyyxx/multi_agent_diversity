"""V2.2 deployment: immutable initial competence and non-regressing team progress."""
import hashlib
import math

from ..current_contract import UNIFIED_TARGET_OR_TEAM_TRANSITION_VERSION
from .schemas import SearchContractError, TransitionDecision


def progress_path(team_gain, target_gain):
    """Observed committed deltas; never used for allocation or stopping."""
    if team_gain > 0 and target_gain > 0:
        return "TARGET_AND_TEAM"
    if team_gain > 0:
        return "TEAM"
    if target_gain > 0:
        return "TARGET"
    return "NONE"


class InitialCompetenceTargetOrTeamProgressV3:
    identity = UNIFIED_TARGET_OR_TEAM_TRANSITION_VERSION

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

    def allows(self, parent, full, target):
        if self.initial_scores is None:
            raise SearchContractError("INITIAL_COMPETENCE_NOT_FROZEN")
        if type(target) is not int or target not in range(5):
            raise SearchContractError("INVALID_TRANSITION_TARGET")
        if "terminal_invalid_delta" not in full.aggregation_diagnostics:
            raise SearchContractError("INVALID_OUTPUT_GUARD_NOT_MEASURED")
        if len(parent.member_scores) != 5 or len(full.member_scores) != 5:
            raise SearchContractError("INVALID_TRANSITION_TEAM_SHAPE")
        invalid_delta = full.aggregation_diagnostics["terminal_invalid_delta"]
        if any(not math.isfinite(v) for v in (*parent.member_scores, *full.member_scores,
                parent.aggregate_score, full.aggregate_score, invalid_delta)):
            raise SearchContractError("NONFINITE_TRANSITION_MEASUREMENT")
        return (full.member_scores[target] >= self.initial_scores[target]
                and full.aggregate_score >= parent.aggregate_score
                and (full.member_scores[target] > parent.member_scores[target]
                     or full.aggregate_score > parent.aggregate_score)
                and (self.invalid_predictions_are_incorrect or invalid_delta <= 0))

    def select(self, parent, candidates):
        if self.initial_scores is None:
            raise SearchContractError("INITIAL_COMPETENCE_NOT_FROZEN")
        feasible = [row for row in candidates if row.promoted and row.full is not None
                    and self.allows(parent, row.full, row.diagnostics.get("target_member"))]
        # Full team first, Full target second; local scores never deploy a winner.
        def key(row):
            full, target = row.full, row.diagnostics["target_member"]
            diagnostics = full.aggregation_diagnostics
            return (full.aggregate_score, full.member_scores[target],
                    -diagnostics.get("team_newly_broken_count", 0),
                    diagnostics.get("mean_soft_vote_utility", 0),
                    0 if self.invalid_predictions_are_incorrect else -diagnostics.get("target_invalid_count", 0),
                    hashlib.sha256(row.candidate.prompt.encode()).hexdigest())
        winner = max(feasible, key=key) if feasible else None
        return TransitionDecision(winner, "INITIAL_COMPETENCE_TARGET_OR_TEAM_PROGRESS"
                                  if winner else "NO_TARGET_OR_TEAM_PROGRESS_WINNER")
