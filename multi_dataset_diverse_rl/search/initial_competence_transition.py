"""Immutable initial floor and strict fixed-peer gain."""
import math
import hashlib
from .. import versions
from .schemas import SearchContractError, TransitionDecision

class InitialCompetenceTransitionV2:
    """Strict ensemble gain above an immutable initial member competence floor."""
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
        # Team score first; remaining safety ties never use local search scores.
        winner = max(feasible, key=lambda r: (r.full.aggregate_score,
            -r.full.aggregation_diagnostics.get("team_newly_broken_count", 0),
            r.full.aggregation_diagnostics.get("mean_soft_vote_utility", 0),
            0 if self.invalid_predictions_are_incorrect else -r.full.aggregation_diagnostics.get("target_invalid_count", 0),
            hashlib.sha256(r.candidate.prompt.encode()).hexdigest())) if feasible else None
        return TransitionDecision(winner, "INITIAL_COMPETENCE_TEAM_GAIN" if winner else "NO_TEAM_GAIN_WINNER")

    def allows(self, parent, full, target):
        if self.initial_scores is None:
            raise SearchContractError("INITIAL_COMPETENCE_NOT_FROZEN")
        if "terminal_invalid_delta" not in full.aggregation_diagnostics:
            raise SearchContractError("INVALID_OUTPUT_GUARD_NOT_MEASURED")
        if len(parent.member_scores) != 5 or len(full.member_scores) != 5:
            raise SearchContractError("INVALID_TRANSITION_TEAM_SHAPE")
        if any(not math.isfinite(v) for v in (*full.member_scores, full.aggregate_score, parent.aggregate_score)):
            raise SearchContractError("NONFINITE_TRANSITION_MEASUREMENT")
        return (full.member_scores[target] >= self.initial_scores[target]
                and full.aggregate_score > parent.aggregate_score
                and (self.invalid_predictions_are_incorrect or full.aggregation_diagnostics["terminal_invalid_delta"] <= 0))
