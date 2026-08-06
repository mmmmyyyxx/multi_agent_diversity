from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.candidate_selection import (
    CandidateEvaluation,
    PromptCompetenceMetrics,
    TeamOutcomeMetrics,
    common_cross_branch_transition_key,
)
from multi_dataset_diverse_rl.member_objectives import member_gain_metrics
from multi_dataset_diverse_rl.responsibility import (
    CandidateMarginalContribution,
    ProtectionContribution,
    ResponsibilityState,
    initialize_repairability_state,
    record_branch_search_outcome,
    reset_state_local_repairability,
)


def _candidate(name: str, *, target: int, vote: int) -> CandidateEvaluation:
    initial = (40, 40, 40, 40, 40)
    counts = list(initial)
    counts[target] += 1
    return CandidateEvaluation(
        prompt=name,
        prompt_hash=name,
        competence=PromptCompetenceMetrics(41, 41 / 75, 0, 0.0, 0),
        team_outcome=TeamOutcomeMetrics(
            (), vote, vote / 75, (), (), (), float(vote)
        ),
        marginal=CandidateMarginalContribution(
            max(vote - 40, 0),
            0,
            vote - 40,
            float(vote - 40),
            0, 0, 0, 0, 0, 0.0,
        ),
        protection=ProtectionContribution(0, 0),
        member_gain=member_gain_metrics(initial, initial, counts, target),
    )


def _incumbent(target: int) -> CandidateEvaluation:
    initial = (40, 40, 40, 40, 40)
    return CandidateEvaluation(
        prompt="parent",
        prompt_hash="parent",
        competence=PromptCompetenceMetrics(40, 40 / 75, 0, 0.0, 0),
        team_outcome=TeamOutcomeMetrics((), 40, 40 / 75, (), (), (), 40.0),
        marginal=CandidateMarginalContribution(0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0.0),
        protection=ProtectionContribution(0, 0),
        member_gain=member_gain_metrics(initial, initial, initial, target),
    )


def main() -> None:
    state = ResponsibilityState()
    initialize_repairability_state(state, range(5))
    parent_hash = "parent-team"

    record_branch_search_outcome(
        state=state,
        agent_id=0,
        normal_completion=True,
        passed_candidate_found=False,
        update_index=0,
    )
    record_branch_search_outcome(
        state=state,
        agent_id=1,
        normal_completion=True,
        passed_candidate_found=True,
        update_index=0,
    )
    assert state.branch_failure_count_by_agent[0] == 1
    winner = _candidate("branch-1", target=1, vote=41)
    assert reset_state_local_repairability(
        state=state,
        agent_ids=range(5),
        old_team_hash=parent_hash,
        new_team_hash="successor-team",
        update_index=0,
    ) is not None
    assert not any(state.branch_failure_count_by_agent.values())

    first = _candidate("branch-a", target=0, vote=41)
    second = _candidate("branch-b", target=1, vote=42)
    candidates = (
        (first, _incumbent(0), 1),
        (second, _incumbent(1), 2),
    )
    selected = max(
        candidates,
        key=lambda row: common_cross_branch_transition_key(
            row[0], row[1], target_selection_rank=row[2]
        ),
    )
    assert selected[0] is second
    record_branch_search_outcome(
        state=state,
        agent_id=0,
        normal_completion=True,
        passed_candidate_found=True,
        update_index=1,
    )
    record_branch_search_outcome(
        state=state,
        agent_id=1,
        normal_completion=True,
        passed_candidate_found=True,
        update_index=1,
    )
    assert state.branch_failure_count_by_agent[0] == 0
    assert state.branch_failure_count_by_agent[1] == 0
    print(json.dumps({
        "ok": True,
        "same_parent_hash": parent_hash,
        "first_scenario_committed": winner.prompt_hash,
        "second_scenario_committed": selected[0].prompt_hash,
        "commit_count_per_update": 1,
        "competition_loser_failure_increment": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
