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
    evaluate_constraints,
)
from multi_dataset_diverse_rl.member_objectives import member_gain_metrics
from multi_dataset_diverse_rl.responsibility import (
    CandidateMarginalContribution,
    ProtectionContribution,
)


def evaluation(
    name: str,
    *,
    target: int,
    vote: int,
    vote_gain: int = 0,
    vote_loss: int = 0,
    pivotal_loss: int = 0,
) -> CandidateEvaluation:
    gains = member_gain_metrics(
        (10, 10, 10, 10, 10),
        (10, 10, 10, 10, 10),
        (target, 10, 10, 10, 10),
        0,
    )
    return CandidateEvaluation(
        prompt=name,
        prompt_hash=name,
        competence=PromptCompetenceMetrics(target, target / 20, 0, 0.0, 0),
        team_outcome=TeamOutcomeMetrics((), vote, vote / 20, (), (), (), 0.5),
        marginal=CandidateMarginalContribution(
            vote_gain,
            vote_loss,
            vote_gain - vote_loss,
            0.0,
            0,
            0,
            0,
            0,
            0,
            0.0,
        ),
        protection=ProtectionContribution(0, pivotal_loss),
        member_gain=gains,
    )


def main() -> None:
    incumbent = evaluation("incumbent", target=10, vote=10)
    five_gain_one_loss = evaluate_constraints(
        evaluation(
            "five-gain-one-loss",
            target=15,
            vote=14,
            vote_gain=5,
            vote_loss=1,
            pivotal_loss=1,
        ),
        incumbent,
    )
    vote_neutral = evaluate_constraints(
        evaluation("vote-neutral", target=11, vote=10), incumbent
    )
    vote_regression = evaluate_constraints(
        evaluation(
            "vote-regression",
            target=11,
            vote=9,
            vote_gain=1,
            vote_loss=2,
        ),
        incumbent,
    )
    assert five_gain_one_loss.hard_feasible
    assert five_gain_one_loss.pivotal_correct_loss_count == 1
    assert vote_neutral.hard_feasible
    assert not vote_regression.hard_feasible
    assert "team_vote_regression" in vote_regression.rejection_reasons
    print(json.dumps({
        "aggregate_acceptance_smoke": {
            "five_gain_one_loss_accepted": True,
            "vote_neutral_target_gain_accepted": True,
            "vote_regression_rejected": True,
        }
    }, indent=2))


if __name__ == "__main__":
    main()
