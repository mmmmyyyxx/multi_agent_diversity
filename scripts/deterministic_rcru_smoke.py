from __future__ import annotations

import hashlib
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
)
from multi_dataset_diverse_rl.member_objectives import member_gain_metrics
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
    robust_contribution_key,
)


def _metrics(
    *,
    delta: int,
    supports: tuple[int, ...],
    lcb: float,
    contribution: int,
    edit_tokens: int,
) -> ResponsibilityContributionMetrics:
    return ResponsibilityContributionMetrics(
        utility=ResponsibilityUtilityMetrics(
            repair_lane="direct_flip",
            active_residual_count=5,
            utility_total=3 + delta,
            incumbent_utility_total=3,
            utility_delta=delta,
            positive_support_count=sum(value > 0 for value in supports),
            negative_support_count=sum(value < 0 for value in supports),
            unchanged_support_count=sum(value == 0 for value in supports),
            per_example_deltas=supports,
            per_example_question_hashes=tuple(
                hashlib.sha256(f"case-{index}".encode()).hexdigest()
                for index in range(5)
            ),
        ),
        coalition=CoalitionContributionMetrics(
            positive_pivotal_count=4 + contribution,
            negative_pivotal_count=1,
            net_contribution=3 + contribution,
            incumbent_positive_pivotal_count=4,
            incumbent_negative_pivotal_count=1,
            incumbent_net_contribution=3,
            positive_pivotal_delta=contribution,
            negative_pivotal_delta=0,
            net_contribution_delta=contribution,
        ),
        robust_support=RobustSupportMetrics(
            bootstrap_replicates=1000,
            bootstrap_lower_quantile=0.10,
            bootstrap_mean_delta=delta / 5,
            bootstrap_lcb=lcb,
            deterministic_seed_hash=hashlib.sha256(
                f"{delta}:{supports}:{contribution}".encode()
            ).hexdigest(),
        ),
        edit=PromptEditMetrics(
            parent_character_count=100,
            candidate_character_count=100 + edit_tokens,
            character_growth=edit_tokens,
            parent_token_count=20,
            candidate_token_count=20 + edit_tokens,
            inserted_token_count=edit_tokens,
            deleted_token_count=0,
            replaced_token_count=0,
            total_edit_token_count=edit_tokens,
            normalized_edit_ratio=edit_tokens / (20 + edit_tokens),
        ),
    )


def _candidate(
    name: str,
    *,
    target_count: int,
    vote_count: int,
    metrics: ResponsibilityContributionMetrics | None,
) -> CandidateEvaluation:
    initial = (45, 45, 45, 45, 45)
    incumbent = initial
    candidate_counts = (target_count, 45, 45, 45, 45)
    prompt_hash = hashlib.sha256(name.encode()).hexdigest()
    return CandidateEvaluation(
        prompt=name,
        prompt_hash=prompt_hash,
        competence=PromptCompetenceMetrics(
            target_count, target_count / 75, 0, 0.0, 0
        ),
        team_outcome=TeamOutcomeMetrics(
            (), vote_count, vote_count / 75, (), (), (), 0.0
        ),
        marginal=CandidateMarginalContribution(
            max(vote_count - 40, 0),
            max(40 - vote_count, 0),
            vote_count - 40,
            0.0,
            0,
            0,
            0,
            0,
            0,
            0.0,
        ),
        protection=ProtectionContribution(0, 0),
        member_gain=member_gain_metrics(
            initial, incumbent, candidate_counts, 0
        ),
        responsibility_contribution=metrics,
    )


def main() -> None:
    incumbent = _candidate(
        "incumbent", target_count=45, vote_count=40, metrics=None
    )
    candidates = {
        "A": _candidate(
            "candidate-a",
            target_count=45,
            vote_count=40,
            metrics=_metrics(
                delta=2,
                supports=(1, 1, 0, 0, 0),
                lcb=0.0,
                contribution=1,
                edit_tokens=2,
            ),
        ),
        "B": _candidate(
            "candidate-b",
            target_count=48,
            vote_count=40,
            metrics=_metrics(
                delta=0,
                supports=(0, 0, 0, 0, 0),
                lcb=0.0,
                contribution=0,
                edit_tokens=3,
            ),
        ),
        "C": _candidate(
            "candidate-c",
            target_count=45,
            vote_count=41,
            metrics=_metrics(
                delta=-1,
                supports=(-1, 0, 0, 0, 0),
                lcb=-1.0,
                contribution=0,
                edit_tokens=3,
            ),
        ),
        "D": _candidate(
            "candidate-d",
            target_count=45,
            vote_count=40,
            metrics=_metrics(
                delta=1,
                supports=(1, 0, 0, 0, 0),
                lcb=0.0,
                contribution=0,
                edit_tokens=3,
            ),
        ),
        "E": _candidate(
            "candidate-e",
            target_count=45,
            vote_count=40,
            metrics=_metrics(
                delta=2,
                supports=(1, 1, 0, 0, 0),
                lcb=0.0,
                contribution=1,
                edit_tokens=6,
            ),
        ),
        "F": _candidate(
            "candidate-f",
            target_count=45,
            vote_count=40,
            metrics=_metrics(
                delta=1,
                supports=(2, -1, 0, 0, 0),
                lcb=0.0,
                contribution=0,
                edit_tokens=3,
            ),
        ),
    }
    decisions = {
        name: evaluate_robust_contribution_constraints(row, incumbent)
        for name, row in candidates.items()
    }
    assert decisions["A"].passed
    assert decisions["B"].rejection_reasons == ("no_vote_or_lane_progress",)
    assert "active_lane_regression" in decisions["C"].rejection_reasons
    assert decisions["D"].passed
    assert decisions["E"].passed
    assert "negative_lane_support" in decisions["F"].rejection_reasons
    feasible = [row for name, row in candidates.items() if decisions[name].passed]
    frontier = responsibility_contribution_pareto_front(feasible)
    selected = max(frontier, key=robust_contribution_key)
    assert selected.prompt_hash == candidates["A"].prompt_hash
    print(json.dumps({
        "ok": True,
        "passed_candidates": sorted(
            name for name, decision in decisions.items() if decision.passed
        ),
        "selected_candidate_hash": selected.prompt_hash,
        "rejections": {
            name: list(decision.rejection_reasons)
            for name, decision in decisions.items()
            if not decision.passed
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
