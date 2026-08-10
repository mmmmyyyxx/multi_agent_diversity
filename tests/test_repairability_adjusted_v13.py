import asyncio
import hashlib
import json

import pytest

import multi_dataset_diverse_rl.responsibility as responsibility_module
from multi_dataset_diverse_rl.candidate_selection import (
    CandidateEvaluation,
    PromptCompetenceMetrics,
    TeamOutcomeMetrics,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.member_objectives import member_gain_metrics
from multi_dataset_diverse_rl.responsibility import (
    CandidateMarginalContribution,
    MemberAwareRepairOpportunity,
    ProtectionContribution,
    RepairLane,
    ResponsibilityState,
    initialize_repairability_state,
    record_branch_search_outcome,
    repairability_adjusted_target_scores,
    reset_state_local_repairability,
    select_repairability_targets,
)
from multi_dataset_diverse_rl.system import (
    CandidateFunnel,
    CandidateRuntime,
    PromptEnsembleOptimizationSystem,
    TargetBranchResult,
)
from multi_dataset_diverse_rl.tcs import StudentPromptCandidate


def opportunity(agent, name, *, direct=0, margin=0):
    return MemberAwareRepairOpportunity(
        agent_id=agent,
        question_hash=name,
        vote_flip_gain=direct,
        margin_gain=margin,
        member_error=True,
        coverage_opportunity=False,
        conversion_opportunity=True,
        dominant_wrong_member=False,
        unique_correct=False,
        pivotal_correct=False,
        oracle_soft_utility_gain=0.0,
    )


def scores_for(
    active,
    state,
    counts=(50, 50, 50, 50, 50),
    *,
    wait_inside_discount=True,
):
    lanes = {
        agent: RepairLane.DIRECT_FLIP for agent in active
    }
    return repairability_adjusted_target_scores(
        active_assignments=active,
        state=state,
        seed=46,
        current_member_correct_counts=counts,
        initial_member_correct_counts=(45, 45, 45, 45, 45),
        member_uplift_tolerance=5,
        legal_assignments=active,
        service_portfolios=active,
        active_lane_by_agent=lanes,
        wait_inside_discount=wait_inside_discount,
    )


def test_score_excludes_direct_flip_margin_and_normalizes_per_state():
    state = ResponsibilityState()
    initialize_repairability_state(state, range(5))
    active = {
        0: (
            opportunity(0, "direct", direct=1, margin=99),
            opportunity(0, "support", direct=0, margin=3),
            opportunity(0, "negative", direct=0, margin=-5),
        ),
        1: (opportunity(1, "other", direct=0, margin=6),),
    }
    rows = {row.agent_id: row for row in scores_for(active, state)}
    assert rows[0].direct_fix_count == 1
    assert rows[0].support_margin_sum == 3
    assert rows[0].normalized_direct_fix == 1.0
    assert rows[0].normalized_support_margin == 0.5
    assert rows[1].normalized_support_margin == 1.0


def test_all_zero_dimensions_failure_discount_and_wait_exploration():
    state = ResponsibilityState()
    initialize_repairability_state(state, range(5))
    state.branch_failure_count_by_agent[0] = 2
    state.updates_since_selected_by_agent.update({0: 0, 1: 10})
    active = {
        0: (opportunity(0, "a"),),
        1: (opportunity(1, "b"),),
    }
    rows = {row.agent_id: row for row in scores_for(active, state)}
    assert rows[0].opportunity_value == rows[1].opportunity_value == 0.0
    assert rows[0].repairability_discount == pytest.approx(1 / 3)
    assert rows[0].expected_update_value == 0.0
    assert rows[1].expected_update_value == pytest.approx(0.05)
    assert rows[1].normalized_wait == 1.0


def test_v15_w1_wait_is_inside_state_local_failure_discount():
    state = ResponsibilityState()
    initialize_repairability_state(state, range(5))
    state.branch_failure_count_by_agent[0] = 8
    state.updates_since_selected_by_agent.update({0: 10, 1: 0})
    active = {
        0: (opportunity(0, "target", direct=0, margin=1),),
        1: (opportunity(1, "maximum", direct=0, margin=9),),
    }
    rows = {
        row.agent_id: row
        for row in scores_for(active, state, counts=(45, 55, 45, 45, 45))
    }
    target = rows[0]
    assert target.opportunity_value == pytest.approx(0.23333333333333334)
    assert target.normalized_wait == 1.0
    assert target.repairability_discount == pytest.approx(1 / 9)
    assert target.expected_update_value == pytest.approx(
        (0.23333333333333334 + 0.05) / 9
    )
    assert target.expected_update_value != pytest.approx(
        0.23333333333333334 / 9 + 0.05
    )

    legacy = {
        row.agent_id: row
        for row in scores_for(
            active,
            state,
            counts=(45, 55, 45, 45, 45),
            wait_inside_discount=False,
        )
    }[0]
    assert legacy.expected_update_value == pytest.approx(
        0.23333333333333334 / 9 + 0.05
    )


def test_v15_w1_boundary_cases_and_failure_monotonicity():
    active = {
        0: (opportunity(0, "a", direct=1),),
        1: (opportunity(1, "b", direct=1),),
    }
    state = ResponsibilityState()
    initialize_repairability_state(state, range(5))
    state.updates_since_selected_by_agent.update({0: 0, 1: 10})
    rows = {row.agent_id: row for row in scores_for(active, state)}
    assert rows[0].expected_update_value == pytest.approx(
        rows[0].opportunity_value
    )
    assert rows[1].expected_update_value == pytest.approx(
        rows[1].opportunity_value + 0.05
    )

    state.branch_failure_count_by_agent[1] = 3
    discounted = {row.agent_id: row for row in scores_for(active, state)}[1]
    assert discounted.expected_update_value == pytest.approx(
        (discounted.opportunity_value + 0.05) / 4
    )
    assert discounted.expected_update_value < rows[1].expected_update_value


def test_v15_w1_ranking_is_deterministic():
    active = {
        0: (opportunity(0, "a", direct=1),),
        1: (opportunity(1, "b", direct=1),),
    }
    state = ResponsibilityState()
    first = scores_for(active, state)
    second = scores_for(active, state)
    assert first == second
    assert [row.agent_id for row in first] == [row.agent_id for row in second]


def test_selector_is_total_top_k_downgrades_and_never_calls_pareto(monkeypatch):
    monkeypatch.setattr(
        responsibility_module,
        "_pareto_front_numbers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("v13 selector called Pareto")
        ),
    )
    state = ResponsibilityState()
    initialize_repairability_state(state, range(5))
    state.frozen_by_agent[0] = True
    active = {
        0: (opportunity(0, "a", direct=1),),
        1: (opportunity(1, "b", margin=2),),
    }
    rows = scores_for(active, state)
    assert {row.agent_id for row in rows} == {0, 1}
    assert state.consecutive_failed_updates_by_agent == {}
    assert len(select_repairability_targets(rows, target_branch_count=1)) == 1
    selected = select_repairability_targets(rows, target_branch_count=2)
    assert len(selected) == 2
    assert select_repairability_targets(
        rows[:1], target_branch_count=2
    ) == rows[:1]


def test_state_local_branch_counters_and_reset_contract():
    state = ResponsibilityState()
    initialize_repairability_state(state, range(3))
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
        normal_completion=False,
        passed_candidate_found=False,
        update_index=0,
    )
    record_branch_search_outcome(
        state=state,
        agent_id=2,
        normal_completion=True,
        passed_candidate_found=True,
        update_index=0,
    )
    assert state.branch_failure_count_by_agent == {0: 1, 1: 0, 2: 0}
    assert state.branch_attempt_count_by_agent == {0: 1, 1: 0, 2: 1}
    assert state.branch_feasible_count_by_agent == {0: 0, 1: 0, 2: 1}
    assert reset_state_local_repairability(
        state=state,
        agent_ids=range(3),
        old_team_hash="same",
        new_team_hash="same",
        update_index=0,
    ) is None
    assert state.branch_failure_count_by_agent[0] == 1
    event = reset_state_local_repairability(
        state=state,
        agent_ids=range(3),
        old_team_hash="same",
        new_team_hash="changed",
        update_index=1,
    )
    assert event is not None
    assert not any(state.branch_failure_count_by_agent.values())
    assert not any(state.branch_attempt_count_by_agent.values())
    assert not any(state.branch_feasible_count_by_agent.values())


@pytest.mark.parametrize(
    "failure_class",
    (
        "transport_failure",
        "teacher_schema_exhausted",
        "critic_schema_exhausted",
        "student_invalid_exhausted_after_upstream_regeneration",
        "student_provider_truncation",
        "proposal_protocol_failure",
    ),
)
def test_operational_terminal_failures_are_not_complete_branch_failures(
    failure_class,
):
    funnel = CandidateFunnel(
        student_calls=8,
        raw_candidate_count=8,
        terminal_failure_class=failure_class,
    )
    assert not PromptEnsembleOptimizationSystem.is_complete_repairability_failure(
        funnel
    )


def evaluation(name, target, *, vote):
    initial = (40, 40, 40, 40, 40)
    counts = list(initial)
    counts[target] += 1
    return CandidateEvaluation(
        prompt=name,
        prompt_hash=hashlib.sha256(name.encode()).hexdigest(),
        competence=PromptCompetenceMetrics(41, 41 / 75, 0, 0.0, 0),
        team_outcome=TeamOutcomeMetrics(
            (), vote, vote / 75, (), (), (), float(vote)
        ),
        marginal=CandidateMarginalContribution(
            max(0, vote - 40), 0, vote - 40, float(vote - 40),
            0, 0, 0, 0, 0, 0.0,
        ),
        protection=ProtectionContribution(0, 0),
        member_gain=member_gain_metrics(initial, initial, counts, target),
    )


def candidate(name, target, profile, *, vote):
    prompt_hash = hashlib.sha256(name.encode()).hexdigest()
    value = CandidateRuntime(
        student_candidate=StudentPromptCandidate(name),
        prompt=name,
        prompt_hash=prompt_hash,
        generation=1,
        parent_prompt_hash="parent",
    )
    value.profile = profile
    value.final_evaluation = evaluation(name, target, vote=vote)
    return value


async def build_dual_system(tmp_path):
    async def solver(_question, _agent_id, _prompt):
        return PromptAnswer("B", "FINAL_ANSWER: B", True)

    system = PromptEnsembleOptimizationSystem(
        Config.from_flat(
            out_dir=str(tmp_path),
            experiment_setting="shared_member_aware_dual_target",
            initialization_mode="provided_prompt_set",
            provided_prompts_json='["p0","p1","p2","p3","p4"]',
        ),
        solver=solver,
    )
    await system.initialize_fixed_probe([{"question": "q", "answer": "A"}])
    system.cached_active_lane_by_agent.update({
        0: RepairLane.DIRECT_FLIP,
        1: RepairLane.COVERAGE,
    })
    assigned = {
        0: [opportunity(0, "q", direct=1)],
        1: [opportunity(1, "q", margin=1)],
    }
    system.ensure_responsibility_current = lambda: ({}, assigned)
    system.select_targets = lambda *_args: ((0, 1), [])
    return system, assigned


def branch(system, target, rank, accepted, *, normal=True):
    return TargetBranchResult(
        target_agent_id=target,
        target_selection_rank=rank,
        parent_team_hash=system.team_prompt_state_hash(),
        active_lane=system.cached_active_lane_by_agent[target],
        assigned_hashes={"q"},
        funnel=CandidateFunnel(stage_a_evaluated=1),
        accepted=accepted,
        incumbent=evaluation(f"incumbent-{target}", target, vote=40),
        evaluated=[] if accepted is None else [accepted],
        normal_completion=normal,
    )


def test_dual_update_commits_only_feasible_branch_and_resets_failures(tmp_path):
    async def run():
        system, _ = await build_dual_system(tmp_path)
        profile = system.active_profiles[1]

        async def evaluate_branch(**kwargs):
            target = kwargs["target"]
            return branch(
                system,
                target,
                kwargs["target_rank"],
                None if target == 0 else candidate(
                    "winner-1", 1, profile, vote=41
                ),
            )

        system._evaluate_target_branch = evaluate_branch
        accepted = await system.update_once(0)
        return system, accepted

    system, accepted = asyncio.run(run())
    assert accepted
    assert [agent.current_prompt for agent in system.agents] == [
        "p0", "winner-1", "p2", "p3", "p4"
    ]
    assert system.responsibility_state.specialization_anchor_by_agent[0] is None
    assert system.responsibility_state.specialization_anchor_by_agent[1] is (
        RepairLane.COVERAGE
    )
    assert not any(
        system.responsibility_state.branch_failure_count_by_agent.values()
    )
    assert {
        row["parent_team_hash"]
        for row in system.dual_target_branch_decisions
    } == {
        system.dual_target_commit_decisions[0]["parent_team_hash"]
    }
    assert system.dual_target_commit_decisions[0]["committed_target_id"] == 1
    sanitized = json.dumps({
        "scores": system.repairability_adjusted_target_scores,
        "branches": system.dual_target_branch_decisions,
        "commits": system.dual_target_commit_decisions,
        "failures": system.repairability_failure_events,
        "resets": system.repairability_reset_events,
    })
    assert "winner-1" not in sanitized
    assert '"prompt"' not in sanitized
    assert '"question"' not in sanitized
    assert "FINAL_ANSWER" not in sanitized


def test_dual_update_global_key_selects_one_and_loser_is_not_failure(tmp_path):
    async def run():
        system, _ = await build_dual_system(tmp_path)
        profiles = system.active_profiles

        async def evaluate_branch(**kwargs):
            target = kwargs["target"]
            return branch(
                system,
                target,
                kwargs["target_rank"],
                candidate(
                    f"candidate-{target}",
                    target,
                    profiles[target],
                    vote=41 + target,
                ),
            )

        system._evaluate_target_branch = evaluate_branch
        await system.update_once(0)
        return system

    system = asyncio.run(run())
    commit = system.dual_target_commit_decisions[0]
    assert commit["committed_target_id"] == 1
    assert sum(
        agent.current_prompt.startswith("candidate-")
        for agent in system.agents
    ) == 1
    loser = next(
        row for row in system.dual_target_branch_decisions
        if row["target_agent_id"] == 0
    )
    assert loser["competition_loser"]
    loser_event = next(
        row for row in system.repairability_failure_events
        if row["agent_id"] == 0
    )
    assert loser_event["passed_candidate_found"]
    assert loser_event["branch_failure_count"] == 0


def test_v13_training_dynamics_omits_legacy_pareto_and_freeze_fields(tmp_path):
    system, _ = asyncio.run(build_dual_system(tmp_path))
    row = system.record_training_dynamics(update_index=-1)
    encoded = json.dumps(row, sort_keys=True)
    assert "target_pareto_front" not in encoded
    assert "target_frozen" not in encoded
    assert "repairability_adjusted_target_score" in row
