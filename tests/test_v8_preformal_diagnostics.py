import copy

from multi_dataset_diverse_rl.diagnostics.candidate_funnel import (
    empty_candidate_channel_funnel,
    record_candidate_stage,
    record_funnel_event,
    validate_candidate_channel_funnel,
)
from multi_dataset_diverse_rl.metrics.vote_conversion import (
    question_vote_conversion_diagnostics,
    summarize_vote_conversion,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.quality_diversity import (
    enumerate_joint_teams,
    hierarchical_quality_bands,
    select_stable_joint_team,
)
from multi_dataset_diverse_rl.utils import plurality_vote_with_diagnostics


def _vote_row(answers, *, gold="A", question_hash="q", invalid_flags=None):
    vote = plurality_vote_with_diagnostics(
        answers, tie_break_method="random", seed=42, question_hash=question_hash,
    )
    match = lambda left, right: str(left).strip().upper() == str(right).strip().upper()
    counts = dict(vote["vote_counts"])
    row = {
        "question_hash": question_hash,
        "individual_correct": [int(match(answer, gold)) for answer in answers],
        "vote_correct": int(match(vote["vote_answer"], gold)),
        "vote_counts": counts,
        "gold_vote_count": sum(int(count) for answer, count in counts.items() if match(answer, gold)),
        "largest_wrong_vote_count": max(
            [int(count) for answer, count in counts.items() if not match(answer, gold)], default=0,
        ),
        "invalid_flags": list(invalid_flags or [0] * len(answers)),
    }
    row.update(question_vote_conversion_diagnostics(row))
    return row


def test_vote_conversion_six_five_agent_cases_reconcile():
    rows = [
        _vote_row(["A", "B", "B", "C", "D"], question_hash="c1-gap"),
        _vote_row(["A", "A", "B", "B", "B"], question_hash="c2-gap"),
        _vote_row(["A", "A", "B", "B", "C"], question_hash="c2-tie"),
        _vote_row(["A", "A", "A", "B", "B"], question_hash="c3-win"),
        _vote_row(["B", "B", "C", "C", "D"], question_hash="c0"),
        _vote_row(
            ["A", "B", "B", "C", ""], question_hash="invalid",
            invalid_flags=[0, 0, 0, 0, 1],
        ),
    ]
    summary = summarize_vote_conversion(rows)
    assert rows[0]["correct_agent_count"] == 1 and rows[0]["vote_correct"] == 0
    assert rows[1]["correct_agent_count"] == 2 and rows[1]["vote_correct"] == 0
    assert rows[2]["correct_agent_count"] == 2 and rows[2]["gold_in_top_tie"] is True
    assert rows[3]["correct_agent_count"] == 3 and rows[3]["vote_correct"] == 1
    assert rows[4]["oracle_correct"] == 0
    assert rows[5]["invalid_agent_count"] == 1
    assert (summary["c0_count"], summary["c1_count"], summary["c2_count"], summary["c3plus_count"]) == (1, 2, 2, 1)
    assert summary["oracle_correct_count"] == 5
    assert summary["oracle_vote_gap_count"] == summary["oracle_correct_count"] - summary["vote_correct_count"]
    assert summary["c3plus_vote_fail_count"] == 0
    assert summary["vote_normalization_anomaly_count"] == 0


def test_vote_conversion_zero_oracle_has_finite_zero_rate():
    summary = summarize_vote_conversion([_vote_row(["B", "B", "C", "C", "D"])])
    assert summary["oracle_correct_count"] == 0
    assert summary["oracle_to_vote_conversion_rate"] == 0.0
    assert summary["dominant_wrong_concentration"] == 0.0


def test_candidate_channel_funnel_counts_and_deduplicates_candidate_identity():
    funnel = empty_candidate_channel_funnel()
    seen = {}
    tcs = {"prompt_hash": "tcs", "generation": 1, "candidate_source": "teacher_critic_student"}
    open_probation = {"prompt_hash": "open-p", "generation": 1, "candidate_source": "open_mechanism_exploration"}
    open_catastrophic = {"prompt_hash": "open-c", "generation": 1, "candidate_source": "open_mechanism_exploration"}
    record_funnel_event(funnel, seen, channel="teacher_critic_student", stage="generation_call_count", identity="tcs-call")
    record_funnel_event(funnel, seen, channel="teacher_critic_student", stage="raw_candidate_count", identity="tcs-call", amount=1)
    record_funnel_event(funnel, seen, channel="open_mechanism_exploration", stage="generation_call_count", identity="open-call")
    record_funnel_event(funnel, seen, channel="open_mechanism_exploration", stage="raw_candidate_count", identity="open-call", amount=3)
    for item, final_stage in (
        (tcs, "safe_count"),
        (open_probation, "probation_count"),
        (open_catastrophic, "catastrophic_count"),
    ):
        for stage in ("schema_valid_candidate_count", "prescreen_pass_count", "evaluated_candidate_count"):
            assert record_candidate_stage(funnel, seen, item, agent_id=0, stage=stage)
        assert record_candidate_stage(funnel, seen, item, agent_id=0, stage=final_stage)
    assert not record_candidate_stage(
        funnel, seen, copy.deepcopy(open_probation), agent_id=0, stage="schema_valid_candidate_count"
    )
    for stage in ("archive_retained_count", "representative_selected_count", "active_selected_count"):
        assert record_candidate_stage(funnel, seen, tcs, agent_id=0, stage=stage)
    validate_candidate_channel_funnel(funnel)
    assert funnel["teacher_critic_student"]["safe_count"] == 1
    assert funnel["open_mechanism_exploration"]["schema_valid_candidate_count"] == 2
    assert funnel["open_mechanism_exploration"]["probation_count"] == 1
    assert funnel["open_mechanism_exploration"]["catastrophic_count"] == 1


def test_additional_diagnostics_do_not_change_quality_band_selection():
    incumbent = {
        "vote_correct_count": 4, "total_agent_correct_count": 12,
        "bottom2_correct_count": 3, "coverage_depth_c1_correct_count": 4,
        "coverage_depth_c2_correct_count": 3, "per_agent_correct_count": [3, 3, 2, 2, 2],
        "vote_acc": 1.0, "mean_individual_acc": 0.6, "bottom2_mean_acc": 0.5,
        "coverage_depth_c1": 1.0, "coverage_depth_c2": 0.75,
        "per_agent_acc": [0.75, 0.75, 0.5, 0.5, 0.5], "answer_vectors": [["A"] * 4] * 5,
        "beam_indices": [0, 0, 0, 0, 0], "prompt_hashes": ["a"] * 5,
    }
    alternate = {**copy.deepcopy(incumbent), "beam_indices": [1, 0, 0, 0, 0], "prompt_hashes": ["b", "a", "a", "a", "a"]}
    without = hierarchical_quality_bands([incumbent, alternate], incumbent, type("Cfg", (), {
        "joint_vote_band_questions": 0, "joint_mean_band_correct_count": 0,
        "joint_bottom2_band_correct_count": 0, "joint_c1_band_questions": 0,
        "joint_c2_band_questions": 0, "joint_allowed_vote_loss_questions": 0,
        "joint_allowed_total_agent_correct_loss": 0, "joint_allowed_bottom2_correct_loss": 0,
        "joint_allowed_c1_loss_questions": 0, "joint_allowed_c2_loss_questions": 0,
        "joint_allowed_per_agent_correct_loss": 0,
    })())
    enriched = copy.deepcopy([incumbent, alternate])
    for team in enriched:
        team.update(summarize_vote_conversion([_vote_row(["A", "A", "B", "C", "D"])]))
        team["candidate_channel_funnel"] = empty_candidate_channel_funnel()
    with_diagnostics = hierarchical_quality_bands(enriched, enriched[0], type("Cfg", (), {
        "joint_vote_band_questions": 0, "joint_mean_band_correct_count": 0,
        "joint_bottom2_band_correct_count": 0, "joint_c1_band_questions": 0,
        "joint_c2_band_questions": 0, "joint_allowed_vote_loss_questions": 0,
        "joint_allowed_total_agent_correct_loss": 0, "joint_allowed_bottom2_correct_loss": 0,
        "joint_allowed_c1_loss_questions": 0, "joint_allowed_c2_loss_questions": 0,
        "joint_allowed_per_agent_correct_loss": 0,
    })())
    assert [team["beam_indices"] for team in without["final"]] == [team["beam_indices"] for team in with_diagnostics["final"]]
    assert [len(level) for level in without["bands"]] == [len(level) for level in with_diagnostics["bands"]]


def test_vote_and_funnel_diagnostics_do_not_change_joint_team_selection():
    def profile(prompt_hash, correctness):
        return {
            "prompt_hash": prompt_hash,
            "answer_vector": ["A" if value else "B" for value in correctness],
            "correctness_vector": list(correctness),
            "mechanism_representation": {
                "family_id": "hard_elimination",
                "normalized_operation_sequence": ["hard_elimination"],
                "mechanism_embedding": [1.0, 0.0],
            },
        }

    beams = [
        [profile("a0", [1, 0, 1, 0]), profile("a1", [0, 1, 1, 0])],
        *[[profile(f"p{agent_id}", [1, 0, 1, 0])] for agent_id in range(1, 5)],
    ]
    teams = enumerate_joint_teams(
        beams, ["A"] * 4, ["q0", "q1", "q2", "q3"],
        vote_fn=plurality_vote_with_diagnostics,
        match_fn=lambda left, right: left == right,
        tie_break_method="random", seed=42,
    )
    incumbent = teams[0]
    cfg = Config(agents=5)
    kwargs = dict(
        initial_per_agent=[0.5] * 5,
        lineage_states=[AgentState("prompt").lineage_state for _ in range(5)],
        probe_size=4,
        config=cfg,
        gold_answers=["A"] * 4,
        question_hashes=["q0", "q1", "q2", "q3"],
        vote_fn=plurality_vote_with_diagnostics,
        match_fn=lambda left, right: left == right,
        tie_break_method="random",
        seed=42,
    )
    baseline = select_stable_joint_team(copy.deepcopy(teams), copy.deepcopy(incumbent), **kwargs)
    enriched_teams = copy.deepcopy(teams)
    for team in enriched_teams:
        team["candidate_channel_funnel"] = empty_candidate_channel_funnel()
        team["diagnostic_marker"] = {"enabled": True}
    enriched = select_stable_joint_team(
        enriched_teams, copy.deepcopy(enriched_teams[0]), **kwargs
    )
    assert baseline["selected"]["prompt_hashes"] == enriched["selected"]["prompt_hashes"]
    assert baseline["selected"]["beam_indices"] == enriched["selected"]["beam_indices"]
    assert baseline["hierarchical_band_counts"] == enriched["hierarchical_band_counts"]
