import csv
import json

import pytest

from multi_dataset_diverse_rl.cli import checkpoint_behavior_config
from multi_dataset_diverse_rl.config import Config, build_parser
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem
from multi_dataset_diverse_rl.tasks import get_task_spec
from multi_dataset_diverse_rl.utils import canonical_aggregation_mode
from scripts.analyze_competence_depth import analyze


def make_system(tmp_path=None, **overrides):
    params = dict(
        agents=5,
        reward_mode="accuracy_only",
        aggregation_mode="plurality",
        competence_depth_enabled=True,
        out_dir=str(tmp_path or "."),
    )
    params.update(overrides)
    cfg = Config(**params)
    system = TraceBeamSearchSystem.__new__(TraceBeamSearchSystem)
    system.cfg = cfg
    system.task_spec = get_task_spec("bbh")
    system.agents = [AgentState("prompt") for _ in range(5)]
    system.specialization_strength = 0.0
    system.previous_epoch_per_agent_acc = []
    return system


def rollout(system, answers, gold="A", question_hash="fixed-question"):
    return system.compute_rollout_metrics(
        ["trace"] * len(answers), answers, gold, prompts=["prompt"] * len(answers),
        question_hash=question_hash,
    )


def test_two_votes_can_win_plurality_without_c3():
    metrics = rollout(make_system(), ["A", "A", "B", "C", "D"])
    assert metrics["gold_vote_count"] == 2
    assert metrics["plurality_vote_correct"] == 1
    assert metrics["plurality_vote_tie"] is False
    assert sum(metrics["individual_correct"]) >= 2
    assert sum(metrics["individual_correct"]) < 3


def test_three_votes_win_plurality():
    metrics = rollout(make_system(), ["A", "A", "A", "B", "C"])
    assert metrics["plurality_vote_correct"] == 1
    assert sum(metrics["individual_correct"]) == 3


def test_plurality_tie_is_deterministic_for_same_question_hash():
    system = make_system(vote_tie_break="random")
    left = rollout(system, ["A", "A", "B", "B", "C"], question_hash="same")
    right = rollout(system, ["A", "A", "B", "B", "C"], question_hash="same")
    assert left["plurality_vote_tie"] is True
    assert left["plurality_vote_answer"] == right["plurality_vote_answer"]
    assert left["plurality_tie_break_question_hash"] == "same"


def test_single_correct_vote_cannot_beat_two_equal_wrong_votes():
    metrics = rollout(make_system(), ["A", "B", "B", "C", "D"])
    assert metrics["plurality_vote_correct"] == 0
    assert metrics["any_correct"] == 1


def test_actual_plurality_counterfactual_finds_non_tie_pivotal_fix():
    system = make_system()
    baseline = rollout(system, ["A", "B", "B", "C", "D"])
    assert baseline["plurality_vote_correct"] == 0
    assert baseline["plurality_pivotal_fix_opportunity_per_agent"][1] == 1
    fixed = rollout(system, ["A", "A", "B", "C", "D"])
    assert fixed["plurality_vote_correct"] == 1
    assert fixed["plurality_vote_tie"] is False


def test_c3_is_not_asserted_equal_to_plurality_vote():
    system = make_system()
    metrics = rollout(system, ["A", "A", "B", "C", "D"])
    summary = system._summarize_rollout_rows([metrics])
    assert summary["coverage_depth_c3"] == 0.0
    assert summary["plurality_vote_acc"] == 1.0
    assert summary["c3_minus_plurality_vote"] == -1.0


def test_majority_alias_maps_to_plurality_without_changing_v7_payload_shape():
    assert canonical_aggregation_mode("majority") == "plurality"
    assert canonical_aggregation_mode("plurality") == "plurality"
    parser_cfg = build_parser().parse_args(["--aggregation_mode", "plurality"])
    assert parser_cfg.aggregation_mode == "plurality"
    legacy_payload = checkpoint_behavior_config(Config())
    assert "effective_aggregation_mode" not in legacy_payload
    v8_payload = checkpoint_behavior_config(Config(competence_depth_enabled=True))
    assert v8_payload["effective_aggregation_mode"] == "plurality"
    assert v8_payload["plurality_boundary_version"] == "plurality_boundary_v1"
    plurality = rollout(make_system(aggregation_mode="plurality"), ["A", "A", "B", "C", "D"])
    majority = rollout(make_system(aggregation_mode="majority"), ["A", "A", "B", "C", "D"])
    assert majority["plurality_vote_answer"] == plurality["plurality_vote_answer"]
    assert majority["requested_aggregation_mode"] == "majority"
    assert majority["effective_aggregation_mode"] == "plurality"


def test_candidate_plurality_transition_metrics_use_actual_vote_flips():
    system = make_system()
    metrics = system._candidate_boundary_error_metrics([{
        "baseline_target_correct": False,
        "candidate_target_correct": True,
        "baseline_vote_correct": False,
        "candidate_vote_correct": True,
        "plurality_pivotal_fix_opportunity": True,
        "plurality_pivotal_fix": True,
        "plurality_pivotal_loss": False,
        "peer_wrong_count": 2,
    }])
    assert metrics["plurality_pivotal_fix_opportunity_rate"] == 1.0
    assert metrics["plurality_pivotal_fix_rate"] == 1.0
    assert metrics["plurality_pivotal_loss_rate"] == 0.0
    assert metrics["pivotal_definition"] == "actual_plurality_counterfactual"


def test_candidate_audit_marks_missing_metrics_instead_of_zero(tmp_path):
    run_dir = tmp_path / "task" / "setting_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "history.json").write_text(json.dumps([{"epoch": 1, "test": {"vote_acc": 0.5}}]), encoding="utf-8")
    (run_dir / "update_logs.jsonl").write_text(json.dumps({
        "event": "candidate_evaluation", "candidate_id": "c1", "metrics": {"reward": 0.2}
    }) + "\n", encoding="utf-8")
    analyze(tmp_path)
    with (tmp_path / "competence_depth_candidate_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        candidate = next(csv.DictReader(handle))
    with (tmp_path / "competence_depth_run_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        run = next(csv.DictReader(handle))
    assert candidate["metric_missing"] == "True"
    assert candidate["depth2_gain_rate"] == ""
    assert float(run["plurality_metric_coverage"]) == 0.0
