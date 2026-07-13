import pytest

from multi_dataset_diverse_rl.config import Config, build_parser


def test_cli_defaults_match_config():
    defaults = Config()
    assert defaults.best_state_selection_mode == "vote_first"
    args = build_parser().parse_args([])
    for field in [
        "val_size",
        "max_tokens",
        "optimizer_max_tokens",
        "evaluator_max_tokens",
        "candidate_eval_batch_size",
        "reward_mode",
        "candidate_selection_mode",
        "best_state_selection_mode",
        "reward_schedule_mode",
        "reward_diversity_warmup_updates",
        "beam_size",
        "num_candidates_per_parent",
        "optimizer_parent_concurrency",
        "optimizer_architecture",
        "teacher_critic_max_rounds",
        "teacher_question_pass_threshold",
        "teacher_temperature",
        "critic_temperature",
        "student_temperature",
        "teacher_max_tokens",
        "critic_max_tokens",
        "student_max_tokens",
        "student_json_max_retries",
        "student_json_repair_max_tokens",
        "student_json_repair_temperature",
        "student_candidate_schema_mode",
        "student_candidate_max_chars_per_field",
        "student_candidate_prompt_max_chars",
        "resume_from_checkpoint",
        "optimizer_fallback_mode",
        "no_effective_evolution_patience",
        "no_effective_evolution_min_optimizer_candidates",
    ]:
        assert getattr(args, field) == getattr(defaults, field)
    assert bool(args.no_effective_evolution_stop_enabled) == defaults.no_effective_evolution_stop_enabled
    assert bool(args.teacher_critic_use_voting_failure) == defaults.teacher_critic_use_voting_failure
    assert bool(args.student_json_retry_on_parse_fail) == defaults.student_json_retry_on_parse_fail
    assert bool(args.student_json_repair_enabled) == defaults.student_json_repair_enabled
    assert bool(args.student_force_minified_json) == defaults.student_force_minified_json


def test_parser_accepts_vote_useful_diversity_and_rejects_removed_mode():
    parser = build_parser()
    assert parser.parse_args(["--reward_mode", "vote_useful_diversity"]).reward_mode == "vote_useful_diversity"
    with pytest.raises(SystemExit):
        parser.parse_args(["--reward_mode", "coverage_useful_diversity"])
