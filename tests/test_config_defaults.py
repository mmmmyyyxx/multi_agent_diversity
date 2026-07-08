from multi_dataset_diverse_rl.config import Config, build_parser


def test_cli_defaults_match_config():
    defaults = Config()
    args = build_parser().parse_args([])
    for field in [
        "val_size",
        "max_tokens",
        "optimizer_max_tokens",
        "evaluator_max_tokens",
        "candidate_eval_batch_size",
        "reward_mode",
        "reward_schedule_mode",
        "reward_diversity_warmup_updates",
        "beam_size",
        "num_candidates_per_parent",
        "optimizer_fallback_mode",
        "no_effective_evolution_patience",
        "no_effective_evolution_min_optimizer_candidates",
    ]:
        assert getattr(args, field) == getattr(defaults, field)
    assert bool(args.no_effective_evolution_stop_enabled) == defaults.no_effective_evolution_stop_enabled


def test_parser_accepts_coverage_useful_diversity_and_deprecated_alias():
    parser = build_parser()
    assert parser.parse_args(["--reward_mode", "coverage_useful_diversity"]).reward_mode == "coverage_useful_diversity"
    assert parser.parse_args(["--reward_mode", "coverage_rescue_diversity"]).reward_mode == "coverage_rescue_diversity"
