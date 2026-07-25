from pathlib import Path

from multi_dataset_diverse_rl.config import ModelConfig


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_matched_gpt4omini_seed42.ps1"
)


def test_matched_gpt4omini_script_freezes_only_the_requested_pilot() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert ModelConfig().agent_model == "deepseek-chat"
    assert ModelConfig().optimizer_model == "deepseek-chat"
    assert ModelConfig().evaluator_model == "deepseek-chat"

    assert (
        '"--settings", '
        '"shared_baseline,shared_independent_accuracy,shared_member_aware_full"'
    ) in text
    assert text.count('"gpt-4o-mini"') == 3
    assert '"--seeds", "42"' in text
    assert '"--train_size", "75"' in text
    assert '"--val_size", "50"' in text
    assert '"--test_size", "125"' in text
    assert '"--epochs", "8"' in text
    assert '"--update_every", "75"' in text
    assert '"--candidate_eval_pool_size", "75"' in text
    assert '"--num_candidates_per_parent", "2"' in text
    assert '"--stage_a_channel_top_k", "2"' in text
    assert '"--stage_b_candidate_budget", "2"' in text
    assert '"--solver_max_tokens", "1800"' in text
    assert '"--solver_invalid_max_retries", "3"' in text
    assert '"--eval_solver_call_concurrency", "8"' in text
    assert '"--resume_from_checkpoint", "0"' in text
    assert '"--resume_completed" "0"' in text

    assert "shared_peer_state_vote_first" not in text
    assert "shared_peer_state_member_pareto" not in text
    assert "shared_member_aware_responsibility" not in text
    assert '"43"' not in text
    assert '"44"' not in text
    assert text.index("preflight_member_aware.py") < text.index(
        "run_task_level_accuracy.py"
    )
