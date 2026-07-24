from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.llm_client import LLMCallResult
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.tcs import response_truncated


def call(text, finish_reason, completion_tokens, limit):
    return LLMCallResult(
        text=text,
        prompt_tokens=2,
        completion_tokens=completion_tokens,
        total_tokens=completion_tokens + 2,
        latency_seconds=0.01,
        finish_reason=finish_reason,
        completion_token_limit=limit,
        hit_completion_limit=finish_reason == "length" or completion_tokens >= limit,
    )


def test_finish_reason_and_incomplete_limit_hit_are_classified_as_truncation():
    assert response_truncated(call("{", "length", 10, 20))
    assert response_truncated(call("{", "stop", 20, 20))
    assert not response_truncated(call('{"ok":true}', "stop", 20, 20))
    assert not response_truncated(call('{"ok":true}', "stop", 10, 20))


def test_role_budget_defaults_and_cost_summary_remain_stable(tmp_path):
    cfg = Config.from_flat(out_dir=str(tmp_path))
    assert (
        cfg.tcs.teacher_max_tokens,
        cfg.tcs.critic_max_tokens,
        cfg.tcs.student_max_tokens,
        cfg.models.solver_max_tokens,
    ) == (600, 300, 1400, 1800)
    system = PromptEnsembleOptimizationSystem(cfg)
    system.llm.calls.append({
        "role": "optimizer",
        "model": "fake",
        "attempt": 1,
        "success": True,
        "status_code": 200,
        "error_type": "",
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
        "latency_seconds": 0.1,
        "finish_reason": "stop",
        "completion_token_limit": 600,
        "hit_completion_limit": False,
    })
    summary = system.cost_summary()
    assert summary["total_tokens"] == 5
    assert summary["optimizer_calls"] == 1
