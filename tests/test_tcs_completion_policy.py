from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.llm_client import LLMCallResult
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.tcs import response_truncated


def call(text: str, finish_reason: str, completion_tokens: int) -> LLMCallResult:
    return LLMCallResult(
        text=text,
        prompt_tokens=2,
        completion_tokens=completion_tokens,
        total_tokens=completion_tokens + 2,
        latency_seconds=0.01,
        finish_reason=finish_reason,
    )


def test_only_provider_finish_reason_is_classified_as_truncation():
    assert response_truncated(call("{", "length", 10))
    assert not response_truncated(call("{", "stop", 20))
    assert not response_truncated(call('{"ok":true}', "stop", 20))


def test_structural_defaults_and_role_cost_summary(tmp_path):
    cfg = Config.from_flat(out_dir=str(tmp_path))
    assert cfg.models.solver_max_tokens == 1800
    assert (
        cfg.tcs.teacher_total_max_chars,
        cfg.tcs.teacher_field_max_chars,
        cfg.tcs.critic_feedback_max_chars,
        cfg.tcs.candidate_prompt_max_chars,
        cfg.tcs.total_candidate_prompt_max_chars,
    ) == (1800, 800, 500, 3000, 5000)
    system = PromptEnsembleOptimizationSystem(cfg)
    system.llm.calls.append({
        "role": "teacher",
        "client_role": "optimizer",
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
    })
    summary = system.cost_summary()
    assert summary["total_tokens"] == 5
    assert summary["optimizer_calls"] == 1
    assert summary["tokens_by_role"]["teacher"] == 5
