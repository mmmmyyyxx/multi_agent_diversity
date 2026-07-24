import asyncio
import json

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.llm_client import LLMCallResult
from scripts.critic_calibration_replay import run


class FakeCalibrationClient:
    def __init__(self, invalid_first=False, truncate=False):
        self.roles = []
        self.invalid_first = invalid_first
        self.truncate = truncate
        self.request_counts = {}
        self.user_prompts = []

    async def chat_result(
        self, _model, system, user, _temperature, _max_tokens, role,
        _logical_role=None,
    ):
        self.roles.append(role)
        self.user_prompts.append(user)
        request_count = self.request_counts.get(system, 0)
        self.request_counts[system] = request_count + 1
        if self.invalid_first and request_count == 0:
            return LLMCallResult("not-json", 2, 1, 3, 0.01, "stop")
        failed = []
        feedback = ""
        if "demonstrated question has answer A" in system:
            failed = ["shortcut_or_copying"]
            feedback = "Remove the memorized answer."
        elif "Think carefully and double-check" in system:
            failed = ["actionable_specificity"]
            feedback = "Specify an executable check."
        elif "No preservation rule is needed" in system:
            failed = ["preservation_or_output_risk"]
            feedback = "Add an operational preservation rule."
        elif "occupation typically performs" in system:
            failed = ["shortcut_or_copying"]
            feedback = "Remove the stereotype shortcut."
        text = json.dumps({
            "failed_checks": failed,
            "risk_case_ids": [],
            "feedback": feedback,
        })
        return LLMCallResult(
            text, 2, 3, 5, 0.01, "length" if self.truncate else "stop"
        )

    def cost_summary(self):
        return {
            "solver_calls": 0,
            "optimizer_calls": 0,
            "evaluator_calls": len(self.roles),
            "total_llm_calls": len(self.roles),
            "total_tokens": 0,
        }


def test_calibration_replay_is_evaluator_only_and_separates_hard_blockers(tmp_path):
    client = FakeCalibrationClient()
    report = asyncio.run(run(
        Config.from_flat(out_dir=str(tmp_path)),
        client,
    ))
    assert report["ok"] is True
    assert report["summary"]["good_acceptance_count"] == 1
    assert report["summary"]["memorizing_rejection_count"] == 1
    assert report["summary"]["classification_correct_count"] == 5
    assert client.roles == ["evaluator"] * 5
    assert (tmp_path / "critic_calibration_report.json").is_file()


def test_calibration_retry_uses_current_three_field_protocol(tmp_path):
    client = FakeCalibrationClient(invalid_first=True)
    report = asyncio.run(run(Config.from_flat(out_dir=str(tmp_path)), client))
    assert report["ok"] is True
    retry_prompts = [
        prompt for prompt in client.user_prompts
        if prompt.startswith("Your previous response was invalid")
    ]
    assert len(retry_prompts) == 5
    assert all("Copy " not in prompt for prompt in retry_prompts)
    assert all(
        "only failed_checks, risk_case_ids, and feedback" in prompt
        for prompt in retry_prompts
    )
    assert report["summary"]["provider_truncation_count"] == 0


def test_calibration_reports_provider_truncation_separately(tmp_path):
    report = asyncio.run(run(
        Config.from_flat(out_dir=str(tmp_path)),
        FakeCalibrationClient(truncate=True),
    ))
    assert report["ok"] is False
    assert report["criteria"]["provider_truncation_zero"] is False
    assert report["summary"]["provider_truncation_count"] == 10
    assert all(
        attempt["finish_reason"] == "length"
        and attempt["response_truncated"] is True
        and attempt["prompt_tokens"] == 2
        and attempt["completion_tokens"] == 3
        for item in report["items"]
        for attempt in item["attempts"]
    )
