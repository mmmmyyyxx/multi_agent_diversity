import asyncio
import json

from multi_dataset_diverse_rl.config import Config
from scripts.critic_calibration_replay import run


class FakeCalibrationClient:
    def __init__(self):
        self.roles = []

    async def chat(self, _model, system, _user, _temperature, _max_tokens, role):
        self.roles.append(role)
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
        return json.dumps({
            "failed_checks": failed,
            "risk_case_ids": [],
            "feedback": feedback,
        })

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
