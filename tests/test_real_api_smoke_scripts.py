import asyncio
import json

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.llm_client import LLMCallResult
from scripts.real_api_role_transport_smoke import run


class FakeTransportClient:
    def __init__(self, invalid_teacher=False):
        self.invalid_teacher = invalid_teacher
        self.roles = []

    async def chat_result(self, _model, _system, _user, _temperature, _max_tokens, role):
        self.roles.append(role)
        return LLMCallResult("Reason\nFINAL_ANSWER: A", 1, 1, 2, 0.01)

    async def chat(self, _model, system, _user, _temperature, _max_tokens, role):
        self.roles.append(role)
        if system == "Return strict JSON only.":
            return json.dumps({"candidates": [{
                "candidate_prompt": "Verify each option before selecting one letter.",
                "target_failure_mechanism": "premature selection",
                "repair_procedure": "compare all options",
                "preservation_rule": "retain verified answers",
                "expected_effect": "fewer selection errors",
            }]})
        if "Audit the Teacher" in system:
            return json.dumps({
                "approved": False,
                "score": 0.5,
                "feedback": "transport-valid rejection",
                "rejection_reasons": ["generic"],
            })
        if self.invalid_teacher:
            return "not-json"
        return json.dumps({
            "target_failure_mechanism": "premature selection",
            "repair_procedure": "compare all options",
            "preservation_rule": "retain verified answers",
            "expected_effect": "fewer selection errors",
        })

    def cost_summary(self):
        return {"total_llm_calls": len(self.roles)}


def test_role_transport_smoke_validates_all_roles_without_requiring_critic_approval(tmp_path):
    client = FakeTransportClient()
    report = asyncio.run(run(
        Config.from_flat(
            out_dir=str(tmp_path),
            answer_format="option_letter",
            num_candidates_per_parent=1,
        ),
        client,
    ))
    assert report["ok"] is True
    assert report["critic"]["decision"]["approved"] is False
    assert report["student"]["schema_valid_count"] == 1
    assert client.roles == ["solver", "optimizer", "evaluator", "optimizer"]
    assert (tmp_path / "role_transport_smoke.json").is_file()


def test_role_transport_still_calls_critic_and_student_when_live_teacher_schema_fails(tmp_path):
    client = FakeTransportClient(invalid_teacher=True)
    report = asyncio.run(run(
        Config.from_flat(
            out_dir=str(tmp_path),
            answer_format="option_letter",
            num_candidates_per_parent=1,
        ),
        client,
    ))
    assert report["ok"] is False
    assert report["teacher"]["schema_valid"] is False
    assert report["critic"]["teacher_input_source"] == "fixed_transport_fixture"
    assert report["critic"]["schema_valid"] is True
    assert report["student"]["schema_valid"] is True
    assert client.roles == ["solver", "optimizer", "evaluator", "optimizer"]
