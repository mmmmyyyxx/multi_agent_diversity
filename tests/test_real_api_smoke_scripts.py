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
                "observed_failure_pattern": "premature selection",
                "generalizable_mechanism": "options are not compared",
                "decision_rule": "compare all options",
                "uncertainty_or_abstention_rule": "retain viable options when unresolved",
                "preservation_conditions": "retain verified answers",
                "evidence_summary": "selection happens before verification",
            }]})
        if "Audit the Teacher" in system:
            facts = json.loads(
                system.split("DERIVED_CASE_FACTS:\n", 1)[1].split(
                    "\nProposalContext:", 1,
                )[0]
            )
            return json.dumps({
                "case_fact_restatements": facts,
                "context_consistent": True,
                "sample_memorization_free": True,
                "executable_change": False,
                "internally_consistent": True,
                "preservation_rule_present": True,
                "output_contract_safe": True,
                "peer_copying_free": True,
                "stereotype_forcing_free": True,
                "non_generic_change": False,
                "blocking_reasons": ["generic"],
                "soft_concerns": [],
                "score": 0.5,
                "feedback": "transport-valid rejection",
            })
        if self.invalid_teacher:
            return "not-json"
        return json.dumps({
            "observed_failure_pattern": "premature selection",
            "generalizable_mechanism": "options are not compared",
            "decision_rule": "compare all options",
            "uncertainty_or_abstention_rule": "retain viable options when unresolved",
            "preservation_conditions": "retain verified answers",
            "evidence_summary": "selection happens before verification",
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
