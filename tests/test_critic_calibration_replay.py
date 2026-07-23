import asyncio
import json

from multi_dataset_diverse_rl.config import Config
from scripts.critic_calibration_replay import run


class FakeCalibrationClient:
    def __init__(self):
        self.roles = []

    async def chat(self, _model, system, _user, _temperature, _max_tokens, role):
        self.roles.append(role)
        facts = json.loads(
            system.split("DERIVED_CASE_FACTS:\n", 1)[1].split(
                "\nProposalContext:", 1,
            )[0]
        )
        checks = {
            "context_consistent": True,
            "sample_memorization_free": True,
            "executable_change": True,
            "internally_consistent": True,
            "preservation_rule_present": True,
            "output_contract_safe": True,
            "peer_copying_free": True,
            "stereotype_forcing_free": True,
            "non_generic_change": True,
        }
        blocking = []
        if "select answer A" in system:
            checks["sample_memorization_free"] = False
            blocking.append("memorizes a supplied question and answer")
        elif "target already answers the error case correctly" in system.lower():
            checks["context_consistent"] = False
            blocking.append("contradicts the derived current state")
        elif "Think carefully and reason step by step" in system:
            checks["executable_change"] = False
            checks["non_generic_change"] = False
            blocking.append("generic instruction only")
        elif "No preservation condition is needed" in system:
            checks["preservation_rule_present"] = False
            blocking.append("no operational preservation condition")
        elif "Always select the antecedent whose occupation" in system:
            checks["stereotype_forcing_free"] = False
            blocking.append("forces a stereotype despite uncertainty")
        return json.dumps({
            "case_fact_restatements": facts,
            **checks,
            "blocking_reasons": blocking,
            "soft_concerns": ["benefit remains unverified"] if not blocking else [],
            "score": 0.4,
            "feedback": "calibration audit",
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
    assert report["summary"]["good_acceptance_count"] == 2
    assert report["summary"]["memorizing_rejection_count"] == 1
    assert report["summary"]["classification_correct_count"] == 7
    assert client.roles == ["evaluator"] * 7
    assert (tmp_path / "critic_calibration_report.json").is_file()
