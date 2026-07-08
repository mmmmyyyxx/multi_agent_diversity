import asyncio
import json

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def _system_without_init(cfg=None, agents=3):
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = cfg or Config(agents=agents, reward_mode="guarded_diversity")
    system.agents = [AgentState(f"agent {i} prompt") for i in range(agents)]
    system.recent_window_records = []
    return system


def _window_record_with_target_error():
    return {
        "question_hash": "q-secret-hash",
        "traces": [
            "I quickly choose. FINAL_ANSWER: A",
            "I compare constraints, eliminate the tempting option, verify consistency, and answer. FINAL_ANSWER: B",
            "I check assumptions and give one answer. FINAL_ANSWER: C",
        ],
        "answers": ["A", "B", "C"],
        "prompts": ["p0", "p1", "p2"],
        "metrics": {
            "individual_correct": [0, 1, 0],
            "vote_correct": 1,
            "vote_answer": "B",
            "invalid_flags": [0, 0, 0],
            "mean_embedding_overlap": 0.0,
            "per_agent_overlap": [0.0, 0.0, 0.0],
        },
        "homogeneous_cases": [],
        "validity_cases": [],
    }


def test_target_error_generation_batch_created():
    system = _system_without_init()
    diagnosis = system._window_overlap_diagnosis([_window_record_with_target_error()])

    batches = system._build_case_generation_batches(agent_id=0, diagnosis=diagnosis)

    target_batches = [b for b in batches if b["batch_type"] == "target_error_repair"]
    assert target_batches
    assert target_batches[0]["cases"][0]["case_type"] == "target_agent_wrong_and_peer_correct"
    assert target_batches[0]["cases"][0]["error_pattern"]


def test_optimizer_prompt_prioritizes_accuracy_repair():
    system = _system_without_init(Config(optimizer_architecture="one_shot"))
    captured = {}

    async def fake_chat(**kwargs):
        captured.update(kwargs)
        return json.dumps({"candidates": []})

    system._chat = fake_chat
    batch = {
        "batch_type": "target_error_repair",
        "cases": [
            {
                "case_id": "case-1",
                "target_agent_id": 0,
                "case_type": "target_agent_wrong_and_peer_correct",
                "target_trace_preview": "short reasoning FINAL_ANSWER: [redacted]",
                "target_answer_preview": {"present": True, "kind": "option_like", "length": 1},
                "peer_behavior_summary": {"num_peer_correct": 1},
                "error_pattern": "premature_answer",
                "repair_hint": "delay final answer until verification",
            }
        ],
    }

    asyncio.run(
        system.propose_candidates(
            agent_id=0,
            parent_prompt="You are a careful solver.",
            overlap_diagnosis={"prompt_roles": [], "per_agent_overlap_pressure": [0.0], "per_agent_invalid_rate": [0.0]},
            num_candidates=1,
            generation_batches=[batch],
        )
    )

    system_prompt = captured["system_prompt"].lower()
    user_prompt = captured["user_prompt"].lower()
    assert "target agent" in system_prompt
    assert "error patterns" in system_prompt
    assert "answer accuracy" in system_prompt
    assert "useful reasoning diversity" in system_prompt
    assert "repair the target agent's observed error patterns" in user_prompt
    assert "do not optimize for trace difference alone" in user_prompt
    assert "reduce full-trace embedding overlap" not in system_prompt


def test_candidate_schema_keeps_accuracy_repair_fields():
    system = _system_without_init(Config(optimizer_architecture="one_shot"))

    async def fake_chat(**kwargs):
        return json.dumps(
            {
                "candidates": [
                    {
                        "candidate_prompt": "List constraints, compare alternatives, verify the selected answer, then output exactly one final answer.",
                        "role_name": "constraint_repairer",
                        "decision_procedure": ["list constraints", "compare alternatives", "verify"],
                        "when_to_use": "missed constraints",
                        "fallback_strategy": "direct verification",
                        "anti_overlap_rule": "repair first",
                        "validity_checks": ["one final answer"],
                        "target_error_pattern": "missed_constraint",
                        "accuracy_repair_rule": "list constraints before selecting",
                        "expected_accuracy_effect": "prevents selecting answers that violate qualifiers",
                        "source_batch_type": "target_error_repair",
                    }
                ]
            }
        )

    system._chat = fake_chat
    candidates = asyncio.run(
        system.propose_candidates(
            agent_id=0,
            parent_prompt="You are a careful solver.",
            overlap_diagnosis={"prompt_roles": [], "per_agent_overlap_pressure": [0.0], "per_agent_invalid_rate": [0.0]},
            num_candidates=1,
            generation_batches=[{"batch_type": "target_error_repair", "cases": [{"case_id": "c1"}]}],
        )
    )

    assert candidates[0]["target_error_pattern"] == "missed_constraint"
    assert candidates[0]["accuracy_repair_rule"] == "list constraints before selecting"
    assert candidates[0]["expected_accuracy_effect"]
    assert candidates[0]["generation_batch_type"] == "target_error_repair"


def test_accuracy_repair_fallback_used_when_optimizer_returns_too_few():
    system = _system_without_init(Config(optimizer_architecture="one_shot", optimizer_fallback_mode="template"))

    async def fake_chat(**kwargs):
        return json.dumps({"candidates": []})

    system._chat = fake_chat
    candidates = asyncio.run(
        system.propose_candidates(
            agent_id=0,
            parent_prompt="You are a careful solver.",
            overlap_diagnosis={"prompt_roles": [], "per_agent_overlap_pressure": [0.0], "per_agent_invalid_rate": [0.0]},
            num_candidates=2,
            generation_batches=[{"batch_type": "target_error_repair", "cases": [{"case_id": "c1"}]}],
        )
    )

    assert len(candidates) == 2
    assert all("accuracy_repair" in c["candidate_source"] for c in candidates)
    assert candidates[0]["role_name"] in {
        "constraint_verifier",
        "option_elimination_specialist",
        "reverse_answer_validator",
        "format_and_answer_auditor",
    }
    assert candidates[0]["accuracy_repair_rule"]


def test_no_gold_leakage_in_optimizer_payload():
    system = _system_without_init()
    record = _window_record_with_target_error()
    record["question"] = "Which private option text should never be copied?"
    record["gold"] = "B"
    diagnosis = system._window_overlap_diagnosis([record])
    batch = system._build_case_generation_batches(agent_id=0, diagnosis=diagnosis)[0]
    payloads = [system._optimizer_case_payload(c) for c in batch["cases"]]
    text = json.dumps(payloads, ensure_ascii=False)

    assert "gold" not in text.lower()
    assert "Which private option text" not in text
    assert '"target_answer":' not in text
    assert '"team_vote_answer":' not in text
    assert "FINAL_ANSWER: A" not in text
    assert '"kind": "option_like"' in text
