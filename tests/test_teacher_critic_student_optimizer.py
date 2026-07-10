import asyncio
import json

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem
from multi_dataset_diverse_rl.tasks import get_task_spec


def _system(cfg=None):
    cfg = cfg or Config(optimizer_architecture="teacher_critic_student", optimizer_fallback_mode="none")
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = cfg
    system.task_spec = get_task_spec("bbh")
    system.agents = [AgentState("shared prompt") for _ in range(int(cfg.agents))]
    system.update_logs = []
    system.recent_window_records = []
    system.optimizer_generation_diagnostics = {}
    system.no_effective_evolution_counter = 0
    system.no_effective_evolution_stopped = False
    system.no_effective_evolution_reason = ""
    return system


def _diagnosis():
    return {
        "prompt_roles": [
            {"agent_id": 0, "prompt_preview": "shared prompt"},
            {"agent_id": 1, "prompt_preview": "peer prompt"},
        ],
        "per_agent_overlap_pressure": [0.7, 0.2],
        "per_agent_invalid_rate": [0.0, 0.0],
        "homogeneous_case_counts": [2, 0],
        "mean_window_overlap": 0.6,
    }


def _batch():
    return [
        {
            "batch_type": "target_error_repair",
            "purpose": "repair abstract target errors",
            "cases": [
                {
                    "case_id": "c1",
                    "case_type": "target_agent_wrong_and_peer_correct",
                    "target_agent_id": 0,
                    "target_correct": False,
                    "peer_correct_available": True,
                    "target_trace_preview": "trace must not leak full question text",
                    "target_answer": "A",
                    "gold": "B",
                    "question": "FULL PRIVATE QUESTION TEXT",
                    "repair_hint": "check stated constraints before committing",
                }
            ],
        }
    ]


def _valid_student_candidate():
    return {
        "candidate_prompt": "Use an abstract constraint audit, compare alternatives, then provide one final answer.",
        "student_interpretation_of_question": "add constraint audit",
        "target_error_pattern": "missed constraint",
        "accuracy_repair_rule": "audit constraints before answer",
        "diversity_contribution": "constraint-first route",
        "error_correlation_reduction": "breaks shared omission pattern",
        "task_alignment_rule": "respect answer format",
        "peer_redundancy_avoidance": "avoid peer wording",
        "expected_accuracy_effect": "fewer constraint mistakes",
        "expected_diversity_effect": "different valid route",
        "risk_control": "stay concise",
        "rationale": "grounded in diagnostics",
    }


def test_teacher_question_rejected_uses_best_scored_teacher_question():
    system = _system(Config(optimizer_architecture="teacher_critic_student", teacher_critic_max_rounds=3))
    called = {"student": 0}

    async def fake_approved(**kwargs):
        return {
            "approved": True,
            "teacher_question": {"socratic_guiding_question": "Best available below-threshold question"},
            "critic_reviews": [
                {"score": 0.2, "passed": False, "quality_critique": "generic"},
                {"score": 0.68, "passed": False, "quality_critique": "best but below threshold"},
                {"score": 0.4, "passed": False, "quality_critique": "worse"},
            ],
            "teacher_critic_rounds": 3,
            "teacher_rewrite_count": 2,
            "teacher_question_forced_best_score": True,
            "teacher_question_forced_best_round": 2,
            "teacher_question_forced_below_threshold": True,
            "teacher_question_forced_best_review": {"score": 0.68, "passed": False, "quality_critique": "best but below threshold"},
        }

    async def fake_student(**kwargs):
        called["student"] += 1
        assert kwargs["approved_teacher_question"]["teacher_question"]["socratic_guiding_question"] == "Best available below-threshold question"
        return [_valid_student_candidate()]

    system.generate_approved_teacher_question = fake_approved
    system.generate_student_candidates = fake_student
    candidates = asyncio.run(
        system.propose_candidates_teacher_critic_student(
            agent_id=0,
            parent_prompt="parent",
            overlap_diagnosis=_diagnosis(),
            num_candidates=2,
            generation_batches=_batch(),
        )
    )

    diagnostics = system._optimizer_generation_diagnostics_for_parent(0, "parent")
    assert len(candidates) == 1
    assert called["student"] == 1
    assert diagnostics["teacher_question_rejected"] is False
    assert diagnostics["teacher_question_approved"] is True
    assert diagnostics["teacher_question_forced_best_score"] is True
    assert diagnostics["teacher_question_forced_best_round"] == 2
    assert diagnostics["teacher_question_forced_below_threshold"] is True
    assert diagnostics["teacher_question_score"] == 0.68


def test_teacher_question_rewrite_then_pass():
    system = _system(Config(optimizer_architecture="teacher_critic_student", teacher_critic_max_rounds=2))
    calls = {"teacher": 0, "critic": 0, "rewrite": 0}

    async def fake_teacher(*args, **kwargs):
        calls["teacher"] += 1
        return {"socratic_guiding_question": "generic first question"}

    async def fake_critic(*args, **kwargs):
        calls["critic"] += 1
        if calls["critic"] == 1:
            return {"passed": False, "score": 0.2, "rewrite_instruction": "make it specific"}
        return {"passed": True, "score": 0.9, "quality_critique": "specific"}

    async def fake_rewrite(*args, **kwargs):
        calls["rewrite"] += 1
        return {"socratic_guiding_question": "Which abstract constraint-checking behavior should the prompt enforce?"}

    system.propose_teacher_question = fake_teacher
    system.critique_teacher_question = fake_critic
    system.rewrite_teacher_question = fake_rewrite

    approved = asyncio.run(system.generate_approved_teacher_question(0, "parent", {"diagnostic_focus": {}}, 2))

    assert approved["approved"] is True
    assert calls == {"teacher": 1, "critic": 2, "rewrite": 1}
    assert approved["teacher_rewrite_count"] == 1


def test_teacher_question_three_rounds_fallback_to_best_score():
    system = _system(Config(optimizer_architecture="teacher_critic_student", teacher_critic_max_rounds=3, teacher_question_pass_threshold=0.75))
    calls = {"teacher": 0, "critic": 0, "rewrite": 0}

    async def fake_teacher(*args, **kwargs):
        calls["teacher"] += 1
        return {"socratic_guiding_question": "q1"}

    async def fake_critic(*args, **kwargs):
        calls["critic"] += 1
        scores = [0.2, 0.68, 0.4]
        return {"passed": False, "score": scores[calls["critic"] - 1], "rewrite_instruction": f"rewrite {calls['critic']}"}

    async def fake_rewrite(*args, **kwargs):
        calls["rewrite"] += 1
        return {"socratic_guiding_question": f"q{calls['rewrite'] + 1}"}

    system.propose_teacher_question = fake_teacher
    system.critique_teacher_question = fake_critic
    system.rewrite_teacher_question = fake_rewrite

    approved = asyncio.run(system.generate_approved_teacher_question(0, "parent", {"diagnostic_focus": {}}, 2))

    assert approved["approved"] is True
    assert approved["teacher_question"]["socratic_guiding_question"] == "q2"
    assert approved["teacher_question_forced_best_score"] is True
    assert approved["teacher_question_forced_best_round"] == 2
    assert approved["teacher_question_forced_below_threshold"] is True
    assert approved["teacher_critic_rounds"] == 3
    assert approved["teacher_rewrite_count"] == 2
    assert calls == {"teacher": 1, "critic": 3, "rewrite": 2}


def test_student_candidates_include_teacher_metadata():
    system = _system()

    async def fake_approved(**kwargs):
        return {
            "approved": True,
            "teacher_question": {"socratic_guiding_question": "Which abstract verification procedure should be added?"},
            "critic_reviews": [{"passed": True, "score": 0.88, "quality_critique": "good"}],
            "teacher_critic_rounds": 1,
            "teacher_rewrite_count": 0,
        }

    async def fake_student(**kwargs):
        return [_valid_student_candidate()]

    system.generate_approved_teacher_question = fake_approved
    system.generate_student_candidates = fake_student
    candidates = asyncio.run(
        system.propose_candidates_teacher_critic_student(
            agent_id=0,
            parent_prompt="parent",
            overlap_diagnosis=_diagnosis(),
            num_candidates=1,
            generation_batches=_batch(),
        )
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["candidate_source"] == "teacher_critic_student"
    assert cand["teacher_question_approved"] is True
    assert cand["teacher_question_score"] == 0.88
    assert cand["teacher_critic_rounds"] == 1
    assert cand["diversity_contribution"] == "constraint-first route"
    assert cand["error_correlation_reduction"] == "breaks shared omission pattern"
    assert cand["task_alignment_rule"] == "respect answer format"
    assert cand["peer_redundancy_avoidance"] == "avoid peer wording"


def test_no_gold_leakage_in_teacher_context():
    system = _system()
    context = system._build_teacher_context(
        agent_id=0,
        parent_prompt="parent prompt",
        target_role_spec={"agent_id": 0},
        peer_role_specs=[],
        window_stats={"target_overlap_pressure": 0.5, "mean_window_overlap": 0.6},
        validity_constraints={"do_not_copy_case_content": True},
        generation_batches=_batch(),
    )
    text = json.dumps(context, ensure_ascii=False)

    assert "FULL PRIVATE QUESTION TEXT" not in text
    assert '"gold"' not in text.lower()
    assert '"target_answer"' not in text
    assert '"B"' not in text


def test_one_shot_optimizer_backward_compatible():
    system = _system(Config(optimizer_architecture="one_shot", optimizer_fallback_mode="none"))

    async def fake_chat(**kwargs):
        return json.dumps(
            {
                "candidates": [
                    {
                        "candidate_prompt": "Check abstract constraints, compare alternatives, and answer once.",
                        "target_error_pattern": "missed_constraint",
                        "accuracy_repair_rule": "list constraints first",
                        "expected_accuracy_effect": "fewer misses",
                    }
                ]
            }
        )

    system._chat = fake_chat
    candidates = asyncio.run(
        system.propose_candidates(
            agent_id=0,
            parent_prompt="parent",
            overlap_diagnosis=_diagnosis(),
            num_candidates=1,
            generation_batches=_batch(),
        )
    )

    assert len(candidates) == 1
    assert candidates[0]["candidate_source"] == "optimizer"
    assert candidates[0]["accuracy_repair_rule"] == "list constraints first"


def test_teacher_critic_student_no_template_fallback():
    system = _system(Config(optimizer_architecture="teacher_critic_student", optimizer_fallback_mode="template"))

    async def fake_approved(**kwargs):
        return {
            "approved": True,
            "teacher_question": {"socratic_guiding_question": "Which abstract repair should be used?"},
            "critic_reviews": [{"passed": True, "score": 0.9}],
            "teacher_critic_rounds": 1,
            "teacher_rewrite_count": 0,
        }

    async def fake_student(**kwargs):
        return []

    system.generate_approved_teacher_question = fake_approved
    system.generate_student_candidates = fake_student
    candidates = asyncio.run(
        system.propose_candidates_teacher_critic_student(
            agent_id=0,
            parent_prompt="parent",
            overlap_diagnosis=_diagnosis(),
            num_candidates=2,
            generation_batches=_batch(),
        )
    )

    diagnostics = system._optimizer_generation_diagnostics_for_parent(0, "parent")
    assert candidates == []
    assert diagnostics["student_candidate_count_raw"] == 0
    assert diagnostics["optimizer_final_candidate_count"] == 0


def test_teacher_critic_student_rejects_missing_student_fields():
    system = _system()

    incomplete = {
        "candidate_prompt": "Check constraints and answer once.",
        "target_error_pattern": "missed_constraint",
    }

    missing = system._missing_optimizer_fields(incomplete, architecture="teacher_critic_student")

    assert "student_interpretation_of_question" in missing
    assert "diversity_contribution" in missing
    assert "error_correlation_reduction" in missing
    assert "task_alignment_rule" in missing
    assert "peer_redundancy_avoidance" in missing
    assert "risk_control" in missing
    assert not system._candidate_has_required_optimizer_fields(incomplete, architecture="teacher_critic_student")


def test_one_shot_schema_remains_prompt_only():
    system = _system(Config(optimizer_architecture="one_shot", optimizer_fallback_mode="none"))

    item = {"candidate_prompt": "Check constraints and answer once."}

    assert system._candidate_has_required_optimizer_fields(item, architecture="one_shot")
    assert system._missing_optimizer_fields(item, architecture="one_shot") == []


def test_student_missing_fields_filtered_with_diagnostics():
    system = _system()

    async def fake_approved(**kwargs):
        return {
            "approved": True,
            "teacher_question": {"socratic_guiding_question": "Which abstract verification should be enforced?"},
            "critic_reviews": [{"passed": True, "score": 0.9}],
            "teacher_critic_rounds": 1,
            "teacher_rewrite_count": 0,
        }

    async def fake_student(**kwargs):
        return [
            {
                "candidate_prompt": "Check constraints and answer once.",
                "target_error_pattern": "missed_constraint",
            }
        ]

    system.generate_approved_teacher_question = fake_approved
    system.generate_student_candidates = fake_student

    candidates = asyncio.run(
        system.propose_candidates_teacher_critic_student(
            agent_id=0,
            parent_prompt="parent",
            overlap_diagnosis=_diagnosis(),
            num_candidates=1,
            generation_batches=_batch(),
        )
    )

    diagnostics = system._optimizer_generation_diagnostics_for_parent(0, "parent")

    assert candidates == []
    assert diagnostics["student_candidate_count_raw"] == 1
    assert diagnostics["student_candidate_count_final"] == 0
    assert diagnostics["student_all_candidates_filtered"] is True
    assert diagnostics["optimizer_schema_filtered_count"] >= 1
    assert "diversity_contribution" in diagnostics["student_missing_required_fields"]


def test_safe_float_parses_critic_score_strings():
    system = _system()

    assert system._safe_float("0.82/1", 0.0) == 0.82
    assert system._safe_float("score: 0.75", 0.0) == 0.75
    assert system._safe_float("0.91 (pass)", 0.0) == 0.91
    assert system._safe_float(None, 0.3) == 0.3
    assert system._safe_float("high", 0.0) == 0.0


def test_teacher_question_rewrite_passes_with_string_score():
    system = _system(Config(optimizer_architecture="teacher_critic_student", teacher_critic_max_rounds=2))
    calls = {"critic": 0, "rewrite": 0}

    async def fake_teacher(*args, **kwargs):
        return {"socratic_guiding_question": "generic"}

    async def fake_critic(*args, **kwargs):
        calls["critic"] += 1
        if calls["critic"] == 1:
            return {"passed": False, "score": "0.2/1", "rewrite_instruction": "make it specific"}
        return {"passed": True, "score": "score: 0.88", "quality_critique": "specific"}

    async def fake_rewrite(*args, **kwargs):
        calls["rewrite"] += 1
        return {"socratic_guiding_question": "Which abstract constraint verification should the Student enforce?"}

    system.propose_teacher_question = fake_teacher
    system.critique_teacher_question = fake_critic
    system.rewrite_teacher_question = fake_rewrite

    approved = asyncio.run(system.generate_approved_teacher_question(0, "parent", {"diagnostic_focus": {}}, 1))

    assert approved["approved"] is True
    assert calls["rewrite"] == 1


def test_student_empty_response_diagnostics():
    system = _system(Config(student_json_retry_on_parse_fail=False))

    async def fake_chat(**kwargs):
        return ""

    system._chat = fake_chat
    result = asyncio.run(system.generate_student_candidates(0, "parent", {}, {}, 1))
    diag = result["diagnostics"]

    assert result["candidates"] == []
    assert diag["student_raw_response_empty"] is True
    assert diag["student_failure_stage"] == "raw_empty"


def test_student_empty_response_retries_and_succeeds():
    system = _system(Config(student_json_retry_on_parse_fail=True, student_json_max_retries=5))
    calls = {"count": 0}

    async def fake_chat(**kwargs):
        calls["count"] += 1
        if "student_json_retry" in kwargs.get("stage", ""):
            return json.dumps({"candidates": [_valid_student_candidate()]})
        return ""

    system._chat = fake_chat
    result = asyncio.run(system.generate_student_candidates(0, "parent", {}, {}, 1))
    diag = result["diagnostics"]

    assert len(result["candidates"]) == 1
    assert calls["count"] == 2
    assert diag["student_json_retry_attempted"] is True
    assert diag["student_json_retry_succeeded"] is True
    assert diag["student_json_repair_attempted"] is False
    assert diag["student_raw_response_empty"] is False
    assert diag["student_failure_stage"] in {"", "none"}


def test_student_non_json_response_diagnostics():
    system = _system(Config(
        student_json_retry_on_parse_fail=False,
        student_json_repair_enabled=False,
    ))

    async def fake_chat(**kwargs):
        return "I will create better prompts, but here is an explanation first..."

    system._chat = fake_chat
    result = asyncio.run(system.generate_student_candidates(0, "parent", {}, {}, 1))
    diag = result["diagnostics"]

    assert result["candidates"] == []
    assert diag["student_json_parse_failed"] is True
    assert diag["student_refusal_or_explanation"] is True
    assert diag["student_failure_stage"] in {"json_parse_failed", "refusal_or_explanation"}


def test_student_json_missing_candidates_key_diagnostics():
    system = _system()

    async def fake_chat(**kwargs):
        return json.dumps({"notes": "no candidates here"})

    system._chat = fake_chat
    result = asyncio.run(system.generate_student_candidates(0, "parent", {}, {}, 1))
    diag = result["diagnostics"]

    assert result["candidates"] == []
    assert diag["student_json_has_candidates_key"] is False
    assert diag["student_failure_stage"] == "missing_candidates_key"


def test_student_candidates_not_list_diagnostics():
    system = _system()

    async def fake_chat(**kwargs):
        return json.dumps({"candidates": {"candidate_prompt": "bad"}})

    system._chat = fake_chat
    result = asyncio.run(system.generate_student_candidates(0, "parent", {}, {}, 1))
    diag = result["diagnostics"]

    assert result["candidates"] == []
    assert diag["student_json_has_candidates_key"] is True
    assert diag["student_candidates_is_list"] is False
    assert diag["student_failure_stage"] == "candidates_not_list"


def test_student_candidates_empty_list_diagnostics():
    system = _system()

    async def fake_chat(**kwargs):
        return json.dumps({"candidates": []})

    system._chat = fake_chat
    result = asyncio.run(system.generate_student_candidates(0, "parent", {}, {}, 1))
    diag = result["diagnostics"]

    assert result["candidates"] == []
    assert diag["student_candidates_is_list"] is True
    assert diag["student_candidates_empty_list"] is True
    assert diag["student_failure_stage"] == "empty_candidates_list"


def test_student_all_candidates_filtered_by_schema_failure_stage():
    system = _system()

    async def fake_approved(**kwargs):
        return {
            "approved": True,
            "teacher_question": {"socratic_guiding_question": "Which abstract verification should be enforced?"},
            "critic_reviews": [{"passed": True, "score": 0.9}],
            "teacher_critic_rounds": 1,
            "teacher_rewrite_count": 0,
        }

    async def fake_student(**kwargs):
        return {
            "candidates": [
                {
                    "candidate_prompt": "Check constraints and answer once.",
                    "target_error_pattern": "missed_constraint",
                }
            ],
            "diagnostics": {
                "student_raw_response_empty": False,
                "student_raw_response_preview": '{"candidates": [...]}',
                "student_json_parse_failed": False,
                "student_json_parse_error": "",
                "student_json_has_candidates_key": True,
                "student_candidates_is_list": True,
                "student_candidates_empty_list": False,
                "student_refusal_or_explanation": False,
                "student_failure_stage": "none",
            },
        }

    system.generate_approved_teacher_question = fake_approved
    system.generate_student_candidates = fake_student
    candidates = asyncio.run(
        system.propose_candidates_teacher_critic_student(
            agent_id=0,
            parent_prompt="parent",
            overlap_diagnosis=_diagnosis(),
            num_candidates=1,
            generation_batches=_batch(),
        )
    )
    diagnostics = system._optimizer_generation_diagnostics_for_parent(0, "parent")

    assert candidates == []
    assert diagnostics["student_candidate_count_raw"] == 1
    assert diagnostics["student_candidate_count_final"] == 0
    assert diagnostics["student_all_candidates_filtered"] is True
    assert diagnostics["student_failure_stage"] == "all_candidates_filtered_schema"


def test_valid_student_candidate_has_no_failure_stage():
    system = _system()

    async def fake_approved(**kwargs):
        return {
            "approved": True,
            "teacher_question": {"socratic_guiding_question": "Which abstract verification should be enforced?"},
            "critic_reviews": [{"passed": True, "score": 0.9}],
            "teacher_critic_rounds": 1,
            "teacher_rewrite_count": 0,
        }

    async def fake_student(**kwargs):
        return {
            "candidates": [_valid_student_candidate()],
            "diagnostics": {
                "student_raw_response_empty": False,
                "student_raw_response_preview": '{"candidates": [...]}',
                "student_json_parse_failed": False,
                "student_json_parse_error": "",
                "student_json_has_candidates_key": True,
                "student_candidates_is_list": True,
                "student_candidates_empty_list": False,
                "student_refusal_or_explanation": False,
                "student_failure_stage": "none",
            },
        }

    system.generate_approved_teacher_question = fake_approved
    system.generate_student_candidates = fake_student
    candidates = asyncio.run(
        system.propose_candidates_teacher_critic_student(
            agent_id=0,
            parent_prompt="parent",
            overlap_diagnosis=_diagnosis(),
            num_candidates=1,
            generation_batches=_batch(),
        )
    )
    diagnostics = system._optimizer_generation_diagnostics_for_parent(0, "parent")

    assert len(candidates) == 1
    assert diagnostics["student_candidate_count_raw"] == 1
    assert diagnostics["student_candidate_count_final"] == 1
    assert diagnostics["student_failure_stage"] in {"", "none"}


def test_student_parse_failure_retry_succeeds():
    system = _system(Config(student_json_repair_enabled=True, student_json_max_retries=1))
    calls = []

    async def fake_chat(**kwargs):
        calls.append(kwargs.get("stage", ""))
        if "student_json_retry" in kwargs.get("stage", ""):
            return json.dumps({"candidates": [_valid_student_candidate()]})
        return '{"candidates": ['

    system._chat = fake_chat
    result = asyncio.run(system.generate_student_candidates(0, "parent", {"approved": True}, {}, 1))
    diag = result["diagnostics"]

    assert len(result["candidates"]) == 1
    assert diag["student_json_retry_attempted"] is True
    assert diag["student_json_retry_succeeded"] is True
    assert diag["student_json_repair_attempted"] is False
    assert diag["student_json_parse_failed"] is False
    assert diag["student_failure_stage"] in {"", "none"}
    assert any("student_json_retry" in stage for stage in calls)


def test_student_parse_failure_retry_fails_then_repair_succeeds():
    system = _system(Config(student_json_repair_enabled=True, student_json_max_retries=1))

    async def fake_chat(**kwargs):
        stage = kwargs.get("stage", "")
        if "student_json_repair" in stage:
            return json.dumps({"candidates": [_valid_student_candidate()]})
        return '{"candidates": [{"candidate_prompt": "broken"'

    system._chat = fake_chat
    result = asyncio.run(system.generate_student_candidates(0, "parent", {"approved": True}, {}, 1))
    diag = result["diagnostics"]

    assert len(result["candidates"]) == 1
    assert diag["student_json_retry_attempted"] is True
    assert diag["student_json_retry_succeeded"] is False
    assert diag["student_json_repair_attempted"] is True
    assert diag["student_json_repair_succeeded"] is True
    assert diag["student_json_parse_failed"] is False
    assert diag["student_failure_stage"] in {"", "none"}


def test_student_parse_failure_retry_and_repair_fail():
    system = _system(Config(student_json_repair_enabled=True, student_json_max_retries=1))

    async def fake_chat(**kwargs):
        return '{"candidates": ['

    system._chat = fake_chat
    result = asyncio.run(system.generate_student_candidates(0, "parent", {"approved": True}, {}, 1))
    diag = result["diagnostics"]

    assert result["candidates"] == []
    assert diag["student_json_retry_attempted"] is True
    assert diag["student_json_retry_succeeded"] is False
    assert diag["student_json_repair_attempted"] is True
    assert diag["student_json_repair_succeeded"] is False
    assert diag["student_json_parse_failed"] is True
    assert diag["student_failure_stage"] == "json_parse_failed"


def test_student_valid_empty_candidates_does_not_trigger_repair():
    system = _system(Config(student_json_repair_enabled=True, student_json_max_retries=1))

    async def fake_chat(**kwargs):
        return json.dumps({"candidates": []})

    system._chat = fake_chat
    result = asyncio.run(system.generate_student_candidates(0, "parent", {"approved": True}, {}, 1))
    diag = result["diagnostics"]

    assert result["candidates"] == []
    assert diag["student_json_repair_attempted"] is False
    assert diag["student_candidates_empty_list"] is True
    assert diag["student_failure_stage"] == "empty_candidates_list"


def test_compact_student_prompt_contains_minified_json_instructions():
    system = _system(Config(student_candidate_schema_mode="compact"))
    captured = {}

    async def fake_chat(**kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt", "")
        captured["user_prompt"] = kwargs.get("user_prompt", "")
        return json.dumps({"candidates": []})

    system._chat = fake_chat
    asyncio.run(system.generate_student_candidates(0, "parent", {"approved": True}, {}, 1))
    prompt = captured["system_prompt"] + "\n" + captured["user_prompt"]

    assert "Return minified JSON only" in prompt
    assert "Do not use Markdown" in prompt
    assert "Do not wrap the JSON in code fences" in prompt
    assert "Every other field must be <=" in prompt
    assert "candidate_prompt must be <=" in prompt


def test_truncate_candidate_text_fields():
    cfg = Config(student_candidate_prompt_max_chars=12, student_candidate_max_chars_per_field=8)
    system = _system(cfg)
    item = {
        "candidate_prompt": "x" * 40,
        "rationale": "y" * 40,
        "count": 3,
    }

    truncated = system._truncate_candidate_text_fields(item)

    assert truncated["candidate_prompt"] == "x" * 12
    assert truncated["rationale"] == "y" * 8
    assert truncated["count"] == 3
