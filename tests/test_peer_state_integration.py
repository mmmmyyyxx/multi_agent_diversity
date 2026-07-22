import asyncio
import json

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.evaluation.fixed_probe import candidate_probe_metrics


QUESTIONS = {"q0": "A", "q1": "A", "q2": "A"}


async def fake_solver(question, agent_id, prompt):
    if "repair-q0" in prompt and question == "q0":
        answer = "A"
    elif question == "q1" and agent_id in {0, 1, 2}:
        answer = "A"
    elif question == "q2" and agent_id in {0, 1}:
        answer = "A"
    else:
        answer = "B"
    return PromptAnswer(answer=answer, trace=f"check FINAL_ANSWER: {answer}", valid=True)


async def fake_optimizer(system_prompt, _user_prompt, _temperature, _max_tokens):
    if "Audit whether" in system_prompt:
        return json.dumps({"approved": True, "score": 1.0, "feedback": "approved"})
    if "strict JSON" in system_prompt:
        return json.dumps({"candidates": [{
            "candidate_prompt": "repair-q0",
            "target_failure_mechanism": "misses uncovered cases",
            "repair_procedure": "check ambiguity before committing",
            "preservation_rule": "retain existing correct decisions",
            "expected_responsibility_effect": "convert q0 to a gold vote",
        }]})
    return "When ambiguity remains, compare interpretations and verify the selected referent."


def test_full_fake_chain_accepts_and_refreshes(tmp_path):
    cfg = Config.from_flat(
        out_dir=str(tmp_path), answer_format="option_letter", num_candidates_per_parent=1,
        stage_a_channel_top_k=1, stage_b_candidate_budget=2,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=fake_solver, optimizer_chat=fake_optimizer)

    async def run():
        data = [{"question": question, "answer": gold} for question, gold in QUESTIONS.items()]
        await system.initialize_fixed_probe(data)
        before, _ = system.current_states_and_credits()
        changed = await system.update_once(0)
        after, _ = system.current_states_and_credits()
        return before, changed, after

    before, changed, after = asyncio.run(run())
    assert before[0].gold_vote_count == 0
    assert changed is True
    target_agent_id = system.candidate_decisions[-1]["target_agent_id"]
    assert system.agents[target_agent_id].current_prompt == "repair-q0"
    assert after[0].gold_vote_count == 1
    assert system.candidate_decisions[-1]["accepted_prompt_hash"]
    assert len(system.responsibility_assignments) == 2
    assert "assigned_credits" in system.responsibility_assignments[-1]
    assert system.fixed_probe.cache_misses > 0


def test_independent_accuracy_context_excludes_peer_state_credit(tmp_path):
    captured = []

    async def capture_optimizer(system_prompt, user_prompt, temperature, max_tokens):
        captured.append(system_prompt + user_prompt)
        return await fake_optimizer(system_prompt, user_prompt, temperature, max_tokens)

    cfg = Config.from_flat(
        out_dir=str(tmp_path), answer_format="option_letter", independent_accuracy_only=True,
        responsibility_assignment_enabled=False, responsibility_conditioned_tcs=False,
        target_selector="round_robin", num_candidates_per_parent=1,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=fake_solver, optimizer_chat=capture_optimizer)

    async def run():
        await system.initialize_fixed_probe([{"question": "q0", "answer": "A"}])
        await system.propose_candidates(0, set())

    asyncio.run(run())
    teacher_text = captured[0]
    assert "peer_answer_histogram" not in teacher_text
    assert "fix_soft_utility_gain" not in teacher_text
    assert "'G':" not in teacher_text


def test_stage_a_only_runs_subset_before_full_probe(tmp_path):
    calls = []

    async def counting_solver(question, agent_id, prompt):
        calls.append((question, agent_id, prompt))
        return await fake_solver(question, agent_id, prompt)

    cfg = Config.from_flat(
        out_dir=str(tmp_path), answer_format="option_letter",
        stage_a_representative_size=1, stage_a_coverage_size=0,
        stage_a_conversion_size=0, stage_a_preservation_size=0,
        stage_a_channel_top_k=1, stage_b_candidate_budget=1,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=counting_solver, optimizer_chat=fake_optimizer)

    async def run():
        data = [{"question": question, "answer": gold} for question, gold in QUESTIONS.items()]
        await system.initialize_fixed_probe(data)
        calls.clear()
        candidate = {
            "prompt": "repair-q0", "prompt_hash": system.prompt_hash("repair-q0"),
            "generation": 1,
        }
        await system.evaluate_candidates(0, [candidate], set())

    asyncio.run(run())
    candidate_calls = [row for row in calls if row[2] == "repair-q0"]
    # One Stage-A question, then only two missing questions are added for Stage B.
    assert len(candidate_calls) == 3


def test_c0_wrong_to_wrong_has_zero_candidate_utility(tmp_path):
    cfg = Config.from_flat(out_dir=str(tmp_path), answer_format="option_letter")
    system = PromptEnsembleOptimizationSystem(cfg, solver=fake_solver, optimizer_chat=fake_optimizer)

    async def run():
        await system.initialize_fixed_probe([{"question": "q0", "answer": "A"}])
        candidate = (PromptAnswer("C", "check FINAL_ANSWER: C", True),)
        return candidate_probe_metrics(
            examples=system.fixed_probe.examples,
            active_profiles=system.active_profiles,
            candidate_profile=candidate,
            target_agent_id=0,
            assigned_question_hashes=set(),
            normalize_answer=system.normalize_answer,
            match_answer=system.match_answer,
            tie_break=cfg.peer_state.vote_tie_break,
            seed=cfg.training.seed,
            tau=cfg.peer_state.soft_vote_tau,
        )

    metrics = asyncio.run(run())
    assert metrics["soft_vote_utility_delta"] == 0.0
    assert metrics["coverage_gain_count"] == 0


def test_unapproved_teacher_never_reaches_student(tmp_path):
    calls = []

    async def rejecting_optimizer(system_prompt, _user_prompt, _temperature, _max_tokens):
        calls.append(system_prompt)
        if "Audit whether" in system_prompt:
            return json.dumps({"approved": False, "score": 0.1, "feedback": "too generic"})
        if "strict JSON" in system_prompt:
            raise AssertionError("Student must not run after all critic rounds reject")
        return "Think step by step"

    cfg = Config.from_flat(out_dir=str(tmp_path), answer_format="option_letter", teacher_critic_max_rounds=3)
    system = PromptEnsembleOptimizationSystem(cfg, solver=fake_solver, optimizer_chat=rejecting_optimizer)

    async def run():
        await system.initialize_fixed_probe([{"question": "q0", "answer": "A"}])
        return await system.propose_candidates(0, set())

    assert asyncio.run(run()) == []
    assert sum("Audit whether" in call for call in calls) == 3
