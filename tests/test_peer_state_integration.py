import asyncio
import json

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer, evaluate_candidate_profile
from multi_dataset_diverse_rl.system import CandidateFunnel, CandidateRuntime, PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.tcs import StudentCandidate


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
    if "Audit the Teacher" in system_prompt:
        return json.dumps({
            "approved": True,
            "score": 1.0,
            "feedback": "approved",
            "rejection_reasons": [],
        })
    if system_prompt == "Return strict JSON only.":
        return json.dumps({"candidates": [{
            "candidate_prompt": "repair-q0",
            "target_failure_mechanism": "misses uncovered cases",
            "repair_procedure": "check ambiguity before committing",
            "preservation_rule": "retain existing correct decisions",
            "expected_effect": "convert q0 to a gold vote",
        }]})
    return json.dumps({
        "target_failure_mechanism": "misses uncovered cases",
        "repair_procedure": "check ambiguity before committing",
        "preservation_rule": "retain existing correct decisions",
        "expected_effect": "convert q0 to a gold vote",
    })


def student_candidate():
    return StudentCandidate(
        candidate_prompt="repair-q0",
        target_failure_mechanism="misses uncovered cases",
        repair_procedure="check ambiguity",
        preservation_rule="retain correct decisions",
        expected_effect="create a correct vote",
    )


def test_full_fake_chain_accepts_and_refreshes_online(tmp_path):
    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        answer_format="option_letter",
        num_candidates_per_parent=1,
        stage_a_channel_top_k=1,
        stage_b_candidate_budget=2,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=fake_solver, optimizer_chat=fake_optimizer)

    async def run():
        data = [{"question": question, "answer": gold} for question, gold in QUESTIONS.items()]
        await system.initialize_fixed_probe(data)
        before, _, _ = system.current_states_and_opportunities()
        changed = await system.update_once(0)
        after, _, _ = system.current_states_and_opportunities()
        return before, changed, after

    before, changed, after = asyncio.run(run())
    assert before[0].gold_vote_count == 0
    assert changed is True
    target_agent_id = system.candidate_decisions[-1]["target_agent_id"]
    assert system.agents[target_agent_id].current_prompt == "repair-q0"
    assert after[0].gold_vote_count == 1
    assert len(system.responsibility_assignments) == 2
    assert "assigned_opportunities" in system.responsibility_assignments[-1]
    assert system.candidate_decisions[-1]["funnel"]["accepted_candidate"] is True
    assert system.tcs_context_history[-1]["context_type"] == "ResponsibilityProposalContext"
    assert len(system.tcs_context_history[-1]["proposal_context_hash"]) == 64
    assert system.tcs_context_history[-1]["forbidden_field_violations"] == []
    assert system.tcs_context_history[-1]["responsibility_specific_field_count"] > 0
    assert [row["role"] for row in system.tcs_rounds] == ["teacher", "critic", "student"]
    assert all(row["schema_valid"] for row in system.tcs_rounds)
    assert system.tcs_rounds[1]["effective_approved"] is True
    assert system.tcs_rounds[2]["raw_count"] == 1


def test_b3_also_refreshes_responsibilities_online(tmp_path):
    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        answer_format="option_letter",
        experiment_setting="shared_peer_state_responsibility",
        num_candidates_per_parent=1,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=fake_solver, optimizer_chat=fake_optimizer)

    async def run():
        await system.initialize_fixed_probe([{"question": "q0", "answer": "A"}])
        return await system.update_once(0)

    assert asyncio.run(run()) is True
    assert len(system.responsibility_assignments) == 2
    assert system.tcs_context_history[-1]["context_type"] == "PeerStateProposalContext"
    assert system.tcs_context_history[-1]["forbidden_field_violations"] == []
    assert not any(
        "assigned" in path or "owner_age" in path or "responsibility" in path
        for path in system.tcs_context_history[-1]["serialized_recursive_field_paths"]
    )


def test_student_wrong_candidate_count_retries_before_stage_a(tmp_path):
    student_calls = 0

    async def count_retry_optimizer(system_prompt, user_prompt, temperature, max_tokens):
        nonlocal student_calls
        if system_prompt != "Return strict JSON only.":
            return await fake_optimizer(system_prompt, user_prompt, temperature, max_tokens)
        student_calls += 1
        count = 2 if student_calls == 1 else 1
        row = {
            "candidate_prompt": "repair-q0",
            "target_failure_mechanism": "misses uncovered cases",
            "repair_procedure": "check ambiguity before committing",
            "preservation_rule": "retain existing correct decisions",
            "expected_effect": "convert q0 to a gold vote",
        }
        return json.dumps({"candidates": [row] * count})

    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        answer_format="option_letter",
        num_candidates_per_parent=1,
    )
    system = PromptEnsembleOptimizationSystem(
        cfg, solver=fake_solver, optimizer_chat=count_retry_optimizer,
    )

    async def run():
        await system.initialize_fixed_probe([{"question": "q0", "answer": "A"}])
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert student_calls == 2
    assert funnel.student_calls == 2
    assert funnel.requested_candidate_count == 1
    assert funnel.raw_candidate_count == 1
    assert funnel.schema_valid_count == 1
    assert len(candidates) == 1


def test_stage_a_pool_budget_is_fixed_and_pools_are_disjoint(tmp_path):
    async def build(setting):
        cfg = Config.from_flat(
            out_dir=str(tmp_path / setting),
            answer_format="option_letter",
            experiment_setting=setting,
            stage_a_representative_size=2,
            stage_a_coverage_size=2,
            stage_a_conversion_size=2,
            stage_a_preservation_size=2,
        )
        system = PromptEnsembleOptimizationSystem(cfg, solver=fake_solver, optimizer_chat=fake_optimizer)
        data = [{"question": f"q{i}", "answer": "A"} for i in range(6)]
        await system.initialize_fixed_probe(data)
        pools = system._pool_indices(0, set())
        return pools

    async def run_all():
        return await asyncio.gather(
            build("shared_independent_accuracy_tcs"),
            build("shared_peer_state_credit_round_robin"),
            build("shared_peer_state_responsibility"),
        )

    b1, b2, b3 = asyncio.run(run_all())
    for pools in (b1, b2, b3):
        groups = [set(pools.coverage), set(pools.conversion), set(pools.preservation), set(pools.representative)]
        assert pools.final_unique_size == 6
        assert len(set().union(*groups)) == 6
        assert sum(len(left & right) for index, left in enumerate(groups) for right in groups[index + 1:]) == 0


def test_independent_accuracy_tcs_excludes_peer_state_fields(tmp_path):
    captured = []

    async def capture_optimizer(system_prompt, user_prompt, temperature, max_tokens):
        captured.append(system_prompt + user_prompt)
        return await fake_optimizer(system_prompt, user_prompt, temperature, max_tokens)

    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        answer_format="option_letter",
        experiment_setting="shared_independent_accuracy_tcs",
        num_candidates_per_parent=1,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=fake_solver, optimizer_chat=capture_optimizer)

    async def run():
        await system.initialize_fixed_probe([{"question": "q0", "answer": "A"}])
        await system.propose_candidates(0, set(), CandidateFunnel())

    asyncio.run(run())
    joined = "\n".join(captured)
    assert "peer_wrong_histogram" not in joined
    assert "oracle_soft_utility_gain" not in joined
    assert '"team_G"' not in joined
    audit = system.tcs_context_history[-1]
    assert audit["context_class"] == "AccuracyProposalContext"
    assert audit["forbidden_field_violations"] == []


def test_independent_accuracy_previous_summary_never_contains_vote_delta(tmp_path):
    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        answer_format="option_letter",
        experiment_setting="shared_independent_accuracy_tcs",
        num_candidates_per_parent=1,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=fake_solver, optimizer_chat=fake_optimizer)

    async def run():
        await system.initialize_fixed_probe([{"question": "q0", "answer": "A"}])
        return await system.update_once(0)

    assert asyncio.run(run()) is True
    summary = system.previous_accuracy_summaries[0].lower()
    assert "vote" not in summary
    assert "correct-count change" in summary


def test_independent_accuracy_tcs_does_not_leak_unique_correct_peer_context(tmp_path):
    captured = []

    async def unique_solver(_question, agent_id, _prompt):
        answer = "A" if agent_id == 0 else "B"
        return PromptAnswer(answer=answer, trace=f"check FINAL_ANSWER: {answer}", valid=True)

    async def capture_optimizer(system_prompt, user_prompt, temperature, max_tokens):
        captured.append(system_prompt + user_prompt)
        return await fake_optimizer(system_prompt, user_prompt, temperature, max_tokens)

    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        answer_format="option_letter",
        experiment_setting="shared_independent_accuracy_tcs",
        num_candidates_per_parent=1,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=unique_solver, optimizer_chat=capture_optimizer)

    async def run():
        await system.initialize_fixed_probe([{"question": "unique", "answer": "A"}])
        await system.propose_candidates(0, set(), CandidateFunnel())

    asyncio.run(run())
    joined = "\n".join(captured)
    assert "peer_wrong_histogram" not in joined
    assert '"unique_correct"' not in joined


def test_stage_a_runs_subset_before_full_probe(tmp_path):
    calls = []

    async def counting_solver(question, agent_id, prompt):
        calls.append((question, agent_id, prompt))
        return await fake_solver(question, agent_id, prompt)

    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        answer_format="option_letter",
        stage_a_representative_size=1,
        stage_a_coverage_size=0,
        stage_a_conversion_size=0,
        stage_a_preservation_size=0,
        stage_a_channel_top_k=1,
        stage_b_candidate_budget=1,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=counting_solver, optimizer_chat=fake_optimizer)

    async def run():
        data = [{"question": question, "answer": gold} for question, gold in QUESTIONS.items()]
        await system.initialize_fixed_probe(data)
        calls.clear()
        row = CandidateRuntime(
            student_candidate=student_candidate(),
            prompt="repair-q0",
            prompt_hash=system.prompt_hash("repair-q0"),
            generation=1,
            parent_prompt_hash="parent",
        )
        await system.evaluate_candidates(0, [row], set(), CandidateFunnel())

    asyncio.run(run())
    candidate_calls = [row for row in calls if row[2] == "repair-q0"]
    assert len(candidate_calls) == 3


def test_c0_wrong_to_wrong_has_zero_real_candidate_utility(tmp_path):
    cfg = Config.from_flat(out_dir=str(tmp_path), answer_format="option_letter")
    system = PromptEnsembleOptimizationSystem(cfg, solver=fake_solver, optimizer_chat=fake_optimizer)

    async def run():
        await system.initialize_fixed_probe([{"question": "q0", "answer": "A"}])
        return evaluate_candidate_profile(
            prompt="wrong",
            prompt_hash="wrong",
            examples=system.fixed_probe.examples,
            active_profiles=system.active_profiles,
            candidate_profile=(PromptAnswer("C", "check FINAL_ANSWER: C", True),),
            target_agent_id=0,
            assigned_question_hashes=set(),
            normalize_answer=system.normalize_answer,
            match_answer=system.match_answer,
            tie_break=cfg.peer_state.vote_tie_break,
            seed=cfg.training.seed,
            tau=cfg.peer_state.soft_vote_tau,
        )

    evaluation = asyncio.run(run())
    assert evaluation.marginal.soft_utility_delta == 0.0
    assert evaluation.marginal.coverage_gain_count == 0


def test_unapproved_teacher_never_reaches_student(tmp_path):
    calls = []

    async def rejecting_optimizer(system_prompt, _user_prompt, _temperature, _max_tokens):
        calls.append(system_prompt)
        if "Audit the Teacher" in system_prompt:
            return json.dumps({
                "approved": False,
                "score": 0.1,
                "feedback": "too generic",
                "rejection_reasons": ["generic"],
            })
        if system_prompt == "Return strict JSON only.":
            raise AssertionError("Student must not run after all critic rounds reject")
        return await fake_optimizer(system_prompt, _user_prompt, _temperature, _max_tokens)

    cfg = Config.from_flat(
        out_dir=str(tmp_path), answer_format="option_letter", teacher_critic_max_rounds=3,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=fake_solver, optimizer_chat=rejecting_optimizer)

    async def run():
        await system.initialize_fixed_probe([{"question": "q0", "answer": "A"}])
        return await system.propose_candidates(0, set(), CandidateFunnel())

    assert asyncio.run(run()) == []
    assert sum("Audit the Teacher" in call for call in calls) == 3
    critic_rounds = [row for row in system.tcs_rounds if row["role"] == "critic"]
    assert len(critic_rounds) == 3
    assert all(row["json_extracted"] and row["schema_valid"] for row in critic_rounds)
    assert all(row["approved_raw"] is False for row in critic_rounds)
    assert all(row["effective_approved"] is False for row in critic_rounds)
    assert all(row["rejection_reasons"] == ["generic"] for row in critic_rounds)
