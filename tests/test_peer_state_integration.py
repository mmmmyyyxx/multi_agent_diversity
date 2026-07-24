import asyncio
from copy import deepcopy
import json

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer, evaluate_candidate_profile
from multi_dataset_diverse_rl.system import CandidateFunnel, CandidateRuntime, PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.tcs import StudentCandidate


QUESTIONS = {"q0": "A", "q1": "A", "q2": "A"}


async def fake_solver(question, agent_id, prompt):
    if "repair-q0" in prompt:
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
        facts = json.loads(
            system_prompt.split("DERIVED_CASE_FACTS:\n", 1)[1].split(
                "\nProposalContext:", 1,
            )[0]
        )
        return json.dumps({
            "case_fact_restatements": facts,
            "context_consistent": True,
            "sample_memorization_free": True,
            "executable_change": True,
            "internally_consistent": True,
            "preservation_rule_present": True,
            "output_contract_safe": True,
            "peer_copying_free": True,
            "stereotype_forcing_free": True,
            "non_generic_change": True,
            "blocking_reasons": [],
            "soft_concerns": ["empirical benefit remains uncertain"],
            "score": 0.2,
            "feedback": "approved",
        })
    if system_prompt == "Return strict JSON only.":
        return json.dumps({"candidates": [{
            "candidate_prompt": "repair-q0",
            "observed_failure_pattern": "misses uncovered cases",
            "generalizable_mechanism": "premature ambiguity decisions",
            "decision_rule": "check ambiguity before committing",
            "uncertainty_or_abstention_rule": "retain ambiguity without exclusion evidence",
            "preservation_conditions": "retain existing correct decisions",
            "evidence_summary": "uncovered cases need another decision check",
        }]})
    return json.dumps({
        "observed_failure_pattern": "misses uncovered cases",
        "generalizable_mechanism": "premature ambiguity decisions",
        "decision_rule": "check ambiguity before committing",
        "uncertainty_or_abstention_rule": "retain ambiguity without exclusion evidence",
        "preservation_conditions": "retain existing correct decisions",
        "evidence_summary": "uncovered cases need another decision check",
    })


def student_candidate():
    return StudentCandidate(
        candidate_prompt="repair-q0",
        observed_failure_pattern="misses uncovered cases",
        generalizable_mechanism="premature ambiguity decisions",
        decision_rule="check ambiguity",
        uncertainty_or_abstention_rule="retain ambiguity without exclusion evidence",
        preservation_conditions="retain correct decisions",
        evidence_summary="uncovered cases need another decision check",
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
    assert [
        row["team_state_version"] for row in system.responsibility_assignments
    ] == [0, 1]
    assert system.responsibility_refresh_count == 2
    prior_owner_age = dict(system.responsibility_state.owner_age_by_question)
    system.ensure_responsibility_current()
    assert len(system.responsibility_assignments) == 2
    assert system.responsibility_state.owner_age_by_question == prior_owner_age
    responsibility_audit = system.responsibility_assignments[-1]
    assert "assigned_opportunities" in responsibility_audit
    assert "member_gain_counts" in responsibility_audit
    assert "improvement_need_by_agent" in responsibility_audit
    assert "owner_candidate_pareto_fronts" in responsibility_audit
    assert "owner_chosen_reasons" in responsibility_audit
    decision = system.candidate_decisions[-1]
    assert decision["funnel"]["accepted_candidate"] is True
    assert len(decision["agent_target_priorities"]) == 5
    assert sum(row["selected"] for row in decision["agent_target_priorities"]) == 1
    assert all(
        {
            "individual_error_count",
            "assigned_load",
            "direct_vote_fix_count",
            "oracle_soft_utility_gain_sum",
            "coverage_opportunity_count",
            "dominant_wrong_count",
            "gain_count",
            "improvement_need",
            "unique_correct_count",
            "pivotal_correct_count",
            "updates_since_selected",
            "overdue",
            "pareto_front",
            "selected",
        }
        <= set(row)
        for row in decision["agent_target_priorities"]
    )
    assert (
        system.tcs_context_history[-1]["context_type"]
        == "MemberAwareResponsibilityProposalContext"
    )
    assert len(system.tcs_context_history[-1]["proposal_context_hash"]) == 64
    assert system.tcs_context_history[-1]["forbidden_field_violations"] == []
    assert system.tcs_context_history[-1]["responsibility_specific_field_count"] > 0
    assert [row["role"] for row in system.tcs_rounds] == ["teacher", "critic", "student"]
    assert all(row["schema_valid"] for row in system.tcs_rounds)
    assert system.tcs_rounds[1]["effective_approved"] is True
    assert system.tcs_rounds[2]["raw_count"] == 1


def test_responsibility_refresh_failure_rolls_back_complete_commit_state(tmp_path):
    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        answer_format="option_letter",
        num_candidates_per_parent=1,
        stage_a_channel_top_k=1,
        stage_b_candidate_budget=2,
    )
    system = PromptEnsembleOptimizationSystem(
        cfg, solver=fake_solver, optimizer_chat=fake_optimizer
    )
    captured = {}

    async def run():
        data = [{"question": question, "answer": gold} for question, gold in QUESTIONS.items()]
        await system.initialize_fixed_probe(data)
        system.ensure_responsibility_current()
        original_refresh = system.refresh_responsibility_after_commit

        def failing_refresh():
            captured["responsibility_state"] = deepcopy(system.responsibility_state)
            target = system.candidate_decisions[-1]["target_agent_id"]
            captured["responsibility_state"].accepted_updates_by_agent[target] -= 1
            captured["cached_owners"] = deepcopy(system.cached_responsibility_owners)
            captured["cached_assignments"] = deepcopy(
                system.cached_responsibility_assignments
            )
            captured["cached_opportunities"] = deepcopy(
                system.cached_member_opportunities
            )
            captured["team_state_version"] = system.team_state_version
            captured["responsibility_state_version"] = (
                system.responsibility_state_version
            )
            captured["refresh_count"] = system.responsibility_refresh_count
            captured["peer_history_length"] = len(system.peer_state_history)
            captured["responsibility_history_length"] = len(
                system.responsibility_assignments
            )
            captured["target_audit_length"] = len(system.target_priority_audit)
            original_refresh()
            raise RuntimeError("synthetic responsibility refresh failure")

        system.refresh_responsibility_after_commit = failing_refresh
        old_prompts = [agent.current_prompt for agent in system.agents]
        old_previous_prompts = [
            agent.previous_active_prompt for agent in system.agents
        ]
        old_profiles = deepcopy(system.active_profiles)
        with pytest.raises(
            RuntimeError, match="synthetic responsibility refresh failure"
        ):
            await system.update_once(0)
        return old_prompts, old_previous_prompts, old_profiles

    old_prompts, old_previous_prompts, old_profiles = asyncio.run(run())
    assert [agent.current_prompt for agent in system.agents] == old_prompts
    assert [agent.previous_active_prompt for agent in system.agents] == old_previous_prompts
    assert system.active_profiles == old_profiles
    assert system.responsibility_state == captured["responsibility_state"]
    assert system.cached_responsibility_owners == captured["cached_owners"]
    assert (
        system.cached_responsibility_assignments
        == captured["cached_assignments"]
    )
    assert system.cached_member_opportunities == captured["cached_opportunities"]
    assert system.team_state_version == captured["team_state_version"]
    assert (
        system.responsibility_state_version
        == captured["responsibility_state_version"]
    )
    assert system.responsibility_refresh_count == captured["refresh_count"]
    assert len(system.peer_state_history) == captured["peer_history_length"]
    assert (
        len(system.responsibility_assignments)
        == captured["responsibility_history_length"]
    )
    assert len(system.target_priority_audit) == captured["target_audit_length"]


def test_b3_also_refreshes_responsibilities_online(tmp_path):
    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        answer_format="option_letter",
        experiment_setting="shared_member_aware_responsibility",
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
            "observed_failure_pattern": "misses uncovered cases",
            "generalizable_mechanism": "premature ambiguity decisions",
            "decision_rule": "check ambiguity before committing",
            "uncertainty_or_abstention_rule": "retain ambiguity without exclusion evidence",
            "preservation_conditions": "retain existing correct decisions",
            "evidence_summary": "uncovered cases need another decision check",
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
            build("shared_independent_accuracy"),
            build("shared_peer_state_vote_first"),
            build("shared_member_aware_responsibility"),
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
        experiment_setting="shared_independent_accuracy",
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
        experiment_setting="shared_independent_accuracy",
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
        experiment_setting="shared_independent_accuracy",
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
            initial_profiles=system.initial_profiles,
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
            facts = json.loads(
                system_prompt.split("DERIVED_CASE_FACTS:\n", 1)[1].split(
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
                "score": 0.1,
                "feedback": "too generic",
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
    assert all(row["effective_approved"] is False for row in critic_rounds)
    assert all(row["blocking_reasons"] == ["generic"] for row in critic_rounds)


def test_critic_fact_misread_retries_same_teacher_before_student(tmp_path):
    teacher_calls = 0
    critic_calls = 0

    async def retrying_optimizer(system_prompt, user_prompt, temperature, max_tokens):
        nonlocal teacher_calls, critic_calls
        if "Audit the Teacher" in system_prompt:
            critic_calls += 1
            facts = json.loads(
                system_prompt.split("DERIVED_CASE_FACTS:\n", 1)[1].split(
                    "\nProposalContext:", 1,
                )[0]
            )
            if critic_calls == 1:
                facts[0]["target_status"] = "misread"
            return json.dumps({
                "case_fact_restatements": facts,
                "context_consistent": True,
                "sample_memorization_free": True,
                "executable_change": True,
                "internally_consistent": True,
                "preservation_rule_present": True,
                "output_contract_safe": True,
                "peer_copying_free": True,
                "stereotype_forcing_free": True,
                "non_generic_change": True,
                "blocking_reasons": [],
                "soft_concerns": [],
                "score": 0.1,
                "feedback": "worth testing",
            })
        if system_prompt != "Return strict JSON only.":
            teacher_calls += 1
        return await fake_optimizer(system_prompt, user_prompt, temperature, max_tokens)

    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        answer_format="option_letter",
        num_candidates_per_parent=1,
        critic_json_max_retries=2,
    )
    system = PromptEnsembleOptimizationSystem(
        cfg,
        solver=fake_solver,
        optimizer_chat=retrying_optimizer,
    )

    async def run():
        await system.initialize_fixed_probe([{"question": "q0", "answer": "A"}])
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert teacher_calls == 1
    assert critic_calls == 2
    assert funnel.critic_invalid_responses == 1
    assert funnel.critic_approved == 1
    assert len(candidates) == 1
    critic_rounds = [row for row in system.tcs_rounds if row["role"] == "critic"]
    assert critic_rounds[0]["schema_valid"] is False
    assert "restatement mismatch" in critic_rounds[0]["parse_error"]
    assert critic_rounds[1]["fact_restatement_valid"] is True


def test_student_candidate_copying_supplied_question_is_rejected_before_stage_a(tmp_path):
    question = (
        "The analyst called the manager after she reviewed the report. "
        "Who does she refer to? (A) analyst (B) manager (C) ambiguous"
    )

    async def memorizing_student(system_prompt, user_prompt, temperature, max_tokens):
        if system_prompt != "Return strict JSON only.":
            return await fake_optimizer(system_prompt, user_prompt, temperature, max_tokens)
        return json.dumps({"candidates": [{
            "candidate_prompt": f"Memorize this example: {question}",
            "observed_failure_pattern": "misses one example",
            "generalizable_mechanism": "memorize the supplied text",
            "decision_rule": "repeat the stored answer",
            "uncertainty_or_abstention_rule": "abstain otherwise",
            "preservation_conditions": "leave other cases unchanged",
            "evidence_summary": "one supplied example is stored verbatim",
        }]})

    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        answer_format="option_letter",
        num_candidates_per_parent=1,
    )
    system = PromptEnsembleOptimizationSystem(
        cfg,
        solver=fake_solver,
        optimizer_chat=memorizing_student,
    )

    async def run():
        await system.initialize_fixed_probe([{"question": question, "answer": "A"}])
        funnel = CandidateFunnel()
        candidates = await system.propose_candidates(0, set(), funnel)
        return funnel, candidates

    funnel, candidates = asyncio.run(run())
    assert candidates == []
    assert funnel.schema_valid_count == 1
    assert funnel.sample_memorization_rejected == 1
    assert funnel.stage_a_evaluated == 0
