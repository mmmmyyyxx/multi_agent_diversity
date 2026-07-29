import pytest
import asyncio
import json

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.proposal_memory import (
    ProposalMemoryEntry,
    ProposalMemoryKey,
    assigned_residual_set_hash,
    entry_from_dict,
    entry_to_dict,
    feedback_for,
)
from multi_dataset_diverse_rl.system import CandidateFunnel, PromptEnsembleOptimizationSystem


def identity():
    return RunIdentity(
        method_version="member_aware_peer_state_v6",
        experiment_setting="shared_member_aware_full",
        git_commit="commit", git_dirty=False, config_fingerprint="config",
        manifest_sha256="manifest", train_file_sha256="train",
        val_file_sha256="val", test_file_sha256="test",
        train_question_set_hash="train-q", val_question_set_hash="val-q",
        test_question_set_hash="test-q",
    )


def system(tmp_path):
    cfg = Config.from_flat(
        out_dir=str(tmp_path), proposal_memory_mode="state_local_v1",
    )
    value = PromptEnsembleOptimizationSystem(cfg)
    value.set_run_identity(identity())
    value.team_state_version = 7
    value.cached_responsibility_owners = {"q-a": 0, "q-b": 0, "q-c": 1}
    return value


def test_residual_set_hash_is_order_independent_and_versioned():
    assert assigned_residual_set_hash(("b", "a", "a")) == assigned_residual_set_hash(("a", "b"))


def test_memory_key_isolates_agent_prompt_state_and_run(tmp_path):
    value = system(tmp_path)
    key = value._proposal_memory_key(
        target_agent_id=0, parent_prompt=value.agents[0].current_prompt,
        assigned_hashes={"q-a", "q-b"},
    )
    entry = ProposalMemoryEntry(key=key, assigned_question_hashes=("q-a", "q-b"))
    value.proposal_memory_entries[key.key_hash()] = entry
    assert value._proposal_memory_entry(key, {"q-b", "q-a"}) is entry
    assert value._proposal_memory_entry(
        value._proposal_memory_key(
            target_agent_id=1, parent_prompt=value.agents[1].current_prompt,
            assigned_hashes={"q-c"},
        ), {"q-c"}
    ) is None
    value.team_state_version += 1
    assert value._proposal_memory_entry(
        value._proposal_memory_key(
            target_agent_id=0, parent_prompt=value.agents[0].current_prompt,
            assigned_hashes={"q-a", "q-b"},
        ), {"q-a", "q-b"}
    ) is None


def test_non_owned_residual_in_entry_fails_closed(tmp_path):
    value = system(tmp_path)
    key = value._proposal_memory_key(
        target_agent_id=0, parent_prompt=value.agents[0].current_prompt,
        assigned_hashes={"q-a", "q-b"},
    )
    value.proposal_memory_entries[key.key_hash()] = ProposalMemoryEntry(
        key=key, assigned_question_hashes=("q-a", "q-c"),
    )
    with pytest.raises(RuntimeError, match="lifecycle/schema mismatch"):
        value._proposal_memory_entry(key, {"q-a", "q-b"})
    with pytest.raises(RuntimeError, match="non-owned residual"):
        value._proposal_memory_key(
            target_agent_id=0, parent_prompt=value.agents[0].current_prompt,
            assigned_hashes={"q-c"},
        )


def test_entry_roundtrip_and_feedback_preserve_cursor_and_tabu():
    key = ProposalMemoryKey("run", 1, 0, "prompt", "residual")
    entry = ProposalMemoryEntry(
        key=key, assigned_question_hashes=("a",), attempt_count=2,
        previous_evidence_bundle_hashes=("bundle",),
        previous_repair_plan_hashes=("plan",), last_failure_stage="zero_repair_behavior",
        rotation_cursor=2, immediate_tabu_bundle_hash="bundle",
    )
    restored = entry_from_dict(entry_to_dict(entry))
    assert restored == entry
    assert feedback_for(restored).rotation_level == "pattern"


def test_rejected_update_writes_a_state_local_entry_and_accepted_event_does_not():
    value = system("proposal-memory")
    assigned = {"q-a", "q-b"}
    key = value._proposal_memory_key(
        target_agent_id=0, parent_prompt=value.agents[0].current_prompt,
        assigned_hashes=assigned,
    )
    value._proposal_memory_attempts[0] = {
        "key": key, "memory_hit": False, "feedback": None,
        "evidence_bundle_hash": "bundle-a", "rotation_cursor": 0,
        "rotation_exhausted": False,
    }
    value._record_proposal_memory_outcome(
        update_index=0, target_agent_id=0, assigned_hashes=assigned,
        evaluated=(), funnel=CandidateFunnel(), accepted=None,
    )
    entry = value.proposal_memory_entries[key.key_hash()]
    assert entry.attempt_count == 1
    assert entry.last_failure_stage == "pipeline"
    assert value.proposal_memory_events[-1]["memory_hit"] is False


def test_accepted_outcome_does_not_create_failure_memory_and_new_state_cannot_hit(tmp_path):
    value = system(tmp_path)
    assigned = {"q-a", "q-b"}
    key = value._proposal_memory_key(
        target_agent_id=0, parent_prompt=value.agents[0].current_prompt,
        assigned_hashes=assigned,
    )
    value._proposal_memory_attempts[0] = {
        "key": key, "memory_hit": False, "feedback": None,
        "evidence_bundle_hash": "bundle-a", "rotation_cursor": 0,
        "rotation_exhausted": False,
    }
    # Acceptance is supplied by the paired evaluator; memory only records its
    # lifecycle result and must never turn an accepted update into a failure entry.
    value._record_proposal_memory_outcome(
        update_index=0, target_agent_id=0, assigned_hashes=assigned,
        evaluated=(), funnel=CandidateFunnel(), accepted=object(),
    )
    assert value.proposal_memory_events[-1]["failure_stage"] == "accepted"
    assert not value.proposal_memory_entries
    value.team_state_version += 1
    successor_key = value._proposal_memory_key(
        target_agent_id=0, parent_prompt=value.agents[0].current_prompt,
        assigned_hashes=assigned,
    )
    assert successor_key != key
    assert value._proposal_memory_entry(successor_key, assigned) is None


def test_end_to_end_memory_hit_rotates_only_the_same_agent_state(tmp_path):
    async def solver(_question, _agent, _prompt):
        return PromptAnswer("B", "FINAL_ANSWER: B", True)

    teacher_requests: list[tuple[str, str]] = []

    async def optimizer(system_prompt, user_prompt, _temperature, _max_tokens):
        if "Check only explicit hard blockers" in system_prompt:
            return json.dumps({"failed_checks": [], "risk_case_ids": [], "feedback": ""})
        if system_prompt == "Return strict JSON only.":
            return json.dumps({"candidate_prompts": ["candidate-a", "candidate-b"]})
        teacher_requests.append((system_prompt, user_prompt))
        return json.dumps({
            "failure_pattern": "owned residual remains", "repair_rule": "Apply a direct consistency check before deciding.",
            "preservation_rule": "Keep the output contract and retain supported conclusions.",
        })

    async def run(mode: str):
        cfg = Config.from_flat(
            out_dir=str(tmp_path / mode), proposal_memory_mode=mode,
            initialization_mode="provided_prompt_set",
            provided_prompts_json=json.dumps([f"p{agent}" for agent in range(5)]),
        )
        value = PromptEnsembleOptimizationSystem(cfg, solver=solver, optimizer_chat=optimizer)
        value.set_run_identity(identity())
        await value.initialize_fixed_probe([
            {"question": f"q{index}", "answer": "A"} for index in range(3)
        ])
        owned = {row.question_hash for row in value.fixed_probe.examples}
        value.cached_responsibility_owners = {question_hash: 0 for question_hash in owned}
        first_funnel = CandidateFunnel()
        first = await value.propose_candidates(0, owned, first_funnel, update_index=0)
        accepted, _, evaluated = await value.evaluate_candidates(0, first, owned, first_funnel)
        if mode == "state_local_v1":
            value._record_proposal_memory_outcome(
                update_index=0, target_agent_id=0, assigned_hashes=owned,
                evaluated=evaluated, funnel=first_funnel, accepted=accepted,
            )
            value.cached_responsibility_owners = {question_hash: 1 for question_hash in owned}
            await value.propose_candidates(1, owned, CandidateFunnel(), update_index=1)
            value.cached_responsibility_owners = {question_hash: 0 for question_hash in owned}
            second = await value.propose_candidates(0, owned, CandidateFunnel(), update_index=2)
            return value, first, second, evaluated
        return value, first, (), evaluated

    async def check():
        off, off_candidates, _, off_evaluated = await run("off")
        on, on_candidates, on_second, on_evaluated = await run("state_local_v1")
        assert [row.prompt_hash for row in off_candidates] == [row.prompt_hash for row in on_candidates]
        assert [row.constraint for row in off_evaluated] == [row.constraint for row in on_evaluated]
        assert off.tcs_context_history[0]["proposal_context_hash"] == on.tcs_context_history[0]["proposal_context_hash"]
        assert off.tcs_context_history[0]["selected_context_pattern_question_hashes"] == on.tcs_context_history[0]["selected_context_pattern_question_hashes"]
        assert off.tcs_context_history[0]["proposal_rotation_cursor"] == on.tcs_context_history[0]["proposal_rotation_cursor"] == 0
        assert off.tcs_rounds[0]["request_hash"] == on.tcs_rounds[0]["request_hash"]
        assert on.proposal_memory_events[0]["memory_hit"] is False
        assert on.tcs_context_history[-1]["proposal_memory_hit"] is True
        assert on.tcs_context_history[-1]["evidence_bundle_hash"] != on.proposal_memory_events[0]["current_evidence_bundle_hash"]
        assert on.proposal_memory_events[0]["target_agent_id"] == 0
        assert len(on_second) == 2
        second_agent_context = on.tcs_context_history[1]
        assert second_agent_context["target_agent_id"] == 1
        assert second_agent_context["proposal_memory_hit"] is False
        assert "proposal_failure_feedback" not in second_agent_context["serialized_recursive_field_paths"]
        assert "change_repair_mechanism" in teacher_requests[-1][0]
        assert "target_agent_id\":1" not in teacher_requests[-1][0]
        on.flush_artifacts()
        memory_files = (
            "proposal_memory_events_sanitized.jsonl",
            "proposal_memory_summary.json",
            "proposal_memory_key_isolation_audit.json",
            "proposal_rotation_trajectory.jsonl",
        )
        contents = {
            name: (tmp_path / "state_local_v1" / name).read_text(encoding="utf-8")
            for name in memory_files
        }
        assert all("candidate-a" not in text and "candidate-b" not in text for text in contents.values())
        assert all("FINAL_ANSWER" not in text and '"question"' not in text for text in contents.values())
        assert len(contents["proposal_memory_events_sanitized.jsonl"].splitlines()) == len(on.proposal_memory_events)
        assert len(contents["proposal_rotation_trajectory.jsonl"].splitlines()) == len(on.proposal_rotation_trajectory)
    asyncio.run(check())
