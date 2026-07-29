import pytest

from multi_dataset_diverse_rl.config import Config
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
