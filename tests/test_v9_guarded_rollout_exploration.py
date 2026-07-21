import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.cli import select_state_conditioned_candidate_eval_batch
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.checkpoint import (
    build_training_checkpoint,
    restore_system_state,
)
from multi_dataset_diverse_rl.qd.joint_controller import JointControllerMixin
from multi_dataset_diverse_rl.strategy_registry import build_policy_bundle
from multi_dataset_diverse_rl.state_conditioned import (
    STATE_SNAPSHOT_VERSION,
    build_fixed_probe_state_snapshot,
    deterministic_exploration_parent_enabled,
    exploration_profile_distance,
    select_state_conditioned_archive,
    select_state_conditioned_parents,
    select_state_conditioned_team,
    validate_fixed_probe_state_snapshot,
)


def _profile(signature, correctness, *, answers=None, trace_axis=0):
    answers = answers or ["A" if value else "B" for value in correctness]
    embeddings = [
        [1.0, 0.0] if trace_axis == 0 else [0.0, 1.0]
        for _ in correctness
    ]
    return {
        "question_hashes": [f"q{index}" for index in range(len(correctness))],
        "answer_vector": list(answers),
        "correctness_vector": list(correctness),
        "invalid_vector": [0] * len(correctness),
        "trace_embedding_vector_per_question": embeddings,
        "rollout_signature_hash": signature,
    }


def _candidate(prompt_hash, accuracy, profile, **metrics):
    return {
        "prompt_hash": prompt_hash,
        "prompt": prompt_hash,
        "generation": 1,
        "metrics": {
            "state_quality_guard_passed": True,
            "candidate_target_accuracy": accuracy,
            "candidate_invalid_rate": 0.0,
            "rollout_profile": profile,
            **metrics,
        },
    }


class _RouteHolder(JointControllerMixin):
    def __init__(self, *, state, rollout):
        self.state = state
        self.rollout = rollout
        self.state_calls = 0
        self.rollout_calls = 0
        self.joint_team_selection_history = []

    def _is_state_conditioned_method(self):
        return self.state

    def _is_rollout_qd_method(self):
        return self.rollout

    def _is_stable_qd_lineage(self):
        return False

    def _fixed_probe_hash(self, probe_data):
        return "probe"

    def _flush_jsonl(self, *args):
        return None

    async def _select_state_conditioned_joint_active_team(self, probe_data, *, epoch):
        self.state_calls += 1
        return {"enabled": True, "combination_count": 1}

    async def _select_rollout_joint_active_team(self, probe_data, *, epoch):
        self.rollout_calls += 1
        return {"enabled": True, "combination_count": 1}


def test_v9_refresh_disables_joint_selector():
    holder = _RouteHolder(state=True, rollout=True)
    result = asyncio.run(holder.refresh_joint_active_team_if_needed(
        [{"question": "q", "answer": "A"}], epoch=1
    ))
    assert holder.state_calls == 0
    assert result["selector_route"] == "sequential_accuracy_first_v1"
    assert result["joint_team_combination_count"] == 0


def test_v9_does_not_call_rollout_selector():
    holder = _RouteHolder(state=True, rollout=True)
    asyncio.run(holder.refresh_joint_active_team_if_needed(
        [{"question": "q", "answer": "A"}], epoch=1
    ))
    assert holder.rollout_calls == 0


def test_v9_direct_joint_selection_is_disabled():
    holder = _RouteHolder(state=True, rollout=True)
    result = asyncio.run(holder.select_joint_active_team(
        [{"question": "q", "answer": "A"}], epoch=1
    ))
    assert holder.state_calls == 0
    assert holder.rollout_calls == 0
    assert result["joint_team_combination_count"] == 0


def test_v8_still_uses_rollout_selector():
    holder = _RouteHolder(state=False, rollout=True)
    asyncio.run(holder.refresh_joint_active_team_if_needed(
        [{"question": "q", "answer": "A"}], epoch=1
    ))
    assert holder.rollout_calls == 1
    assert holder.state_calls == 0


def test_v9_policy_bundle_has_no_rollout_archive_or_joint_selector():
    bundle = build_policy_bundle(Config(method_version="v9_state_conditioned_error"))
    assert bundle.archive_policy.name == "none"
    assert bundle.joint_selector.name == "none"


def _pool_fixture():
    source = [
        {"question": f"question {index}", "answer": "A"}
        for index in range(16)
    ]
    states = ["C0", "C1", "C2", "C3PLUS"] * 4
    records = []
    for row, state in zip(source, states):
        records.append({
            "question_hash": hashlib.sha1(row["question"].encode("utf-8")).hexdigest()[:12],
            "state": state,
            "option_count": 4,
        })
    return source, records


def test_representative_pool_invariant():
    source, records = _pool_fixture()
    common = dict(
        candidate_eval_batch_size=8,
        candidate_batch_representative_size=4,
        candidate_batch_coverage_size=2,
        candidate_batch_conversion_size=2,
        seed=42,
    )
    enabled = select_state_conditioned_candidate_eval_batch(
        source, source, Config(**common), 1, 1, state_records=records
    )
    disabled = select_state_conditioned_candidate_eval_batch(
        source,
        source,
        Config(
            **common,
            state_coverage_enabled=False,
            state_c2_correct_conversion_enabled=False,
            state_c2_wrong_split_enabled=False,
        ),
        1,
        1,
        state_records=records,
    )
    primary = lambda rows: [row["question"] for row in rows if row["_candidate_pool_primary"]]
    assert primary(enabled) == primary(disabled)


def test_state_pools_disjoint_and_targeted_fallback_explicit():
    source, records = _pool_fixture()
    batch = select_state_conditioned_candidate_eval_batch(
        source,
        source,
        Config(
            candidate_eval_batch_size=12,
            candidate_batch_representative_size=8,
            candidate_batch_coverage_size=2,
            candidate_batch_conversion_size=2,
            seed=9,
        ),
        1,
        1,
        state_records=records,
    )
    assert len({row["question"] for row in batch}) == len(batch)
    for row in batch:
        if row["_candidate_pool_fallback_for"]:
            assert row["_candidate_pool"] == "representative"
            assert not row["_candidate_pool_target_match"]


def test_full_probe_snapshot_used_and_stale_snapshot_rejected():
    source, records = _pool_fixture()
    snapshot = {
        "snapshot_version": STATE_SNAPSHOT_VERSION,
        "snapshot_epoch": 0,
        "probe_hash": "probe",
        "active_prompt_hashes": ["a", "b"],
        "record_count": len(records),
        "records": records,
    }
    batch = select_state_conditioned_candidate_eval_batch(
        source,
        source,
        Config(candidate_eval_batch_size=8, candidate_batch_representative_size=4),
        1,
        1,
        state_snapshot=snapshot,
        active_prompt_hashes=["a", "b"],
        probe_hash="probe",
    )
    assert {row["_state_snapshot_version"] for row in batch} == {STATE_SNAPSHOT_VERSION}
    with pytest.raises(ValueError, match="prompt hashes are stale"):
        validate_fixed_probe_state_snapshot(snapshot, ["a", "changed"], "probe")


def test_snapshot_builder_covers_all_states():
    profiles = [
        _profile("a", [0, 0, 0, 1]),
        _profile("b", [0, 0, 1, 1]),
        _profile("c", [0, 1, 1, 1]),
        _profile("d", [0, 0, 0, 0]),
        _profile("e", [0, 0, 0, 0]),
    ]

    def vote_fn(answers, **kwargs):
        counts = {answer: answers.count(answer) for answer in set(answers)}
        return {"vote_answer": max(sorted(counts), key=counts.get), "vote_counts": counts}

    snapshot = build_fixed_probe_state_snapshot(
        profiles,
        ["A"] * 4,
        ["q0", "q1", "q2", "q3"],
        [4] * 4,
        ["a", "b", "c", "d", "e"],
        snapshot_epoch=0,
        probe_hash="probe",
        vote_fn=vote_fn,
        match_fn=lambda left, right: left == right,
        tie_break_method="first",
        seed=42,
    )
    assert [row["state"] for row in snapshot["records"]] == ["C0", "C1", "C2", "C3"]


def _archive_candidates():
    return [
        _candidate("inc", 0.90, _profile("inc", [1, 1, 0, 0])),
        _candidate("acc", 0.95, _profile("acc", [1, 1, 1, 0])),
        _candidate("cov", 0.94, _profile("cov", [1, 0, 1, 0]), c0_to_c1_count=3),
        _candidate("c2c", 0.94, _profile("c2c", [0, 1, 1, 0]), c2_to_c3_count=3),
        _candidate("split", 0.94, _profile("split", [1, 1, 0, 1]), c2_wrong_cluster_reduction=3),
        _candidate("explore", 0.94, _profile("explore", [0, 0, 0, 1], trace_axis=1)),
    ]


def test_exploration_slot_preserves_exploit_slots_and_uses_distinct_signature():
    cfg = Config(
        state_rollout_exploration_enabled=True,
        state_representative_capacity=6,
        qd_archive_size_per_agent=6,
    )
    archive = select_state_conditioned_archive(_archive_candidates(), "inc", 6, cfg)
    slots = {item["state_archive_slot"] for item in archive}
    assert slots == {
        "incumbent", "overall_accuracy", "coverage_repair", "c2_correct",
        "c2_split", "rollout_exploration",
    }
    assert len({item["metrics"]["rollout_profile"]["rollout_signature_hash"] for item in archive}) == 6


@pytest.mark.parametrize(
    "failed_field",
    ["accuracy_guard_passed", "vote_loss_guard_passed", "c1_to_c0_guard_passed"],
)
def test_exploration_cannot_bypass_quality_guards(failed_field):
    cfg = Config(state_rollout_exploration_enabled=True, qd_archive_size_per_agent=6)
    items = _archive_candidates()
    unsafe = _candidate(
        "unsafe", 0.99, _profile("unsafe", [0, 0, 0, 0], trace_axis=1)
    )
    unsafe["metrics"].update({"state_quality_guard_passed": False, failed_field: False})
    archive = select_state_conditioned_archive([*items, unsafe], "inc", 6, cfg)
    assert "unsafe" not in {item["prompt_hash"] for item in archive}


def test_wrong_answer_distance_not_used_in_generic_exploration_score():
    cfg = Config()
    left = _profile("left", [1, 0], answers=["A", "B"])
    right = _profile("right", [1, 0], answers=["A", "C"])
    assert exploration_profile_distance(left, right, cfg)["exploration_rollout_distance"] == 0.0


def test_exploration_slot_uses_max_min_distance_and_is_deterministic():
    cfg = Config(state_rollout_exploration_enabled=True, qd_archive_size_per_agent=6)
    base = _archive_candidates()[:-1]
    near = _candidate("near", 0.94, _profile("near", [1, 1, 0, 0]))
    far = _candidate("far", 0.94, _profile("far", [0, 0, 0, 1], trace_axis=1))
    first = select_state_conditioned_archive([*base, near, far], "inc", 6, cfg)
    second = select_state_conditioned_archive([*base, near, far], "inc", 6, cfg)
    exploration = next(item for item in first if item["state_archive_slot"] == "rollout_exploration")
    assert exploration["prompt_hash"] == "far"
    assert [item["prompt_hash"] for item in first] == [item["prompt_hash"] for item in second]


def test_exploration_parent_probability_is_deterministic_and_stagnation_forces_it():
    kwargs = dict(
        seed=42, epoch=2, step=10, agent_id=3, probability=0.15,
        stagnation_count=0, stagnation_patience=2,
    )
    assert deterministic_exploration_parent_enabled(**kwargs) == deterministic_exploration_parent_enabled(**kwargs)
    assert deterministic_exploration_parent_enabled(**{**kwargs, "probability": 0.0}) is False
    assert deterministic_exploration_parent_enabled(**{**kwargs, "probability": 0.0, "stagnation_count": 2}) is True


def test_at_most_one_exploration_parent_and_it_is_not_automatically_active():
    cfg = Config(
        state_rollout_exploration_enabled=True,
        state_exploration_parent_enabled=True,
        state_exploration_parent_probability=1.0,
    )
    archive = [
        {**item, "state_archive_slot": "rollout_exploration"}
        if item["prompt_hash"] in {"explore", "split"} else item
        for item in _archive_candidates()
    ]
    parents, sources, diagnostics = select_state_conditioned_parents(
        archive, "inc", cfg, seed=42, epoch=1, step=10, agent_id=0
    )
    assert sources.count("rollout_exploration") == 1
    assert diagnostics["exploration_parent_count"] == 1
    assert sum(item.get("state_archive_slot") == "rollout_exploration" for item in parents) == 1

    selected = select_state_conditioned_team([
        {"prompt_hashes": ["task"], "total_agent_correct_count": 10, "vote_correct_count": 5, "trace_diversity_tiebreak": 0.0},
        {"prompt_hashes": ["explore"], "total_agent_correct_count": 9, "vote_correct_count": 5, "trace_diversity_tiebreak": 1.0},
    ], cfg, probe_size=2, num_agents=5)["selected"]
    assert selected["prompt_hashes"] == ["task"]


def test_trace_distance_only_breaks_complete_task_key_tie():
    common = {
        "total_agent_correct_count": 10,
        "vote_correct_count": 2,
        "c0_count": 1,
        "coverage_depth_c2": 2,
        "c2_strict_vote_correct_count": 1,
        "c2_vote_correct_count": 1,
        "c2_mean_largest_wrong_vote": 2.0,
        "bottom2_correct_count": 2,
        "mean_gold_plurality_margin": 0.0,
        "invalid_count": 0,
    }
    teams = [
        {**common, "prompt_hashes": ["low"], "trace_diversity_tiebreak": 0.0},
        {**common, "prompt_hashes": ["high"], "trace_diversity_tiebreak": 1.0},
    ]
    assert select_state_conditioned_team(
        teams, Config(state_trace_tiebreak_enabled=True), probe_size=2, num_agents=5
    )["selected"]["prompt_hashes"] == ["high"]
    assert select_state_conditioned_team(
        teams, Config(state_trace_tiebreak_enabled=False), probe_size=2, num_agents=5
    )["selected"]["prompt_hashes"] == ["low"]


def test_exploration_lineage_and_snapshot_checkpoint_roundtrip():
    cfg = Config(
        method_version="v9_state_conditioned_error",
        reward_mode="rollout_state_conditioned",
        candidate_selection_mode="state_conditioned_accuracy_first",
        best_state_selection_mode="state_conditioned_vote_first",
        epochs=1,
        train_size=1,
    )
    agent = SimpleNamespace(
        initial_prompt="p", current_prompt="p", prompt_beam=[], history=["p"],
        accept_count=0, reject_count=0, trajectory_state_dict=lambda: {},
    )
    source = SimpleNamespace(
        agents=[agent], recent_window_records=[], specialization_strength=0.0,
        fixed_probe_state_snapshot={
            "snapshot_version": STATE_SNAPSHOT_VERSION,
            "probe_hash": "probe",
            "active_prompt_hashes": ["p"],
            "records": [],
        },
        exploration_parent_use_count=2,
        exploration_descendant_count=4,
        exploration_descendant_safe_count=3,
        exploration_descendant_archive_count=2,
        exploration_descendant_active_count=1,
        exploration_descendant_vote_gain_count=1,
        exploration_descendant_c0_to_c1_count=1,
        exploration_descendant_c1_to_c2_count=1,
        exploration_descendant_c2_to_c3_count=0,
        exploration_descendant_state_gain_count=2,
        state_parent_selection_source_counts={"rollout_exploration": 2},
        state_active_selection_source_counts={"overall_accuracy": 1},
    )
    payload = build_training_checkpoint(
        cfg, source, epoch_index=0, cursor=0, order=[0], train_accumulators={},
        best_score=0.0, best_epoch=0, epochs_without_improvement=0,
        stopped_early=False, no_effective_evolution_counter=0,
        no_effective_evolution_stopped=False, no_effective_evolution_reason="",
    )
    restored_agent = SimpleNamespace(
        initial_prompt="", current_prompt="", prompt_beam=[], history=[],
        accept_count=0, reject_count=0,
    )
    restored = SimpleNamespace(
        agents=[restored_agent],
        _make_beam_item=lambda prompt, *args: {"prompt": prompt},
    )
    restore_system_state(restored, payload["state"])
    assert restored.fixed_probe_state_snapshot == source.fixed_probe_state_snapshot
    assert restored.exploration_descendant_state_gain_count == 2
    assert restored.state_parent_selection_source_counts == {"rollout_exploration": 2}
    assert restored.state_active_selection_source_counts == {"overall_accuracy": 1}
