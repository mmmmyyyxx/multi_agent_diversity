from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.lineage import empty_lineage_state, lineage_drift, update_lineage_state


def selected(signature=("hard_elimination",), correctness=(1, 0, 1), rescue=(1, 0, 0), embedding=(1.0, 0.0)):
    return {
        "prompt": "prompt",
        "prompt_hash": "hash",
        "mechanism_representation": {
            "normalized_operation_sequence": list(signature),
            "mechanism_embedding": list(embedding),
        },
        "behavior_profile": {
            "correctness_vector": list(correctness),
            "error_vector": [1 - value for value in correctness],
            "rescue_vector": list(rescue),
            "accuracy": sum(correctness) / len(correctness),
        },
    }


def state_only(record):
    return {key: value for key, value in record.items() if key not in {"old_status", "new_status", "reason"}}


def test_lineage_transitions_uncommitted_provisional_committed():
    cfg = Config(lineage_provisional_epochs=2, lineage_commit_epochs=3)
    state = empty_lineage_state()
    first = update_lineage_state(state, selected(), epoch=1, quality_gate_passed=True, config=cfg)
    second = update_lineage_state(state_only(first), selected(), epoch=2, quality_gate_passed=True, config=cfg)
    third = update_lineage_state(state_only(second), selected(), epoch=3, quality_gate_passed=True, config=cfg)
    assert first["new_status"] == "uncommitted"
    assert second["new_status"] == "provisional"
    assert third["new_status"] == "committed"


def test_drift_is_disabled_before_commit_and_enabled_after_commit():
    cfg = Config()
    candidate = selected(("weighted_scoring", "counterfactual_check"), embedding=(0.0, 1.0))
    assert lineage_drift(candidate, empty_lineage_state(), cfg)["lineage_drift_penalty"] == 0.0
    committed = empty_lineage_state()
    committed.update({
        "lineage_status": "committed",
        "lineage_anchor_mechanism_signature": ["hard_elimination"],
        "lineage_anchor_mechanism_embedding": [1.0, 0.0],
        "lineage_anchor_correctness_vector": [1, 0, 1],
        "lineage_anchor_rescue_vector": [1, 0, 0],
    })
    assert lineage_drift(candidate, committed, cfg)["lineage_drift_penalty"] > 0.0


def test_committed_lineage_switch_requires_two_confirmations():
    cfg = Config(lineage_switch_confirmation_epochs=2)
    state = empty_lineage_state()
    state.update({
        "lineage_status": "committed",
        "lineage_anchor_mechanism_signature": ["hard_elimination"],
        "lineage_anchor_mechanism_embedding": [1.0, 0.0],
    })
    alternative = selected(("weighted_scoring",), embedding=(0.0, 1.0))
    first = update_lineage_state(state, alternative, epoch=2, quality_gate_passed=True, config=cfg)
    assert first["reason"] == "lineage_switch_pending"
    assert first["lineage_anchor_mechanism_signature"] == ["hard_elimination"]
    second = update_lineage_state(state_only(first), alternative, epoch=3, quality_gate_passed=True, config=cfg)
    assert second["reason"] == "lineage_switch_committed"
    assert second["lineage_anchor_mechanism_signature"] == ["weighted_scoring"]


def test_pending_switch_is_cancelled_when_agent_returns_to_anchor():
    cfg = Config(lineage_switch_confirmation_epochs=2)
    state = empty_lineage_state()
    state.update({
        "lineage_status": "committed",
        "lineage_anchor_mechanism_signature": ["hard_elimination"],
        "lineage_anchor_mechanism_embedding": [1.0, 0.0],
    })
    pending = update_lineage_state(
        state, selected(("weighted_scoring",), embedding=(0.0, 1.0)),
        epoch=2, quality_gate_passed=True, config=cfg,
    )
    cancelled = update_lineage_state(
        state_only(pending), selected(), epoch=3, quality_gate_passed=True, config=cfg,
    )
    assert cancelled["reason"] == "anchor_retained"
    assert cancelled["pending_lineage_count"] == 0
    assert cancelled["lineage_switch_cancel_count"] == 1
