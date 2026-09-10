import pytest

from multi_dataset_diverse_rl.team_search.ledger import TwoLayerLedgerRecord


def test_two_layer_ledger_preserves_cost_attribution() -> None:
    row = TwoLayerLedgerRecord(
        seed=1,
        update_index=2,
        target_member=3,
        local_optimizer_backend="gepa",
        local_search_id="search",
        local_candidate_id="local",
        local_parent_ids=("parent",),
        local_generation=1,
        team_candidate_id="team",
        phase="local_optimizer_solver_eval",
        evaluation_stage="reflection_minibatch",
        logical_role="task_solver",
        client_role="solver",
        input_tokens=10,
        output_tokens=2,
        total_tokens=12,
        logical_call_id="logical",
        provider_attempt_id="attempt",
        cache_hit=False,
    )
    assert row.total_tokens == 12


def test_two_layer_ledger_rejects_ambiguous_accounting() -> None:
    with pytest.raises(ValueError):
        TwoLayerLedgerRecord(
            1, 2, 3, "gepa", "s", None, (), None, None,
            "unknown", "stage", "role", "client", 1, 1, 3, "l", "p", False,
        )
