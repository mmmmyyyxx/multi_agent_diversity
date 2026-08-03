import json
import sqlite3
from pathlib import Path

from multi_dataset_diverse_rl.evaluation.persistent_solver_cache import PersistentSolverCache
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer
from scripts.audit_final_method_stage import (
    _comparison_cache_chain,
    _expected_matrix,
    _matched_observation_consistency,
    _priority_key,
)
from scripts.final_method_source_identity import build_source_identity
from scripts.run_task_level_accuracy import (
    _merge_ready_solver_cache,
    _sqlite_backup,
    _sqlite_content_sha256,
)


def test_formal_stage_matrices_are_exact():
    assert _expected_matrix("pilot") == (
        ("disambiguation_qa",),
        (46,),
        (
            "shared_baseline",
            "shared_independent_accuracy",
            "shared_peer_state_vote_first",
            "shared_peer_state_member_pareto",
            "shared_member_aware_responsibility",
            "shared_member_aware_full",
        ),
        8,
        False,
    )
    assert _expected_matrix("disambiguation")[3:] == (32, True)
    assert _expected_matrix("cross_task") == (
        ("geometric_shapes", "ruin_names"),
        (44, 45, 46),
        ("shared_baseline", "shared_member_aware_full"),
        32,
        True,
    )


def test_max_wait_gate_orders_wait_then_direct_margin_deficit_seed():
    rows = [
        {
            "updates_since_selected": 9,
            "D_i": 2,
            "S_i": 5,
            "d_i": 1,
            "seeded_rank": "b",
        },
        {
            "updates_since_selected": 9,
            "D_i": 2,
            "S_i": 5,
            "d_i": 1,
            "seeded_rank": "a",
        },
        {
            "updates_since_selected": 8,
            "D_i": 99,
            "S_i": 99,
            "d_i": 99,
            "seeded_rank": "0",
        },
    ]
    assert min(rows, key=_priority_key) is rows[1]


def test_sqlite_frozen_cache_clone_is_independent(tmp_path):
    source = tmp_path / "frozen.sqlite"
    destination = tmp_path / "mutable.sqlite"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE entries(value TEXT)")
        connection.execute("INSERT INTO entries VALUES ('frozen')")
    _sqlite_backup(source, destination)
    assert _sqlite_content_sha256(source) == _sqlite_content_sha256(destination)
    with sqlite3.connect(destination) as connection:
        connection.execute("INSERT INTO entries VALUES ('mutable')")
    with sqlite3.connect(source) as connection:
        assert connection.execute("SELECT value FROM entries").fetchall() == [("frozen",)]
    with sqlite3.connect(destination) as connection:
        assert connection.execute("SELECT value FROM entries").fetchall() == [
            ("frozen",),
            ("mutable",),
        ]


def test_source_identity_is_hash_only_and_covers_formal_scripts():
    workspace = Path(__file__).resolve().parents[1]
    identity = build_source_identity(workspace)
    encoded = json.dumps(identity)
    assert identity["source_identity_version"] == "final_method_source_identity_v1"
    assert len(identity["source_tree_hash"]) == 64
    assert len(identity["git_diff_hash"]) == 64
    assert all(
        value != "missing"
        for value in identity["experiment_script_hashes"].values()
    )
    assert str(workspace) not in encoded


def test_cumulative_reference_merges_ready_entries_for_later_settings(tmp_path):
    reference = tmp_path / "reference.sqlite"
    local = tmp_path / "local.sqlite"
    next_local = tmp_path / "next.sqlite"
    reference_cache = PersistentSolverCache(reference)
    metadata = {
        "model_request_identity": "request-v1",
        "solver_model": "solver",
        "endpoint_identity": "endpoint",
        "output_contract_version": "contract",
        "parser_version": "parser",
        "temperature": 0.0,
        "max_tokens": 1800,
        "evaluation_replica_seed": 46,
        "prompt_hash": "prompt",
        "question_hash": "question",
    }
    assert reference_cache._claim_or_read("baseline", metadata)[0] == "owner"
    reference_cache._store("baseline", PromptAnswer("A", "trace-a", True))
    _sqlite_backup(reference, local)
    local_cache = PersistentSolverCache(local)
    assert local_cache._claim_or_read("new-prompt", metadata)[0] == "owner"
    local_cache._store("new-prompt", PromptAnswer("B", "trace-b", True))
    _merge_ready_solver_cache(local, reference)
    assert PersistentSolverCache(reference).ready_entry_count() == 2
    _sqlite_backup(reference, next_local)
    assert PersistentSolverCache(next_local).ready_entry_count() == 2


def test_matched_observation_gate_rejects_unchanged_prompt_metric_drift():
    prompt_hashes = [str(index) * 64 for index in range(5)]
    baseline = {
        "complete": True,
        "task": "toy",
        "seed": 46,
        "setting": "shared_baseline",
        "final_prompt_hashes": prompt_hashes,
        "selected_test": {
            "per_agent_correct_counts": [10, 10, 10, 10, 10],
            "vote_correct_count": 10,
        },
    }
    unchanged = {
        **baseline,
        "setting": "shared_member_aware_full",
        "selected_test": {
            "per_agent_correct_counts": [12, 12, 12, 12, 12],
            "vote_correct_count": 12,
        },
    }
    findings = []
    rows = _matched_observation_consistency([baseline, unchanged], findings)
    assert rows[0]["passed"] is False
    assert rows[0]["mismatched_unchanged_member_ids"] == [0, 1, 2, 3, 4]
    assert len(findings) == 1


def test_comparison_cache_chain_requires_previous_post_run_reference():
    rows = [
        {
            "complete": True,
            "task": "toy",
            "seed": 46,
            "setting": "shared_baseline",
            "comparison_cache_match": {
                "matched": True,
                "starting_cache_sha256": "a",
                "reference_cache_sha256": "a",
                "post_run_reference_cache_sha256": "b",
            },
        },
        {
            "complete": True,
            "task": "toy",
            "seed": 46,
            "setting": "shared_member_aware_full",
            "comparison_cache_match": {
                "matched": True,
                "starting_cache_sha256": "wrong",
                "reference_cache_sha256": "wrong",
                "post_run_reference_cache_sha256": "c",
            },
        },
    ]
    findings = []
    audit = _comparison_cache_chain(rows, findings)
    assert audit[0]["chain_continuity"] is True
    assert audit[1]["chain_continuity"] is False
    assert len(findings) == 1
