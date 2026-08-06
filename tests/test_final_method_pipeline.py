import asyncio
import json
import sqlite3
from pathlib import Path

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.persistent_solver_cache import PersistentSolverCache
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer, PromptQuestionEvaluator
from multi_dataset_diverse_rl.persistence.identity import (
    PROMPT_QUESTION_EVALUATOR_VERSION,
    solver_request_components,
    solver_request_identity,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.audit_final_method_stage import (
    _comparison_cache_chain,
    _expected_matrix,
    _matched_observation_consistency,
    _portable_config_value,
    _priority_key,
)
from scripts.final_method_source_identity import build_source_identity
from scripts.run_task_level_accuracy import (
    _compare_unchanged_test_observations,
    _merge_ready_solver_cache,
    _solver_cache_snapshot,
    _sqlite_backup,
    _sqlite_content_sha256,
)


def test_stage_gate_reports_dataset_paths_without_machine_prefix():
    absolute = "D:\\workspace\\strict_splits_bbh_seed42\\task\\opt.csv"
    assert _portable_config_value("train_path", absolute) == (
        "strict_splits_bbh_seed42/task/opt.csv"
    )
    assert _portable_config_value("train_size", 75) == 75


def test_formal_stage_matrices_are_exact():
    assert _expected_matrix("pilot") == (
        ("disambiguation_qa",),
        (46,),
        (
            "shared_baseline",
            "shared_independent_accuracy",
            "shared_peer_state_vote_first",
            "shared_peer_state_member_first_safe",
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
    assert _expected_matrix("strict_v2_witness") == (
        ("disambiguation_qa",),
        (46,),
        ("shared_baseline", "shared_peer_state_member_first_safe"),
        0,
        True,
    )
    assert _expected_matrix("strict_v2_disambiguation") == (
        ("disambiguation_qa",),
        (44, 45, 46),
        (
            "shared_baseline",
            "shared_peer_state_member_first_safe",
            "shared_member_aware_responsibility",
            "shared_member_aware_full",
        ),
        32,
        True,
    )


def test_first_frontier_tie_break_orders_wait_then_seed_only():
    rows = [
        {
            "updates_since_selected": 9,
            "D_i": 99,
            "S_i": 99,
            "d_i": 99,
            "seeded_rank": "b",
        },
        {
            "updates_since_selected": 9,
            "D_i": 0,
            "S_i": 0,
            "d_i": 0,
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
    assert identity["source_identity_version"] == "final_method_source_identity_v2"
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
    audit = _merge_ready_solver_cache(local, reference)
    assert audit["gate"] == "PASS"
    assert audit["new_entries_merged"] == 1
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
                "gate": "PASS",
                "matched": True,
                "cache_chain_continuity": True,
                "starting_cache_sha256": "a",
                "parent_reference_hash": "a",
                "result_reference_hash": "b",
                "exact_request_conflict_count": 0,
                "missing_reference_count": 0,
                "unexpected_provider_recall_count": 0,
            },
        },
        {
            "complete": True,
            "task": "toy",
            "seed": 46,
            "setting": "shared_member_aware_full",
            "comparison_cache_match": {
                "gate": "PASS",
                "matched": True,
                "cache_chain_continuity": True,
                "starting_cache_sha256": "wrong",
                "parent_reference_hash": "wrong",
                "result_reference_hash": "c",
                "exact_request_conflict_count": 0,
                "missing_reference_count": 0,
                "unexpected_provider_recall_count": 0,
            },
        },
    ]
    findings = []
    audit = _comparison_cache_chain(rows, findings)
    assert audit[0]["chain_continuity"] is True
    assert audit[1]["chain_continuity"] is False
    assert len(findings) == 1


def _cache_metadata() -> dict:
    return {
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


def test_first_exact_observation_is_never_overwritten_and_conflict_fails(tmp_path):
    reference = tmp_path / "reference.sqlite"
    local = tmp_path / "local.sqlite"
    cache = PersistentSolverCache(reference)
    assert cache._claim_or_read("K", _cache_metadata())[0] == "owner"
    cache._store("K", PromptAnswer("A", "FINAL_ANSWER: A", True))
    _sqlite_backup(reference, local)
    conflicting = json.dumps(
        vars(PromptAnswer("B", "FINAL_ANSWER: B", True)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with sqlite3.connect(local) as connection:
        connection.execute(
            "UPDATE solver_cache SET answer_json = ? WHERE cache_key = 'K'",
            (conflicting,),
        )

    before = _solver_cache_snapshot(reference)["entries"]["K"]["observation_hash"]
    audit = _merge_ready_solver_cache(local, reference)
    after = _solver_cache_snapshot(reference)["entries"]["K"]["observation_hash"]
    assert audit["gate"] == "FAIL"
    assert audit["exact_request_conflict_count"] == 1
    assert before == after


def _evaluator(path: Path) -> PromptQuestionEvaluator:
    return PromptQuestionEvaluator(
        model_request_identity="request-v1",
        parser_version="parser",
        temperature=0.0,
        decoding_seed=46,
        cache_metadata={
            "solver_model": "solver",
            "endpoint_identity": "endpoint",
            "output_contract_version": "contract",
            "max_tokens": 1800,
        },
        shared_cache=PersistentSolverCache(path),
    )


def _evaluate(evaluator: PromptQuestionEvaluator, question_hash: str, solve):
    return evaluator.evaluate(
        question=question_hash,
        question_hash=question_hash,
        prompt="prompt",
        prompt_hash="prompt",
        agent_id=0,
        solve=solve,
    )


def test_cross_setting_clone_hits_merged_observation_without_provider(tmp_path):
    reference = tmp_path / "reference.sqlite"
    PersistentSolverCache(reference)
    s3 = tmp_path / "s3.sqlite"
    _sqlite_backup(reference, s3)

    async def produce(*_args):
        return PromptAnswer("A", "FINAL_ANSWER: A", True)

    first = asyncio.run(_evaluate(_evaluator(s3), "q", produce))
    assert first.answer == "A"
    assert _merge_ready_solver_cache(s3, reference)["gate"] == "PASS"
    s4 = tmp_path / "s4.sqlite"
    _sqlite_backup(reference, s4)

    async def forbidden(*_args):
        raise AssertionError("provider must not be called")

    second_evaluator = _evaluator(s4)
    second = asyncio.run(_evaluate(second_evaluator, "q", forbidden))
    assert second.answer == "A"
    assert second_evaluator.shared_cache.hits == 1
    assert second_evaluator.shared_cache.misses == 0


def test_setting_name_does_not_change_solver_request_identity_or_cache_key(tmp_path):
    common = {
        "seed": 46,
        "out_dir": str(tmp_path / "out"),
        "shared_solver_cache_path": str(tmp_path / "cache.sqlite"),
    }
    s3 = Config.from_flat(
        **common, experiment_setting="shared_peer_state_member_first_safe",
    )
    s4 = Config.from_flat(
        **common, experiment_setting="shared_member_aware_responsibility",
    )
    assert solver_request_identity(s3) == solver_request_identity(s4)
    left = PromptQuestionEvaluator(
        model_request_identity=solver_request_identity(s3),
        parser_version=s3.peer_state.parser_version,
        temperature=s3.models.temperature,
        decoding_seed=s3.training.seed,
        cache_metadata=solver_request_components(s3),
        version=PROMPT_QUESTION_EVALUATOR_VERSION,
    )
    right = PromptQuestionEvaluator(
        model_request_identity=solver_request_identity(s4),
        parser_version=s4.peer_state.parser_version,
        temperature=s4.models.temperature,
        decoding_seed=s4.training.seed,
        cache_metadata=solver_request_components(s4),
        version=PROMPT_QUESTION_EVALUATOR_VERSION,
    )
    assert left.key("prompt", "question") == right.key("prompt", "question")
    changed_contract = Config.from_flat(
        **common,
        experiment_setting="shared_peer_state_member_first_safe",
        answer_format="yes_no",
    )
    assert solver_request_identity(s3) != solver_request_identity(changed_contract)


def test_exact_question_message_bytes_cannot_alias_in_solver_cache(tmp_path):
    system = PromptEnsembleOptimizationSystem(Config.from_flat(out_dir=str(tmp_path)))
    compact = system.build_probe([{"question": "a b", "answer": "x"}])
    spaced = system.build_probe([{"question": "a  b", "answer": "x"}])
    left_hash = compact.examples[0].question_hash
    right_hash = spaced.examples[0].question_hash
    assert left_hash != right_hash
    assert system.prompt_question_evaluator.key("prompt", left_hash) != (
        system.prompt_question_evaluator.key("prompt", right_hash)
    )


def test_unchanged_prompt_per_question_observation_drift_fails():
    base_member = {
        "q": {
            "cache_key": "k",
            "observation_hash": "A",
            "parsed_answer_hash": "answer",
            "response_hash": "response",
            "valid": True,
            "terminal_invalid": False,
            "correct": True,
        }
    }
    prior = {
        "prompt_hashes": ["same"] * 5,
        "per_member": [dict(base_member) for _ in range(5)],
        "per_agent_correct_counts": [1] * 5,
        "team_vote_vector_hash": "team-a",
        "team_vote_correct_count": 1,
    }
    current = {
        **prior,
        "per_member": [dict(base_member) for _ in range(5)],
    }
    current["per_member"][2] = {
        "q": {**base_member["q"], "observation_hash": "B"}
    }
    audit = _compare_unchanged_test_observations(
        prior, current, prior_setting="shared_baseline",
    )
    assert audit["passed"] is False
    assert audit["per_question_drift_count"] == 1
    assert audit["per_question_drifts"][0]["agent_id"] == 2


def test_baseline_test_entries_enter_reference_and_all_replays_hit(tmp_path):
    reference = tmp_path / "reference.sqlite"
    PersistentSolverCache(reference)
    baseline = tmp_path / "baseline.sqlite"
    _sqlite_backup(reference, baseline)
    baseline_evaluator = _evaluator(baseline)

    async def produce(question, *_args):
        return PromptAnswer(question, f"FINAL_ANSWER: {question}", True)

    for question_hash in ("test-q1", "test-q2"):
        asyncio.run(_evaluate(baseline_evaluator, question_hash, produce))
    audit = _merge_ready_solver_cache(baseline, reference)
    assert audit["new_entries_merged"] == 2

    optimized = tmp_path / "optimized.sqlite"
    _sqlite_backup(reference, optimized)
    optimized_evaluator = _evaluator(optimized)

    async def forbidden(*_args):
        raise AssertionError("baseline test observation must be reused")

    for question_hash in ("test-q1", "test-q2"):
        asyncio.run(_evaluate(optimized_evaluator, question_hash, forbidden))
    assert optimized_evaluator.shared_cache.hits == 2
    assert optimized_evaluator.shared_cache.misses == 0
