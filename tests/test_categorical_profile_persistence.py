from __future__ import annotations

import asyncio
import json

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.categorical_profiles import (
    endpoint_identifiability_snapshot,
    legal_option_labels,
    sanitized_categorical_profile,
)
from multi_dataset_diverse_rl.evaluation.fixed_probe import ProbeExample
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalPromptCandidate
from multi_dataset_diverse_rl.governance.artifacts import scan_sanitized_artifacts
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.team_search.schemas import TeamSearchAssignment
from multi_dataset_diverse_rl.team_search.system_runtime import (
    SystemTeamCandidateEvaluator,
    SystemTeamCommitter,
)


def _answer(value: str, *, valid: bool = True, terminal: bool = False) -> PromptAnswer:
    return PromptAnswer(
        answer=value,
        trace="private reasoning that must never be persisted",
        valid=valid,
        terminal_invalid=terminal,
    )


def _normalize(value: str) -> str:
    return value.strip().upper()


def _match(left: str, right: str) -> bool:
    return _normalize(left) == _normalize(right)


def test_sanitized_profile_keeps_categories_and_excludes_raw_material() -> None:
    examples = (
        ProbeExample("secret question\n(A) one\n(B) two", "row-1", "A"),
        ProbeExample("another secret\n(A) one\n(B) two", "row-2", "B"),
    )
    payload = sanitized_categorical_profile(
        examples=examples,
        profile=(_answer("A"), _answer("private free text")),
        normalize_answer=_normalize,
        match_answer=_match,
        parent_team_hash="parent",
        update_index=2,
        target_member=3,
        candidate_hash="candidate-hash",
        candidate_id="candidate-id",
        evaluation_stage="team_full_eval",
    )
    assert payload["row_count"] == 2
    assert payload["rows"][0] == {
        "example_id": "row-1",
        "normalized_choice": "A",
        "correct": True,
        "terminal_invalid": False,
        "candidate_hash": "candidate-hash",
    }
    assert payload["rows"][1]["normalized_choice"].startswith("VALUE_SHA256:")
    serialized = json.dumps(payload)
    for secret in ("secret question", "private free text", "private reasoning"):
        assert secret not in serialized


def test_invalid_answer_is_an_explicit_category() -> None:
    example = ProbeExample("q\n(A) one\n(B) two", "row", "A")
    payload = sanitized_categorical_profile(
        examples=(example,),
        profile=(_answer("ignored", valid=False, terminal=True),),
        normalize_answer=_normalize,
        match_answer=_match,
        parent_team_hash="parent",
        update_index=0,
        target_member=0,
        candidate_hash="hash",
        candidate_id="id",
        evaluation_stage="team_full_eval",
    )
    assert payload["rows"][0]["normalized_choice"] == "INVALID"
    assert payload["rows"][0]["correct"] is False
    assert payload["rows"][0]["terminal_invalid"] is True


def test_endpoint_snapshot_records_p_i_and_correctable_example_ids() -> None:
    example = ProbeExample("q\n(A) one\n(B) two\n(C) three", "row", "A")
    profiles = tuple((_answer(value),) for value in ("A", "A", "B", "B", "C"))
    snapshot = endpoint_identifiability_snapshot(
        examples=(example,),
        profiles=profiles,
        normalize_answer=_normalize,
        team_prompt_state_hash="team",
        update_index=1,
        trigger="test_commit",
        committed_target_member=4,
        committed_candidate_hash="candidate",
    )
    assert snapshot["endpoint_structurally_identifiable"] is True
    assert snapshot["immediate_vote_change_possible"] is True
    assert set(snapshot["p_i"]) == {"0", "1", "2", "3", "4"}
    assert snapshot["p_i"]["4"] == snapshot["by_member"]["4"]["pivotal_capable_count"]
    assert snapshot["by_member"]["4"]["pivotal_correctable_example_ids"] == ["row"]
    assert snapshot["by_member"]["4"]["pivotal_correctable_count"] == 1
    assert "pivotal_capable_row_indices" not in snapshot["by_member"]["4"]


def test_homogeneous_state_has_zero_structural_opportunities() -> None:
    example = ProbeExample("q\n(A) one\n(B) two", "row", "B")
    profiles = tuple((_answer("A"),) for _ in range(5))
    snapshot = endpoint_identifiability_snapshot(
        examples=(example,),
        profiles=profiles,
        normalize_answer=_normalize,
        team_prompt_state_hash="team",
        update_index=-1,
        trigger="initialization",
    )
    assert snapshot["total_member_row_opportunities"] == 0
    assert snapshot["immediate_vote_change_possible"] is False


def test_option_label_parser_supports_both_repository_formats() -> None:
    assert legal_option_labels("q\n(A) x\n(B) y") == ("A", "B")
    assert legal_option_labels("q\nA. x\nB) y") == ("A", "B")


def test_fixed_probe_initialization_atomically_materializes_replay_evidence(
    tmp_path,
) -> None:
    async def solver(_question: str, _agent_id: int, _prompt: str) -> PromptAnswer:
        return _answer("A")

    system = PromptEnsembleOptimizationSystem(
        Config.from_flat(out_dir=str(tmp_path), task_type="mmlu"),
        solver=solver,
    )
    asyncio.run(
        system.initialize_fixed_probe(
            [{"question": "sensitive stem\n(A) first\n(B) second", "answer": "A"}]
        )
    )

    profile_paths = sorted((tmp_path / "team_full_categorical_profiles").glob("*.json"))
    state_paths = sorted((tmp_path / "endpoint_identifiability_states").glob("*.json"))
    assert len(profile_paths) == 5
    assert len(state_paths) == 1
    assert (
        json.loads(state_paths[0].read_text(encoding="utf-8"))[
            "total_member_row_opportunities"
        ]
        == 0
    )
    published = "\n".join(
        path.read_text(encoding="utf-8") for path in [*profile_paths, *state_paths]
    )
    assert "sensitive stem" not in published
    assert "private reasoning" not in published
    assert scan_sanitized_artifacts(tmp_path) == []


def test_team_full_evaluation_persists_before_return(tmp_path) -> None:
    async def solver(_question: str, _agent_id: int, prompt: str) -> PromptAnswer:
        return _answer("B" if prompt == "replacement" else "A")

    async def run() -> None:
        system = PromptEnsembleOptimizationSystem(
            Config.from_flat(out_dir=str(tmp_path), task_type="mmlu"),
            solver=solver,
        )
        await system.initialize_fixed_probe(
            [{"question": "stem\n(A) first\n(B) second", "answer": "B"}]
        )
        evaluator = SystemTeamCandidateEvaluator(
            system=system,
            shadow_probe=system.fixed_probe,
            loop=asyncio.get_running_loop(),
            stage=lambda _stage: None,
            accounting=lambda: {
                "successful_provider_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
            update_index_reader=lambda: 0,
        )
        assignment = TeamSearchAssignment(
            target_member=0,
            parent_prompt=system.agents[0].current_prompt,
            evidence=(),
            optimization_context="",
            responsibility_identity="test",
        )
        candidate = LocalPromptCandidate("candidate-id", "replacement", 1.0, {}, (), 1)
        await asyncio.to_thread(evaluator.evaluate_full, assignment, candidate)

    asyncio.run(run())
    paths = sorted((tmp_path / "team_full_categorical_profiles").glob("*.json"))
    assert len(paths) == 6
    full = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
        if json.loads(path.read_text(encoding="utf-8"))["evaluation_stage"]
        == "team_full_eval"
    ]
    assert len(full) == 1
    assert full[0]["candidate_id"] == "candidate-id"
    assert full[0]["rows"][0]["normalized_choice"] == "B"


def test_two_layer_commit_immediately_persists_successor_p_i(tmp_path) -> None:
    async def solver(_question: str, _agent_id: int, prompt: str) -> PromptAnswer:
        return _answer("B" if prompt == "replacement procedure" else "A")

    async def run() -> None:
        system = PromptEnsembleOptimizationSystem(
            Config.from_flat(out_dir=str(tmp_path), task_type="mmlu"),
            solver=solver,
        )
        await system.initialize_fixed_probe(
            [{"question": "stem\n(A) first\n(B) second", "answer": "B"}]
        )
        evaluator = SystemTeamCandidateEvaluator(
            system=system,
            shadow_probe=system.fixed_probe,
            loop=asyncio.get_running_loop(),
            stage=lambda _stage: None,
            accounting=lambda: {
                "successful_provider_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
            update_index_reader=lambda: 0,
        )
        assignment = TeamSearchAssignment(
            target_member=0,
            parent_prompt=system.agents[0].current_prompt,
            evidence=(),
            optimization_context="",
            responsibility_identity="test",
        )
        candidate = LocalPromptCandidate(
            "candidate-id", "replacement procedure", 1.0, {}, (), 1
        )
        evaluation, _ = await asyncio.to_thread(
            evaluator.evaluate_full, assignment, candidate
        )
        committer = SystemTeamCommitter(
            system=system,
            evaluator=evaluator,
            update_index_reader=lambda: 0,
        )
        committer.commit(
            assignment=assignment,
            candidate=candidate,
            evaluation=evaluation,
        )

    asyncio.run(run())
    states = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "endpoint_identifiability_states").glob("*.json")
    ]
    assert len(states) == 2
    committed = [
        state for state in states if state["trigger"] == "two_layer_team_commit"
    ]
    assert len(committed) == 1
    assert committed[0]["committed_target_member"] == 0
    assert committed[0]["committed_candidate_hash"]
