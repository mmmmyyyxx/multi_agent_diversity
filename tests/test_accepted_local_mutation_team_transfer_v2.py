from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from infrastructure.common_solver_contract_v1.contract import canonical_question_payload
from multi_dataset_diverse_rl.answer_formats import canonical_answer
from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash, validate_manifest
from multi_dataset_diverse_rl.team_search.accepted_mutation_replay import (
    full_team_result, minibatch_full_gate_audit, summarize_mandatory_full_transfer,
)
from scripts import run_accepted_local_mutation_team_transfer_v2 as runner


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/accepted_local_mutation_team_transfer_v2_prep_20260919"
V1_REPORT = ROOT / "reports/accepted_local_mutation_team_transfer_v1_prep_20260919"
MANIFEST = ROOT / "experiments/manifests/accepted_local_mutation_team_transfer_v2.yaml"
PRIVATE = ROOT / "runs/accepted_local_mutation_team_transfer_v2_prep/private_bundle.json"
V1_PRIVATE = ROOT / "runs/accepted_local_mutation_team_transfer_v1_prep/private_bundle.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_v2_preserves_all_five_v1_mutations_exactly():
    current = read(PRIVATE)
    previous = read(V1_PRIVATE)
    assert current["baseline_team_hash"] == previous["baseline_team_hash"]
    assert current["accepted_mutations"] == previous["accepted_mutations"]
    assert [(row["target_member"], row["source_proposal_index"])
            for row in current["accepted_mutations"]] == [
        (1, 2), (2, 8), (3, 5), (4, 2), (4, 4),
    ]


def test_v2_manifest_is_authorized_only_for_frozen_replay_and_mandatory_full():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    schema = read(ROOT / "infrastructure/experiment_manifest.schema.json")
    assert validate_manifest(manifest, schema) == []
    assert manifest["artifacts"]["preregistration"]["sha256"] == preregistration_hash(manifest)
    assert manifest["api_authorization"]["authorized"] is True
    assert manifest["api_authorization"]["allowed_roles"] == ["solver"]
    assert manifest["api_authorization"]["allowed_phases"] == ["accepted_mutation_team_replay"]
    assert manifest["design"]["mandatory_full_candidate_count"] == 5
    assert manifest["design"]["team_minibatch_role"] == "TEAM_MINIBATCH_DIAGNOSTIC_GATE_V1"
    assert manifest["budget"]["limit"]["successful_solver_provider_calls"] == 800
    assert manifest["budget"]["limit"]["Validation50"] == 0
    assert manifest["budget"]["limit"]["Test50"] == 0


def test_v1_is_recorded_as_superseded_without_modification():
    evidence = read(REPORT / "v1_supersession.json")
    assert evidence["status"] == "SUPERSEDED_BEFORE_EXECUTION_BY_V2_DUE_TO_TEAMMINIBATCH_CENSORING"
    assert evidence["v1_files_modified"] is False
    assert evidence["v1_report_tree"] == build_sha256_manifest(V1_REPORT)


def test_cost_and_zero_api_freeze_are_exact():
    cost = read(REPORT / "cost_envelope.json")
    assert cost["mandatory_logical_solver_rows"] == 560
    assert cost["mandatory_exact_cache_hits"] == 60
    assert cost["mandatory_successful_solver_provider_calls"] == 500
    assert cost["total_successful_solver_provider_ceiling"] == 800
    assert cost["transport_attempt_ceiling"] == 3200
    assert read(REPORT / "api_ledger_summary.json")["provider_calls"] == 0
    handoff = read(REPORT / "EXPERIMENT_HANDOFF.json")
    assert handoff["READY_TO_RUN"] is True
    assert handoff["authorization"] == "EXPLICIT_V2_EXECUTION_AUTHORIZED_2026-09-19"


def test_runner_has_no_search_commit_scheduler_or_realizability_path():
    source = (ROOT / "scripts/run_accepted_local_mutation_team_transfer_v2.py").read_text()
    tree = ast.parse(source)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name and ("gepa" in name.lower() or "committer" in name.lower()) for name in imports)
    assert "ReflectionLM" not in source
    assert "SystemTeamCommitter" not in source
    assert ".commit(" not in source
    assert "if not minibatch_pass(metrics)" not in source
    assert "persistent_realizability" in source  # explicit zero in isolation evidence


def test_full_result_summary_and_gate_audit_are_independent_of_minibatch():
    assert [full_team_result(value) for value in (2, 0, -1)] == [
        "TEAM_POSITIVE", "TEAM_EQUAL", "TEAM_NEGATIVE",
    ]
    rows = [
        {"execution_status": "COMPLETE", "full_status": "PASS", "team_vote_delta": 1,
         "team_minibatch_status": "PASS"},
        {"execution_status": "COMPLETE", "full_status": "PASS", "team_vote_delta": 1,
         "team_minibatch_status": "FAIL"},
        {"execution_status": "COMPLETE", "full_status": "PASS", "team_vote_delta": 0,
         "team_minibatch_status": "PASS"},
        {"execution_status": "COMPLETE", "full_status": "PASS", "team_vote_delta": 0,
         "team_minibatch_status": "FAIL"},
        {"execution_status": "COMPLETE", "full_status": "PASS", "team_vote_delta": -1,
         "team_minibatch_status": "FAIL"},
    ]
    summary = summarize_mandatory_full_transfer(rows)
    assert summary["team_positive_count"] == 2
    assert summary["team_positive_denominator"] == 5
    assert "MEMBER_SPECIFIC_TRANSFER_HETEROGENEITY" in summary["interpretation_states"]
    assert minibatch_full_gate_audit(rows) == {
        "correct_promotion": 1,
        "false_negative_filtering": 1,
        "false_positive_promotion": 1,
        "correct_filtering": 2,
    }


class _FakeCompletions:
    def __init__(self, answers_by_member):
        self.answers_by_member = answers_by_member
        self.runtime = None

    async def create(self, **request):
        question = request["messages"][1]["content"]
        mutation_id = self.runtime.stage["mutation_id"]
        member = (1 if mutation_id == "shared_control"
                  else int(mutation_id.split("_", 1)[0].removeprefix("member")))
        answer = self.answers_by_member[member][question]
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=f"FINAL_ANSWER: {answer}"),
                finish_reason="stop",
            )],
        )


def test_fake_provider_minibatch_failure_does_not_suppress_full(tmp_path):
    private = read(PRIVATE)
    answers = {}
    for member in range(1, 5):
        task = private["phase_a_tasks"][f"seed78_update0_member{member}"]
        answers[member] = {
            canonical_question_payload(row["input_payload"]): row["parent_output"]
            for row in task["search_examples"]
        }
    completions = _FakeCompletions(answers)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    runtime = runner.Runtime(client=client, ledger_path=tmp_path / "provider.jsonl")
    completions.runtime = runtime
    result = asyncio.run(runner.replay(private, runtime))
    assert result["full_evaluated_candidates"] == 5
    assert all(row["team_minibatch_status"] == "FAIL" for row in result["cases"])
    assert all(row["full_status"] == "PASS" for row in result["cases"])
    assert all(row["full_team_result"] == "TEAM_EQUAL" for row in result["cases"])
    assert result["accounting"]["provider_successes"] == 500
    assert result["accounting"]["solver"]["cache_hits"] == 60
    assert result["isolation"] == {
        "GEPA": 0, "Reflection": 0, "write_back": 0,
        "persistent_realizability_update": 0, "Validation50": 0, "Test50": 0,
    }
    second_completions = _FakeCompletions(answers)
    second_client = SimpleNamespace(chat=SimpleNamespace(completions=second_completions))
    second_runtime = runner.Runtime(client=second_client, ledger_path=tmp_path / "provider2.jsonl")
    second_completions.runtime = second_runtime
    second = asyncio.run(runner.replay(private, second_runtime))
    assert second["cases"] == result["cases"]
    assert second["transfer_table"] == result["transfer_table"]
    assert second["transfer_summary"] == result["transfer_summary"]
    assert second["team_minibatch_gate_audit"] == result["team_minibatch_gate_audit"]


def test_fake_provider_minibatch_pass_also_reaches_same_mandatory_full(tmp_path):
    private = read(PRIVATE)
    answers = {}
    for member in range(1, 5):
        task = private["phase_a_tasks"][f"seed78_update0_member{member}"]
        mapping = {
            canonical_question_payload(row["input_payload"]): row["gold"]
            for row in task["search_examples"]
        }
        mapping.update({
            canonical_question_payload(row["question"]): canonical_answer(row["answer"], "option_letter")
            for row in private["shadow50"]
        })
        answers[member] = mapping
    completions = _FakeCompletions(answers)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    runtime = runner.Runtime(client=client, ledger_path=tmp_path / "provider.jsonl")
    completions.runtime = runtime
    result = asyncio.run(runner.replay(private, runtime))
    assert result["full_evaluated_candidates"] == 5
    assert all(row["team_minibatch_status"] == "PASS" for row in result["cases"])
    assert all(row["full_status"] == "PASS" for row in result["cases"])
    assert result["accounting"]["provider_successes"] <= 800


def test_v2_report_is_sanitized_and_hash_replayable():
    assert scan_sanitized_artifacts(REPORT) == []
    assert read(REPORT / "sanitization_manifest.json")["status"] == "PASS"
    assert read(REPORT / "sha256_manifest.json") == build_sha256_manifest(REPORT)
    assert all(read(REPORT / "fact_assertions.json").values())
